# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Synchronous complete-leg NVSHMEM transport for offline static Plans.

Only tracked symmetric root buffers participate in remote gets. Local kernel
banks are populated by D2D unpack after confirmed transfer completion. This
bringup path deliberately synchronizes and must run outside CUDA Graph capture.
"""

import torch

from flashinfer.moe_ep.megaflux_plan import checked_nbytes
from .identity import validate_plan_binding
from .physical import PhysicalSlotFrontend


def bind_static_plan(workspace, plan):
    config = validate_plan_binding(workspace, plan)
    workspace._frontend = PhysicalSlotFrontend(config, plan)


class StaticReplicaTransport:
    def __init__(self, plan, bank, rank):
        from nvshmem import core

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("allocate static transport before capture")
        self.rank = rank
        self.config = plan.config
        self.closed = False
        c = self.config
        # Distinct exports from an owner can exceed one receiver's R. Reserve
        # the conservative bound uniformly across PEs to keep symmetric order.
        self.export_capacity = min(
            c.num_owned_experts_per_rank, (c.world_size - 1) * c.replica_budget
        )
        self.sizes = []
        self.roots = []
        self.exports = []
        self.receivers = []
        for payload, offset, _, residual in bank:
            sizes = (
                checked_nbytes(payload[0].numel(), payload.element_size()),
                checked_nbytes(offset[0].numel(), offset.element_size()),
                checked_nbytes(1, residual.element_size()),
            )
            total = checked_nbytes(sum(sizes), 1)
            checked_nbytes(self.export_capacity + c.replica_budget, total)
            self.sizes.append(sizes)
            exports, receivers = [], []
            for target, count in (
                (exports, self.export_capacity),
                (receivers, c.replica_budget),
            ):
                for _ in range(count):
                    root = core.tensor((total,), dtype=torch.uint8)
                    self.roots.append(root)
                    buf, _, _ = core.tensor_get_buffer(root)
                    if root.numel() != total or buf.size < total:
                        raise RuntimeError("symmetric buffer is smaller than full leg")
                    target.append(root)
            self.exports.append(exports)
            self.receivers.append(receivers)

    @staticmethod
    def _pieces(leg, slot):
        payload, offset, _, residual = leg
        # Packed K is stride-one; flatten the underlying (N, packedK) storage,
        # preserving the Humming byte layout rather than re-quantizing.
        return (
            payload[slot].transpose(0, 1).reshape(-1).view(torch.uint8),
            offset[slot].reshape(-1).view(torch.uint8),
            residual[slot : slot + 1].view(torch.uint8),
        )

    @staticmethod
    def rendezvous():
        from nvshmem import core

        stream = torch.cuda.current_stream()
        core.barrier_all(stream=stream)
        stream.synchronize()

    def copy(self, plan, bank):
        from nvshmem import core, bindings

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("static synchronous replica transport cannot capture")
        if self.closed or plan.config != self.config:
            raise ValueError("transport geometry/version differs from Plan")
        c = self.config
        by_owner = [
            sorted(
                {
                    d.logical_expert
                    for d in plan.replica_descriptors
                    if d.owner_rank == r
                }
            )
            for r in range(c.world_size)
        ]
        if any(len(experts) > self.export_capacity for experts in by_owner):
            raise ValueError("distinct owner exports exceed symmetric capacity")
        for index, expert in enumerate(by_owner[self.rank]):
            for leg_id, leg in enumerate(bank):
                at = 0
                root = self.exports[leg_id][index]
                for piece, size in zip(
                    self._pieces(leg, expert % c.num_owned_experts_per_rank),
                    self.sizes[leg_id],
                    strict=True,
                ):
                    root[at : at + size].copy_(piece)
                    at += size
        # Publish all owners' complete snapshots before any remote get.
        self.rendezvous()
        stream = torch.cuda.current_stream()
        for d in plan.replica_descriptors:
            if d.receiver_rank != self.rank:
                continue
            export = by_owner[d.owner_rank].index(d.logical_expert)
            slot = d.receiver_slot
            for leg_id, leg in enumerate(bank):
                src = self.exports[leg_id][export]
                dst = self.receivers[leg_id][slot - c.num_owned_experts_per_rank]
                src_buf, _, _ = core.tensor_get_buffer(src)
                dst_buf, _, _ = core.tensor_get_buffer(dst)
                # get(tensor,tensor) uses allocator buffer capacity; explicitly
                # transfer exact logical bytes through the same NVSHMEM binding.
                bindings.getmem_on_stream(
                    dst_buf.handle,
                    src_buf.handle,
                    sum(self.sizes[leg_id]),
                    d.owner_rank,
                    stream.cuda_stream,
                )
                core.quiet(stream=stream)
                stream.synchronize()
                at = 0
                for piece, size in zip(
                    self._pieces(leg, slot), self.sizes[leg_id], strict=True
                ):
                    piece.copy_(dst[at : at + size])
                    at += size
        # All bytes (including offsets/residuals) are visible before the first
        # TMA/GEMM consumer. No communication is left outstanding on return.
        self.rendezvous()

    def close(self):
        from nvshmem import core

        if not self.closed:
            self.rendezvous()
            for root in reversed(self.roots):
                core.free_tensor(root)
            self.roots.clear()
            self.closed = True


class DeviceReplicaTransport:
    """GPU-selected exports with conservative synchronous root gets.

    This correctness path fetches every owner's bounded export root on every
    PE, then unpacks only GPU-selected replicas. Extra bytes are charged work;
    selective device-initiated transport is a later optimization.
    """

    def __init__(self, config, bank, rank, *, allocate_cache=True):
        from nvshmem import core
        from .planner import planner_module

        self.config, self.rank = config, rank
        self.module = planner_module()
        self.export_capacity = min(
            config.num_owned_experts_per_rank,
            (config.world_size - 1) * config.replica_budget,
        )
        self.roots, self.exports, self.caches, self.pieces = [], [], [], []
        self.cache_sizes = []
        self.closed = False
        S = config.physical_slot_capacity
        for payload, offset, _, residual in bank:
            pieces = (
                payload.transpose(1, 2).reshape(S, -1).view(torch.uint8),
                offset.reshape(S, -1).view(torch.uint8),
                residual.reshape(S, -1).view(torch.uint8),
            )
            self.pieces.append(pieces)
            size = sum(p.shape[1] for p in pieces)
            checked_nbytes((config.world_size + 1) * self.export_capacity, size)
            if not self.export_capacity:
                continue
            source = core.tensor((self.export_capacity, size), dtype=torch.uint8)
            self.roots.append(source)
            self.exports.append(source)
            self.cache_sizes.append(size)
            if allocate_cache:
                cache = core.tensor(
                    (config.world_size * self.export_capacity, size),
                    dtype=torch.uint8,
                )
                self.roots.append(cache)
                self.caches.append(cache)

    def allocate_caches(self, capacity):
        """Allocate symmetric receive caches with ``capacity`` rows per PE."""
        from nvshmem import core

        if self.caches:
            raise RuntimeError("replica receive caches were already allocated")
        for size in self.cache_sizes:
            cache = core.tensor(
                (self.config.world_size * capacity, size), dtype=torch.uint8
            )
            self.roots.append(cache)
            self.caches.append(cache)

    def copy(self, plan):
        from nvshmem import core, bindings

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("synchronous device-plan transport cannot capture")
        if not self.export_capacity:
            return
        c = self.config
        for pieces, root in zip(self.pieces, self.exports, strict=True):
            self.module.weights(
                plan.q, plan.slot_to_expert, *pieces, root, self.rank, False
            )
        StaticReplicaTransport.rendezvous()
        stream = torch.cuda.current_stream()
        for root, cache in zip(self.exports, self.caches, strict=True):
            source_buf, source_bytes, _ = core.tensor_get_buffer(root)
            cache_buf, cache_bytes, _ = core.tensor_get_buffer(cache)
            if source_buf.size < source_bytes or cache_buf.size < cache_bytes:
                raise RuntimeError("registered symmetric root capacity mismatch")
            for owner in range(c.world_size):
                # Bounds-checked arithmetic within tracked symmetric roots.
                # Never interpret an ordinary local weight pointer remotely.
                assert (owner + 1) * source_bytes <= cache_bytes
                bindings.getmem_on_stream(
                    cache_buf.handle + owner * source_bytes,
                    source_buf.handle,
                    source_bytes,
                    owner,
                    stream.cuda_stream,
                )
        core.quiet(stream=stream)
        stream.synchronize()
        for pieces, cache in zip(self.pieces, self.caches, strict=True):
            self.module.weights(
                plan.q, plan.slot_to_expert, *pieces, cache, self.rank, True
            )
        StaticReplicaTransport.rendezvous()

    def close(self):
        from nvshmem import core

        if not self.closed:
            StaticReplicaTransport.rendezvous()
            for root in reversed(self.roots):
                core.free_tensor(root)
            self.roots.clear()
            self.closed = True
