# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Preallocated GPU Plan bringup; CPU oracle is used only by tests.

The control kernel is deliberately serial and bounded. This establishes exact
rules before optimizing planner latency. No CUDA tensors are read on the host.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

from flashinfer.moe_ep.megaflux_plan import checked_nbytes


@lru_cache(maxsize=1)
def planner_module():
    from flashinfer.jit.core import gen_jit_spec

    path = Path(__file__).with_suffix(".cu")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    # Editable checkouts may be used before build_backend creates data links.
    root = path.parents[5]
    includes = [
        str(root / name) for name in ("csrc", "include") if (root / name).is_dir()
    ]
    return gen_jit_spec(
        f"megaflux_planner_{digest}", [path], extra_include_paths=includes
    ).build_and_load()


class DevicePlan:
    """Stable capacity-sized buffers; valid record counts remain on GPU."""

    def __init__(self, config, *, device, block_capacity=None):
        c = config
        if c.world_size > 8 or c.num_logical_experts > 512 or c.replica_budget > 4:
            raise ValueError("device planner bringup supports P<=8, E<=512, R<=4")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("allocate and warm the GPU planner before capture")
        self.config = c
        self.module = planner_module()
        E, P, R = c.num_logical_experts, c.world_size, c.replica_budget
        capacity = c.block_capacity if block_capacity is None else block_capacity
        if capacity < 0:
            raise ValueError("negative block capacity")

        def alloc(shape):
            elements = 1
            for size in shape:
                elements *= size
            checked_nbytes(elements, 8)
            return torch.zeros(shape, dtype=torch.int64, device=device)

        self.source_counts = alloc((P, E))
        self.q = alloc((E, P))
        self.expert_counts = alloc((E,))
        self.expert_block_offsets = alloc((E + 1,))
        self.slot_to_expert = alloc((P, c.physical_slot_capacity))
        self.blocks = alloc((capacity, 8))
        self.replicas = alloc((P * R, 8))
        self.statistics = alloc((4, P))
        self.scratch = alloc((40000,))
        self.meta = alloc((6,))  # status, overflow, B, N, routes, epoch

    def build(self, source_counts, *, epoch, identity=False, requested_capacity=None):
        c = self.config
        if (
            source_counts.shape != self.source_counts.shape
            or source_counts.dtype != torch.int64
            or source_counts.device != self.source_counts.device
        ):
            raise ValueError("source counts differ from the configured GPU ABI")
        self.source_counts.copy_(source_counts)
        self.module.plan(
            self.source_counts,
            self.q,
            self.expert_counts,
            self.expert_block_offsets,
            self.slot_to_expert,
            self.blocks,
            self.replicas,
            self.statistics,
            self.meta,
            epoch,
            c.weight_version,
            identity,
            self.blocks.shape[0] if requested_capacity is None else requested_capacity,
            self.scratch,
        )
        return self


class LocalHistogram:
    def __init__(self, config, *, device):
        self.config = config
        self.module = planner_module()
        self.counts = torch.empty(
            config.num_logical_experts, dtype=torch.int64, device=device
        )
        self.ordinal = torch.empty(
            (config.max_tokens_per_source, config.top_k),
            dtype=torch.int64,
            device=device,
        )
        self.status = torch.empty(1, dtype=torch.int32, device=device)

    def run(self, padded_ids):
        if padded_ids.shape != self.ordinal.shape or padded_ids.dtype != torch.int64:
            raise ValueError("histogram requires fixed-capacity int64 routing IDs")
        self.module.histogram(padded_ids, self.counts, self.ordinal, self.status)
        return self.counts


class DeviceRoutes:
    """Preallocated source chunks and receiver tables for NCCL all-to-all."""

    def __init__(self, config, *, device):
        c = config
        self.module = planner_module()
        P, S, A = (
            c.world_size,
            c.physical_slot_capacity,
            c.max_tokens_per_source * c.top_k,
        )
        checked_nbytes(P * S * A, 4)
        checked_nbytes(A * 5, 8)
        self.send_counts = torch.empty((P, S), dtype=torch.int64, device=device)
        self.send_words = torch.empty((P, S, A), dtype=torch.int32, device=device)
        self.local_records = torch.empty((A, 5), dtype=torch.int64, device=device)
        self.counts = torch.empty_like(self.send_counts)
        self.received_words = torch.empty_like(self.send_words)
        self.words = torch.empty((S, P, A), dtype=torch.int32, device=device)

    def materialize(self, ids, histogram, plan, *, rank):
        self.module.materialize(
            ids,
            histogram.ordinal,
            plan.source_counts,
            plan.q,
            plan.slot_to_expert,
            plan.expert_counts,
            plan.expert_block_offsets,
            self.send_counts,
            self.send_words,
            self.local_records,
            rank,
        )

    def exchange(self, *, group=None):
        import torch.distributed as dist

        dist.all_to_all_single(self.counts, self.send_counts, group=group)
        dist.all_to_all_single(self.received_words, self.send_words, group=group)
        self.words.copy_(self.received_words.permute(1, 0, 2))


class DistributedPlanner:
    """GPU statistics, deterministic commit and fixed-size route exchange.

    Both candidate and owner-only Plans are generated on GPU. A collective MAX
    selects the latter for every rank on capacity failure, before route exchange.
    Invalid routing is a collective abort contract, not a local rerun.
    """

    def __init__(
        self,
        config,
        *,
        device,
        group=None,
        requested_capacity=None,
        debug_agreement=False,
    ):
        self.config = config
        self.group = group
        self.requested_capacity = requested_capacity
        self.histogram = LocalHistogram(config, device=device)
        self.candidate = DevicePlan(config, device=device)
        self.identity = DevicePlan(config, device=device)
        self.committed = DevicePlan(config, device=device)
        self.routing = DeviceRoutes(config, device=device)
        self.invalid = torch.empty(1, dtype=torch.int32, device=device)
        self.fallback = torch.empty(1, dtype=torch.int32, device=device)
        self.debug_agreement = debug_agreement
        self.hash = torch.empty(2, dtype=torch.int64, device=device)
        self.peer_hashes = torch.empty(
            config.world_size * 2, dtype=torch.int64, device=device
        )
        self.hash_weights = {}
        if debug_agreement:
            for name in (
                "q",
                "slot_to_expert",
                "blocks",
                "replicas",
                "statistics",
                "meta",
            ):
                field = getattr(self.candidate, name)
                self.hash_weights[name] = torch.arange(
                    1, field.numel() + 1, dtype=torch.int64, device=device
                ).reshape(field.shape)

    def run(self, ids, *, rank, epoch, external_failure=None):
        import torch.distributed as dist

        self.histogram.run(ids)
        self.invalid.copy_(self.histogram.status)
        dist.all_reduce(self.invalid, op=dist.ReduceOp.MAX, group=self.group)
        torch._assert_async(
            self.invalid.eq(0).all(), "invalid collective MegaFlux routes"
        )
        counts = self.candidate.source_counts
        dist.all_gather_into_tensor(
            counts.view(-1), self.histogram.counts, group=self.group
        )
        device_epoch = isinstance(epoch, torch.Tensor)
        self.candidate.build(
            counts,
            epoch=1 if device_epoch else epoch,
            requested_capacity=self.requested_capacity,
        )
        self.identity.build(counts, epoch=1 if device_epoch else epoch, identity=True)
        if device_epoch:
            self.candidate.meta[5:6].copy_(epoch)
            self.identity.meta[5:6].copy_(epoch)
        self.fallback.copy_(self.candidate.meta[1:2])
        if external_failure is not None:
            self.fallback.copy_(torch.maximum(self.fallback, external_failure))
        if self.debug_agreement:
            self.hash.zero_()
            for field_id, (name, weight) in enumerate(
                self.hash_weights.items(), start=1
            ):
                field = getattr(self.candidate, name)
                if name in ("blocks", "replicas"):
                    count = self.candidate.meta[2 if name == "blocks" else 3]
                    field = field * (weight[:, :1] <= count * 8)
                values = (field + field_id) * weight
                self.hash[0].add_(values.sum())
                self.hash[1].add_((values * weight).sum())
            dist.all_gather_into_tensor(self.peer_hashes, self.hash, group=self.group)
            mismatch = ~self.peer_hashes.reshape(-1, 2).eq(self.hash).all()
            self.fallback.copy_(torch.maximum(self.fallback, mismatch.to(torch.int32)))
        dist.all_reduce(self.fallback, op=dist.ReduceOp.MAX, group=self.group)
        for name in (
            "source_counts",
            "q",
            "expert_counts",
            "expert_block_offsets",
            "slot_to_expert",
            "blocks",
            "replicas",
            "statistics",
            "meta",
        ):
            torch.where(
                self.fallback.ne(0),
                getattr(self.identity, name),
                getattr(self.candidate, name),
                out=getattr(self.committed, name),
            )
        self.routing.materialize(ids, self.histogram, self.committed, rank=rank)
        self.routing.exchange(group=self.group)
        return self.committed
