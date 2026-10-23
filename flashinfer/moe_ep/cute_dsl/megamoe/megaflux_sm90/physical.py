# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Physical-slot adapter and fixed kernel-ready bank, without transport."""

import torch

from flashinfer.moe_ep.megaflux_plan import checked_nbytes

from .identity import IdentityFrontend, validate_plan_binding


def bind_physical_plan(workspace, plan):
    c = validate_plan_binding(workspace, plan)
    if plan.replica_descriptors:
        raise ValueError("physical-slot identity stage has no replica transport")
    workspace._frontend = PhysicalSlotFrontend(c, plan)


def allocate_physical_weight_bank(transformed, *, owned, capacity):
    """Copy owned weights once; keep packed storage-K stride and fixed slots."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("allocate physical weights before capture")
    if capacity < owned or owned <= 0:
        raise ValueError("physical capacity must cover all owned experts")
    bank = []
    for payload, offset, placeholder, residual in transformed:
        if (
            payload.shape[0] != owned
            or offset.shape[0] != owned
            or residual.shape != (owned,)
        ):
            raise ValueError("kernel-ready input must contain exactly the owned slots")
        _, packed_k, n = payload.shape
        checked_nbytes(capacity * n * packed_k, payload.element_size())
        checked_nbytes(capacity * offset[0].numel(), offset.element_size())
        checked_nbytes(capacity, residual.element_size())
        storage = torch.zeros(
            (capacity, n, packed_k), dtype=payload.dtype, device=payload.device
        )
        weight = storage.transpose(1, 2)
        folded = torch.zeros(
            (capacity, *offset.shape[1:]), dtype=offset.dtype, device=offset.device
        )
        scale = torch.ones((capacity,), dtype=residual.dtype, device=residual.device)
        weight[:owned].copy_(payload)
        folded[:owned].copy_(offset)
        scale[:owned].copy_(residual)
        bank.append((weight, folded, placeholder, scale))
    return tuple(bank)


def lower_receiver_metadata(plan, rank):
    """Lower actual executing slots, independently from logical ownership."""
    p = plan.config
    cap = p.physical_slot_capacity
    checked_nbytes(p.world_size * cap, 8)
    checked_nbytes(cap * p.world_size * p.max_tokens_per_source * p.top_k, 4)
    counts = torch.zeros((p.world_size, cap), dtype=torch.int64)
    words = torch.full(
        (cap, p.world_size, p.max_tokens_per_source * p.top_k), -1, dtype=torch.int32
    )
    token_prefix = sf_prefix = 0
    for slot in range(cap):
        blocks = sorted(
            (
                b
                for b in plan.block_descriptors
                if b.execution_rank == rank and b.physical_slot == slot
            ),
            key=lambda b: b.expert_block,
        )
        for j, b in enumerate(blocks):
            if plan.slot_to_expert[rank][slot] != b.logical_expert:
                raise ValueError("physical slot/expert identity mismatch")
            if (b.destination_row_offset, b.destination_sf_row_offset) != (
                token_prefix + j * p.beta,
                sf_prefix + j * p.beta,
            ):
                raise ValueError(
                    "Plan offsets must follow physical-slot token/SF prefixes"
                )
            for route in b.routes:
                r = route.source_rank
                at = int(counts[r, slot])
                words[slot, r, at] = (
                    route.source_token * p.top_k + route.source_topk_slot
                )
                counts[r, slot] += 1
        rows = sum(b.valid_rows for b in blocks)
        token_prefix += (rows + p.beta - 1) // p.beta * p.beta
        sf_prefix += (rows + p.sf_padding - 1) // p.sf_padding * p.sf_padding
    if (token_prefix, sf_prefix) != (
        plan.rank_padded_rows[rank],
        plan.rank_sf_rows[rank],
    ):
        raise ValueError("Plan pool capacity does not match physical slots")
    return counts, words


class PhysicalSlotFrontend(IdentityFrontend):
    def __init__(self, config, plan):
        self.physical_slot_capacity = plan.config.physical_slot_capacity
        self.num_owned_experts_per_rank = plan.config.num_owned_experts_per_rank
        self.num_logical_experts = plan.config.num_logical_experts
        counts, words = lower_receiver_metadata(plan, config.rank)
        device = torch.device("cuda", torch.cuda.current_device())
        super().__init__(config, counts.to(device), words.to(device))

    def _weight_expert_capacity(self):
        return self.physical_slot_capacity

    def _mega_compile_key(self):
        return (
            *super()._mega_compile_key(),
            "physical_slots_v1",
            self.physical_slot_capacity,
        )

    def _construct_kernel(self, kernel_type, **kwargs):
        logical = self.num_logical_experts
        owned = self.num_owned_experts_per_rank
        capacity = self.physical_slot_capacity

        class PhysicalKernel(kernel_type):
            def _build_local_region_specs(self):
                # Parent construction has now established physical pool/comm
                # capacity. Restore logical E before building region tables.
                self.num_total_experts = logical
                self.num_logical_experts = logical
                self.num_owned_experts_per_rank = owned
                self.physical_slot_capacity = capacity
                self.token_comm.num_total_experts = logical
                self.token_comm.num_logical_experts = logical
                self.token_comm.num_owned_experts_per_rank = owned
                self.token_comm.physical_slot_capacity = capacity
                return super()._build_local_region_specs()

        return super()._construct_kernel(PhysicalKernel, **kwargs)
