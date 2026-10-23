# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""R=0 Plan receiver-metadata adapter; projection arithmetic is unchanged.

Offline Plans are uploaded before warmup. Three device copies restore their
receiver metadata each invocation because the original tail resets counters.
This adapter does not run an online GPU planner or support replicas.
"""

import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64
import torch

from flashinfer.moe_ep.megaflux_plan import PlanStatus, pack_plan_records
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
    MegaMoEHopperMxfp4Frontend,
    hopper_mxfp4_identity_regions,
    hopper_mxfp4_token_comm_type,
)


_TokenComm = hopper_mxfp4_token_comm_type()


class IdentityTokenComm(_TokenComm):  # type: ignore[valid-type,misc]
    """Use Plan-populated receiver tables instead of router atomic tickets."""

    @cute.jit
    def dispatch_prep(
        self,
        token_comm_storage,
        topk_idx,
        expert_send_count,
        src_token_topk_idx,
        token_rank_mask,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        local_rank,
        num_tokens,
        num_sms,
    ):
        # Source words/counts were copied from the Plan on the current stream.
        # No atomic ticket or router read is needed for this identity adapter.
        pass

    @cute.jit
    def dispatch_barrier(
        self,
        expert_send_count,
        expert_recv_count,
        expert_recv_count_sum,
        nvlink_barrier_signal,
        grid_sync_counter,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        local_rank,
        num_sms,
        nvlink_barrier_counter,
    ):
        i = (
            sm_idx * Int32(self.active_dispatch_threads)
            + warp_idx * Int32(32)
            + lane_idx
        )
        stride = Int32(num_sms * self.active_dispatch_threads)
        while i < Int32(self.num_experts_per_rank):
            count = Int64(expert_recv_count_sum[i]) & Int64(0xFFFFFFFF)
            expert_recv_count_sum[i] = (
                Int64(num_sms * self.world_size) << Int64(32)
            ) | count
            i += stride
        # Keep the original grid/NVLink rendezvous and tail barrier sequence.
        # It publishes source inputs and local Plan metadata before scheduling.
        self.nvlink_barrier(
            nvlink_barrier_signal,
            nvlink_barrier_counter,
            grid_sync_counter,
            peer_rank_ptr_mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            num_sms=num_sms,
            prologue_grid_sync=True,
            epilogue_grid_sync=True,
            sync_threads=self.active_dispatch_threads,
        )


class IdentityFrontend(MegaMoEHopperMxfp4Frontend):
    def __init__(self, config, receiver_counts, source_words):
        super().__init__(config)
        self._identity_counts = receiver_counts
        self._identity_totals = receiver_counts.sum(dim=0)
        self._identity_words = source_words

    def _mega_compile_key(self):
        return (*super()._mega_compile_key(), "megaflux_identity_v1")

    def _construct_kernel(self, kernel_type, **kwargs):
        kernel = super()._construct_kernel(kernel_type, **kwargs)
        # The communication object contains constructor-time configuration,
        # not per-invocation data. Preserve its full barrier/warp policy.
        comm = IdentityTokenComm.__new__(IdentityTokenComm)
        comm.__dict__.update(kernel.token_comm.__dict__)
        kernel.token_comm = comm
        return kernel

    def run(self, inputs, *, num_tokens=None, sync=True, reset_counters=False):
        if reset_counters:
            raise ValueError("identity adapter requires normal tail-managed reset")
        if (
            self._resolve_num_tokens(inputs, num_tokens)
            != self.config.num_tokens_per_rank
        ):
            raise ValueError("identity adapter launches the full padded capacity")
        self.prepare_launch(inputs, num_tokens=num_tokens)
        regions = hopper_mxfp4_identity_regions(self)
        regions["expert_recv_count"].copy_(self._identity_counts)
        regions["expert_recv_count_sum"].copy_(self._identity_totals)
        regions["src_token_topk_idx"].copy_(self._identity_words)
        return super().run(inputs, num_tokens=num_tokens, sync=sync)

    def make_launch_thunk(self, *args, **kwargs):
        raise NotImplementedError("identity Plan metadata must be restored per launch")


def bind_identity_plan(workspace, plan):
    """Upload a trusted offline identity Plan before compilation/capture.

    Caller must stage inputs matching this Plan's original route identities.
    The Plan is static until explicitly rebound; binding during capture or
    replacing an already compiled workspace is rejected.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("bind the identity Plan before capture")
    c = workspace._frontend.config
    p = plan.config
    if workspace._frontend._mega is not None:
        raise RuntimeError("bind the identity Plan before frontend warmup")
    if plan.status != PlanStatus.OK or plan.overflow:
        raise ValueError("identity executor requires a successful Plan")
    if p.replica_budget != 0 or plan.replica_descriptors:
        raise ValueError("identity executor requires R=0")
    if p.layout_id != "humming_sm90_m64_k128_gateup8_residual_x64_v1":
        raise ValueError("identity Plan uses an unsupported weight layout")
    if c.num_tokens_per_rank * c.num_topk >= (1 << 31):
        raise ValueError("source metadata words exceed the signed int32 ABI")
    if c.world_size * c.num_tokens_per_rank * c.num_topk >= (1 << 32):
        raise ValueError("identity token counts exceed the packed counter ABI")
    if (p.num_logical_experts, p.world_size, p.top_k, p.max_tokens_per_source) != (
        c.num_total_experts,
        c.world_size,
        c.num_topk,
        c.num_tokens_per_rank,
    ):
        raise ValueError("Plan and workspace shapes differ")
    if (
        c.mma_tiler_mnk != (128, 32, 128)
        or c.cluster_shape_mnk != (1, 1, 1)
        or c.pingpong
        or c.dedup_dispatch
        or c.grouped_token_back
        or c.in_kernel_fc2_reduce
        or c.tail_split_pairs
    ):
        raise ValueError("identity executor requires the frozen no-dedup N32 tactic")
    pack_plan_records(plan)  # Refuse unsuccessful Plans before any upload.
    owned = p.num_owned_experts_per_rank
    # Physical capacity equals owned only for this deliberately restricted R=0 ABI.
    counts = torch.tensor(plan.source_expert_counts, dtype=torch.int64)
    counts = counts[:, c.rank * owned : (c.rank + 1) * owned].contiguous()
    words = torch.full(
        (owned, c.world_size, c.num_tokens_per_rank * c.num_topk), -1, dtype=torch.int32
    )
    positions = [[0] * c.world_size for _ in range(owned)]
    for block in plan.block_descriptors:
        if block.execution_rank != block.logical_expert // owned:
            raise ValueError("identity Plan contains relocated work")
        if block.execution_rank == c.rank:
            e = block.physical_slot
            for route in block.routes:
                r = route.source_rank
                words[e, r, positions[e][r]] = (
                    route.source_token * c.num_topk + route.source_topk_slot
                )
                positions[e][r] += 1
    device = workspace.x.device
    workspace._frontend = IdentityFrontend(c, counts.to(device), words.to(device))
