# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Attributed copy of the vendor epilogue task loop, with phase-local reads.

Derived from epilogue_fp8_swapab.py:SwapABFp8GluEpilogue.run. Arithmetic,
barriers, stores and return semantics are retained. The only changes acquire
ready before the current phase's residual read, avoid the unused other-phase
residual read, and record progress after FC1 completion.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Int64
from cutlass.cute.typing import Float32
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
    hopper_mxfp4_pipeline_support,
)
from .pipeline_flags import acquire_leg, record_progress

_support = hopper_mxfp4_pipeline_support()
_Base = _support["SwapABFp8GluEpilogue"]
BlockPhase = _support["BlockPhase"]
EpiWarpCount = _support["EpiWarpCount"]
WarpThreadCount = _support["WarpThreadCount"]
GpuReleaseFlagBatchTracker = _support["GpuReleaseFlagBatchTracker"]
consume_initial_pingpong_work = _support["consume_initial_pingpong_work"]
consume_next_pingpong_work = _support["consume_next_pingpong_work"]
_red_or_release_gpu_b64 = _support["_red_or_release_gpu_b64"]
red_add_release_gpu_s32 = _support["red_add_release_gpu_s32"]
iket = _support["iket"]


class PipelineEpilogue(_Base):  # type: ignore[valid-type,misc]
    @cute.jit
    def run(
        self,
        sched_consumer,
        sched_ext,
        smem_fc1_output_buffer: cute.Tensor,
        smem_fc1_amax: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        smem_activation_sf: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        gmem_fc2_output: cute.Tensor,
        gmem_fc1_done_counter: cute.Tensor,
        local_warp_idx: int,
        tidx,
        fc1_activation_dequant_scale: cute.Tensor,
        fc1_weight_dequant_scale: cute.Tensor,
        fc2_activation_dequant_scale: cute.Tensor,
        fc2_weight_dequant_scale: cute.Tensor,
        norm_const,
        tiled_mma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        accumulators: cute.Tensor,
        accum_temp: cute.Tensor,
        run_wgmma_task_tile: cutlass.Constexpr,
        ab_pipeline,
        weight_sf_pipeline,
        ab_consumer_state,
        k_tile_cnt_fc1,
        k_tile_cnt_fc2,
        _iket_active,
        n_half,
        warpgroup_idx,
        math_wg_order_barrier,
        epi_wg_order_barrier,
        fc1_offload_full_mbar_ptr=None,
        fc1_offload_empty_mbar_ptr=None,
        fc1_offload_mbox=None,
        token_comm_args=None,
        gmem_fc1_c=None,
    ) -> None:
        """
        Run the full FP8 fc1+fc2 fused MMA+epilogue task-tile loop.

        ``token_comm_args`` (MegaMoE path) is forwarded to the fc2 task tile so
        the fc2 STG is routed to the source rank's combine output.
        """
        task_tile_boundary_bar = pipeline.NamedBarrier(
            barrier_id=(
                self._epilog_sync_bar_id + warpgroup_idx
                if cutlass.const_expr(self._pingpong)
                else self._epilog_sync_bar_id
            ),
            num_threads=(
                EpiWarpCount * WarpThreadCount
                if cutlass.const_expr(self._pingpong)
                else WarpThreadCount * len(self._epilogue_warp_ids)
            ),
        )

        valid_hidden = cutlass.Int32(gmem_fc2_output.shape[1])
        if _iket_active:
            iket.range_push("swapab_sched_consume_initial")
        if cutlass.const_expr(self._pingpong):
            (
                sched_consumer,
                work_tile_info,
                ab_consumer_state,
            ) = consume_initial_pingpong_work(
                sched_consumer,
                warpgroup_idx,
                ab_consumer_state,
                k_tile_cnt_fc1,
                k_tile_cnt_fc2,
            )
        else:
            work_tile_info = sched_consumer.consume_work()
        if _iket_active:
            iket.range_pop()

        flag_tracker = GpuReleaseFlagBatchTracker(
            flag_addr=Int64(0),
            cumulated_flags=cutlass.Int32(0),
            phase=cutlass.Int32(work_tile_info.phase),
            tid=tidx
            % (
                EpiWarpCount * WarpThreadCount
                if self._pingpong
                else len(self._epilogue_warp_ids) * WarpThreadCount
            ),
        )
        if cutlass.const_expr(self._pingpong):
            math_wg_order_state = math_wg_order_barrier.state.clone()
            epi_wg_order_state = epi_wg_order_barrier.state.clone()
        else:
            math_wg_order_state = ab_consumer_state.clone()
            epi_wg_order_state = ab_consumer_state.clone()

        # Offload: one empty-mbar phase bit per warp (its WG's slot).
        fc1_off_phase = cutlass.Int32(0)
        while work_tile_info.is_valid_tile:
            # Outer task-tile range, emitted by local warp 0 of each epilogue
            # warpgroup. It starts
            # before expert-scale setup and contains the representative WGMMA
            # child range plus that warp's representative epilogue child
            # ranges. FC1 also includes final TMA STG issue; async drain,
            # task-boundary synchronization, and next-tile scheduling follow
            # after the matching pop below.
            if _iket_active:
                if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                    if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
                        iket.range_push("swapab_fc1_task_bw")
                    elif cutlass.const_expr(self.fp8_scale_mode == "mxfp4_hybrid"):
                        iket.range_push("swapab_fc1_task_mxfp4_hybrid")
                    else:
                        iket.range_push("swapab_fc1_task_pt")
                else:
                    if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
                        iket.range_push("swapab_fc2_task_bw")
                    elif cutlass.const_expr(self.fp8_scale_mode == "mxfp4_hybrid"):
                        iket.range_push("swapab_fc2_task_mxfp4_hybrid")
                    else:
                        iket.range_push("swapab_fc2_task_pt")

            expert_idx = work_tile_info.expert_idx
            if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
                fc1_act_weight_dequant_scale = Float32(1.0)
                fc2_act_dequant_scale = Float32(1.0)
                fc2_act_weight_dequant_scale = Float32(1.0)
            elif cutlass.const_expr(self.fp8_scale_mode == "mxfp4_hybrid"):
                # Activation scales were applied in the hybrid mainloop.
                # Only Humming's per-expert residual remains for epilogue.
                fc1_act_weight_dequant_scale = Float32(1.0)
                fc2_act_dequant_scale = Float32(1.0)
                fc2_act_weight_dequant_scale = Float32(1.0)
                if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                    acquire_leg(
                        sched_ext.fc1_ready_counter_ptr.toint()
                        + Int64(self._mf_ready_offset)
                        + Int64(8) * Int64(expert_idx),
                        sched_ext.fc1_ready_counter_ptr.toint()
                        + Int64(self._mf_epoch_offset),
                    )
                    fc1_act_weight_dequant_scale = Float32(
                        fc1_weight_dequant_scale[expert_idx]
                    )
                else:
                    acquire_leg(
                        sched_ext.fc1_ready_counter_ptr.toint()
                        + Int64(self._mf_ready_offset)
                        + Int64(8) * (Int64(self._mf_capacity) + Int64(expert_idx)),
                        sched_ext.fc1_ready_counter_ptr.toint()
                        + Int64(self._mf_epoch_offset),
                    )
                    fc2_act_weight_dequant_scale = Float32(
                        fc2_weight_dequant_scale[expert_idx]
                    )
                    if expert_idx >= cutlass.Int32(self._mf_owned):
                        record_progress(
                            sched_ext.fc1_ready_counter_ptr.toint()
                            + Int64(self._mf_telemetry_offset + 16)
                        )
            else:
                fc1_act_weight_dequant_scale = Float32(
                    fc1_activation_dequant_scale[0]
                ) * Float32(fc1_weight_dequant_scale[expert_idx])
                fc2_act_dequant_scale = Float32(fc2_activation_dequant_scale[0])
                fc2_act_weight_dequant_scale = fc2_act_dequant_scale * Float32(
                    fc2_weight_dequant_scale[expert_idx]
                )

            if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                if cutlass.const_expr(self._fc1_store_offload):
                    # Acquire this WG's store slot from the server before
                    # the R2S reuses it (pre-armed once at init).
                    cute.arch.mbarrier_wait(
                        fc1_offload_empty_mbar_ptr + warpgroup_idx,
                        fc1_off_phase & cutlass.Int32(1),
                    )
                    fc1_off_phase = fc1_off_phase + cutlass.Int32(1)
                if cutlass.const_expr(self._wgmma_fragment_count == 1):
                    (
                        ab_consumer_state,
                        math_wg_order_state,
                        epi_wg_order_state,
                    ) = self._run_fc1_half_tile(
                        work_tile_info=work_tile_info,
                        local_warp_idx=local_warp_idx,
                        tiled_mma=tiled_mma,
                        tCrA=tCrA,
                        tCrB=tCrB,
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        n_half=0,
                        store_stage_idx=(warpgroup_idx if self._pingpong else 0),
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        k_tile_cnt_fc1=k_tile_cnt_fc1,
                        k_tile_cnt_fc2=k_tile_cnt_fc2,
                        _iket_active=_iket_active,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        smem_fc1_amax=smem_fc1_amax,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        sched_ext=sched_ext,
                        gmem_fc1_output=gmem_fc1_output,
                        gmem_fc1_output_sf=gmem_fc1_output_sf,
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        gmem_topk_scores=gmem_topk_scores,
                        tidx=tidx,
                        fc1_act_weight_dequant_scale=fc1_act_weight_dequant_scale,
                        fc2_act_dequant_scale=fc2_act_dequant_scale,
                        norm_const=norm_const,
                        run_wgmma_task_tile=run_wgmma_task_tile,
                        storage_group_idx=(warpgroup_idx if self._pingpong else 0),
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
                        gmem_fc1_c=gmem_fc1_c,
                    )
                    self._store_fc1_task_tile(
                        work_tile_info=work_tile_info,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        sched_ext=sched_ext,
                        gmem_fc1_output=gmem_fc1_output,
                        local_warp_idx=local_warp_idx,
                        n_half=0,
                        store_stage_idx=(warpgroup_idx if self._pingpong else 0),
                        storage_group_idx=(warpgroup_idx if self._pingpong else 0),
                        _iket_active=_iket_active,
                        gmem_fc1_done_counter=gmem_fc1_done_counter,
                        fc1_offload_full_mbar_ptr=fc1_offload_full_mbar_ptr,
                        fc1_offload_mbox=fc1_offload_mbox,
                    )
                elif n_half == cutlass.Int32(0):
                    (
                        ab_consumer_state,
                        math_wg_order_state,
                        epi_wg_order_state,
                    ) = self._run_fc1_half_tile(
                        work_tile_info=work_tile_info,
                        local_warp_idx=local_warp_idx,
                        tiled_mma=tiled_mma,
                        tCrA=tCrA,
                        tCrB=tCrB,
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        n_half=0,
                        store_stage_idx=cutlass.Int32(0),
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        k_tile_cnt_fc1=k_tile_cnt_fc1,
                        k_tile_cnt_fc2=k_tile_cnt_fc2,
                        _iket_active=_iket_active,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        smem_fc1_amax=smem_fc1_amax,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        sched_ext=sched_ext,
                        gmem_fc1_output=gmem_fc1_output,
                        gmem_fc1_output_sf=gmem_fc1_output_sf,
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        gmem_topk_scores=gmem_topk_scores,
                        tidx=tidx,
                        fc1_act_weight_dequant_scale=fc1_act_weight_dequant_scale,
                        fc2_act_dequant_scale=fc2_act_dequant_scale,
                        norm_const=norm_const,
                        run_wgmma_task_tile=run_wgmma_task_tile,
                        storage_group_idx=(warpgroup_idx if self._pingpong else 0),
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
                        gmem_fc1_c=gmem_fc1_c,
                    )
                    self._store_fc1_task_tile(
                        work_tile_info=work_tile_info,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        sched_ext=sched_ext,
                        gmem_fc1_output=gmem_fc1_output,
                        local_warp_idx=local_warp_idx,
                        n_half=0,
                        store_stage_idx=cutlass.Int32(0),
                        storage_group_idx=(warpgroup_idx if self._pingpong else 0),
                        _iket_active=_iket_active,
                        gmem_fc1_done_counter=gmem_fc1_done_counter,
                        fc1_offload_full_mbar_ptr=fc1_offload_full_mbar_ptr,
                        fc1_offload_mbox=fc1_offload_mbox,
                    )
                else:
                    (
                        ab_consumer_state,
                        math_wg_order_state,
                        epi_wg_order_state,
                    ) = self._run_fc1_half_tile(
                        work_tile_info=work_tile_info,
                        local_warp_idx=local_warp_idx,
                        tiled_mma=tiled_mma,
                        tCrA=tCrA,
                        tCrB=tCrB,
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        n_half=1,
                        store_stage_idx=cutlass.Int32(1),
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        k_tile_cnt_fc1=k_tile_cnt_fc1,
                        k_tile_cnt_fc2=k_tile_cnt_fc2,
                        _iket_active=_iket_active,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        smem_fc1_amax=smem_fc1_amax,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        sched_ext=sched_ext,
                        gmem_fc1_output=gmem_fc1_output,
                        gmem_fc1_output_sf=gmem_fc1_output_sf,
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        gmem_topk_scores=gmem_topk_scores,
                        tidx=tidx,
                        fc1_act_weight_dequant_scale=fc1_act_weight_dequant_scale,
                        fc2_act_dequant_scale=fc2_act_dequant_scale,
                        norm_const=norm_const,
                        run_wgmma_task_tile=run_wgmma_task_tile,
                        storage_group_idx=(warpgroup_idx if self._pingpong else 1),
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
                        gmem_fc1_c=gmem_fc1_c,
                    )
                    self._store_fc1_task_tile(
                        work_tile_info=work_tile_info,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        sched_ext=sched_ext,
                        gmem_fc1_output=gmem_fc1_output,
                        local_warp_idx=local_warp_idx,
                        n_half=1,
                        store_stage_idx=cutlass.Int32(1),
                        storage_group_idx=(warpgroup_idx if self._pingpong else 1),
                        _iket_active=_iket_active,
                        gmem_fc1_done_counter=gmem_fc1_done_counter,
                        fc1_offload_full_mbar_ptr=fc1_offload_full_mbar_ptr,
                        fc1_offload_mbox=fc1_offload_mbox,
                    )
                if expert_idx < cutlass.Int32(self._mf_owned):
                    record_progress(
                        sched_ext.fc1_ready_counter_ptr.toint()
                        + Int64(self._mf_telemetry_offset)
                    )
                else:
                    record_progress(
                        sched_ext.fc1_ready_counter_ptr.toint()
                        + Int64(self._mf_telemetry_offset + 8)
                    )
            else:
                # Preserve the vendor's constexpr/runtime branch distinction.
                if cutlass.const_expr(self._wgmma_fragment_count == 1):  # noqa: SIM114
                    (
                        ab_consumer_state,
                        math_wg_order_state,
                        epi_wg_order_state,
                    ) = self._run_fc2_half_tile(
                        work_tile_info=work_tile_info,
                        local_warp_idx=local_warp_idx,
                        tiled_mma=tiled_mma,
                        tCrA=tCrA,
                        tCrB=tCrB,
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        n_half=0,
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        k_tile_cnt_fc1=k_tile_cnt_fc1,
                        k_tile_cnt_fc2=k_tile_cnt_fc2,
                        _iket_active=_iket_active,
                        sched_ext=sched_ext,
                        gmem_fc2_output=gmem_fc2_output,
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        valid_hidden=valid_hidden,
                        tidx=tidx,
                        fc2_act_weight_dequant_scale=fc2_act_weight_dequant_scale,
                        token_comm_args=token_comm_args,
                        run_wgmma_task_tile=run_wgmma_task_tile,
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
                    )
                elif n_half == cutlass.Int32(0):
                    (
                        ab_consumer_state,
                        math_wg_order_state,
                        epi_wg_order_state,
                    ) = self._run_fc2_half_tile(
                        work_tile_info=work_tile_info,
                        local_warp_idx=local_warp_idx,
                        tiled_mma=tiled_mma,
                        tCrA=tCrA,
                        tCrB=tCrB,
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        n_half=0,
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        k_tile_cnt_fc1=k_tile_cnt_fc1,
                        k_tile_cnt_fc2=k_tile_cnt_fc2,
                        _iket_active=_iket_active,
                        sched_ext=sched_ext,
                        gmem_fc2_output=gmem_fc2_output,
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        valid_hidden=valid_hidden,
                        tidx=tidx,
                        fc2_act_weight_dequant_scale=fc2_act_weight_dequant_scale,
                        token_comm_args=token_comm_args,
                        run_wgmma_task_tile=run_wgmma_task_tile,
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
                    )
                else:
                    (
                        ab_consumer_state,
                        math_wg_order_state,
                        epi_wg_order_state,
                    ) = self._run_fc2_half_tile(
                        work_tile_info=work_tile_info,
                        local_warp_idx=local_warp_idx,
                        tiled_mma=tiled_mma,
                        tCrA=tCrA,
                        tCrB=tCrB,
                        accumulators=accumulators,
                        accum_temp=accum_temp,
                        n_half=1,
                        ab_pipeline=ab_pipeline,
                        weight_sf_pipeline=weight_sf_pipeline,
                        ab_consumer_state=ab_consumer_state,
                        k_tile_cnt_fc1=k_tile_cnt_fc1,
                        k_tile_cnt_fc2=k_tile_cnt_fc2,
                        _iket_active=_iket_active,
                        sched_ext=sched_ext,
                        gmem_fc2_output=gmem_fc2_output,
                        smem_activation_sf=smem_activation_sf,
                        smem_weight_sf=smem_weight_sf,
                        valid_hidden=valid_hidden,
                        tidx=tidx,
                        fc2_act_weight_dequant_scale=fc2_act_weight_dequant_scale,
                        token_comm_args=token_comm_args,
                        run_wgmma_task_tile=run_wgmma_task_tile,
                        math_wg_order_barrier=math_wg_order_barrier,
                        math_wg_order_state=math_wg_order_state,
                        epi_wg_order_barrier=epi_wg_order_barrier,
                        epi_wg_order_state=epi_wg_order_state,
                    )
            # Matches the selected swapab_fc1/fc2_task_* range above.
            if _iket_active:
                iket.range_pop()

            cur_was_linear1 = work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
            # The swap-AB token axis is GEMM N. Keep one FC1-done slot per CTA
            # token tile because cluster peers produce independent workspace
            # rows and do not form a cooperative two-CTA MMA.
            cur_fc1_counter_slot = (
                cutlass.Int32(self._cluster_n)
                * work_tile_info.cumulative_token_block_count
                + work_tile_info.tile_n_idx
            )
            cur_fc2_expert_idx = work_tile_info.expert_idx

            if cutlass.const_expr(self._early_fc1_pub):
                if cur_was_linear1:
                    if cutlass.const_expr(self._mxfp4_fc1_ready_k256):
                        # Match the store helper's full-warp election. Its
                        # issuing lane fully drains BOTH data and scales
                        # before publishing this post-SwiGLU K64 fragment.
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)
                        cute.arch.fence_acq_rel_gpu()
                        if local_warp_idx == cutlass.Int32(0):
                            with cute.arch.elect_one():
                                output_n_tile = (
                                    work_tile_info.tile_m_idx
                                    * cutlass.Int32(self._wgmma_fragment_count)
                                    + n_half
                                )
                                _red_or_release_gpu_b64(
                                    gmem_fc1_done_counter.iterator
                                    + cur_fc1_counter_slot,
                                    Int64(1) << output_n_tile,
                                )
                    else:
                        # Preserve ordinary FP8 / whole-tile code exactly.
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        cute.arch.fence_acq_rel_gpu()
                        if local_warp_idx == cutlass.Int32(0):
                            if cute.arch.lane_idx() == cutlass.Int32(0):
                                _early_ptr = (
                                    gmem_fc1_done_counter.iterator
                                    + cur_fc1_counter_slot
                                )
                                red_add_release_gpu_s32(_early_ptr, cutlass.Int32(1))

            if cutlass.const_expr(self._pingpong_order):
                # Match CUTLASS ping-pong ordering: retire the current FC1
                # bulk-store group and hand epilogue ownership to the peer WG
                # before waiting for the descriptor two positions ahead.
                if _iket_active:
                    if cur_was_linear1:
                        iket.range_push("pp_swap_fc1_retire_handoff")
                    else:
                        iket.range_push("pp_swap_fc2_retire_handoff")
                if cur_was_linear1:
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    cute.arch.fence_acq_rel_gpu()
                    flag_tracker = flag_tracker.accumulate(
                        work_tile_info.phase,
                        1,
                        (gmem_fc1_done_counter.iterator + cur_fc1_counter_slot).toint(),
                    )
                else:
                    if cutlass.const_expr(self._token_back_by_dispatch):
                        # All four warps must finish STG before the leader
                        # publishes this task's FC2 completion counter.
                        task_tile_boundary_bar.arrive_and_wait()
                        cute.arch.fence_acq_rel_gpu()
                        fc2_flag_addr = (
                            token_comm_args.fc2_done_counter.iterator
                            + cur_fc2_expert_idx
                        ).toint()
                    else:
                        fc2_flag_addr = Int64(0)
                    flag_tracker = flag_tracker.accumulate(
                        work_tile_info.phase,
                        1,
                        fc2_flag_addr,
                        not self._token_back_by_dispatch,
                    )
                epi_wg_order_state = epi_wg_order_barrier.arrive(epi_wg_order_state)
                if _iket_active:
                    iket.range_pop()

            if _iket_active:
                iket.range_push("swapab_sched_consume_next")
            if cutlass.const_expr(self._pingpong):
                (
                    sched_consumer,
                    work_tile_info,
                    ab_consumer_state,
                ) = consume_next_pingpong_work(
                    sched_consumer,
                    ab_consumer_state,
                    k_tile_cnt_fc1,
                    k_tile_cnt_fc2,
                )
            else:
                work_tile_info = sched_consumer.consume_work()
            if _iket_active:
                iket.range_pop()

            # Drain fc1 TMA/STG stores before publishing the fc1-done counter.
            # Offload: the empty-warp server owns the drain + publish, so the
            # epilogue neither drains nor accumulates here.
            if cur_was_linear1 and cutlass.const_expr(
                not self._pingpong_order
                and not self._fc1_store_offload
                and not self._mxfp4_fc1_ready_k256
            ):
                if _iket_active:
                    iket.range_push("swapab_fc1_store_drain")
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.fence_acq_rel_gpu()
                if _iket_active:
                    iket.range_pop()

            if _iket_active:
                iket.range_push("swapab_task_boundary_barrier")
            if cutlass.const_expr(not self._pingpong):
                task_tile_boundary_bar.arrive_and_wait()
            if _iket_active:
                iket.range_pop()

            if cur_was_linear1 and cutlass.const_expr(not self._pingpong_order):
                if _iket_active:
                    iket.range_push("swapab_fc1_done_flag_accumulate")
                if cutlass.const_expr(
                    not self._early_fc1_pub and not self._fc1_store_offload
                ):
                    flag_tracker = flag_tracker.accumulate(
                        work_tile_info.phase,
                        self._epi_fc1_batch,
                        (gmem_fc1_done_counter.iterator + cur_fc1_counter_slot).toint(),
                    )
                if _iket_active:
                    iket.range_pop()
            elif cutlass.const_expr(not self._pingpong_order):
                if cutlass.const_expr(self._token_back_by_dispatch):
                    # Fence before (deferred) counter release: make the fc2
                    # pool-output STG writes device-visible.  The release
                    # atomic in flag_tracker.fire() then signals completion.
                    cute.arch.fence_acq_rel_gpu()
                    fc2_flag_addr = (
                        token_comm_args.fc2_done_counter.iterator + cur_fc2_expert_idx
                    ).toint()
                else:
                    fc2_flag_addr = Int64(0)
                no_fire: cutlass.Constexpr = not self._token_back_by_dispatch
                if _iket_active:
                    iket.range_push("swapab_fc2_done_flag_accumulate")
                flag_tracker = flag_tracker.accumulate(
                    work_tile_info.phase,
                    self._epi_fc2_batch,
                    fc2_flag_addr,
                    no_fire,
                )
                if _iket_active:
                    iket.range_pop()

        if cutlass.const_expr(self._fc1_store_offload):
            # Retire this WG's slot and send a sentinel so the server counts
            # one finished WG (valid_tokens = -1).
            cute.arch.mbarrier_wait(
                fc1_offload_empty_mbar_ptr + warpgroup_idx,
                fc1_off_phase & cutlass.Int32(1),
            )
            if local_warp_idx == cutlass.Int32(0):
                if cute.arch.lane_idx() == cutlass.Int32(0):
                    fc1_offload_mbox[warpgroup_idx * cutlass.Int32(5) + 3] = Int64(-1)
            cute.arch.mbarrier_arrive(fc1_offload_full_mbar_ptr + warpgroup_idx)

        if _iket_active:
            iket.range_push("swapab_done_flag_fire_tail")
        flag_tracker.fire()
        if _iket_active:
            iket.range_pop()
