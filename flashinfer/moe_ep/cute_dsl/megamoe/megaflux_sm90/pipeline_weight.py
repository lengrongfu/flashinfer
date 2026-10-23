# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Attributed selected-weight helper with a full-leg readiness acquire.

Derived from kernel_fp8_glu_fc12_swapab.py:_issue_selected_weight_tma_tile.
TMA and offset pipeline behavior is unchanged after acquisition.
"""

from typing import Any, Callable

import cutlass
import cutlass.cute as cute
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
    hopper_mxfp4_pipeline_support,
)
from .pipeline_flags import acquire_leg

_support = hopper_mxfp4_pipeline_support()
BlockPhase = _support["BlockPhase"]
iket = _support["iket"]


class PipelineWeightMixin:
    _mf_ready_offset: int
    _mf_capacity: int
    _mf_epoch_offset: int
    wgmma_warpgroup_count: int
    _uses_weight_aux_pipeline: Callable[..., Any]
    _tma_load_a_with_weight_sf_task_tile: Callable[..., Any]
    _tma_load_a_task_tile: Callable[..., Any]

    @cute.jit
    def _issue_selected_weight_tma_tile(
        self,
        range_name,
        tma_atom_weight: cute.CopyAtom,
        tma_tensor_weight: cute.Tensor,
        weight_sf_gemm: cute.Tensor,
        sA: cute.Tensor,
        sWeightSf: cute.Tensor,
        ab_producer,
        weight_sf_producer,
        work_tile_info,
        k_tile_cnt,
        ext,
        tidx,
        tma_cta_coord,
        tma_cta_layout,
        mcast_mask,
        _iket_active,
    ):
        """Issue one selected phase's weight/offset load."""
        leg = cutlass.Int64(0)
        if work_tile_info.phase != cutlass.Int32(BlockPhase.Linear1):
            leg = cutlass.Int64(1)
        acquire_leg(
            ext.fc1_ready_counter_ptr.toint()
            + cutlass.Int64(self._mf_ready_offset)
            + 8 * (leg * self._mf_capacity + cutlass.Int64(work_tile_info.expert_idx)),
            ext.fc1_ready_counter_ptr.toint() + cutlass.Int64(self._mf_epoch_offset),
        )
        if _iket_active:
            iket.range_push(range_name)
        real_a, desc_ptr_a = ext.get_gmem_tensor(
            "a",
            tma_tensor_weight,
            work_tile_info,
        )
        if cutlass.const_expr(self._uses_weight_aux_pipeline()):
            output_scale_block_base = work_tile_info.tile_m_idx * cutlass.Int32(
                self.wgmma_warpgroup_count
            )
            ab_producer, weight_sf_producer = self._tma_load_a_with_weight_sf_task_tile(
                tma_atom=tma_atom_weight,
                real_a=real_a,
                desc_ptr_a=desc_ptr_a,
                sA=sA,
                weight_sf_gemm=weight_sf_gemm,
                smem_weight_sf=sWeightSf,
                ab_producer=ab_producer,
                weight_sf_producer=weight_sf_producer,
                work_tile_info=work_tile_info,
                tile_m_idx=work_tile_info.tile_m_idx,
                output_scale_block_base=output_scale_block_base,
                k_tile_cnt=k_tile_cnt,
                tidx=tidx,
                tma_cta_coord=tma_cta_coord,
                tma_cta_layout=tma_cta_layout,
                mcast_mask=mcast_mask,
                _iket_active=_iket_active,
            )
        else:
            ab_producer = self._tma_load_a_task_tile(
                tma_atom_weight,
                real_a,
                desc_ptr_a,
                sA,
                ab_producer,
                work_tile_info.tile_m_idx,
                k_tile_cnt,
                tma_cta_coord,
                tma_cta_layout,
                mcast_mask,
                _iket_active,
            )
        if _iket_active:
            iket.range_pop()
        return ab_producer, weight_sf_producer
