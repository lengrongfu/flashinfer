# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Explicit SM90 offline identity Plan backend."""

from typing import Any

from ......core.kernel.registry import register_mega_kernel
from ..fp8_mxfp4_bf16_pull_cutedsl.backend import Sm90PullMxfp4MegaKernelBackend
from .config import Sm90_Mxfp4_MegaFluxIdentityConfig


@register_mega_kernel("sm90_fp8_mxfp4_bf16_megaflux_identity")
class Sm90MegaFluxIdentityBackend(Sm90PullMxfp4MegaKernelBackend):
    def __init__(self, config: Sm90_Mxfp4_MegaFluxIdentityConfig):
        super().__init__(config)
        assert config.plan is not None
        self._identity_plan = config.plan
        self._planned_ids: Any = None

    @classmethod
    def kernel_name(cls):
        return "sm90_fp8_mxfp4_bf16_megaflux_identity"

    def validate_init(self, bootstrap, fleet_params):
        super().validate_init(bootstrap, fleet_params)
        p = self._identity_plan.config
        if (p.num_logical_experts, p.world_size, p.max_tokens_per_source) != (
            fleet_params.num_experts,
            bootstrap.world_size,
            fleet_params.max_tokens_per_rank,
        ):
            raise ValueError("identity Plan and fleet shapes differ")

    def validate_forward(self, t, fleet_params, *, quantize_input):
        import torch

        super().validate_forward(t, fleet_params, quantize_input=quantize_input)
        if self._planned_ids is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm the offline identity backend before capture")
            c = self._identity_plan.config
            ids = [[-1] * c.top_k for _ in range(c.max_tokens_per_source)]
            for block in self._identity_plan.block_descriptors:
                for route in block.routes:
                    if route.source_rank == self.ep_rank:
                        ids[route.source_token][route.source_topk_slot] = (
                            block.logical_expert
                        )
            self._planned_ids = torch.tensor(
                ids, dtype=t.topk_ids.dtype, device=t.topk_ids.device
            )
        n = t.topk_ids.shape[0]
        matches = (t.topk_ids == self._planned_ids[:n]).all()
        matches = matches & (self._planned_ids[n:] == -1).all()
        # Device assertion is capture-compatible and orders before dispatch.
        # A violated offline contract aborts execution; it is not local fallback.
        torch._assert_async(matches, "routing differs from the bound identity Plan")

    def _allocate_workspace(self, fleet_params):
        from ......cute_dsl.megamoe.megaflux_sm90 import bind_identity_plan

        workspace = super()._allocate_workspace(fleet_params)
        try:
            bind_identity_plan(workspace, self._identity_plan)
        except Exception:
            workspace.destroy()
            raise
        return workspace
