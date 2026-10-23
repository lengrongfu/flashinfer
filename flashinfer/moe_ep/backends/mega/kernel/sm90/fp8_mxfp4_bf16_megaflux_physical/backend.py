# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Owner-only execution through an expanded fixed physical weight bank."""

from ......core.kernel.registry import register_mega_kernel
from ..fp8_mxfp4_bf16_megaflux_identity.backend import Sm90MegaFluxIdentityBackend
from ..fp8_mxfp4_bf16_pull_cutedsl.backend import Sm90PullMxfp4MegaKernelBackend


@register_mega_kernel("sm90_fp8_mxfp4_bf16_megaflux_physical")
class Sm90MegaFluxPhysicalBackend(Sm90MegaFluxIdentityBackend):
    @classmethod
    def kernel_name(cls):
        return "sm90_fp8_mxfp4_bf16_megaflux_physical"

    def _allocate_workspace(self, fleet_params):
        from ......cute_dsl.megamoe.megaflux_sm90.physical import bind_physical_plan

        workspace = Sm90PullMxfp4MegaKernelBackend._allocate_workspace(
            self, fleet_params
        )
        try:
            bind_physical_plan(workspace, self._identity_plan)
        except Exception:
            workspace.destroy()
            raise
        return workspace

    def _prepare_bank(self, workspace, transformed_weights):
        from ......cute_dsl.megamoe.megaflux_sm90.physical import (
            allocate_physical_weight_bank,
        )

        bank = getattr(workspace, "_megaflux_weight_bank", None)
        source_key = tuple(t.data_ptr() for leg in transformed_weights for t in leg)
        if bank is None:
            p = self._identity_plan.config
            bank = allocate_physical_weight_bank(
                transformed_weights,
                owned=p.num_owned_experts_per_rank,
                capacity=p.physical_slot_capacity,
            )
            workspace._megaflux_weight_bank = bank
            workspace._megaflux_source_key = source_key
        elif source_key != workspace._megaflux_source_key:
            raise ValueError("physical bank does not support runtime weight updates")
        return bank

    def compute(self, workspace, transformed_weights, *, output):
        bank = self._prepare_bank(workspace, transformed_weights)
        return super().compute(workspace, bank, output=output)

    def _forget_local_workspace_state(self, workspace):
        super()._forget_local_workspace_state(workspace)
        workspace._megaflux_weight_bank = None
