# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Offline single replica, full synchronous W1/W2 transfer each invocation."""

from ......core.kernel.registry import register_mega_kernel
from ..fp8_mxfp4_bf16_megaflux_physical.backend import Sm90MegaFluxPhysicalBackend
from ..fp8_mxfp4_bf16_pull_cutedsl.backend import Sm90PullMxfp4MegaKernelBackend


@register_mega_kernel("sm90_fp8_mxfp4_bf16_megaflux_static")
class Sm90MegaFluxStaticBackend(Sm90MegaFluxPhysicalBackend):
    @classmethod
    def kernel_name(cls):
        return "sm90_fp8_mxfp4_bf16_megaflux_static"

    def _allocate_workspace(self, fleet_params):
        from ......cute_dsl.megamoe.megaflux_sm90.static import bind_static_plan

        workspace = Sm90PullMxfp4MegaKernelBackend._allocate_workspace(
            self, fleet_params
        )
        try:
            bind_static_plan(workspace, self._identity_plan)
        except Exception:
            workspace.destroy()
            raise
        return workspace

    def validate_forward(self, t, fleet_params, *, quantize_input):
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("static synchronous replica transport cannot capture")
        return super().validate_forward(t, fleet_params, quantize_input=quantize_input)

    def compute(self, workspace, transformed_weights, *, output):
        from ......cute_dsl.megamoe.megaflux_sm90.static import StaticReplicaTransport

        bank = self._prepare_bank(workspace, transformed_weights)
        transport = getattr(workspace, "_megaflux_transport", None)
        if transport is None:
            transport = StaticReplicaTransport(self._identity_plan, bank, self.ep_rank)
            workspace._megaflux_transport = transport
        transport.copy(self._identity_plan, bank)
        return Sm90PullMxfp4MegaKernelBackend.compute(
            self, workspace, bank, output=output
        )

    def rebind_plan(self, workspace, plan):
        """Collective offline switch after all old consumers have retired.

        Every PE must call with the same Plan. Geometry, weight version and
        device addresses stay fixed. This does not implement online planning.
        """
        import torch
        from ......cute_dsl.megamoe.megaflux_sm90.physical import (
            lower_receiver_metadata,
        )
        from ......cute_dsl.megamoe.megaflux_sm90.static import StaticReplicaTransport
        from ......megaflux_plan import pack_plan_records

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("rebind static Plans outside capture")
        old = self._identity_plan
        if plan.config != old.config or plan.invocation_epoch <= old.invocation_epoch:
            raise ValueError("static rebind needs same geometry and a newer epoch")
        if len(plan.replica_descriptors) > 1:
            raise ValueError("static bringup supports at most one replica")
        pack_plan_records(plan)
        counts, words = lower_receiver_metadata(plan, self.ep_rank)
        StaticReplicaTransport.rendezvous()
        frontend = workspace._frontend
        frontend._identity_counts.copy_(counts)
        frontend._identity_totals.copy_(counts.sum(dim=0))
        frontend._identity_words.copy_(words)
        self._planned_ids = None
        self._identity_plan = plan
        self._kernel_config.plan = plan

    def _forget_local_workspace_state(self, workspace):
        transport = getattr(workspace, "_megaflux_transport", None)
        if transport is not None:
            transport.close()
            workspace._megaflux_transport = None
        super()._forget_local_workspace_state(workspace)
