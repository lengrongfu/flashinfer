# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Independent complete-leg streams and guarded persistent consumers."""

from ......core.kernel.registry import register_mega_kernel
from ..fp8_mxfp4_bf16_megaflux_online.backend import Sm90MegaFluxOnlineBackend


@register_mega_kernel("sm90_fp8_mxfp4_bf16_megaflux_pipeline")
class Sm90MegaFluxPipelineBackend(Sm90MegaFluxOnlineBackend):
    @classmethod
    def kernel_name(cls):
        return "sm90_fp8_mxfp4_bf16_megaflux_pipeline"

    def _allocate_workspace(self, fleet_params):
        from ......cute_dsl.megamoe.megaflux_sm90.pipeline import (
            PipelineFrontend,
            PipelineState,
        )

        workspace = super()._allocate_workspace(fleet_params)
        try:
            state = PipelineState(self._identity_plan.config, device=workspace.x.device)
            workspace._megaflux_pipeline = state
            workspace._frontend = PipelineFrontend(
                workspace._frontend.config,
                self._identity_plan,
                state,
                self._kernel_config.communication_clusters,
            )
        except Exception:
            workspace.destroy()
            raise
        return workspace

    def _make_transport(self, config, bank, workspace):
        from ......cute_dsl.megamoe.megaflux_sm90.pipeline import (
            PipelinedReplicaTransport,
        )

        return PipelinedReplicaTransport(
            config,
            bank,
            self.ep_rank,
            workspace._megaflux_pipeline,
            self._kernel_config,
            group=self.ep_comm_group,
        )

    def _before_planning(self, workspace):
        # The preceding transfer can still read the old q/slot map even if
        # its expert had no work. Retire it before rebuilding that storage.
        workspace._megaflux_pipeline.join()

    def _bind_pipeline_state(self, workspace):
        from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
            prepare_hopper_mxfp4_pipeline,
        )

        state = workspace._megaflux_pipeline
        if state.ready is None:
            views = prepare_hopper_mxfp4_pipeline(
                workspace, workspace._megaflux_weight_bank
            )
            state.__dict__.update(views)
        return state

    def _prepare_transport_epoch(self, workspace):
        self._bind_pipeline_state(workspace).epoch.fill_(workspace._megaflux_epoch)

    def _forget_local_workspace_state(self, workspace):
        super()._forget_local_workspace_state(workspace)
        workspace._megaflux_pipeline = None
