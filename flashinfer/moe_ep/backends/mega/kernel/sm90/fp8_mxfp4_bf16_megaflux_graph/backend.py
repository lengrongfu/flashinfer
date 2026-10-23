# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Device epochs and a fully joined captured transfer/compute DAG."""

from ......core.kernel.registry import register_mega_kernel
from ..fp8_mxfp4_bf16_megaflux_pipeline.backend import Sm90MegaFluxPipelineBackend
from ..fp8_mxfp4_bf16_pull_cutedsl.backend import Sm90PullMxfp4MegaKernelBackend


@register_mega_kernel("sm90_fp8_mxfp4_bf16_megaflux_graph")
class Sm90MegaFluxGraphBackend(Sm90MegaFluxPipelineBackend):
    @classmethod
    def kernel_name(cls):
        return "sm90_fp8_mxfp4_bf16_megaflux_graph"

    def _allocate_workspace(self, fleet_params):
        workspace = super()._allocate_workspace(fleet_params)
        workspace._megaflux_pipeline.graph_mode = True
        return workspace

    def validate_forward(self, t, fleet_params, *, quantize_input):
        return Sm90PullMxfp4MegaKernelBackend.validate_forward(
            self, t, fleet_params, quantize_input=quantize_input
        )

    def validate_capture_ready(self, workspace, transformed_weights):
        import torch

        if torch.cuda.is_current_stream_capturing():
            state = workspace._megaflux_pipeline
            if (
                state.ready is None
                or workspace._frontend._mega is None
                or getattr(workspace, "_megaflux_transport", None) is None
            ):
                raise RuntimeError(
                    "warm all MegaFlux ranks, banks and kernels before capture"
                )

    def _before_planning(self, workspace):
        # Every previous invocation rejoins both transfer branches after compute.
        # On the caller's serialized stream the old DAG has retired in full.
        state = self._bind_pipeline_state(workspace)
        state.control(4)

    def _planner_epoch(self, workspace):
        return workspace._megaflux_pipeline.epoch

    def _prepare_transport_epoch(self, workspace):
        # Advance once on GPU; never fill a captured constant Python epoch.
        pass

    def compute(self, workspace, transformed_weights, *, output):
        result = super().compute(workspace, transformed_weights, output=output)
        workspace._megaflux_pipeline.join()
        return result

    def _forget_local_workspace_state(self, workspace):
        import torch

        # A captured DAG may execute on runtime-managed streams. Drain the
        # device context, not only the original capture-stream event objects.
        torch.cuda.synchronize()
        # Internal captured events can cease to be waitable after the Graph is
        # released. The context-wide drain already retired every branch.
        state = getattr(workspace, "_megaflux_pipeline", None)
        if state is not None:
            state.issued = False
        super()._forget_local_workspace_state(workspace)
