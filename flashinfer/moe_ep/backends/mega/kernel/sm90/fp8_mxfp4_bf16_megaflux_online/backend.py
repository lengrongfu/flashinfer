# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GPU Plan and route generation, synchronous full-leg transport."""

from ......core.kernel.registry import register_mega_kernel
from ..fp8_mxfp4_bf16_megaflux_physical.backend import Sm90MegaFluxPhysicalBackend
from ..fp8_mxfp4_bf16_pull_cutedsl.backend import Sm90PullMxfp4MegaKernelBackend


@register_mega_kernel("sm90_fp8_mxfp4_bf16_megaflux_online")
class Sm90MegaFluxOnlineBackend(Sm90MegaFluxPhysicalBackend):
    @classmethod
    def kernel_name(cls):
        return "sm90_fp8_mxfp4_bf16_megaflux_online"

    def _allocate_workspace(self, fleet_params):
        import torch
        import torch.distributed as dist

        p = self._identity_plan.config
        manifest = torch.tensor(
            [
                p.num_logical_experts,
                p.world_size,
                p.replica_budget,
                p.max_tokens_per_source,
                p.top_k,
                p.weight_version,
                1,
            ],
            dtype=torch.int64,
            device="cuda",
        )
        all_manifests = torch.empty(
            p.world_size * manifest.numel(), dtype=torch.int64, device="cuda"
        )
        dist.all_gather_into_tensor(all_manifests, manifest, group=self.ep_comm_group)
        fleet = all_manifests.reshape(p.world_size, -1)
        geometry = [0, 1, 2, 3, 4, 6]
        torch._assert_async(
            fleet[:, geometry].eq(manifest[geometry]).all(),
            "inconsistent collective MegaFlux geometry/layout",
        )
        self._manifest_fallback = (
            (~fleet[:, 5].eq(manifest[5]).all()).reshape(1).to(torch.int32)
        )
        return super()._allocate_workspace(fleet_params)

    def validate_forward(self, t, fleet_params, *, quantize_input):
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("online synchronous bringup cannot capture")
        return Sm90PullMxfp4MegaKernelBackend.validate_forward(
            self, t, fleet_params, quantize_input=quantize_input
        )

    def compute(self, workspace, transformed_weights, *, output):
        from ......cute_dsl.megamoe.megaflux_sm90.planner import DistributedPlanner

        bank = self._prepare_bank(workspace, transformed_weights)
        planner = getattr(workspace, "_megaflux_planner", None)
        if planner is None:
            p = self._identity_plan.config
            planner = DistributedPlanner(
                p,
                device=workspace.topk_idx.device,
                group=self.ep_comm_group,
                requested_capacity=self._kernel_config.requested_block_capacity,
                debug_agreement=self._kernel_config.debug_plan_agreement,
            )
            workspace._megaflux_planner = planner
            workspace._megaflux_transport = self._make_transport(p, bank, workspace)
            workspace._megaflux_epoch = 0
        self._before_planning(workspace)
        workspace._megaflux_epoch += 1
        plan = planner.run(
            workspace.topk_idx,
            rank=self.ep_rank,
            epoch=workspace._megaflux_epoch,
            external_failure=self._manifest_fallback,
        )
        self._prepare_transport_epoch(workspace)
        workspace._megaflux_transport.copy(plan)
        frontend = workspace._frontend
        frontend._identity_counts.copy_(planner.routing.counts)
        frontend._identity_totals.copy_(planner.routing.counts.sum(dim=0))
        frontend._identity_words.copy_(planner.routing.words)
        return Sm90PullMxfp4MegaKernelBackend.compute(
            self, workspace, bank, output=output
        )

    def _make_transport(self, config, bank, workspace):
        from ......cute_dsl.megamoe.megaflux_sm90.static import DeviceReplicaTransport

        return DeviceReplicaTransport(config, bank, self.ep_rank)

    def _before_planning(self, workspace):
        pass

    def _prepare_transport_epoch(self, workspace):
        pass

    def _forget_local_workspace_state(self, workspace):
        transport = getattr(workspace, "_megaflux_transport", None)
        if transport is not None:
            transport.close()
            workspace._megaflux_transport = None
        workspace._megaflux_planner = None
        super()._forget_local_workspace_state(workspace)
