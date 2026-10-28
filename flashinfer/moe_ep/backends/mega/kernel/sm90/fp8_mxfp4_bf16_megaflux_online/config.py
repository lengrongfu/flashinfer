# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GPU planning with synchronous transport; explicit correctness bringup."""

from dataclasses import dataclass
from ......cute_dsl.megamoe.megaflux_sm90.benefit import BenefitCosts, ReplicaAdmission
from ......megaflux_plan import PlanConfig, empty_workspace_plan
from ..fp8_mxfp4_bf16_megaflux_physical.config import Sm90_Mxfp4_MegaFluxPhysicalConfig


@dataclass
class Sm90_Mxfp4_MegaFluxOnlineConfig(Sm90_Mxfp4_MegaFluxPhysicalConfig):
    kernel_name: str = "sm90_fp8_mxfp4_bf16_megaflux_online"
    plan_config: PlanConfig | None = None
    requested_block_capacity: int | None = None
    debug_plan_agreement: bool = False
    benefit_costs: BenefitCosts | None = None
    min_replica_routes: int = 0
    replica_admission: ReplicaAdmission | None = None
    metadata_transport: str = "nccl"

    def __post_init__(self):
        if self.metadata_transport not in ("nccl", "nvlink"):
            raise ValueError("metadata_transport must be nccl or nvlink")
        if self.min_replica_routes < 0:
            raise ValueError("negative replica planning threshold")
        p = self.plan_config
        if (
            p is None
            or p.world_size > 8
            or p.num_logical_experts > 512
            or p.replica_budget > 4
        ):
            raise ValueError("online bringup requires PlanConfig with P<=8 E<=512 R<=4")
        if (
            self.requested_block_capacity is not None
            and self.requested_block_capacity < 0
        ):
            raise ValueError("negative requested block capacity")
        if self.plan is not None:
            raise ValueError(
                "online backend consumes routing on GPU, not an offline Plan"
            )
        # Shape-only empty bootstrap metadata; no routing is planned on CPU.
        self.plan = empty_workspace_plan(p)
        super().__post_init__()

    def _expected_kernel_name(self):
        return "sm90_fp8_mxfp4_bf16_megaflux_online"
