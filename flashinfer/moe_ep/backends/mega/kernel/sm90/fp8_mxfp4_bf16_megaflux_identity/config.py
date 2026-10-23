# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Frozen offline identity backend configuration."""

from dataclasses import dataclass
from typing import Literal

from ......megaflux_plan import ForwardPlan, PlanStatus
from ..fp8_mxfp4_bf16_pull_cutedsl.config import (
    Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
)


@dataclass
class Sm90_Mxfp4_MegaFluxIdentityConfig(Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig):
    """Static offline R=0 Plan; staged routing must match the supplied Plan.

    There is no online routing planner or automatic selection. Construct a
    new layer/workspace when changing the Plan in this development version.
    """

    kernel_name: str = "sm90_fp8_mxfp4_bf16_megaflux_identity"
    plan: ForwardPlan | None = None
    swap_ab: bool | None = True
    pingpong: bool | None = False
    mma_tiler_mnk: tuple[int, int, int] | None = (128, 32, 128)
    cluster_shape_mnk: tuple[int, int, int] | None = (1, 1, 1)
    dedup_dispatch: bool | None = False
    fold_producer_warps: bool | None = False
    token_back_mode: (
        Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"] | None
    ) = "epi_warps"

    def __post_init__(self):
        super().__post_init__()
        if self.kernel_name != "sm90_fp8_mxfp4_bf16_megaflux_identity":
            raise ValueError("identity config cannot select another backend")
        if self.plan is None or self.plan.status != PlanStatus.OK or self.plan.overflow:
            raise ValueError("a successful offline identity Plan is required")
        if self.plan.config.replica_budget != 0 or self.plan.replica_descriptors:
            raise ValueError("MegaFlux identity backend requires R=0")
        if self.plan.config.top_k != self.top_k:
            raise ValueError("Plan top_k differs from backend top_k")
        if self.knobs is not None:
            raise ValueError("identity backend does not support autotuning")
        if (
            self.swap_ab is not True
            or self.pingpong is not False
            or self.mma_tiler_mnk != (128, 32, 128)
            or self.cluster_shape_mnk != (1, 1, 1)
            or self.dedup_dispatch is not False
            or self.fold_producer_warps is not False
            or self.token_back_mode != "epi_warps"
        ):
            raise ValueError(
                "identity backend requires frozen no-dedup N32/C1x1 geometry"
            )
