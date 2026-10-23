# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in static synchronous single-replica bringup configuration."""

from dataclasses import dataclass
from ..fp8_mxfp4_bf16_megaflux_physical.config import Sm90_Mxfp4_MegaFluxPhysicalConfig


@dataclass
class Sm90_Mxfp4_MegaFluxStaticConfig(Sm90_Mxfp4_MegaFluxPhysicalConfig):
    kernel_name: str = "sm90_fp8_mxfp4_bf16_megaflux_static"

    def _expected_kernel_name(self):
        return "sm90_fp8_mxfp4_bf16_megaflux_static"

    def _validate_plan_kind(self):
        if self.plan.config.replica_budget < 1:
            raise ValueError("static replica transport requires positive R")
        if len(self.plan.replica_descriptors) > 1:
            raise ValueError("static bringup supports at most one replica")
