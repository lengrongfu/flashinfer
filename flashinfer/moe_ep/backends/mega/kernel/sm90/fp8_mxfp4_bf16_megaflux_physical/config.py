# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Physical-slot identity configuration; unused replica slots are reserved."""

from dataclasses import dataclass

from ..fp8_mxfp4_bf16_megaflux_identity.config import Sm90_Mxfp4_MegaFluxIdentityConfig


@dataclass
class Sm90_Mxfp4_MegaFluxPhysicalConfig(Sm90_Mxfp4_MegaFluxIdentityConfig):
    kernel_name: str = "sm90_fp8_mxfp4_bf16_megaflux_physical"

    def _expected_kernel_name(self):
        return "sm90_fp8_mxfp4_bf16_megaflux_physical"

    def _validate_plan_kind(self):
        if self.plan.replica_descriptors:
            raise ValueError("physical-slot identity config does not copy replicas")
