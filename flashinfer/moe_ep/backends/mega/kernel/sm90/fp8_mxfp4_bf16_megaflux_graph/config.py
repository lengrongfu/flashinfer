# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Graph qualification of the fixed-capacity W1/W2 pipeline."""

from dataclasses import dataclass
from ..fp8_mxfp4_bf16_megaflux_pipeline.config import Sm90_Mxfp4_MegaFluxPipelineConfig


@dataclass
class Sm90_Mxfp4_MegaFluxGraphConfig(Sm90_Mxfp4_MegaFluxPipelineConfig):
    kernel_name: str = "sm90_fp8_mxfp4_bf16_megaflux_graph"

    def _expected_kernel_name(self):
        return "sm90_fp8_mxfp4_bf16_megaflux_graph"
