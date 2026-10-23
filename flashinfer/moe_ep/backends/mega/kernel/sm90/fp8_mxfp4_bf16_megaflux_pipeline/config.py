# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Explicit W1/W2 overlap configuration, before Graph qualification."""

from dataclasses import dataclass
from ..fp8_mxfp4_bf16_megaflux_online.config import Sm90_Mxfp4_MegaFluxOnlineConfig


@dataclass
class Sm90_Mxfp4_MegaFluxPipelineConfig(Sm90_Mxfp4_MegaFluxOnlineConfig):
    kernel_name: str = "sm90_fp8_mxfp4_bf16_megaflux_pipeline"
    selective_weight_copy: bool = False
    synchronize_weight_copy: bool = False
    communication_clusters: int = 2
    pipeline_delays_us: tuple[int, int, int, int] = (0, 0, 0, 0)

    def __post_init__(self):
        if self.communication_clusters < 2:
            raise ValueError("reserve at least two clusters for W1/W2 progress")
        if len(self.pipeline_delays_us) != 4 or any(
            d < 0 or d > 1_000_000 for d in self.pipeline_delays_us
        ):
            raise ValueError("four delay values in [0,1000000] microseconds required")
        super().__post_init__()

    def _expected_kernel_name(self):
        return "sm90_fp8_mxfp4_bf16_megaflux_pipeline"
