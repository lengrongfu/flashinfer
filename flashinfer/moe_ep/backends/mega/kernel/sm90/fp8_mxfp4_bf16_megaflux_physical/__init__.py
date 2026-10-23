# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in physical-slot MegaFlux identity stage."""

from .backend import Sm90MegaFluxPhysicalBackend
from .config import Sm90_Mxfp4_MegaFluxPhysicalConfig

__all__ = ["Sm90MegaFluxPhysicalBackend", "Sm90_Mxfp4_MegaFluxPhysicalConfig"]
