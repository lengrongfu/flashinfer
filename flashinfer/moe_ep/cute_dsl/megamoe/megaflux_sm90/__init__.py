# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Maintained SM90 MegaFlux executor development entry points (explicit opt-in)."""


def bind_identity_plan(workspace, plan):
    """Bind an offline R=0 Plan to a Hopper MXFP4 workspace before warmup."""
    from .identity import bind_identity_plan as bind

    return bind(workspace, plan)
