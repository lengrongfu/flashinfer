# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""NVLink mapped symmetric export reads with full-leg pipeline gates."""

import pytest


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
@pytest.mark.parametrize("budget", [0, 1, 2, 4])
@pytest.mark.parametrize("routing", ["sparse_owner", "zero_source", "all_masked"])
def test_selective_pipeline(monkeypatch, budget, routing):
    from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref
    from tests.moe_ep.test_megaflux_identity import _run_offline_backend

    monkeypatch.setattr(ref, "TOKENS_PER_RANK", 129)
    _run_offline_backend(
        routing,
        physical_budget=budget,
        online=True,
        pipeline=True,
        selective=True,
        delays=(20000, 20000, 20000, 20000),
    )


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
def test_selective_gated_forward(monkeypatch):
    from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref
    from tests.moe_ep.test_megaflux_identity import _run_offline_backend

    monkeypatch.setattr(ref, "TOKENS_PER_RANK", 129)
    _run_offline_backend(
        "sparse_owner",
        physical_budget=1,
        online=True,
        pipeline=True,
        selective=True,
        benefit_reject=True,
        delays=(0, 20000, 0, 0),
    )


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
def test_selective_gated_accepts_improving_plan(monkeypatch):
    from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref
    from tests.moe_ep.test_megaflux_identity import _run_offline_backend

    monkeypatch.setattr(ref, "TOKENS_PER_RANK", 129)
    _run_offline_backend(
        "sparse_owner",
        physical_budget=1,
        online=True,
        pipeline=True,
        selective=True,
        benefit_accept=True,
        delays=(0, 20000, 0, 0),
    )
