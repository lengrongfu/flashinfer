# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Same-Plan synchronous/overlap comparison with real delay injection."""

import pytest


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "delays",
    [
        (0, 0, 0, 0),
        (20000, 0, 0, 0),
        (0, 20000, 0, 0),
        (0, 0, 20000, 0),
        (0, 0, 0, 20000),
    ],
)
@pytest.mark.parametrize(
    "routing_pattern", ["sparse_owner", "zero_source", "all_masked"]
)
def test_pipeline_full_leg_delay_and_progress(monkeypatch, delays, routing_pattern):
    from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref
    from tests.moe_ep.test_megaflux_identity import _run_offline_backend

    monkeypatch.setattr(ref, "TOKENS_PER_RANK", 129)
    _run_offline_backend(
        routing_pattern, physical_budget=1, online=True, pipeline=True, delays=delays
    )


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
@pytest.mark.parametrize("budget", [0, 2, 4])
def test_pipeline_expanded_replica_capacity(monkeypatch, budget):
    from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref
    from tests.moe_ep.test_megaflux_identity import _run_offline_backend

    monkeypatch.setattr(ref, "TOKENS_PER_RANK", 129)
    _run_offline_backend(
        "sparse_owner",
        physical_budget=budget,
        online=True,
        pipeline=True,
        delays=(0, 20000, 0, 0),
    )


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
def test_pipeline_resource_audit(monkeypatch):
    from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref
    from tests.moe_ep.test_megaflux_identity import _run_offline_backend

    monkeypatch.setattr(ref, "TOKENS_PER_RANK", 129)
    _run_offline_backend(
        "sparse_owner",
        physical_budget=1,
        online=True,
        pipeline=True,
        delays=(0, 20000, 0, 0),
        resource_audit=True,
    )


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
def test_pipeline_e384_k6_identity_and_full_leg_progress(monkeypatch):
    from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref
    from tests.moe_ep.test_megaflux_identity import _run_offline_backend

    _, world, _ = ref._launcher_ranks()
    if world != 8:
        pytest.skip("E384/K6 acceptance uses EP8")
    monkeypatch.setattr(ref, "TOKENS_PER_RANK", 129)
    monkeypatch.setattr(ref, "LOCAL_EXPERTS", 48)
    _run_offline_backend(
        "sparse_owner",
        physical_budget=1,
        online=True,
        pipeline=True,
        delays=(0, 20000, 0, 0),
        top_k=6,
    )
