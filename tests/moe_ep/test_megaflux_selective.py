# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""NVLink mapped symmetric export reads with full-leg pipeline gates."""

import pytest


@pytest.mark.arch_hopper
def test_compact_cache_preserves_nonzero_owner_row():
    import torch

    from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.pipeline import (
        pipeline_module,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    module = pipeline_module()
    # Receiver rank 0 imports expert 4 from owner rank 1 into replica slot 4.
    q = torch.zeros((8, 2), dtype=torch.int64, device=device)
    q[4] = 1
    slots = torch.tensor(
        [[0, 1, 2, 3, 4], [4, 5, 6, 7, -1]], dtype=torch.int64, device=device
    )
    roots = [
        torch.full((28,), 0xA5, dtype=torch.uint8, device=device),
        torch.arange(28, dtype=torch.uint8, device=device),
    ]
    peers = torch.tensor([root.data_ptr() for root in roots], device=device)
    cache = torch.full((2, 28), 0x5A, dtype=torch.uint8, device=device)
    module.selective_get(q, slots, peers, cache, 0, 1, 1)
    payload = torch.zeros((5, 16), dtype=torch.uint8, device=device)
    offset = torch.zeros((5, 8), dtype=torch.uint8, device=device)
    residual = torch.zeros((5, 4), dtype=torch.uint8, device=device)
    for part in range(3):
        module.unpack_part(q, slots, payload, offset, residual, cache, 0, 1, part, True)
    actual = torch.cat((payload[4], offset[4], residual[4]))
    torch.testing.assert_close(actual, roots[1], atol=0, rtol=0)


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
