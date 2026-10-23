# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Owner-streamed oracle must match the established global reference exactly."""

import os

import pytest
import torch
import torch.distributed as dist


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
def test_owner_streamed_reference_matches_global():
    from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl.tuner import (
        create_tuning_weights,
    )

    if "RANK" not in os.environ:
        pytest.skip("launch with torchrun")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("nccl")
    try:
        raw = create_tuning_weights(
            local_experts=ref.LOCAL_EXPERTS,
            hidden=ref.HIDDEN,
            intermediate=ref.INTERMEDIATE,
            rank=rank,
            device="cuda",
        )
        hidden = ref._all_gather_stack(
            torch.randn(3, ref.HIDDEN, device="cuda", dtype=torch.bfloat16)
        )
        ids = ref._all_gather_stack(
            torch.tensor(
                [[rank * ref.LOCAL_EXPERTS, ((rank + 1) % world) * ref.LOCAL_EXPERTS]]
                * 3,
                device="cuda",
            )
        )
        scores = torch.full(ids.shape, 0.5, device="cuda")
        expected = ref._global_route_reference(
            hidden,
            ids,
            scores,
            ref._gather_raw_weights(raw),
            require_all_experts=False,
            return_terms=True,
        )
        owned = ref._RawGlobalWeights(
            **{
                name: getattr(raw, name).unsqueeze(0)
                for name in ("w13", "w2", "w13_scale", "w2_scale")
            }
        )
        actual = ref._global_route_reference(
            hidden,
            ids,
            scores,
            owned,
            require_all_experts=False,
            owned_rank=rank,
            return_terms=True,
        )
        dist.all_reduce(actual, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        if created:
            dist.destroy_process_group()
