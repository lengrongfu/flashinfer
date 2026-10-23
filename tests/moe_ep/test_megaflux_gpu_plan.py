# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fieldwise GPU planner gates. Host reads here are test oracle inspection only."""

import random

import pytest
import torch

from flashinfer.moe_ep.megaflux_plan import PlanConfig, plan_forward, pack_plan_records
from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.planner import (
    DevicePlan,
    LocalHistogram,
    DeviceRoutes,
)


@pytest.mark.arch_hopper
@pytest.mark.parametrize("budget", [0, 1, 2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_gpu_plan_matches_every_cpu_field(budget, seed):
    rng = random.Random(seed)
    c = PlanConfig(16, 4, budget, 145, 3, weight_version=7)
    ids = []
    for rank in range(c.world_size):
        n = [0, 1, 65, 145][rank] if seed == 0 else rng.randrange(146)
        source = []
        for _ in range(n):
            experts = rng.sample(range(c.num_logical_experts), c.top_k)
            source.append([e if rng.random() > 0.25 else -1 for e in experts])
        ids.append(source)
    _check(ids, c)


@pytest.mark.arch_hopper
@pytest.mark.parametrize("budget", [0, 1, 2, 4])
@pytest.mark.parametrize("tokens", [0, 1, 31, 32, 33, 65, 129])
def test_gpu_hot_expert_global_tail(budget, tokens):
    c = PlanConfig(16, 4, budget, 129, 1)
    _check([[[0]] * tokens, [], [], []], c)


def _check(ids, c, capacity=None):
    expected = plan_forward(ids, c, invocation_epoch=17, block_capacity=capacity)
    histogram = LocalHistogram(c, device="cuda")
    source_counts = []
    for source in ids:
        padded = source + [[-1] * c.top_k] * (c.max_tokens_per_source - len(source))
        tensor = torch.tensor(padded, dtype=torch.int64, device="cuda").reshape(
            c.max_tokens_per_source, c.top_k
        )
        histogram.run(tensor)
        assert histogram.status.item() == 0
        ordinal = histogram.ordinal.cpu().tolist()
        seen = [0] * c.num_logical_experts
        for row, words in zip(padded, ordinal, strict=True):
            for e, at in zip(row, words, strict=True):
                assert at == (-1 if e < 0 else seen[e])
                if e >= 0:
                    seen[e] += 1
        assert histogram.counts.cpu().tolist() == seen
        source_counts.append(histogram.counts.clone())
    counts = torch.stack(source_counts)
    gpu = DevicePlan(c, device="cuda", block_capacity=capacity).build(counts, epoch=17)
    assert gpu.meta.cpu().tolist() == [
        int(expected.status),
        int(expected.overflow),
        len(expected.block_descriptors),
        len(expected.replica_descriptors),
        sum(expected.expert_counts),
        17,
    ]
    assert gpu.source_counts.cpu().tolist() == [
        list(x) for x in expected.source_expert_counts
    ]
    assert gpu.expert_counts.cpu().tolist() == list(expected.expert_counts)
    assert gpu.expert_block_offsets.cpu().tolist() == list(
        expected.expert_block_offsets
    )
    assert gpu.slot_to_expert.cpu().tolist() == [
        list(x) for x in expected.slot_to_expert
    ]
    assert gpu.statistics.cpu().tolist() == list(
        map(
            list,
            (
                expected.rank_task_counts,
                expected.num_active_physical_slots,
                expected.rank_padded_rows,
                expected.rank_sf_rows,
            ),
        )
    )
    if not expected.overflow:
        packed = pack_plan_records(expected)
        assert gpu.blocks[: len(packed.blocks)].cpu().tolist() == [
            list(x) for x in packed.blocks
        ]
        assert gpu.replicas[: len(packed.replicas)].cpu().tolist() == [
            list(x) for x in packed.replicas
        ]
    if not expected.overflow:
        from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.physical import (
            lower_receiver_metadata,
        )

        routing = DeviceRoutes(c, device="cuda")
        route_records = []
        expected_receivers = [
            lower_receiver_metadata(expected, r) for r in range(c.world_size)
        ]
        for rank, source in enumerate(ids):
            padded = source + [[-1] * c.top_k] * (c.max_tokens_per_source - len(source))
            tensor = torch.tensor(padded, dtype=torch.int64, device="cuda").reshape(
                c.max_tokens_per_source, c.top_k
            )
            histogram.run(tensor)
            routing.materialize(tensor, histogram, gpu, rank=rank)
            torch.testing.assert_close(
                routing.send_counts.cpu(),
                torch.stack([x[0][rank] for x in expected_receivers]),
            )
            torch.testing.assert_close(
                routing.send_words.cpu(),
                torch.stack([x[1][:, rank] for x in expected_receivers]),
            )
            route_records.extend(
                x for x in routing.local_records.cpu().tolist() if x[0] >= 0
            )
        assert sorted(route_records, key=lambda x: (x[3], x[4])) == [
            list(x) for x in packed.routes
        ]
    return gpu


@pytest.mark.arch_hopper
def test_gpu_capacity_failure_is_explicit_and_empty():
    _check([[[0]] * 65, [], [], []], PlanConfig(16, 4, 1, 129, 1), capacity=2)


@pytest.mark.arch_hopper
def test_gpu_rejects_invalid_and_duplicate_routes():
    c = PlanConfig(4, 2, 1, 2, 2)
    h = LocalHistogram(c, device="cuda")
    for ids in ([[0, 0], [-1, -1]], [[4, 1], [-1, -1]], [[-2, 1], [-1, -1]]):
        h.run(torch.tensor(ids, dtype=torch.int64, device="cuda"))
        assert h.status.item() == 1


@pytest.mark.arch_hopper
@pytest.mark.parametrize("budget", [0, 1, 2, 4])
def test_gpu_target_expert_count_multiple_hot_experts(budget):
    c = PlanConfig(384, 8, budget, 512, 6)
    _check([[[0, 1, 2, 3, 4, 5]] * 512 for _ in range(8)], c)
