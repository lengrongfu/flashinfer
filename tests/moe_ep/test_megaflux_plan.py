# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only Plan conservation tests; no claim about GPU execution.

Run without the CUDA-dependent repository conftest:
python3 -m pytest --noconftest tests/moe_ep/test_megaflux_plan.py
"""

import importlib.util
from collections import Counter
from dataclasses import replace
from pathlib import Path
import random
import sys

import pytest


# Load the zero-dependency leaf without executing flashinfer.__init__.
_PATH = Path(__file__).resolve().parents[2] / "flashinfer/moe_ep/megaflux_plan.py"
_SPEC = importlib.util.spec_from_file_location("megaflux_plan_cpu", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
PlanConfig = _MODULE.PlanConfig
PlanStatus = _MODULE.PlanStatus
plan_forward = _MODULE.plan_forward


def _verify(ids, c, plan):
    assert plan.status == PlanStatus.OK
    assert not plan.overflow
    expected = Counter(
        (e, r, t, k)
        for r, source in enumerate(ids)
        for t, row in enumerate(source)
        for k, e in enumerate(row)
        if e >= 0
    )
    actual = Counter(
        (
            b.logical_expert,
            route.source_rank,
            route.source_token,
            route.source_topk_slot,
        )
        for b in plan.block_descriptors
        for route in b.routes
    )
    assert actual == expected
    assert all(n == 1 for n in actual.values())
    packed = _MODULE.pack_plan_records(plan)
    assert len(packed.blocks) == len(plan.block_descriptors)
    assert len(packed.replicas) == len(plan.replica_descriptors)
    assert len(packed.routes) == sum(expected.values())
    assert all(
        len(row) == len(_MODULE.BLOCK_COLUMNS) and row[-1] == 0 for row in packed.blocks
    )
    assert all(
        len(row) == len(_MODULE.REPLICA_COLUMNS) and row[-2:] == (0, 0)
        for row in packed.replicas
    )
    for rank, token, topk, block, row in packed.routes:
        descriptor = plan.block_descriptors[block]
        assert ids[rank][token][topk] == descriptor.logical_expert
        assert descriptor.routes[row] == _MODULE.Route(rank, token, topk)
    assert sum(plan.expert_counts) == sum(expected.values())
    assert plan.abi_version == 1
    assert plan.source_expert_counts == tuple(
        tuple(
            sum(e == expert for row in source for e in row)
            for expert in range(c.num_logical_experts)
        )
        for source in ids
    )
    assert len(plan.expert_block_offsets) == c.num_logical_experts + 1
    assert plan.expert_block_offsets[-1] == len(plan.block_descriptors)
    assert len(plan.block_descriptors) <= c.block_capacity
    for e in range(c.num_logical_experts):
        blocks = [b for b in plan.block_descriptors if b.logical_expert == e]
        assert [b.expert_block for b in blocks] == list(range(len(blocks)))
        assert sum(b.valid_rows < c.beta for b in blocks) <= 1
        assert sum(b.valid_rows for b in blocks) == plan.expert_counts[e]
        ordered_routes = [route for b in blocks for route in b.routes]
        assert ordered_routes == sorted(ordered_routes)
        assert (
            len(blocks)
            == plan.expert_block_offsets[e + 1] - plan.expert_block_offsets[e]
        )
    for r in range(c.world_size):
        own = c.num_owned_experts_per_rank
        slots = plan.slot_to_expert[r]
        assert len(slots) == c.physical_slot_capacity
        assert slots[:own] == tuple(range(r * own, (r + 1) * own))
        occupied = [e for e in slots if e >= 0]
        assert len(occupied) == len(set(occupied))
        replicas = [d for d in plan.replica_descriptors if d.receiver_rank == r]
        assert len(replicas) <= c.replica_budget
        assert len(replicas) == sum(e >= 0 for e in slots[own:])
        rank_blocks = [b for b in plan.block_descriptors if b.execution_rank == r]
        assert plan.rank_task_counts[r] == len(rank_blocks)
        assert plan.num_active_physical_slots[r] == len(
            {b.physical_slot for b in rank_blocks}
        )
        rows, sf_rows = set(), set()
        for b in rank_blocks:
            assert slots[b.physical_slot] == b.logical_expert
            assert 0 < b.valid_rows <= c.beta
            assert b.valid_rows == len(b.routes)
            assert b.destination_row_offset % c.beta == 0
            assert b.destination_row_offset + c.beta <= plan.rank_padded_rows[r]
            assert b.destination_sf_row_offset + b.valid_rows <= plan.rank_sf_rows[r]
            block_rows = set(
                range(b.destination_row_offset, b.destination_row_offset + b.valid_rows)
            )
            block_sf_rows = set(
                range(
                    b.destination_sf_row_offset,
                    b.destination_sf_row_offset + b.valid_rows,
                )
            )
            assert not rows.intersection(block_rows)
            assert not sf_rows.intersection(block_sf_rows)
            rows.update(block_rows)
            sf_rows.update(block_sf_rows)
        for s in {b.physical_slot for b in rank_blocks}:
            first = min(
                b.destination_sf_row_offset for b in rank_blocks if b.physical_slot == s
            )
            assert first % c.sf_padding == 0
        for d in replicas:
            assert slots[d.receiver_slot] == d.logical_expert
            assert d.owner_rank == d.logical_expert // own
            assert d.owner_slot == d.logical_expert % own
            assert d.weight_version == c.weight_version
            assert d.layout_id == c.layout_id
    identity = plan_forward(
        ids, c, invocation_epoch=plan.invocation_epoch, identity=True
    )
    assert max(plan.rank_task_counts) <= max(identity.rank_task_counts)


@pytest.mark.parametrize("budget", [0, 1, 2, 4])
@pytest.mark.parametrize("seed", range(12))
def test_random_ragged_masked_route_conservation(seed, budget):
    rng = random.Random(seed)
    c = PlanConfig(16, 4, budget, 145, 3, weight_version=7)
    ids = []
    for _ in range(c.world_size):
        source = []
        for _ in range(rng.randrange(c.max_tokens_per_source + 1)):
            choices = rng.sample(range(c.num_logical_experts), c.top_k)
            source.append([e if rng.random() > 0.25 else -1 for e in choices])
        ids.append(source)
    plan = plan_forward(ids, c, invocation_epoch=17)
    _verify(ids, c, plan)
    assert plan == plan_forward(ids, c, invocation_epoch=17)


@pytest.mark.parametrize("tokens", [0, 1, 31, 32, 33, 63, 64, 65, 129])
@pytest.mark.parametrize("budget", [0, 1, 2, 4])
def test_hot_expert_global_tail_and_empty_sources(tokens, budget):
    c = PlanConfig(16, 4, budget, 129, 1)
    ids = [[[0]] * tokens, [], [], []]
    plan = plan_forward(ids, c, invocation_epoch=1)
    _verify(ids, c, plan)
    if budget and tokens >= 65:
        assert plan.replica_descriptors
        assert any(plan.rank_task_counts[1:])


def test_move_not_capped_by_donor_overload():
    # tau=3; rank 0 moves three blocks to rank 2 despite overload of two.
    # It then receives one block from expert 1. An overload cap would leave
    # expert 0's allocation at [3,0,2] instead of [2,0,3].
    c = PlanConfig(3, 3, 1, 160, 1)
    q = _MODULE._search([5, 4, 0], c, False, False)
    assert q == [[2, 0, 3], [1, 3, 0], [0, 0, 0]]


@pytest.mark.parametrize("budget", [0, 1, 2, 4])
def test_multiple_hot_experts_fanout_and_slot_exhaustion(budget):
    c = PlanConfig(32, 8, budget, 257, 3)
    # All three hot experts are owned by rank 0; legal distinct top-k.
    # Ragged sources include an entirely empty rank that can execute replicas.
    ids = [[[0, 1, 2]] * n for n in [257, 193, 129, 65, 33, 1, 0, 31]]
    plan = plan_forward(ids, c, invocation_epoch=3)
    _verify(ids, c, plan)
    assert plan == plan_forward(ids, c, invocation_epoch=3)
    if budget:
        assert len(plan.replica_descriptors) > 1
        assert max(plan.rank_task_counts) < max(
            plan_forward(ids, c, invocation_epoch=3, identity=True).rank_task_counts
        )


def test_identity_and_capacity_failure_are_explicit():
    c = PlanConfig(8, 2, 2, 65, 2)
    ids = [[[0, 1]] * 65, []]
    plan = plan_forward(ids, c, invocation_epoch=2, identity=True)
    _verify(ids, c, plan)
    assert not plan.replica_descriptors
    assert all(b.execution_rank == 0 for b in plan.block_descriptors)
    failure = plan_forward(ids, c, invocation_epoch=2, block_capacity=5)
    assert failure.status == PlanStatus.CAPACITY_EXCEEDED
    assert failure.overflow
    assert not failure.block_descriptors
    assert not failure.replica_descriptors
    assert failure.rank_task_counts == (0, 0)
    assert failure.expert_counts == plan.expert_counts
    with pytest.raises(ValueError, match="unsuccessful Plan"):
        _MODULE.pack_plan_records(failure)


@pytest.mark.parametrize(
    "ids", [[[[0, 0]], []], [[[8, -1]], []], [[[-2, -1]], []], [[[0]], []], [[]]]
)
def test_invalid_routes_rejected(ids):
    with pytest.raises((ValueError, TypeError)):
        plan_forward(ids, PlanConfig(8, 2, 1, 8, 2), invocation_epoch=1)


def test_configuration_and_int64_capacity_validation():
    c = PlanConfig(384, 8, 2, 4096, 6)
    assert c.num_owned_experts_per_rank == 48
    assert c.physical_slot_capacity == 50
    assert c.max_assignments == 8 * 4096 * 6
    assert c.block_capacity == (c.max_assignments + 31) // 32 + 384
    for change in (
        {"beta": 128},
        {"world_size": 7},
        {"top_k": 385},
        {"replica_budget": -1},
    ):
        with pytest.raises(ValueError):
            replace(c, **change)
    with pytest.raises(OverflowError):
        replace(c, max_tokens_per_source=1 << 62)
    with pytest.raises(OverflowError):
        _MODULE.checked_nbytes(1 << 62, 8)
    with pytest.raises(ValueError):
        plan_forward([[]] * 8, c, invocation_epoch=0)
