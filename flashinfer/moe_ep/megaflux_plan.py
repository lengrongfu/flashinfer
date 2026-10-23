# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-only MegaFlux forward Plan ABI and deterministic test oracle.

This module does not enable a backend or execute GPU work. It intentionally
has no Torch/CUDA imports. The eventual GPU planner must reproduce these
rules, including ties and the bounded refinement pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from operator import index
from typing import Sequence


_I64_MAX = (1 << 63) - 1
PLAN_ABI_VERSION = 1
# Wire records are signed int64. The final column is reserved and zero.
BLOCK_COLUMNS = (
    "logical_expert",
    "expert_block",
    "execution_rank",
    "physical_slot",
    "destination_row_offset",
    "valid_rows",
    "destination_sf_row_offset",
    "reserved",
)
REPLICA_COLUMNS = (
    "logical_expert",
    "owner_rank",
    "owner_slot",
    "receiver_rank",
    "receiver_slot",
    "weight_version",
    "layout_index",
    "reserved",
)
ROUTE_COLUMNS = (
    "source_rank",
    "source_token",
    "source_topk_slot",
    "block_index",
    "row_in_block",
)


def _integer(name: str, value: int, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result > _I64_MAX:
        raise OverflowError(f"{name} exceeds signed int64")
    if result < minimum:
        raise ValueError(f"{name} must be in [{minimum}, {_I64_MAX}]")
    return result


def _ceil(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple


def _align(value: int, multiple: int) -> int:
    return _ceil(value, multiple) * multiple


def checked_nbytes(count: int, element_bytes: int) -> int:
    """Check allocation arithmetic before converting Python ints to int64."""
    result = _integer("count", count) * _integer("element_bytes", element_bytes, 1)
    if result > _I64_MAX:
        raise OverflowError("allocation byte size exceeds signed int64")
    return result


class PlanStatus(IntEnum):
    OK = 0
    CAPACITY_EXCEEDED = 1


@dataclass(frozen=True)
class PlanConfig:
    num_logical_experts: int
    world_size: int
    replica_budget: int
    max_tokens_per_source: int
    top_k: int
    # Frozen first supported geometry: swap-AB (128,32,128), cluster (1,1,1).
    # beta is a token-cluster block, not the independent SF padding.
    beta: int = 32
    sf_padding: int = 128
    tactic_id: str = "sm90_mxfp4_m128_n32_k128_c1x1_v1"
    layout_id: str = "humming_sm90_m64_k128_gateup8_residual_x64_v1"
    weight_version: int = 0

    def __post_init__(self) -> None:
        for name in (
            "num_logical_experts",
            "world_size",
            "top_k",
            "beta",
            "sf_padding",
        ):
            _integer(name, getattr(self, name), 1)
        for name in ("replica_budget", "max_tokens_per_source", "weight_version"):
            _integer(name, getattr(self, name))
        if self.num_logical_experts % self.world_size:
            raise ValueError("logical experts must be divisible by world_size")
        if self.top_k > self.num_logical_experts:
            raise ValueError("top_k exceeds logical experts")
        if (
            self.beta != 32
            or self.sf_padding != 128
            or self.tactic_id != ("sm90_mxfp4_m128_n32_k128_c1x1_v1")
        ):
            raise ValueError("only the frozen N32/C1x1 tactic is supported")
        if not self.layout_id:
            raise ValueError("layout_id must be nonempty")
        checked_nbytes(self.max_assignments, 8 * len(ROUTE_COLUMNS))
        checked_nbytes(self.block_capacity, 64)
        checked_nbytes(self.block_capacity, self.sf_padding)
        checked_nbytes(self.world_size * self.physical_slot_capacity, 8)
        checked_nbytes(self.world_size * self.replica_budget, 64)
        checked_nbytes(self.world_size * self.num_logical_experts, 8)

    @property
    def num_owned_experts_per_rank(self) -> int:
        return self.num_logical_experts // self.world_size

    @property
    def physical_slot_capacity(self) -> int:
        return self.num_owned_experts_per_rank + self.replica_budget

    @property
    def max_assignments(self) -> int:
        return self.world_size * self.max_tokens_per_source * self.top_k

    @property
    def block_capacity(self) -> int:
        return _ceil(self.max_assignments, self.beta) + self.num_logical_experts


@dataclass(frozen=True, order=True)
class Route:
    source_rank: int
    source_token: int
    source_topk_slot: int


@dataclass(frozen=True)
class BlockDescriptor:
    logical_expert: int
    expert_block: int
    execution_rank: int
    physical_slot: int
    destination_row_offset: int
    valid_rows: int
    destination_sf_row_offset: int
    routes: tuple[Route, ...]


@dataclass(frozen=True)
class ReplicaDescriptor:
    logical_expert: int
    owner_rank: int
    owner_slot: int
    receiver_rank: int
    receiver_slot: int
    weight_version: int
    layout_id: str


@dataclass(frozen=True)
class ForwardPlan:
    """Reference representation of ABI v1.

    Device arrays use signed int64 for block descriptors (including row
    offsets), counts, prefixes and route ordinals. ``routes`` is CPU oracle
    data; it must be materialized as a separate fixed-capacity route table on
    device. String layout/tactic identifiers must be resolved collectively
    before capture. No tensor conversion or device upload occurs here.
    """

    config: PlanConfig
    invocation_epoch: int
    status: PlanStatus
    overflow: bool
    expert_counts: tuple[int, ...]
    source_expert_counts: tuple[tuple[int, ...], ...]
    expert_block_offsets: tuple[int, ...]
    slot_to_expert: tuple[tuple[int, ...], ...]
    block_descriptors: tuple[BlockDescriptor, ...]
    replica_descriptors: tuple[ReplicaDescriptor, ...]
    rank_task_counts: tuple[int, ...]
    num_active_physical_slots: tuple[int, ...]
    rank_padded_rows: tuple[int, ...]
    rank_sf_rows: tuple[int, ...]

    @property
    def abi_version(self) -> int:
        return PLAN_ABI_VERSION


@dataclass(frozen=True)
class PackedPlanRecords:
    """ABI v1 record arrays, ready for test-owned int64 tensor conversion.

    Only active prefixes are emitted; an executor must allocate fixed
    capacities before capture and copy into those arrays. Layout index 0
    refers to ``layout_ids[0]``, agreed across ranks before Plan commit.
    """

    blocks: tuple[tuple[int, ...], ...]
    replicas: tuple[tuple[int, ...], ...]
    routes: tuple[tuple[int, ...], ...]
    layout_ids: tuple[str, ...]


def pack_plan_records(plan: ForwardPlan) -> PackedPlanRecords:
    """Serialize successful reference records without Torch or device access."""
    if plan.status != PlanStatus.OK or plan.overflow:
        raise ValueError("cannot serialize an unsuccessful Plan for execution")
    blocks = tuple(
        (
            b.logical_expert,
            b.expert_block,
            b.execution_rank,
            b.physical_slot,
            b.destination_row_offset,
            b.valid_rows,
            b.destination_sf_row_offset,
            0,
        )
        for b in plan.block_descriptors
    )
    replicas = tuple(
        (
            d.logical_expert,
            d.owner_rank,
            d.owner_slot,
            d.receiver_rank,
            d.receiver_slot,
            d.weight_version,
            0,
            0,
        )
        for d in plan.replica_descriptors
    )
    routes = tuple(
        (route.source_rank, route.source_token, route.source_topk_slot, block, row)
        for block, b in enumerate(plan.block_descriptors)
        for row, route in enumerate(b.routes)
    )
    return PackedPlanRecords(blocks, replicas, routes, (plan.config.layout_id,))


def _loads(q: list[list[int]], p: int) -> list[int]:
    return [sum(row[r] for row in q) for r in range(p)]


def _replicas(q: list[list[int]], owned: int, p: int) -> int:
    return sum(q[e][r] > 0 for e in range(len(q)) for r in range(p) if r != e // owned)


def _search(
    blocks: list[int], c: PlanConfig, largest_home: bool, fanout: bool
) -> list[list[int]]:
    p, owned = c.world_size, c.num_owned_experts_per_rank
    q = [[b if r == e // owned else 0 for r in range(p)] for e, b in enumerate(blocks)]
    load = _loads(q, p)
    slots = [0] * p
    target = _ceil(sum(blocks), p)

    def move(e: int, receiver: int) -> bool:
        owner = e // owned
        amount = min(q[e][owner], max(0, target - load[receiver]))
        if not amount:
            return False
        if not q[e][receiver]:
            slots[receiver] += 1
        q[e][owner] -= amount
        q[e][receiver] += amount
        load[owner] -= amount
        load[receiver] += amount
        return True

    for _ in range(c.replica_budget + 2):
        changed = False

        # Priorities are frozen at round entry; rank breaks ties.
        def priority(r):
            value = (
                max(q[e][r] for e in range(r * owned, (r + 1) * owned))
                if largest_home
                else load[r] - target
            )
            return (-value, r)

        for donor in sorted(range(p), key=priority):
            if load[donor] <= target:
                continue
            home = range(donor * owned, (donor + 1) * owned)
            # Reuse existing copies before allocating any new replica.
            existing = sorted(
                ((e, r) for e in home for r in range(p) if r != donor and q[e][r]),
                key=lambda er: (load[er[1]], er[0], er[1]),
            )
            for e, r in existing:
                if load[donor] <= target:
                    break
                changed |= move(e, r)
            for e in sorted(home, key=lambda e: (-q[e][donor], e)):
                if load[donor] <= target:
                    break
                while q[e][donor] and load[donor] > target:
                    receivers = [
                        r
                        for r in range(p)
                        if r != donor
                        and not q[e][r]
                        and slots[r] < c.replica_budget
                        and load[r] < target
                    ]
                    if not receivers:
                        break
                    receiver = min(receivers, key=lambda r: (load[r], r))
                    changed |= move(e, receiver)
                    if not fanout:
                        break
        if not changed:
            break
    return q


def _assignment(blocks: list[int], c: PlanConfig, identity: bool) -> list[list[int]]:
    p, owned = c.world_size, c.num_owned_experts_per_rank
    if identity or not c.replica_budget:
        return [
            [b if r == e // owned else 0 for r in range(p)]
            for e, b in enumerate(blocks)
        ]
    candidates = [
        _search(blocks, c, home, fanout)
        for home in (False, True)
        for fanout in (False, True)
    ]
    # Enumeration order is the final deterministic candidate tie-break.
    q = min(candidates, key=lambda x: (max(_loads(x, p)), _replicas(x, owned, p)))
    load = _loads(q, p)
    ceiling = max(load)
    copies = sorted(
        (
            (e, r)
            for e in range(len(q))
            for r in range(p)
            if r != e // owned and q[e][r]
        ),
        key=lambda er: (q[er[0]][er[1]], er[0], er[1]),
    )
    for e, r in copies:
        amount, owner = q[e][r], e // owned
        # The owner always has storage, even when assigned no blocks.
        destinations = [
            d
            for d in range(p)
            if d != r and (d == owner or q[e][d]) and load[d] + amount <= ceiling
        ]
        if destinations:
            d = min(destinations, key=lambda d: (d != owner, load[d], d))
            q[e][d] += amount
            q[e][r] = 0
            load[d] += amount
            load[r] -= amount
    # Explicit bound: at most one transfer per original block. No new copies.
    for _ in range(sum(blocks)):
        donor = min(range(p), key=lambda r: (-load[r], r))
        receiver = min(range(p), key=lambda r: (load[r], r))
        if load[donor] - load[receiver] <= 1:
            break
        shared = [
            e
            for e in range(len(q))
            if q[e][donor] and (receiver == e // owned or q[e][receiver])
        ]
        if not shared:
            break
        e = min(shared)
        q[e][donor] -= 1
        q[e][receiver] += 1
        load[donor] -= 1
        load[receiver] += 1
    return q


def plan_forward(
    topk_ids: Sequence[Sequence[Sequence[int]]],
    config: PlanConfig,
    *,
    invocation_epoch: int,
    identity: bool = False,
    block_capacity: int | None = None,
) -> ForwardPlan:
    """Build a CPU oracle plan; capacity failure never returns partial tasks.

    Sources may have different valid token counts. ``-1`` is padding; other
    out-of-range IDs and repeated experts within a token are rejected.
    Destination blocks are ordered by expert, then execution rank. SF offsets
    are independently padded per slot, not multiplied by beta/128.
    """
    c = config
    epoch = _integer("invocation_epoch", invocation_epoch, 1)
    capacity = (
        c.block_capacity
        if block_capacity is None
        else _integer("block_capacity", block_capacity)
    )
    if len(topk_ids) != c.world_size:
        raise ValueError("one route array per source rank is required")
    routes: list[list[Route]] = [[] for _ in range(c.num_logical_experts)]
    counts = [[0] * c.num_logical_experts for _ in range(c.world_size)]
    for rank, source in enumerate(topk_ids):
        if len(source) > c.max_tokens_per_source:
            raise ValueError("source token count exceeds configured capacity")
        for token, row in enumerate(source):
            if len(row) != c.top_k:
                raise ValueError("route row width must equal top_k")
            seen = set()
            for k, value in enumerate(row):
                if isinstance(value, bool):
                    raise ValueError("expert IDs must be integers, not bools")
                e = _integer("expert ID + 1", index(value) + 1) - 1
                if e == -1:
                    continue
                if e >= c.num_logical_experts or e in seen:
                    raise ValueError("expert IDs must be in range and unique per token")
                seen.add(e)
                counts[rank][e] += 1
                routes[e].append(Route(rank, token, k))
    expert_counts = tuple(map(len, routes))
    blocks = [_ceil(n, c.beta) for n in expert_counts]
    offsets = [0]
    for n in blocks:
        offsets.append(offsets[-1] + n)
    owned = c.num_owned_experts_per_rank
    slots = [
        list(range(r * owned, (r + 1) * owned)) + [-1] * c.replica_budget
        for r in range(c.world_size)
    ]
    overflow = offsets[-1] > capacity
    q = (
        _assignment(blocks, c, identity)
        if not overflow
        else [[0] * c.world_size for _ in blocks]
    )
    replicas = []
    slot_index = {
        (e, r): e % owned
        for r in range(c.world_size)
        for e in range(r * owned, (r + 1) * owned)
    }
    for r in range(c.world_size):
        remote = [
            e for e in range(c.num_logical_experts) if e // owned != r and q[e][r]
        ]
        for s, e in enumerate(remote, start=owned):
            slots[r][s] = e
            slot_index[e, r] = s
            replicas.append(
                ReplicaDescriptor(
                    e, e // owned, e % owned, r, s, c.weight_version, c.layout_id
                )
            )
    descriptors = []
    data_end, sf_end, tasks, active = ([0] * c.world_size for _ in range(4))
    for e, allocation in enumerate(q):
        block = 0
        for r, n in enumerate(allocation):
            if not n:
                continue
            active[r] += 1
            begin = block * c.beta
            rows = min(n * c.beta, len(routes[e]) - begin)
            for j in range(n):
                first = (block + j) * c.beta
                chunk = tuple(routes[e][first : first + c.beta])
                descriptors.append(
                    BlockDescriptor(
                        e,
                        block + j,
                        r,
                        slot_index[e, r],
                        data_end[r] + j * c.beta,
                        len(chunk),
                        sf_end[r] + j * c.beta,
                        chunk,
                    )
                )
            data_end[r] += _align(rows, c.beta)
            sf_end[r] += _align(rows, c.sf_padding)
            tasks[r] += n
            block += n
    return ForwardPlan(
        c,
        epoch,
        PlanStatus.CAPACITY_EXCEEDED if overflow else PlanStatus.OK,
        overflow,
        expert_counts,
        tuple(map(tuple, counts)),
        tuple(offsets),
        tuple(map(tuple, slots)),
        tuple(descriptors),
        tuple(replicas),
        tuple(tasks),
        tuple(active),
        tuple(data_end),
        tuple(sf_end),
    )
