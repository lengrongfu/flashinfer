# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Device-only benefit selection, including repeated captured decisions."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.benefit import (
    BenefitCosts,
    BenefitGate,
)


@pytest.mark.arch_hopper
@pytest.mark.parametrize("capture", [False, True])
def test_benefit_gate_changes_without_host_decision(capture):
    candidate = SimpleNamespace(
        statistics=torch.zeros((4, 8), dtype=torch.int64, device="cuda")
    )
    identity = SimpleNamespace(statistics=torch.zeros_like(candidate.statistics))
    gate = BenefitGate(BenefitCosts(10, 3, 5, 2), device="cuda")
    identity.statistics[0].fill_(10)
    candidate.statistics[0].fill_(5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gate.run(candidate, identity)
        if capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                gate.run(candidate, identity)
    stream.synchronize()
    addresses = (gate.reject.data_ptr(), gate.metrics.data_ptr())
    for peak, reject in ((5, 0), (9, 1), (10, 1), (0, 0), (5, 0)):
        candidate.statistics[0].fill_(peak)
        if capture:
            graph.replay()
        else:
            gate.run(candidate, identity)
        assert gate.reject.item() == reject
        assert gate.metrics.cpu().tolist() == [100, peak * 10 + 8, 8, reject]
        assert addresses == (gate.reject.data_ptr(), gate.metrics.data_ptr())


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
def test_collective_benefit_reject_commits_identity():
    import os
    import torch.distributed as dist

    from flashinfer.moe_ep.megaflux_plan import PlanConfig
    from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.planner import (
        DistributedPlanner,
    )

    if "RANK" not in os.environ:
        pytest.skip("launch with torchrun")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("nccl")
    try:
        config = PlanConfig(world * 2, world, 1, 129, 1)
        # Only rank zero rejects: the collective must commit identity everywhere.
        costs = BenefitCosts(10, 1e9 if rank == 0 else 0, 0)
        planner = DistributedPlanner(config, device="cuda", benefit_costs=costs)
        ids = torch.zeros((129, 1), dtype=torch.int64, device="cuda")
        plan = planner.run(ids, rank=rank, epoch=7)
        assert planner.fallback.item() == 1
        assert plan.q.equal(planner.identity.q)
        assert plan.slot_to_expert.equal(planner.identity.slot_to_expert)
        assert plan.blocks.equal(planner.identity.blocks)
        assert plan.meta.equal(planner.identity.meta)
        assert planner.benefit_gate.metrics[2].item() == costs.fixed_us
        dist.barrier()
    finally:
        if created:
            dist.destroy_process_group()
