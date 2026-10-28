# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Admission must reject unprofitable copies without freezing graph decisions."""

import os

import pytest
import torch

from flashinfer.moe_ep.megaflux_plan import PlanConfig
from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.benefit import ReplicaAdmission
from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.planner import DevicePlan


@pytest.fixture(autouse=True)
def rank_device():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


@pytest.mark.arch_hopper
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("copy_ns", [100004, 2000000, 100000000])
def test_admission_dynamic_owner_and_hotspot(capture, copy_ns):
    config = PlanConfig(384, 8, 1, 1024, 6)
    policy = ReplicaAdmission(3000, 8000, 3300000, copy_ns, 300000)
    counts = torch.zeros((8, 384), dtype=torch.int64, device="cuda")
    plan = DevicePlan(config, device="cuda")
    owner = DevicePlan(config, device="cuda")

    def run():
        plan.build(counts, epoch=1, replica_admission=policy)

    run()
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    for pattern, live in [
        ("balanced", 512),
        ("hot", 128),
        ("hot", 512),
        ("hot", 1024),
        ("balanced", 512),
    ]:
        counts.zero_()
        if pattern == "balanced":
            counts.fill_(live // 64)
        else:
            counts[:, :6].fill_(live)
        graph.replay() if capture else run()
        owner.build(counts, epoch=1, identity=True)
        replicas = plan.meta[3].item()
        if pattern == "hot" and live >= 512 and copy_ns < 1000000:
            assert replicas > 0
            assert plan.statistics[0].max() < owner.statistics[0].max()
        else:
            assert replicas == 0
            for name in (
                "q",
                "expert_counts",
                "expert_block_offsets",
                "slot_to_expert",
                "statistics",
                "meta",
            ):
                assert getattr(plan, name).equal(getattr(owner, name)), name
            n = owner.meta[2].item()
            assert plan.blocks[:n].equal(owner.blocks[:n])


@pytest.mark.arch_hopper
def test_admission_equal_savings_rejects():
    config = PlanConfig(16, 4, 1, 128, 1)
    counts = torch.zeros((4, 16), dtype=torch.int64, device="cuda")
    counts[:, 0] = 128
    plan = DevicePlan(config, device="cuda")
    # Perfect balance saves 12 blocks, exactly the fixed cost: reject.
    plan.build(counts, epoch=1, replica_admission=ReplicaAdmission(1000, 1, 12, 0, 0))
    assert plan.meta[3].item() == 0


def test_admission_calibration_scope():
    from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.benefit import (
        h200_replica_admission,
    )

    shape = dict(experts=384, world_size=8, hidden=5120, intermediate=2304, top_k=6)
    assert h200_replica_admission(device_name="NVIDIA H200", **shape) is not None
    assert h200_replica_admission(device_name="NVIDIA H100", **shape) is None
    assert (
        h200_replica_admission(device_name="NVIDIA H200", enabled=False, **shape)
        is None
    )
    shape["hidden"] = 4096
    assert h200_replica_admission(device_name="NVIDIA H200", **shape) is None


@pytest.mark.parametrize(
    "values",
    [
        (999, 1, 0, 0, 0),
        (1000, 0, 0, 0, 0),
        (3000, 8000, -1, 1, 1),
        (3000, 8000, 1, float("nan"), 1),
    ],
)
def test_admission_invalid_calibration(values):
    with pytest.raises(ValueError):
        ReplicaAdmission(*values)


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
def test_admission_transport_graph(monkeypatch):
    if "RANK" not in os.environ:
        pytest.skip("launch with torchrun")
    from tests.moe_ep import test_megaflux_graph as graph_test

    original = graph_test.Sm90_Mxfp4_MegaFluxGraphConfig

    def with_admission(**kwargs):
        return original(
            **kwargs,
            replica_admission=ReplicaAdmission(1500, 8000, 0, 10, 0),
        )

    monkeypatch.setattr(graph_test, "Sm90_Mxfp4_MegaFluxGraphConfig", with_admission)
    monkeypatch.setenv("MEGAFLUX_SELECTIVE", "1")
    graph_test.test_graph_replay_plan_slot_epoch(monkeypatch, 1, False)
