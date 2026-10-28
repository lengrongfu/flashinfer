# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Peer metadata must preserve rank order and retire buffers across Graph replays."""

import pytest
import torch


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
@pytest.mark.parametrize("check_hash", [False, True])
def test_metadata_queued_graph_replays(check_hash):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.metadata import NvlinkMetadata
    from flashinfer.moe_ep.megaflux_plan import PlanConfig
    from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref

    rank, world, local = ref._launcher_ranks()
    if world < 2:
        pytest.skip("launch through torchrun with 2 or 8 ranks")
    bootstrap = BootstrapConfig(world_size=world, rank=rank, device=local)
    ensure_moe_ep_cuda_device(bootstrap)
    config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=ref.INTERMEDIATE, top_k=world, load_balance_mode="static"
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(config).runtime_requirements(bootstrap)
    )
    transport = graph = None
    try:
        c = PlanConfig(world * 2, world, 1, 33, world)
        E, S, A = c.num_logical_experts, c.physical_slot_capacity, 33 * world
        transport = NvlinkMetadata(c, device=torch.device("cuda", local))
        step = torch.zeros(1, dtype=torch.int64, device="cuda")
        indices = torch.arange(world * S * A, dtype=torch.int32, device="cuda").view(
            world, S, A
        )
        experts = torch.arange(E, dtype=torch.int64, device="cuda")
        counts = torch.empty((world, E), dtype=torch.int64, device="cuda")
        invalid = torch.empty(1, dtype=torch.int32, device="cuda")
        fallback = torch.empty_like(invalid)
        hashes = torch.empty(world * 2, dtype=torch.int64, device="cuda")
        route_counts = torch.empty((world, S), dtype=torch.int64, device="cuda")
        words = torch.empty((S, world, A), dtype=torch.int32, device="cuda")

        def exchange():
            transport.begin()
            transport.counts.copy_(experts + step * 100 + rank)
            transport.status.copy_((step % 2) * (rank == world - 1))
            transport.hash[0:1].copy_(step)
            transport.hash[1:2].copy_(step + (step % 3 == 0) * rank)
            transport.statistics(counts, invalid)
            fallback.copy_((step % 5 == 0) * (rank == 0))
            transport.vote(fallback, hashes, check_hash=check_hash)
            transport.send_counts.fill_(rank)
            transport.send_words.copy_(indices + rank * 100000 + step * 1000000)
            transport.routes(route_counts, words)

        exchange()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            exchange()
        snapshots = []
        # Queue changing inputs and copies without a per-iteration host/device
        # rendezvous; faster peers must not clobber a slower peer's read arena.
        for iteration in range(64):
            step.fill_(iteration)
            if rank == iteration % world:
                torch.cuda._sleep(10000)
            graph.replay()
            snapshots.append(
                tuple(
                    x.clone()
                    for x in (counts, invalid, fallback, hashes, route_counts, words)
                )
            )
        torch.cuda.synchronize()
        for iteration, actual in enumerate(snapshots):
            actual = [x.cpu() for x in actual]
            expected_counts = torch.stack(
                [torch.arange(E) + iteration * 100 + r for r in range(world)]
            )
            torch.testing.assert_close(actual[0], expected_counts, atol=0, rtol=0)
            assert actual[1].item() == iteration % 2
            assert actual[2].item() == int(
                iteration % 5 == 0 or (check_hash and iteration % 3 == 0)
            )
            expected_hashes = torch.tensor(
                [
                    [iteration, iteration + (iteration % 3 == 0) * r]
                    for r in range(world)
                ]
            ).flatten()
            torch.testing.assert_close(actual[3], expected_hashes, atol=0, rtol=0)
            torch.testing.assert_close(
                actual[4], torch.arange(world)[:, None].expand(world, S), atol=0, rtol=0
            )
            expected_words = torch.stack(
                [
                    torch.arange(world * S * A, dtype=torch.int32).view(world, S, A)[
                        rank
                    ]
                    + r * 100000
                    + iteration * 1000000
                    for r in range(world)
                ],
                dim=1,
            )
            torch.testing.assert_close(actual[5], expected_words, atol=0, rtol=0)
    finally:
        if graph is not None:
            torch.cuda.synchronize()
            graph.reset()
        if transport is not None:
            transport.close()
        finalize_moe_ep_runtime(runtime)
