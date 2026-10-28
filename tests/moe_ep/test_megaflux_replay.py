# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Serving replay must update routes and clear padding as local batches shrink."""

import pytest
import torch


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
@pytest.mark.parametrize("decode_policy", [0, 1, 33])
@pytest.mark.parametrize(
    "metadata_transport,force_fallback",
    [("nccl", False), ("nvlink", False), ("nvlink", True)],
)
def test_replay_changing_local_batch_sizes(
    monkeypatch, decode_policy, metadata_transport, force_fallback
):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_graph import (
        Sm90_Mxfp4_MegaFluxGraphConfig,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.replay import (
        MegaFluxGraphReplay,
    )
    from flashinfer.moe_ep.megaflux_plan import PlanConfig
    from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref

    monkeypatch.setattr(ref, "TOKENS_PER_RANK", 33)
    rank, world, local = ref._launcher_ranks()
    if world < 2:
        pytest.skip("launch through torchrun with 2 or 8 ranks")
    bootstrap = BootstrapConfig(world_size=world, rank=rank, device=local)
    ensure_moe_ep_cuda_device(bootstrap)
    common = dict(
        intermediate_size=ref.INTERMEDIATE,
        top_k=world,
        gate_up_clamp=ref.GATE_UP_CLAMP,
        load_balance_mode="static",
    )
    original_config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(**common)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(original_config).runtime_requirements(bootstrap)
    )
    baseline = layer = replay = None
    try:
        raw = ref._make_raw_weights(rank)
        args = dict(
            bootstrap=BootstrapConfig(
                world_size=world, rank=rank, device=local, auto_bootstrap=False
            ),
            fleet_params=FleetParams(
                num_experts=world * ref.LOCAL_EXPERTS,
                max_tokens_per_rank=ref.TOKENS_PER_RANK,
                token_hidden_size=ref.HIDDEN,
            ),
            weights=raw,
        )
        baseline = MoEEpLayer(
            **args,
            backend=MegaConfig(megakernel=original_config, quantize_input=True),
        )
        layer = MoEEpLayer(
            **args,
            backend=MegaConfig(
                megakernel=Sm90_Mxfp4_MegaFluxGraphConfig(
                    **common,
                    selective_weight_copy=True,
                    metadata_transport=metadata_transport,
                    debug_plan_agreement=metadata_transport == "nvlink",
                    requested_block_capacity=0
                    if force_fallback and rank == 0
                    else None,
                    min_replica_routes=(world * world * decode_policy + 1)
                    if decode_policy
                    else 0,
                    plan_config=PlanConfig(
                        world * ref.LOCAL_EXPERTS,
                        world,
                        1,
                        ref.TOKENS_PER_RANK,
                        world,
                    ),
                ),
                quantize_input=True,
            ),
        )
        # Allocate/compile both collective workspaces before queuing a replay.
        # Lazy NVSHMEM allocation while another rank is executing the captured
        # collectives can synchronize the device in the middle of a round.
        baseline.warmup()
        replay = MegaFluxGraphReplay(
            layer,
            max_tokens=ref.TOKENS_PER_RANK,
            hidden_size=ref.HIDDEN,
            top_k=world,
            device=torch.device("cuda", local),
        )
        graph = replay._graph
        sizes = (33, 1, 0, 17, 32, 2)
        for step in range(12):
            hidden, ids, weights = ref._make_tokens_and_routes(
                rank, world, launch=step, routing_pattern="sparse_owner"
            )
            n = sizes[(step + rank) % len(sizes)]
            if step < 3:
                n = (33, 1, 0)[step]
            hidden, ids, weights = hidden[:n], ids[:n], weights[:n]
            wire_ids = ids.to(torch.int32) if step % 2 else ids
            actual = replay(hidden, wire_ids, weights).clone()
            expected = baseline(
                MoEEpTensors(hidden_states=hidden, topk_ids=ids, topk_weights=weights)
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert replay._graph is graph
            assert replay._inputs.topk_ids[n:].eq(-1).all().item()
        replay.close()
        with pytest.raises(RuntimeError, match="closed"):
            replay(hidden, ids, weights)
    finally:
        if replay is not None:
            replay.close()
        if baseline is not None:
            baseline.destroy()
        if layer is not None:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)
