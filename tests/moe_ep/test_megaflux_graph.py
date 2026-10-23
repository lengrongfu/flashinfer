# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Queued Graph replay with changing routes, physical identities and epochs."""

import os
import traceback

import pytest
import torch
import torch.distributed as dist

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpLayer,
    MoEEpTensors,
    bootstrap_moe_ep_runtime,
    finalize_moe_ep_runtime,
    ensure_moe_ep_cuda_device,
)
from flashinfer.moe_ep.megaflux_plan import PlanConfig, plan_forward, pack_plan_records
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_graph import (
    Sm90_Mxfp4_MegaFluxGraphConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl import (
    Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
)
from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "budget,force_fallback",
    [(0, False), (1, False), (2, False), (4, False), (1, True)],
    ids=["r0", "r1", "r2", "r4", "fallback"],
)
def test_graph_replay_plan_slot_epoch(monkeypatch, budget, force_fallback):
    monkeypatch.setattr(ref, "TOKENS_PER_RANK", 129)
    from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90 import physical

    allocate = physical.allocate_physical_weight_bank
    guards = []

    def guarded_bank(transformed, *, owned, capacity):
        expanded = allocate(transformed, owned=owned, capacity=capacity + 1)
        result = []
        for payload, offset, placeholder, residual in expanded:
            for pattern, guard in (
                (
                    0xA5,
                    payload[capacity:].transpose(1, 2).reshape(-1).view(torch.uint8),
                ),
                (0x5A, offset[capacity:].reshape(-1).view(torch.uint8)),
                (0xC3, residual[capacity:].view(torch.uint8)),
            ):
                guard.fill_(pattern)
                guards.append((guard, guard.clone()))
            result.append(
                (
                    payload[:capacity],
                    offset[:capacity],
                    placeholder,
                    residual[:capacity],
                )
            )
        return tuple(result)

    monkeypatch.setattr(physical, "allocate_physical_weight_bank", guarded_bank)
    rank, world, local = ref._launcher_ranks()
    if world < 2:
        pytest.skip("launch through torchrun with 2/8 ranks")
    bootstrap = BootstrapConfig(world_size=world, rank=rank, device=local)
    ensure_moe_ep_cuda_device(bootstrap)
    initial = create_mega_kernel(
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=ref.INTERMEDIATE, top_k=world
        )
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, initial.runtime_requirements(bootstrap)
    )
    layer = baseline = graph = None
    try:
        raw = ref._make_raw_weights(rank)
        global_raw = ref._gather_raw_weights(raw)
        c = PlanConfig(
            world * ref.LOCAL_EXPERTS, world, budget, ref.TOKENS_PER_RANK, world
        )
        # Different stream delays per PE expose reuse of an export while a slow
        # receiver is still reading it. Collective sequences remain identical.
        delays = (
            2000 if rank % 3 == 0 else 0,
            5000 if rank == world - 1 else 0,
            1000 if rank == 1 else 0,
            1000 if rank == 0 else 0,
        )
        config = Sm90_Mxfp4_MegaFluxGraphConfig(
            intermediate_size=ref.INTERMEDIATE,
            top_k=world,
            plan_config=c,
            load_balance_mode="static",
            gate_up_clamp=ref.GATE_UP_CLAMP,
            debug_plan_agreement=True,
            requested_block_capacity=0 if force_fallback and rank == 0 else None,
            pipeline_delays_us=delays,
        )
        fleet = FleetParams(
            num_experts=c.num_logical_experts,
            max_tokens_per_rank=c.max_tokens_per_source,
            token_hidden_size=ref.HIDDEN,
        )
        args = dict(
            bootstrap=BootstrapConfig(
                world_size=world, rank=rank, device=local, auto_bootstrap=False
            ),
            fleet_params=fleet,
            weights=raw,
        )
        layer = MoEEpLayer(
            **args,
            backend=MegaConfig(
                megakernel=config, quantize_input=True, preprocess_weights=True
            ),
        )
        original_config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=ref.INTERMEDIATE,
            top_k=world,
            gate_up_clamp=ref.GATE_UP_CLAMP,
            load_balance_mode="static",
            swap_ab=True,
            pingpong=False,
            mma_tiler_mnk=(128, 32, 128),
            cluster_shape_mnk=(1, 1, 1),
            dedup_dispatch=False,
            fold_producer_warps=False,
            token_back_mode="epi_warps",
        )
        baseline = MoEEpLayer(
            **args,
            backend=MegaConfig(
                megakernel=original_config, quantize_input=True, preprocess_weights=True
            ),
        )
        fixed = _tokens(rank, world, 2)
        capture_stream = torch.cuda.Stream(device=local)
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            ws = layer._ensure_workspace()
            cold = torch.cuda.CUDAGraph()
            with (
                pytest.raises(RuntimeError, match="warm all MegaFlux"),
                torch.cuda.graph(cold, stream=capture_stream),
            ):
                layer._kernel.validate_capture_ready(ws, layer._transformed)
            cold = None
            for _ in range(3):
                layer.forward(fixed)
                baseline.forward(fixed)
            torch.cuda.synchronize()
            state = ws._megaflux_pipeline
            start_epoch = state.epoch.item()
            assert start_epoch == 3
            addresses = _addresses(ws)
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                captured = layer.forward(fixed)
            torch.cuda.synchronize()
            assert state.epoch.item() == start_epoch
            replays = int(os.environ.get("MEGAFLUX_GRAPH_REPLAYS", "128"))
            assert replays > 0
            checked = 0
            while checked < replays:
                pending = []
                for step in range(checked, min(checked + 8, replays)):
                    source = _tokens(rank, world, step)
                    fixed.hidden_states.copy_(source.hidden_states)
                    fixed.topk_ids.copy_(source.topk_ids)
                    fixed.topk_weights.copy_(source.topk_weights)
                    graph.replay()
                    # Queue the entire burst without a host read or a CPU Plan.
                    pending.append((source, captured.clone(), _snapshot(ws)))
                torch.cuda.synchronize()
                assert _addresses(ws) == addresses
                for guard, expected_guard in guards:
                    torch.testing.assert_close(guard, expected_guard, atol=0, rtol=0)
                for offset, (source, actual, snapshot) in enumerate(pending):
                    step = checked + offset
                    torch.testing.assert_close(
                        actual, baseline.forward(source), atol=0, rtol=0
                    )
                    global_ids = ref._all_gather_stack(source.topk_ids)
                    expected = plan_forward(
                        global_ids.cpu().tolist(),
                        c,
                        invocation_epoch=start_epoch + step + 1,
                        identity=force_fallback,
                    )
                    _assert_snapshot(snapshot, expected, rank, force_fallback)
                    if not any(expected.expert_counts):
                        assert actual.eq(0).all().item()
                    if step % 16 in (2, 7) or step == replays - 1:
                        # The instruction-exact test-owned WGMMA/SiLU helpers
                        # launch on CUDA's default stream. Validate there with
                        # explicit producer/consumer dependencies; this is test
                        # inspection outside the captured production DAG.
                        default = torch.cuda.default_stream()
                        default.wait_stream(capture_stream)
                        with torch.cuda.stream(default):
                            expected_output = ref._global_route_reference(
                                ref._all_gather_stack(source.hidden_states),
                                global_ids,
                                ref._all_gather_stack(source.topk_weights),
                                global_raw,
                                require_all_experts=False,
                            )
                            ref._assert_matches_reference(
                                ref._all_gather_stack(actual),
                                expected_output,
                                launch=step,
                            )
                        capture_stream.wait_stream(default)
                checked += len(pending)
            assert state.epoch.item() == start_epoch + replays
            _assert_final_replica_bytes(layer, ws, rank)
            if rank == 0:
                print(
                    "MegaFlux Graph accepted",
                    {
                        "world": world,
                        "R": budget,
                        "replays": replays,
                        "fallback": force_fallback,
                        "final_epoch": state.epoch.item(),
                        "stable_addresses": True,
                    },
                    flush=True,
                )
    except Exception:
        traceback.print_exc()
        os._exit(1)
    finally:
        graph = None
        torch.cuda.synchronize()
        if baseline is not None:
            baseline.destroy()
        if layer is not None:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)


def _tokens(rank, world, step):
    hidden, ids, weights = ref._make_tokens_and_routes(
        rank, world, launch=step, routing_pattern="sparse_owner"
    )
    valid = (step * 17 + rank * 13) % (ref.TOKENS_PER_RANK + 1)
    mode = step % 8
    if mode == 0:
        valid = 0
    elif mode == 1:
        valid = 0 if rank == 0 else ref.TOKENS_PER_RANK
    elif mode == 2:
        valid = ref.TOKENS_PER_RANK
    elif mode == 3:
        valid = 32
    elif mode == 4:
        valid = 33
    ids = (
        ids // ref.LOCAL_EXPERTS * ref.LOCAL_EXPERTS
        + (ids % ref.LOCAL_EXPERTS + step // world) % ref.LOCAL_EXPERTS
    )
    ids[valid:] = -1
    return MoEEpTensors(hidden_states=hidden, topk_ids=ids, topk_weights=weights)


def _snapshot(ws):
    plan = ws._megaflux_planner.committed
    fields = {
        name: getattr(plan, name).clone()
        for name in (
            "source_counts",
            "expert_counts",
            "expert_block_offsets",
            "slot_to_expert",
            "q",
            "blocks",
            "replicas",
            "statistics",
            "meta",
        )
    }
    state = ws._megaflux_pipeline
    fields.update(
        {
            "state_" + name: getattr(state, name).clone()
            for name in ("ready", "epoch", "chunks", "identity", "retired")
        }
    )
    fields["fallback"] = ws._megaflux_planner.fallback.clone()
    return fields


def _assert_snapshot(fields, plan, rank, force_fallback):
    packed = pack_plan_records(plan)
    epoch = plan.invocation_epoch
    assert fields["meta"].cpu().tolist() == [
        0,
        0,
        len(packed.blocks),
        len(packed.replicas),
        sum(plan.expert_counts),
        epoch,
    ]
    assert fields["source_counts"].cpu().tolist() == list(
        map(list, plan.source_expert_counts)
    )
    assert fields["expert_counts"].cpu().tolist() == list(plan.expert_counts)
    assert fields["expert_block_offsets"].cpu().tolist() == list(
        plan.expert_block_offsets
    )
    assert fields["slot_to_expert"].cpu().tolist() == list(
        map(list, plan.slot_to_expert)
    )
    assert fields["blocks"][: len(packed.blocks)].cpu().tolist() == list(
        map(list, packed.blocks)
    )
    assert fields["replicas"][: len(packed.replicas)].cpu().tolist() == list(
        map(list, packed.replicas)
    )
    assert fields["statistics"].cpu().tolist() == list(
        map(
            list,
            (
                plan.rank_task_counts,
                plan.num_active_physical_slots,
                plan.rank_padded_rows,
                plan.rank_sf_rows,
            ),
        )
    )
    assert fields["state_epoch"].item() == epoch
    assert fields["state_ready"].eq(epoch).all().item()
    assert fields["state_retired"].eq(epoch - 1).all().item()
    identities = fields["state_identity"].cpu()
    assert identities[:, :, 0].tolist() == [list(plan.slot_to_expert[rank])] * 2
    assert (
        identities[:, :, 4].eq(epoch).all() and identities[:, :, 5].eq(epoch - 1).all()
    )
    if plan.config.replica_budget:
        assert fields["state_chunks"].eq(epoch).all().item()
    assert fields["fallback"].item() == int(
        force_fallback and bool(plan.block_descriptors)
    )


def _addresses(ws):
    tensors = [ws._frontend._mega.local_workspace, ws._frontend._mega.shared_workspace]
    for leg in ws._megaflux_weight_bank:
        tensors.extend((leg[0], leg[1], leg[3]))
    tensors.extend(ws._megaflux_transport.roots)
    p = ws._megaflux_planner
    for plan in (p.candidate, p.identity, p.committed):
        tensors.extend(
            getattr(plan, name)
            for name in (
                "source_counts",
                "q",
                "expert_counts",
                "expert_block_offsets",
                "slot_to_expert",
                "blocks",
                "replicas",
                "statistics",
                "meta",
                "scratch",
            )
        )
    tensors.extend(
        getattr(p.routing, name)
        for name in (
            "send_counts",
            "send_words",
            "local_records",
            "counts",
            "received_words",
            "words",
        )
    )
    tensors.extend(
        (
            p.histogram.ordinal,
            p.histogram.counts,
            p.histogram.status,
            p.invalid,
            p.fallback,
            p.hash,
            p.peer_hashes,
        )
    )
    tensors.extend(p.hash_weights.values())
    state = ws._megaflux_pipeline
    tensors.extend(
        getattr(state, name)
        for name in ("ready", "epoch", "chunks", "identity", "retired", "telemetry")
    )
    return tuple(t.data_ptr() for t in tensors)


def _assert_final_replica_bytes(layer, ws, rank):
    from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.static import (
        StaticReplicaTransport,
    )

    slots = ws._megaflux_planner.committed.slot_to_expert[rank].cpu().tolist()
    own = ref.LOCAL_EXPERTS
    for source, bank in zip(layer._transformed, ws._megaflux_weight_bank, strict=True):
        original = torch.stack(
            [
                torch.cat(StaticReplicaTransport._pieces(source, slot))
                for slot in range(own)
            ]
        )
        gathered = ref._all_gather_stack(original)
        for slot, expert in enumerate(slots[own:], start=own):
            if expert >= 0:
                torch.testing.assert_close(
                    torch.cat(StaticReplicaTransport._pieces(bank, slot)),
                    gathered[expert // own, expert % own],
                    atol=0,
                    rtol=0,
                )
