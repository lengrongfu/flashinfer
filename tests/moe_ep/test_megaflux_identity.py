# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GPU checks for the offline R=0 identity receiver adapter."""

import pytest


@pytest.mark.arch_hopper
def test_identity_config_rejects_replica_budget_and_tactic_changes():
    from dataclasses import replace
    from flashinfer.moe_ep.megaflux_plan import PlanConfig, plan_forward
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_identity import (
        Sm90_Mxfp4_MegaFluxIdentityConfig,
    )

    p = PlanConfig(4, 2, 0, 32, 2)
    ids = [[[0, 1]], [[2, 3]]]
    plan = plan_forward(ids, p, invocation_epoch=1)
    config = Sm90_Mxfp4_MegaFluxIdentityConfig(
        intermediate_size=128, top_k=2, plan=plan
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    assert create_mega_kernel(config)._workspace_pool_request(None) is None
    for changes in (
        {"dedup_dispatch": True},
        {"mma_tiler_mnk": (128, 64, 128)},
        {"pingpong": True},
        {"token_back_mode": "standalone_warps"},
        {"knobs": "auto"},
        {"kernel_name": "sm90_fp8_mxfp4_bf16_pull_cutedsl"},
    ):
        with pytest.raises(ValueError):
            replace(config, **changes)
    with pytest.raises(ValueError, match="R=0"):
        replace(
            config,
            plan=plan_forward(ids, replace(p, replica_budget=1), invocation_epoch=1),
        )


@pytest.mark.arch_hopper
@pytest.mark.parametrize("active_dispatch_warps", [1, 2, 4])
def test_identity_matches_independent_oracle(monkeypatch, active_dispatch_warps):
    from tests.moe_ep import test_sm90_pull_mxfp4_kernel_vs_reference as original

    original._require_hopper()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    from flashinfer.moe_ep.megaflux_plan import PlanConfig, plan_forward
    from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90 import bind_identity_plan

    pkg = original._sm90_tree()
    problem = original._single_rank_problem()
    ids = problem["topk_ids"].cpu().tolist()
    ids += [[-1] * problem["topk"]] * (problem["max_tokens"] - len(ids))
    plan = plan_forward(
        [ids],
        PlanConfig(problem["experts"], 1, 0, problem["max_tokens"], problem["topk"]),
        invocation_epoch=1,
    )

    class PlanPackage:
        def get_symm_buffer_for_hopper_mxfp4_mega_moe(self, *args, **kwargs):
            workspace = pkg.get_symm_buffer_for_hopper_mxfp4_mega_moe(*args, **kwargs)
            bind_identity_plan(workspace, plan)
            return workspace

        def __getattr__(self, name):
            return getattr(pkg, name)

    monkeypatch.setattr(original, "_sm90_tree", lambda: PlanPackage())
    original.test_sm90_mxfp4_kernel_matches_independent_reference(
        monkeypatch, active_dispatch_warps
    )


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "routing_pattern", ["cross_rank", "zero_source", "all_masked", "sparse_owner"]
)
def test_identity_backend_multirank_and_graph(routing_pattern):
    _run_offline_backend(routing_pattern)


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
@pytest.mark.parametrize("budget", [1, 2, 4])
@pytest.mark.parametrize("routing_pattern", ["cross_rank", "zero_source", "all_masked"])
def test_expanded_physical_slots_identity(routing_pattern, budget):
    _run_offline_backend(routing_pattern, physical_budget=budget)


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
@pytest.mark.parametrize("routing_pattern", ["cross_rank", "zero_source", "all_masked"])
def test_static_replica_full_bytes_and_slot_reuse(monkeypatch, routing_pattern):
    from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref

    monkeypatch.setattr(ref, "TOKENS_PER_RANK", 129)
    _run_offline_backend(routing_pattern, physical_budget=1, static_replica=True)


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
@pytest.mark.parametrize("budget", [0, 1, 2, 4])
@pytest.mark.parametrize(
    "routing_pattern", ["sparse_owner", "zero_source", "all_masked"]
)
def test_online_gpu_plan_and_changing_routes(monkeypatch, routing_pattern, budget):
    from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref

    monkeypatch.setattr(ref, "TOKENS_PER_RANK", 129)
    _run_offline_backend(routing_pattern, physical_budget=budget, online=True)


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
def test_online_collective_capacity_fallback(monkeypatch):
    from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref

    monkeypatch.setattr(ref, "TOKENS_PER_RANK", 129)
    _run_offline_backend(
        "sparse_owner", physical_budget=1, online=True, requested_capacity=0
    )


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
def test_online_collective_version_fallback(monkeypatch):
    from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref

    monkeypatch.setattr(ref, "TOKENS_PER_RANK", 129)
    _run_offline_backend(
        "sparse_owner", physical_budget=1, online=True, version_mismatch=True
    )


def _run_offline_backend(
    routing_pattern,
    physical_budget=None,
    static_replica=False,
    online=False,
    requested_capacity=None,
    version_mismatch=False,
    pipeline=False,
    delays=(0, 0, 0, 0),
    resource_audit=False,
    top_k=None,
):
    import torch
    import torch.distributed as dist
    from tests.moe_ep import test_moe_ep_sm90_pull_mxfp4_mega_multirank as ref
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
    from flashinfer.moe_ep.megaflux_plan import PlanConfig, plan_forward
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_identity import (
        Sm90_Mxfp4_MegaFluxIdentityConfig,
        Sm90MegaFluxIdentityBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl import (
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    rank, world, local = ref._launcher_ranks()
    selected_topk = world if top_k is None else top_k
    if world < 2:
        pytest.skip("launch with torchrun --nproc_per_node=2/4/8")
    bootstrap = BootstrapConfig(world_size=world, rank=rank, device=local)
    ensure_moe_ep_cuda_device(bootstrap)
    # Bootstrap first, then obtain the global offline Plan outside forward.
    initial = create_mega_kernel(
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=ref.INTERMEDIATE,
            top_k=selected_topk,
        )
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, initial.runtime_requirements(bootstrap)
    )
    layer = None
    baseline = None
    synchronous = None
    graph = None
    try:
        raw = ref._make_raw_weights(rank)
        global_raw = ref._gather_raw_weights(raw)
        hidden, ids, weights = ref._make_tokens_and_routes(
            rank, world, launch=0, routing_pattern=routing_pattern
        )
        selected_topk = world if top_k is None else top_k
        ids = ids[:, :selected_topk].contiguous()
        weights = weights[:, :selected_topk].contiguous()
        global_hidden = ref._all_gather_stack(hidden)
        global_ids = ref._all_gather_stack(ids)
        global_weights = ref._all_gather_stack(weights)
        static_spec = None
        if static_replica:
            expert = ref.LOCAL_EXPERTS if routing_pattern == "zero_source" else 0
            receiver = 0 if routing_pattern == "zero_source" else world - 1
            moved = {"all_masked": 0, "zero_source": 1, "cross_rank": 2}[
                routing_pattern
            ]
            static_spec = (expert, receiver, moved)
        plan = plan_forward(
            global_ids.cpu().tolist(),
            PlanConfig(
                world * ref.LOCAL_EXPERTS,
                world,
                physical_budget or 0,
                ref.TOKENS_PER_RANK,
                selected_topk,
                weight_version=int(version_mismatch and rank == 0),
            ),
            invocation_epoch=1,
            identity=(not static_replica and not online)
            or requested_capacity == 0
            or version_mismatch,
            static_replica=static_spec,
        )
        config_type = Sm90_Mxfp4_MegaFluxIdentityConfig
        if physical_budget is not None:
            from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_physical import (
                Sm90_Mxfp4_MegaFluxPhysicalConfig,
            )

            config_type = Sm90_Mxfp4_MegaFluxPhysicalConfig
        if static_replica:
            from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_static import (
                Sm90_Mxfp4_MegaFluxStaticConfig,
            )

            config_type = Sm90_Mxfp4_MegaFluxStaticConfig
        if online:
            from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_online import (
                Sm90_Mxfp4_MegaFluxOnlineConfig,
            )

            config_type = Sm90_Mxfp4_MegaFluxOnlineConfig
        if pipeline:
            from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_pipeline import (
                Sm90_Mxfp4_MegaFluxPipelineConfig,
            )

            config_type = Sm90_Mxfp4_MegaFluxPipelineConfig
        plan_args = (
            {
                "plan_config": plan.config,
                "requested_block_capacity": requested_capacity if rank == 0 else None,
                "debug_plan_agreement": True,
            }
            if online
            else {"plan": plan}
        )
        config = config_type(
            intermediate_size=ref.INTERMEDIATE,
            top_k=selected_topk,
            **plan_args,
            gate_up_clamp=ref.GATE_UP_CLAMP,
            load_balance_mode="static",
        )
        assert isinstance(create_mega_kernel(config), Sm90MegaFluxIdentityBackend)
        fleet = FleetParams(
            num_experts=world * ref.LOCAL_EXPERTS,
            max_tokens_per_rank=ref.TOKENS_PER_RANK,
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
            top_k=selected_topk,
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
        if pipeline:
            from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_online import (
                Sm90_Mxfp4_MegaFluxOnlineConfig,
            )

            synchronous = MoEEpLayer(
                **args,
                backend=MegaConfig(
                    megakernel=Sm90_Mxfp4_MegaFluxOnlineConfig(
                        intermediate_size=ref.INTERMEDIATE,
                        top_k=selected_topk,
                        plan_config=plan.config,
                        gate_up_clamp=ref.GATE_UP_CLAMP,
                        load_balance_mode="static",
                    ),
                    quantize_input=True,
                    preprocess_weights=True,
                ),
            )
        inputs = MoEEpTensors(hidden_states=hidden, topk_ids=ids, topk_weights=weights)
        actual = layer.forward(inputs).clone()
        from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.identity import (
            IdentityFrontend,
            IdentityTokenComm,
        )

        ws = layer._ensure_workspace()
        if pipeline:
            torch.cuda.synchronize()
        assert isinstance(ws._frontend, IdentityFrontend)
        assert isinstance(ws._frontend._mega.kernel.token_comm, IdentityTokenComm)
        assert ws is not baseline._ensure_workspace()
        if physical_budget is not None:
            kernel = ws._frontend._mega.kernel
            cap = ref.LOCAL_EXPERTS + physical_budget
            assert kernel.num_logical_experts == world * ref.LOCAL_EXPERTS
            assert kernel.num_total_experts == world * ref.LOCAL_EXPERTS
            assert kernel.num_owned_experts_per_rank == ref.LOCAL_EXPERTS
            assert kernel.physical_slot_capacity == cap
            assert kernel.num_experts_per_rank == cap
            assert kernel.token_comm.num_experts_per_rank == cap
            assert (
                kernel.token_comm.num_total_experts == plan.config.num_logical_experts
            )
            assert ws._frontend._identity_counts.shape == (world, cap)
            for leg, source in zip(
                ws._megaflux_weight_bank, layer._transformed, strict=True
            ):
                assert leg[0].shape[0] == cap and leg[0].stride(1) == 1
                for bank_tensor, original_tensor in zip(
                    (leg[0], leg[1], leg[3]),
                    (source[0], source[1], source[3]),
                    strict=True,
                ):
                    torch.testing.assert_close(
                        bank_tensor[: ref.LOCAL_EXPERTS],
                        original_tensor,
                        atol=0,
                        rtol=0,
                    )
        if pipeline:
            sync_output = synchronous.forward(inputs)
            torch.testing.assert_close(actual, sync_output, atol=0, rtol=0)
            torch.testing.assert_close(
                ws._megaflux_planner.committed.q,
                synchronous._ensure_workspace()._megaflux_planner.committed.q,
                atol=0,
                rtol=0,
            )
            from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.pipeline import (
                PipelineFrontend,
            )

            assert isinstance(ws._frontend, PipelineFrontend)
            from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.pipeline_epilogue import (
                PipelineEpilogue,
            )

            assert isinstance(ws._frontend._mega.kernel.epilogue, PipelineEpilogue)
            hardware = torch.cuda.get_device_properties(local).multi_processor_count
            assert ws._frontend.persistent_clusters == hardware - 2
            if rank == 0:
                compiled = ws._frontend._mega.compiled
                kernel = ws._frontend._mega.kernel
                native_resources = []
                if resource_audit:
                    from cuda.bindings import driver as cuda

                    library = cuda.CUlibrary(int(compiled.library))
                    for symbol in compiled.kernel_info:
                        status, handle = cuda.cuLibraryGetKernel(
                            library, symbol.encode()
                        )
                        assert status == cuda.CUresult.CUDA_SUCCESS
                        values = {}
                        for label, attr in (
                            (
                                "registers",
                                cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
                            ),
                            (
                                "static_smem",
                                cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                            ),
                            (
                                "max_dynamic_smem",
                                cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            ),
                            (
                                "local_bytes",
                                cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                            ),
                        ):
                            status, value = cuda.cuKernelGetAttribute(
                                attr, handle, cuda.CUdevice(local)
                            )
                            assert status == cuda.CUresult.CUDA_SUCCESS
                            values[label] = value
                        native_resources.append((symbol, values))
                print(
                    "MegaFlux pipeline resources",
                    {
                        "threads": kernel.threads_per_cta,
                        "epi_register_policy": kernel.epi_reg_cnt,
                        "persistent_clusters": ws._frontend.persistent_clusters,
                        "reserved_clusters": 2,
                        "compiled_type": str(type(compiled)),
                        "native": native_resources,
                        "kernel_info": str(getattr(compiled, "kernel_info", None)),
                    },
                    flush=True,
                )
        original_output = baseline.forward(inputs)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, original_output, atol=0, rtol=0)
        expected = ref._global_route_reference(
            global_hidden,
            global_ids,
            global_weights,
            global_raw,
            require_all_experts=routing_pattern == "cross_rank",
        )
        ref._assert_matches_reference(ref._all_gather_stack(actual), expected, launch=0)
        if pipeline:
            config.pipeline_delays_us = delays
        # Repeat on the same workspace; metadata counters were tail-reset.
        torch.testing.assert_close(layer.forward(inputs), actual, atol=0, rtol=0)
        if pipeline:
            torch.cuda.synchronize()
            state = ws._megaflux_pipeline
            times = state.telemetry.cpu().tolist()
            identities = state.identity.cpu()
            assert identities[:, :, 0].tolist() == [list(plan.slot_to_expert[rank])] * 2
            assert identities[0, :, 1].eq(0).all() and identities[1, :, 1].eq(1).all()
            assert identities[:, :, 2].eq(plan.config.weight_version).all()
            assert identities[:, :, 3].eq(0).all()
            assert identities[:, :, 4].eq(2).all() and identities[:, :, 5].eq(1).all()
            assert state.retired.eq(1).all().item()
            if physical_budget:
                assert state.chunks.eq(2).all().item()
                assert state.ready.eq(2).all().item()
                assert times[3] > 0 and times[4] > 0
                owns = any(
                    b.execution_rank == rank and b.physical_slot < ref.LOCAL_EXPERTS
                    for b in plan.block_descriptors
                )
                replica = any(
                    b.execution_rank == rank and b.physical_slot >= ref.LOCAL_EXPERTS
                    for b in plan.block_descriptors
                )
                if owns:
                    assert times[0] > 0
                    if delays[0]:
                        assert times[0] < times[3], times
                    if any(delays[1:]):
                        assert times[0] < times[4], times
                if replica:
                    assert times[1] >= times[3] > 0, times
                    assert times[2] >= times[4] > 0, times
                    if any(delays[1:]) and not delays[0]:
                        assert times[1] < times[4], times
            gathered_times = ref._all_gather_stack(state.telemetry).cpu().tolist()
            if rank == 0:
                print("MegaFlux pipeline telemetry", delays, gathered_times, flush=True)
        if static_replica or online:

            def assert_replica_bytes(current_plan):
                from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.static import (
                    StaticReplicaTransport,
                )

                for source_leg, bank_leg in zip(
                    layer._transformed, ws._megaflux_weight_bank, strict=True
                ):
                    # Gather the original owner's kernel-ready bytes, independent
                    # of transport export/staging buffers.
                    source_bytes = torch.cat(
                        [
                            torch.cat(StaticReplicaTransport._pieces(source_leg, slot))
                            for slot in range(ref.LOCAL_EXPERTS)
                        ]
                    ).reshape(ref.LOCAL_EXPERTS, -1)
                    gathered = ref._all_gather_stack(source_bytes)
                    for d in current_plan.replica_descriptors:
                        if d.receiver_rank == rank:
                            actual_bytes = torch.cat(
                                StaticReplicaTransport._pieces(
                                    bank_leg, d.receiver_slot
                                )
                            )
                            torch.testing.assert_close(
                                actual_bytes,
                                gathered[d.owner_rank, d.owner_slot],
                                atol=0,
                                rtol=0,
                            )

            assert_replica_bytes(plan)
            if online:

                def assert_device_plan(expected_plan):
                    from flashinfer.moe_ep.megaflux_plan import pack_plan_records

                    committed = ws._megaflux_planner.committed
                    packed = pack_plan_records(expected_plan)
                    assert committed.blocks[: len(packed.blocks)].cpu().tolist() == [
                        list(row) for row in packed.blocks
                    ]
                    assert committed.replicas[
                        : len(packed.replicas)
                    ].cpu().tolist() == [list(row) for row in packed.replicas]
                    assert committed.slot_to_expert.cpu().tolist() == [
                        list(row) for row in expected_plan.slot_to_expert
                    ]
                    assert committed.statistics.cpu().tolist() == list(
                        map(
                            list,
                            (
                                expected_plan.rank_task_counts,
                                expected_plan.num_active_physical_slots,
                                expected_plan.rank_padded_rows,
                                expected_plan.rank_sf_rows,
                            ),
                        )
                    )
                    assert ws._megaflux_planner.fallback.item() == int(
                        version_mismatch
                        or (
                            requested_capacity == 0
                            and bool(expected_plan.block_descriptors)
                        )
                    )
                    for peer_q in ref._all_gather_stack(committed.q):
                        torch.testing.assert_close(peer_q, committed.q, atol=0, rtol=0)

                assert_device_plan(plan)
                shifted = torch.where(
                    ids >= 0, (ids + ref.LOCAL_EXPERTS) % fleet.num_experts, ids
                )
                shifted[(rank * 17) % ref.TOKENS_PER_RANK :] = -1
                next_inputs = MoEEpTensors(
                    hidden_states=hidden, topk_ids=shifted, topk_weights=weights
                )
                next_actual = layer.forward(next_inputs).clone()
                torch.testing.assert_close(
                    next_actual, baseline.forward(next_inputs), atol=0, rtol=0
                )
                next_global_ids = ref._all_gather_stack(shifted)
                next_plan = plan_forward(
                    next_global_ids.cpu().tolist(),
                    plan.config,
                    invocation_epoch=3,
                    identity=requested_capacity == 0 or version_mismatch,
                )
                assert_device_plan(next_plan)
                assert_replica_bytes(next_plan)
                next_expected = ref._global_route_reference(
                    global_hidden,
                    next_global_ids,
                    global_weights,
                    global_raw,
                    require_all_experts=False,
                )
                ref._assert_matches_reference(
                    ref._all_gather_stack(next_actual), next_expected, launch=1
                )
                return
            new_plan = plan_forward(
                global_ids.cpu().tolist(),
                plan.config,
                invocation_epoch=2,
                static_replica=(static_spec[0] + 1, static_spec[1], static_spec[2]),
            )
            # Same addresses and physical slot, different logical expert.
            backend = layer._kernel
            backend.rebind_plan(ws, new_plan)
            torch.testing.assert_close(layer.forward(inputs), actual, atol=0, rtol=0)
            assert_replica_bytes(new_plan)
            return
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            captured = layer.forward(inputs)
        for _ in range(16):
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            torch.testing.assert_close(captured, actual, atol=0, rtol=0)
    except Exception:
        # A rank-local failure must not enter collective free while peers are
        # still executing GPU barriers. torchrun terminates the other ranks.
        import os
        import traceback

        traceback.print_exc()
        os._exit(1)
    finally:
        graph = None
        if synchronous is not None:
            synchronous.destroy()
        if baseline is not None:
            baseline.destroy()
        if layer is not None:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)
