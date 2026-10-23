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


def _run_offline_backend(routing_pattern, physical_budget=None):
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
    if world < 2:
        pytest.skip("launch with torchrun --nproc_per_node=2/4/8")
    bootstrap = BootstrapConfig(world_size=world, rank=rank, device=local)
    ensure_moe_ep_cuda_device(bootstrap)
    # Bootstrap first, then obtain the global offline Plan outside forward.
    initial = create_mega_kernel(
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=ref.INTERMEDIATE,
            top_k=world,
        )
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, initial.runtime_requirements(bootstrap)
    )
    layer = None
    baseline = None
    graph = None
    try:
        raw = ref._make_raw_weights(rank)
        global_raw = ref._gather_raw_weights(raw)
        hidden, ids, weights = ref._make_tokens_and_routes(
            rank, world, launch=0, routing_pattern=routing_pattern
        )
        global_hidden = ref._all_gather_stack(hidden)
        global_ids = ref._all_gather_stack(ids)
        global_weights = ref._all_gather_stack(weights)
        plan = plan_forward(
            global_ids.cpu().tolist(),
            PlanConfig(
                world * ref.LOCAL_EXPERTS,
                world,
                physical_budget or 0,
                ref.TOKENS_PER_RANK,
                world,
            ),
            invocation_epoch=1,
            identity=True,
        )
        config_type = Sm90_Mxfp4_MegaFluxIdentityConfig
        if physical_budget is not None:
            from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_physical import (
                Sm90_Mxfp4_MegaFluxPhysicalConfig,
            )

            config_type = Sm90_Mxfp4_MegaFluxPhysicalConfig
        config = config_type(
            intermediate_size=ref.INTERMEDIATE,
            top_k=world,
            plan=plan,
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
        inputs = MoEEpTensors(hidden_states=hidden, topk_ids=ids, topk_weights=weights)
        actual = layer.forward(inputs).clone()
        from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.identity import (
            IdentityFrontend,
            IdentityTokenComm,
        )

        ws = layer._ensure_workspace()
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
        # Repeat on the same workspace; metadata counters were tail-reset.
        torch.testing.assert_close(layer.forward(inputs), actual, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            captured = layer.forward(inputs)
        for _ in range(16):
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            torch.testing.assert_close(captured, actual, atol=0, rtol=0)
    finally:
        graph = None
        if baseline is not None:
            baseline.destroy()
        if layer is not None:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)
