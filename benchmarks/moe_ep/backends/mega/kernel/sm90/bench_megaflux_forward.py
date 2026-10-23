# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Six-way end-to-end ablation, with identical routes, weights and compute tactic.

Fixed variants freeze the warmed GPU Plan and receiver tables outside timing.
All transfers and input/output work remain inside each measured forward.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("NVSHMEM_DISABLE_NVLS", "1")


def main():
    import numpy as np
    import torch
    import torch.distributed as dist
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
        bootstrap_moe_ep_runtime,
        finalize_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_identity import (
        Sm90_Mxfp4_MegaFluxIdentityConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl.tuner import (
        create_tuning_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.megaflux_plan import PlanConfig, plan_forward
    from flashinfer.moe_ep.sm90_routing import generate_sm90_routing_numpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--gate-block-us", type=float)
    parser.add_argument("--gate-fixed-us", type=float)
    parser.add_argument("--gate-transfer-us", type=float)
    parser.add_argument("--gate-margin-us", type=float, default=0)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=2304)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--routing", choices=("balanced", "hotspot"), default="balanced"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    if args.gate_block_us is not None and (
        args.gate_fixed_us is None or args.gate_transfer_us is None
    ):
        parser.error("explicit block cost requires fixed and transfer costs")
    if args.tokens <= 0 or args.iters <= 0 or args.warmup < 1:
        parser.error("positive tokens/iters and at least one warmup are required")
    rank, world, local = (
        int(os.environ.get(k, d))
        for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0))
    )
    bootstrap = BootstrapConfig(world_size=world, rank=rank, device=local)
    ensure_moe_ep_cuda_device(bootstrap)
    common = dict(
        intermediate_size=args.intermediate,
        top_k=args.top_k,
        swap_ab=True,
        pingpong=False,
        mma_tiler_mnk=(128, 32, 128),
        cluster_shape_mnk=(1, 1, 1),
        dedup_dispatch=False,
        fold_producer_warps=False,
        token_back_mode="epi_warps",
        load_balance_mode="static",
        gate_up_clamp=10.0,
    )
    original_config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(**common)
    backend = create_mega_kernel(original_config)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, backend.runtime_requirements(bootstrap)
    )
    layers = []
    try:
        if args.routing == "balanced":
            routes = generate_sm90_routing_numpy(
                routing_profile="block_permutation_v1",
                world_size=world,
                tokens=args.tokens,
                topk=args.top_k,
                total_experts=args.experts,
                seed=1234,
            )
        else:
            routes = np.broadcast_to(
                np.arange(args.top_k), (world, args.tokens, args.top_k)
            ).copy()
        plan = plan_forward(
            routes.tolist(),
            PlanConfig(args.experts, world, 0, args.tokens, args.top_k),
            invocation_epoch=1,
        )
        identity_config = Sm90_Mxfp4_MegaFluxIdentityConfig(**common, plan=plan)
        device = torch.device("cuda", local)
        raw = create_tuning_weights(
            local_experts=args.experts // world,
            hidden=args.hidden,
            intermediate=args.intermediate,
            rank=rank,
            device=device,
        )
        transformed = preprocess_mega_weights(
            raw, intermediate_size=args.intermediate, hidden_size=args.hidden
        )
        del raw
        gen = torch.Generator(device=device).manual_seed(42 + rank)
        hidden = torch.randn(
            args.tokens, args.hidden, dtype=torch.bfloat16, device=device, generator=gen
        )
        weights = torch.softmax(
            torch.randn(args.tokens, args.top_k, device=device, generator=gen), dim=-1
        )
        ids = torch.tensor(routes[rank], device=device, dtype=torch.int64)
        from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_pipeline import (
            Sm90_Mxfp4_MegaFluxPipelineConfig,
        )
        from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.benefit import (
            BenefitCosts,
        )

        pc = PlanConfig(args.experts, world, args.budget, args.tokens, args.top_k)
        costs = BenefitCosts(
            1 if args.gate_block_us is None else args.gate_block_us,
            args.gate_fixed_us or 0,
            args.gate_transfer_us or 0,
            args.gate_margin_us,
        )

        def pipeline_config(**extra):
            return Sm90_Mxfp4_MegaFluxPipelineConfig(
                **common, plan_config=pc, selective_weight_copy=True, **extra
            )

        outputs = []
        results = {}
        fixed_signature = None
        for name, config in (
            ("original", original_config),
            ("identity", identity_config),
            ("fixed_sync", pipeline_config(synchronize_weight_copy=True)),
            ("fixed_pipeline", pipeline_config()),
            ("online_pipeline", pipeline_config()),
            ("online_gated_pipeline", pipeline_config(benefit_costs=costs)),
        ):
            if name == "online_gated_pipeline" and args.gate_block_us is None:
                observed = results["online_pipeline"]
                block_us = results["original"]["p50_us"] / max(
                    1, max(plan.rank_task_counts)
                )
                fixed_us = (
                    observed["diagnostic_planner_us"]
                    + observed["diagnostic_staging_and_launch_us"]
                )
                transfer_us = max(
                    0,
                    max(observed.get("diagnostic_leg_completion_us", [0]))
                    - observed["diagnostic_staging_and_launch_us"],
                )
                costs = BenefitCosts(
                    block_us, fixed_us, transfer_us, args.gate_margin_us
                )
                config.benefit_costs = costs
            layer = MoEEpLayer(
                bootstrap=BootstrapConfig(
                    world_size=world, rank=rank, device=local, auto_bootstrap=False
                ),
                fleet_params=FleetParams(
                    num_experts=args.experts,
                    max_tokens_per_rank=args.tokens,
                    token_hidden_size=args.hidden,
                ),
                weights=None,
                backend=MegaConfig(
                    megakernel=config,
                    quantize_input=True,
                    preprocess_weights=False,
                    transformed_weights=transformed,
                ),
            )
            layers.append(layer)
            output = torch.empty_like(hidden)
            inputs = MoEEpTensors(
                hidden_states=hidden, topk_ids=ids, topk_weights=weights, output=output
            )
            for _ in range(args.warmup):
                layer.forward(inputs)
            if name == "identity":
                from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.identity import (
                    IdentityFrontend,
                )

                if not isinstance(
                    layer._ensure_workspace()._frontend, IdentityFrontend
                ):
                    raise RuntimeError(
                        "identity benchmark did not select the Plan frontend"
                    )
            ws = layer._ensure_workspace()
            if name.startswith("fixed_"):
                planner = ws._megaflux_planner

                # Routes are immutable in this ablation; freeze outside timing.
                def frozen_run(ids, *, rank, epoch, external_failure=None):
                    if isinstance(epoch, torch.Tensor):
                        planner.committed.meta[5:6].copy_(epoch)
                    else:
                        planner.committed.meta[5].fill_(epoch)
                    return planner.committed

                planner.run = frozen_run
            torch.cuda.synchronize()
            samples = []
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(args.iters):
                dist.barrier()
                start.record()
                layer.forward(inputs)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
            values = torch.tensor(samples, device=device, dtype=torch.float64)
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
            maximum = values.cpu().numpy()
            results[name] = dict(
                rank_max_us=maximum.tolist(),
                p50_us=float(np.quantile(maximum, 0.5)),
                p95_us=float(np.quantile(maximum, 0.95)),
                p99_us=float(np.quantile(maximum, 0.99)),
            )
            if hasattr(ws, "_megaflux_planner"):
                planner = ws._megaflux_planner
                transport = ws._megaflux_transport
                # Separate diagnostic pass: event instrumentation is outside samples.
                phase_events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                leg_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                original_run, original_copy = planner.run, transport.copy

                def timed_run(*pos, **kw):
                    phase_events[0].record()
                    result = original_run(*pos, **kw)
                    phase_events[1].record()
                    return result

                def timed_copy(*pos, **kw):
                    phase_events[2].record()
                    result = original_copy(*pos, **kw)
                    phase_events[3].record()
                    for event, stream in zip(
                        leg_events, ws._megaflux_pipeline.streams, strict=True
                    ):
                        event.record(stream)
                    return result

                planner.run, transport.copy = timed_run, timed_copy
                layer.forward(inputs)
                torch.cuda.synchronize()
                planner.run, transport.copy = original_run, original_copy
                state = ws._megaflux_pipeline
                results[name]["diagnostic_planner_us"] = (
                    phase_events[0].elapsed_time(phase_events[1]) * 1000
                )
                results[name]["diagnostic_staging_and_launch_us"] = (
                    phase_events[2].elapsed_time(phase_events[3]) * 1000
                )
                if state.issued:
                    results[name]["diagnostic_leg_completion_us"] = [
                        phase_events[2].elapsed_time(event) * 1000
                        for event in leg_events
                    ]
                phase_values = torch.tensor(
                    [
                        results[name]["diagnostic_planner_us"],
                        results[name]["diagnostic_staging_and_launch_us"],
                        *results[name].get("diagnostic_leg_completion_us", [0, 0]),
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(phase_values, op=dist.ReduceOp.MAX)
                phase_values = phase_values.cpu().tolist()
                results[name]["diagnostic_planner_us"] = phase_values[0]
                results[name]["diagnostic_staging_and_launch_us"] = phase_values[1]
                results[name]["diagnostic_leg_completion_us"] = phase_values[2:]
                results[name]["diagnostic_progress_globaltimer_ns"] = (
                    state.telemetry.cpu().tolist()
                )
                signature = hashlib.sha256(
                    planner.committed.q.cpu().numpy().tobytes()
                    + planner.committed.slot_to_expert.cpu().numpy().tobytes()
                ).hexdigest()
                results[name]["assignment_sha256"] = signature
                if name == "fixed_sync":
                    fixed_signature = signature
                if name == "fixed_pipeline":
                    assert signature == fixed_signature, (
                        "fixed ablations used different Plans"
                    )
                counts = planner.committed.statistics.cpu().tolist()
                results[name]["rank_blocks"] = counts[0]
                results[name]["rank_padded_rows"] = counts[2]
                results[name]["rank_sf_rows"] = counts[3]
                results[name]["replicas"] = planner.committed.meta[3].item()
                results[name]["selective_transport"] = transport.selective
                replica_count = (
                    planner.committed.slot_to_expert[:, pc.num_owned_experts_per_rank :]
                    >= 0
                ).sum(1)
                leg_bytes = sum(root.shape[1] for root in transport.exports)
                export_q = planner.committed.q.clone()
                experts = torch.arange(args.experts, device=device)
                export_q[experts, experts // pc.num_owned_experts_per_rank] = 0
                export_counts = export_q.ne(0).any(1).reshape(world, -1).sum(1)
                results[name]["export_staging_bytes_by_owner"] = (
                    (export_counts * leg_bytes).cpu().tolist()
                )
                results[name]["receiver_cache_capacity_bytes_per_rank"] = sum(
                    root.numel() for root in transport.caches
                )
                results[name]["weight_bytes_by_receiver"] = (
                    (replica_count * leg_bytes).cpu().tolist()
                    if transport.selective
                    else [world * transport.export_capacity * leg_bytes] * world
                )
                if planner.benefit_gate is not None:
                    results[name]["gate_costs_us"] = vars(costs)
                    results[name]["gate_local_metrics_us"] = (
                        planner.benefit_gate.metrics.cpu().tolist()
                    )
                    results[name]["collective_fallback"] = planner.fallback.item()
            outputs.append(output.clone())
            torch.testing.assert_close(outputs[0], outputs[-1], atol=0, rtol=0)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "variant": name,
                            "result": results[name],
                            "bitwise_original": True,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            layer.destroy()
            layers.pop()
            torch.cuda.empty_cache()
        for output in outputs[1:]:
            torch.testing.assert_close(outputs[0], output, atol=0, rtol=0)
        if rank == 0:
            print(
                json.dumps(
                    dict(
                        gpu=torch.cuda.get_device_name(),
                        world_size=world,
                        torch=torch.__version__,
                        workload=vars(args),
                        tactic=common,
                        weight_seed="0x4D584650+rank",
                        activation_seed="42+rank",
                        routing_seed=1234 if args.routing == "balanced" else None,
                        beta=plan.config.beta,
                        rank_blocks=plan.rank_task_counts,
                        rank_padded_rows=plan.rank_padded_rows,
                        rank_sf_rows=plan.rank_sf_rows,
                        valid_routing_assignments=sum(plan.expert_counts),
                        replica_count=0,
                        metadata_copy_bytes_per_rank=(
                            (world + 1) * (args.experts // world) * 8
                            + (args.experts // world)
                            * world
                            * args.tokens
                            * args.top_k
                            * 4
                        ),
                        results=results,
                    ),
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        for layer in reversed(layers):
            layer.destroy()
        finalize_moe_ep_runtime(runtime)


if __name__ == "__main__":
    main()
