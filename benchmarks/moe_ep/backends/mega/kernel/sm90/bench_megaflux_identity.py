# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Original vs offline identity Plan, including per-invocation metadata copies.

Run through torchrun in the target GPU container. All ranks use identical
weights, routing, frozen tactics and preallocated outputs for both variants.
JSON includes raw per-iteration rank-maximum CUDA-event times and quantiles.
CPU Plan construction/upload and warmup are outside the measured region;
this is the stage-three identity overhead ablation, not online MegaFlux.
"""

import argparse
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
        outputs = []
        results = {}
        for name, config in (
            ("original", original_config),
            ("identity", identity_config),
        ):
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
            outputs.append(output)
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
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
