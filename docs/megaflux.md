# SM90 MXFP4 MegaFlux forward

This experimental opt-in implementation adds bounded expert replication, a
GPU forward Plan, complete-leg W1/W2 transfer/compute overlap, device epochs,
CUDA Graph replay and collective benefit fallback to the SM90 MXFP4 MegaMoE
backend. It preserves the Humming packed weights, folded offsets, residual x64,
FP8 WGMMA arithmetic and BF16 per-route return contract. Original backend
selection is unchanged. The implementation is in the maintained
`cute_dsl/megamoe/megaflux_sm90` tree; vendored kernel `src` files are unchanged.

Optional single-node NVLink metadata exchange is selected with
`metadata_transport="nvlink"`; the default remains NCCL. See
[the transport contract and measurements](megaflux-nvlink-metadata.md).
Stability, model accuracy and 16K/32K output measurements are tracked in
[the long-context campaign report](megaflux-long-context.md).

Use explicit imports:

```python
from flashinfer.moe_ep.megaflux_plan import PlanConfig
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_graph import (
    Sm90_Mxfp4_MegaFluxGraphConfig,
)
from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.benefit import BenefitCosts

config = Sm90_Mxfp4_MegaFluxGraphConfig(
    intermediate_size=2304,
    top_k=6,
    plan_config=PlanConfig(384, 8, 1, 512, 6),
    selective_weight_copy=True,
    # Optional: supply measured microsecond costs for this exact workload.
    # benefit_costs=BenefitCosts(block_us, fixed_us, transfer_us, margin_us),
    gate_up_clamp=10.0,
    load_balance_mode="static",
)
```

Pass `config` as `MegaConfig(megakernel=config, ...)` to `MoEEpLayer` with the
usual bootstrap, fleet parameters and MXFP4 weight preprocessing. Logical E384,
EP8 and R1 imply L48 owned and C49 physical slots per rank. The planner supports
P<=8, E<=512 and R<=4; configuration and routing capacities are fixed at workspace
creation. Only -1 is a padded route, and duplicate experts in one valid top-k row
are invalid. The input is a fixed-capacity padded IDs tensor. Weight version and
packed layout are immutable for a workspace. Manifest/version/capacity failures
commit owner-only routing collectively rather than leaving a partial Plan.

Warm all allocations, JIT compilation and communication before Graph capture.
The Graph config increments the epoch on GPU and rejoins W1/W2 streams before
return. One workspace supports one in-flight forward on a serialized stream;
use distinct workspaces for overlapping requests. Explicitly destroy layers
before finalizing NVSHMEM. The eager pipeline config is available in the sibling
`fp8_mxfp4_bf16_megaflux_pipeline` package.

Selective copies use mapped NVSHMEM symmetric export roots on a single-node
NVLink clique. An initialization-time collective capability check falls back to
conservative full-root gets when any peer is unmapped. No ordinary weight-bank
pointer is interpreted remotely. Cache allocation remains conservative; only
remote load traffic is reduced. Two SM clusters are reserved for W1/W2 progress.

A benefit model is optional and has no assumed universal calibration. Rejection
still pays statistics, planning, routing exchange, pack/unpack launches and
synchronization. Measure full forward. The serial bounded GPU planner is a
correctness-first implementation and remains a latency bottleneck; this backend
is not automatically selected and should not be presumed faster than original.

See [Plan ABI](megaflux-plan-abi.md), [GPU planner](megaflux-gpu-planner.md),
[pipeline](megaflux-w1-w2-pipeline.md), [Graph](megaflux-graph.md),
[selective transport](megaflux-selective-transport.md),
[benefit gate](megaflux-benefit-gate.md) and
[forward benchmark](megaflux-forward-benchmark.md) for acceptance evidence,
raw rank-max iterations, calibration and measured limitations. Tests and
benchmarks requiring GPUs run in `/root/flashinfer` via `ssh p2p-test-lrf`.
