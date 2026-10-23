# SM90 MegaFlux offline identity executor

This development backend executes an explicit, static, owner-only Plan (R=0).
It does not implement expert replication, GPU planning, transfer readiness,
changing-Plan Graph replay or a benefit gate. Its purpose is to establish the
identity execution boundary before adding physical replica slots.

**Evidence correction:** the initial layer-level results in commit `b7e999563`
used an inherited workspace-pool factory that bypassed the adapter. Those
layer-level identity claims and timings are withdrawn. Directly bound
single-rank adapter tests were valid. The corrected backend is unpooled;
tests/benchmarks assert the actual frontend and communication types and
workspace separation. `superseded-pooled-*` logs preserve invalid evidence;
current identity logs and the table below are replacement runs.

## Enable explicitly

Importing the development backend registers its distinct kernel name. The
original config and default selection continue to use the original backend.

```python
from flashinfer.moe_ep import MegaConfig
from flashinfer.moe_ep.megaflux_plan import PlanConfig, plan_forward
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_identity import (
    Sm90_Mxfp4_MegaFluxIdentityConfig,
)

# offline_ids_by_rank: host [P][tokens][top_k] IDs, with -1 padding.
# This is a test/offline workflow, not a production router CPU readback.
plan = plan_forward(
    offline_ids_by_rank,
    PlanConfig(E, P, 0, token_capacity, top_k),
    invocation_epoch=1,
)
backend = MegaConfig(
    megakernel=Sm90_Mxfp4_MegaFluxIdentityConfig(
        intermediate_size=intermediate,
        top_k=top_k,
        plan=plan,
        load_balance_mode="static",
    ),
    quantize_input=True,
    preprocess_weights=True,
)
# Pass backend to the usual MoEEpLayer constructor with the same FleetParams.
```

Plan/fleet shapes must agree before allocation. Use a new layer/workspace to
change the Plan in this version. Supply matching source IDs to every forward;
the backend checks them with a device assertion before dispatch, including
during Graph capture/replay. A contract violation aborts execution; no rank
silently falls back. It is not a recoverable error protocol: a device assertion
can leave peers waiting, so discard the failed distributed invocation/processes.
Negative-path collective recovery and changing-Plan commits remain later work.

Warm all ranks before capture. Routing weights and activations may change
while IDs remain fixed. No dedup or grouped return is used. The kernel geometry
is frozen to swap-AB `(128,32,128)`, cluster `(1,1,1)`, beta 32, SF padding 128,
without pingpong or tail-split. R>0 and incompatible geometries are rejected.
The existing packed source-word and count limits (int32 and low uint32 fields)
are checked in addition to the Plan's int64 capacity checks.

## How it executes

The maintained implementation lives under
`flashinfer/moe_ep/cute_dsl/megamoe/megaflux_sm90/`. It accesses the kernel drop
only through public shim exports. No vendored `src/` bytes are changed.

1. Before warmup, lower the trusted CPU Plan into per-source receiver counts
   and stable `(physical_slot, source_rank, source_ordinal)` source words.
2. Each forward restores those arrays with three D2D copies into the existing
   receiver metadata workspace. These copies are part of the measured path.
3. A maintained communication subclass replaces atomic-ticket dispatch prep
   with Plan metadata. It publishes the expected count-completion bits using
   the actual persistent grid size and preserves the grid/NVLink rendezvous.
4. Existing dispatch pull consumes the Plan's counts and source words. For
   R=0, physical slot order equals owner expert order, so its token/SF prefixes
   and scheduler work are the exact lowering of the Plan's blocks.
5. The original FC1/FC2, Humming scales/residual compensation, clamp, source
   metadata, per-route BF16 return and final top-k reduction are reused.
6. Original tail cleanup remains intact. Counters are restored next invocation,
   rather than relying on the previous invocation's values.

The shim adds a compile-time construction hook; its default creates the same
original kernel. The adapter preserves the communication constructor's complete
warp/barrier policy. Raw launch thunks that bypass metadata restoration are
explicitly unsupported.

This is an R=0 lowering, not the completed physical-capacity refactor. Before
R>0, token/SF offsets must be materialized in physical slot order; logical E
must remain separate from expanded weight/TMA axes, pool/counter capacity and
scheduler limits. The adapter deliberately rejects replicas so the existing
owner-only dimensions cannot silently be reused for replicated work.

## GPU validation

All GPU runs use `ssh p2p-test-lrf`, checkout `/root/flashinfer`, 8 H200 GPUs,
with the environment described in `megaflux-sm90-bringup.md`.

```bash
MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q \
  tests/moe_ep/test_megaflux_identity.py -m 'arch_hopper and not gpu_2'

PYTHONPATH=/tmp/megaflux-deps NCCL_NVLS_ENABLE=0 NVSHMEM_DISABLE_NVLS=1 \
python3 -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -q \
  tests/moe_ep/test_megaflux_identity.py -m 'gpu_2 and arch_hopper'
```

Single rank: 3 passed, dispatch warps 1/2/4. EP8: 4 passed per rank, including
normal cross-rank routing, zero-valid source, all-masked input and sparse owner
placement. Each EP case compares exactly to the original backend, checks the
independent numerical oracle, repeats the same workspace and replays a Graph
16 times. This establishes fixed-Plan R=0 correctness only. Target-shape paired
benchmark also requires bitwise output equality between the two backends.
The earlier EP2 layer result also used the pooled baseline and is superseded.
The corrected physical-slot EP2 check passes cross-rank execution and 16 Graph
replays (1 passed per rank); see `megaflux-physical-slots.md`. The config
rejection test passes (1 passed). Original single-rank + identity regression: 6 passed;
original EP8 regression: 10 passed per rank. Raw results are retained under
`megaflux-results/identity-*.txt`.

## Identity overhead ablation

```bash
PYTHONPATH=/tmp/megaflux-deps NCCL_NVLS_ENABLE=0 NVSHMEM_DISABLE_NVLS=1 \
python3 -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmarks/moe_ep/backends/mega/kernel/sm90/bench_megaflux_identity.py \
  --tokens 512 --warmup 5 --iters 30
```

This uses identical kernel-ready weights, activations, routing, static tactic
and preallocated outputs for both paths. It includes staging, route validation,
Plan metadata restore, compute and final reduction. CUDA events include host
launch gaps; barriers before the start event and the later timing all-reduce
are excluded. Raw per-iteration times are reduced by MAX across ranks before
quantiles. Offline Plan construction/upload and warmup are excluded; no online
planner is claimed. The script also supports `--routing hotspot`.

H200 EP8, H5120/I2304/E384/top-k6:

| Routes / tokens per rank | Complete forward | p50 us | p95 us | p99 us |
| --- | --- | --- | --- | --- |
| Balanced / 64 | Original | 847.440 | 1873.974 | 2408.974 |
| Balanced / 64 | Identity Plan | 1042.272 | 1194.406 | 2180.195 |
| Balanced / 512 | Original | 1421.200 | 1621.320 | 2285.267 |
| Balanced / 512 | Identity Plan | 1614.480 | 1905.011 | 2125.114 |
| Hotspot / 512 | Original | 6467.056 | 6925.122 | 7053.839 |
| Hotspot / 512 | Identity Plan | 6597.152 | 6717.200 | 6726.427 |

One paired run per workload; these differences are not evidence of a speedup. This
static Plan removes original dispatch preprocessing while adding validation
and metadata copies. Results are not directly comparable to the earlier
atomic-scheduler baseline or compute-only timings. Raw results and full
workload parameters: `megaflux-results/identity-bench-{64,512,hotspot-512}.txt`.
The real adapter adds about 193 us to balanced-512 p50 in this run; that
overhead must be charged when judging replication benefits.
The legal hotspot selects distinct experts 0..5, all on rank 0. All 768 token
blocks remain on rank 0: this confirms the identity limitation rather than
claiming load balancing. Timing outliers are retained in the raw data.

Pending: physical capacity beyond owned, static replicas, complete-leg byte
verification, GPU planner/consistent fallback, delay injection, epoch/slot
retirement, changing-plan Graph stress, benefit gate and complete ablations.
