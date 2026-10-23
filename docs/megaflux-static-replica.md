# SM90 MegaFlux static synchronous replica

This opt-in development backend runs one manually selected replica. It follows
the accepted physical-capacity gate and retains the original Humming WGMMA
arithmetic and BF16 per-route return. It does not implement online planning,
W1/W2 overlap or CUDA Graph capture.

## Transfer contract

`StaticReplicaTransport` reserves identical symmetric root buffers on every
PE. Per-leg export capacity is `min(owned, (P-1)*R)` distinct experts per
owner, independently from the R receiver slots. The ordinary physical bank
is never passed as a remote source. Export roots hold the complete packed
payload, folded offset bytes and FP32 residual scale; the fixed unit
placeholder stays local. Packing traverses the underlying `(N, packedK)`
Humming storage, preserving packed-K stride one without quantization.

Each invocation packs active owner exports, publishes them with a stream
barrier and synchronizes. Receivers issue exact-byte
`nvshmem.bindings.getmem_on_stream` between registered symmetric roots,
then quiet and synchronize before D2D unpack into the physical bank.
A final stream barrier/synchronize finishes all payload/offset/residual
writes before any TMA or GEMM consumer. The exact-byte binding avoids
`nvshmem.core.get` copying allocator buffer capacity instead of the logical
leg length. No transfer or replica is cached across invocations.

The path synchronizes the host and rejects capture. All PEs must call
forward, rebind and destroy in the same order. Weight updates require a new
workspace. This first implementation intentionally supports at most one
replica descriptor globally; unused reserved slots are not active experts.

## Offline use

```python
from flashinfer.moe_ep.megaflux_plan import PlanConfig, plan_forward
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_static import (
    Sm90_Mxfp4_MegaFluxStaticConfig,
)

# global_ids is an offline source/rank/token/top-k array, before forward.
plan = plan_forward(
    global_ids, PlanConfig(E, P, R, max_tokens, top_k),
    invocation_epoch=1, static_replica=(expert, receiver_rank, moved_blocks),
)
config = Sm90_Mxfp4_MegaFluxStaticConfig(
    intermediate_size=intermediate, top_k=top_k, plan=plan,
    load_balance_mode="static",
)
# Pass config as MegaConfig(megakernel=config, quantize_input=True,
#                            preprocess_weights=True) to MoEEpLayer.
```

`moved_blocks` counts whole global beta=32 blocks, including a possible global
tail. Zero reserves a dormant descriptor, allowing empty-expert transfer
validation. Each source's routing must match the offline Plan exactly.
`backend.rebind_plan(workspace, newer_plan)` is collective and outside
capture; it retires old consumers before updating the existing metadata
buffers. Geometry, capacity, layout and weight version must match, and the
invocation epoch must increase. New experts are fully re-exported and copied
before slot reuse. These are offline lifecycle guarantees, not the eventual
device epoch/ready-flag protocol.

## Validation

Run from `/root/flashinfer` through `ssh p2p-test-lrf`:

```bash
PYTHONPATH=/tmp/megaflux-deps NCCL_NVLS_ENABLE=0 NVSHMEM_DISABLE_NVLS=1 \
  torchrun --standalone --nproc_per_node=8 -m pytest \
  tests/moe_ep/test_megaflux_identity.py -q \
  -k 'static_replica or expanded_physical_slots_identity' --disable-warnings
```

H200 acceptance: EP2 static **3 passed per rank**; EP8 static plus physical
identity regression **12 passed per rank**. Raw results are in
`megaflux-results/static-replica-ep{2,8}.txt`. Host Plan suite: **105 passed**;
Ruff and focused mypy checks pass.

The static tests use H=128/I=128/M=129, E=4P and K=P, R=1. They compare both
legs' full byte content against gathered original owner tensors, compare
output bitwise against the original backend and independently against the
FP32 route oracle, repeat the workspace and change the slot's expert at
epoch 2. Cases cover cross-rank routing/global tail, a zero-input receiver
that still executes remote expert work, and all-masked/dormant replicas.
Physical identity tests retain the R=1/2/4 Graph regression. Target-shape
replica latency and overlap are not established by these small-shape gates.
