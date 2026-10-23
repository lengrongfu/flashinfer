# SM90 MegaFlux W1/W2 pipeline

The explicit `sm90_fp8_mxfp4_bf16_megaflux_pipeline` backend keeps the online
GPU Plan and complete-leg export layout unchanged and changes transfer timing.
It is a forward correctness implementation; selective transfer and benefit
measurement remain later work. CUDA Graph qualification is a separate stage.

## Data and readiness

Every owner packs both complete kernel-ready legs into tracked symmetric
export roots. A stream-ordered NVSHMEM barrier publishes these snapshots.
Separate W1 and W2 CUDA streams perform exact-byte root gets. Each owner-root
chunk is marked complete only after NVSHMEM quiet. Each stream unpacks packed
payload, folded offsets and residual scales, then publishes its leg/slot epoch
with a system release store. Merely enqueueing a get never publishes ready.
The conservative all-owner export-root traffic is still charged overhead.

Ready flags, current epoch, chunk completion, per-leg slot identity/retirement
and telemetry are appended local
workspace regions, beyond the original zeroed counter prefix. Consumer
addresses derive from the existing runtime FC1-ready pointer plus checked
layout offsets. Device addresses are not embedded as compilation constants.
Owned experts are ready immediately for both legs; replicas acquire the
current leg's exact epoch. After acquisition, the TMA producer issues
`fence.proxy.async.global` before reading weights or folded offsets. Residual
consumers also acquire readiness before their first generic load.

The original hybrid epilogue eagerly reads both legs' residuals during FC1.
The maintained BSD-attributed task-loop copy reads only the active phase's
residual and preserves its arithmetic, barriers, stores and token return.
The maintained weight helper adds acquisition before the original TMA/offset
load. Original vendor source files remain unchanged; shim exports provide
narrow dependencies and workspace views.

## Progress and retirement

The frontend reduces actual persistent launch capacity by two clusters,
reserving independent execution capacity for the two communication streams.
This changes `max_active_clusters` passed to the kernel, not only a scheduler
hint. The original dispatch/grid/NVLink barrier counts derive from this
reduced launch count. CTA shape, WGMMA fragments, shared storage, named-barrier
layout and per-role register policy retain the frozen N32/C1x1 design; the
additional readiness code uses scalar pointer/epoch registers. Independent
progress must be established by delay tests, not assumed from stream creation.

Before rebuilding the next Plan or repacking exports, the main stream joins
both old transfer completion events. A pre-pack rendezvous retires old compute and every cross-rank export reader
before any owner overwrites its root; a separate post-pack rendezvous publishes
the new complete snapshots. Per-slot records retain expert, projection, model
version, layout, current epoch and previous retired epoch. Destruction drains both transfer streams
before freeing symmetric roots. The stage-six backend rejects capture.

## Validation

The test compares a separate synchronous online layer executing the same q,
the original backend, and the independent FP32 route oracle. It also checks
complete replica bytes, GPU Plan fields, repeated forward and a changed Plan
on the same workspace. Delay cases independently inject 20ms before W1,
before W2, before W2 folded offsets, and before W2 residual stores.

Within each GPU, telemetry records first owned FC1 completion, first replica
FC1 completion, first replica FC2 residual read, and W1/W2 ready publication.
Tests require replica FC1/FC2 timestamps after the corresponding publication.
With W2 delayed they require FC1 completion before W2 publication, establishing
compute progress while the second leg remains unavailable. GPU clocks are
compared only within one rank. All-masked and zero-input-source cases exercise
no-work retirement and reuse; expanded R=2/4 cases exercise larger banks.

```bash
PYTHONPATH=/tmp/megaflux-deps NCCL_NVLS_ENABLE=0 NVSHMEM_DISABLE_NVLS=1 \
  torchrun --standalone --nproc_per_node=8 -m pytest \
  tests/moe_ep/test_megaflux_pipeline.py -q -x -s --disable-warnings
```

Accepted H200 evidence: EP2 main 18 + resource audit 1 cases/rank; EP8 main
18 + resource/E384-K6 supplemental 2 cases/rank; standalone atomic telemetry
1 passed. Main math is H128/I128/M129, with E=4P/K=P and R=0/1/2/4. The
supplemental EP8 case uses E384/K6 at H128/I128; H5120/I2304 timing remains
part of the final benchmark stage. Raw logs are `megaflux-results/pipeline-*.txt`.

Native fused-kernel attributes: 131 registers/thread, 0 static shared bytes,
0 local bytes; configured maximum dynamic shared memory 232448 bytes (this
is the driver limit, not a measurement of allocated dynamic storage). The
CTA has 384 threads with the original epilogue register policy of 216.
131*384=50304 registers before allocation rounding already excludes two
such CTAs per 65536-register SM. The original per-CTA shared-storage layout
has no new arrays; flags/identity live in global workspace. Actual launch is
130 persistent clusters on 132-SM H200, with two clusters reserved. Delay
telemetry demonstrates independent communication progress under this bound.

The shape-only bootstrap metadata constructor does not invoke the CPU routing
oracle. Ruff, focused mypy and all 105 host Plan tests also pass. Capture and
benefit/latency claims are not established by this stage.

## Consumer wait diagnostics

The v3 private pipeline workspace appends two int64 nanosecond maxima to the
five existing progress timestamps (16 additional local bytes/rank). An acquire
that immediately observes the current epoch records no wait. A failed acquire
starts `%globaltimer`, polls with the same system acquire, and atomically MAXs
its elapsed duration before the async proxy fence. Both weight/offset TMA and
residual epilogue consumers contribute to the corresponding W1/W2 maximum.
The counter measures a real individual consumer stall; it is not a sum across
threads or the global critical-path loss, since other owned work may overlap.
No per-forward host readback is added. Control initialization resets both
counters every invocation, including captured replay.

EP8 accepts 14 full-forward cases/rank (including both gated accept and reject)
and five Graph cases/rank x128 replays after this instrumentation. Injecting
W1/W2 start delays of 20ms and additional W2 offset/residual delays yields
roughly 19ms W1 and 59ms W2 maximum waits on replica receivers. Owner-only rows
remain zero. Raw evidence: `consumer-wait-ep8.txt` and
`consumer-wait-graph-ep8.txt`. The compile key advances to v3 for the changed
local workspace layout; packed Plan ABI and numerical computation are unchanged.
