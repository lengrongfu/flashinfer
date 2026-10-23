# End-to-end forward ablation

Run `benchmarks/moe_ep/backends/mega/kernel/sm90/bench_megaflux_forward.py`
through torchrun in `/root/flashinfer` with EP8. The six variants share weights,
route seed 1234, activation seed 42+rank, gate clamp, beta, no dedup, return
reduction, tile and cluster geometry. Fixed sync/pipeline variants use the same
deterministic warmed GPU Plan and freeze its routing tables outside timing;
there is no CPU Plan construction in a timed invocation. Quantization, input
staging, export pack, copies, compute and reduction remain inside timing.
Layers are destroyed sequentially to bound memory. Every final output must
match the original bitwise.

Example (GPU work must run via `ssh p2p-test-lrf`):

```sh
PYTHONPATH=/tmp/megaflux-deps NCCL_NVLS_ENABLE=0 NVSHMEM_DISABLE_NVLS=1 \
FLASHINFER_CUDA_ARCH_LIST=9.0a MAX_JOBS=2 \
torchrun --standalone --nproc_per_node=8 \
benchmarks/moe_ep/backends/mega/kernel/sm90/bench_megaflux_forward.py \
--tokens 512 --routing hotspot --warmup 5 --iters 30 --budget 1
```

Defaults are H5120/I2304/E384/K6. Also run balanced T64 and T512. Raw CUDA-event
microseconds are reduced with MAX over ranks for every measured iteration;
JSON reports p50/p95/p99 rather than reducing separate rank percentiles. It
records blocks, padding, replica count, full-leg traffic, owner export staging
and receiver cache allocation separately. Diagnostic phase events run in an
additional invocation outside the timing samples. Planning and staging/launch
spans are rank maxima; leg completion spans start before staging, so they
include staging as well as transfer. GPU globaltimer progress is diagnostic and
must not be confused with an isolated exposed-weight-wait measurement.

When no explicit gate coefficients are supplied, the benchmark fits a conservative
model from its own preceding observations: block cost is original p50 divided by
owner-only maximum block count; fixed overhead is online diagnostic planner plus
staging/launch; additional exposed transfer estimate is the positive difference
between maximum leg completion and staging/launch. Gate coefficients are frozen
before measuring gated forward and persisted in JSON. Optional explicit
`--gate-block-us`, `--gate-fixed-us`, `--gate-transfer-us`, `--gate-margin-us`
allow replaying a calibration. This simple model can overcount shared base work
and does not account for overlap perfectly. It is an empirical gate, not a
proof that a candidate speeds up. Single diagnostic samples on shared GPUs are
noisy; compare actual end-to-end samples and persist the co-tenant limitation.

A rejected gate still pays statistics, planning, agreement, routing exchange,
packing kernel launches and barriers. Selective identity transfer performs no
remote weight loads; its launch/unpack overhead remains timed. The conservative
fallback instead pays full-root traffic, even for identity. Do not omit either
cost when reporting the rejected path. Current optimized copies retain the
conservative cache allocation and do not claim capacity savings.

## Initial target run, before tiled byte copies

All six variants passed bitwise original-output comparison on all three target
workloads (30 samples, five warmups). These measurements exposed a bottleneck
and are retained as `forward-pre-tiled-*.txt`, not presented as final optimized
performance. p50 rank-maximum microseconds:

| Workload | Original | Identity | Fixed sync | Fixed pipeline | Online | Gated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced T64 | 821 | 983 | 4748 | 4739 | 8894 | 8923 |
| Balanced T512 | 1417 | 1580 | 105699 | 109249 | 111975 | 9901 |
| Hotspot T512 | 6425 | 6585 | 107819 | 111440 | 111467 | 15278 |

The gate rejects with these conservative observed costs, but rejection still
pays planning, routing and the old export-clearing overhead. The initial
single-CTA byte-copy path dominates when full-size replicas are active. The
next optimization removes unreachable export-row clearing and tiles pack,
peer loads and unpack over CTAs with aligned four-byte copies. Its acceptance
and rerun results must be recorded separately. The serial GPU planner remains
an explicitly measured limitation rather than being hidden outside timing.
