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

## Graph mode and consumer wait metrics

`--cuda-graph` captures each of the same six variants after warmup, then times
replay. Pipeline variants use the Graph-qualified executor with device epochs
and joined communication branches. Fixed variants still freeze only the Plan
and routing tables; each replay transfers full selected weights. All final
outputs must match captured original bitwise. Per-phase CUDA-event diagnostics
are omitted (`null`, with an explicit unavailable flag) in Graph mode rather
than injecting a different eager timeline into the reported replay costs.
Graph calibration uses a measured end-to-end residual: original p50 divided by
owner maximum blocks is the block coefficient, and the nonnegative difference
between preceding online p50 and modeled candidate compute is the fixed cost.
Transfer is included in that residual; a zero separate transfer coefficient does
not remove its paid time. Coefficients are workload-specific, frozen before the
gated sample window and persisted, not learned inside forward.

The v3 pipeline adds directly measured `max_consumer_weight_wait_us` for W1/W2,
reduced MAX over all ranks in an untimed inspection. A ready acquire that succeeds
immediately leaves zero; a failed acquire measures its actual polling duration.
This is maximum individual consumer stall, not a sum or a global critical-path
penalty, because other owned work can overlap. Progress timestamps retain only
the original five values. All pipeline end-to-end times include the wait-counter
instrumentation; only failed acquires issue MAX atomics. Native compiled kernel
attributes (registers, local bytes, static shared memory and dynamic shared-memory
limit), actual persistent clusters and communication reservation are recorded.
The maximum dynamic shared-memory attribute is a configured limit, not measured
per-launch usage.

`--reference` runs the owner-streamed instruction oracle outside timing, as
described in `megaflux-target-reference.md`. Its EP8 equivalence test first
checks exact agreement with the established global reference at a small shape.
All final eager target workloads run this independent check. Graph workloads
share those weights/routes and require bitwise equality to their original.

GPU occupancy changed during the session: earlier checks found roughly
126–131GiB allocated per GPU with active co-tenants, while a later snapshot
(`forward-shared-gpu-environment.txt`) found roughly 10GiB and zero utilization
between work. There was no exclusive reservation or clock control. Keep paired
raw samples and this limitation; a single snapshot does not prove isolation
throughout a benchmark or support broad speedup claims.

## Final tiled-copy and wait-counter target results

All six eager and six Graph target workloads pass bitwise comparison with the
original path. Each eager workload also passes the owner-sharded independent
instruction reference. Fixed sync and pipeline Plans share the same assignment
hash. The benchmark uses H5120/I2304/E384/K6/EP8, five warmups and 30 rank-max
samples. Values below are p50 microseconds; p95/p99, every sample, byte counts,
phase diagnostics, resource attributes and calibration inputs are in the linked
raw logs.

| Workload | Original | Identity | Fixed sync | Fixed pipeline | Online pipeline | Online gated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eager balanced T64 | 823 | 966 | 1673 | 1668 | 5873 | 5433 |
| Eager balanced T512 | 1426 | 1566 | 3305 | 3083 | 7869 | 7315 |
| Eager hotspot T512 | 6450 | 6598 | 3643 | 3756 | 8360 | 12661 |
| Graph balanced T64 | 475 | 473 | 731 | 714 | 4693 | 4691 |
| Graph balanced T512 | 1084 | 1082 | 2900 | 2689 | 7449 | 6277 |
| Graph hotspot T512 | 6048 | 6033 | 3182 | 3255 | 7852 | 11666 |

The online GPU planner and route exchange dominate online latency. The calibrated
gate rejects the replica Plan in these runs; the rejected path still pays online
planning, so it is several times slower than the fixed-plan and original paths.
This confirms safe selection, not useful online speedup. Fixed-plan selective
replication improves the hotspot median in both eager and Graph runs, while
balanced runs regress. The wait counter observes nonzero W1 consumer waits for
the skewed workload (about 345 us eager and 296 us Graph for one fixed pipeline
sample, and about 353/289 us online); these per-forward diagnostics are not
summed into a critical-path estimate. Measurements ran on a shared H200 host
without clock or device isolation, so they are workload evidence rather than a
universal speedup claim.

Raw final runs: [eager balanced T64](megaflux-results/forward-final-eager-balanced-64.txt),
[eager balanced T512](megaflux-results/forward-final-eager-balanced-512.txt),
[eager hotspot T512](megaflux-results/forward-final-eager-hotspot-512.txt),
[Graph balanced T64](megaflux-results/forward-final-graph-balanced-64.txt),
[Graph balanced T512](megaflux-results/forward-final-graph-balanced-512.txt),
[Graph hotspot T512](megaflux-results/forward-final-graph-hotspot-512.txt).
