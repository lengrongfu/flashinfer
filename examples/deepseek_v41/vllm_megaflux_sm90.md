# DeepSeek V4.1 MXFP4 with MegaFlux on SM90

`patch_vllm_megaflux_sm90.py` adds the opt-in
`flashinfer_moe_ep_megaflux_sm90` backend to an installed vLLM 0.32.x package.
It connects vLLM's DeepSeek-V4 routed-expert path to FlashInfer's SM90
MegaFlux graph backend, retaining packed MXFP4 weights and UE8M0 scales. The
patch backs up the four vLLM modules it changes under `/tmp`, preserving their
relative paths, before editing.
It is a deployment patch for the pinned vLLM release, not an upstream vLLM
change.

The adapter also accepts `FLASHINFER_MOE_EP_MEGAFLUX_METADATA=nvlink` to select
the optional NVLink peer metadata transport; `nccl` remains the default. See
[the implementation and measurements](../../docs/megaflux-nvlink-metadata.md).
This switch requires every EP peer to be mapped in the same NVSHMEM group.

Apply it in the serving environment where this FlashInfer checkout is
importable:

```bash
PYTHONPATH=/root/flashinfer:/tmp/megaflux-deps \
  python3 /root/flashinfer/examples/deepseek_v41/patch_vllm_megaflux_sm90.py
```

The FlashInfer checkout must include its MoE-EP runtime dependencies and must
be visible in `PYTHONPATH`. Start a DeepSeek-V4.1 model with EP=8 and explicitly
select the backend:

```bash
PYTHONPATH=/root/flashinfer:/tmp/megaflux-deps \
VLLM_ENGINE_READY_TIMEOUT_S=1800 \
FLASHINFER_MOE_EP_MEGAFLUX_METADATA=nvlink \
NVSHMEM_DISABLE_CUDA_VMM=1 NVSHMEM_SYMMETRIC_SIZE=16G \
vllm serve /gpfs/rd/models/DeepSeek-V4.1-Flash-1017/ \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 --data-parallel-size-local 8 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.85 --enforce-eager \
  --trust-remote-code \
  --max-model-len 512 --max-num-batched-tokens 128 --max-num-seqs 1 \
  --served-model-name deepseek-v41-megaflux \
  --reasoning-parser deepseek_v41 \
  --kernel-config '{"moe_backend":"flashinfer_moe_ep_megaflux_sm90"}'
```

The validated DeepSeek V4.1 EP=8 startup used a 16 GiB static NVSHMEM heap and
`gpu-memory-utilization=0.85`; it allocated about 634k KV-cache tokens per rank.
Keep the `NVSHMEM_*` settings consistent across ranks. The pipeline uses a
compact selective receive cache with `world_size * replica_budget` rows per
layer; the non-selective fallback retains the larger owner-export cache.
Eager mode avoids full-model CUDA graph pools during bring-up.
The adapter nevertheless captures each warmed MegaFlux MoE in its own
fixed-capacity CUDA Graph. This removes repeated Python/kernel launch work
without capturing the entire vLLM model. Input buffers are updated on every
call, padding IDs are cleared to `-1`, and GPU planning, replica identities
and epochs remain dynamic. Set `FLASHINFER_MOE_EP_MEGAFLUX_CUDA_GRAPH=0`
before starting the service to disable this local replay path for comparison.

On H200, the adapter selects owner-only execution below a calibrated global
route threshold: `EP * top_k * min(max_num_batched_tokens, 256) + 1`. H200 EP8
capacity sweeps found replica planning/copying slower for both balanced and
hotspot batches through 256 tokens per rank. Owner-only descriptors are built
cooperatively on GPU. Larger route sets retain replica planning. Override the
threshold with `FLASHINFER_MOE_EP_MEGAFLUX_MIN_REPLICA_ROUTES`; `49` restores
the earlier one-token-per-rank band for EP8/top-k6, while `0` always considers
replica planning. This is an explicit `min_replica_routes` backend setting
(default zero outside this adapter). Capacity/version fallback remains
collective, and its alternate plan is only built when the GPU decision needs it.
Other SM90 GPUs retain the earlier `EP * top_k + 1` default until separately
calibrated; the parallel owner-only planner applies to all supported geometries.

The command above is a bounded bring-up configuration: a 512-token context,
128 tokens per scheduler step, and one sequence per DP rank. Larger limits
change the per-layer workspace and memory requirements and need separate
validation.

DeepSeek V4.1 enables thinking by default. Keep
`--reasoning-parser deepseek_v41` when testing normal chat requests: without
it, vLLM 0.32.0 can return HTTP 200 while placing reasoning text and
`</think>` in `message.content`. With the parser, reasoning and the final
answer are returned separately. For a non-thinking request, set
`"chat_template_kwargs": {"thinking": false}` in the request body.

The implementation is currently limited to DeepSeek-V4 routing with MXFP4
experts on compute capability 9.0, linear expert placement, BF16 activations,
and no EPLB or dual-batch overlap. It uses one reserved replica slot per rank
and static load balancing. Other vLLM backends remain selectable and unchanged.

The graph backend's W1/W2 selective weight copy and compute pipeline are
exercised through `MoEEpMegaLayer`. Confirm `Using FlashInferMoeEpExperts`
and `FLASHINFER_MOE_EP_MEGAFLUX_SM90` in the worker logs, then run:

```bash
python3 examples/deepseek_v41/verify_vllm_megaflux_sm90.py \
  --output /tmp/megaflux-results/vllm-e2e-responses.json
```

The verifier checks answer content and normal completion, serial and four-way
concurrent requests, a prompt spanning multiple 128-token prefill chunks,
multi-step decode, and streaming responses with thinking both disabled and
left at its default. It saves full requests/responses and exits nonzero on
any failure; HTTP 200 alone is not a correctness check. Run it on the serving
host (through the SSH session when using a remote container).

On eight H200 GPUs with vLLM 0.32.0, this bounded configuration passed all
13 checks on 2026-10-23, including non-streaming and streaming default-thinking
requests. This is an integration smoke check, not a model accuracy benchmark
or validation of larger serving limits.

Compare latency only against a separately run vLLM baseline with identical model, topology,
prompt, and generation parameters.

On 2026-10-24, the local MoE replay path was compared with vLLM's automatic
Humming backend using the bounded eight-H200 configuration above. Each case
ran twice (seeds 42 and 43), after two warmup requests, with fixed input/output
lengths and EOS ignored. Throughput below is total output tokens divided by
total measured duration across both rounds:

| Input/output tokens | Client concurrency | Native Humming tokens/s | Optimized MegaFlux tokens/s |
| --- | ---: | ---: | ---: |
| 32/32 | 1 | 5.43 | 6.95 |
| 32/32 | 8 | 16.32 | 25.65 |
| 256/32 | 8 | 18.02 | 32.01 |

Mean time per output token after the first was 135–137 ms for MegaFlux versus
177–178 ms for Humming. Eight-concurrent short-input throughput varied from
20.86 to 33.32 tokens/s across the MegaFlux rounds; these small runs include
DP queuing and are not production saturation results. The native baseline
was measured earlier in the same session, not in an interleaved A/B/A run.

One benchmark case can be reproduced with vLLM 0.32.0 as follows (use seed 43
for the second round; use 24 prompts for the eight-concurrent cases):

```bash
vllm bench serve --backend vllm \
  --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
  --model /gpfs/rd/models/DeepSeek-V4.1-Flash-1017/ \
  --served-model-name deepseek-v41-megaflux --tokenizer-mode deepseek_v41 \
  --dataset-name random --random-input-len 32 --random-output-len 32 \
  --random-range-ratio 0 --num-prompts 8 --max-concurrency 1 \
  --num-warmups 2 --ignore-eos --temperature 0 --seed 42 \
  --save-result --save-detailed --result-dir /tmp/megaflux-results \
  --result-filename megaflux-short-c1-r1.json
```

The optimization also fixes compact replica-cache unpacking for nonzero
owner ranks. Small semantic chat checks alone did not expose this byte-layout
error; the regression now checks the owner-indexed cache row directly, and
multi-rank replay compares changing routes and batch sizes bit-for-bit with
the original fused path. Older MegaFlux timings include that defect and are
historical references rather than correctness-qualified baselines.

### Expanded-context interleaved A/B/A validation (2026-10-24)

The earlier small-load gains did **not** carry over to this larger serving
configuration. Native Humming A1, optimized MegaFlux B, then native Humming A2
each ran three rounds with seeds 42, 43, and 44. This is one A/B/A sequence
with three workload rounds per stage, not three independent service cycles.
No kernel or serving configuration changed during the sequence except the
explicit MegaFlux `--kernel-config` selection for B.

The common settings were eight H200 GPUs, vLLM 0.32.0, TP=1, DP=8, EP=8,
`--enforce-eager`, `--max-model-len 8192`, `--max-num-batched-tokens 256`,
`--max-num-seqs 8` per DP rank, GPU memory utilization 0.85, and the same
16 GiB static NVSHMEM heap. MegaFlux retained local MoE CUDA Graph replay.
Each case used four warmup requests, fixed input length, 64 output tokens,
EOS ignored, temperature zero, and the random dataset. The cases used
24, 32, and 64 measured requests per round, respectively.

Throughput is total output tokens divided by total measured duration over
three rounds; larger is better:

| Input tokens / concurrency | Native A1 tokens/s | MegaFlux B tokens/s | Native A2 tokens/s | B vs A1 | B vs A2 | A2 vs A1 drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 / 8 | 36.98 | 34.82 | 36.31 | -5.85% | -4.12% | -1.81% |
| 4096 / 16 | 46.61 | 43.32 | 45.38 | -7.05% | -4.54% | -2.63% |
| 7168 / 32 | 60.58 | 54.79 | 58.11 | -9.57% | -5.72% | -4.08% |

Per-round output throughput preserves the observed variation:

| Input / concurrency | A1 rounds 1, 2, 3 | B rounds 1, 2, 3 | A2 rounds 1, 2, 3 |
| --- | --- | --- | --- |
| 1024 / 8 | 36.87, 36.60, 37.47 | 34.71, 34.80, 34.94 | 35.26, 37.13, 36.59 |
| 4096 / 16 | 44.13, 47.91, 48.00 | 43.46, 41.81, 44.81 | 41.92, 48.33, 46.39 |
| 7168 / 32 | 57.58, 59.73, 64.90 | 55.65, 53.94, 54.79 | 58.53, 56.78, 59.08 |

P95 below pools individual requests across the three rounds, rather than
averaging the reported percentiles. TPOT is `(latency - TTFT) / (output_len - 1)`.
Lower latency is better:

| Input / concurrency | P95 TTFT seconds A1 / B / A2 | P95 TPOT ms A1 / B / A2 | P95 end-to-end seconds A1 / B / A2 |
| --- | --- | --- | --- |
| 1024 / 8 | 2.92 / 3.24 / 3.04 | 192.97 / 205.72 / 197.19 | 14.87 / 15.81 / 15.16 |
| 4096 / 16 | 13.27 / 13.28 / 13.53 | 218.10 / 236.54 / 213.83 | 26.04 / 26.70 / 25.64 |
| 7168 / 32 | 33.10 / 37.92 / 31.60 | 227.84 / 254.50 / 229.25 | 46.69 / 53.74 / 45.51 |

All 27 measured cases completed: 1080/1080 requests, 5,529,600 input tokens
and 69,120 output tokens. Every saved input/output length was checked against
the intended workload. Each service stage also passed all 13 integration
smoke checks (39/39 total). No runtime OOM, CUDA/NCCL error, or traceback was
found before service shutdown. Ten-second GPU sampling observed peak memory
of 123829 / 123849 / 123835 MiB for A1 / B / A2, active SM clocks of 1980 MHz
in every stage, and peak temperatures of 49 / 44 / 49 degrees C.

The logs contain DP statistics-order warnings (39 / 25 / 31 for A1 / B / A2).
After all their requests finished, explicitly stopping A1 and B also produced
one and two API output-handler `EngineDeadError` tracebacks during shutdown.
These are retained in the evidence and excluded from runtime-error counts;
zero failed requests does not mean the logs were warning-free. The complete
experiment, including service startup and switching, lasted about 54 minutes;
measured benchmark durations totaled 23.86 minutes. These are length-controlled
synthetic workloads, not captured production traffic or a production-duration
stability or model-accuracy qualification.

For this expanded configuration, the optimization does not establish a
throughput benefit: B is below both native aggregates and has worse P95 TPOT.
The 4K TTFT is effectively unchanged, while 7K tail latency is worse. A1/A2
drift and round-to-round queuing variation limit precision, especially for
the larger cases. The bounded 512-context results above must not be treated
as production-load gains.

A follow-up profiling target is fixed-capacity replay: it captures and
quantizes the scheduler capacity even for a much smaller live decode batch.
Increasing that capacity from 128 to 256 can add padding work. This is a
code-based hypothesis, not a cause isolated by this experiment: context,
sequence limit, scheduler capacity, and workload all changed relative to the
earlier bounded runs. Rank-synchronized graph buckets and an isolated capacity
sweep need their own correctness and A/B/A validation before claiming a fix.

Raw evidence is saved on the serving host in
`/tmp/megaflux-results/aba-expanded-20261024/`, with an archive at
`/tmp/megaflux-results/aba-expanded-20261024.tar.gz`. The directory includes
the protocol, every server/client command, all detailed benchmark JSON files,
correctness responses, logs, source hashes, GPU telemetry, and the controller,
analysis, and integrity-check scripts. The local copy is
`/tmp/megaflux-aba-evidence/aba-expanded-20261024/`; its `comparison.json`,
`validation.json`, and `telemetry-summary.json` contain the reported values.
The healthy native A2 service was left running on port 8000 after the test.

### Capacity profiling and targeted planner optimization (2026-10-24)

An EP8 single-layer controlled experiment used the production geometry
E=384, hidden=5120, intermediate=2304, top-k=6, identical deterministic
weights/routes, five warmups, and three rounds of 40 CUDA-event samples.
Each sample takes the maximum across ranks. Separate PyTorch CUDA traces
were captured outside the timing samples. These are controlled single-layer
measurements, not end-to-end serving throughput.
These diagnostic processes used a 2 GiB static NVSHMEM heap with NVLS disabled;
the serving comparison keeps its previous 16 GiB heap and serving defaults.
Every diagnostic contrast holds its communication environment fixed.

With four live tokens per rank and a fixed 256-token workspace, changing
the captured input length from 8 to 256 increased median time from 4618 to
4637 microseconds, about 19 microseconds (0.4%). Holding capture length at
8 and changing workspace capacity from 64 to 256 increased time from 4529
to 4618 microseconds, about 89 microseconds (2.0%). Enlarging both workspace
and capture from 64 to 256 cost about 103 microseconds (2.3%). Thus padding
and capacity do add cost, but do not explain the dominant regression.

The CUDA trace instead showed the full replica `build_plan` kernel taking
about 2491 microseconds per invocation. The old route threshold was 49:
four tokens/rank at EP8/top-k6 means 192 routes and enters this full serial
search, whereas the earlier single-token decode used owner-only planning.
The owner-only path also constructed its descriptors serially.

The following ablation holds workspace/captured capacity at 256 and varies
only the route policy and owner descriptor implementation. Values are the
median of three rank-max GPU P50 measurements, in microseconds:

| Live tokens/rank | Old threshold 49 | Owner-only, old serial builder | Owner-only, parallel builder |
| --- | ---: | ---: | ---: |
| 1 | 1619 | 1624 | 579 |
| 4 | 4638 | 1795 | 724 |
| 8 | 4782 | 1946 | 853 |
| 32 | 3134 | 2116 | 991 |
| 128 | 3122 | 2128 | 1003 |
| 256 | 5552 | 2420 | 1285 |

A second sweep routes every token to the same six experts on one owner.
Before optimizing descriptor construction, full replica planning versus
owner-only took 4492 versus 1947 microseconds at 32 tokens/rank, 4816 versus
3240 at 128, and 5215 versus 4943 at 256. This bounds both balanced and
extremely skewed synthetic cases at the tested capacities; it is not a
universal routing-distribution guarantee.

The final implementation builds owner counts, offsets, q, slots, statistics
and descriptors cooperatively, preserving oracle ordering, 32-row data
padding, 128-row scale padding, overflow, and GPU-conditional fallback.
An independent dispatch kernel sets a device flag to skip serial replica
search. Keeping the cooperative code outside the serial search kernel
avoids increasing the latter's register allocation. Full-search timings
remain approximately 4644 / 3139 / 5558 microseconds for 4 / 32 / 256 live
tokens, consistent with the old timings within about 0.2%.

The vLLM adapter now uses the calibrated owner-only band on H200 through
`min(max_num_batched_tokens, 256)` tokens per rank. At the tested capacity,
the default threshold is 12289 globally valid routes; replica selection is
therefore intentionally inactive for these small batches. This is a policy
optimization, not faster replica balancing. Larger route sets can still
use replica planning, and the threshold can be overridden. Other SM90
devices retain the old default until measured separately.

The final planner passed 77 single-GPU oracle tests, including production
E=384/E=512, R=0/1/4, empty inputs, partial tiles and capacity overflow.
The first cooperative version passed all 18 EP8 replay/selective tests per
rank; the final separate-kernel version passed all five targeted EP8 tests
per rank, covering changing batch sizes, policy transitions, int32/int64
routes, and collective fallback/acceptance. Single-layer replay outputs in
all timing sweeps matched the corresponding eager path bit-for-bit.

The profiler scripts, full sweep results, traces, pre-change source backups
and regression logs are saved under
`/tmp/megaflux-results/capacity-opt-20261024/` on the serving host, with a
local snapshot under `/tmp/megaflux-capacity-evidence/capacity-opt-20261024/`.

### Optimized same-load A/B/A serving validation (2026-10-24)

After the controlled profiling above, the final implementation was frozen
and tested in a new native A1 -> optimized MegaFlux B -> native A2 sequence.
All serving and client parameters match the expanded experiment: eight H200,
vLLM 0.32.0, TP=1/DP=8/EP=8, eager, 8192 context, 256 batch tokens, 8 sequences
per DP rank, utilization 0.85, 16 GiB static NVSHMEM heap, 64 output tokens,
four warmup requests, fixed input lengths, and seeds 42/43/44. Each stage
ran all three workloads for three rounds. There were no mid-run kernel or
policy changes; source and installed-adapter hashes were checked afterward.

Throughput is total output tokens / total measured duration across rounds:

| Input / concurrency | Native A1 tokens/s | Optimized B tokens/s | Native A2 tokens/s | B vs A1 | B vs A2 | A2/A1 drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1k-c8 | 36.09 | 46.75 | 36.05 | +29.54% | +29.71% | -0.13% |
| 4k-c16 | 45.18 | 53.64 | 45.34 | +18.72% | +18.32% | +0.34% |
| 7k-c32 | 57.58 | 74.56 | 59.08 | +29.48% | +26.19% | +2.60% |

Per-round throughput, preserving all observed variation:

| Input / concurrency | A1 rounds 1, 2, 3 | B rounds 1, 2, 3 | A2 rounds 1, 2, 3 |
| --- | --- | --- | --- |
| 1k-c8 | 36.00, 35.74, 36.54 | 47.57, 47.61, 45.17 | 35.78, 36.21, 36.15 |
| 4k-c16 | 43.64, 47.27, 44.80 | 51.91, 56.50, 52.72 | 46.44, 41.25, 49.02 |
| 7k-c32 | 57.96, 61.02, 54.17 | 74.45, 74.46, 74.76 | 57.34, 58.46, 61.61 |

P95 values pool individual requests across the three rounds:

| Input / concurrency | P95 TTFT seconds A1 / B / A2 | P95 TPOT ms A1 / B / A2 | P95 end-to-end seconds A1 / B / A2 |
| --- | --- | --- | --- |
| 1k-c8 | 2.79 / 2.59 / 3.19 | 198.59 / 151.50 / 195.82 | 14.88 / 11.78 / 15.19 |
| 4k-c16 | 13.73 / 11.24 / 14.09 | 222.09 / 174.74 / 220.53 | 27.16 / 21.04 / 26.68 |
| 7k-c32 | 31.36 / 24.41 / 30.33 | 228.67 / 182.71 / 234.47 | 45.28 / 35.38 / 44.13 |

B exceeded both native aggregates in every workload, and every individual B
round exceeded every native round for that workload. The 18.32%–29.71%
throughput gains are larger than the observed native aggregate drift
(-0.13%, +0.34%, +2.60%). P95 TPOT decreased about 20%–24%; the 7K case
also reduced P95 TTFT from 30.33–31.36 seconds to 24.41 seconds. These are
measured gains for this configuration, not statistical confidence intervals
or guarantees for arbitrary traffic/hardware.

All 1080 formal requests completed with zero failures: 5,529,600 input tokens
and 69,120 output tokens. All recorded lengths matched the protocol. Each
stage passed 13 integration checks (39/39 total). The A/B/A sequence took
49.42 minutes including startup/switching, with 22.11 minutes of measured
benchmark duration. No pre-shutdown runtime traceback, OOM, or CUDA/NCCL
error appeared. DP statistics-order warnings remained (20 / 34 / 28), and
stopping A1 and B produced one API output-handler traceback each after
sampling ended. These shutdown logs are retained, not counted as request
failures or silently removed.

Ten-second sampling found peak GPU memory of 123835 / 123847 / 123837 MiB,
peak temperatures of 49 / 45 / 49 degrees C, and active SM clocks of
1980 MHz in all three stages. The change did not materially increase GPU
memory usage. These bounded synthetic runs do not establish production
model accuracy or long-duration availability.

The main regression was serial planning and unprofitable replica work at
small batches, amplified as concurrency crossed the old route threshold
and long inputs required more prefill steps. The captured capacity remains
256; no graph bucketing was introduced. Replica search itself was preserved,
and B intentionally uses owner-only execution within the calibrated band.

Serving evidence is under `/tmp/megaflux-results/aba-capacity-opt-20261024/`.
The combined profiler/serving archive is
`/tmp/megaflux-results/capacity-opt-full-20261024.tar.gz`; its local copy is
`/tmp/megaflux-capacity-evidence/results-final.tar.gz`. It includes all raw
requests/metrics, source snapshots, traces, protocol, commands, hashes,
validation, telemetry, and reproduction scripts. The native A2 service was
left healthy on port 8000 after the experiment.

### Replica-planning capacity sweep: 512/1024 (2026-10-24)

The next controlled EP8 sweep increased workspace and captured capacity to
512 and 1024, retaining E=384, hidden=5120, intermediate=2304, top-k=6,
R=1, deterministic weights and routes, three rounds of 40 rank-max CUDA-event
samples, and separate CUDA traces. These single-layer processes used a
4 GiB static NVSHMEM heap with NVLS disabled. The policies were forced
owner-only (threshold 1,000,000,000), current threshold 12289, and forced
replica search (threshold 0). Every replay matched its eager result and
the corresponding owner-only output bit-for-bit.

Selected median GPU times, in milliseconds, compare owner-only against the
current policy after the latter really enters replica search:

| Capacity | Routing | Live tokens/rank | Owner-only ms | Replica path ms | Committed replicas | Logical remote weight payload/invocation |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 512 | Balanced | 384 | 1.634 | 5.181 | 1 | 18.80 MB |
| 512 | Balanced | 512 | 1.840 | 6.643 | 4 | 75.20 MB |
| 1024 | Balanced | 512 | 2.101 | 6.893 | 4 | 75.20 MB |
| 1024 | Balanced | 1024 | 3.154 | 7.233 | 4 | 75.20 MB |
| 512 | Hotspot | 384 | 5.657 | 5.720 | 6 | 112.80 MB |
| 512 | Hotspot | 512 | 7.362 | 6.086 | 6 | 112.80 MB |
| 1024 | Hotspot | 512 | 7.626 | 6.339 | 6 | 112.80 MB |
| 1024 | Hotspot | 768 | 11.002 | 7.067 | 6 | 112.80 MB |
| 1024 | Hotspot | 1024 | 14.345 | 7.792 | 6 | 112.80 MB |

Hotspot routes every token to the same six logical experts owned by one
rank. A refined sweep at 384, 400, 416, 432, 448, 480 and 512 tokens/rank
places the crossover near 384–400. Gains near the crossover are small:
at 400 the replica path improves inverse latency by about 2%–3%, while
at 416 it improves by 5%–6%, and at 512 by about 21%. At 1024 hotspot
tokens/rank the measured speedup is about 1.84x (latency decreases about
46%), not an 84% latency decrease. Balanced routes show no benefit through
1024; sparse load differences cannot repay serial planning and weight
transfer. There is no universal token-only crossover independent of skew.

Payload is the exact logical copied expert bytes implied by committed
replica descriptors and the two transport-leg buffer widths, not measured
NVLink wire traffic. Planning being eligible is distinct from committing a
nonzero number of replicas; both are recorded in the raw data.

Raw sweeps and traces are on the serving host in
`/tmp/megaflux-results/replica-scale-20261024/`: `sweep.json`,
`refined/sweep.json`, and the corresponding scripts/logs/traces.

The serving comparison uses two fixed profiles: 4096 input tokens at
concurrency 64 (128 requests/round), and 7168 input tokens at concurrency
128 (256 requests/round). Each stage runs three rounds with seeds 42/43/44,
64 output tokens, ignore-EOS, and four benchmark warmup requests. All stages
first run identical concurrent preconditioning with both profiles, one
request per concurrency slot and eight output tokens. An initial attempt
without concurrent native preconditioning is excluded from the comparison.

For each capacity, A1 and A2 are the native default backend (the logs select
HUMMING), B is MegaFlux with threshold 12289, and C is the same MegaFlux
configuration with threshold 1,000,000,000 to force owner-only planning.
The order is A1, B, C, A2, with three benchmark rounds per stage; these are
not three independent server-restart A/B/A cycles. Common settings are
TP1/DP8/EP8, max-model-len 8192, max-num-seqs 32 per DP rank, enforce-eager,
GPU utilization 0.85 at capacity 512 and 0.90 at capacity 1024, local
MegaFlux graph replay enabled, static 16 GiB
NVSHMEM heap and CUDA VMM disabled. This compares against native eager
serving, not a native whole-model CUDA Graph configuration.

At capacity 1024, the original 0.85 MegaFlux startup reported -3.8 GiB of
available KV-cache memory and failed before serving requests. Its native A1
measurements and failed startup log are retained in `cap1024-u085/` and
excluded from the final comparison. All four 1024 stages are rerun at 0.90;
no 0.85 native result is paired with a 0.90 MegaFlux result. The 512/1024
serving results therefore do not isolate capacity alone: the memory budget
also differs. The fixed single-layer sweeps isolate routing/capacity effects.

An installed-adapter diagnostic tap reads the first MoE layer's committed
plan on each rank only during separate B/C preconditioning. Its enable file
is removed before formal timing; no plan metadata is copied to CPU during
measured requests. These samples establish actual nonzero replica commits
and distinguish them from planner eligibility. They are not all-layer or
formal-request replica measurements.

Completed serving results (output tokens/second, total output divided by
summed duration across three rounds):

| Capacity | Input/concurrency | Native A1 | MegaFlux replicas B | MegaFlux owner-only C | Native A2 | B vs C |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 4096/64 | 180.4 | 221.1 | 235.5 | 185.2 | -6.1% |
| 512 | 7168/128 | 152.1 | 178.0 | 189.0 | 153.0 | -5.8% |
| 1024 | 4096/64 | 237.1 | 258.7 | 293.8 | 240.1 | -11.9% |
| 1024 | 7168/128 | 273.3 | 266.7 | 334.8 | 268.3 | -20.4% |

B improves over both native brackets by 19.3%–22.5% and 16.4%–17.1%
at capacity 512, and 7.8%–9.1% for 1024/4096/64. At 1024/7168/128,
B is 0.6%–2.4% slower than native, comparable to the -1.85% native bracket
drift; this does not establish a reliable improvement. Owner-only C is
22.4%–30.5% faster than both native brackets across these four profiles.
These comparisons identify the best measured configuration, not an isolated
causal estimate of every planner cost: changing routing also changes
scheduling and subsequent batch composition.

At 1024/7168/128, pooled P95 end-to-end latency is 34.66/37.64/28.12/34.57
seconds for A1/B/C/A2; P95 TTFT is 21.29/23.01/17.26/21.11 seconds.
B's P95 TPOT is 246.8 ms, versus C's 183.3 ms and native 215.1/218.3 ms.
Thus the replica penalty is visible in request latency as well as throughput.
All round-level throughput, pooled latency metrics and comparison arithmetic
are retained in `comparison.json` and the individual benchmark JSON files.

The untimed first-layer rank-0 diagnostics contain nonzero committed replicas
in 280/371 B samples at 512 and 156/219 at 1024, with at most seven replicas
and 131,604,536 logical remote weight bytes per sampled invocation. C has
zero replicas; all B/C samples have zero fallback. Global metadata is counted
once per epoch, not summed over the eight redundant rank observations.
The enable files are absent and no probe samples fall inside formal timing
windows. This proves that the served B configuration actually uses replicas
during diagnosis, without pretending to observe its formal-request frequency.

The 512 owner-only diagnostic's eligible samples have owner load peak/mean
median 1.82 and P95 1.87, much less severe than the artificial one-owner
hotspot. The 1024 owner-only diagnostic has no sample above 12288 routes
(observed maximum 9270); it cannot establish large-batch owner skew.
Diagnostic CPU reads perturb scheduling and B/C do not have identical
observed batch distributions. Together with the controlled sweeps, the
results support an imbalance-dependent admission policy rather than simply
lowering a route-count threshold. No replica benefit crossover was observed
for this real-model workload through the tested capacities/concurrencies.

A fixed-live-token owner-only comparison raises workspace/capture capacity
from 512 to 1024 and adds about 0.25–0.28 ms: balanced 16 tokens/rank rises
from 1.081 to 1.336 ms (+23.6%), and balanced 512 rises from 1.840 to
2.101 ms (+14.2%). This measures the combined workspace/capture effect,
not quantization padding alone. Larger capacity can improve serving batch
packing despite this per-layer overhead. Its memory cost is substantial:
MegaFlux's available KV cache falls from 1,104,484 tokens/rank at 512/0.85
to 391,756 at 1024/0.90 (47.82 maximum 8192-token requests/rank, versus
configured max-num-seqs 32). Cross-capacity serving gains also include the
changed memory budget and must not be attributed solely to token capacity.

The final accepted campaign contains 48 formal cases, 9216 successful
requests, zero failed requests, 589824 output tokens and 104 passing smoke
checks. This is not an all-attempts stability guarantee: an earlier 512 C
attempt crashed in native CUDA Graph replay after 65 successes and 63
failures. Its root cause remains unproven. The original full service log was
overwritten during retry; saved excerpts, failed benchmark data and this
limitation are retained under `cap512/excluded-C-attempt1/`. The separate
1024/0.85 startup failure is also retained. Successful retries do not establish
that the earlier crash is fixed. Accepted service logs also contain scheduler
statistics ordering warnings; no preemption is recorded in those logs.

`validation.json` audits request counts, shapes, smoke checks, source hashes,
probe timing and runtime logs. `source-snapshots/` and `snapshot-sha256.json`
preserve the exact measured sources including the temporary adapter tap.
The tap is restored to its pre-probe source after validation and snapshotting;
`probe-restoration.json` records the verified hashes. The complete evidence
archive is `/tmp/megaflux-results/replica-scale-20261024.tar.gz` on the host.

### Skew and transport-aware replica admission (2026-10-24)

The vLLM adapter enables an optional calibrated admission policy only for
H200, EP8, E384, hidden5120, intermediate2304 and top-k6. Set
`FLASHINFER_MOE_EP_MEGAFLUX_ADMISSION=0` to retain the previous route-threshold
policy. Other shapes/devices retain their existing behavior. The low-level
planner and backend default remains uncalibrated (`replica_admission=None`).

The policy stays entirely on device and uses padded owner block counts.
Before serial search, require peak/mean >=3 and discounted ideal block savings
(reserving 1/P of potential savings) to exceed fixed cost, the full P*R copy
budget and a safety margin. This is a conservative opportunity filter, not
an optimality bound. After search/compaction, require actual critical-path
block savings to exceed fixed cost plus the actual committed replica count's
transport cost and margin. Equality rejects. Rejection emits exact owner-only
records before any expert weight transfer. Configuration agreement includes
all admission parameters to prevent inconsistent plans between ranks.

The initial H200 coefficients are 8 us/block, 3300 us fixed, 300 us margin,
and approximately 100 us/copy. The last term is 18,800,648 logical expert
bytes divided by an effective 188 GB/s coefficient; it is a fitted accounting
term covering both transport legs, not a measured NVLink bandwidth claim.
The coefficients come from the preceding controlled sweeps, not a universal
hardware model. A first trial charged only one optimistic copy during the
preflight: 384–416-token hotspots then paid search costs before rejecting and
regressed about 3 ms. That trial is retained separately; the conservative
preflight addresses it before freezing serving measurements.

After tightening preflight, the final controlled sweep gives the following
median-of-three-round single-layer CUDA times. All eager, replay and
cross-policy outputs agree bit-for-bit. Values are milliseconds:

| Capacity | Routing/live tokens per rank | Owner-only | Previous policy | Admission | Admission replicas |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | balanced/384 | 1.639 | 5.185 | 1.645 | 0 |
| 512 | balanced/512 | 1.843 | 6.650 | 1.847 | 0 |
| 512 | hotspot/384 | 5.747 | 5.738 | 5.767 | 0 |
| 512 | hotspot/400 | 5.955 | 5.785 | 5.976 | 0 |
| 512 | hotspot/416 | 6.169 | 5.836 | 6.190 | 0 |
| 512 | hotspot/512 | 7.458 | 6.110 | 6.380 | 6 |
| 1024 | balanced/1024 | 3.149 | 7.255 | 3.145 | 0 |
| 1024 | hotspot/512 | 7.612 | 6.349 | 6.617 | 6 |
| 1024 | hotspot/1024 | 14.289 | 7.832 | 8.097 | 6 |

The guard deliberately skips the small gains at 400–416 rather than paying
search before discovering insufficient margin. Accepted hotspot candidates
pay about 0.27 ms additional final accounting in this implementation, but
still outperform owner-only by about 14% at 512 and 43% at 1024 hotspot
live tokens/rank. The complete final sweep is `sweep/sweep.json`; the first
optimistic-preflight trial is preserved under `initial-sweep/`.

The regression checks exposed a separate pre-existing `NameError` in the
non-selective pipeline's full-root copy path (`size` was undefined). The
copy size and per-owner destination offset now use the already validated
`source_bytes`. An uncaptured-log reproduction establishes that cause;
it does not explain the earlier serving campaign's native CUDA Graph crash.
After repair, EP8 tests pass all 14 cases per rank, including the affected
non-selective Graph path, active-admission selective transport Graph replay,
dynamic GPU admission, equality rejection, copy-cost sensitivity and
calibration scope. The serving-path selective Graph suite separately passes
all five budget/fallback cases with 128 changing-input replays each.

The new serving campaign uses three independently restarted A1/B/A2 brackets
per capacity (18 services, 36 formal cases), rather than three rounds inside
one server bracket. Each bracket uses a matched seed 42/43/44 and identical
preconditioning across backends. The workloads and memory budgets remain
4096/64 and 7168/128, capacity512/0.85 and capacity1024/0.90. Raw commands,
logs, diagnostics, source snapshots and results are retained at
`/tmp/megaflux-results/admission-20261024/`. Prior route-threshold and
owner-only serving results are historical references, not fresh controls.

This admission campaign stopped on the second 512-capacity B service: the
7168/128 workload reproduced a host SIGSEGV in `cuGraphLaunch` /
`cudaGraphLaunch` / `CUDAGraph::replay`. The failed benchmark reported 250
completed and six failed requests, but 43 completed responses were also
truncated. Its throughput is therefore invalid. The complete service log,
failed results, 43 frozen source hashes and failure audit are preserved;
no 1024-capacity bracket completed in this campaign. A passing single bracket
or the separate kernel tests do not establish serving stability.

Follow-up diagnostics with `NCCL_GRAPH_MIXING_SUPPORT=0` completed three
distinct-seed profile pairs (1152 requests), with zero failures and exactly
64 output tokens per response. These runs used native debuggers and are
diagnostics only, not formal performance comparisons or proof of a fix.
All 43 frozen compute source hashes remained unchanged. The matched
graph-mixing-enabled control subsequently could not initialize: each GPU
had approximately 132 GiB allocated, without a corresponding live process
visible inside the container. This resource failure provides no evidence
about Graph stability. Its startup log is retained separately under
`graph-debug/ncclmix1/`; the successful disabled diagnostics and their audit
are under `graph-debug/ncclmix0/` and `graph-debug/ncclmix0-audit.json`.

The root cause remains open. NCCL documents limitations around mixed eager
and captured communication, but its deadlock guidance does not by itself
explain this host segmentation fault. Complete the controlled stability
investigation before accepting a mitigation, then rerun all three A/B/A
brackets at both capacities with matched runtime settings. Do not combine
these diagnostic results with the incomplete formal campaign.

After resources became available, fresh matched controls under
`/tmp/megaflux-results/admission-20261025/` reproduced failures with both
graph-mixing settings. With mixing enabled, native GDB caught a host null
pointer dereference inside `libcuda.so.1` during `cuGraphLaunch` (driver
590.44.01, ELF offset `0x27b4ba`). The third 7168/128 case had only 208 full
64-token responses out of 256 requests. With mixing disabled, the second
7168/128 case reported asynchronous `cudaErrorLaunchFailure`, followed by a
worker abort; only 217 responses were complete. Neither failed case provides
valid throughput. Disabling graph mixing is therefore not a validated fix.

A separate CPU-side communication audit did not observe steady-state eager
calls on the captured EP process group through the instrumented Torch
distributed entry points. Eager DP communication and barriers used other
groups. This is bounded diagnostic evidence, not an exhaustive concurrency
proof or an explanation of the device failure.

The expanded standalone stress used 40 independent layers, weights,
workspaces and Graphs on each of eight ranks. It completed 2000 changing-route
steps, including hotspot, balanced, shrinking and zero-token-rank inputs:
640,000 total Graph launches without a reported CUDA error. This test has no
independent functional output oracle and excludes the rest of the model.
An initial diagnostic setup retained redundant raw weight packs and ran out
of memory; that attempt is preserved separately and excluded from stability
results. Releasing source packs follows the documented layer ownership
contract.

Full-model `CUDA_LAUNCH_BLOCKING=1` diagnostics subsequently completed three
distinct-seed profile pairs: 1152 requests, zero failures and exactly 64
output tokens per response, with all 43 compute source hashes unchanged.
Blocking changes execution overlap and performance; this pass is a timing
clue, not a repair or a performance result. Whole-model memcheck and a
targeted fix remain required before starting a fresh formal A/B/A campaign.

Whole-model Compute Sanitizer attempts confirmed injection into all eight GPU
workers, but the unlimited-reporting run exhausted GPU memory while loading a
Humming kernel module during initial correctness checks. It reported no precise
device-memory violation before that resource failure. A subsequent diagnostic
reduced the KV-cache budget (`gpu-memory-utilization=0.82`, capacity still 512);
the sanitizer frontend itself then exited with SIGSEGV before GPU initialization.
These attempts are incomplete diagnostics, not reproductions of the original
Graph crash or clean memcheck results. Tool startup and instrumentation memory
requirements must be resolved before interpreting subsequent checks.

A separately unpacked CUDA 13.1 sanitizer (2025.4.1, build 37093031) subsequently
reached serving startup with the reduced KV-cache budget, but exhausted its own
device memory during the first correctness requests. It explicitly reported
untracked launches across both MegaFlux and other model kernels. That run has
invalid checking coverage and must not be reported as a clean memcheck pass.
The next diagnostic uses `--force-synchronization-limit 128` to bound tool
memory use. This changes execution timing and cannot validate the original
asynchronous serving mode, even if memory checking completes successfully.
The synchronization-limited run returned correct arithmetic and capital answers
without a device-memory report, but each two-token request took several minutes.
Its original full diagnostic schedule was stopped after those two checks; the
same instrumented service is instead used for bounded 4096/7168-token probes
with eight concurrent requests and eight output tokens. These probes are
explicitly separate from the required 512/1024-capacity, 64/128-concurrency
formal campaign and do not establish its stability.

The bounded run completed all eight 4096-token requests with exactly eight output
tokens each, without an untracked-launch or precise device-memory report. The
7168-token case was interrupted by `RPC call to sample_tokens timed out`; none
of its eight requests completed. No precise device-memory fault was observed
before that timeout. Preserve this as partial diagnostic evidence: the original
Graph failure was not explained, and neither a full memcheck pass nor a stability
fix was established.

A temporary Graph-lifetime control retained each raw Graph
(`keep_graph=True`) and explicitly instantiated it. All eight workers captured
40 layers each, but rank 7 then failed the KV-cache capacity check before any
HTTP validation. Rank 0 still reported available cache memory. This startup
failure does not establish either the original crash or a Graph-retention
memory regression; the lifetime hypothesis remains unvalidated.

After the owned experiment processes exited, GPU 7 retained approximately
16 GiB with no container-visible compute process or device-file holder. A
separate minimal driver-context lifecycle (16-byte allocation, successful free
and context destruction, process exit) was followed by another approximately
16 GiB increase in reported usage. This observation does not distinguish a
driver resource issue from a holder outside the container. Further context
probes and full-model starts were stopped; no GPU reset was performed.
The next lifetime control is prepared with CPU-only per-rank memory-budget
logging and a guard requiring at least 130000 MiB free on every GPU, but has
not started. The 43 frozen compute-source hashes remain unchanged. A host-side
ownership check and resource recovery are required before continuing the
eight-GPU stability tests and the formal A/B/A campaign.

After resources recovered, the retained-template control completed 1152
full-output requests and 13 correctness checks. Reverting to the original
Graph construction reproduced a host `cuGraphLaunch` segfault during the
second 4096/c64 case. Both controls reported identical per-rank memory budgets.
However, applying the retained-template change to the actual source and removing
all diagnostic hooks still reproduced the same segfault in the third 7168/c128
case. The candidate was reverted: its eight-rank numerical regression pass and
earlier serving pass do not establish a repair. The validation chain stopped
before capacity 1024, and no formal stability gate or new A/B/A campaign was
created. A separate control now tests NCCL graph buffer registration with
`NCCL_GRAPH_REGISTER=0`, mixing still enabled and the original Graph construction;
this control also reproduced the same host segfault in its first 7168/c128
case and is not a repair. The next diagnostic disables async scheduling while
explicitly retaining Gloo DP synchronization, the original Graph construction
and default NCCL graph registration. This removes both two-batch scheduler
overlap and asynchronous output handling; it does not yet isolate those two
components and is not a validated serving configuration.

The synchronous-scheduler control also reproduced the same host segfault in
its first 7168/c128 case. Removing the periodic 20-step synchronization from
the 40-independent-layer stress still completed 2000 steps (640,000 global
Graph launches), including the final device synchronization and a zero launcher
exit status. That diagnostic excludes the rest of the model and has no
independent numerical oracle.

A subsequent per-Graph completion-event control passed five seed pairs and
the sixth 4096/c64 case, then segfaulted in the sixth 7168/c128 case. Its
periodic records covered 1,843,200 global replay calls with zero waits for a
previous invocation. This does not support overlapping reuse of the same
Graph in the observed calls, and neither the event guard nor async-scheduler
disablement is a repair. No new formal A/B/A campaign has started.
