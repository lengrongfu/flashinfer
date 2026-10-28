# SM90 MegaFlux stability, accuracy and long-context campaign

Status: GPU validation, both full-length serving backends and the independent
restart control passed. MegaFlux failed the 128-concurrency 32K-input control.
The matched native high-concurrency control passed. All five attempts are
complete; high-concurrency MegaFlux stability remains unqualified.

## Configuration and scope

Tests run through `ssh p2p-test-lrf` in `/root/flashinfer`, on eight H200 GPUs
with DeepSeek-V4.1-Flash-1017 and vLLM 0.32.0. Production kernels are unchanged
from the preceding NVLink metadata experiment. Both serving backends use
TP1/DP8/EP8, eager model execution, maximum batched tokens 512, maximum sequences
32, memory fraction 0.85 and a 65536-token model context. MegaFlux retains its
local MoE Graphs, NVLink metadata, 12289-route threshold and existing admission.
The 37 captured source files match between native A and MegaFlux B. The startup
log reports 71.48 GiB available for KV cache with native and 8.33 GiB with
MegaFlux B under the same memory fraction. Thus this implementation consumes
substantially more non-KV memory; its throughput result does not establish an
equivalent maximum serving capacity. Both services report the same Engram
shared-memory shortage and shard the offloaded tables across DP replicas.

The performance cases are 16384 input + 16384 output and 32768 input + 32768
output, each with eight requests at concurrency eight. EOS is ignored, and
every actual input/output length is audited. A separate 4096-input/128-output
prewarm precedes measurement; the benchmark's full-length readiness request and
full-length warmups are disabled. These concurrency settings differ from the
earlier 64/128-concurrency experiment: compare the backends within this campaign.
The performance inputs are vLLM's synthetic random-token workload with seed 42,
not the GSM8K or retrieval prompts. The installed vLLM completion client records
input and output lengths from the response usage. Synthetic throughput does not
establish the same speedup on natural long-form reasoning traffic.
Input and output lengths grow together, so these pairs do not isolate context
length as a single variable. Each main case has eight requests in one service
lifetime per backend; tail latency and run-to-run variability need larger
samples before treating these measurements as a production SLO estimate.

The initial service sequence was native long-output, MegaFlux long-output, then a fresh
MegaFlux restart control with 512 outputs at the same long input lengths. Every
attempt gets its own directory, source hashes, commands, logs, telemetry and
request records. The experiment directory is
`/tmp/megaflux-results/long-context-20261027` in the container.
An additional fresh MegaFlux service ran the long-input/512-output cases
at concurrency 128. After its 32K-input failure, a matched fresh native service
ran the same high-concurrency sequence as an attribution control.

## Long-output serving results

All four formal cases completed eight requests with exact input/output lengths
and zero failures or truncations: 32 requests and 786432 output tokens total.
The offline audit independently reproduces token totals, throughput and all
reported P50/P95/P99 TTFT, TPOT and end-to-end latency percentiles.

| Input / output tokens | Native output tokens/s | MegaFlux output tokens/s | Change |
| --- | ---: | ---: | ---: |
| 16384 / 16384 | 41.6368 | 55.5761 | +33.48% |
| 32768 / 32768 | 41.5286 | 55.5451 | +33.75% |

| Input / output tokens | Native P95 TTFT (s) | MegaFlux P95 TTFT (s) | Native P95 TPOT (ms) | MegaFlux P95 TPOT (ms) |
| --- | ---: | ---: | ---: | ---: |
| 16384 / 16384 | 11.6935 | 14.9038 | 191.359 | 143.055 |
| 32768 / 32768 | 22.7488 | 22.6584 | 191.881 | 143.306 |

The 16K case took 3147.986 seconds with native and 2358.424 seconds with
MegaFlux (52.47 versus 39.31 minutes). The 32K case took 6312.377 versus
4719.477 seconds (105.21 versus 78.66 minutes). P95 TPOT improved by 25.24%
and 25.32% respectively. P95 TTFT regressed by 27.45% at 16K and was nearly
unchanged at 32K (-0.40%). This is a decode-throughput gain under the recorded
eager serving configuration, with a first-token latency tradeoff. The protocol
includes short prewarming but does not isolate first-use compilation,
scheduling and pure prefill-kernel costs within TTFT. It does not separately
attribute gains to replication, local Graph replay and NVLink metadata.

Native pre-workload checks passed 13/13 semantic tests and 6/6 retrieval cases.
GSM8K was 60/64; all four incorrect cases ended normally, without truncation.
All 53756 chosen-token probabilities in its four probes are finite. The native
post-workload semantic check also passed 13/13. Native before/after top-1
agreement is 98.63%, 99.07%, 98.99% and 99.14% at 512, 4096, 16384 and 32768
tokens respectively. Mean absolute chosen-logprob differences are 0.0465,
0.0339, 0.0328 and 0.0291. Thus the native service itself is not bit-invariant
across these observations; its cause is not established by this test.
Cross-backend differences must be interpreted alongside this control.
Before/after probes have identical requests but do not enforce identical
scheduler batches or co-running requests. Their deltas alone cannot separate
batch/load effects from changes caused by elapsed service time.

MegaFlux's pre-workload semantic checks passed 13/13. Its GSM8K result is
61/64, versus native 60/64, with 63 identical extracted answers, no native-only
correct answer and one MegaFlux-only correct answer. Both backends passed all
six retrieval cases. The subset shows no observed answer regression, but a
one-question difference does not establish an accuracy improvement.

| Probe tokens | Native before/after top-1 agreement | Native vs MegaFlux top-1 agreement | Native before/after mean absolute logprob delta | Native vs MegaFlux mean absolute logprob delta |
| --- | ---: | ---: | ---: | ---: |
| 512 | 98.63% | 98.63% | 0.04651 | 0.05356 |
| 4096 | 99.07% | 98.85% | 0.03390 | 0.03374 |
| 16384 | 98.99% | 98.74% | 0.03284 | 0.03965 |
| 32768 | 99.14% | 99.09% | 0.02911 | 0.03004 |

All compared probabilities are finite and the prompt token IDs match exactly.
The differences are of a similar scale to the native within-service control;
this is descriptive evidence, not a statistical equivalence guarantee.
Per-position tails, including maximum deviations, are retained in the JSON.
Across the four probes, the largest chosen-logprob difference is 6.9253 for
native versus MegaFlux B, compared with 6.2150 for native before versus after.
Small mean differences therefore do not imply uniformly small errors or
bitwise model equivalence. Top-1 comparisons use the rank returned by the
service, including its handling of tied probability values.

MegaFlux B's post-workload semantic check passed 13/13, with all four logprob
observations finite. Its before/after top-1 agreement at 512/4096/16384/32768
tokens is 98.83%/98.88%/98.78%/99.12%; mean absolute chosen-logprob deltas are
0.04684/0.03752/0.04109/0.02991, with a maximum of 6.3634. These observations
also show numerical variation rather than bitwise model equivalence.

Both long-output lifetimes completed their source guards, and the 37 source
hashes agree across them. Neither has runtime `ERROR` lines or out-of-order
warnings before shutdown. Their peak sampled GPU memory was 124026 MiB
(native) and 124029 MiB (MegaFlux); maximum temperature was 43°C for both.
All eight GPUs had zero corrected/uncorrected volatile ECC errors at the native
checkpoint and the MegaFlux 32K one-hour checkpoint. Restart and high-load
controls are separate from these long-output counts.

## Restart and high-concurrency controls

Fresh MegaFlux service C completed both long-input/512-output workloads:

| Input / output tokens | Concurrency | Completed | Output tokens/s | P95 TTFT (s) | P95 TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16384 / 512 | 8 | 8/8 | 46.3177 | 14.6700 | 144.900 |
| 32768 / 512 | 8 | 8/8 | 39.4276 | 27.8016 | 148.645 |

All lengths were exact, with zero failed requests and 8192 total output tokens.
Before/after semantic checks passed 13/13 each. GSM8K was 61/64 and retrieval
6/6; all 64 extracted GSM8K answers match MegaFlux B. Source hashes match A/B,
and snapshot verification passed. Peak sampled GPU memory was 124035 MiB,
maximum temperature 43°C, and available KV memory at startup was 8.33 GiB.
There were no runtime `ERROR` or out-of-order lines. During explicit shutdown,
an `AsyncLLM` / `EngineDeadError` traceback was logged, as in native A.

All C probability probes were finite. Across the four lengths, C before/after
top-1 agreement ranges from 98.63% to 99.06%, and mean absolute logprob deltas
from 0.03000 to 0.05738. Its maximum logprob delta is 7.9459. B versus C top-1
agreement ranges from 98.43% to 99.09%; exact answer agreement therefore does
not imply exact token-probability agreement. Full per-length values are retained.

Fresh service D passed the concurrency-128 16K-input/512-output case: 128/128
exact-length requests, zero failures, 65536 output tokens in 163.089 seconds,
401.8414 tokens/s, P95 TTFT 85.3664 seconds and P95 TPOT 170.562 ms.
Its pre-workload accuracy was again 61/64 GSM8K and 6/6 retrieval, with the same
three incorrect math answers as B/C.

**D failed the concurrency-128 32K-input/512-output case.** The client marked
123 requests completed and five failed, but only 50/128 recorded output lengths
were 512. Most interrupted streams had partial output; two records reported
33022 and 32958 output tokens despite only 253 and 188 inter-token samples,
respectively. These counts and the raw 667.7 tokens/s summary are invalid for
performance comparison. The runner rejected the workload and did not execute
its post-workload probability/semantic probes or its final lifetime source guard.

The first explicit worker exception was on DP5/EP5 at 14:51:29 UTC:
`cudaErrorLaunchFailure (719)`, reported from the DeepSeek attention indexer's
`fused_indexer_q_rope_quant` CuTe-DSL executor. CUDA errors may surface after
the originating operation; this traceback does not prove the indexer or MegaFlux
is the root cause. It occurred before shutdown and is a genuine runtime failure.
All failure logs and partial responses are retained under
`megaflux-highconcurrency-D`. The service was stopped by the runner, and no GPU
compute processes remained before the next control started.

Fresh native service E completed the same `--short --concurrency 128` sequence:

| Input / output tokens | Completed | Output tokens/s | P95 TTFT (s) | P95 TPOT (ms) |
| --- | ---: | ---: | ---: | ---: |
| 16384 / 512 | 128/128 | 318.2160 | 106.3767 | 213.797 |
| 32768 / 512 | 128/128 | 203.8079 | 205.6985 | 216.715 |

Both had exact lengths and zero failures. E's pre/post semantic checks passed
13/13 each, with finite probability probes and no runtime `ERROR` or
out-of-order lines. Its GSM8K result was 60/64 and retrieval 6/6; all 64 math
answers match native A. The startup log confirms `HummingGroupedExperts`.
E's peak sampled GPU memory was 124085 MiB, maximum temperature 50°C, and
available KV memory 71.48 GiB. D's peak was 124057 MiB and maximum temperature
50°C, with available KV memory 8.33 GiB.

Across three MegaFlux lifetimes, the same fixed GSM8K subset yielded 61/64
with identical extracted answers; both native lifetimes yielded 60/64 with
identical answers. All five passed the same six retrieval cases before load.
These are repeated observations of the same questions, not 320 independent
accuracy questions. D has no post-failure accuracy result. Native A versus E
has a maximum chosen-logprob delta of 7.9984; native A versus D reaches 8.1544.
This further limits any claim of numerical equivalence despite unchanged
extracted answers.

The one-run matched control supports a MegaFlux-path stability concern in
this configuration, but does not identify the originating kernel or prove a
universal failure rate. High-concurrency MegaFlux stability has **not** passed.
The prior crash root cause is not established or claimed fixed. These controls
remain separate from the concurrency-eight long-output performance comparison.

The SSH connections closed during the native 32K stage. The native service
nevertheless completed both workloads, after-probes and source-hash checks,
then reached its cleanup step; the following MegaFlux service had not started.
The remaining B/C/D attempts were resumed using a detached container-side
driver and monitor, independent of the interactive SSH connection. The
completed native attempt was retained without replacement.
More precisely, the native runner returned zero after its explicit SIGTERM
cleanup. Its log contains no `ERROR` lines before shutdown, but contains an
`AsyncLLM output_handler` / `EngineDeadError` traceback during shutdown,
forced child-process termination and resource-tracker warnings about eight
semaphores and sixteen shared-memory objects. These are retained as cleanup
limitations, separately from the successful requests and post-workload probes.

## Completed GPU checks

- Metadata, changing-batch Graph replay and owner-streamed reference: 12 tests
  passed on every EP8 rank. The replay comparisons against original MegaMoE
  are bit-exact, including empty/asymmetric ranks and forced fallback.
- H5120/I2304/E384/K6/T512 full forward: all six variants agree bit-for-bit for
  balanced and hotspot routing. The independent instruction-reference relative
  L2 error is 0.0016593 and 0.0016598 respectively, within the established
  MXFP4/FP8 numerical contract. This is not a BF16-weight accuracy claim.
- Forty independently weighted/captured layers, 2000 steps per rank, EP8:
  640000 layer Graph replays completed. Inputs alternate hotspot, shrinking,
  asymmetric and balanced routes with NCCL interleaved between rounds. There
  is no periodic host synchronization. All 100 snapshots per rank of the first
  four rows of the final layer's hotspot output are finite. Sampling every
  twentieth step selects only the hotspot mode. This stress checks execution
  and sampled finite outputs; it
  does not independently verify every one of the 640000 layer outputs.
- The stress process emitted a process-group destruction warning at exit.
  Its successful exit does not imply warning-free cleanup; subsequent service
  launch checks that GPUs are idle before starting.

## Model accuracy protocol

The fixed sample contains 64 questions drawn with seed 20261027 from the
[official GSM8K test set](https://github.com/openai/grade-school-math).
The downloaded test JSONL has 1319 rows and SHA256
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
Prompts are zero-shot, non-thinking, temperature zero, with up to 2048 output
tokens and a required `####` numeric final answer. Missing final answers,
incorrect numbers and truncated responses count as incorrect. This fixed
subset is not a full GSM8K qualification or an external leaderboard score.

Six retrieval cases place a passcode at 5%, 50% and 95% of approximately 16K
and 32K input contexts. The actual token counts are retained in the response
usage. Four teacher-forced probes use identical token IDs at lengths 512, 4096,
16384 and 32768. Saved chosen-token log probabilities support mean/p99/max
absolute differences, mean negative log likelihood and top-1 agreement;
they do not support a full-vocabulary KL divergence. Logprob and semantic
checks repeat after the long workload to measure within-service drift.

Forced 16K/32K generation tests duration, output length and performance. It
does not establish the semantic quality of those forced continuations.

## Reproduction

The runner refuses to overwrite an attempt or start alongside active GPU
processes. Use a fresh attempt name on the idle eight-GPU host:

```bash
cd /root/flashinfer
python3 examples/deepseek_v41/run_megaflux_long_context.py \
  --root /tmp/megaflux-results/long-context-20261027 \
  --name native-new --backend native
python3 examples/deepseek_v41/run_megaflux_long_context.py \
  --root /tmp/megaflux-results/long-context-20261027 \
  --name megaflux-new --backend nvlink
python3 examples/deepseek_v41/run_megaflux_long_context.py \
  --root /tmp/megaflux-results/long-context-20261027 \
  --name megaflux-restart-new --backend nvlink --short
python3 examples/deepseek_v41/run_megaflux_long_context.py \
  --root /tmp/megaflux-results/long-context-20261027 \
  --name megaflux-c128-new --backend nvlink --short --concurrency 128
python3 examples/deepseek_v41/run_megaflux_long_context.py \
  --root /tmp/megaflux-results/long-context-20261027 \
  --name native-c128-new --backend native --short --concurrency 128
```

The root must contain the recorded GSM8K test JSONL and frozen accuracy
fixtures. `measure_megaflux_accuracy.py` creates those fixtures from the test
JSONL if absent. `compare_megaflux_accuracy.py LEFT RIGHT --output result.json`
compares the saved `accuracy-before` or `accuracy-after` directories.
`audit_megaflux_long_context.py ATTEMPT... --output audited-summary.json`
independently checks saved request lengths, token totals, throughput and
P50/P95/P99 latency arithmetic, and verifies source snapshots against their
recorded hashes. The raw benchmark JSON retains inter-token samples and text;
the compact audit retains per-request lengths and first-token/end-to-end times.

## Evidence and validation status

[Compact evidence](megaflux-results/long-context-20261027/README.md) includes
commands, environment, exact-length audits, paired accuracy comparisons,
source hashes and compressed full logs. The complete raw archive is retained
at `.fi_workspace/megaflux-results/long-context-20261027.tar.gz` in both the
local and remote repositories. Its locations and SHA256 are recorded in the
compact manifest; it includes all raw responses, inter-token samples and source
snapshots, including the failed attempt.

All five startup source manifests contain the same 37 hashes. A/B/C/E completed
their lifetime guards; D skipped its guard on failure. A separate post-campaign
check verifies that the current selected files still match all five startup
manifests. This does not convert D's failed workload into a passing result.
The final arithmetic audit intentionally exits nonzero for D's incomplete
32K workload and marks its throughput invalid; every other saved workload
passes. Diagnostic arrays in the compact audit are capped excerpts with full
line counts, while the complete logs remain available.

The test helpers and report pass the repository's focused pre-commit checks,
including Ruff lint and formatting. No production kernels were changed
in this validation campaign. All owned serving processes have stopped; final
GPU usage and ECC observations are retained in `post-run-checks.json`.
