# SM90 MegaFlux NVLink metadata exchange

This is an opt-in extension to the existing SM90 MXFP4 MegaFlux forward, not a
new implementation of its planner or weight pipeline. The existing implementation
adapts [MegaFlux](https://arxiv.org/html/2610.00671) to Hopper's FP8 WGMMA and Humming
MXFP4 representation; its serial bounded planner differs from the paper's four
parallel warp searches. Forward, complete-leg W1/W2 readiness, dynamic routing,
CUDA Graph replay and the vLLM adapter were already present before this change.

Set `metadata_transport="nvlink"` in `Sm90_Mxfp4_MegaFluxGraphConfig` (or its
online/pipeline base configs). The default remains `"nccl"`. After applying
`examples/deepseek_v41/patch_vllm_megaflux_sm90.py`, select the new path in vLLM with
`FLASHINFER_MOE_EP_MEGAFLUX_METADATA=nvlink`. All EP ranks must select the same
transport; the collective manifest includes this setting.

The transport requires one NVSHMEM EP group, matching rank order and direct
mapping of every peer. Initialization checks this collectively and fails clearly
on unsupported topology. Only addresses returned by `nvshmem_ptr` for a tracked
symmetric arena are used remotely. Allocation, compilation and capability checks
precede Graph capture. There is no host Plan readback during forward.

Each invocation performs four stream-ordered NVSHMEM barriers: retire the prior
invocation's readers, publish counts/status, publish fallback/hash votes, and
publish routes. These sections are disjoint within the arena. The next invocation
cannot overwrite them until all preceding reads retire. The route kernel gathers
directly into `[slot, source, route]` order. NCCL is still used for setup; its five
steady-state metadata collectives are replaced on this explicit path. The memory
ordering relies on the [NVSHMEM barrier contract](https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html)
and serialized use of the shared team on the caller's stream.

## GPU validation on 2026-10-27

Environment: `ssh p2p-test-lrf`, `/root/flashinfer`, eight NVIDIA H200 GPUs,
PyTorch 2.13.0+cu130, vLLM 0.32.0. Tests use the existing isolated NVSHMEM Python
dependencies under `/tmp/megaflux-deps`. The initial eight-rank suite passed eight
cases per rank: two metadata cases with 64 queued changing-input Graph replays,
and six complete-forward comparisons spanning NCCL/NVLink and three decode
planning policies. The NVLink follow-up passed six cases per rank, enabling
Plan hash agreement and rank-0-only forced capacity fallback. Outputs match the
original fused MegaMoE bit-for-bit for shrinking, zero and asymmetric local
batch sizes. Transport checks cover rank order, invalid-route status reduction,
fallback votes, hash mismatch and reuse of the same arena without per-replay
host synchronization.

Review subsequently added EP group-size validation, a checked total arena size,
and a 64-bit host grid calculation. The final metadata build again passed both
queued-replay tests on all eight ranks. These guards do not alter the tested
workload's GPU arithmetic. The full-forward measurements below precede these
guards; their exact source hashes are saved alongside the samples.

Raw logs and complete per-iteration benchmark samples are in
[`megaflux-results/nvlink-metadata-20261027/`](megaflux-results/nvlink-metadata-20261027/).
Remote commands, original source backups and service attempts are retained in
`/tmp/megaflux-results/nvlink-metadata-20261027/`.

## Full-forward measurement

H5120/I2304/E384/K6, EP8, 512 tokens/rank, one replica slot/rank, five warmups,
30 CUDA Graph samples. Each sample is the maximum GPU elapsed time across ranks.
All six variants in each run match the original output bit-for-bit. Fixed sync
and fixed pipeline use identical assignment hashes and exclude online planning
while retaining each invocation's required replica-weight transfers. The table reports median
milliseconds; the JSON retains p95/p99 and all samples.

| Route / metadata | Original | Fixed sync | Fixed pipeline | Online pipeline | Online benefit gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced / NCCL | 1.081 | 2.926 | 2.669 | 5.653 | 4.411 |
| Balanced / NVLink | 1.087 | 2.906 | 2.669 | 5.593 | 4.397 |
| Hotspot / NCCL | 6.048 | 3.167 | 3.241 | 6.077 | 9.834 |
| Hotspot / NVLink | 6.015 | 3.172 | 3.228 | 6.066 | 9.839 |

These single sweeps do not establish a material metadata speedup. Fixed-plan
replication helps the hotspot, but the online search consumes that benefit.
The benchmark's optional benefit gate is calibrated from its preceding samples;
it is distinct from the separately calibrated H200 vLLM admission policy.
No paper-level or end-to-end speedup follows from these numbers.

Reproduce each combination with:

```bash
cd /root/flashinfer
PYTHONPATH=/root/flashinfer:/tmp/megaflux-deps \
NCCL_NVLS_ENABLE=0 NVSHMEM_DISABLE_NVLS=1 NVSHMEM_DISABLE_CUDA_VMM=1 \
NVSHMEM_SYMMETRIC_SIZE=16G \
python3 -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmarks/moe_ep/backends/mega/kernel/sm90/bench_megaflux_forward.py \
  --cuda-graph --metadata-transport nvlink --tokens 512 \
  --routing hotspot --warmup 5 --iters 30
```

Use `nccl` for the transport control and `balanced` for the other routing case.
The new transport is also a diagnostic intervention for the previously recorded
full-model CUDA Graph host crash. Passing these kernel tests does not establish
that the serving crash is repaired.

## vLLM serving experiment

DeepSeek-V4.1-Flash-1017, vLLM 0.32.0, TP1/DP8/EP8 on eight H200 GPUs,
8192 context, 512 maximum batched tokens, 32 maximum sequences per rank,
GPU-memory utilization 0.85, eager model execution and a 16 GiB static NVSHMEM
heap. MegaFlux retains its local MoE Graphs and enables `metadata=nvlink`,
the existing 12289-route threshold and calibrated H200 admission. Native uses
automatic `HummingGroupedExperts`. No debugger, blocking CUDA launch mode,
runtime probe or model-source instrumentation is used.

Every service first runs the 13-case semantic/streaming verifier and the same
4K/c64 and 7K/c128 preconditioning (seed1042, eight output tokens). Formal cases
use 128 and 256 requests, respectively, 64 output tokens, ignored EOS,
temperature zero, exact input lengths and four warmup requests. Completed
counts alone are insufficient: every saved output length must equal 64.

Native A1 and the three-seed MegaFlux B service completed. B passed all 1152
formal requests (seeds42/43/44) with zero failures or truncations and 13/13
semantic checks. Its output throughput by seed was 236.39/241.34/230.78 tokens/s
at 4K/c64, and 191.86/187.20/196.25 tokens/s at 7K/c128. These are three rounds
inside one service lifetime, not independent restarted trials. A1 passed all
384 formal requests and 13/13 semantic checks. The separately recorded native
A2 retry also passed all 384 formal requests and 13/13 semantic checks. Matched
seed42 measurements, with throughput in output tokens/s:

| Input / concurrency | Native A1 | MegaFlux B | Native A2 retry | B gain vs A1 / A2 retry |
| --- | ---: | ---: | ---: | ---: |
| 4096 / 64 | 137.07 | 236.39 | 184.69 | +72.5% / +28.0% |
| 7168 / 128 | 117.37 | 191.86 | 152.14 | +63.5% / +26.1% |

| Input / concurrency | Native A1 P95 TTFT / TPOT | MegaFlux B P95 TTFT / TPOT | Native A2 retry P95 TTFT / TPOT |
| --- | ---: | ---: | ---: |
| 4096 / 64 | 15.83 s / 310.95 ms | 11.67 s / 176.00 ms | 13.95 s / 218.95 ms |
| 7168 / 128 | 58.72 s / 318.17 ms | 32.96 s / 175.01 ms | 40.53 s / 218.78 ms |

Native throughput changed by +34.7% and +29.6% between the two successful
services. Its cause remains undetermined. These observed gains are not a
confidence interval or an established speedup bound. Across A1, B and the A2
retry, the valid formal samples contain 1920 complete requests, 11796480 input
tokens and 122880 output tokens. All individual lengths and throughput
arithmetic were checked against the saved responses.

The first following native A2 attempt failed during 4K/c64. The first reported
worker exception surfaced in native `pynccl.reduce_scatterv` / `ncclReduce` as
`NCCL error: unhandled cuda error`. That service selected Humming and did not
enable MegaFlux. The log does not establish the originating CUDA fault. The
client reported 56 completed and 72 failed requests; all 56 completed responses
were truncated, leaving **zero** full-length responses. Its throughput is
invalid and is excluded. The complete original log, responses, source hashes
and failed audit are retained under `native-A2/`. The successful A2 retry used
identical source, environment and serving parameters and never overwrote this
failure. All recorded hashes also match between B, failed A2 and the A2 retry:
35 actual source files, plus nine pre-existing macOS metadata files. The 34
FlashInfer source files match the local workspace; the remaining source is the
installed vLLM adapter. This is therefore not an all-attempts-passed A/B/A
stability result.

This comparison measures the integrated backend, including local Graph replay,
owner execution, metadata and admission. It does not isolate the contribution
of dynamic replicas or the new metadata transport, and there is no fresh
NCCL-MegaFlux serving control in this experiment. The prior full-model Graph
failure must not be called repaired merely because this B service completed.

The reported memory allocated while loading is 45.83 GiB for native versus
89.37 GiB for MegaFlux; the latter includes prepared backend allocations.
Rank0 reports 71.48 versus 8.61 GiB available KV-cache memory. Both accommodate
this workload, and none of the completed services records a preemption. Ten-second
sampling saw active SM clocks of 1980 MHz in both, and peak memory of
123895/123905 MiB for A1/B. Both logs contain DP statistics-order warnings
(116/220 respectively); the A2 retry contains 98. The A2 retry also logs one
eight-line `EngineDeadError` traceback during intentional shutdown after
`ALL_DONE`, following completion of the request audits. Zero failed requests
does not imply warning-free logs. The harness stopped only its own services;
all eight GPUs were idle with zero allocated device memory afterward.

Per-request JSON results are preserved losslessly as `.json.gz` alongside
commands, response checks and source hashes in the local evidence directory.
The original uncompressed results, full service logs and source snapshots
remain in the remote experiment directory. Full service logs are also retained
locally as `server.log.gz`. A complete 321-file raw evidence archive is saved
locally at `/tmp/megaflux-nvlink-metadata-20261027.tar.gz` (SHA256
`d9fdecd856e29cfca54e278ec11199d6e14930b682f38f3e9967e3ec6b884381`).

[`serving-summary.json`](megaflux-results/nvlink-metadata-20261027/serving-summary.json)
contains the audited metrics and excluded attempt.
[`serving_check.py.gz`](megaflux-results/nvlink-metadata-20261027/serving_check.py.gz)
preserves the exact experiment harness; it reproduces one service attempt and refuses
to reuse an existing output directory. Run on the idle eight-GPU host:

```bash
cd /root/flashinfer
python3 /tmp/megaflux-results/nvlink-metadata-20261027/serving_check.py \
  --name native-new --backend native --seeds 42
python3 /tmp/megaflux-results/nvlink-metadata-20261027/serving_check.py \
  --name megaflux-new --backend nvlink --seeds 42 43 44
```

The new transport remains opt-in. Further repeated service lifetimes and a
matched NCCL-MegaFlux control are needed before attributing a serving gain to
this transport or claiming that the historical Graph crash is resolved.
