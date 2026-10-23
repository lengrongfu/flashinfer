# MegaFlux forward on SM90 MXFP4: bring-up ledger

Status: original-backend baseline and CPU reference established; an offline
R=0 identity receiver adapter now passes single-GPU and EP8 validation. The
physical-capacity identity gate now also passes R=1/2/4 reserved-slot tests;
replication remains incomplete. Initial pooled layer-level evidence is
superseded as described in `megaflux-identity-executor.md`. See
`megaflux-plan-abi.md` and `megaflux-identity-executor.md` for the precise
contracts, evidence and limitations.

## Scope and revision

- Task reference: `ad57e22e41ce7474b5ab8ac75191bff3debc5941`.
- Inspected checkout: `608772bd85dc4cb5a4f606e7f70009dd9330e024`, branch
  `feat/add-makeflux`; initially clean. No reset to the task reference.
- Algorithm: [MegaFlux, arXiv:2610.00671](https://arxiv.org/html/2610.00671).
  The paper's implementation and reported performance target Blackwell;
  this task targets Hopper and retains the existing Humming/WGMMA arithmetic.
- Forward only, one NVLink domain, one model weight version, one invocation
  per workspace. New execution must be explicitly selected. Preserve the
  original backend, logical expert IDs, top-k weights, activation and clamp.
- Priority workload: H=5120, I=2304 (post-SwiGLU), E=384, top-k=6, EP=8.
  These are validation parameters, not algorithm constants.

## Source map

Paths below are relative to the repository root. `DROP` denotes
`flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel`.
`BACKEND` denotes
`flashinfer/moe_ep/backends/mega/kernel/sm90/fp8_mxfp4_bf16_pull_cutedsl`.

| Responsibility | Current symbols / files |
| --- | --- |
| Layer lifecycle | `flashinfer/moe_ep/modes/mega_layer.py`: `MoEEpMegaLayer`, `_preprocess_weights`, `_ensure_workspace`, `create_workspace`, `forward`, `compute_staged` |
| Backend boundary | `BACKEND/backend.py`: `Sm90PullMxfp4MegaKernelBackend`, `_fused_frontend_config_kwargs`, `_allocate_workspace`, `stage_inputs`, `compute` |
| Numerical/config contract | `BACKEND/config.py`: `Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig` |
| Raw -> kernel-ready weights | `BACKEND/weights.py`: `preprocess_mega_weights`, `_preprocess_humming_leg_chunked`, `_interleave_gate_up_8`, `validate_transformed_mega_weights` |
| Input quantization/masking | `BACKEND/staging.py`: `stage_mega_moe_inputs`, `validate_sm90_mxfp4_forward_inputs` |
| Public kernel boundary | `DROP/__init__.py`: exported Hopper MXFP4 helpers |
| Adaptation/compilation | `DROP/shim/hopper_mxfp4.py`: `MegaMoEHopperMxfp4Config`, `_ensure_mega_compiled`, `get_symm_buffer_for_hopper_mxfp4_mega_moe`, `hopper_mxfp4_mega_moe` |
| Shared launch lifecycle | `DROP/shim/hopper_fp8.py`: `run`, `prepare_launch`, `make_launch_thunk`, `_build_mega_runtime_kwargs` |
| Pool sizing/region tables | `DROP/src/moe_hopper_fp8/megamoe_kernel_fp8.py`: `_pool_shapes`, `_build_local_region_specs`, `_build_shared_region_specs`, `__call__` |
| Dispatch/return/reset | `DROP/src/src/token_comm.py`: `TokenSrcMetadata`, `dispatch_prep`, `dispatch_barrier`, `dispatch_pull`, `token_back_by_push`, `kernel_tail`, `tail_reset_counters` |
| Scheduler/task identity | `DROP/src/moe_nvfp4_swapab/fc1_fc2_fuse_sched.py`: `fc12_expert_phase_tiles`; `custom_ext.py`: `GluMxFp8WorkTileInfo`, `get_gmem_tensor` |
| Weight load issue | `DROP/src/moe_hopper_fp8/kernel_fp8_glu_fc12_swapab.py`: `_issue_selected_weight_tma_tile` |
| MXFP4 offsets and arithmetic | `DROP/src/moe_hopper_fp8/kernel_mxfp4_fp8_glu_fc12_swapab.py`: `_tma_load_a_with_weight_sf_task_tile`, `_copy_weight_scale_cpasync`, K256 variants, `_mma_mxfp4_per_tensor_1xacc` |
| Offline tactic selection | `BACKEND/tuner.py`, `DROP/shim/mxfp4_tuner.py` |

### Weight ABI

Let L=E/P, H=hidden, I=post-SwiGLU intermediate. Each leg is
`(packed_weight, folded_offset, activation_placeholder, weight_scale)`.

| Tensor | FC1 | FC2 | Type / layout |
| --- | --- | --- | --- |
| Raw payload | `(L, 2I, H/2)` | `(L, H, I/2)` | uint8 packed E2M1 |
| Raw scale | `(L, 2I, H/32)` | `(L, H, I/32)` | uint8 E8M0 |
| Kernel payload | `(L, H/2, 2I)` | `(L, I/2, H)` | uint8, storage-K stride 1 |
| Folded offset | `(L, 2I/64, H/128, 16, 16)` | `(L, H/64, I/128, 16, 16)` | contiguous uint8, 16-byte aligned |
| Activation placeholder | `(1,)` | `(1,)` | contiguous float32 ones |
| Residual compensation | `(L,)` | `(L,)` | contiguous float32, Humming residual times 64 |

Preprocessing interleaves FC1 gate/up in groups of 8, applies Humming with
fixed exponent range 11, then transposes payload dimensions 1 and 2. The
normal allocated payload strides are `(I*H, 1, H/2)` for FC1 and
`(H*I/2, 1, I/2)` for FC2. Runtime validation requires storage-K stride 1;
it does not require every user-supplied payload to have those exact strides.
Never make the transposed view contiguous. Replication must copy payload,
folded offsets and residual compensation; it must not re-run preprocessing.

### Execution and identity

`forward` prepares weights/workspace, allocates an owned output if needed,
stages activation/scales/top-k inputs, and calls `compute_staged` -> backend
`compute` -> public Hopper shim -> fused launch and standalone top-k reduce.
Within the persistent kernel, dispatch preparation and count publication
precede pulls; arrival counters gate FC1, FC1 completion gates FC2, and FC2
completion gates return. These stages overlap across tasks.

Current ownership is `rank = expert_id // L`, `local_expert = expert_id % L`.
This arithmetic occurs in dispatch preparation AND count publication, not
just the weight selector. Scheduler `expert_idx` selects weight/TMA slices
through `custom_ext.get_gmem_tensor`. The current kernel also derives logical
E as `world_size * num_experts_per_rank`; increasing this field to add replica
slots would incorrectly change logical E.

Return identity is already represented by `TokenSrcMetadata`: one int64,
low 32 bits source token, high 32 bits `(source_rank << 16) | source_topk`.
Preserve this identity through physical relocation and preserve the existing
standalone top-k reduction order. Config already supports disabling dispatch
dedup and requires grouped return and in-kernel FC2 reduction to be false
for the production MXFP4 contract.

### Granularity and capacity

For swap-AB, `token_tile_size = mma_tiler_mnk[1]`, and
`cluster_tile_tokens = mma_tiler_mnk[1] * cluster_shape_mnk[1]`.
The MXFP4 shim passes token padding `mma_tiler_mnk[1]`; SF padding is separately
128. Scheduler state tracks distinct data-row, SF-row and token-block
cumulatives. `fc12_expert_phase_tiles` uses ceil(tokens / cluster token tile),
with additional handling for optional tail-split pairs.

Proposed first geometry to validate: `(128,32,128)`, cluster `(1,1,1)`,
pingpong false, no dedup, no grouped return, no tail-split. Its candidate
block size is 32, not 128. This is a source-derived candidate, NOT a frozen,
GPU-validated planner ABI yet. SF offsets need independent 128-row alignment.

Current owner-only pool formulas use
`raw = P * max_tokens_per_rank * min(top_k,L) + L*(token_padding-1)`,
rounded to token padding. SF and task capacity derive from that pool.
These owner-only bounds must be re-derived for relocated work, alongside
every slot-indexed counter, task loop and metadata plane.

### Synchronization and reset audit

- `dispatch_barrier` performs a software grid rendezvous, publishes per-source
  counts to owners, fences at system scope, then completes the NVLink barrier.
- `fc1_tma_b_predispatch_spin` waits for the full valid cluster token block.
- `_issue_selected_weight_tma_tile` selects the leg-specific weight tensor,
  then invokes the payload/auxiliary load path. New leg-ready checks must
  precede this issue point and cover folded offsets and residual readers.
- Existing weight/auxiliary producer-consumer pipelines order shared-memory
  consumption; they do not establish completion of a new remote weight copy.
- `kernel_tail` drains work, resets shared accumulating counters between two
  NVLink barriers, then resets local accumulating counters. The phase-flip
  signal lives outside the zeroed prefix. New epoch/readiness state must be
  isolated from these bulk resets.
- Layer workspace creation is forbidden during capture; fixed knobs are
  required for explicit workspace profiles. Warmup and all allocations must
  precede capture.

## Implementation boundaries and stage gates

`kernel_src/README.md` requires verbatim vendored `src/`. Upper backends may
only use a drop's package exports. The package docstring identifies
`flashinfer/moe_ep/cute_dsl/` as the maintained implementation location.
Implement the new executor there using public helpers, or create a separately
tracked drop/shim with provenance; do not mutate the existing vendored tree.

1. Reproduce original correctness and baseline; persist input recipe, complete
   resolved tactic, environment, output comparison and raw latency samples.
2. Define Plan ABI and deterministic CPU oracle/property tests only after gate 1.
   Separate logical E, owned L, physical capacity L+R and active slots.
3. Identity-plan executor, R=0; compare output, exact routes and overhead.
4. One static replica and fixed weight bank, synchronous complete-leg copies.
5. GPU planner matching the oracle and collective identity fallback.
6. Separate W1/W2 readiness and independently progressing transfer workers;
   validate injected delays against the same synchronous plan.
7. Epoch/retirement/Graph stress and GPU benefit gating.
8. Restore optimizations only with dependency proofs; complete ablations.

No later stage is accepted based on CPU tests or compilation alone.

## Remote environment (2026-10-23)

All GPU work must execute through `ssh p2p-test-lrf`.

- 8 NVIDIA H200 GPUs; every GPU pair reports NV18 in `nvidia-smi topo -m`.
- System interpreter is `python3`, Python packages under `/usr/local/lib/python3.12`.
- PyTorch 2.13.0+cu130, CUDA reported by Torch 13.0.
- nvidia-cutlass-dsl 4.8.0, nvidia-nvshmem-cu13 3.4.5,
  apache-tvm-ffi 0.1.11, pytest 9.1.1.
- Installed flashinfer-python 0.7.0.post1 is NOT established as the current
  checkout. `find_spec('nvshmem')` returned None; the NVSHMEM runtime library
  wheel alone does not establish that required Python bindings are available.
- `/usr/local/cuda/bin/nvcc` exists. Local git submodules are uninitialized.
- Inherited `NO_COLOR=1` breaks wstunnel argument parsing. Connection succeeds
  with `env -u NO_COLOR ssh p2p-test-lrf ...` and network-enabled execution.
- User selected the existing remote `/root/flashinfer` checkout after the
  initial source-transfer request was rejected. No repository transfer was
  needed. Remote branch `main`, clean HEAD
  `6a84f79d2aa4c25aa272fa1ee4beea09ae6eee12` has identical `flashinfer/moe_ep`,
  `tests/moe_ep` and `benchmarks/moe_ep` trees to local HEAD (verified with
  `git diff --quiet 608772bd85dc4cb5a4f606e7f70009dd9330e024 HEAD -- ...`).
- Original single-GPU independent-oracle test: **3 passed**, dispatch warps
  1/2/4, 8.45 seconds. Raw output: `megaflux-results/baseline-single.txt`.
- Initial EP8 run failed before kernel execution with `ModuleNotFoundError:
  nvshmem` in `core/runtime/bootstrap.py`. Installed `nvshmem4py-cu13==0.3.1`
  with `--no-deps --target /tmp/megaflux-deps`; import succeeds. Global packages
  were not replaced. EP8 retry uses `PYTHONPATH=/tmp/megaflux-deps`,
  `NCCL_NVLS_ENABLE=0`, `NVSHMEM_DISABLE_NVLS=1`.
- EP8 retry: **10 passed per rank**, 23 deselected per rank. Includes the
  existing independent numerical oracle, sparse owners, zero-input sources,
  all-masked routing, repeated calls and 16-replay outer CUDA Graph cases.
  Rank-local reported durations vary; final output reports 117.36 seconds.
  Raw output: `megaflux-results/baseline-ep8.txt`.
- Target-shape benchmark completed with 3 warmups and 20 iterations, balanced
  block-permutation routing, all three points `pass` (benchmark completion,
  not an independent numerical accuracy check). Complete effective tactic
  and output: `megaflux-results/baseline-bench.txt`.

| Tokens/rank | Compute max-rank median (us) | E2E max of rank means (us) |
| --- | --- | --- |
| 64 | 696.112007 | 1096.66 |
| 512 | 1232.847989 | 1593.90 |
| 4096 | 6473.776102 | 6885.61 |

These are the existing benchmark's warm CUDA-event metrics, NOT
per-iteration rank-max p50/p95/p99. Direct launch includes host launch gaps.
No MegaFlux speedup has been measured. Effective tactic uses static geometry
`(128,32,128)`, cluster `(1,1,1)`, group_hint 132, atomic-counter scheduling,
pingpong false, dedup false, grouped return false, BF16 combine, epi return,
FC1 store offload true, early publication false, producer folding false,
two scheduler stages, no tail-split and no FC2 N8 tail.

## Baseline commands after source transfer and dependency validation

Run inside `/root/flashinfer`, with `python3` and its matching
`torchrun` environment. Preserve complete stdout/stderr rather than recording
only a speedup ratio.

```bash
MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 python3 -m pytest -v -m arch_hopper \
  tests/moe_ep/test_sm90_pull_mxfp4_kernel_vs_reference.py

PYTHONPATH=/tmp/megaflux-deps NCCL_NVLS_ENABLE=0 NVSHMEM_DISABLE_NVLS=1 \
python3 -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -v \
  tests/moe_ep/test_moe_ep_sm90_pull_mxfp4_mega_multirank.py \
  -m 'gpu_2 and arch_hopper'

PYTHONPATH=/tmp/megaflux-deps NCCL_NVLS_ENABLE=0 NVSHMEM_DISABLE_NVLS=1 \
python3 -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmarks/moe_ep/backends/mega/kernel/sm90/bench_moe_ep_sm90_mega.py \
  --backend sm90_fp8_mxfp4_bf16_pull_cutedsl \
  --hidden 5120 --intermediate 2304 --num-experts 384 --top-k 6 \
  --tokens 64,512,4096 --mxfp4-tactic-source explicit \
  --swap-ab --mma-tiler 128,32,128 --cga 1,1 --pingpong off \
  --token-back epi_warps --no-fold-producer-warps
```

The existing benchmark supplies deterministic weights (seed 0x4D584650+rank),
activations (42+rank) and routing (1234), and records the effective tactic.
The baseline command is a starting measurement, not the final requested
p50/p95/p99 ablation suite. Freeze the full emitted tactic after validation,
then use identical weights/routing/return modes for all comparisons. Existing
small-shape numerical tests do not establish target-shape numerical accuracy.

CPU reference validation: 96 tests passed on the container's Python 3.12,
covering deterministic plans, conservation, packing, multiple hotspots,
R=0/1/2/4, global tails, masks, ragged and empty sources, descriptor capacity
failure and int64 bounds. Raw output: `megaflux-results/planner-cpu.txt`.
This establishes only the CPU reference. It is not a GPU planner or backend.

Identity adapter validation: single-rank 3 dispatch-warp variants and EP8
4 routing cases per rank pass the independent numerical oracle. EP8 results
are bitwise equal to the original backend, including repeated calls and
16 fixed-Plan Graph replays. Original-backend regression also passes (single
rank and all 10 EP8 cases). Paired H5120/I2304/E384/top-k6 benchmark requires
bitwise equality at the target shape and records per-iteration rank-max
latency. See `megaflux-identity-executor.md` and raw identity result files.

Pending: target-shape independent numerical correctness, GPU Plan comparison,
weight transfer,
delay injection, CUDA Graph stress, all MegaFlux performance ablations.
