# DeepSeek V4.1 MXFP4 with MegaFlux on SM90

`patch_vllm_megaflux_sm90.py` adds the opt-in
`flashinfer_moe_ep_megaflux_sm90` backend to an installed vLLM 0.32.x package.
It connects vLLM's DeepSeek-V4 routed-expert path to FlashInfer's SM90
MegaFlux graph backend, retaining packed MXFP4 weights and UE8M0 scales. The
patch backs up the three vLLM modules it changes under `/tmp` before editing.
It is a deployment patch for the pinned vLLM release, not an upstream vLLM
change.

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
NVSHMEM_DISABLE_CUDA_VMM=1 NVSHMEM_SYMMETRIC_SIZE=16G \
vllm serve /gpfs/rd/models/DeepSeek-V4.1-Flash-1017/ \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 --data-parallel-size-local 8 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.85 --enforce-eager \
  --kernel-config '{"moe_backend":"flashinfer_moe_ep_megaflux_sm90"}'
```

The validated DeepSeek V4.1 EP=8 startup used a 16 GiB static NVSHMEM heap and
`gpu-memory-utilization=0.85`; it allocated about 634k KV-cache tokens per rank.
Keep the `NVSHMEM_*` settings consistent across ranks. The pipeline uses a
compact selective receive cache with `world_size * replica_budget` rows per
layer; the non-selective fallback retains the larger owner-export cache.
Eager mode avoids allocating additional CUDA graph pools during bring-up.

The implementation is currently limited to DeepSeek-V4 routing with MXFP4
experts on compute capability 9.0, linear expert placement, BF16 activations,
and no EPLB or dual-batch overlap. It uses one reserved replica slot per rank
and static load balancing. Other vLLM backends remain selectable and unchanged.

The graph backend's W1/W2 selective weight copy and compute pipeline are
exercised through `MoEEpMegaLayer`. Validate an installation by issuing a
generation request and confirming both `Using FlashInferMoeEpExperts` and a
successful `/v1/chat/completions` response in the worker logs. Compare latency
only against a separately run vLLM baseline with identical model, topology,
prompt, and generation parameters.
