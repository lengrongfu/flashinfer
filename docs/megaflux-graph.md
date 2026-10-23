# SM90 MegaFlux device epochs and CUDA Graph

The opt-in `sm90_fp8_mxfp4_bf16_megaflux_graph` backend qualifies the W1/W2
pipeline for warmed, fixed-capacity capture. One workspace supports one
inflight invocation; launch/replay invocations on a serialized stream and keep
the layer alive until its Graph users are finished.

Before capture, all ranks preprocess weights, allocate physical/symmetric
banks and Plan/route/flag storage, compile the kernels and execute warmup.
Cold capture fails before staging or allocating lazy resources. This path
adds a GPU epoch increment per invocation. Candidate and identity metadata
receive that device epoch before collective agreement; a Python value frozen
during capture never determines replay readiness.

Source retirement/publication remain NVSHMEM stream barriers, without host
stream synchronization. W1/W2 streams branch from publication and publish
complete-leg ready epochs exactly as in the accepted pipeline. After compute,
both branches rejoin the main stream. Thus the captured DAG includes no-work
transfers as well as consumers before the next replay can overwrite q, slots,
exports or physical weights. A pre-pack global rendezvous retires all remote
export readers. Slot records carry current and retired epochs, and appended
flag regions remain outside the original tail-zero prefix.

Destruction drains the CUDA context before releasing symmetric buffers.
Internal captured event objects may no longer be waitable after Graph release;
cleanup therefore uses the completed context drain instead of waiting those
stale event handles. Non-capture pipeline behavior retains its existing stream
synchronization and host epoch path.

## Enablement

```python
from flashinfer.moe_ep.megaflux_plan import PlanConfig
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_graph import (
    Sm90_Mxfp4_MegaFluxGraphConfig,
)

config = Sm90_Mxfp4_MegaFluxGraphConfig(
    intermediate_size=intermediate, top_k=top_k,
    plan_config=PlanConfig(E, P, R, max_tokens, top_k),
    load_balance_mode="static", debug_plan_agreement=True,
)
# Construct MoEEpLayer with MegaConfig, warm every rank on the capture stream,
# then capture forward. Change routing/masks/activations in fixed input buffers
# before replay; padded capacity, weight version/layout and addresses stay fixed.
```

## Accepted GPU evidence

H200 EP2 and EP8 each pass **5 cases per rank**, each with **128 Graph replays**
(640 per rank): R=0/1/2/4 and a capacity failure injected only on rank 0.
Each case queues bursts of eight replays without a CPU Plan or host reads,
with changing experts, source lengths (including 0/32/33/129), all-padding
transitions and different W1/W2/offset/residual delays per PE.

Every replay compares bitwise with the original backend and compares active
Plan records, counts, prefixes, slots, epochs and retirement state to the CPU
oracle. Nonempty iterations sample the independent instruction-exact numerical
reference. That test reference launches on CUDA's default stream, so its
inspection uses explicit stream dependencies outside capture. All persistent
bank, workspace, Plan, scratch, route, histogram, hash and flag addresses stay
fixed. Trailing byte canaries around payload, folded offsets and residuals
remain unchanged, and final replica bytes match original owners.

The device epoch advances from 3 after warmup to 131. Cold capture rejection,
Graph destruction and symmetric teardown pass. Raw evidence:
`megaflux-results/graph-ep2.txt` and `graph-ep8.txt`. This container has no
Compute Sanitizer; these are canary/bounds/output checks, not a memcheck claim.
No target-shape latency or speedup claim is established by this gate.

```bash
PYTHONPATH=/tmp/megaflux-deps NCCL_NVLS_ENABLE=0 NVSHMEM_DISABLE_NVLS=1 \
  MEGAFLUX_GRAPH_REPLAYS=128 torchrun --standalone --nproc_per_node=8 \
  -m pytest tests/moe_ep/test_megaflux_graph.py -q -x -s --disable-warnings
```
