# SM90 MegaFlux GPU planning with synchronous transport

The opt-in `sm90_fp8_mxfp4_bf16_megaflux_online` backend consumes current
routing on GPU. It retains the frozen beta=32/N32/C1x1 Humming tactic and
physical weight bank. The CPU Plan implementation is only an oracle in tests;
a shape-only initializer provides empty metadata before compilation and never
invokes the CPU routing oracle.

`LocalHistogram` assigns stable local occurrences by scanning token/top-k order
for each expert. GPU all-gather exchanges C[source,expert], retaining source
prefixes. The bounded CUDA control kernel implements all four candidates,
fixed donor priorities, existing-copy reuse, uncapped donor movement, candidate
tie breaks, replica pruning and refinement exactly as the CPU oracle. Control
scratch is explicitly preallocated in global memory; large per-thread arrays
would cause excessive CUDA local-memory backing. The first serial control
implementation establishes correctness, not low planner latency.

The Plan materializer writes physical slots, block and replica records,
separate token/SF pool prefixes and stable source route records. Each source
combines its local ordinal with the C prefix, then sends fixed-capacity route
word chunks by GPU all-to-all. The executing rank receives counts and source
words directly; no `.cpu()`, `.item()` or host routing branch occurs in forward.

Both candidate and owner-only Plans are generated on GPU. A collective MAX
commits identity on capacity failure before route exchange. Optional two-word
Plan hashes detect cross-rank disagreement and feed the same collective
fallback. A weight-version disagreement also commits identity on every rank.
Invalid IDs/duplicates abort collectively before dispatch. Incompatible fleet
geometry/layout fails initialization; post-commit communication errors abort
rather than replay work on an individual owner.

## Transport and limitations

`DeviceReplicaTransport` uses tracked symmetric complete-leg export roots.
GPU kernels select and pack the distinct experts exported by each owner.
Each rank synchronously gets every owner's bounded export root, then GPU
kernels unpack only the selected replica slots. This conservative correctness
path copies inactive export padding and exports needed by other receivers.
Those bytes and all planner/route/packing work are real overhead; they must
be charged in benchmarks. It is not a selective-transfer performance result.

Export capacity is `min(owned,(P-1)*R)` per owner, independently of R receiver
slots. Every invocation republishes and transfers both legs completely,
including folded offsets and residuals, before GEMM. No runtime weight update
or capture is supported. P<=8, E<=512 and R<=4 are explicit bringup bounds.

## Use

```python
from flashinfer.moe_ep.megaflux_plan import PlanConfig
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_online import (
    Sm90_Mxfp4_MegaFluxOnlineConfig,
)

config = Sm90_Mxfp4_MegaFluxOnlineConfig(
    intermediate_size=intermediate, top_k=top_k,
    plan_config=PlanConfig(E, P, R, max_tokens, top_k, weight_version=version),
    load_balance_mode="static", debug_plan_agreement=True,
)
# Pass as MegaConfig(megakernel=config, quantize_input=True,
#                    preprocess_weights=True) to MoEEpLayer.
```

## GPU gates

`tests/moe_ep/test_megaflux_gpu_plan.py` compares every active record and
statistic, source occurrence, receiver count/word and route record to the CPU
oracle. Coverage includes R=0/1/2/4, ragged sources, masking, empty input,
global tails, capacity failure, invalid routes and E384/P8/K6/M512 with six
hot experts. All 50 cases pass on H200.

`tests/moe_ep/test_megaflux_identity.py -k online` runs the actual backend on
EP8. It compares both complete weight legs bytewise, output bitwise to the
original backend and independently to the numerical oracle; then it changes
routes and source lengths on the same workspace. Plan records are compared to
the oracle and cross-rank q arrays are checked. The capacity test injects a
failure on rank 0 only, requiring all ranks to commit identity.

Accepted H200 results: 50 fieldwise GPU cases; EP2 4 online cases per rank;
EP8 14 online cases per rank after moving control scratch to global storage.
EP8 includes rank-0-only capacity failure and cross-rank version disagreement.
Raw logs: `megaflux-results/gpu-planner.txt`, `online-ep2.txt`, `online-ep8.txt`.
Ruff and focused mypy pass. No pipeline/Graph or speedup claim follows from
these correctness gates.
