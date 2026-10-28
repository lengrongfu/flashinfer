# MegaFlux NVLink metadata implementation plan

**Goal:** Reduce SM90 MegaFlux forward metadata overhead and isolate captured NCCL communication from the outstanding full-model Graph failure.

**Architecture:** Keep the existing planner, weight pipeline and mathematical result. Add an explicitly selected, single-node NVLink metadata transport using NVSHMEM-owned symmetric roots. Each invocation retires old readers, publishes counts/status, publishes the collective fallback vote, and publishes routes. Peer-read kernels fuse gathers/reductions and the route transpose. NCCL remains the default and initializes/checks capabilities outside capture.

**Tech Stack:** CUDA C++, TVM-FFI JIT, NVSHMEM, PyTorch, CuTe DSL, vLLM 0.32, H200 EP8.

1. Preserve the dirty local and remote sources. Add the metadata transport and a validated backend option; include it in collective geometry agreement. Never use ordinary Torch allocation addresses as peer pointers.
2. Compare every exchanged field against NCCL, with asymmetric ranks, stale-buffer reuse, masked/empty routes and rank-local fallback. Capture and queue repeated changing-input exchanges before synchronizing.
3. Run exact-output MegaFlux replay regression on eight H200 GPUs for both transports. Keep the existing NCCL route as a control. Run focused lint and source review.
4. Expose an opt-in vLLM environment selection and benchmark rank-maximum full forward with identical weights/routes. Preserve raw samples and exact source hashes.
5. Run a clean full-model serving diagnostic using the prior failing workload, including response-length and semantic checks. Treat a pass as bounded evidence, not a proven driver fix. Compare serving throughput only on complete matching cases; retain failed attempts.

The preceding implementation and measurements already exist in `/root/flashinfer` and `docs/megaflux*.md`. Do not replace them or claim them as newly implemented. The previous retained-Graph, NCCL mixing/registration, synchronous scheduler and completion-event controls did not fix the full-model crash.

## Execution result

Implemented and validated the opt-in transport on H200 EP8, including queued
Graph replays, rank-local fallback and exact complete-forward output checks.
All four full-forward benchmark combinations preserve raw samples. The patched
vLLM integration passed semantic checks and 1152 complete formal requests in one
three-seed MegaFlux service. Native A1 and an independently named A2 retry each
passed 384 formal requests; the first A2 attempt failed in the native backend
and remains in the evidence. Baseline variability and historical Graph
stability are unresolved. The completed implementation, audited metrics,
commands and limitations are recorded in
[`megaflux-nvlink-metadata.md`](../megaflux-nvlink-metadata.md).
