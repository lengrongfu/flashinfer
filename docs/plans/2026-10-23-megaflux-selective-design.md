# Selective weight transport and final ablation

Use GPU-selected reads from NVSHMEM peer mappings on the single-node NVLink
clique. This avoids loading another NVSHMEM device-library instance into the
Python runtime. The alternative is an RDC-linked NVSHMEM get kernel with separate
device-state initialization; it adds initialization/linkage complexity without
benefiting this H200 clique. Keep the accepted full-root gets as collective
fallback when a peer is not mapped. Never map private kernel banks remotely.

At initialization obtain each owner's pointer to each tracked symmetric export
root using `nvshmem_ptr`. Agree collectively on mapping availability and upload
stable pointer tables. Each forward retains prepack retirement and postpack
publication barriers. A per-replica CTA derives owner and sorted export index
from committed GPU q and slot map, loads only that full leg into the existing
bounded receiver cache, then returns. The communication stream orders subsequent
unpack, full-leg completion markers and release publication after these loads.
The compute consumers retain epoch-qualified acquire and async proxy fence.
Unused cache rows remain unreachable. Identity Plans produce no remote loads.
R=0 requires no peer roots. Pointer tables live until transport teardown after
all consumers/streams retire. Initial capability readback is outside forward.

Keep this opt-in until EP2/EP8 gates pass. Test active and empty Plans, budgets
0/1/2/4, changing experts and epochs, full packed-byte comparison, independent
numerical oracle, asymmetric scale/residual delays, queued Graph replay and
collective benefit rejection. Exact bytes are selected replicas times full-leg
bytes, not bounded cache allocation sizes. Retain current cache allocation shape
for conservative fallback and existing verified unpack indexing.

Six-way benchmark: original, identity, frozen GPU Plan with sync copies, same
frozen Plan with pipelining, online pipeline, online gated pipeline. The sync
variant waits on the same two leg events before compute with the same kernel
and SM reservation. Frozen Plan skips only planner/routing updates outside the
measured forward; transfers and quantization remain inside. Retain raw rank-max
iteration times and separate diagnostic events for planning, staging/launch and
leg completion. Explicit cost coefficients are inputs, not claimed calibration.
Run target H5120/I2304/E384/K6/EP8 balanced T64/T512 and legal hotspot T512.
Current co-tenant GPU load must be recorded; do not interpret noisy samples as
isolated hardware performance or assume a speedup.
