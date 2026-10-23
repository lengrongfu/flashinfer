# Selective symmetric-peer transport

`Sm90_Mxfp4_MegaFluxPipelineConfig(selective_weight_copy=True)` (also inherited by
the Graph config) enables NVLink-mapped reads of GPU-selected full weight legs.
The default remains the conservative accepted full-root transport. At workspace
initialization, all ranks agree on opt-in and mapping availability in the EP
process group. If any rank cannot map any export root, all ranks use full-root
gets. Initialization may inspect that capability scalar on the host; forward
never reads q, replicas or routing back to the host.

The peer addresses are returned by `nvshmem_ptr` for tracked symmetric export
roots, never private kernel banks. NVIDIA documents direct loads/stores via
these mappings in its [NVSHMEM communication model](https://docs.nvidia.com/nvshmem/api/latest/using.html#communication-model).
The owner snapshot is published through the existing NVSHMEM barrier before
reads start. A GPU CTA per selected receiver replica copies exactly its owner's
full leg into the bounded local cache. The source export index follows the same
sorted logical-expert enumeration as the pack kernel. Four-byte coalesced loads
cover aligned rows; arbitrary tails retain a byte path. Each source and cache
row is bounded by the export capacity. Inactive slots issue no peer loads.

W1 and W2 remain on independent communication streams. The copy kernel completes
before part-wise unpack, including folded offsets and residual scale. Existing
delay injection, per-owner completion markers, release publication, consumer
acquire and async proxy fence remain in force. Previous compute and receiver
streams join before the global retirement barrier allows an owner to overwrite
its export. Pointer tables are stable through Graph replay and freed only after
transport retirement. Cache allocation is still conservative P times export
capacity, so this change reduces traffic rather than claiming memory savings.

`synchronize_weight_copy=True` inserts main-stream waits for both leg events
before compute. It provides the same-Plan synchronous ablation using identical
copy kernels, persistent compute, physical layout and SM reservation. No host
synchronization or routing readback is added by this switch.

Tests in `test_megaflux_selective.py` compare complete bytes, bitwise original
outputs, an independent numerical reference, shifted experts, empty routes,
tails and injected W1/W2/offset/residual delays at R=0/1/2/4. The requested mapped
path must actually be active on the H200 test clique. An additional full forward
case rejects on rank zero only and checks collective identity selection. Graph
tests enable the path with `MEGAFLUX_SELECTIVE=1`, check stable peer-table
addresses and replay each R/fallback case 128 times with changing routes.

H200 acceptance (2026-10-23): EP2 12 cases/rank pass; EP8 the same 12
cases/rank pass, followed by a separately passing gated-forward case/rank.
Selective EP8 Graph passes 5 cases/rank, each with 128 queued replays. Raw logs
are in `docs/megaflux-results/selective-*.txt`. The initial EP8 combined run
completed the 12 transport cases then failed the additional gate case because
the synchronous comparator did not have the gate enabled, violating its
same-Plan assertion. The comparator configuration was corrected; the separate
EP8 gated log is the acceptance evidence. No kernel arithmetic was changed to
resolve that test configuration mismatch. Performance acceptance is separate.

## Tiled copy acceptance

The initial target benchmark exposed roughly 100ms full-leg transfer spans with
single-CTA byte loops. The maintained pack, selected peer-read and part-unpack
kernels now distribute each row over 32 CTAs, using four-byte copies where row
and part alignment permit, with a byte fallback. Each CTA owns disjoint strided
indices. Inactive export rows are no longer cleared: no committed replica can
reference them, while the conservative transport still charges full-root DMA.
Global retirement, publication and full-leg readiness remain unchanged.

H200 EP8 accepts 63 cases/rank (13 full forward including gated fallback, plus
50 fieldwise GPU Plan tests) and five Graph cases/rank x128 queued replays.
Raw results: `tiled-copy-ep8.txt`, `tiled-copy-graph-ep8.txt`. Full-size benchmark
and owner-streamed independent-reference results follow separately.
