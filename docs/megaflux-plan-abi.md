# MegaFlux forward Plan ABI v1 (CPU reference only)

Implementation: `flashinfer/moe_ep/megaflux_plan.py`. This is an internal,
dependency-free oracle. It is not exported from `flashinfer.moe_ep`, does not
register a backend, and cannot execute or upload a Plan. Production forward
must eventually construct these records on GPU without CPU routing readback.

## Frozen geometry and identities

The first reference configuration uses SM90 MXFP4 swap-AB `(128,32,128)`,
cluster `(1,1,1)`, beta=32, independent SF padding=128. Other granularities
are rejected until their complete pool/cluster contract is validated. The
exact numerical/communication tactic from the baseline ledger must also be
frozen by the future executor before materialization, not autotuned afterward.

`PlanConfig` separates logical E, owned E/P, physical capacity E/P+R, maximum
routing assignments P*max_tokens_per_source*top_k, and conservative block
capacity ceil(max_assignments/beta)+E. `num_active_physical_slots` counts
slots with positive work per rank; zero-work owned slots retain their storage
and identity. One workspace has one model version/layout and one invocation
epoch greater than zero. Capacity/byte arithmetic is checked against signed
int64; storage allocation is outside this module.

## Records

All integer device fields are signed int64. `pack_plan_records` serializes
active prefixes only. Allocate capacity-sized device storage before capture;
use explicit valid record counts. Padding is not implicit active work.

| Array | Shape / columns |
| --- | --- |
| expert_counts | E |
| source_expert_counts | P x E, stable source-prefix oracle |
| expert_block_offsets | E+1, exclusive block prefixes |
| slot_to_expert | P x (E/P+R), -1 for unused replica slots |
| block records | B x 8: expert, expert block ordinal, execution rank, physical slot, token row offset, valid rows, SF row offset, reserved zero |
| replica records | N x 8: expert, owner rank, owner slot, receiver rank, receiver slot, weight version, layout index, reserved zero |
| route records | assignments x 5: source rank, source token, source top-k position, global block index, row within block |
| rank_task_counts | P, token blocks per rank (not individual FC1/FC2 feature tiles) |
| rank_padded_rows / rank_sf_rows | P each, separately padded pool requirements |

`PackedPlanRecords.layout_ids` is a host layout identity table. Layout index
0 names the single configured layout; this mapping must be agreed across EP
before capture. Strings do not travel in device records. Python `Route`
objects attached to blocks are oracle data, serialized separately from the
fixed-width block records.

The record array order is logical expert, execution rank, expert block.
Owned slots keep expert modulo owned; replica slots are assigned in increasing
expert ID per receiver. Within each expert, routes are ordered by source rank,
source token, then top-k position. Blocks are cut once globally; only the last
block can be partial. Separate token/SF slot prefixes preserve independent
padding. Returned outputs retain source top-k identity; routing weights are
never modified by this planner.

## Deterministic load-only oracle

`identity=True` and R=0 retain ownership. Otherwise four bounded greedy
candidates combine total-load or largest-home-expert donor priority with
single- or multi-copy fan-out. Donor order is frozen at round entry. Existing
replicas are visited before new copies, ordered by receiver load at visit
entry, then expert and receiver IDs. New copies choose lowest load then rank;
home experts are scanned by decreasing remaining blocks then expert ID.
Movement is min(remaining owner blocks, receiver target deficit), without a
donor-overload cap. Each search uses at most R+2 rounds.

Candidate selection minimizes maximum blocks/rank, then positive remote-copy
count, then candidate enumeration order. Replica deletion visits increasing
allocation at pass entry, then expert/rank, and prefers owner then least-loaded
existing copy under the selected maximum. Refinement transfers single blocks
between the most/least-loaded ranks sharing an existing expert, with rank and
expert tie-breaks, for at most the total original block count. No new replicas
are opened by refinement. There is no backward gradient-reduction penalty or
forward benefit gate in this reference stage.

These extra tie-break and bound choices make underspecified paper details
reproducible; the future GPU implementation must use the same rules.

## Failure contract and validation

Malformed shapes/IDs, duplicates within a token, unsupported tactic and invalid
epoch fail explicitly. -1 is the only padding sentinel. Sources may contain
different token counts, including zero. Exceeding descriptor capacity returns
`CAPACITY_EXCEEDED`, `overflow=True`, statistics and empty task/replica arrays;
serialization rejects that Plan. No partial tasks can execute. This is not yet
the collective fallback protocol: GPU agreement/commit belongs to stage five.

Run the host property suite without CUDA-dependent repository conftests:

```bash
python3 -m pytest --noconftest tests/moe_ep/test_megaflux_plan.py -q
```

The tests cover conservation, stable ordering, record packing, independent SF
padding, logical/physical identities, deterministic planning, global tails,
R=0/1/2/4, multiple hot experts, fan-out, slot exhaustion, empty sources,
all-empty input, masks, descriptor overflow and signed-int64 allocation bounds.
Passing this suite establishes only the CPU reference planner.
