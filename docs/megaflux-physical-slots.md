# SM90 MegaFlux physical-slot identity gate

This stage adds physical capacity without assigning any work to a replica.
It is a separate opt-in backend:
`fp8_mxfp4_bf16_megaflux_physical.Sm90_Mxfp4_MegaFluxPhysicalConfig`.
Supply `plan_forward(..., identity=True)` with a nonzero `replica_budget` to
reserve unused slots. Plans with replica descriptors are rejected until the
transport stage is enabled separately.

The original and R=0 identity backend retain their contracts. Offline Plan
backends are unpooled: inheriting the original pool factory bypasses their
frontend allocation hooks and would select the wrong executor.

## Capacity audit

| Concept | Source / consumers |
| --- | --- |
| Logical E | Plan/config and kernel/communication `num_total_experts`, restored before region construction |
| Owned L=E/P | `num_owned_experts_per_rank`; original expert slots and raw/preprocessed model weights |
| Physical C=L+R | weight expert axis, static TMA shape, scheduler bounds and legacy kernel/comm `num_experts_per_rank` alias |
| Active physical slots | nonempty Plan slot counts, independent of capacity |

The maintained frontend overrides the shim's weight-capacity hook. A
construction adapter restores logical E after the base constructor establishes
physical pool capacity, and before local/shared region layout is finalized.
This sizes logical sender counters by E while sizing receiver counts/source
words, FC1/FC2 counters, token/SF/task pools, source metadata and scheduler
loops by C. Original atomic route preparation and owner-based publication are
bypassed by the Plan communication adapter, not patched with a new divisor.
Dedup, grouped return and in-kernel reduction remain disabled.

Kernel-ready owned weights are copied once into a fixed bank with C slots.
Payload storage is allocated `(C,N,packed_K)` then viewed as
`(C,packed_K,N)`; storage-K stride remains 1. Folded offsets and residual-x64
are copied too; the fixed unit activation placeholder is retained. There is
no new quantization or GEMM. TMA descriptors bind the expanded fixed bank.
Allocations and receiver metadata byte products are checked in int64 before
allocation. Each workspace owns its bank; runtime weight changes are rejected.

Plan records remain ordered by logical expert/rank, but destination token/SF
prefixes now follow physical-slot order. This distinction matters when a low
logical expert occupies a receiver's replica slot after its owned slots.

## Evidence

On `p2p-test-lrf:/root/flashinfer`, H200 EP8, the combined matrix passed
**13 cases per rank**: four actual R=0 identity cases and nine owner-only
expanded-bank cases (R=1/2/4 x normal/zero-source/all-masked). Tests assert the
actual Plan frontend and communication type, separate workspace ownership,
unchanged logical E, expanded slot dimensions and exact owned tensor copies.
Each case compares exactly to the original backend and independent numerical
oracle, repeats the workspace and replays its fixed Plan Graph 16 times.
An EP2 R=2 case also passes. CPU Plan tests: 96 pass, now checking every
physical-slot token/SF prefix. No replica transport performance is claimed.

```bash
PYTHONPATH=/tmp/megaflux-deps NCCL_NVLS_ENABLE=0 NVSHMEM_DISABLE_NVLS=1 \
python3 -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -q \
  tests/moe_ep/test_megaflux_identity.py -m 'gpu_2 and arch_hopper'
```

Raw evidence: `megaflux-results/physical-identity-ep{2,8}.txt`,
`megaflux-results/planner-physical-order.txt`. Corrected R=0 timing evidence
is explained in `megaflux-identity-executor.md`.
