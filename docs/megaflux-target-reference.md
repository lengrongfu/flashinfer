# Bounded-memory target-shape reference

The instruction-sequence reference now optionally evaluates `owned_rank` only
and returns unreduced BF16 per-route terms. It decodes one owned expert at a time
from raw MXFP4 payload and E8M0 scale bytes. This avoids gathering/decoding all
E384 experts on each GPU, keeping target H5120/I2304 oracle temporaries bounded.
Default global-reference behavior and arithmetic remain unchanged.

Each logical route has exactly one owner contribution and zeros from other
ranks. A SUM of BF16 terms therefore transports those terms without duplicating
or changing their value. The existing FP32 top-k reduction is applied only after
all terms are present, retaining the established reference's summation order.
No production Plan, weight decoder, scheduler or transfer result constructs the
reference. The test-owned WGMMA K32 sequence, activation quantization, grouped
FC2 scaling and FMA promotion remain the same as the global reference.

`test_megaflux_target_reference.py` compares owner-streamed and global BF16 terms
exactly on H200 EP8: one case/rank passes. Raw evidence is
`megaflux-results/owner-reference-ep8.txt`. The forward benchmark's `--reference`
option regenerates deterministic raw weights outside timing, gathers activations
and routes, evaluates owned terms, reduces them and checks the final output.
All six variants already must match original bitwise, so this independently
validates the shared output against the instruction reference.

H5120/I2304/E384/K6/EP8 balanced T64/T512 and hotspot T512 runs pass this oracle;
raw `forward-tiled-*.txt` records relative L2 error around 0.00166, with the
original reference tolerance unchanged. This is the implementation's MXFP4/FP8
numerical contract, not a claim of exact full-precision BF16 expert GEMMs.
