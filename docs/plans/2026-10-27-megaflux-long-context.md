# MegaFlux stability, accuracy and long-output validation

Use `ssh p2p-test-lrf`, `/root/flashinfer`, DeepSeek-V4.1-Flash-1017 and eight
H200 GPUs. Preserve the previously tested production source. Evidence lives
under `/tmp/megaflux-results/long-context-20261027` in the container.

1. Re-run EP8 metadata, replay and independent-reference tests. Check all six
   complete-forward variants against the independent instruction reference at
   H5120/I2304/E384/K6/T512 for balanced and hotspot routing.
2. Run 40 independent layers for 2000 changing-input steps with NVLink metadata,
   no periodic host synchronization, interleaved NCCL and finite-output snapshots.
3. Run a fresh native service, a fresh MegaFlux service, then a separate
   MegaFlux restart control. Use identical model/configuration except the backend;
   keep TP1/DP8/EP8, batched tokens 512, max sequences 32 and memory fraction 0.85.
4. Set max model length to 65536. Measure 16384 input + 16384 output and 32768
   input + 32768 output, eight requests and concurrency eight. Ignore EOS and
   audit every input/output length. Use an explicit short prewarm, no hidden
   full-length endpoint precheck or full-length warmup. The restart control uses
   the same long inputs with 512 output tokens.
5. Compare a frozen, seeded 64-question GSM8K test subset, six retrieval cases
   at early/middle/late positions in 16K/32K contexts, and teacher-forced token
   probabilities at 512/4096/16384/32768 tokens. Keep requests, complete responses,
   fixture/data hashes and answer extraction. Repeat logprob and semantic checks
   after the workload to detect drift within a service lifetime.
6. Report request failures/truncations, kernel numerical checks, paired exact
   match scores, chosen-token probability differences, top-1 agreement, latency,
   throughput, duration and memory. Retain every failed attempt separately.

After the three initial lifetimes, run an additional fresh MegaFlux service
with `--short --concurrency 128`. This exercises 16K/32K input prefill and 512
output tokens under the concurrency that previously exposed serving failures.
Treat it as a separate stability stress, not a performance comparison with the
concurrency-eight long-output cases.

The concurrency-128 MegaFlux service failed during the 32K-input case with a
runtime CUDA launch failure. Preserve that attempt and exclude its incomplete
throughput result. Add a fresh native `--short --concurrency 128` control with
the unchanged runner to investigate attribution. Report skipped post-workload
checks explicitly; a failure remains part of the campaign even if a later
control succeeds.

This is a bounded stability campaign, not a production availability guarantee.
GSM8K uses zero-shot, non-thinking prompts and a fixed subset, so it does not
establish full-benchmark accuracy parity. The probability probe measures a
fixed teacher-forced corpus, not a full-vocabulary divergence. Forced long
generation validates length/performance and does not establish answer quality.
