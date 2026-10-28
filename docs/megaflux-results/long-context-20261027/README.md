# Long-context campaign evidence

These files summarize the 2026-10-27 EP8 H200 campaign. See
[the report](../../megaflux-long-context.md) for results and limitations.

- `audited-summary.json`: independent arithmetic checks, exact request lengths,
  per-request latencies, source snapshot verification, telemetry and log review.
- Accuracy comparison JSON: paired answers and per-length chosen-token logprob
  differences; every comparison records its fixture hash and missing records.
- Attempt directories: commands, environment, audits, semantic observations,
  accuracy summaries and compressed original logs/telemetry.
- Root logs and commands: GPU regression, independent reference and 40-layer
  replay stress. `stress_nvlink.py.gz` retains the exact stress source.
- `compact-manifest.json`: hashes for every copied file and the complete archive;
  JSON copies are pretty-printed, while compressed logs/source preserve bytes.

The complete archive includes raw benchmark inter-token samples and generated
text, all accuracy request/response records, the original GSM8K test JSONL and
license, per-attempt source snapshots and test/reference source snapshots.
It is kept outside tracked evidence in the existing ignored `.fi_workspace`
directory on both the local and remote repositories. The manifest records its
absolute locations and checksum. No full benchmark response has been replaced
by a compact summary in that archive.

`file-manifest.json` records SHA-256 checksums for the raw experiment files,
excluding itself. The MegaFlux concurrency-128 32K attempt failed; its
invalid throughput and partial responses are retained, not treated as passes.
Diagnostic excerpts are capped at 200 matches with full counts and logs saved.
The source guards cover 37 selected implementation/runner files per lifetime;
model weights are identified by path and file metadata rather than full-content
hashes. GPU test/reference sources were snapshotted after the GPU tests, with
that timing recorded explicitly. Shutdown diagnostics are retained even when
the request audits and runner exits succeeded.
