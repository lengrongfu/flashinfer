# CUDA Graph and device epoch qualification

Start after the independent W1/W2 pipeline acceptance commit. Preserve that
non-capture backend and add a separate explicit Graph-qualified configuration.

- Compile and allocate the complete fixed-capacity fleet, physical banks,
  symmetric exports, route buffers, flags and communication events before
  capture. Reject cold capture before staging inputs.
- Advance the local epoch with a GPU kernel each invocation. Capture must not
  freeze a Python epoch value and thereby accept stale ready flags. Copy the
  device epoch into candidate/identity Plan metadata before agreement/commit.
- Keep source publication as an NVSHMEM stream barrier. Remove the host stream
  synchronization only on the qualified path; branch communication streams
  from a captured event after publication.
- Join both communication branches back into the main stream after compute.
  This completes the captured DAG and retires no-work transfers as well as
  consumers. Same-stream Graph launches serialize complete invocations before
  reuse of q, slot maps, exports and physical banks.
- Keep delayed full-leg publication and acquire/proxy visibility unchanged.
  Weight version and layout are fixed for a workspace; source masks and
  executing slot identities may change without reallocating addresses.
- Replay with changing hot experts, replica receivers/slots, ragged valid rows,
  global tails and all-masked transitions. Require exact equality to the
  original backend, CPU Plan fields in test inspection, unchanged addresses,
  monotonic device epochs, correct complete-leg bytes and no stale outputs.
- Run EP2 first and EP8 after that passes. Store actual capture/replay evidence
  and commit qualification separately from later benefit gating or latency
  optimization.
