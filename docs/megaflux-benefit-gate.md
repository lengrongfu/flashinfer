# GPU benefit gate

The online, pipeline and graph configs accept optional `benefit_costs`, a
`BenefitCosts(block_us, fixed_us, transfer_us, margin_us)` instance. All values
are finite nonnegative microseconds and block cost must be positive. There is
no uncalibrated default and the existing load-only behavior remains the default.

The GPU compares the owner-only maximum block count times `block_us` with the
candidate maximum times `block_us`, plus `fixed_us`, `transfer_us` and the
margin. Equality rejects. Transfer cost is fully exposed in this conservative
model. Calibration must include the actual geometry and actual implementation:
statistics/exchange, planning/materialization, export staging, synchronization,
and full-root padded traffic in the currently accepted transport. Charging only
active replicas would undercount its traffic. This model does not establish a
performance improvement by itself.

The rejection enters the same collective MAX as capacity and weight-version
failure, before route materialization and transport. One rejecting rank makes
all ranks commit the owner-only Plan. No per-invocation host tensor readback is
introduced. The gate's stable device `metrics` buffer contains original estimate,
candidate estimate (excluding margin), paid fixed/transfer cost, and local reject.
A rejection does not skip existing planner, rendezvous or conservative transfer
work; the paid cost remains visible. Gate-disabled and gate-rejected measurements
must still time that work end to end.

Acceptance tests: ordinary CUDA and captured/replayed decisions, equality
rejection, changing maxima and stable result addresses; EP8 collective asymmetric
rank-zero rejection compares committed q, physical slots, blocks and epoch to the
GPU identity Plan. These establish selection correctness, not calibrated benefit,
full forward gate acceptance or selective transport performance.
