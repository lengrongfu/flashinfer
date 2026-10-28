# MegaFlux replica admission implementation plan

**Goal:** Avoid unprofitable replicas under the measured H200 serving workload while retaining large, strongly skewed hotspot gains.

**Architecture:** Optional calibrated integer-nanosecond admission policy in the existing device planner. A cooperative preflight checks owner block peak/mean and an optimistic savings bound before serial search. After compaction, compare actual critical-path block savings against fixed planning cost, committed-copy transport cost and a safety margin; rejected plans emit owner descriptors. No forward host reads or added collective branches. Legacy policy remains selectable for A/B comparison and uncalibrated hardware retains existing behavior.

**Tech Stack:** CUDA C++, Python, torch distributed, vLLM 0.32, EP8 H200.

1. Preserve dirty sources and prior evidence. Add validated calibration values and wire optional policy through backend/config/adapter without changing low-level default oracle semantics.
2. Test balanced reject, strong hotspot accept, insufficient savings reject, copy-cost sensitivity, equality reject and graph replay changing decisions. Run existing fieldwise planner/collective/graph tests over SSH.
3. Run fixed single-layer owner/legacy/admission sweeps for both capacities; exact outputs must agree and large-hotspot benefit must remain. Freeze calibrated values before serving measurements.
4. Run native/admission/native independent restart brackets with three seeds, for capacities 512 (.85) and 1024 (.90), profiles 4096/64 and 7168/128, output64 and same preconditioning. Retain legacy and owner-only prior results as historical references, not fresh brackets. Record actual admitted copies in separate untimed diagnostics; no probes during timing.
5. Verify raw counts, failures, latency and throughput, bracket drift, source hashes and restored diagnostics. Preserve all attempts, archive evidence and update the serving guide. Report failure limitations and actual scope without claiming production stability.

Calibration seed from previous controlled sweeps (not fitted to new serving outcomes): block8us, fixed3300us, margin300us, per-copy transport computed from exact expert bytes at188GB/s; peak/mean minimum3. These are explicit measured-geometry parameters, not a universal cost model. Controlled sweep can invalidate them before freeze; record changes.

2026-10-26 stability continuation:

- Preserve the controlled lifetime experiment and reversal before changing sources. Retaining the raw Graph plus explicit instantiation passed 1152 full-output requests; reverting construction reproduced a host `cuGraphLaunch` segfault in the second pair. Both modes had identical per-rank memory budgets. This supports a lifetime workaround; it does not prove the driver's internal fault mechanism.
- The retained-template candidate passed the eight-rank exact-output replay regression, but its clean full-model run reproduced the same host segfault in the third 7168/c128 case. The candidate was reverted and is not a fix. The stability chain stopped before 1024; no stability gate or formal campaign was created.
- Freeze the changed sources, then validate clean, uninstrumented serving at both 512/.85 and 1024/.90 with three distinct-seed pairs each. Require all output lengths, request counts and correctness checks before recording a stability gate.
- Only after that evidence exists, prepare and run a new native/MegaFlux/native campaign with three independent restart brackets per capacity. Failed earlier campaigns and diagnostic timings remain excluded. Collect all-layer admission activity separately from timed requests, and retain raw data and source hashes.
