# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Epoch-qualified local ready acquisition and progress telemetry."""

from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def acquire_leg(ready, epoch, wait_counter=None, *, loc=None, ip=None):
    operands = [ready.ir_value(loc=loc, ip=ip), epoch.ir_value(loc=loc, ip=ip)]
    if wait_counter is None:
        asm = (
            "{ .reg .u64 e,v; .reg .pred p; ld.global.u64 e, [$1]; "
            "MF_WAIT: ld.acquire.sys.global.u64 v, [$0]; "
            "setp.ne.u64 p,v,e; @p bra MF_WAIT; fence.proxy.async.global; }"
        )
        constraints = "l,l"
    else:
        operands.append(wait_counter.ir_value(loc=loc, ip=ip))
        # Only a real failed acquire starts the timer or updates telemetry.
        # Every waiting caller contributes, so MAX does not assume warp timing.
        asm = (
            "{ .reg .u64 e,v,t,d,old; .reg .pred p; ld.global.u64 e, [$1]; "
            "ld.acquire.sys.global.u64 v, [$0]; setp.ne.u64 p,v,e; @!p bra MF_DONE; "
            "mov.u64 t, %globaltimer; "
            "MF_WAIT: ld.acquire.sys.global.u64 v, [$0]; "
            "setp.ne.u64 p,v,e; @p bra MF_WAIT; "
            "mov.u64 d, %globaltimer; sub.u64 d,d,t; "
            "atom.global.max.u64 old, [$2], d; "
            "MF_DONE: fence.proxy.async.global; }"
        )
        constraints = "l,l,l"
    llvm.inline_asm(
        None,
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def record_progress(address, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip)],
        "{ .reg .u64 t,old; mov.u64 t, %globaltimer; "
        "atom.global.cas.b64 old, [$0], 0, t; }",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# Call sites derive addresses from the runtime FC1-ready pointer plus static
# region offsets; model/workspace device addresses are never compile constants.
