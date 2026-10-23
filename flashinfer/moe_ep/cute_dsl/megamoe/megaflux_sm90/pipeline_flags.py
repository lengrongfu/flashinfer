# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Epoch-qualified local ready acquisition and progress telemetry."""

from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def acquire_leg(ready, epoch, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [ready.ir_value(loc=loc, ip=ip), epoch.ir_value(loc=loc, ip=ip)],
        "{ .reg .u64 e,v; .reg .pred p; ld.global.u64 e, [$1]; "
        "MF_WAIT: ld.acquire.sys.global.u64 v, [$0]; "
        "setp.ne.u64 p,v,e; @p bra MF_WAIT; fence.proxy.async.global; }",
        "l,l",
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
