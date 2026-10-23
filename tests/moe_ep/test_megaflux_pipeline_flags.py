# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Device telemetry helper smoke test independent of persistent kernel tails."""

import pytest
import torch
import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from flashinfer.moe_ep.cute_dsl.megamoe.megaflux_sm90.pipeline_flags import (
    record_progress,
)


@cute.kernel
def _record_kernel(out: cute.Tensor):
    record_progress(out.iterator.toint())


@cute.jit
def _record(out: cute.Tensor, stream: cuda.CUstream):
    _record_kernel(out).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@pytest.mark.arch_hopper
def test_pipeline_progress_atomic():
    tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
    arg = from_dlpack(tensor, assumed_align=8)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(_record, arg, stream)
    compiled(arg, stream)
    torch.cuda.synchronize()
    assert tensor.item() > 0
