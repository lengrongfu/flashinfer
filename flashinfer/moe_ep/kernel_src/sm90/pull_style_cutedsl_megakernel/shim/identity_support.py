# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Narrow public bridge for maintained SM90 Plan communication adapters."""


def hopper_mxfp4_token_comm_type():
    """Expose the communication extension base without bypassing the shim."""
    from src.token_comm import TokenInPullTokenBackPush

    return TokenInPullTokenBackPush


def hopper_mxfp4_identity_regions(frontend):
    """Return typed views of the existing receiver metadata ABI.

    Layout interpretation remains in the shim. These views do not allocate
    storage and remain valid only for the current compiled workspace.
    """
    import math
    import torch

    mega = frontend._mega
    if mega is None:
        raise RuntimeError("prepare the frontend before requesting Plan regions")
    result = {}
    for name, dtype, itemsize in (
        ("expert_recv_count", torch.int64, 8),
        ("expert_recv_count_sum", torch.int64, 8),
        ("src_token_topk_idx", torch.int32, 4),
    ):
        spec = mega.kernel._shared_region_by_name[name]
        offset = mega.kernel._shared_offsets[name]
        nbytes = math.prod(spec.shape) * itemsize
        result[name] = (
            mega.shared_workspace.narrow(0, offset, nbytes).view(dtype).view(spec.shape)
        )
    return result


def hopper_mxfp4_pipeline_support():
    """Expose narrow epilogue dependencies for an attributed maintained loop.

    The vendor implementation remains unchanged. The maintained adapter needs
    phase-local residual reads because the original eagerly reads both legs.
    """
    from moe_hopper_fp8 import epilogue_fp8_swapab as epi

    names = (
        "SwapABFp8GluEpilogue",
        "BlockPhase",
        "EpiWarpCount",
        "WarpThreadCount",
        "GpuReleaseFlagBatchTracker",
        "consume_initial_pingpong_work",
        "consume_next_pingpong_work",
        "_red_or_release_gpu_b64",
        "red_add_release_gpu_s32",
        "iket",
    )
    return {name: getattr(epi, name) for name in names}


def hopper_mxfp4_pipeline_region_specs(capacity, world_size):
    import cutlass
    from moe_nvfp4_swapab.megamoe_kernel import _RegionSpec

    return [
        _RegionSpec(name, cutlass.Int64, shape, 16)
        for name, shape in (
            ("megaflux_ready", (2, capacity)),
            ("megaflux_epoch", (1,)),
            ("megaflux_telemetry", (7,)),
            ("megaflux_chunks", (2, world_size)),
            ("megaflux_identity", (2, capacity, 6)),
            ("megaflux_retired", (capacity,)),
        )
    ]


def prepare_hopper_mxfp4_pipeline(workspace, bank):
    """Compile once and expose local flag views through the runtime workspace."""
    import math
    import torch
    from .hopper_mxfp4 import _build_mxfp4_inputs

    frontend = workspace._frontend
    frontend.prepare_launch(_build_mxfp4_inputs(workspace, *bank))
    mega = frontend._mega
    views = {}
    for short in ("ready", "epoch", "telemetry", "chunks", "identity", "retired"):
        name = "megaflux_" + short
        spec = mega.kernel._local_region_by_name[name]
        offset = mega.kernel._local_offsets[name]
        views[short] = (
            mega.local_workspace.narrow(0, offset, math.prod(spec.shape) * 8)
            .view(torch.int64)
            .view(spec.shape)
        )
    return views
