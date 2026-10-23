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
