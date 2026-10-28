# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity MoE graph replay for an eager, serialized serving caller."""

import torch

from flashinfer.moe_ep.tensors import MoEEpTensors


class MegaFluxGraphReplay:
    """Capture once on every EP rank, then update live inputs before replay.

    Only the MoE is captured; the caller may remain eager. Padding is masked
    with expert ID -1, so ranks can have different live token counts, including
    zero. Planning, routes, replica identities and epochs are recomputed on GPU
    on every replay. Returned output is borrowed until the next invocation.
    Calls must be serialized on the caller's stream, as for the underlying
    layer. Close on all ranks before destroying that layer.
    """

    @torch.inference_mode()
    def __init__(self, layer, *, max_tokens, hidden_size, top_k, device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare MegaFlux replay before outer graph capture")
        self._layer = layer
        self._inputs = MoEEpTensors(
            hidden_states=torch.zeros(
                (max_tokens, hidden_size), dtype=torch.bfloat16, device=device
            ),
            topk_ids=torch.full(
                (max_tokens, top_k), -1, dtype=torch.int64, device=device
            ),
            topk_weights=torch.zeros(
                (max_tokens, top_k), dtype=torch.float32, device=device
            ),
        )
        # Warm the exact padded shape before capture, including quantization
        # and collective kernels. All EP ranks must construct this together.
        layer.warmup(self._inputs)
        self._stream = torch.cuda.Stream(device=device)
        self._stream.wait_stream(torch.cuda.current_stream(device))
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, stream=self._stream):
            self._output = layer.forward(self._inputs, return_workspace_view=True)
        torch.cuda.current_stream(device).wait_stream(self._stream)

    @torch.inference_mode()
    def __call__(self, hidden_states, topk_ids, topk_weights):
        if self._graph is None:
            raise RuntimeError("MegaFlux replay has been closed")
        target = self._inputs
        n = hidden_states.shape[0]
        capacity, hidden = target.hidden_states.shape
        top_k = target.topk_ids.shape[1]
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[1] != hidden
            or n > capacity
            or topk_ids.shape != (n, top_k)
            or topk_weights.shape != (n, top_k)
        ):
            raise ValueError("inputs differ from the captured MegaFlux capacity")
        for source, destination, dtypes in (
            (hidden_states, target.hidden_states, (torch.bfloat16,)),
            (topk_ids, target.topk_ids, (torch.int32, torch.int64)),
            (topk_weights, target.topk_weights, (torch.float32,)),
        ):
            if source.device != destination.device or source.dtype not in dtypes:
                raise ValueError(
                    f"expected MegaFlux input dtype {dtypes} on {destination.device}, "
                    f"got {source.dtype} on {source.device}"
                )
            destination[:n].copy_(source)
        # Clear IDs after every update: shrinking batches must not replay old
        # routes. Other tail values are unreachable through the masked IDs.
        target.topk_ids[n:].fill_(-1)
        self._graph.replay()
        return self._output[:n]

    def close(self):
        if self._graph is not None:
            torch.cuda.synchronize(self._inputs.hidden_states.device)
            self._graph = None
            self._output = self._inputs = self._stream = self._layer = None
