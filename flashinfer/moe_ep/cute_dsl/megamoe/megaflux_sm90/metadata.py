# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Single-node NVLink metadata exchange with captured, device-only publication."""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

from flashinfer.moe_ep.megaflux_plan import checked_nbytes


@lru_cache(maxsize=1)
def metadata_module():
    from flashinfer.jit.core import gen_jit_spec

    path = Path(__file__).with_suffix(".cu")
    root = path.parents[5]
    includes = [str(root / p) for p in ("csrc", "include") if (root / p).is_dir()]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return gen_jit_spec(
        f"megaflux_metadata_{digest}", [path], extra_include_paths=includes
    ).build_and_load()


class NvlinkMetadata:
    """One symmetric arena per planner; one serialized invocation per arena.

    All ranks execute begin, statistics, vote and routes in that order. The
    entry barrier retires every reader of the preceding invocation. Each
    later barrier publishes a disjoint section before peer loads. Thus no
    sender can overwrite a section while a slower receiver still reads it.
    All barriers use the caller's stream and are captured with the kernels.
    """

    def __init__(self, config, *, device, group=None):
        from nvshmem import core, bindings
        import torch.distributed as dist

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("allocate NVLink metadata before graph capture")
        self.config = config
        self.rank = dist.get_rank(group)
        if (core.my_pe(), core.n_pes()) != (
            self.rank,
            config.world_size,
        ) or dist.get_world_size(group) != config.world_size:
            raise ValueError("metadata requires the matching NVSHMEM EP rank order")
        self.module = metadata_module()
        E, P, S, A = (
            config.num_logical_experts,
            config.world_size,
            config.physical_slot_capacity,
            config.max_tokens_per_source * config.top_k,
        )
        # int64 counts/status, fallback/hash, then route counts and int32 words.
        words_at = E + 4 + P * S
        nbytes = checked_nbytes(
            checked_nbytes(words_at, 8) + checked_nbytes(P * S * A, 4), 1
        )
        self.root = core.tensor((nbytes,), dtype=torch.uint8)
        buf, logical_bytes, _ = core.tensor_get_buffer(self.root)
        if logical_bytes != nbytes or buf.size < nbytes:
            raise RuntimeError("symmetric metadata arena capacity mismatch")
        addresses = [bindings.ptr(buf.handle, r) for r in range(P)]
        unavailable = torch.tensor(
            [any(not p for p in addresses)], dtype=torch.int32, device=device
        )
        dist.all_reduce(unavailable, op=dist.ReduceOp.MAX, group=group)
        if unavailable.item():
            core.free_tensor(self.root)
            self.root = None
            raise RuntimeError("NVLink metadata requires all EP peers to be mapped")
        self.peers = torch.tensor(addresses, dtype=torch.int64, device=device)
        header = self.root[: words_at * 8].view(torch.int64)
        self.counts = header[:E]
        self.status = header[E : E + 1].view(torch.int32)[:1]
        self.fallback = header[E + 1 : E + 2]
        self.hash = header[E + 2 : E + 4]
        self.hash.zero_()
        self.send_counts = header[E + 4 :].view(P, S)
        self.send_words = self.root[words_at * 8 :].view(torch.int32).view(P, S, A)

    @staticmethod
    def _barrier():
        from nvshmem import core

        core.barrier_all(stream=torch.cuda.current_stream())

    def begin(self):
        self._barrier()

    def statistics(self, counts, invalid):
        self._barrier()
        self.module.statistics(self.peers, counts, invalid)

    def vote(self, fallback, hashes, *, check_hash):
        self.fallback.copy_(fallback)
        self._barrier()
        self.module.vote(
            self.peers, fallback, hashes, self.config.num_logical_experts, check_hash
        )

    def routes(self, counts, words):
        self._barrier()
        self.module.routes(
            self.peers, counts, words, self.config.num_logical_experts, self.rank
        )

    def close(self):
        from nvshmem import core

        if self.root is not None:
            self._barrier()
            torch.cuda.current_stream().synchronize()
            core.free_tensor(self.root)
            self.root = None
