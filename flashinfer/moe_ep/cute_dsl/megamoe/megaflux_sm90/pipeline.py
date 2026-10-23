# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Independent W1/W2 streams and epoch-qualified full-leg consumers."""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

from .physical import PhysicalSlotFrontend
from .static import DeviceReplicaTransport


@lru_cache(maxsize=1)
def pipeline_module():
    from flashinfer.jit.core import gen_jit_spec

    path = Path(__file__).with_suffix(".cu")
    root = path.parents[5]
    includes = [
        str(root / name) for name in ("csrc", "include") if (root / name).is_dir()
    ]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return gen_jit_spec(
        f"megaflux_pipeline_{digest}", [path], extra_include_paths=includes
    ).build_and_load()


class PipelineState:
    def __init__(self, config, *, device):
        self.config = config
        self.module = pipeline_module()
        self.ready = self.epoch = self.telemetry = self.chunks = None
        self.identity = self.retired = None
        self.streams = [torch.cuda.Stream(device=device) for _ in range(2)]
        self.start = torch.cuda.Event()
        self.done = [torch.cuda.Event() for _ in range(2)]
        self.issued = False
        self.graph_mode = False

    def join(self):
        if self.issued:
            for event in self.done:
                torch.cuda.current_stream().wait_event(event)

    def control(self, action, *, leg=0, owner=0, delay_us=0):
        self.module.control(
            self.ready,
            self.epoch,
            self.telemetry,
            self.chunks,
            action,
            leg,
            owner,
            self.config.num_owned_experts_per_rank,
            delay_us,
        )


class PipelineFrontend(PhysicalSlotFrontend):
    def __init__(self, config, plan, state, reserved_clusters):
        self.state = state
        self.reserved_clusters = reserved_clusters
        super().__init__(config, plan)

    def _resolve_max_active_clusters(self, value):
        if value <= self.reserved_clusters:
            raise ValueError(
                "insufficient cluster capacity for independent communication"
            )
        self.persistent_clusters = value - self.reserved_clusters
        return self.persistent_clusters

    def _mega_compile_key(self):
        return (
            *super()._mega_compile_key(),
            "megaflux_pipeline_runtime_flags_v2",
            self.reserved_clusters,
        )

    def _construct_kernel(self, kernel_type, **kwargs):
        from .pipeline_weight import PipelineWeightMixin
        from .pipeline_epilogue import PipelineEpilogue

        from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
            hopper_mxfp4_pipeline_region_specs,
        )

        capacity, world = self.physical_slot_capacity, self.config.world_size

        class GatedKernel(PipelineWeightMixin, kernel_type):
            def _build_local_region_specs(self):
                return [
                    *super()._build_local_region_specs(),
                    *hopper_mxfp4_pipeline_region_specs(capacity, world),
                ]

            def _setup_attributes(self):
                super()._setup_attributes()
                epilogue = PipelineEpilogue.__new__(PipelineEpilogue)
                epilogue.__dict__.update(self.epilogue.__dict__)
                for name in ("ready", "epoch", "telemetry"):
                    value = (
                        self._local_offsets["megaflux_" + name]
                        - self._local_offsets["l1_arrival_count"]
                    )
                    self.__dict__["_mf_" + name + "_offset"] = value
                    epilogue.__dict__["_mf_" + name + "_offset"] = value
                for target in (self, epilogue):
                    target._mf_capacity = capacity
                    target._mf_owned = capacity - self._mf_replica_budget
                self.epilogue = epilogue

        kernel = super()._construct_kernel(GatedKernel, **kwargs)
        kernel._mf_replica_budget = capacity - self.num_owned_experts_per_rank

        return kernel


class PipelinedReplicaTransport(DeviceReplicaTransport):
    def __init__(self, config, bank, rank, state, backend_config):
        super().__init__(config, bank, rank)
        self.state, self.backend_config = state, backend_config

    def copy(self, plan):
        from nvshmem import core, bindings

        if torch.cuda.is_current_stream_capturing() and not self.state.graph_mode:
            raise RuntimeError("warm and validate pipeline before the Graph stage")
        state, c = self.state, self.config
        state.control(0)
        if not self.export_capacity:
            state.module.bind_identity(
                plan.slot_to_expert,
                state.epoch,
                state.retired,
                state.identity,
                self.rank,
                c.weight_version,
            )
            return
        # All PEs retire old local consumers and their receiver streams before
        # any owner overwrites exports still readable by another PE.
        self._rendezvous()
        state.module.bind_identity(
            plan.slot_to_expert,
            state.epoch,
            state.retired,
            state.identity,
            self.rank,
            c.weight_version,
        )
        for pieces, root in zip(self.pieces, self.exports, strict=True):
            self.module.weights(
                plan.q, plan.slot_to_expert, *pieces, root, self.rank, False
            )
        # Old compute/transfers are retired and every owner export is published.
        # No further cross-rank barrier runs on a communication stream.
        self._rendezvous()
        main = torch.cuda.current_stream()
        state.start.record(main)
        delays = self.backend_config.pipeline_delays_us
        for leg, stream in enumerate(state.streams):
            stream.wait_event(state.start)
            with torch.cuda.stream(stream):
                if delays[leg]:
                    state.control(1, leg=leg, delay_us=delays[leg])
                root, cache = self.exports[leg], self.caches[leg]
                src, size, _ = core.tensor_get_buffer(root)
                dst, cache_size, _ = core.tensor_get_buffer(cache)
                if (
                    src.size < size
                    or dst.size < cache_size
                    or cache_size != c.world_size * size
                ):
                    raise RuntimeError("symmetric pipeline root capacity mismatch")
                for owner in range(c.world_size):
                    bindings.getmem_on_stream(
                        dst.handle + owner * size,
                        src.handle,
                        size,
                        owner,
                        stream.cuda_stream,
                    )
                    core.quiet(stream=stream)
                    # This marker follows actual get completion, not enqueue.
                    state.control(2, leg=leg, owner=owner)
                pieces = self.pieces[leg]
                for part in range(3):
                    extra_delay = delays[part + 1] if leg == 1 and part > 0 else 0
                    if extra_delay:
                        state.control(1, leg=leg, delay_us=extra_delay)
                    state.module.unpack_part(
                        plan.q, plan.slot_to_expert, *pieces, cache, self.rank, part
                    )
                state.control(3, leg=leg)
                state.done[leg].record(stream)
        state.issued = True

    def _rendezvous(self):
        from nvshmem import core

        stream = torch.cuda.current_stream()
        core.barrier_all(stream=stream)
        if not self.state.graph_mode:
            stream.synchronize()

    def close(self):
        self.state.join()
        for stream in self.state.streams:
            stream.synchronize()
        super().close()
