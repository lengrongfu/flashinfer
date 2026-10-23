# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Explicit calibrated critical-path model; all invocation decisions stay on GPU."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class BenefitCosts:
    """Microsecond costs measured outside the forward path for this geometry.

    ``fixed_us`` includes statistics, planning, materialization, synchronization
    and staging. ``transfer_us`` charges the actual transport implementation,
    including padded full-root traffic for the conservative transport. Neither
    term disappears when the gate rejects a candidate. No default calibration
    is supplied: fitting a model is a benchmark responsibility.
    """

    block_us: float
    fixed_us: float
    transfer_us: float
    margin_us: float = 0.0

    def __post_init__(self):
        for name in ("block_us", "fixed_us", "transfer_us", "margin_us"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.block_us == 0:
            raise ValueError("block_us must be positive")


class BenefitGate:
    """Record [original, candidate, paid overhead, reject] on the device.

    The estimate conservatively exposes the entire transfer cost. Equality
    rejects. The caller combines this decision with collective safety fallback
    before committing routing; no tensor is inspected by Python.
    """

    def __init__(self, costs, *, device):
        self.costs = costs
        self.metrics = torch.empty(4, dtype=torch.float64, device=device)
        self.reject = torch.empty(1, dtype=torch.int32, device=device)

    def run(self, candidate, identity):
        c = self.costs
        self.metrics[0].copy_(identity.statistics[0].amax())
        self.metrics[0].mul_(c.block_us)
        self.metrics[1].copy_(candidate.statistics[0].amax())
        self.metrics[1].mul_(c.block_us).add_(c.fixed_us + c.transfer_us)
        self.metrics[2].fill_(c.fixed_us + c.transfer_us)
        self.reject.copy_(self.metrics[1:2] + c.margin_us >= self.metrics[0:1])
        self.metrics[3:4].copy_(self.reject)
        return self.reject
