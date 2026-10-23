# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Add the opt-in SM90 MegaFlux backend to an installed vLLM 0.32.x package.

This source patch keeps the H100 path explicit and does not alter other
FlashInfer MoE-EP backends. Run once in the serving environment before launch.
"""

from importlib.metadata import version
from pathlib import Path
import shutil
import tempfile


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if new in source:
        return
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one patch anchor in {path}, found {count}: {old!r}")
    path.write_text(source.replace(old, new))


def main() -> None:
    installed = version("vllm")
    if not installed.startswith("0.32."):
        raise RuntimeError(f"this integration patch targets vLLM 0.32.x, found {installed}")
    root = Path(__import__("vllm").__file__).resolve().parent
    kernel = root / "config/kernel.py"
    adapter = root / "model_executor/layers/fused_moe/flashinfer_moe_ep.py"
    oracle = root / "model_executor/layers/fused_moe/oracle/mxfp4.py"
    targets = (kernel, adapter, oracle)
    backup_root = Path(tempfile.mkdtemp(prefix="vllm-megaflux-sm90-backup-"))
    for path in targets:
        shutil.copy2(path, backup_root / path.name)

    replace_once(
        kernel,
        '    "flashinfer_moe_ep_mega_deep_gemm",\n',
        '    "flashinfer_moe_ep_mega_deep_gemm",\n'
        '    "flashinfer_moe_ep_megaflux_sm90",\n',
    )
    replace_once(
        kernel,
        'FLASHINFER_MOE_EP_DEEP_GEMM = "flashinfer_moe_ep_mega_deep_gemm"\n',
        'FLASHINFER_MOE_EP_DEEP_GEMM = "flashinfer_moe_ep_mega_deep_gemm"\n'
        'FLASHINFER_MOE_EP_MEGAFLUX_SM90 = "flashinfer_moe_ep_megaflux_sm90"\n',
    )
    replace_once(
        kernel,
        '        FLASHINFER_MOE_EP_DEEP_GEMM,\n',
        '        FLASHINFER_MOE_EP_DEEP_GEMM,\n'
        '        FLASHINFER_MOE_EP_MEGAFLUX_SM90,\n',
    )
    replace_once(
        kernel,
        '    if moe_backend != FLASHINFER_MOE_EP_DEEP_GEMM:\n',
        '    if moe_backend not in (FLASHINFER_MOE_EP_DEEP_GEMM, FLASHINFER_MOE_EP_MEGAFLUX_SM90):\n',
    )

    replace_once(
        adapter,
        '    FLASHINFER_MOE_EP_DEEP_GEMM,\n',
        '    FLASHINFER_MOE_EP_DEEP_GEMM,\n'
        '    FLASHINFER_MOE_EP_MEGAFLUX_SM90,\n',
    )
    replace_once(
        adapter,
        'FlashInferMoeEpKernel = Literal["cutedsl", "deep_gemm"]\n',
        'FlashInferMoeEpKernel = Literal["cutedsl", "deep_gemm", "megaflux_sm90"]\n',
    )
    replace_once(
        adapter,
        '_SUPPORTED_CAPABILITIES = frozenset({(10, 0), (10, 3)})\n',
        '_SUPPORTED_CAPABILITIES = frozenset({(9, 0), (10, 0), (10, 3)})\n',
    )
    replace_once(
        adapter,
        '    FLASHINFER_MOE_EP_DEEP_GEMM: FlashInferMoeEpBackendSpec(\n'
        '        kernel="deep_gemm",\n'
        '        weight_formats=frozenset({"mxfp4"}),\n'
        '    ),\n',
        '    FLASHINFER_MOE_EP_DEEP_GEMM: FlashInferMoeEpBackendSpec(\n'
        '        kernel="deep_gemm",\n'
        '        weight_formats=frozenset({"mxfp4"}),\n'
        '    ),\n'
        '    FLASHINFER_MOE_EP_MEGAFLUX_SM90: FlashInferMoeEpBackendSpec(\n'
        '        kernel="megaflux_sm90",\n'
        '        weight_formats=frozenset({"mxfp4"}),\n'
        '    ),\n',
    )
    replace_once(
        adapter,
        '    spec = flashinfer_moe_ep_backend_spec(moe.moe_backend)\n'
        '    unsupported: list[str] = []\n',
        '    spec = flashinfer_moe_ep_backend_spec(moe.moe_backend)\n'
        '    unsupported: list[str] = []\n'
        '    capability = current_platform.get_device_capability()\n'
        '    if spec.kernel == "megaflux_sm90" and (capability is None or (capability.major, capability.minor) != (9, 0)):\n'
        '        unsupported.append("SM90 MegaFlux requires compute capability 9.0")\n'
        '    if spec.kernel == "cutedsl" and capability is not None and capability.major == 9:\n'
        '        unsupported.append("the CuTeDSL MoE-EP backend is SM100/SM103; use the SM90 MegaFlux backend")\n',
    )
    replace_once(
        adapter,
        '    if spec.kernel == "deep_gemm" and vllm_config.parallel_config.enable_eplb:\n'
        '        unsupported.append("EPLB")\n',
        '    if spec.kernel in ("deep_gemm", "megaflux_sm90") and vllm_config.parallel_config.enable_eplb:\n'
        '        unsupported.append("EPLB")\n',
    )
    # Add imports to the API loader using a stable anchor in its guarded import.
    replace_once(
        adapter,
        '                Nvfp4CutedslMegaMoeConfig,\n'
        '            )\n',
        '                Nvfp4CutedslMegaMoeConfig,\n'
        '            )\n'
        '            from flashinfer.moe_ep.megaflux_plan import PlanConfig\n'
        '            from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_megaflux_graph import (\n'
        '                Sm90_Mxfp4_MegaFluxGraphConfig,\n'
        '            )\n',
    )
    replace_once(
        adapter,
        '        self.Nvfp4CutedslMegaMoeConfig = Nvfp4CutedslMegaMoeConfig\n',
        '        self.Nvfp4CutedslMegaMoeConfig = Nvfp4CutedslMegaMoeConfig\n'
        '        self.PlanConfig = PlanConfig\n'
        '        self.Sm90_Mxfp4_MegaFluxGraphConfig = Sm90_Mxfp4_MegaFluxGraphConfig\n',
    )
    replace_once(
        adapter,
        '        self._top_k = moe.experts_per_token\n',
        '        self._top_k = moe.experts_per_token\n'
        '        self._topk_indices_dtype = (\n'
        '            torch.int64 if spec.kernel == "megaflux_sm90" else torch.int32\n'
        '        )\n',
    )
    replace_once(
        adapter,
        '    def topk_indices_dtype(self) -> torch.dtype:\n'
        '        return torch.int32\n',
        '    def topk_indices_dtype(self) -> torch.dtype:\n'
        '        return self._topk_indices_dtype\n',
    )
    replace_once(
        adapter,
        '        epilogue = cast(FlashInferMoeEpEpilogue, self._epilogue)\n'
        '        return self._moe_ep_tensors_cls(\n',
        '        epilogue = cast(FlashInferMoeEpEpilogue, self._epilogue)\n'
        '        if topk_ids.dtype != self._topk_indices_dtype:\n'
        '            topk_ids = topk_ids.to(self._topk_indices_dtype)\n'
        '        return self._moe_ep_tensors_cls(\n',
    )
    replace_once(
        adapter,
        '            torch.arange(self._top_k, dtype=torch.int32, device=self._device)\n',
        '            torch.arange(self._top_k, dtype=self.topk_indices_dtype, device=self._device)\n',
    )
    replace_once(
        adapter,
        '        (fc1_weight, fc1_sf), (fc2_weight, fc2_sf) = self._mega_layer._transformed\n'
        '        return (\n'
        '            _expert_major_bytes(fc1_weight),\n'
        '            _expert_major_bytes(fc1_sf),\n'
        '            _expert_major_bytes(fc2_weight),\n'
        '            _expert_major_bytes(fc2_sf),\n'
        '        )\n',
        '        transformed = self._mega_layer._transformed\n'
        '        if self._topk_indices_dtype is torch.int64:\n'
        '            # MegaFlux stores each FC leg as (weight, folded_offset,\n'
        '            # activation_placeholder, residual_scale); EPLB is disabled\n'
        '            # for this backend, so expose the first two expert-major views.\n'
        '            fc1, fc2 = transformed\n'
        '            return (\n'
        '                _expert_major_bytes(fc1[0]),\n'
        '                _expert_major_bytes(fc1[1]),\n'
        '                _expert_major_bytes(fc2[0]),\n'
        '                _expert_major_bytes(fc2[1]),\n'
        '            )\n'
        '        (fc1_weight, fc1_sf), (fc2_weight, fc2_sf) = transformed\n'
        '        return (\n'
        '            _expert_major_bytes(fc1_weight),\n'
        '            _expert_major_bytes(fc1_sf),\n'
        '            _expert_major_bytes(fc2_weight),\n'
        '            _expert_major_bytes(fc2_sf),\n'
        '        )\n',
    )
    replace_once(
        adapter,
        '    if spec.kernel == "deep_gemm":\n'
        '        return FlashInferMoeEpWeights(w13, w2, w13_scale, w2_scale)\n',
        '    if spec.kernel in ("deep_gemm", "megaflux_sm90"):\n'
        '        return FlashInferMoeEpWeights(w13, w2, w13_scale, w2_scale)\n',
    )
    replace_once(
        adapter,
        '        if spec.kernel == "cutedsl":\n'
        '            # In-kernel fc2 reduce keeps its default (off); FlashInfer 0.7.1rc1\n',
        '        if spec.kernel == "megaflux_sm90":\n'
        '            if moe.routing_method is not RoutingMethodType.DeepseekV4:\n'
        '                raise ValueError("SM90 MegaFlux vLLM integration currently requires DeepseekV4 routing")\n'
        '            plan_config = api.PlanConfig(\n'
        '                num_logical_experts=moe.num_experts,\n'
        '                world_size=ep_group.world_size,\n'
        '                replica_budget=1,\n'
        '                max_tokens_per_source=moe.max_num_tokens,\n'
        '                top_k=moe.experts_per_token,\n'
        '            )\n'
        '            megakernel = api.Sm90_Mxfp4_MegaFluxGraphConfig(\n'
        '                intermediate_size=moe.intermediate_size,\n'
        '                top_k=moe.experts_per_token,\n'
        '                plan_config=plan_config,\n'
        '                selective_weight_copy=True,\n'
        '                gate_up_clamp=moe.swiglu_limit,\n'
        '                load_balance_mode="static",\n'
        '            )\n'
        '        elif spec.kernel == "cutedsl":\n'
        '            # In-kernel fc2 reduce keeps its default (off); FlashInfer 0.7.1rc1\n',
    )

    replace_once(
        oracle,
        '    FLASHINFER_MOE_EP_DEEP_GEMM = "FLASHINFER_MOE_EP_DEEP_GEMM"\n',
        '    FLASHINFER_MOE_EP_DEEP_GEMM = "FLASHINFER_MOE_EP_DEEP_GEMM"\n'
        '    FLASHINFER_MOE_EP_MEGAFLUX_SM90 = "FLASHINFER_MOE_EP_MEGAFLUX_SM90"\n',
    )
    replace_once(
        oracle,
        '    Mxfp4MoeBackend.FLASHINFER_MOE_EP_DEEP_GEMM,\n',
        '    Mxfp4MoeBackend.FLASHINFER_MOE_EP_DEEP_GEMM,\n'
        '    Mxfp4MoeBackend.FLASHINFER_MOE_EP_MEGAFLUX_SM90,\n',
    )
    replace_once(
        oracle,
        '        "flashinfer_moe_ep_mega_deep_gemm": [\n'
        '            Mxfp4MoeBackend.FLASHINFER_MOE_EP_DEEP_GEMM\n'
        '        ],\n',
        '        "flashinfer_moe_ep_mega_deep_gemm": [\n'
        '            Mxfp4MoeBackend.FLASHINFER_MOE_EP_DEEP_GEMM\n'
        '        ],\n'
        '        "flashinfer_moe_ep_megaflux_sm90": [\n'
        '            Mxfp4MoeBackend.FLASHINFER_MOE_EP_MEGAFLUX_SM90\n'
        '        ],\n',
    )
    print(f"Patched vLLM {installed} at {root}; backups: {backup_root}")




if __name__ == "__main__":
    main()
