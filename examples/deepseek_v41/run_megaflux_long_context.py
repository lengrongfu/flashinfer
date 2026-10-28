# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""One isolated service lifetime with matched accuracy and long-output workloads."""

import argparse
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--backend", choices=("native", "nvlink"), required=True)
    parser.add_argument(
        "--short", action="store_true", help="512-output restart control"
    )
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.concurrency < 1:
        parser.error("use a single directory name and positive concurrency")
    root = args.root / args.name
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("MEGAFLUX_") or key in (
            "LD_PRELOAD",
            "CUDA_LAUNCH_BLOCKING",
            "NCCL_GRAPH_REGISTER",
        ):
            env.pop(key)
    env.update(
        PYTHONPATH=f"{repo}:/tmp/megaflux-deps",
        VLLM_ENGINE_READY_TIMEOUT_S="1800",
        NVSHMEM_DISABLE_CUDA_VMM="1",
        NVSHMEM_SYMMETRIC_SIZE="16G",
        FLASHINFER_MOE_EP_MEGAFLUX_CUDA_GRAPH="1",
        FLASHINFER_MOE_EP_MEGAFLUX_METADATA="nvlink",
        FLASHINFER_MOE_EP_MEGAFLUX_MIN_REPLICA_ROUTES="12289",
        FLASHINFER_MOE_EP_MEGAFLUX_ADMISSION="1",
        NCCL_GRAPH_MIXING_SUPPORT="1",
    )
    server = None
    done = threading.Event()

    def state(stage, **data):
        result = dict(stage=stage, time=time.time(), **data)
        (root / "status.json").write_text(json.dumps(result, indent=2) + "\n")
        with (root / "events.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)

    def telemetry():
        while not done.wait(10):
            with (root / "telemetry.csv").open("a") as f:
                f.write(f"# {time.time()}\n")
                f.flush()
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw",
                        "--format=csv,noheader",
                    ],
                    stdout=f,
                    check=False,
                )

    def execute(name, command, timeout=14400):
        (root / f"{name}.command.json").write_text(json.dumps(command, indent=2) + "\n")
        state(name)
        with (root / f"{name}.log").open("w") as f:
            result = subprocess.run(
                command,
                cwd=repo,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}")

    def accuracy(name, phase="all"):
        execute(
            name,
            [
                "python3",
                "examples/deepseek_v41/measure_megaflux_accuracy.py",
                "--dataset",
                str(args.root / "gsm8k-test.jsonl"),
                "--fixtures",
                str(args.root / "accuracy-fixtures.json.gz"),
                "--output",
                str(root / name),
                "--phase",
                phase,
            ],
        )

    def benchmark(length, output, count, name):
        command = [
            "vllm",
            "bench",
            "serve",
            "--backend",
            "vllm",
            "--base-url",
            "http://127.0.0.1:8000",
            "--endpoint",
            "/v1/completions",
            "--model",
            "/gpfs/rd/models/DeepSeek-V4.1-Flash-1017/",
            "--served-model-name",
            "deepseek-v41-megaflux",
            "--tokenizer-mode",
            "deepseek_v41",
            "--dataset-name",
            "random",
            "--random-input-len",
            str(length),
            "--random-output-len",
            str(output),
            "--random-range-ratio",
            "0",
            "--num-prompts",
            str(count),
            "--max-concurrency",
            str(args.concurrency),
            "--num-warmups",
            "0",
            "--ready-check-timeout-sec",
            "0",
            "--ignore-eos",
            "--seed",
            "42",
            "--temperature",
            "0",
            "--save-result",
            "--save-detailed",
            "--result-dir",
            str(root),
            "--result-filename",
            name + ".json",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--metric-percentiles",
            "50,95,99",
            "--disable-tqdm",
        ]
        execute(name, command)
        data = json.loads((root / f"{name}.json").read_text())
        audit = dict(
            expected=count,
            completed=data["completed"],
            failed=data["failed"],
            input_count=len(data["input_lens"]),
            output_count=len(data["output_lens"]),
            complete_inputs=sum(n == length for n in data["input_lens"]),
            complete_outputs=sum(n == output for n in data["output_lens"]),
            total_output_tokens=sum(data["output_lens"]),
        )
        (root / f"{name}.audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        if (
            any(
                audit[key] != count
                for key in (
                    "completed",
                    "input_count",
                    "output_count",
                    "complete_inputs",
                    "complete_outputs",
                )
            )
            or audit["failed"]
        ):
            raise RuntimeError(f"incomplete workload: {name}: {audit}")
        state(
            "case_passed",
            name=name,
            **audit,
            **{
                k: data.get(k)
                for k in ("duration", "output_throughput", "p95_ttft_ms", "p95_tpot_ms")
            },
        )

    try:
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        free = [
            int(v)
            for v in subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines()
        ]
        if active or len(free) != 8 or min(free) < 130000:
            raise RuntimeError("requires eight idle H200 GPUs; no service started")
        paths = list(
            (repo / "flashinfer/moe_ep/cute_dsl/megamoe/megaflux_sm90").glob("*.py")
        )
        paths += list(
            (repo / "flashinfer/moe_ep/cute_dsl/megamoe/megaflux_sm90").glob("*.cu")
        )
        paths += list(
            (repo / "flashinfer/moe_ep/backends/mega/kernel/sm90").glob(
                "*megaflux*/*.py"
            )
        )
        paths += [
            repo / "flashinfer/moe_ep/megaflux_plan.py",
            Path(__file__).resolve(),
            repo / "examples/deepseek_v41/measure_megaflux_accuracy.py",
            Path(
                "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/flashinfer_moe_ep.py"
            ),
        ]
        paths = [p for p in paths if not p.name.startswith("._")]
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        (root / "source-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
        for p in paths:
            destination = root / "source-snapshot" / p.relative_to("/")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, destination)
        command = [
            "vllm",
            "serve",
            "/gpfs/rd/models/DeepSeek-V4.1-Flash-1017/",
            "--tensor-parallel-size",
            "1",
            "--data-parallel-size",
            "8",
            "--data-parallel-size-local",
            "8",
            "--enable-expert-parallel",
            "--gpu-memory-utilization",
            "0.85",
            "--enforce-eager",
            "--trust-remote-code",
            "--max-model-len",
            "65536",
            "--max-num-batched-tokens",
            "512",
            "--max-num-seqs",
            "32",
            "--served-model-name",
            "deepseek-v41-megaflux",
            "--reasoning-parser",
            "deepseek_v41",
        ]
        if args.backend == "nvlink":
            command += [
                "--kernel-config",
                '{"moe_backend":"flashinfer_moe_ep_megaflux_sm90"}',
            ]
        (root / "server.command.json").write_text(json.dumps(command, indent=2) + "\n")
        (root / "server.env.json").write_text(
            json.dumps(
                {
                    k: v
                    for k, v in env.items()
                    if k.startswith(("NVSHMEM_", "NCCL_", "FLASHINFER_", "VLLM_"))
                    or k == "PYTHONPATH"
                },
                indent=2,
            )
            + "\n"
        )
        with (root / "server.log").open("w") as log:
            server = subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (root / "server.pid").write_text(str(server.pid) + "\n")
        state("starting", pid=server.pid)
        threading.Thread(target=telemetry, daemon=True).start()
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode}")
            try:
                urllib.request.urlopen(
                    "http://127.0.0.1:8000/health", timeout=2
                ).close()
                break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError("service readiness timeout")
        execute(
            "semantic-before",
            [
                "python3",
                "examples/deepseek_v41/verify_vllm_megaflux_sm90.py",
                "--output",
                str(root / "semantic-before.json"),
            ],
        )
        accuracy("accuracy-before")
        benchmark(4096, 128, args.concurrency, "prewarm-4096-out128")
        for length in (16384, 32768):
            output = 512 if args.short else length
            benchmark(
                length,
                output,
                args.concurrency,
                f"traffic-in{length}-out{output}-c{args.concurrency}",
            )
        accuracy("accuracy-after", phase="logprobs")
        execute(
            "semantic-after",
            [
                "python3",
                "examples/deepseek_v41/verify_vllm_megaflux_sm90.py",
                "--output",
                str(root / "semantic-after.json"),
            ],
        )
        if any(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest
            for p, digest in hashes.items()
        ):
            raise RuntimeError("source changed during service lifetime")
        state("ALL_DONE")
    except BaseException as exc:
        state("FAILED", error=repr(exc))
        raise
    finally:
        done.set()
        if server is not None:
            with suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=20)


if __name__ == "__main__":
    main()
