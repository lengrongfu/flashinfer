# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Audit saved long-context attempts without launching or contacting a service."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text())


def percentile(values, p):
    values = sorted(values)
    at = (len(values) - 1) * p
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def audit_benchmark(path):
    data = read(path)
    command = read(path.with_suffix(".command.json"))
    expected = {
        key: int(command[command.index(flag) + 1])
        for key, flag in (
            ("count", "--num-prompts"),
            ("input", "--random-input-len"),
            ("output", "--random-output-len"),
        )
    }
    problems = []

    def check(condition, label):
        if not condition:
            problems.append(label)

    n = expected["count"]
    check(data["completed"] == n and data["failed"] == 0, "request completion")
    for key in ("input_lens", "output_lens", "ttfts", "latencies", "itls", "errors"):
        check(len(data[key]) == n, f"{key} count")
    check(not any(data["errors"]), "request errors")
    successful = [i for i, error in enumerate(data["errors"]) if not error]
    for key in ("input", "output"):
        lengths = data[f"{key}_lens"]
        check(all(v == expected[key] for v in lengths), f"{key} lengths")
        check(
            sum(lengths[i] for i in successful) == data[f"total_{key}_tokens"],
            f"{key} total",
        )
    check(math.isfinite(data["duration"]) and data["duration"] > 0, "duration")
    for key in ("ttfts", "latencies"):
        check(all(math.isfinite(v) and v >= 0 for v in data[key]), key)
    check(
        all(math.isfinite(v) and v >= 0 for row in data["itls"] for v in row),
        "inter-token latencies",
    )
    check(
        math.isclose(
            sum(data["output_lens"][i] for i in successful) / data["duration"],
            data["output_throughput"],
            rel_tol=1e-9,
        ),
        "throughput arithmetic",
    )
    if len(data["latencies"]) == len(data["ttfts"]) == len(data["output_lens"]):
        tpots = [
            (data["latencies"][i] - data["ttfts"][i]) / (data["output_lens"][i] - 1)
            for i in successful
            if data["output_lens"][i] > 1
        ]
        for label, values in (
            ("ttft", [data["ttfts"][i] for i in successful]),
            ("tpot", tpots),
            ("e2el", [data["latencies"][i] for i in successful]),
        ):
            for p in (50, 95, 99):
                check(
                    bool(values)
                    and math.isclose(
                        percentile(values, p / 100) * 1000,
                        data[f"p{p}_{label}_ms"],
                        rel_tol=1e-8,
                        abs_tol=1e-5,
                    ),
                    f"p{p}_{label} arithmetic",
                )
    result = {
        key: value for key, value in data.items() if not isinstance(value, (list, dict))
    }
    result.update(
        expected=expected, audit_problems=problems, complete_workload=not problems
    )
    for key in ("input_lens", "output_lens", "ttfts", "latencies", "errors"):
        result[key] = data[key]
    result["inter_token_sample_counts"] = [len(row) for row in data["itls"]]
    result["raw_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def audit_attempt(root):
    result = dict(path=str(root), status=read(root / "status.json"))
    result["lifetime_source_guard_completed"] = result["status"]["stage"] == "ALL_DONE"
    result["benchmarks"] = {
        path.stem: audit_benchmark(path)
        for path in sorted(root.glob("*.json"))
        if path.stem.startswith(("traffic-", "prewarm-"))
        and not path.stem.endswith((".audit", ".command"))
    }
    result["semantics"] = {}
    for path in sorted(root.glob("semantic-*.json")):
        if path.stem.endswith(".command"):
            continue
        rows = read(path)
        result["semantics"][path.stem] = dict(
            total=len(rows),
            passed=sum(row["passed"] for row in rows),
            failed=[row["name"] for row in rows if not row["passed"]],
        )
    hashes = read(root / "source-sha256.json")
    result["source_snapshot_count"] = len(hashes)
    result["source_snapshot_mismatches"] = [
        name
        for name, expected in hashes.items()
        if hashlib.sha256(
            (root / "source-snapshot" / name.lstrip("/")).read_bytes()
        ).hexdigest()
        != expected
    ]
    lines = (root / "server.log").read_text(errors="replace").splitlines()
    shutdown = next(
        (i for i, line in enumerate(lines) if "[shutdown]" in line), len(lines)
    )
    before = [line for line in lines[:shutdown] if " ERROR " in line]
    after = [line for line in lines[shutdown:] if " ERROR " in line]
    out_of_order = [line for line in lines if "out of order" in line.lower()]
    warnings = [line for line in lines if "WARNING" in line or "UserWarning" in line]
    # Keep the audit readable after a failure fans out across all DP workers.
    # The complete server.log is retained alongside this summary.
    excerpt_limit = 200
    result["log"] = dict(
        shutdown_start_line=shutdown + 1 if shutdown < len(lines) else None,
        diagnostic_excerpt_limit=excerpt_limit,
        error_line_count_before_shutdown=len(before),
        error_line_count_during_shutdown=len(after),
        out_of_order_line_count=len(out_of_order),
        warning_line_count=len(warnings),
        error_lines_before_shutdown=before[:excerpt_limit],
        error_lines_during_shutdown=after[:excerpt_limit],
        out_of_order_warning_lines=out_of_order[:excerpt_limit],
        warnings=warnings[:excerpt_limit],
        backend_lines=[
            line for line in lines if "Using " in line and "Experts" in line
        ],
        kv_cache_lines=[line for line in lines if "Available KV cache memory:" in line],
    )
    telemetry = []
    for line in (root / "telemetry.csv").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        telemetry.append([float(re.sub(r"[^0-9.]", "", s)) for s in line.split(",")])
    result["telemetry"] = dict(
        samples_per_gpu=dict(Counter(int(row[0]) for row in telemetry)),
        peak_memory_mib=max(row[1] for row in telemetry),
        max_temperature_c=max(row[3] for row in telemetry),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {path.name: audit_attempt(path) for path in args.attempts}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, attempt in result.items():
        print(name, attempt["status"]["stage"])
        for case, data in attempt["benchmarks"].items():
            print(
                case,
                data["completed"],
                data["output_throughput"] if data["complete_workload"] else "INVALID",
                data["audit_problems"],
            )
    if any(
        attempt["status"]["stage"] != "ALL_DONE"
        or attempt["source_snapshot_mismatches"]
        or any(data["audit_problems"] for data in attempt["benchmarks"].values())
        for attempt in result.values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
