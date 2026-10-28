# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compare saved answers and chosen-token log probabilities on identical inputs."""

import argparse
import gzip
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def quantile(values, p):
    values = sorted(values)
    at = (len(values) - 1) * p
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def reported_top1(row):
    # Use the server's rank to preserve its tie-breaking on equal logprobs.
    tokens = [key for key, value in row.items() if value["rank"] == 1]
    if len(tokens) != 1:
        raise ValueError("expected exactly one server-ranked top-1 token")
    return tokens[0]


def compare(left, right):
    left_summary = json.loads((left / "summary.json").read_text())
    right_summary = json.loads((right / "summary.json").read_text())
    if left_summary["fixture_sha256"] != right_summary["fixture_sha256"]:
        raise ValueError("fixtures differ")
    left_files = {p.name for p in left.glob("*.json.gz")}
    right_files = {p.name for p in right.glob("*.json.gz")}
    result = dict(
        left=str(left),
        right=str(right),
        fixture_sha256=left_summary["fixture_sha256"],
        only_left=sorted(left_files - right_files),
        only_right=sorted(right_files - left_files),
        answers={},
        logprobs={},
        failed_observations=[],
    )
    for name in sorted(left_files & right_files):
        a, b = read(left / name), read(right / name)
        if a["request"] != b["request"]:
            raise ValueError(f"requests differ: {name}")
        if not a["transport_ok"] or not b["transport_ok"]:
            result["failed_observations"].append(name)
            continue
        if a["kind"] != "logprobs":
            category = result["answers"].setdefault(a["kind"], [])
            category.append(
                dict(
                    name=a["name"],
                    left_correct=a["correct"],
                    right_correct=b["correct"],
                    answer_agrees=a["extracted"] == b["extracted"],
                    left_answer=a["extracted"],
                    right_answer=b["extracted"],
                )
            )
            continue
        ac, bc = a["response"]["choices"][0], b["response"]["choices"][0]
        ids = ac["prompt_token_ids"]
        if ids != bc["prompt_token_ids"]:
            raise ValueError(f"teacher-forced tokens differ: {name}")
        la, lb, matches, exact = [], [], 0, 0
        for i, token in enumerate(ids[1:], start=1):
            ap, bp = ac["prompt_logprobs"][i], bc["prompt_logprobs"][i]
            if not all(
                math.isfinite(v["logprob"]) for row in (ap, bp) for v in row.values()
            ):
                raise ValueError(f"non-finite probabilities: {name}:{i}")
            la.append(ap[str(token)]["logprob"])
            lb.append(bp[str(token)]["logprob"])
            matches += reported_top1(ap) == reported_top1(bp)
            exact += la[-1] == lb[-1]
        errors = [abs(x - y) for x, y in zip(la, lb, strict=True)]
        result["logprobs"][name] = dict(
            scored_tokens=len(errors),
            top1_agreement=matches / len(errors),
            exact_chosen_logprob_fraction=exact / len(errors),
            mean_abs_logprob_delta=statistics.mean(errors),
            p99_abs_logprob_delta=quantile(errors, 0.99),
            max_abs_logprob_delta=max(errors),
            left_mean_nll=-statistics.mean(la),
            right_mean_nll=-statistics.mean(lb),
        )
    for kind, rows in list(result["answers"].items()):
        result["answers"][kind] = dict(
            total=len(rows),
            left_correct=sum(r["left_correct"] for r in rows),
            right_correct=sum(r["right_correct"] for r in rows),
            left_only_correct=sum(
                r["left_correct"] and not r["right_correct"] for r in rows
            ),
            right_only_correct=sum(
                r["right_correct"] and not r["left_correct"] for r in rows
            ),
            answer_agreement=sum(r["answer_agrees"] for r in rows),
            cases=rows,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "answers"}, indent=2))
    print(
        json.dumps(
            {
                k: {n: v for n, v in row.items() if n != "cases"}
                for k, row in result["answers"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
