# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Save matched GSM8K, retrieval and teacher-forced logprob observations."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import time
import urllib.error
import urllib.request


def save(path, data):
    payload = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()
    path.write_bytes(gzip.compress(payload, mtime=0))


def numeric_answer(text):
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if not matches:
        return None
    try:
        return str(Decimal(matches[-1].replace(",", "")).normalize())
    except InvalidOperation:
        return None


def fixtures(path, dataset, model_path):
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(model_path, tokenizer_mode="deepseek_v41")
    data = dataset.read_bytes()
    rows = [json.loads(line) for line in data.splitlines()]
    indices = random.Random(20261027).sample(range(len(rows)), 64)
    cases = []
    for index in indices:
        row = rows[index]
        cases.append(
            dict(
                name=f"gsm8k-{index}",
                kind="gsm8k",
                expected=numeric_answer(row["answer"]),
                prompt=row["question"]
                + "\nSolve step by step. End your answer with #### followed by the final number.",
                max_tokens=2048,
            )
        )
    filler = "The catalogue entry describes an ordinary blue book on a wooden shelf. "
    for length in (16384, 32768):
        tokens = tokenizer.encode(filler * (length // 8), add_special_tokens=False)[
            : length - 256
        ]
        for slot, fraction in enumerate((0.05, 0.5, 0.95)):
            at = int(len(tokens) * fraction)
            code = str(81234567 + length + slot * 173)
            prompt = (
                "Read the archive and find the secret passcode for Project Cedar.\n"
                + tokenizer.decode(tokens[:at])
                + f"\nThe secret passcode for Project Cedar is {code}.\n"
                + tokenizer.decode(tokens[at:])
                + "\nWhat is the secret passcode for Project Cedar? Reply with only its digits."
            )
            cases.append(
                dict(
                    name=f"retrieval-{length}-position-{slot}",
                    kind="retrieval",
                    expected=code,
                    prompt=prompt,
                    max_tokens=64,
                    target_context=length,
                    needle_fraction=fraction,
                )
            )
    corpus = "\n\n".join(
        row["question"] + "\n" + re.sub(r"<<.*?>>", "", row["answer"])
        for row in rows[:160]
    )
    tokens = tokenizer.encode(corpus, add_special_tokens=False)
    tokens = (tokens * (32768 // len(tokens) + 1))[:32768]
    for length in (512, 4096, 16384, 32768):
        cases.append(
            dict(
                name=f"logprobs-{length}",
                kind="logprobs",
                prompt_token_ids=tokens[:length],
            )
        )
    result = dict(
        dataset_url="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        dataset_sha256=hashlib.sha256(data).hexdigest(),
        dataset_rows=len(rows),
        sample_seed=20261027,
        sample_indices=indices,
        cases=cases,
    )
    save(path, result)
    return result


def observe(case, args):
    name, kind = case["name"], case["kind"]
    if kind == "logprobs":
        endpoint = "/completions"
        payload = dict(
            model=args.model,
            prompt=case["prompt_token_ids"],
            max_tokens=1,
            temperature=0,
            seed=0,
            prompt_logprobs=1,
            return_token_ids=True,
        )
    else:
        endpoint = "/chat/completions"
        payload = dict(
            model=args.model,
            messages=[dict(role="user", content=case["prompt"])],
            max_tokens=case["max_tokens"],
            temperature=0,
            seed=0,
            chat_template_kwargs={"thinking": False},
        )
    record = dict(name=name, kind=kind, request=payload, transport_ok=False)
    started = time.monotonic()
    try:
        request = urllib.request.Request(
            args.base_url.rstrip("/") + endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=1200) as response:
            record["response"] = result = json.load(response)
        choice = result["choices"][0]
        record.update(transport_ok=True, finish_reason=choice["finish_reason"])
        if kind == "logprobs":
            ids = choice["prompt_token_ids"]
            values = choice["prompt_logprobs"]
            if ids != case["prompt_token_ids"] or len(values) != len(ids):
                raise ValueError("teacher-forced prompt/token count mismatch")
            if any(
                values[i] is None or str(ids[i]) not in values[i]
                for i in range(1, len(ids))
            ):
                raise ValueError("missing teacher-forced token probability")
            record["scored_tokens"] = len(ids) - 1
        else:
            content = choice["message"].get("content") or ""
            answer = (
                numeric_answer(content)
                if kind == "gsm8k"
                else content.strip().rstrip(".。")
            )
            record.update(
                expected=case["expected"],
                extracted=answer,
                correct=answer == case["expected"]
                and choice["finish_reason"] == "stop",
            )
    except urllib.error.HTTPError as exc:
        record.update(transport_ok=False, error=exc.read().decode())
    except Exception as exc:
        record.update(transport_ok=False, error=repr(exc))
    record["seconds"] = time.monotonic() - started
    save(args.output / (name + ".json.gz"), record)
    summary = {k: v for k, v in record.items() if k not in ("request", "response")}
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="deepseek-v41-megaflux")
    parser.add_argument(
        "--model-path", default="/gpfs/rd/models/DeepSeek-V4.1-Flash-1017/"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "logprobs"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = fixtures(args.fixtures, args.dataset, args.model_path)
    cases = data["cases"]
    observations = []
    if args.phase == "all":
        with ThreadPoolExecutor(max_workers=8) as pool:
            observations.extend(
                pool.map(
                    lambda c: observe(c, args),
                    [c for c in cases if c["kind"] != "logprobs"],
                )
            )
    # Score sequentially to keep the logprob memory demand bounded.
    observations.extend(observe(c, args) for c in cases if c["kind"] == "logprobs")
    summary = dict(
        fixture_sha256=hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
        observations=observations,
        transport_failures=sum(not r["transport_ok"] for r in observations),
    )
    for kind in ("gsm8k", "retrieval"):
        selected = [r for r in observations if r["kind"] == kind]
        summary[kind] = dict(
            total=len(selected), correct=sum(r.get("correct", False) for r in selected)
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    raise SystemExit(bool(summary["transport_failures"]))


if __name__ == "__main__":
    main()
