# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Check a running vLLM service, including default-thinking response parsing."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


def check(
    base_url, model, timeout, name, prompt, expected, *, thinking=False, stream=False
):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": 128,
        "stream": stream,
    }
    # Omission deliberately tests the tokenizer's default thinking mode.
    if not thinking:
        payload["chat_template_kwargs"] = {"thinking": False}
    record = {"name": name, "request": payload, "expected": expected}
    started = time.monotonic()
    try:
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            record["status"] = response.status
            if stream:
                chunks, content, reasoning, finish, done = [], "", "", None, False
                for raw_line in response:
                    line = raw_line.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(data)
                    chunks.append(chunk)
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        content += delta.get("content") or ""
                        reasoning += (
                            delta.get("reasoning")
                            or delta.get("reasoning_content")
                            or ""
                        )
                        finish = choice.get("finish_reason") or finish
                record.update(chunks=chunks, done=done)
                if not done:
                    raise ValueError("stream ended without [DONE]")
            else:
                result = json.load(response)
                record["response"] = result
                choice = result["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                reasoning = (
                    message.get("reasoning") or message.get("reasoning_content") or ""
                )
                finish = choice["finish_reason"]
        if name == "multi-step-decode":
            numbers = [int(n) for n in re.findall(r"\d+", content)]
            correct = numbers == list(range(1, 31))
        else:
            correct = content.strip().rstrip(".。") == expected
        record.update(
            content=content,
            reasoning=reasoning,
            finish_reason=finish,
            passed=correct and finish == "stop" and "</think>" not in content,
        )
        if thinking:
            record["passed"] &= bool(reasoning)
        if name == "chunked-prefill":
            prompt_tokens = record["response"]["usage"]["prompt_tokens"]
            record["passed"] &= prompt_tokens > 128
    except urllib.error.HTTPError as exc:
        record.update(status=exc.code, error=exc.read().decode(), passed=False)
    except Exception as exc:
        record.update(error=repr(exc), passed=False)
    record["seconds"] = time.monotonic() - started
    print(
        json.dumps(
            {
                key: record[key]
                for key in ("name", "passed", "seconds", "content", "error")
                if key in record
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="deepseek-v41-megaflux")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        ("arithmetic", "What is 17 + 25? Reply with only the integer.", "42"),
        (
            "capital",
            "What is the capital of France? Reply with only the city name.",
            "Paris",
        ),
        ("chinese", "中国的首都是哪里？只回答城市名称。", "北京"),
        ("repeat", "Repeat exactly this word and nothing else: megaflux", "megaflux"),
    ]

    def run(case, **kwargs):
        return check(args.base_url, args.model, args.timeout, *case, **kwargs)

    records = [run(case) for case in cases]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records.extend(
            pool.map(
                run,
                [
                    (f"concurrent-{name}", prompt, answer)
                    for name, prompt, answer in cases
                ],
            )
        )
    records.append(
        run(
            (
                "chunked-prefill",
                "The marker is irrelevant. " * 90 + "\n" + cases[0][1],
                "42",
            )
        )
    )
    records.append(
        run(
            (
                "multi-step-decode",
                "List the integers from 1 to 30, separated by commas. "
                "Output only the list.",
                None,
            )
        )
    )
    records.append(run(("streaming", *cases[1][1:]), stream=True))
    records.append(run(("default-thinking", *cases[0][1:]), thinking=True))
    records.append(
        run(("streaming-thinking", *cases[0][1:]), thinking=True, stream=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    passed = sum(record["passed"] for record in records)
    print(f"{passed}/{len(records)} passed; evidence: {args.output}")
    raise SystemExit(0 if passed == len(records) else 1)


if __name__ == "__main__":
    main()
