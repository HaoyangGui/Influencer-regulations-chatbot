#!/usr/bin/env python3
"""Ask every question in an evaluation dataset and save the chatbot responses."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "evaluation" / "dataset" / "evaluation_dataset_quote_updated.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "output" / "evaluation_results.json"


def load_questions(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(questions, list):
        raise ValueError(f"Expected a JSON object containing a 'questions' list: {input_path}")

    required = ("id", "category", "difficulty", "question_type", "question")
    invalid = [index for index, item in enumerate(questions, start=1)
               if not isinstance(item, dict) or any(field not in item for field in required)]
    if invalid:
        raise ValueError(f"Questions missing required fields at entries: {invalid}")
    return questions


def run_batch(
    questions: list[dict[str, Any]],
    server_url: str,
    max_tokens: int,
    top_k: int,
    timeout: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    endpoint = server_url.rstrip("/") + "/answer"

    for position, item in enumerate(questions, start=1):
        question = str(item["question"])
        record: dict[str, Any] = {
            "id": item["id"],
            "category": item["category"],
            "difficulty": item["difficulty"],
            "question_type": item["question_type"],
            "question": question,
        }
        started = time.perf_counter()
        try:
            response = requests.post(
                endpoint,
                json={"question": question, "top_k": top_k, "max_tokens": max_tokens},
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("Server response was not a JSON object")
            record["response"] = body
            print(f"[{position}/{len(questions)}] {item['id']} completed")
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            print(f"[{position}/{len(questions)}] {item['id']} failed: {exc}")
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        results.append(record)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--server-url", default=os.getenv("RAG_SERVER_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    questions = load_questions(args.input)
    results = run_batch(questions, args.server_url, args.max_tokens, args.top_k, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "input_dataset": str(args.input),
            "server_url": args.server_url.rstrip("/"),
            "question_count": len(questions),
            "completed_count": sum("response" in item for item in results),
            "failed_count": sum("error" in item for item in results),
        },
        "results": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
