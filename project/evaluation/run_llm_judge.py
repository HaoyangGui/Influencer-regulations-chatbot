#!/usr/bin/env python3
"""Run an LLM-as-a-judge evaluation over the fixed benchmark dataset."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm import LLMClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
DEFAULT_BENCHMARK = EVALUATION_DIR / "dataset" / "evaluation_benchmark.json"
DEFAULT_ANSWERS = EVALUATION_DIR / "dataset" / "evaluation_question_answer.json"
DEFAULT_PROMPT = EVALUATION_DIR / "judge_prompt.txt"
DEFAULT_OUTPUT = EVALUATION_DIR / "output"
ERRORS = (
    "None", "Incorrect legal conclusion", "Incomplete answer",
    "Unsupported evidence", "Incorrect citation", "Fabricated quotation",
    "Failure to abstain", "Instruction violation",
)
DIMENSIONS = ("correctness", "completeness", "evidence_support", "instruction_following")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def build_items(benchmark_path: Path, answers_path: Path) -> list[dict[str, Any]]:
    benchmark = load_json(benchmark_path)
    answers = load_json(answers_path)
    benchmark_items = {item["id"]: item for item in benchmark["results"]}
    answer_items = {item["id"]: item for item in answers["results"]}
    if set(benchmark_items) != set(answer_items):
        raise ValueError("Benchmark and evaluation_question_answer IDs do not match")
    items = []
    for answer in answers["results"]:
        item = benchmark_items[answer["id"]]
        if any(item[field] != answer[field] for field in ("category", "difficulty", "question_type", "question")):
            raise ValueError(f"Metadata mismatch for {answer['id']}")
        reference = item.get("reference", {})
        response = answer.get("response") or {}
        items.append({
            "id": item["id"],
            "category": item["category"],
            "difficulty": item["difficulty"],
            "question_type": item["question_type"],
            "question": item["question"],
            "answerability": {
                "unanswerable": reference.get("unanswerable", False),
                "requires_multi_hop": reference.get("requires_multi_hop", False),
            },
            "key_legal_points": (reference.get("gold_answer") or {}).get("key_points", []),
            "relevant_evidence": reference.get("gold_quotations", []),
            "chatbot_answer": response.get("answer", ""),
            "retrieved_chunks": response.get("evidence", []),
        })
    return items


def parse_judgement(text: str, question_id: str) -> dict[str, Any]:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text.strip(), re.DOTALL)
    parsed = json.loads(match.group(1) if match else text.strip())
    if not isinstance(parsed, dict):
        raise ValueError("Judge output is not a JSON object")
    scores = {}
    for field in DIMENSIONS:
        value = int(parsed[field])
        if value not in (0, 1, 2):
            raise ValueError(f"{field} must be 0, 1, or 2")
        scores[field] = value
    total = round(sum(scores.values()) / 8, 4)
    if abs(float(parsed.get("total_score", total)) - total) > 1e-6:
        raise ValueError("total_score does not match the rubric")
    error = parsed.get("main_error")
    if error not in ERRORS:
        raise ValueError(f"Invalid main_error: {error}")
    return {
        "question_id": question_id,
        **scores,
        "total_score": total,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "main_error": error,
    }


def answerability(item: dict[str, Any]) -> str:
    if item["answerability"].get("unanswerable"):
        return "unanswerable"
    return "answerable"


def metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "number_of_questions": len(results),
        **{
            f"mean_{field}": round(mean(result[field] for result in results), 4) if results else 0.0
            for field in (*DIMENSIONS, "total_score")
        },
    }


def qualitative_summary(results: list[dict[str, Any]]) -> dict[str, list[str]]:
    if not results:
        return {"strengths": [], "weaknesses": [], "common_errors": [], "recommendations": []}
    averages = {field: mean(result[field] for result in results) for field in DIMENSIONS}
    errors = Counter(result["main_error"] for result in results if result["main_error"] != "None")
    strongest = max(averages, key=averages.get)
    weakest = min(averages, key=averages.get)
    return {
        "strengths": [f"{strongest} was strongest (mean {averages[strongest]:.2f}/2)."],
        "weaknesses": [f"{weakest} was weakest (mean {averages[weakest]:.2f}/2)."],
        "common_errors": [f"{name}: {count}" for name, count in errors.most_common(3)]
        or ["No non-None main errors were recorded."],
        "recommendations": [
            "Keep requiring claims to cite retrieved chunks and quotations to match verbatim.",
            f"Strengthen prompt guidance for {weakest}.",
        ],
    }


def write_summary_markdown(path: Path, report: dict[str, Any]) -> None:
    overall = report["summary"]["overall"]
    lines = ["# LLM-as-a-Judge Summary", "", f"Prompt version: `{report['prompt_version']}`", "",
             "## Overall", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in overall.items())
    for section, values in report["summary"]["qualitative_summary"].items():
        lines.extend(["", f"## {section.replace('_', ' ').title()}", ""])
        lines.extend(f"- {value}" for value in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    items = build_items(args.benchmark, args.answers)
    template = args.prompt.read_text(encoding="utf-8")
    client = LLMClient()
    judged: list[dict[str, Any]] = []
    args.output.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output / f"llm_judge_{args.prompt_version}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for position, item in enumerate(items, start=1):
            evaluation_item = json.dumps(item, ensure_ascii=False, indent=2)
            judge_input = f"{template}\n\nEvaluation item:\n{evaluation_item}"
            try:
                raw = client.generate_answer(judge_input, max_tokens=args.max_tokens)
                result = parse_judgement(raw, item["id"])
            except (ValueError, json.JSONDecodeError):
                raw = client.generate_answer(judge_input, max_tokens=args.max_tokens * 2)
                result = parse_judgement(raw, item["id"])
            result["answerability"] = answerability(item)
            result["prompt_version"] = args.prompt_version
            judged.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{position}/{len(items)}] {item['id']} judged")

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_answerability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result, item in zip(judged, items):
        by_category[item["category"]].append(result)
        by_answerability[result["answerability"]].append(result)
    overall = metrics(judged)
    summary = {
        "overall": {
            **overall,
            "number_of_answerable_questions": len(by_answerability["answerable"]),
            "number_of_partially_answerable_questions": len(by_answerability["partially_answerable"]),
            "number_of_unanswerable_questions": len(by_answerability["unanswerable"]),
        },
        "by_category": {key: metrics(value) for key, value in sorted(by_category.items())},
        "by_answerability": {key: metrics(value) for key, value in sorted(by_answerability.items())},
        "error_distribution": {error: sum(result["main_error"] == error for result in judged) for error in ERRORS},
        "qualitative_summary": qualitative_summary(judged),
    }
    report = {
        "prompt_version": args.prompt_version,
        "individual_results": judged,
        "summary": summary,
        "prompt_comparison": [{
            "prompt_version": args.prompt_version,
            **{f"mean_{field}": overall[f"mean_{field}"] for field in (*DIMENSIONS, "total_score")},
        }],
    }
    json_path = args.output / f"llm_judge_{args.prompt_version}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_markdown(args.output / f"llm_judge_{args.prompt_version}_summary.md", report)
    print(f"Saved {jsonl_path}, {json_path}, and summary Markdown")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-version", default="V1")
    parser.add_argument("--max-tokens", type=int, default=1536)
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    run(args)


if __name__ == "__main__":
    main()
