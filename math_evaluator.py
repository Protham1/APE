"""
math_evaluator.py — Standalone Math QA Evaluator

Usage:
    python math_evaluator.py                        # uses DEFAULT_PROMPT
    python math_evaluator.py "your prompt here"     # uses custom prompt

Input  : math_dataset.json
Output : eval_result.json  (consumed by APE loop later)

Two-call design per question:
    Call 1 — Solver   : LLM answers the math question
    Call 2 — Extractor: LLM pulls just the number out (min tokens)

Key design: ALL prompts must end with the FINAL ANSWER tag instruction.
The extractor looks for this tag specifically — avoids grabbing wrong
numbers from intermediate steps in long hard-problem solutions.
"""

import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

SOLVER_MODEL    = "gpt-4o-mini"
EXTRACTOR_MODEL = "gpt-4o-mini"
DATASET_FILE    = "math_dataset.json"
OUTPUT_FILE     = "eval_result.json"
TOLERANCE       = 0.01

# This suffix is appended to EVERY prompt — APE mutations should not change this
# It's what makes the extractor reliable on long hard-problem responses
ANSWER_TAG_SUFFIX = "\nAt the end of your solution always write: FINAL ANSWER: <number>"

# Default base prompt — APE loop mutates only this part, suffix stays fixed
DEFAULT_BASE_PROMPT = "Solve the following math problem step by step."

# Extractor — looks for the tag specifically, ignores all intermediate numbers
EXTRACTOR_SYSTEM = (
    "The math solution contains a line starting with 'FINAL ANSWER:'. "
    "Extract only the number after 'FINAL ANSWER:'. "
    "If it is a fraction like 4/9, convert it to decimal. "
    "If it is an expression like 'C(8,3) = 56', extract just 56. "
    "Return probabilities and fractions as decimals (e.g. 0.4 not 4/10). "
    "For time answers return decimal hours (e.g. 2.4 not '2h 24min'). "
    "Reply with ONLY the number. If no FINAL ANSWER line exists reply NULL."
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Call 1: Solver ────────────────────────────────────────────────────────────

def solve(question: str, base_prompt: str) -> str:
    full_prompt = base_prompt + ANSWER_TAG_SUFFIX
    response = client.chat.completions.create(
        model       = SOLVER_MODEL,
        temperature = 0.0,
        max_tokens  = 600,
        messages    = [
            {"role": "system", "content": full_prompt},
            {"role": "user",   "content": question},
        ],
    )
    return response.choices[0].message.content.strip()

# ── Call 2: Extractor ─────────────────────────────────────────────────────────

def extract_number(solver_response: str) -> float | None:
    response = client.chat.completions.create(
        model       = EXTRACTOR_MODEL,
        temperature = 0.0,
        max_tokens  = 20,
        messages    = [
            {"role": "system", "content": EXTRACTOR_SYSTEM},
            {"role": "user",   "content": solver_response},
        ],
    )
    raw = response.choices[0].message.content.strip()
    try:
        return float(raw)
    except ValueError:
        return None

# ── Grader ────────────────────────────────────────────────────────────────────

def is_correct(predicted: float | None, expected: float) -> bool:
    if predicted is None:
        return False
    return abs(predicted - expected) <= TOLERANCE

# ── Main Evaluator ────────────────────────────────────────────────────────────

def evaluate(base_prompt: str, verbose: bool = True) -> dict:
    with open(DATASET_FILE, encoding="utf-8") as f:
        dataset = json.load(f)[:20]

    results = []
    correct = 0

    if verbose:
        print(f"\n── Evaluating prompt ──────────────────────────────────")
        print(f"  {base_prompt[:80]}{'...' if len(base_prompt) > 80 else ''}")
        print(f"───────────────────────────────────────────────────────\n")

    for q in dataset:
        solver_response = solve(q["question"], base_prompt)
        predicted       = extract_number(solver_response)
        passed          = is_correct(predicted, q["answer"])

        if passed:
            correct += 1

        result = {
            "id"        : q["id"],
            "difficulty": q["difficulty"],
            "topic"     : q["topic"],
            "expected"  : q["answer"],
            "predicted" : predicted,
            "correct"   : passed,
        }
        results.append(result)

        if verbose:
            status = "✓" if passed else "✗"
            print(f"  [{status}] Q{q['id']:02d} [{q['difficulty']:6}] "
                  f"expected={q['answer']} got={predicted}")

    total     = len(dataset)
    score     = round(correct / total, 4)

    breakdown = {}
    for diff in ["easy", "medium", "hard"]:
        subset  = [r for r in results if r["difficulty"] == diff]
        n       = len(subset)
        n_right = sum(1 for r in subset if r["correct"])
        breakdown[diff] = {
            "correct": n_right,
            "total"  : n,
            "score"  : round(n_right / n, 4) if n else 0,
        }

    output = {
        "prompt"   : base_prompt,
        "score"    : score,
        "correct"  : correct,
        "total"    : total,
        "breakdown": breakdown,
        "results"  : results,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    if verbose:
        print(f"\n── Final Score ─────────────────────────────────────────")
        print(f"  Overall : {correct}/{total} = {score:.0%}")
        for diff, stats in breakdown.items():
            print(f"  {diff:6}  : {stats['correct']}/{stats['total']} = {stats['score']:.0%}")
        print(f"\n  Output saved → {OUTPUT_FILE}")
        print(f"───────────────────────────────────────────────────────\n")

    return output

# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_BASE_PROMPT
    evaluate(base_prompt)