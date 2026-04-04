"""
main.py — Entry point for the APE + Router Pipeline

Flow:
    User Input → Router → math_qa  → solve() → print answer
                        → email_gen → evaluate_email() → print evaluation
"""

import json
from router import classify
from email_evaluator import evaluate_email
from math_evaluator import solve, extract_number

# Best prompt found so far — swap this out once APE loop runs
MATH_PROMPT = "You are a math olympiad expert, think step by step."


def run_pipeline(query: str):
    route = classify(query)

    print(f"\n[Router] Task: {route.task_type.value} ({route.confidence:.0%} confidence)")
    print(f"[Router] Query: {route.clean_query}\n")

    if route.task_type.value == "math_qa":
        raw_solution = solve(route.clean_query, MATH_PROMPT)
        answer       = extract_number(raw_solution)

        print("── Math Solution ───────────────────────────────")
        print(raw_solution)
        print(f"\n── Extracted Answer: {answer} ──────────────────\n")

    elif route.task_type.value == "email_gen":
        result = evaluate_email(route.clean_query)

        print("── Email Evaluation ────────────────────────────")
        print(f"  Score          : {result.get('score', 'N/A'):.0%}")
        print(f"  Clarity        : {result.get('clarity')}/5")
        print(f"  Professionalism: {result.get('professionalism')}/5")
        print(f"  Grammar        : {result.get('grammar')}/5")
        print(f"  Tone           : {result.get('tone')}")
        print(f"  Issues         : {result.get('issues')}")
        print(f"  Suggestion     : {result.get('suggestion')}")
        print("────────────────────────────────────────────────\n")

    else:
        print(f"[main] Unknown task type: {route.task_type.value}")


if __name__ == "__main__":
    print("=== APE + Router Pipeline ===")
    query = input("Enter your query:\n").strip()

    if not query:
        print("[error] Empty input. Exiting.")
        exit(1)

    run_pipeline(query)