

import os
import json
import sys
from enum import Enum
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

ROUTER_MODEL   = "gpt-4o-mini"          # cheapest; only does classification
OUTPUT_FILE    = "router_output.json"   # consumed by next script in pipeline

TASK_TYPES = ["math_qa", "email_gen"]   # extend this list as you add task types
TASK_TYPES = ["math_qa", "email_gen", "email_eval"] 

# ── Pydantic schema (enforces structured output) ─────────────────────────────

class TaskType(str, Enum):
    math_qa   = "math_qa"
    email_gen = "email_gen"
    email_eval = "email_eval"

class RouterOutput(BaseModel):
    task_type  : TaskType
    confidence : float          # 0.0 – 1.0
    reasoning  : str            # one-line explanation (useful for debugging)
    clean_query: str            # stripped / normalized version of user input

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a task classifier for a prompt-routing system.

Classify the user's input into exactly one of these task types:
- math_qa   : any mathematical problem, calculation, word problem, algebra, arithmetic
- email_gen : requests to write, draft, compose, or improve an email of any kind
- email_eval : evaluate or review an email

Respond ONLY with a valid JSON object matching this schema exactly:
{
  "task_type"  : "<math_qa | email_gen>",
  "confidence" : <float between 0.0 and 1.0>,
  "reasoning"  : "<one sentence explaining why>",
  "clean_query": "<user query, lightly cleaned — fix typos, remove filler words>"
}

Do not include any text outside the JSON object.
""".strip()

# ── Core classifier function ──────────────────────────────────────────────────

def classify(user_query: str) -> RouterOutput:
    """
    Send user_query to gpt-4o-mini and return a validated RouterOutput.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model       = ROUTER_MODEL,
        temperature = 0.0,          # deterministic — classification, not generation
        max_tokens  = 200,
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_query},
        ],
    )

    raw_json = response.choices[0].message.content.strip()

    try:
        parsed = RouterOutput.model_validate_json(raw_json)
    except Exception as e:
        raise ValueError(
            f"Router returned invalid JSON.\nRaw response:\n{raw_json}\nError: {e}"
        )

    return parsed

# ── Output writer (for downstream scripts) ────────────────────────────────────

def save_output(result: RouterOutput, query: str) -> None:
    """
    Write structured output to router_output.json.
    This file is the handoff contract between router.py and the next script.
    """
    payload = {
        "original_query": query,
        **result.model_dump(),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[router] Output saved → {OUTPUT_FILE}")

# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        # Accept query as CLI argument: python router.py "solve 2x + 3 = 9"
        user_query = " ".join(sys.argv[1:])
    else:
        # Interactive mode
        print("=== Task Router ===")
        user_query = input("Enter your query: ").strip()

    if not user_query:
        print("[error] Empty query. Exiting.")
        sys.exit(1)

    print(f"\n[router] Classifying: '{user_query}'")

    result = classify(user_query)
    save_output(result, user_query)

    # ── Pretty print for dev visibility ──
    print("\n── Classification Result ──────────────────")
    print(f"  Task Type  : {result.task_type.value}")
    print(f"  Confidence : {result.confidence:.0%}")
    print(f"  Reasoning  : {result.reasoning}")
    print(f"  Clean Query: {result.clean_query}")
    print("───────────────────────────────────────────\n")

    return result

if __name__ == "__main__":
    main()