import os
import json
import sys
import traceback
from openai import OpenAI
from dotenv import load_dotenv
from router import classify

load_dotenv()

print("[DEBUG] Script started...")

# ── Config ────────────────────────────────────────────────────────────────

SOLVER_MODEL = "gpt-4o-mini"

BEST_PROMPT_FILES = {
    "math_qa"    : "JSON/best_prompt.json",
    "email_gen"  : "JSON/email_best_prompt.json",
    "email_eval" : "JSON/email_best_prompt.json",
}

FALLBACK_PROMPTS = {
    "math_qa"    : "You are a math expert. Think step by step.",
    "email_gen"  : "Write a professional email clearly and politely.",
    "email_eval" : "You are an expert email reviewer. Give structured, actionable feedback.",
}

# 🔴 Validate API key early
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[FATAL] OPENAI_API_KEY not found in .env")
    input("Press Enter to exit...")
    sys.exit(1)

client = OpenAI(api_key=api_key)


# ── Load best prompt ──────────────────────────────────────────────────────

def load_best_prompt(task_type: str) -> str:
    try:
        path = BEST_PROMPT_FILES.get(task_type)

        if not path or not os.path.exists(path):
            print(f"[WARN] No prompt file for '{task_type}', using fallback.")
            return FALLBACK_PROMPTS.get(task_type, "You are a helpful assistant.")

        with open(path) as f:
            data = json.load(f)

        prompt = data.get("best_prompt")

        if not prompt:
            print(f"[WARN] 'best_prompt' missing in {path}, using fallback.")
            return FALLBACK_PROMPTS.get(task_type, "You are a helpful assistant.")

        return prompt

    except Exception as e:
        print("[ERROR] load_best_prompt failed:", e)
        return "You are a helpful assistant."


# ── Generate answer ───────────────────────────────────────────────────────

def generate_answer(system_prompt: str, user_query: str, task_type: str) -> str:
    try:
        if task_type == "email_eval":
            user_message = f"Please evaluate the following email:\n\n{user_query}"
        else:
            user_message = user_query

        print("[DEBUG] Calling OpenAI for answer...")

        response = client.chat.completions.create(
            model=SOLVER_MODEL,
            temperature=0.3,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("[ERROR] OpenAI generation failed:")
        traceback.print_exc()
        return "ERROR: Failed to generate response."


# ── Pipeline ──────────────────────────────────────────────────────────────

def run(user_query: str) -> dict:
    try:
        print("\n" + "="*55)
        print(f"QUERY: {user_query}")
        print("="*55)

        # Step 1 — Routing
        print("\n[1/3] Routing query...")
        route = classify(user_query)

        task = route.task_type.value
        print(f"Task       : {task}")
        print(f"Confidence : {route.confidence:.0%}")
        print(f"Reasoning  : {route.reasoning}")

        # Step 2 — Prompt
        print("\n[2/3] Loading best prompt...")
        best_prompt = load_best_prompt(task)

        # Step 3 — Answer
        print("\n[3/3] Generating answer...")
        answer = generate_answer(best_prompt, route.clean_query, task)

        print("\n" + "="*55)
        print("ANSWER:")
        print("="*55)
        print(answer)
        print("="*55)

        result = {
            "query": user_query,
            "task_type": task,
            "confidence": route.confidence,
            "best_prompt": best_prompt,
            "answer": answer,
        }

        os.makedirs("JSON", exist_ok=True)

        with open("JSON/pipeline_result.json", "w") as f:
            json.dump(result, f, indent=2)

        print("\n[INFO] Saved → JSON/pipeline_result.json")

        return result

    except Exception as e:
        print("\n[FATAL ERROR in run()]")
        traceback.print_exc()
        input("Press Enter to exit...")
        return {}


# ── Entry ─────────────────────────────────────────────────────────────────

def main():
    try:
        if len(sys.argv) > 1:
            user_query = " ".join(sys.argv[1:])
        else:
            print("=== AN2 Router Pipeline ===")
            user_query = input("Enter your query: ").strip()

        if not user_query:
            print("[ERROR] Empty query.")
            return

        run(user_query)

    except Exception:
        print("\n[FATAL ERROR in main()]")
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()