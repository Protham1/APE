

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL       = "gpt-4o-mini"
OUTPUT_FILE = "email_eval.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Score is NOT in the schema — computed deterministically from sub-scores
# This avoids inconsistent LLM self-scoring
SYSTEM_PROMPT = """
You are an email quality evaluator.

Return ONLY valid JSON, no extra text:

{
  "clarity":          <int 0-5>,
  "professionalism":  <int 0-5>,
  "grammar":          <int 0-5>,
  "tone":             <"formal" | "neutral" | "informal">,
  "issues":           [<max 3 short strings>],
  "suggestion":       <one short fix as a string>
}
""".strip()


def evaluate_email(email_text: str) -> dict:
    response = client.chat.completions.create(
        model       = MODEL,
        temperature = 0.0,
        max_tokens  = 250,
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": email_text},
        ],
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 0.0, "error": "bad json", "raw": raw}

    # Compute score deterministically: avg of 3 sub-scores normalized to 0-1
    clarity         = result.get("clarity", 0)
    professionalism = result.get("professionalism", 0)
    grammar         = result.get("grammar", 0)
    result["score"] = round((clarity + professionalism + grammar) / 15, 4)

    return result


def save_output(email_text: str, result: dict) -> None:
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"email": email_text, "evaluation": result}, f, indent=2)
    print(f"\n[evaluator] Output saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    print("=== Email Evaluator ===")
    email = input("Paste your email below (press Enter twice when done):\n")

    if not email.strip():
        print("[error] Empty input. Exiting.")
        exit(1)

    result = evaluate_email(email)
    save_output(email, result)

    print("\n── Evaluation Result ───────────────────────────")
    print(f"  Score          : {result.get('score', 'N/A'):.0%}")
    print(f"  Clarity        : {result.get('clarity')}/5")
    print(f"  Professionalism: {result.get('professionalism')}/5")
    print(f"  Grammar        : {result.get('grammar')}/5")
    print(f"  Tone           : {result.get('tone')}")
    print(f"  Issues         : {result.get('issues')}")
    print(f"  Suggestion     : {result.get('suggestion')}")
    print("────────────────────────────────────────────────\n")