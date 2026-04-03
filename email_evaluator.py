import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "gpt-4o-mini"   
OUTPUT_FILE = "email_eval.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an email evaluator.

Return ONLY JSON:

{
  "score": float (0 to 1),
  "clarity": 0-5,
  "professionalism": 0-5,
  "grammar": 0-5,
  "tone": "formal | neutral | informal",
  "issues": ["max 3 short issues"],
  "suggestion": "one short fix"
}
""".strip()


def evaluate_email(email_text: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        max_tokens=120,   
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": email_text}
        ],
    )

    raw = response.choices[0].message.content.strip()

    try:
        return json.loads(raw)
    except:
        return {"score": 0.0, "error": "bad json", "raw": raw}


def save_output(email_text, result):
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "email": email_text,
            "evaluation": result
        }, f, indent=2)


if __name__ == "__main__":
    email = input("Enter email:\n")
    result = evaluate_email(email)
    save_output(email, result)

    print("\nResult:\n", json.dumps(result, indent=2))