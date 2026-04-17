import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from evaluators.email_evaluator import evaluate_email

load_dotenv()

# --- Config ---
MODEL = "gpt-4o-mini"   # cheap
N_GENERATIONS = 2
N_MUTATIONS = 5
TOP_K = 2

OUTPUT_FILE = "email_best_prompt.json"
LOG_FILE = "email_ape_log.json"

SEED_PROMPT = "Write a professional email clearly and politely."

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Email Generator ---
def generate_email(prompt: str, task: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        max_tokens=150,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": task}
        ],
    )
    return response.choices[0].message.content.strip()


# --- Fitness Function ---
def evaluate_prompt(prompt: str, dataset: list) -> float:
    scores = []

    for item in dataset:
        email = generate_email(prompt, item["task"])
        eval_result = evaluate_email(email)

        score = eval_result.get("score", 0.0)

        if len(email) > 200:
            score -= 0.1

        # clamp score between 0 and 1
        score = max(0, min(score, 1))
        scores.append(score)

    return sum(scores) / len(scores)


# --- Prompt Mutator ---
MUTATOR_SYSTEM = """
You improve prompts for email writing.

Given a prompt, generate a better version.

Keep it short and clear.
""".strip()


def mutate_prompt(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        max_tokens=60,
        messages=[
            {"role": "system", "content": MUTATOR_SYSTEM},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content.strip()


# --- APE Loop ---
def run_ape():
    with open("JSON/email_dataset.json") as f:
        dataset = json.load(f)

    population = [SEED_PROMPT]
    log = []

    for gen in range(N_GENERATIONS):
        print(f"\n=== Generation {gen} ===")

        scored = []

        for prompt in population:
            score = evaluate_prompt(prompt, dataset)
            scored.append((prompt, score))

            print(f"Score: {score:.3f}")
            print(f"Prompt: {prompt}\n")
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Keep top K
        top_prompts = [p for p, _ in scored[:TOP_K]]

        # Log
        log.append({
            "generation": gen,
            "top_score": scored[0][1],
            "best_prompt": scored[0][0]
        })

        # Mutate
        new_population = top_prompts.copy()

        for p in top_prompts:
            for _ in range(N_MUTATIONS):
                new_population.append(mutate_prompt(p))

        population = new_population

    # Best prompt
    best_prompt = scored[0][0]

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"best_prompt": best_prompt}, f, indent=2)

    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

    


if __name__ == "__main__":
    run_ape()