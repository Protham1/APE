

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from math_evaluator import evaluate

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MUTATOR_MODEL  = "gpt-4o-mini"
N_GENERATIONS  = 2
N_MUTATIONS    = 5
TOP_K          = 2          # survivors per generation
DATASET_FILE   = "math_dataset.json"
OUTPUT_FILE    = "best_prompt.json"
LOG_FILE       = "ape_log.json"

SEED_PROMPT = "You are a math olympiad expert, think step by step."

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Mutator system prompt ─────────────────────────────────────────────────────

MUTATOR_SYSTEM = """
You are a prompt engineer optimizing a system prompt for a math-solving LLM.

You will receive:
- The current best prompt
- A list of questions the model got wrong (with topics and difficulties)

Generate exactly 5 improved prompt variations. Each should try a DIFFERENT strategy:
  1. Change the persona (e.g. professor, competition coach, tutor)
  2. Change the reasoning style (e.g. verify answer, work backwards, identify traps)
  3. Add topic-specific instructions (based on which topics are failing)
  4. Change the output structure (e.g. label each step, state assumptions first)
  5. Combine the best elements of strategies above

Rules:
- Do NOT include the FINAL ANSWER tag — that is added automatically
- Each prompt must be a system prompt (second-person, instructional)
- Keep each prompt under 80 words
- Return ONLY a JSON array of 5 strings, no extra text

Example format:
["prompt 1", "prompt 2", "prompt 3", "prompt 4", "prompt 5"]
""".strip()


# ── Generate mutations ────────────────────────────────────────────────────────

def generate_mutations(best_prompt: str, failed_questions: list) -> list[str]:
    # Summarise failures for the mutator — topic + difficulty only, saves tokens
    failure_summary = [
        f"Q{q['id']} [{q['difficulty']}] [{q['topic']}] expected={q['expected']}"
        for q in failed_questions
    ]

    user_msg = (
        f"Current best prompt:\n{best_prompt}\n\n"
        f"Failed questions ({len(failure_summary)}):\n" +
        "\n".join(failure_summary)
    )

    response = client.chat.completions.create(
        model       = MUTATOR_MODEL,
        temperature = 0.8,      # higher temp = more diverse mutations
        max_tokens  = 600,
        messages    = [
            {"role": "system", "content": MUTATOR_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
    )

    raw = response.choices[0].message.content.strip()

    try:
        mutations = json.loads(raw)
        if isinstance(mutations, list) and len(mutations) == 5:
            return mutations
        raise ValueError("Expected list of 5")
    except Exception as e:
        print(f"  [mutator] Parse error: {e}\n  Raw: {raw[:200]}")
        # Fallback — return slight variations of seed to keep loop alive
        return [
            best_prompt + f" Focus on careful arithmetic.",
            best_prompt + f" Always verify your answer.",
            best_prompt + f" Identify the problem type before solving.",
            best_prompt + f" Show all intermediate steps clearly.",
            best_prompt + f" Be especially careful with probability and fractions.",
        ]


# ── APE Loop ──────────────────────────────────────────────────────────────────

def run_ape():
    log         = []
    best_prompt = SEED_PROMPT

    # ── Evaluate seed prompt first ──
    print(f"\n{'='*55}")
    print(f"  SEED EVALUATION")
    print(f"{'='*55}")
    seed_result = evaluate(best_prompt, verbose=True)
    best_score  = seed_result["score"]

    log.append({
        "generation" : 0,
        "type"       : "seed",
        "prompt"     : best_prompt,
        "score"      : best_score,
        "breakdown"  : seed_result["breakdown"],
    })

    print(f"\n  Seed score: {best_score:.0%}")

    # ── Generation loop ──
    for gen in range(1, N_GENERATIONS + 1):
        print(f"\n{'='*55}")
        print(f"  GENERATION {gen}")
        print(f"{'='*55}")

        # Get failing questions from last best run
        failed = [r for r in seed_result["results"] if not r["correct"]]
        print(f"\n  Generating {N_MUTATIONS} mutations based on {len(failed)} failures...\n")

        mutations = generate_mutations(best_prompt, failed)

        # ── Score all mutations ──
        gen_results = []
        for i, prompt in enumerate(mutations, 1):
            print(f"\n  ── Mutation {i}/{N_MUTATIONS} ──────────────────────────")
            print(f"  {prompt[:90]}{'...' if len(prompt) > 90 else ''}")
            result = evaluate(prompt, verbose=True)
            gen_results.append({
                "prompt"    : prompt,
                "score"     : result["score"],
                "breakdown" : result["breakdown"],
                "results"   : result["results"],
            })
            log.append({
                "generation": gen,
                "type"      : "mutation",
                "mutation"  : i,
                "prompt"    : prompt,
                "score"     : result["score"],
                "breakdown" : result["breakdown"],
            })

        # ── Rank and select top-k ──
        gen_results.sort(key=lambda x: x["score"], reverse=True)
        top = gen_results[:TOP_K]

        print(f"\n  ── Generation {gen} Rankings ──────────────────────")
        for rank, r in enumerate(gen_results, 1):
            marker = "✓ SURVIVES" if rank <= TOP_K else "✗"
            print(f"  [{rank}] {r['score']:.0%}  {marker}  {r['prompt'][:60]}...")

        # ── Update best ──
        if top[0]["score"] > best_score:
            best_score  = top[0]["score"]
            best_prompt = top[0]["prompt"]
            seed_result = top[0]   # use top result's failures for next gen
            print(f"\n  ✓ New best prompt found! Score: {best_score:.0%}")
        else:
            print(f"\n  ~ No improvement this generation. Best stays at {best_score:.0%}")
            seed_result = top[0]   # still use top mutation's failures

    # ── Save outputs ──
    best_output = {
        "best_prompt" : best_prompt,
        "best_score"  : best_score,
        "seed_prompt" : SEED_PROMPT,
        "improvement" : round(best_score - log[0]["score"], 4),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(best_output, f, indent=2)

    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

    # ── Final summary ──
    print(f"\n{'='*55}")
    print(f"  APE COMPLETE")
    print(f"{'='*55}")
    print(f"  Seed prompt score : {log[0]['score']:.0%}")
    print(f"  Best prompt score : {best_score:.0%}")
    print(f"  Improvement       : +{best_output['improvement']:.0%}")
    print(f"  Best prompt       : {best_prompt[:80]}...")
    print(f"\n  Saved → {OUTPUT_FILE}")
    print(f"  Log   → {LOG_FILE}")
    print(f"{'='*55}\n")

    return best_output


if __name__ == "__main__":
    run_ape()