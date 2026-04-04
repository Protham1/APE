# APE — Automated Prompt Engineer

A system that automatically finds the best prompt for solving math problems, then routes user queries to the right pipeline.

Instead of manually tweaking prompts, APE runs an optimization loop — it generates prompt variations, scores them against a dataset, and evolves toward the highest-performing prompt across multiple generations.

---

## How It Works

```
User Input
    ↓
Router (gpt-4o-mini classifier)
    ↓
┌─────────────┬──────────────────┐
│   Math QA   │  Email Evaluator │
│             │                  │
│ Best prompt │  LLM-as-judge    │
│ from APE    │  rubric scoring  │
└─────────────┴──────────────────┘
```

The APE optimization loop runs offline once:

```
Seed prompt → evaluate on 50 questions
    ↓
Generate 5 mutations (persona, reasoning style, format)
    ↓
Score each mutation
    ↓
Top 2 survive → repeat for N generations
    ↓
Best prompt saved to best_prompt.json
```

---

## Project Structure

```
APE/
├── main.py               # Entry point — routes query, loads best prompt
├── router.py             # Classifies input as math_qa or email_gen
├── math_evaluator.py     # Solves + grades math questions
├── email_evaluator.py    # LLM-as-judge email scoring
├── ape_loop.py           # APE optimization loop
├── math_dataset.json     # 50 questions (25 medium, 25 hard)
├── best_prompt.json      # Output of APE loop (auto-generated)
├── ape_log.json          # Full generation history (auto-generated)
├── requirements.txt
└── .env                  # Your OpenAI API key (never commit this)
```

---

## Setup

```bash
git clone https://github.com/Protham1/APE.git
cd APE
pip install -r requirements.txt
cp .env.example .env      # add your OpenAI API key
```

---

## Usage

**Step 1 — Run the APE optimization loop (once)**
```bash
python ape_loop.py
```
This runs ~1100 API calls on `gpt-4o-mini` and takes around 15-25 minutes. It saves the best prompt found to `best_prompt.json`.

**Step 2 — Run the pipeline**
```bash
python main.py
```
The pipeline automatically loads the optimized prompt. If `best_prompt.json` doesn't exist yet it falls back to the default seed prompt.

---

## Example Queries

```
Enter your query: solve 3x + 7 = 22
→ Router detects: math_qa
→ Uses APE-optimized prompt
→ FINAL ANSWER: 5

Enter your query: draft a follow-up email to a client about a delayed project
→ Router detects: email_gen
→ Evaluates on clarity, professionalism, grammar
→ Score: 87%
```

---

## Tech Stack

| Component       | Details                        |
|----------------|--------------------------------|
| Language        | Python 3.10+                   |
| Models          | gpt-4o-mini (all components)   |
| Routing         | Zero-shot classification       |
| Math eval       | Deterministic answer matching  |
| Email eval      | LLM-as-judge with rubric       |
| Prompt search   | Evolutionary mutation loop     |

---

## Dataset

50 math questions with no easy questions — intentionally hard to give APE room to improve:

| Difficulty | Count | Topics                              |
|-----------|-------|-------------------------------------|
| Medium    | 25    | Arithmetic, Algebra, Probability    |
| Hard      | 25    | Arithmetic, Algebra, Probability    |

---

## APE Results

| Generation | Best Score | Prompt Strategy         |
|-----------|------------|-------------------------|
| Seed      | ~75%       | Default step-by-step    |
| Gen 1     | TBD        | Mutated variations      |
| Gen 2     | TBD        | Refined best survivors  |

*(Update this table after running `ape_loop.py`)*