# 📌 APE System (Math QA + Email Generator)

## 🚀 Overview
This project implements a lightweight **Autonomous Prompt Engineering (APE)** system.

It supports:
- 🧮 Solving math problems (Math QA)
- ✉️ Generating emails based on user intent

The system uses an **evaluation loop** to refine outputs and improve performance over time.

---

## ⚙️ Features
- **Math QA**
  - Solves basic to intermediate math problems
  - Evaluated using a dataset (`math_dataset.json`)

- **Email Generation (via routing)**
  - Uses task classification to determine output type

- **Evaluation System**
  - Compares model outputs with expected answers
  - Stores results in `eval_result.json`

- **Routing Logic**
  - Determines whether input is math or email (`Router.py`)
  - Outputs stored in `router_output.json`

---

## 📂 Project Structure

AN2/
├── .env # Environment variables
├── .gitignore
├── eval_result.json # Stores evaluation results
├── math_dataset.json # Dataset for math QA testing
├── math_evaluator.py # Evaluates math outputs
├── README.md
├── requirements.txt
├── router_output.json # Stores router decisions
├── Router.py # Task routing logic

---


---

## 🛠️ Tech Stack
- Python
- JSON (for dataset + outputs)
- LLM (API or local model)

---

## ▶️ How to Run
```bash
git clone https://github.com/Protham1/APE.git
cd AN2
pip install -r requirements.txt
python Router.py
python math_evaluator.py