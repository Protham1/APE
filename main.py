import json
from Router import classify
from email_evaluator import evaluate_email


def run_pipeline(query: str):
    route = classify(query)

    print("\n[Router Output]")
    print(route)

    if route.task_type.value == "email_eval":
        result = evaluate_email(query)

        print("\n[Evaluation]")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    q = input("Enter query:\n")
    run_pipeline(q)