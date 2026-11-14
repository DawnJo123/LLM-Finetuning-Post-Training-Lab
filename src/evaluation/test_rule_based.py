
from src.evaluation.dataset import load_evaluation_dataset
from src.evaluation.rule_based import evaluate_rule_based


def main():
    dataset = load_evaluation_dataset(
        "data/evaluation/baseline_eval.jsonl"
    )

    instruction_examples = [
        example for example in dataset
        if example["category"] == "instruction_following"
    ]

    sample_responses = {
        "IF-001": (
            "- Overfitting occurs when a model memorizes training data.\n"
            "- It learns noise instead of general patterns.\n"
            "- Consequently, it performs poorly on unseen data."
        ),
        "IF-002": (
            "Backpropagation calculates how much each network parameter "
            "contributes to prediction error. This information helps "
            "adjust the parameters during training."
        ),
        "IF-003": (
            "def is_even(n):\n"
            "    return n % 2 == 0"
        ),
        "IF-004": (
            "| Model type | Training data | Main objective |\n"
            "|---|---|---|\n"
            "| Supervised learning | Labeled data | Predict outputs |\n"
            "| Unsupervised learning | Unlabeled data | Discover patterns |"
        ),
        "IF-005": (
            "A CNN processes images using convolutional filters "
            "to identify patterns such as edges, textures, and shapes."
        ),
    }

    for example in instruction_examples:
        result = evaluate_rule_based(
            response=sample_responses[example["id"]],
            checks=example["evaluation"]["checks"],
        )

        print(
            f"{example['id']} | "
            f"Passed: {result['passed']} | "
            f"Score: {result['score']:.2f}"
        )

        for check in result["checks"]:
            print(
                f"  {check['type']}: "
                f"{'PASS' if check['passed'] else 'FAIL'} "
                f"(actual: {check['actual']})"
            )

        print()


if __name__ == "__main__":
    main()
