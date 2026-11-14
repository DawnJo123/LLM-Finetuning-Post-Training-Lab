
from src.evaluation.dataset import load_evaluation_dataset
from src.evaluation.coding import evaluate_coding


def main():
    dataset = load_evaluation_dataset(
        "data/evaluation/baseline_eval.jsonl"
    )

    coding_examples = [
        example for example in dataset
        if example["category"] == "coding"
    ]

    sample_responses = {
        "CODE-001": """
def remove_duplicates(nums):
    return list(dict.fromkeys(nums))
""",

        "CODE-002": """
def count_frequency(items):
    frequency = {}
    for item in items:
        frequency[item] = frequency.get(item, 0) + 1
    return frequency
""",

        "CODE-003": """
def is_palindrome(text):
    cleaned = ''.join(
        char.lower() for char in text if char.isalnum()
    )
    return cleaned == cleaned[::-1]
""",

        "CODE-004": """
def calculate_mean(values):
    if not values:
        return None
    return sum(values) / len(values)
""",

        "CODE-005": """
import torch

def normalize_tensor(x):
    minimum = x.min()
    maximum = x.max()

    if maximum == minimum:
        return torch.zeros_like(x)

    return (x - minimum) / (maximum - minimum)
""",
    }

    for example in coding_examples:
        result = evaluate_coding(
            response=sample_responses[example["id"]],
            evaluation=example["evaluation"],
        )

        print(
            f"{example['id']} | "
            f"Passed: {result['passed']} | "
            f"Score: {result['score']:.2f} | "
            f"Tests: {result['passed_checks']}/"
            f"{result['total_checks']}"
        )

        if "error" in result:
            print(f"  Error: {result['error']}")

        for check in result["checks"]:
            print(
                f"  Test {check['test_case']}: "
                f"{'PASS' if check['passed'] else 'FAIL'}"
            )

            if not check["passed"]:
                print(f"    Details: {check}")

        print()


if __name__ == "__main__":
    main()
