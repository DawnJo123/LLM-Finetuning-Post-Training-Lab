
from src.evaluation.dataset import load_evaluation_dataset


def main():
    dataset = load_evaluation_dataset(
        "data/evaluation/baseline_eval.jsonl"
    )

    print(f"Loaded {len(dataset)} evaluation examples.")

    for example in dataset:
        print(
            f"{example['id']} | "
            f"{example['category']} | "
            f"{example['task_type']}"
        )


if __name__ == "__main__":
    main()
