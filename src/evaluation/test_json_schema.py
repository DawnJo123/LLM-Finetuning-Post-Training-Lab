
from src.evaluation.dataset import load_evaluation_dataset
from src.evaluation.json_schema import evaluate_json_schema


def test_invalid_responses():
    test_cases = [
        {
            "name": "Invalid JSON",
            "response": '{"person": "Alex",}',
            "evaluation": {"method": "json_schema"},
        },
        {
            "name": "Extra object field",
            "response": (
                '{"name":"ResNet50","framework":"PyTorch",'
                '"parameters_millions":25.6,'
                '"is_pretrained":true,"extra":"unexpected"}'
            ),
            "evaluation": {
                "method": "json_schema",
                "required_keys": [
                    "name", "framework",
                    "parameters_millions", "is_pretrained"
                ],
                "exact_keys": True,
            },
        },
        {
            "name": "Invalid numeric range",
            "response": '{"sentiment":"positive","confidence":1.5}',
            "evaluation": {
                "method": "json_schema",
                "required_keys": ["sentiment", "confidence"],
                "numeric_ranges": {
                    "confidence": {"min": 0, "max": 1}
                },
            },
        },
        {
            "name": "Incorrect integer type",
            "response": (
                '{"learning_rate":0.001,"batch_size":32.5,'
                '"optimizer":"AdamW","use_gpu":true}'
            ),
            "evaluation": {
                "method": "json_schema",
                "required_keys": [
                    "learning_rate", "batch_size",
                    "optimizer", "use_gpu"
                ],
                "field_types": {
                    "learning_rate": "number",
                    "batch_size": "integer",
                    "optimizer": "string",
                    "use_gpu": "boolean",
                },
            },
        },
    ]

    print("\nINVALID RESPONSE TESTS")
    print("=" * 45)

    for case in test_cases:
        result = evaluate_json_schema(
            case["response"],
            case["evaluation"],
        )

        status = "PASS" if not result["passed"] else "FAIL"

        print(f"{case['name']}: {status}")

    

def main():
    dataset = load_evaluation_dataset(
        "data/evaluation/baseline_eval.jsonl"
    )

    structured_examples = [
        example for example in dataset
        if example["category"] == "structured_output"
    ]

    sample_responses = {
        "STRUCT-001": (
            '{"name":"ResNet50","framework":"PyTorch",'
            '"parameters_millions":25.6,"is_pretrained":true}'
        ),
        "STRUCT-002": (
            '{"person":"Alex","role":"Machine Learning Engineer",'
            '"company":"NeuralTech","location":"Bangalore"}'
        ),
        "STRUCT-003": (
            '[{"algorithm":"Linear Regression","category":"supervised"},'
            '{"algorithm":"K-Means","category":"unsupervised"},'
            '{"algorithm":"Q-Learning","category":"reinforcement learning"}]'
        ),
        "STRUCT-004": (
            '{"sentiment":"positive","confidence":0.95}'
        ),
        "STRUCT-005": (
            '{"learning_rate":0.001,"batch_size":32,'
            '"optimizer":"AdamW","use_gpu":true}'
        ),
    }

    for example in structured_examples:
        result = evaluate_json_schema(
            response=sample_responses[example["id"]],
            evaluation=example["evaluation"],
        )

        print(
            f"{example['id']} | "
            f"Passed: {result['passed']} | "
            f"Score: {result['score']:.2f} | "
            f"Checks: {result['passed_checks']}/{result['total_checks']}"
        )

        for check in result["checks"]:
            print(
                f"  {check['type']}: "
                f"{'PASS' if check['passed'] else 'FAIL'} "
                f"(actual: {check['actual']})"
            )

        print()

    test_invalid_responses()
if __name__ == "__main__":
    main()
