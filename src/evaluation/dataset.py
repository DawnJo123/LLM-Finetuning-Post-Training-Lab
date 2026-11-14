
import json
from pathlib import Path


VALID_CATEGORIES = {
    "instruction_following",
    "technical_knowledge",
    "reasoning",
    "summarization",
    "structured_output",
    "coding",
}

VALID_EVALUATION_METHODS = {
    "rubric",
    "rule_based",
    "json_schema",
    "unit_tests",
}

REQUIRED_FIELDS = {
    "id",
    "category",
    "task_type",
    "prompt",
    "reference_answer",
    "evaluation",
}


def validate_example(example: dict, line_number: int) -> None:
    """Validate an individual evaluation record."""

    record_id = example.get("id", "<unknown>")

    # Validate required fields
    missing = REQUIRED_FIELDS - example.keys()
    if missing:
        raise ValueError(
            f"Missing fields in record {record_id}, "
            f"line {line_number}: {missing}"
        )

    # Validate string fields
    string_fields = [
        "id",
        "category",
        "task_type",
        "prompt",
        "reference_answer",
    ]

    for field in string_fields:
        if not isinstance(example[field], str) or not example[field].strip():
            raise ValueError(
                f"Field '{field}' must be a non-empty string "
                f"in record {record_id}, line {line_number}"
            )

    # Validate category
    if example["category"] not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{example['category']}' "
            f"in record {record_id}, line {line_number}"
        )

    # Validate evaluation object
    evaluation = example["evaluation"]

    if not isinstance(evaluation, dict):
        raise ValueError(
            f"Evaluation must be an object "
            f"in record {record_id}, line {line_number}"
        )

    method = evaluation.get("method")

    if method not in VALID_EVALUATION_METHODS:
        raise ValueError(
            f"Invalid evaluation method '{method}' "
            f"in record {record_id}, line {line_number}"
        )

    # Validate method-specific configuration
    if method == "rubric":
        criteria = evaluation.get("criteria")

        if not isinstance(criteria, list) or not criteria:
            raise ValueError(
                f"Rubric evaluation requires non-empty criteria "
                f"in record {record_id}"
            )

    elif method == "rule_based":
        checks = evaluation.get("checks")

        if not isinstance(checks, list) or not checks:
            raise ValueError(
                f"Rule-based evaluation requires non-empty checks "
                f"in record {record_id}"
            )

    elif method == "json_schema":
        schema_fields = {
            "required_keys",
            "structure",
            "item_required_keys",
        }

        if not schema_fields.intersection(evaluation.keys()):
            raise ValueError(
                f"JSON schema evaluation requires structural "
                f"configuration in record {record_id}"
            )

    elif method == "unit_tests":
        test_cases = evaluation.get("test_cases")

        if not isinstance(test_cases, list) or not test_cases:
            raise ValueError(
                f"Unit test evaluation requires non-empty test_cases "
                f"in record {record_id}"
            )


def load_evaluation_dataset(file_path: str) -> list[dict]:
    """Load and validate evaluation examples from a JSONL file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    examples = []
    seen_ids = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                example = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            if not isinstance(example, dict):
                raise ValueError(
                    f"Each JSONL record must be an object "
                    f"on line {line_number}"
                )

            validate_example(example, line_number)

            record_id = example["id"]

            if record_id in seen_ids:
                raise ValueError(
                    f"Duplicate ID '{record_id}' on line {line_number}"
                )

            seen_ids.add(record_id)
            examples.append(example)

    return examples
