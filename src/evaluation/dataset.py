import json
from pathlib import Path

VALID_CATEGORIES={
    "instruction_following",
    "technical_knowledge",
    "reasoning",
    "summarization",
    "structured_output",
    "coding"
}

REQUIRED_FIELDS={
    "id",
    "category",
    "task_type",
    "prompt",
    "reference_answer",
    "evaluation",
}

def load_evaluation_dataset(file_path:str)-> list[dict]:
    """Load and validate evaluation examples froma JSON file"""

    path=Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    examples=[]
    seen_ids=set()

    with path.open("r",encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                example=json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}:{error}"
                ) from error

            missing= REQUIRED_FIELDS - example.keys()
            if missing:
                raise ValueError(
                    f"Missing fields on line {line_number}: {missing}"
                )

            if example["category"] not in VALID_CATEGORIES:
                raise ValueError(
                    f"Invalid category on line {line_number}: "
                    f"{example["category"]}"
                )

            if example["id"] in seen_ids:
                raise ValueError(f"Duplicate ID:{example["id"]}")


            if not isinstance(example["evaluation"], dict):
                raise ValueError(
                    f"Evaluation must be an object on line {line_number}"
                )

            seen_ids.add(example["id"])
            examples.append(example)

        return examples