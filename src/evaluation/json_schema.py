
import json


def check_type(value, expected_type: str) -> bool:
    """Check a Python value against a JSON field type."""

    if expected_type == "string":
        return isinstance(value, str)

    if expected_type == "boolean":
        return isinstance(value, bool)

    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)

    raise ValueError(f"Unsupported field type: {expected_type}")


def evaluate_json_schema(response: str, evaluation: dict) -> dict:
    """Parse and validate a model response against its JSON schema configuration."""

    results = []

    def add_result(check_type: str, passed: bool, expected, actual):
        results.append({
            "type": check_type,
            "passed": passed,
            "expected": expected,
            "actual": actual,
        })

    # 1. Parse JSON
    try:
        parsed = json.loads(response)
        add_result("valid_json", True, "Valid JSON", "Valid JSON")
    except json.JSONDecodeError as error:
        add_result("valid_json", False, "Valid JSON", str(error))
        return {
            "passed": False,
            "score": 0.0,
            "passed_checks": 0,
            "total_checks": 1,
            "checks": results,
        }

    # 2. Validate the expected top-level structure
    structure = evaluation.get("structure", "object")

    if structure == "object":
        correct_structure = isinstance(parsed, dict)
    elif structure == "array":
        correct_structure = isinstance(parsed, list)
    else:
        raise ValueError(f"Unsupported JSON structure: {structure}")

    add_result(
        "structure",
        correct_structure,
        structure,
        "object" if isinstance(parsed, dict)
        else "array" if isinstance(parsed, list)
        else type(parsed).__name__,
    )

    if not correct_structure:
        passed_count = sum(result["passed"] for result in results)
        return {
            "passed": False,
            "score": passed_count / len(results),
            "passed_checks": passed_count,
            "total_checks": len(results),
            "checks": results,
        }

    # 3. Validate JSON object fields
    if structure == "object":
        required_keys = evaluation.get("required_keys", [])
        actual_keys = list(parsed.keys())

        missing_keys = [
            key for key in required_keys if key not in parsed
        ]

        add_result(
            "required_keys",
            len(missing_keys) == 0,
            required_keys,
            actual_keys,
        )

        if evaluation.get("exact_keys", False):
            exact_keys = (
                len(parsed) == len(required_keys)
                and set(parsed.keys()) == set(required_keys)
            )
            add_result(
                "exact_keys",
                exact_keys,
                required_keys,
                actual_keys,
            )

        # Validate field types
        field_types = evaluation.get("field_types", {})

        for field, expected_type in field_types.items():
            if field not in parsed:
                passed = False
                actual_type = "missing"
            else:
                value = parsed[field]
                passed = check_type(value, expected_type)
                actual_type = type(value).__name__

            add_result(
                f"field_type:{field}",
                passed,
                expected_type,
                actual_type,
            )

        # Validate allowed values
        allowed_values = evaluation.get("allowed_values", {})

        for field, allowed in allowed_values.items():
            actual = parsed.get(field)
            passed = field in parsed and actual in allowed

            add_result(
                f"allowed_values:{field}",
                passed,
                allowed,
                actual,
            )

        # Validate numeric ranges
        numeric_ranges = evaluation.get("numeric_ranges", {})

        for field, limits in numeric_ranges.items():
            value = parsed.get(field)

            passed = (
                field in parsed
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and limits["min"] <= value <= limits["max"]
            )

            add_result(
                f"numeric_range:{field}",
                passed,
                limits,
                value,
            )

    # 4. Validate JSON arrays
    elif structure == "array":
        expected_count = evaluation.get("expected_item_count")

        if expected_count is not None:
            add_result(
                "item_count",
                len(parsed) == expected_count,
                expected_count,
                len(parsed),
            )

        required_keys = evaluation.get("item_required_keys", [])
        field_types = evaluation.get("item_field_types", {})
        item_exact_keys = evaluation.get("item_exact_keys", False)

        for index, item in enumerate(parsed):
            is_object = isinstance(item, dict)

            add_result(
                f"item_{index}_is_object",
                is_object,
                "object",
                type(item).__name__,
            )

            if not is_object:
                continue

            missing_keys = [
                key for key in required_keys if key not in item
            ]

            add_result(
                f"item_{index}_required_keys",
                len(missing_keys) == 0,
                required_keys,
                list(item.keys()),
            )

            if item_exact_keys:
                exact_keys = (
                    len(item) == len(required_keys)
                    and set(item.keys()) == set(required_keys)
                )

                add_result(
                    f"item_{index}_exact_keys",
                    exact_keys,
                    required_keys,
                    list(item.keys()),
                )

            for field, expected_type in field_types.items():
                if field not in item:
                    passed = False
                    actual_type = "missing"
                else:
                    passed = check_type(item[field], expected_type)
                    actual_type = type(item[field]).__name__

                add_result(
                    f"item_{index}_field_type:{field}",
                    passed,
                    expected_type,
                    actual_type,
                )

        # Check that required category values are represented.
        required_categories = evaluation.get(
            "required_category_values", []
        )

        if required_categories:
            actual_categories = [
                item.get("category")
                for item in parsed
                if isinstance(item, dict) and "category" in item
            ]

            missing_categories = [
                category for category in required_categories
                if category not in actual_categories
            ]

            add_result(
                "required_category_values",
                len(missing_categories) == 0,
                required_categories,
                actual_categories,
            )

    passed_count = sum(result["passed"] for result in results)
    total_count = len(results)

    return {
        "passed": passed_count == total_count,
        "score": passed_count / total_count if total_count else 0.0,
        "passed_checks": passed_count,
        "total_checks": total_count,
        "checks": results,
    }
