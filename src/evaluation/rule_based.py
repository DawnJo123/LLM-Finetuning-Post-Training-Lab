
import ast
import re


def count_words(text: str) -> int:
    """Count words in a response."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def evaluate_check(response: str, check: dict) -> dict:
    """Evaluate one rule-based check."""

    check_type = check["type"]
    expected = check["expected"]
    passed = False
    actual = None

    lines = [
        line.strip()
        for line in response.strip().splitlines()
        if line.strip()
    ]

    if check_type == "bullet_count":
        bullet_lines = [
            line for line in lines
            if re.match(r"^[-*+]\s+", line)
        ]
        actual = len(bullet_lines)
        passed = actual == expected

    elif check_type == "max_words_per_bullet":
        bullet_lines = [
            line for line in lines
            if re.match(r"^[-*+]\s+", line)
        ]
        actual = [
            count_words(re.sub(r"^[-*+]\s+", "", line))
            for line in bullet_lines
        ]
        passed = bool(actual) and all(
            word_count <= expected for word_count in actual
        )

    elif check_type == "forbidden_terms":
        actual = [
            term for term in expected
            if term.lower() in response.lower()
        ]
        passed = len(actual) == 0

    elif check_type == "code_only":
        try:
            if "```" in response:
                raise SyntaxError("Markdown code fences are not allowed")

            ast.parse(response.strip())
            actual = "Valid Python syntax"
            passed = True

        except SyntaxError:
            actual = "Invalid Python-only response"
            passed = False

    elif check_type == "contains":
        actual = [
            item for item in expected
            if item in response
        ]
        passed = len(actual) == len(expected)

    elif check_type == "required_headers":
        table_headers = []

        for line in lines:
            if line.startswith("|") and line.endswith("|"):
                cells = [
                    cell.strip()
                    for cell in line.strip("|").split("|")
                ]

                # Skip Markdown separator rows.
                if cells and all(
                    re.fullmatch(r":?-{3,}:?", cell)
                    for cell in cells
                ):
                    continue

                table_headers = cells
                break

        actual = table_headers
        passed = (
            table_headers == expected
            and len(table_headers) == len(expected)
        )

    elif check_type == "no_extra_text":
        is_bullet_list = all(
            re.match(r"^[-*+]\s+", line)
            for line in lines
        )

        is_markdown_table = all(
            line.startswith("|") and line.endswith("|")
            for line in lines
        )

        actual = (
            "bullet_list" if is_bullet_list
            else "markdown_table" if is_markdown_table
            else "extra_or_unexpected_text"
        )

        passed = (
            bool(lines)
            and (is_bullet_list or is_markdown_table)
        ) == expected

    elif check_type == "max_words":
        actual = count_words(response)
        passed = actual <= expected

    else:
        raise ValueError(f"Unsupported check type: {check_type}")

    return {
        "type": check_type,
        "passed": passed,
        "expected": expected,
        "actual": actual,
    }


def evaluate_rule_based(response: str, checks: list[dict]) -> dict:
    """Evaluate a response against all configured checks."""

    if not isinstance(response, str):
        raise TypeError("Response must be a string.")

    if not checks:
        raise ValueError("At least one evaluation check is required.")

    results = [
        evaluate_check(response, check)
        for check in checks
    ]

    passed_count = sum(
        result["passed"] for result in results
    )

    total_count = len(results)

    return {
        "passed": passed_count == total_count,
        "score": passed_count / total_count,
        "passed_checks": passed_count,
        "total_checks": total_count,
        "checks": results,
    }
