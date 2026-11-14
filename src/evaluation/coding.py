
import json
import re
import subprocess
import sys
import tempfile

from pathlib import Path


# This script runs in a separate Python process.
RUNNER_SCRIPT = r'''
import contextlib
import io
import json
import sys
import traceback

payload = json.load(sys.stdin)
namespace = {}

try:
    with contextlib.redirect_stdout(sys.stderr):
        exec(compile(payload["code"], "<model_response>", "exec"), namespace)

    function = namespace.get(payload["function_name"])

    if not callable(function):
        raise ValueError(
            f"Function '{payload['function_name']}' was not found."
        )

except Exception as error:
    print(json.dumps({
        "setup_error": str(error),
        "results": []
    }))
    sys.exit(0)


results = []

for index, test_case in enumerate(payload["test_cases"]):
    try:
        with contextlib.redirect_stdout(sys.stderr):
            args = test_case["input"]
            if payload.get("input_types"):
                import torch
                args = [
                    torch.tensor(arg) if input_type == "torch_tensor" else arg
                    for arg, input_type in zip(args, payload["input_types"])
                ]

            actual=function(*args)

        expected = test_case["expected"]

        # Convert PyTorch tensors into ordinary Python lists.
        if hasattr(actual, "detach") and hasattr(actual, "cpu"):
            actual = actual.detach().cpu().tolist()

        if payload["comparison"] == "allclose":
            import torch

            passed = torch.allclose(
                torch.as_tensor(actual),
                torch.as_tensor(expected),
                atol=payload["tolerance"],
                rtol=0,
            )
        else:
            passed = actual == expected

        results.append({
            "test_case": index + 1,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
        })

    except Exception as error:
        results.append({
            "test_case": index + 1,
            "passed": False,
            "error": str(error),
        })


print(json.dumps({"results": results}))
'''


def extract_code(response: str) -> str:
    """Remove a surrounding Markdown Python code fence if present."""

    response = response.strip()

    match = re.fullmatch(
        r"```(?:python)?\s*(.*?)\s*```",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return match.group(1).strip() if match else response


def evaluate_coding(
    response: str,
    evaluation: dict,
    timeout: int = 60,
) -> dict:
    """Execute generated Python code against configured test cases."""

    if not isinstance(response, str):
        raise TypeError("Response must be a string.")

    function_name = evaluation.get("function_name")
    test_cases = evaluation.get("test_cases", [])

    if not function_name:
        raise ValueError("Evaluation requires a function_name.")

    if not test_cases:
        raise ValueError("At least one test case is required.")

    comparison = evaluation.get("comparison", "exact")

    if comparison not in {"exact", "allclose"}:
        raise ValueError(f"Unsupported comparison method: {comparison}")

    payload = {
        "code": extract_code(response),
        "function_name": function_name,
        "test_cases": test_cases,
        "comparison": comparison,
        "input_types": evaluation.get("input_types", []),
        "tolerance": evaluation.get("tolerance", 0.0001),
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = Path(temp_dir) / "runner.py"
        script_path.write_text(RUNNER_SCRIPT, encoding="utf-8")

        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(script_path),
                ],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "score": 0.0,
                "passed_checks": 0,
                "total_checks": len(test_cases),
                "error": f"Execution timed out after {timeout} seconds.",
                "checks": [],
            }

    try:
        output = json.loads(process.stdout)
    except json.JSONDecodeError:
        return {
            "passed": False,
            "score": 0.0,
            "passed_checks": 0,
            "total_checks": len(test_cases),
            "error": process.stderr or "Could not parse evaluator output.",
            "checks": [],
        }

    if "setup_error" in output:
        return {
            "passed": False,
            "score": 0.0,
            "passed_checks": 0,
            "total_checks": len(test_cases),
            "error": output["setup_error"],
            "checks": [],
        }

    results = output["results"]

    passed_count = sum(
        result["passed"] for result in results
    )

    total_count = len(test_cases)

    return {
        "passed": passed_count == total_count,
        "score": passed_count / total_count,
        "passed_checks": passed_count,
        "total_checks": total_count,
        "checks": results,
    }
