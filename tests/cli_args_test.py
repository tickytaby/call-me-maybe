import subprocess
import sys


def run_cli(args):
    return subprocess.run(
        [sys.executable, "../src/__main__.py", *args],
        capture_output=True,
        text=True,
    )


def test_filenotfounderror():
    result = run_cli(
        [
            "--functions_definition",
            "non-existent-file",
            "--input",
            "../data/input/function_calling_tests.json",
            "--output",
            "../data/output/output.json",
        ]
    )
    assert result.returncode == 1


def test_permissionerror():
    result = run_cli(
        [
            "--functions_definition",
            "../data/input/nopermissions.json",
            "--input",
            "../data/input/function_calling_tests.json",
            "--output",
            "../data/output/output.json",
        ]
    )
    assert result.returncode == 2


def test_invalidjson():
    result = run_cli(
        [
            "--functions_definition",
            "../data/input/invalid.json",
            "--input",
            "../data/input/function_calling_tests.json",
            "--output",
            "../data/output/output.json",
        ]
    )
    assert result.returncode == 3


def test_failed_write_output():
    result = run_cli(
        [
            "--functions_definition",
            "../data/input/functions_definition.json",
            "--input",
            "../data/input/function_calling_tests.json",
            "--output",
            "../data/output/nopermissions_output.json",
        ]
    )
    assert result.returncode == 7


def test_valid_run():
    result = run_cli(
        [
            "--functions_definition",
            "../data/input/functions_definition.json",
            "--input",
            "../data/input/function_calling_tests.json",
            "--output",
            "../data/output/valid_run_test.json",
        ]
    )
    assert result.returncode == 0
