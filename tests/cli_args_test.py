"""CLI error-handling tests. They all fail before the model is loaded."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FUNCTIONS = ROOT / "data/input/functions_definition.json"
PROMPTS = ROOT / "data/input/function_calling_tests.json"


def run_cli(*args: str | Path) -> subprocess.CompletedProcess[str]:
    """Run ``python -m src`` from the repo root, like the subject does."""
    return subprocess.run(
        [sys.executable, "-m", "src", *map(str, args)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )


def write(tmp_path: Path, name: str, content: str) -> Path:
    """Write a fixture file and return its path."""
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_missing_functions_file(tmp_path: Path) -> None:
    result = run_cli("--functions_definition", tmp_path / "nope.json")
    assert result.returncode == 1
    assert "not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_input_file(tmp_path: Path) -> None:
    result = run_cli("--functions_definition", FUNCTIONS,
                     "--input", tmp_path / "nope.json")
    assert result.returncode == 1
    assert "not found" in result.stderr


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root ignores file permissions")
def test_unreadable_file(tmp_path: Path) -> None:
    path = write(tmp_path, "locked.json", "[]")
    path.chmod(0)
    result = run_cli("--functions_definition", path)
    assert result.returncode == 1
    assert "permission" in result.stderr


@pytest.mark.parametrize("content", ["[{", "", "not json", '[{"a": 1,}]'])
def test_invalid_json(tmp_path: Path, content: str) -> None:
    path = write(tmp_path, "bad.json", content)
    result = run_cli("--functions_definition", path)
    assert result.returncode == 1
    assert "not valid JSON" in result.stderr


@pytest.mark.parametrize("tools", [
    [],
    {"name": "fn"},
    [{"description": "no name"}],
    [{"name": "fn", "parameters": {"a": {}}}],
    [{"name": "fn", "parameters": "a"}],
])
def test_invalid_functions_schema(tmp_path: Path, tools: object) -> None:
    path = write(tmp_path, "tools.json", json.dumps(tools))
    result = run_cli("--functions_definition", path)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("prompts", [{"prompt": "hi"}, "hi", 3])
def test_prompts_not_a_list(tmp_path: Path, prompts: object) -> None:
    path = write(tmp_path, "prompts.json", json.dumps(prompts))
    result = run_cli("--functions_definition", FUNCTIONS, "--input", path)
    assert result.returncode == 1
    assert "JSON array" in result.stderr


def test_help() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    assert "--functions_definition" in result.stdout
