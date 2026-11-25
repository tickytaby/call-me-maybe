"""Entry point: ``uv run python -m src [--functions_definition F] [--input I]
[--output O]``.

Translates every natural-language prompt of the input file into a function
call, using constrained decoding (see ``decoder.py``), and writes the calls
to the output file as a JSON array.
"""

import argparse
import json
import os
import sys
import time
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .decoder import ConstrainedDecoder, ModelGenerationError
from .models import SUPPORTED_TYPES, FunctionCall, PromptItem, ToolDefinition
from .vocab import Vocabulary

DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


class InputError(Exception):
    """An input file is missing, unreadable, or malformed."""


def error(message: str) -> None:
    """Print an error or warning message to stderr.

    Args:
        message: Text to print.
    """
    print(message, file=sys.stderr)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        Namespace with ``functions_definition``, ``input`` and ``output``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into function calls.",
    )
    parser.add_argument("--functions_definition", default=DEFAULT_FUNCTIONS,
                        help="JSON file describing the available functions.")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="JSON file with the prompts to process.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Where to write the resulting function calls.")
    return parser.parse_args()


def load_json(path: str, what: str) -> Any:
    """Read and parse a JSON file, turning every failure into InputError.

    Args:
        path: File to read.
        what: Description used in error messages.

    Returns:
        The parsed JSON value.

    Raises:
        InputError: If the file is missing, unreadable or not valid JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise InputError(f"{what} file not found: {path}")
    except IsADirectoryError:
        raise InputError(f"{what} path is a directory: {path}")
    except PermissionError:
        raise InputError(f"no permission to read {what} file: {path}")
    except UnicodeDecodeError:
        raise InputError(f"{what} file is not UTF-8 text: {path}")
    except json.JSONDecodeError as e:
        raise InputError(f"{what} file is not valid JSON ({path}): {e}")
    except OSError as e:
        raise InputError(f"cannot read {what} file {path}: {e}")


def load_tools(path: str) -> list[ToolDefinition]:
    """Load and validate the function definitions.

    Args:
        path: Path to ``functions_definition.json``.

    Returns:
        The validated tools, duplicates removed (first one wins).

    Raises:
        InputError: If the file is invalid or defines no function.
    """
    raw = load_json(path, "functions definition")
    try:
        tools = TypeAdapter(list[ToolDefinition]).validate_python(raw)
    except ValidationError as e:
        raise InputError(f"invalid functions definition in {path}:\n{e}")
    if not tools:
        raise InputError(f"no functions defined in {path}")
    unique: dict[str, ToolDefinition] = {}
    for tool in tools:
        if tool.name in unique:
            error(f"warning: duplicate function {tool.name!r} ignored")
            continue
        for key, schema in tool.parameters.items():
            if schema.type not in SUPPORTED_TYPES:
                error(f"warning: {tool.name}.{key} has unsupported type "
                      f"{schema.type!r}; it will be decoded as a string")
        unique[tool.name] = tool
    return list(unique.values())


def load_prompts(path: str) -> list[str]:
    """Load the prompts, skipping (with a warning) malformed entries.

    Args:
        path: Path to the prompts file.

    Returns:
        The prompt strings, in file order.

    Raises:
        InputError: If the file is invalid or is not a JSON array.
    """
    raw = load_json(path, "input")
    if not isinstance(raw, list):
        raise InputError(f"{path} must contain a JSON array of objects")
    prompts: list[str] = []
    for i, item in enumerate(raw):
        try:
            prompts.append(PromptItem.model_validate(item).prompt)
        except ValidationError:
            error(f"warning: entry {i} of {path} is not "
                  f'{{"prompt": "<text>"}}; skipped')
    return prompts


def write_output(path: str, calls: list[FunctionCall]) -> None:
    """Write the calls as a JSON array, creating the directory if needed.

    Args:
        path: Output file path.
        calls: Calls to write.

    Raises:
        OSError: If the file cannot be written.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    data = [c.model_dump() for c in calls]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def load_model() -> tuple[Any, Vocabulary]:
    """Load the LLM and build the vocabulary tables.

    Returns:
        The ``Small_LLM_Model`` and its Vocabulary.
    """
    from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

    llm = Small_LLM_Model()
    vocab = Vocabulary.from_vocab_file(llm.get_path_to_vocab_file())
    return llm, vocab


def run(args: argparse.Namespace) -> int:
    """Process every prompt and write the results.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code: 0 on success, 1 on input error, 2 if the model
        cannot be loaded, 3 if the output cannot be written, 4 if some
        prompts failed (the others are still written).
    """
    try:
        tools = load_tools(args.functions_definition)
        prompts = load_prompts(args.input)
    except InputError as e:
        error(f"error: {e}")
        return 1

    print(f"Loading model ({len(tools)} functions, {len(prompts)} prompts)...")
    try:
        llm, vocab = load_model()
    except Exception as e:
        error(f"error: could not load the model: {e}")
        return 2
    decoder = ConstrainedDecoder(llm=llm, vocab=vocab, tools=tools)

    calls: list[FunctionCall] = []
    failed = 0
    start = time.perf_counter()
    for i, prompt in enumerate(prompts, 1):
        t0 = time.perf_counter()
        try:
            result = decoder.decode_call(prompt)
        except (ModelGenerationError, ValueError) as e:
            failed += 1
            error(f"[{i}/{len(prompts)}] FAILED {prompt!r}: {e}")
            continue
        calls.append(FunctionCall(
            prompt=prompt, name=result.name, parameters=result.parameters
        ))
        print(f"[{i}/{len(prompts)}] {prompt!r}\n    -> {result.json_text}"
              f"  ({result.passes} passes, {time.perf_counter() - t0:.1f}s)")
        for key in result.truncated:
            error(f"    warning: value of {key!r} hit the token limit "
                  "and was truncated")

    try:
        write_output(args.output, calls)
    except (OSError, ValueError) as e:
        error(f"error: could not write {args.output}: {e}")
        return 3
    print(f"Wrote {len(calls)} calls to {args.output} "
          f"in {time.perf_counter() - start:.1f}s")
    if failed:
        error(f"{failed} prompt(s) failed and were left out of the output")
        return 4
    return 0


def main() -> int:
    """Program entry point.

    Returns:
        Process exit code.
    """
    try:
        return run(parse_args())
    except KeyboardInterrupt:
        error("interrupted")
        return 130
    except Exception as e:  # last line of defence: never show a traceback
        error(f"unexpected error: {type(e).__name__}: {e}")
        return 5


if __name__ == "__main__":
    sys.exit(main())
