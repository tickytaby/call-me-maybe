# Failure modes — status update

Re-audit of `failure_modes.md` against the current working tree (pydantic
migration: `src/models.py` + `TypeAdapter` validation in `src/__main__.py`,
plus new `tests/cli_args_test.py` cases and `data/input/bad_schema_*.json`
fixtures).

## Solved

### 1. `tools_list` isn't the expected shape — SOLVED
`src/__main__.py:474-478` now does
`tools = TypeAdapter(list[ToolDefinition]).validate_python(tools_list)`
right after the JSON load, before anything indexes into it. A dict-shaped
top-level (`{"tools": [...]}`) or a list of dicts missing `"name"` both raise
`pydantic.ValidationError`, caught and turned into a clean
`return 8` with the schema error printed — no more raw `TypeError`/`KeyError`.
Covered by `tests/cli_args_test.py::test_invalid_tools_schema` using the new
fixture `data/input/bad_schema_tools.json` (a tool list with no `"name"`
field), asserting `returncode == 8`.

### 2. Malformed tool schema inside `Util.get_parameters` — SOLVED
The old free-form `Util.get_parameters`/`Tool(BaseModel)` combo (which
manually reached into `tool["parameters"]` and each `v["type"]`) is gone.
`ParameterSchema` (`src/models.py:6-7`) declares
`type: Literal["number", "string", "boolean"]`, and `ToolDefinition.parameters`
(`src/models.py:13`) is typed `dict[str, ParameterSchema]`. The same
`TypeAdapter(list[ToolDefinition]).validate_python(...)` call that fixes #1
also rejects a missing `"parameters"` key (defaults to `{}` via
`Field(default_factory=dict)` instead of `KeyError`ing), a non-dict
`"parameters"`, or a parameter entry missing/misvaluing `"type"` — all become
`ValidationError` → `return 8`, exercised by the same
`test_invalid_tools_schema` fixture.

### 4. `prompts` isn't a list of `{"prompt": ...}` dicts — SOLVED
`src/__main__.py:503-507` validates the loaded prompts with
`TypeAdapter(list[PromptItem]).validate_python(prompts)`
(`PromptItem` in `src/models.py:17-18` requires a non-empty `prompt: str`)
before the old unguarded `prompts_str = [p["prompt"] for p in prompts]` list
comprehension runs. A wrong key (`"text"` instead of `"prompt"`) or a
dict-instead-of-list payload now raises `ValidationError` → `return 9`
instead of an uncaught `KeyError`/`TypeError`. Covered by
`test_invalid_prompts_schema` using the new
`data/input/bad_schema_prompts.json` fixture (`{"text": ...}` instead of
`{"prompt": ...}`), asserting `returncode == 9`.

## Not solved yet

### 3. `tools_list` is empty (`[]`) — STILL OPEN
`TypeAdapter(list[ToolDefinition]).validate_python([])` succeeds — an empty
list is a structurally valid `list[ToolDefinition]`, so pydantic doesn't
reject it. `tools_dict` ends up `{}`, the trie stays empty, `choose_fn`'s
`while len(valid_token_ids):` loop (`src/__main__.py:107-116`) never runs,
`answer` decodes to `""`, and `fill_in_parameters(..., tool_name="")` still
does the unguarded `tool = tools[tool_name]` (`src/__main__.py:348`) →
uncaught `KeyError`. The pydantic migration validates *shape*, not
"non-empty," so this failure mode is untouched.

### 5. `Small_LLM_Model()` construction — STILL OPEN
`src/__main__.py:480`, `llm = Small_LLM_Model()`, still has no
`try`/`except` around it. Missing/corrupt checkpoints, CUDA OOM, etc. still
produce a raw traceback.

### 6. Model-chosen tool name doesn't match a dict key — STILL OPEN
`fill_in_parameters` (`src/__main__.py:345-348`) still does
`tool = tools[tool_name]` with no guard. This is now the *only* remaining
path (besides #3 above) by which a bad `tool_name` reaches `fill_in_parameters`
uncaught — pydantic validation constrains the *input files*, not what the
model itself decodes at runtime.

### 7. Unbounded "thinking" loop in `choose_fn` — STILL OPEN
`src/__main__.py:99-103`,
`while idx != END_THINK_TOK: ...`, is still uncapped, unlike
`_let_model_think`'s `MAX_THINK_TOKENS`/`MAX_PLAN_THINK_TOKENS`-bounded loops.
Same hang/context-overflow risk as before.

### 8. Context-length overflow anywhere `get_logits_from_input_ids` is called — STILL OPEN
None of the call sites (`src/__main__.py:93, 100, 109/118, ...` plus the
`_fill_*_value`/`_let_model_think` helpers) wrap this call. Unchanged.

### 9. Empty-token encode edge case — PARTIALLY MITIGATED, NOT SOLVED
`src/__main__.py:484`,
`tokenized_tools = [llm.encode(tool).tolist()[0] for tool in tool_names]`,
is still unguarded. `ToolDefinition.name` now requires
`Field(min_length=1)` (`src/models.py:11`), which rules out the empty-string
case, but does **not** rule out a name that is non-empty text yet still
encodes to zero tokens (e.g. a name consisting only of whitespace, or a
character sequence the tokenizer maps to nothing) — `min_length` is a string
length check, not a tokenizer-output check. `IndexError` is still reachable.

## Net effect

The pydantic migration (`src/models.py` + the two new `TypeAdapter` calls in
`src/__main__.py`, backed by `tests/cli_args_test.py::test_invalid_tools_schema`
and `test_invalid_prompts_schema`) closes the **file-validation** gaps: items
1, 2, and 4 from the original audit. It does not touch the **model/runtime**
gaps (5, 7, 8), the **empty-tools-list** gap (3), or the **encode-edge-case**
gap (9), and it narrows but doesn't close the **bad-decoded-tool-name** gap
(6) — `fill_in_parameters`'s `tools[tool_name]` lookup is still an unguarded
`KeyError` waiting to happen, now reachable only via runtime model behavior
(items 3 and 6) rather than via malformed input files.
