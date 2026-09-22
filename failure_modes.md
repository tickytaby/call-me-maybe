# Failure modes not covered by try/except

Audit of `src/__main__.py`. Each item is a way the script crashes (or hangs)
with an unhandled exception in production, even though the file-loading code
around it looks defensive.

## Data loaded from files, used without validation

### 1. `tools_list` isn't the expected shape — `src/__main__.py:501-502`
```python
for tool in tools_list:
    tools_dict[tool["name"]] = Tool(tool)
```
`functions_definition.json` parses as valid JSON (satisfies the `try` block
that loads it) but has the wrong top-level shape.
- `{"tools": [...]}` instead of a top-level list → iterating a dict yields
  string keys → `tool["name"]` raises `TypeError: string indices must be
  integers`.
- `[{"description": "no name field"}]` → `KeyError: 'name'`.

### 2. Malformed tool schema inside `Util.get_parameters` — `src/__main__.py:59-72`
- `tool["parameters"]` missing → `KeyError: 'parameters'`.
- `"parameters"` present but not a dict, e.g. `"parameters": []` →
  `.items()` → `AttributeError: 'list' object has no attribute 'items'`.
- A parameter entry missing `"type"`, e.g. `{"foo": {}}` → `KeyError: 'type'`.

### 3. `tools_list` is empty (`[]`)
The trie ends up empty, so in `choose_fn` (`src/__main__.py:130`) the
`while len(valid_token_ids):` loop never runs and `answer` decodes to
essentially nothing/garbage. That string is then used as `tool_name` in
`fill_in_parameters` → `tools[tool_name].template` (`src/__main__.py:382`) →
`KeyError` on the bogus tool name.

### 4. `prompts` isn't a list of `{"prompt": ...}` dicts — `src/__main__.py:524`
```python
prompts_str = [p["prompt"] for p in prompts]
```
This line sits *outside* the `try` block that loads `args.input`
(`src/__main__.py:516-522`), so a structurally wrong (but JSON-valid) file
still crashes uncaught.
- `[{"text": "do a thing"}]` (wrong key) → `KeyError: 'prompt'`.
- `{"prompts": [...]}` (dict, not list) → iterating gives keys →
  `TypeError: string indices must be integers`.

## Model / runtime

### 5. `Small_LLM_Model()` construction — `src/__main__.py:499`
No `try`/`except` around model instantiation at all. Missing/corrupted
checkpoint files, CUDA OOM, incompatible weight format, network failure
fetching weights, etc. all produce an unhandled traceback and default exit
code 1, even though every file-loading step before it succeeded.

### 6. Model-chosen tool name doesn't match a dict key — `fill_in_parameters`, `src/__main__.py:382`
`choose_fn` reconstructs a tool name by decoding token ids picked via the
trie (`src/__main__.py:140`). If `llm.decode` doesn't round-trip
byte-for-byte with the `llm.encode(tool)` used to build the trie (tokenizer
whitespace/casing quirks), the returned string can fail to be a key in
`tools_dict` → `tools[tool_name]` → `KeyError`, uncaught.

### 7. Unbounded "thinking" loop in `choose_fn` — `src/__main__.py:121-126`
```python
while idx != END_THINK_TOK:
    logits = llm.get_logits_from_input_ids(conversation)
    idx = _argmax(logits)
    conversation.append(idx)
    thought.append(idx)
```
Unlike `_let_model_think` (capped by `MAX_THINK_TOKENS` /
`MAX_PLAN_THINK_TOKENS`), this loop has no iteration cap. If the model never
emits `END_THINK_TOK`, this either hangs the process indefinitely or grows
`conversation` until the underlying model raises its own error when the
context window is exceeded — and that error isn't caught either. Production
DoS/hang risk, not just a missing-exception gap.

### 8. Context-length overflow anywhere `get_logits_from_input_ids` is called
e.g. `src/__main__.py:116, 123, 132, 241, 269, 294, 348`. None of these calls
are wrapped. A long tool list, a long user prompt, or the unbounded loop in
item 7 can push `conversation`/`convo` past the model's max context, and
whatever error the SDK raises propagates uncaught.

### 9. Empty-token encode edge case — `src/__main__.py:505`
```python
tokenized_tools = [llm.encode(tool).tolist()[0] for tool in tool_names]
```
If a tool name is `""` or otherwise encodes to zero tokens, `.tolist()[0]`
raises `IndexError: list index out of range`.

## Summary table

| # | Location | Trigger example | Exception |
|---|---|---|---|
| 1 | `main():501-502` | `functions_definition.json` = `{"tools": [...]}` | `TypeError` |
| 2 | `Util.get_parameters:61` | tool missing `"parameters"` | `KeyError` |
| 2 | `Util.get_parameters:63` | param entry missing `"type"` | `KeyError` |
| 3 | `main():506-528` | `functions_definition.json` = `[]` | `KeyError` |
| 4 | `main():524` | test file entries missing `"prompt"` key | `KeyError` |
| 5 | `main():499` | model weights missing/corrupt/OOM | unhandled SDK exception |
| 6 | `fill_in_parameters:382` | decoded tool name != any dict key | `KeyError` |
| 7 | `choose_fn:122-126` | model never emits `END_THINK_TOK` | infinite loop / context overflow |
| 9 | `main():505` | tool name encodes to 0 tokens | `IndexError` |

Two structural gaps stand out for production hardening: (a) everything
derived from `tools_list`/`prompts` *after* the JSON parses successfully is
completely unvalidated, and (b) `Small_LLM_Model()` itself and every
`get_logits_from_input_ids` call have zero error handling, so any
runtime/inference failure takes the whole process down with a raw traceback.
