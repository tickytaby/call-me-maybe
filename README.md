*This project has been created as part of the 42 curriculum by &lt;login&gt;.*

# call me maybe

## Description

`call me maybe` turns natural-language requests into structured function calls
using a small language model (Qwen/Qwen3-0.6B). Given the request
*"What is the sum of 2 and 3?"* and a list of available functions, it does not
answer `5`. Instead it produces the call that would compute it:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

Small models are unreliable at producing structured output when only asked to.
This project uses **constrained decoding**: at every generation step, the
tokens that would break the JSON structure or the function's schema are
removed before the next token is chosen. The output is therefore always valid
JSON that matches the schema, by construction rather than by luck.

## Instructions

Requirements: Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/). The model
(about 1.2 GB) is downloaded from the Hugging Face Hub on first run.

```bash
make install        # uv sync: creates .venv with numpy, pydantic, llm_sdk
make run            # processes data/input/ -> data/output/function_calling_results.json
make lint           # flake8 . and mypy . with the subject's flags
make test           # unit and CLI tests (pytest)
make debug          # same as run, under pdb
make clean          # removes caches
```

## Example usage

```bash
# defaults: data/input/functions_definition.json + data/input/function_calling_tests.json
uv run python -m src

# custom files
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json

# extra arguments through make
make run ARGS="--input data/input/regex.json --output data/output/regex.json"
```

For every prompt the program prints the JSON the model generated and how many
forward passes it needed:

```
[1/22] 'What is the sum of 2 and 3?'
    -> {"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}  (5 passes, 0.2s)
```

Exit codes: `0` success, `1` invalid or missing input file, `2` model could
not be loaded, `3` output could not be written, `4` some prompts failed (the
others are still written), `5` unexpected error. Error messages go to stderr,
and no Python traceback is ever shown.

## Algorithm explanation

### The generation loop

A language model generates text one token at a time. For each step, the
model receives every token so far (`get_logits_from_input_ids`) and returns one
score (logit) per vocabulary token. Normally the highest-scoring token is
appended and the step repeats.

Constrained decoding inserts one step: before picking, every token that
would make the output invalid is excluded, which is the same as setting its
logit to −∞. The code restricts the `argmax` to the allowed token ids
(`ConstrainedDecoder.choose`), which is mathematically identical for greedy
decoding.

### One JSON object per prompt, with forced structure

For each prompt, the model writes a complete JSON object:

```
{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}
└──forced──┘└── trie ─────┘└──── forced ────┘└num┘└f┘└num┘└f┘
```

The model only *chooses* at the open slots: the function name and each
argument value. Everything else (braces, keys, quotes, commas) is known in
advance once the function is chosen, so it is appended directly
(`force`) **without running the model**. Also, when the grammar allows exactly
one token, it is taken without a model call. Only real choices cost a forward
pass.

### The slots

| Slot | Allowed tokens |
|---|---|
| Function name | A **trie** of the tokenized function names: only tokens that continue some name. Where one name is a prefix of another (`fn_add` / `fn_add_numbers`), a closing-quote token is also allowed, so both remain choosable. |
| `number` / `integer` | A state machine over the JSON number grammar `-? (0 \| [1-9][0-9]*) (.[0-9]+)? ([eE][+-]?[0-9]+)?` (no fraction or exponent for `integer`). Only ASCII digits count: `isdigit()` would also accept `²`. The number ends when the model picks a token such as `,`, `}` or whitespace, which is not copied: the real punctuation is forced instead. |
| `boolean` | A trie over `true` / `false`. |
| `string` | First token: one starting with ` "`. Then: any token without control characters, with only valid JSON escapes (`\"`, `\\`, `\n`...), or one containing the closing quote, which ends the string. |

Types not in this list are decoded as strings, with a warning at startup.

### Token tables (`vocab.py`)

The masks are computed once at startup from the model's `vocab.json`
(`get_path_to_vocab_file`). Qwen uses byte-level BPE: each token is a byte
string, written in `vocab.json` with GPT-2's byte-to-character table (`Ġ` is
a space, `Ċ` a newline). The table is inverted to get every token's raw
bytes. Each token is then classified: digits, number terminators, and, for
strings, whether it is valid content in each lexer state (normal / after a
backslash / at the opening quote), and where its closing quote is. Escape state
can span tokens (`\` then `d`), so the string decoder tracks it from token to
token.

### Why the opening quote is chosen by the model

If `"key": "` were forced, the context would end with a lone ` "` token. Qwen
normally merges the quote with the following characters (` "*`, ` "hello`), so
after a lone ` "` the model is somewhere it rarely is in training. The first
version of this rewrite produced `" *"` instead of `"*"` because of this. So
for strings the forced text stops at `"key":`, and the model picks a token
that starts with ` "` itself. This is known as *token healing*. For the same
reason, all forced text between two slots is tokenized in one piece.

### Result

The generated text is parsed with `json.loads`, which cannot fail because every
token was checked, and numbers are converted to `float` (`int` for
`integer`). The output file is written with `json.dump`.

## Design decisions

- **The model writes JSON, rather than separate answers per field.** An earlier
  version first let the model "think" freely, then asked for each value
  separately with prose cues. Values often leaked surrounding syntax
  (`{'s': 'abc`) that had to be cleaned with regexes. Writing the call as JSON
  matches how the model was trained for tool calls, and the string grammar
  makes post-processing unnecessary.
- **No free "thinking".** Each generated token needs a full forward pass, and
  the SDK has no KV cache, so every pass re-reads the whole conversation.
  Thinking accounted for about 70% of all passes in the earlier version
  (80 tokens per prompt, just to pick among 5 names).
- **One generic prompt.** The system prompt lists every function as
  `description -> name(param: type)` and gives two examples with made-up
  functions that are not in the test files. Nothing is specific to a
  particular function name. The description comes *before* the name: with the
  name first, the model picked by surface similarity ("Remove..." went to
  `fn_reverse_string`). A shorter prompt (287 instead of 356 tokens, about 20%
  faster on CPU) was tried and dropped, because it lost accuracy.
- **Greedy decoding** (argmax), so results are deterministic.
- **Pydantic** models for every class: input schemas, output, the trie, the
  vocabulary tables and the decoder itself.
- **Robust inputs:** both files are fully validated *before* the model is
  loaded. A malformed prompt entry is skipped with a warning instead of
  aborting the run. Duplicate function names are reported.
- **Bonus direction:** the masks are built from `vocab.json`, not from
  `decode()`. `encode` is still used for forced text and function names.

## Performance analysis

Measured with `tests/score.py`. "Before" is the previous version of this project
(free reasoning + per-field prompts); "after" is the current one. GPU: RTX 5070 Laptop
(fp16); CPU: 16-core laptop, GPU disabled (fp32).

| Prompt set | Before | After |
|---|---|---|
| Provided tests (22 prompts) | 21/22 | **22/22** |
| Edge cases, custom functions (8) | 2/8 | **8/8** |
| Hard regex set (14) | 8/14 | 6/14 |

| Timing (22 provided prompts) | Before | After |
|---|---|---|
| Forward passes per prompt | ~110–140 | **4–27 (183 in total)** |
| GPU | 92 s | **7 s** |
| CPU only | ~50 min (extrapolated: 7 min for 3 prompts) | **4 min 18 s** (model loading included) |

The regex set is the one place where accuracy dropped. All of its remaining
errors are the model deriving the wrong pattern, never invalid output. For
example, it uses `[aeiou]+` (which merges "ui" into one match) or a vowel class
for "consonants". The earlier version reasoned for up to 192 tokens before
these values. That helped slightly here but made every prompt about 15× slower.

Accuracy is measured with `tests/score.py` against hand-written expected
calls (`tests/fixtures/expected_*.json`). Regex substitutions are scored by
**executing** them, since many regexes are equally correct (`\d+`, `[0-9]+`,
`34|233` all give the same result on the example).

- **Reliability:** every output in every run parsed as valid JSON and matched
  the schema. This is guaranteed by construction.
- **Speed:** the cost is about one forward pass per generated token of a value.
  A number prompt needs about 5–10 passes, a short string about 4, and a
  three-string regex call about 20.
- **Known limits:** string values are capped at 96 tokens and numbers at 48
  (a warning is printed if a value is cut). `\uXXXX` escapes are not generated.
  When a request gives no value (*"Say hi"*), the model invents one, because
  the schema requires an argument.

## Challenges faced

- **Speed.** The first version spent about 110–140 forward passes per prompt,
  mostly on free-form reasoning, and took about 7 minutes for 3 prompts on CPU.
  Removing reasoning, forcing all known structure, and skipping single-choice
  steps brought it down to about 4–20 passes per prompt.
- **Token boundaries.** Model tokens do not line up with JSON syntax: one
  token can contain the end of a string and the comma after it (`",`), or
  start with a space and a quote (` "*`). The string grammar works on the bytes
  of each token and accepts a closing quote anywhere inside it.
- **Prefix collisions in function names.** A trie that only stops at leaves
  makes `fn_add` unreachable when `fn_add_numbers` exists. Fixed by also
  allowing a closing quote at nodes where a name ends.
- **Unicode digits.** `"²".isdigit()` is `True`, but `int("²")` raises. The
  digit class uses ASCII only.

## Testing strategy

- `tests/vocab_test.py`: unit tests for the token tables (ASCII digits, string
  lexer states, escapes, opening and closing quotes), with no model needed.
- `tests/cli_args_test.py`: missing files, unreadable files, invalid JSON,
  invalid schemas, wrong top-level types. Each must exit cleanly with a
  message and no traceback.
- `tests/score.py` with three prompt sets: the provided tests, a harder regex
  set (`data/input/regex.json`), and an edge-case set with a custom function
  file (`tests/fixtures/edge_*.json`). The edge-case set covers a name that
  prefixes another, two string parameters, booleans, negative and exponent
  numbers, very large numbers, long strings, and strings ending in `...` or
  containing an apostrophe.
- Timing runs with the GPU disabled (`CUDA_VISIBLE_DEVICES=""`) to check the
  CPU case.

```bash
make score     # runs the program, then scores it against the expected calls
```

## Resources

- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B): chat template, and
  the empty `<think></think>` block that disables reasoning.
- [ECMA-404, the JSON grammar](https://www.json.org/json-en.html): number and string syntax used by the state machines.
- Willard & Louf, *Efficient Guided Generation for Large Language Models*
  (2023): constrained decoding with finite-state machines over a vocabulary.
- [Guidance: token healing](https://github.com/guidance-ai/guidance): why forcing a partial token hurts the model.
- [Hugging Face tokenizer docs: byte-level BPE](https://huggingface.co/learn/nlp-course/chapter6/5).
- [Pydantic documentation](https://docs.pydantic.dev/).

### How AI was used

Claude (Anthropic), through Claude Code, was used to:

- audit the project against the subject, and list missing requirements and
  failure cases;
- design and implement the constrained-JSON rewrite (`src/vocab.py`,
  `src/decoder.py`, `src/prompt.py`, `src/__main__.py`, `src/models.py`);
- write the tests, the scoring script and the expected-answer fixtures;
- draft this README.

Every change was measured (accuracy with `tests/score.py`, forward passes,
CPU and GPU time) before it was kept, and a prompt variant that scored worse
was discarded.
