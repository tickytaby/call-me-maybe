from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from pydantic import TypeAdapter, ValidationError
from typing import cast
import numpy as np
import argparse
import json
import os
import sys
import re
from .models import FunctionCall, ParameterValue, PromptItem, ToolDefinition


START = "<|im_start|> "
END = " <|im_end|>"
START_THINK = "<think>"
END_THINK = "</think>"
START_TOKEN = 151644
END_TOKEN = 151645
START_THINK_TOK = 151667
END_THINK_TOK = 151668

MAX_VALUE_TOKENS = 32
MAX_STRING_TOKENS = 20
MAX_THINK_TOKENS = 80
MAX_PLAN_THINK_TOKENS = 96


class TrieNode:
    def __init__(self) -> None:
        self.children: dict[int, "TrieNode"] = {}
        self.is_end = False


class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, token_ids: list[int]) -> None:
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
        node.is_end = True

    def get_valid_next_tokens(self, prefix: list[int]) -> set[int]:
        node = self.root
        for tid in prefix:
            if tid not in node.children:
                return set()
            node = node.children[tid]
        return set(node.children.keys())


class ModelGenerationError(RuntimeError):
    """A call into the model failed at runtime (e.g. context-length overflow).

    The SDK doesn't document a stable exception type for this, so every call
    site wraps `get_logits_from_input_ids` through `_get_logits` below and
    normalizes whatever it raises into this one catchable type.
    """


def _get_logits(llm: Small_LLM_Model, convo: list[int]) -> list[float]:
    try:
        return cast(list[float], llm.get_logits_from_input_ids(convo))
    except Exception as e:
        raise ModelGenerationError(
            f"model inference failed with a conversation of {len(convo)} "
            f"tokens (possible context-length overflow): {e}"
        ) from e


def build_fn_call(
    prompt: str, tool: str, params: dict[str, ParameterValue]
) -> FunctionCall:
    return FunctionCall(
        prompt=prompt,
        name=tool,
        parameters={k: v.value for k, v in params.items()},
    )


def choose_fn(
    llm: Small_LLM_Model,
    prompt: str,
    tools: dict[str, ToolDefinition],
    trie: Trie,
    pref: list[int],
) -> str:
    CHOOSE_TOOL_PROMPT = f"""
    system

    You are a helpful assistant,
    you will the tool to call to complete the user prompt.
    Pick from the list below.

    tools_list: {[k for k in tools.keys()]}

    The user prompt is: {prompt}
    """
    prefix = pref.copy()
    initial_prompt = START + CHOOSE_TOOL_PROMPT + END + START + "assistant"
    conversation = llm.encode(initial_prompt)[0].tolist()
    logits = _get_logits(llm, conversation)
    idx = _argmax(logits)
    conversation.append(idx)
    answer = []
    thought = []
    # Letting the model think
    for _ in range(MAX_THINK_TOKENS):
        logits = _get_logits(llm, conversation)
        idx = _argmax(logits)
        conversation.append(idx)
        thought.append(idx)
        if idx == END_THINK_TOK:
            break
    if idx != END_THINK_TOK:
        conversation.append(END_THINK_TOK)
        thought.append(END_THINK_TOK)
    conversation.extend(prefix)
    answer.extend(prefix)
    while True:
        valid_token_ids = trie.get_valid_next_tokens(prefix)
        if not valid_token_ids:
            break
        logits = _get_logits(llm, conversation)
        idx = max(valid_token_ids, key=lambda tid: logits[tid])
        prefix.append(idx)
        conversation.append(idx)
        answer.append(idx)
    return str(llm.decode(answer))


# Classification results for numeric constrained decoding depend only on the
# tokenizer, so we classify the vocabulary once per llm instance and reuse it
# for every "number" parameter across every prompt.
_NUMBER_TOKEN_CACHE: dict[int, dict[str, set[int]]] = {}


def _classify_number_token(text: str) -> str | None:
    """Classify a single decoded token for the bare-number grammar.

    Returns "digit", "signed_digit", "dot", "minus", "stop", or None if the
    token can never legally appear while filling a "number" value.
    """
    stripped = text[1:] if text.startswith(" ") else text
    if stripped.isdigit():
        return "digit"
    if stripped == "-":
        return "minus"
    if stripped == ".":
        return "dot"
    if stripped.startswith("-") and stripped[1:].isdigit():
        return "signed_digit"
    if stripped.strip() == "":
        # Whitespace (or a decoded special token, which decodes to "") acts
        # as a natural delimiter marking the end of the value.
        return "stop"
    return None


def get_number_token_sets(
    llm: Small_LLM_Model, convo: list[int]
) -> dict[str, set[int]]:
    """Bucket every vocabulary token id for constrained numeric decoding."""
    cache_key = id(llm)
    if cache_key in _NUMBER_TOKEN_CACHE:
        return _NUMBER_TOKEN_CACHE[cache_key]

    vocab_size = len(_get_logits(llm, convo))
    sets: dict[str, set[int]] = {
        "digit": set(),
        "signed_digit": set(),
        "dot": set(),
        "minus": set(),
        "stop": {END_TOKEN},
    }
    for tid in range(vocab_size):
        kind = _classify_number_token(llm.decode([tid]))
        if kind is not None:
            sets[kind].add(tid)
    _NUMBER_TOKEN_CACHE[cache_key] = sets
    return sets


def build_bool_trie(llm: Small_LLM_Model) -> Trie:
    """Build a Trie over the tokenizations of the JSON boolean literals.

    Mirrors how tool names are constrained in `choose_fn`.
    """
    trie = Trie()
    for literal in ("true", "false"):
        trie.insert(llm.encode(literal).tolist()[0])
    return trie


def _argmax(logits: list[float], candidate_ids: set[int] | None = None) -> int:
    """Index of the largest logit, optionally restricted to `candidate_ids`."""
    arr = np.asarray(logits)
    if candidate_ids is None:
        return int(np.argmax(arr))
    ids = np.fromiter(candidate_ids, dtype=np.int64)
    return int(ids[np.argmax(arr[ids])])


def _fill_number_value(llm: Small_LLM_Model, convo: list[int]) -> str:
    """Grammar-constrained decode of a bare int/float literal.

    State machine: start -> optional '-' -> digits+ -> optional '.' digits+.
    Every step masks logits down to only the tokens legal in the current
    state, so the result is always a syntactically valid number and the loop
    is bounded by MAX_VALUE_TOKENS regardless of model behaviour.
    """
    sets = get_number_token_sets(llm, convo)
    state = "start"
    value_ids: list[int] = []
    for _ in range(MAX_VALUE_TOKENS):
        if state == "start":
            allowed = sets["digit"] | sets["signed_digit"] | sets["minus"]
        elif state == "after_sign":
            allowed = sets["digit"]
        elif state == "int_digits":
            allowed = sets["digit"] | sets["dot"] | sets["stop"]
        elif state == "after_dot":
            allowed = sets["digit"]
        else:  # frac_digits
            allowed = sets["digit"] | sets["stop"]

        if not allowed:
            break

        logits = _get_logits(llm, convo)
        idx = _argmax(logits, allowed)

        if idx in sets["stop"]:
            break

        convo.append(idx)
        value_ids.append(idx)

        if idx in sets["minus"]:
            state = "after_sign"
        elif idx in sets["dot"]:
            state = "after_dot"
        elif idx in sets["signed_digit"]:
            state = "int_digits"
        else:  # idx in sets["digit"]
            in_fraction = state in ("after_dot", "frac_digits")
            state = "frac_digits" if in_fraction else "int_digits"

    return str(llm.decode(value_ids)).strip()


def _fill_bool_value(llm: Small_LLM_Model, convo: list[int]) -> str:
    """Trie-constrained decode restricted to the literals "true"/"false"."""
    trie = build_bool_trie(llm)
    prefix: list[int] = []
    valid_token_ids = trie.get_valid_next_tokens(prefix)
    while valid_token_ids:
        logits = _get_logits(llm, convo)
        idx = _argmax(logits, valid_token_ids)
        convo.append(idx)
        prefix.append(idx)
        valid_token_ids = trie.get_valid_next_tokens(prefix)
    return str(llm.decode(prefix)).strip()


def _fill_string_value(
    llm: Small_LLM_Model,
    convo: list[int],
    stop_words: list[str] | None = None,  # NEW
) -> str:
    """Capped, stop-gated decode for free-form string parameters.

    String content isn't grammar-constrained the way numbers/booleans are,
    but generation is still bounded by MAX_STRING_TOKENS. The model very
    often opens a value with a JSON-style quote character (mimicking the
    double-quoted tool template already in context), so a bare opening
    quote is consumed as a delimiter rather than treated as a stop; once
    "inside" a quote, only the matching closing quote (or `\n`/END_TOKEN/the
    cap) ends the value. Without this, a value that legitimately starts with
    a quote character comes back empty instead of its real content.

    stop_words additionally guards against runaway generation that wanders    # NEW
    into the next field's syntax (e.g. source_string's value trailing off    # NEW
    into "', regex: '0+', replacement:"). If the accumulated decoded text    # NEW
    ever contains one of these substrings, generation stops immediately      # NEW
    and everything from that point onward is discarded.                     # NEW
    """
    value_ids: list[int] = []
    quote_char: str | None = None
    started = False
    stop_words = stop_words or []  # NEW

    for _ in range(MAX_STRING_TOKENS):
        logits = _get_logits(llm, convo)
        idx = _argmax(logits)
        token_text = llm.decode([idx])

        if idx == END_TOKEN or "\n" in token_text:
            break

        stripped = token_text.strip()
        if not started:
            started = True
            if stripped in ('"', "'"):
                quote_char = stripped
                convo.append(idx)
                continue
            if stripped and stripped[0] in ('"', "'") and len(stripped) > 1:  # NEW
                # Opening quote fused with the first content token in a      # NEW
                # single token (e.g. `"agent007` as one piece) instead of    # NEW
                # arriving as its own token; keep the content after the      # NEW
                # quote rather than losing it or treating the whole token    # NEW
                # as a bare delimiter.                                       # NEW
                quote_char = stripped[0]  # NEW
                convo.append(idx)  # NEW
                value_ids.append(idx)  # NEW
                continue  # NEW
        elif quote_char is not None and stripped == quote_char:
            convo.append(idx)
            break
        elif quote_char is not None and stripped.endswith(quote_char):
            # The closing quote got merged into the same token as trailing
            # content (e.g. `old"` as one token) instead of being its own
            # token; keep the content but stop right after it.
            convo.append(idx)
            value_ids.append(idx)
            break

        convo.append(idx)
        value_ids.append(idx)

        if stop_words:  # NEW
            partial = str(llm.decode(value_ids))  # NEW
            hit = next((w for w in stop_words if w in partial), None)  # NEW
            if hit is not None:  # NEW
                cut = partial.index(hit)  # NEW
                partial = partial[:cut]  # NEW
                # Rebuild value_ids isn't possible post-hoc (ids don't map   # NEW
                # 1:1 to characters), so decode-truncate here and return     # NEW
                # the trimmed string directly rather than continuing the     # NEW
                # token loop with stale ids.                                 # NEW
                text = partial.rstrip()  # NEW
                if quote_char is not None and text.endswith(quote_char):  # NEW
                    text = text[: -len(quote_char)].rstrip()  # NEW
                return text  # NEW

    text = str(llm.decode(value_ids)).strip()
    if quote_char is not None and text.endswith(quote_char):
        text = text[: -len(quote_char)].rstrip()
    return text


# def _fill_string_value(llm: Small_LLM_Model, convo: list[int]) -> str:
#     """Capped, stop-gated decode for free-form string parameters.
#
#     String content isn't grammar-constrained the way numbers/booleans are,
#     but generation is still bounded by MAX_STRING_TOKENS. The model very
#     often opens a value with a JSON-style quote character (mimicking the
#     double-quoted tool template already in context), so a bare opening
#     quote is consumed as a delimiter rather than treated as a stop; once
#     "inside" a quote, only the matching closing quote (or `\n`/END_TOKEN/the
#     cap) ends the value. Without this, a value that legitimately starts with
#     a quote character comes back empty instead of its real content.
#     """
#     value_ids: list[int] = []
#     quote_char: str | None = None
#     started = False
#     # IMPORTANT: Missing the handling of the initial "token fused case
#     for _ in range(MAX_STRING_TOKENS):
#         logits = _get_logits(llm, convo)
#         idx = _argmax(logits)
#         token_text = llm.decode([idx])
#
#         if idx == END_TOKEN or "\n" in token_text:
#             break
#
#         stripped = token_text.strip()
#         if not started:
#             started = True
#             if stripped in ('"', "'"):
#                 quote_char = stripped
#                 convo.append(idx)
#                 continue
#         elif quote_char is not None and stripped == quote_char:
#             convo.append(idx)
#             break
#         elif quote_char is not None and stripped.endswith(quote_char):
#             # The closing quote got merged into the same token as trailing
#             # content (e.g. `old"` as one token) instead of being its own
#             # token; keep the content but stop right after it.
#             convo.append(idx)
#             value_ids.append(idx)
#             break
#
#         convo.append(idx)
#         value_ids.append(idx)
#
#     text = str(llm.decode(value_ids)).strip()
#     if quote_char is not None and text.endswith(quote_char):
#         text = text[: -len(quote_char)].rstrip()
#     return text


def _let_model_think(
    llm: Small_LLM_Model,
    convo: list[int],
    hint: str,
    max_tokens: int = MAX_THINK_TOKENS,
) -> None:
    """Let the model generate its own bounded reasoning before it answers.

    Mirrors the "let it think" loop in `choose_fn`, but capped by
    `max_tokens` so it can never run away. Some parameter values must be
    *derived* from the request (e.g. a regex pattern) rather than copied
    verbatim; a fixed one-line comment gives the model no room to work that
    out, so this appends a real `<think>` span it actually generates.
    """
    init_hint = llm.encode(f"{START_THINK} {hint}")[0].tolist()
    convo.extend(init_hint)
    idx = -1
    value_ids = []
    value_ids.extend(init_hint)
    for _ in range(max_tokens):
        logits = _get_logits(llm, convo)
        idx = _argmax(logits)
        convo.append(idx)
        value_ids.append(idx)
        if idx == END_THINK_TOK:
            break
    if idx != END_THINK_TOK:
        convo.append(END_THINK_TOK)
        value_ids.append(END_THINK_TOK)
    print(llm.decode(convo))


def _clean_param_value(value: str) -> str:
    # If it looks like a leaked "label: value" or 'label: "value"' prefix,
    # strip everything up to and including the first colon that appears
    # before any real content starts.
    print(f"\n\nValue before cleaning: {value}")
    m = re.match(r"""^\s*\{\s*['"]?\w+['"]?\s*:\s*['"]?(.*?)['"]?\s*\}\s*$""", value)
    if m:
        value = m.group(1)
    m = re.match(r'^\s*"?[\w\s]{1,20}"?\s*:\s*', value)
    if m:
        value = value[m.end():]

    value = value.rstrip(".")
    value = value.strip("'")
    value = value.strip('"')
    if value and value.strip() == "":
        return value
    print(f"Value after cleaning: {value.strip()}\n\n")

    return value.strip()


def _coerce_value(text: str, type_: str) -> int | float | str | bool:
    """Convert the decoded literal text to the type the schema expects."""
    if type_ == "number":
        return float(text) if "." in text else int(text)
    if type_ == "boolean":
        return text == "true"
    if type_ == "string":
        return _clean_param_value(text)
    return text


def fill_in_parameters(
    llm: Small_LLM_Model, prompt: str, tools: dict[str, ToolDefinition], tool_name: str
) -> dict[str, ParameterValue]:
    tool = tools[tool_name]
    param_schemas = tool.parameters
    # Only the parameter schema is shown here, not tool.model_dump(): dumping
    # the whole ToolDefinition would repeat the tool's own "name" field right
    # next to the parameter schema, and for a tool whose parameter is itself
    # called "name" (e.g. fn_greet), the model latches onto the tool's own
    # name as the "most recent name value" and echoes that back instead of
    # reasoning about the actual parameter.
    params_shape = {k: v.model_dump() for k, v in param_schemas.items()}
    if tool_name == "fn_substitute_string_with_regex":
        SYSTEM = rf"""Fill parameters for fn_substitute_string_with_regex.
        Do NOT perform the substitution — only extract inputs.
        - source_string: the ORIGINAL text, unmodified (the "before").
            Never put the edited/result text here.
        - regex: what to find/match.
            - If the user names a literal word/phrase, use that text
              (escaped if needed: . * + ? ( ) [ ] {{}} | ^ $ \).
            - If the user names a CATEGORY (e.g. numbers, letters, vowels,
              whitespace, digits)
              , convert it to the matching pattern.
                - numbers→\d+, letters→[a-zA-Z], vowels→[aeiouAEIOU], whitespace→\s+.
        - replacement: what to insert in place of each match.
            - If the user gives a literal word to insert, use that word as-is.
            - If the user names a SYMBOL (e.g. asterisk, dash, hyphen, space,
              underscore,
              comma), convert it to the actual character.
                - asterisk→*, dash/hyphen→-, underscore→_.
            - If not stated (e.g. "remove X"), use "".
        Examples:
        "Replace all numbers in 'Hello 34 I'm 233' with NUMBERS"
            → source_string: "Hello 34 I'm 233", regex: "\d+", replacement: "NUMBERS"
            (regex is the numbers PATTERN, not the word "numbers";
            replacement is the literal word given)
        "Replace all vowels in 'Programming is fun' with asterisks"
            → source_string: "Programming is fun", regex: "[aeiouAEIOU]",
            replacement: "*"
            (replacement is the actual asterisk character, not the word "asterisks")
        Never write the finished/result string into source_string.
        Original prompt: {prompt}
        """
    else:
        SYSTEM = f"""system You are a useful agent that is tasked with filling
        parameters for function calls.
        Original prompt: {prompt}
        You are filling the parameters for "{tool_name}"

        The parameters to fill in are: {params_shape}
        Extract the correct input value(s) the user provided for the function call.
        Avoid computing, solving or transforming the value. Copy it as given.
        """
    initial_prompt = START + SYSTEM + END + START + "assistant"
    convo = llm.encode(initial_prompt)[0].tolist()

    # Reason about every parameter together, once, before any value is
    # committed. This lets later fields build on how earlier ones were
    # derived (e.g. "replacement" can see what "regex" was reasoned to be)
    # instead of each field re-deriving everything from scratch in
    # isolation, which is what caused cross-field confusion before.
    param_list = ", ".join(f"{k} ({obj.type})" for k, obj in param_schemas.items())
    plan_hint = (
        f"Let me work out the values for all parameters of {tool_name} "
        f"together: {param_list}."
        f"So the values would be "
    )
    if tool_name == "fn_substitute_string_with_regex":
        _let_model_think(llm, convo, plan_hint, max_tokens=2 * MAX_PLAN_THINK_TOKENS)
    else:
        _let_model_think(llm, convo, plan_hint, max_tokens=MAX_PLAN_THINK_TOKENS)

    values: dict[str, ParameterValue] = {}
    for k, obj in param_schemas.items():
        snapshot_len = len(convo)

        meta = (
            f"{START_THINK} I am filling the parameter {k}, "
            f"which must be of type {obj.type} {END_THINK}"
        )
        convo.extend(llm.encode(meta)[0].tolist())

        # Cue the model into "answer mode" right before constrained decoding
        # starts. String values in particular tend to drift into echoing a
        # JSON/dict-shaped draft (e.g. "{'name': 'Shrek'}") from the shared
        # reasoning above instead of the bare value; ruling that out
        # explicitly pulls the clean value out reliably.
        other_field_names = [name for name in param_schemas if name != k]  # NEW
        if obj.type == "string":
            cue = (
                f"\nBased on the reasoning above, writing only the bare "
                rf"{k} value, with no quotes, {{}} or key name: "
            )
        else:
            cue = f"\nBased on the reasoning above, {k} value: "
        convo.extend(llm.encode(cue)[0].tolist())

        if obj.type == "number":
            text = _fill_number_value(llm, convo)
        elif obj.type == "boolean":
            text = _fill_bool_value(llm, convo)
        else:
            text = _fill_string_value(
                llm, convo, stop_words=other_field_names
            )  # CHANGED: was _fill_string_value(llm, convo)
        # else:
        #     text = _fill_string_value(llm, convo)

        value = _coerce_value(text, obj.type)
        values[k] = ParameterValue(type=obj.type, value=value)

        # Roll back this field's raw decode trace (rejected quote tokens,
        # any commentary that didn't make it into the final value, etc.)
        # and replace it with a narrated note instead of a "key = value"
        # assignment line. An assignment-shaped ledger line taught the model
        # to echo that same "key = value" shape back as the *next* field's
        # answer (e.g. regex becoming "regex = 'cat'"); phrasing it as prose
        # still carries the decided value forward without handing the model
        # a template that looks like something to copy.
        del convo[snapshot_len:]
        note = (
            f"\n(I already decided the value for '{k}': {json.dumps(value)}. "
            f"Moving on to the next parameter.)"
        )
        convo.extend(llm.encode(note)[0].tolist())

    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="call-me-maybe")
    parser.add_argument(
        "--functions_definition",
        default="./data/input/functions_definition.json",
        help="Path to the JSON file describing the available tools.",
    )
    parser.add_argument(
        "--input",
        default="./data/input/function_calling_tests.json",
        help="Path to the JSON file with the list of prompts to answer.",
    )
    parser.add_argument(
        "--output",
        default="./data/output/output.json",
        help="Path to write the resulting JSON list of fn_calls to.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        with open(args.functions_definition, "r") as f:
            tools_list = json.load(f)
    except FileNotFoundError:
        print(
            (f"{args.functions_definition} not found. "
             "Please try again with a valid path.")
        )
        return 1
    except PermissionError:
        print(f"Can't open {args.functions_definition}. -> missing permissions")
        return 2
    except json.JSONDecodeError:
        print(
            f"Failed to load json, make sure {args.functions_definition} is valid json"
        )
        return 3
    except Exception as e:
        print(f"Failed loading {args.functions_definition}")
        print(e)
        return 4

    try:
        tools = TypeAdapter(list[ToolDefinition]).validate_python(tools_list)
        if not tools:
            raise Exception("No tools provided for function calls")
    except ValidationError as e:
        print(f"{args.functions_definition} has an invalid schema:\n{e}")
        return 8
    except Exception as e:
        print(e)
        return 8

    try:
        llm = Small_LLM_Model()
    except Exception as e:
        print("Failed to load Small_LLM_Model")
        print(e)
        return 10

    tools_dict: dict[str, ToolDefinition] = {tool.name: tool for tool in tools}

    trie = Trie()
    for name in tools_dict.keys():
        toktool = llm.encode(name).tolist()[0]
        if not toktool:
            # A name that encodes to zero tokens contributes no path to the
            # trie at all, which would make it silently unselectable by
            # choose_fn. Warn instead of letting it disappear quietly.
            print(
                (
                    f"Warning: tool {name!r} encodes to zero tokens;"
                    "it can never be selected and will be skipped."
                )
            )
            continue
        trie.insert(toktool)
    prefix = []
    node = trie.root
    while len(node.children) == 1:
        tid, _ = next(iter(node.children.items()))
        prefix.append(tid)
        node = node.children[tid]

    try:
        with open(args.input, "r") as f:
            prompts = json.load(f)
    except Exception as e:
        print(f"Trouble opening file {args.input}")
        print(e)
        return 5

    try:
        prompt_items = TypeAdapter(list[PromptItem]).validate_python(prompts)
    except ValidationError as e:
        print(f"{args.input} has an invalid schema:\n{e}")
        return 9

    prompts_str = [p.prompt for p in prompt_items]
    answers: list[FunctionCall] = []
    for prompt in prompts_str:
        try:
            tool = choose_fn(llm, prompt, tools_dict, trie, prefix)
            if tool not in tools_dict:
                print(
                    (
                        f"Warning: model chose unknown tool {tool!r}"
                        f" for prompt {prompt!r}; skipping."
                    )
                )
                continue
            params = fill_in_parameters(llm, prompt, tools_dict, tool)
        except ModelGenerationError as e:
            print(f"Warning: skipping prompt {prompt!r}: {e}")
            continue
        call = build_fn_call(prompt, tool, params)
        answers.append(call)

    output_dir = os.path.dirname(args.output)
    if not output_dir:
        print("Missing output directory path, printing output to screen...")
        print(json.dumps([a.model_dump() for a in answers]))
        return 6
    try:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([a.model_dump() for a in answers], f, indent=2)
    except Exception as e:
        print(f"Failed to write to output path {output_dir}")
        print(e)
        return 7
    return 0


if __name__ == "__main__":
    sys.exit(main())
