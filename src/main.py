from llm_sdk import Small_LLM_Model
import numpy as np
import json
from typing import Any
import time
import copy


START = "<|im_start|> "
END = " <|im_end|>"
START_THINK = "<think>"
END_THINK = "</think>"
START_TOKEN = 151644
END_TOKEN = 151645
START_THINK_TOK = 151667
END_THINK_TOK = 151668

# Hard safety caps so a per-parameter value can never generate forever, no
# matter how the model behaves.
MAX_VALUE_TOKENS = 32
MAX_STRING_TOKENS = 20
MAX_THINK_TOKENS = 40
MAX_PLAN_THINK_TOKENS = 96


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_ids: list[int]):
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

    def is_complete(self, prefix: list[int]) -> bool:
        node = self.root
        for tid in prefix:
            if tid not in node.children:
                return False
            node = node.children[tid]
        return node.is_end

    def display(self, node=None, prefix="", depth=0):
        if node is None:
            node = self.root
        marker = " [END]" if node.is_end else ""
        if depth > 0:  # skip printing the root itself
            print("  " * (depth - 1) + f"└─ {prefix}{marker}")
        for tid, child in node.children.items():
            self.display(child, prefix=str(tid), depth=depth + 1)


class Util:
    @classmethod
    def get_name(cls, tool: dict[str, str]) -> str:
        return tool["name"]

    @classmethod
    def get_parameters(cls, tool: dict[str, Any]) -> dict:
        out = {}
        parameters_dict = tool["parameters"]
        for k, v in parameters_dict.items():
            if v["type"] == "number":
                # A JSON "number" can legally decode to either an int (e.g.
                # "2") or a float (e.g. "3.0"); accept both.
                typ = (int, float)
            elif v["type"] == "string":
                typ = str
            elif v["type"] == "boolean":
                typ = bool
            else:
                typ = v["type"]
            out[k] = typ
        return out

    @classmethod
    def build_fn_call(cls, prompt: str, tool: str, params: dict) -> str:
        output = {}
        output["prompt"] = prompt
        output["name"] = tool
        output["parameters"] = {k: v["value"] for k, v in params.items()}
        return json.dumps(output)


class Tool:
    name: str
    parameters: dict[str, type | tuple[type, ...]]

    def __init__(self, tool: dict):
        self.name = Util.get_name(tool)
        self.parameters = Util.get_parameters(tool)
        self.template = tool

    def as_string(self) -> str:
        out = f'{{\n\t"name": "{self.name}",\n\t"parameters": {{\n'
        for k, v in self.parameters.items():
            out += f'\t\t"{k}": {str(v)},\n'
        out += "}"
        return out

    def validate_tool_call(self, call: str) -> int:
        """1:  success"""
        """ 0:  malformed json"""
        """-1:  wrong type param"""
        try:
            data = json.loads(call)
        except Exception as e:
            print("failed to json.load in validation call")
            print(e)
            return 0
        try:
            if "parameters" not in data.keys() or not isinstance(
                data["parameters"], dict
            ):
                print("parameters voice missing")
                raise Exception()
            for k, v in data["parameters"].items():
                print("key:", k, "value:", v)
                print(
                    f"checking that {v} ({type(v)}) is of type {self.parameters[k]} ({self.parameters[k]}"
                )
                print()
                if not isinstance(v, self.parameters[k]):
                    print(f"Expecting type {self.parameters[k]} for {k}, got {v}")
                    return 404
        except Exception:
            return -1
        return 1


def choose_fn(
    llm: Small_LLM_Model, prompt: str, tools: dict[str, Tool], trie: Trie, pref: list
) -> str:
    CHOOSE_TOOL_PROMPT = f"""
    system

    You are a helpful assistant, you will the tool to call to complete the user prompt. Pick from the list below.
    tools_list: {[k for k in tools.keys()]}

    The user prompt is: {prompt}
    """
    prefix = pref.copy()
    initial_prompt = START + CHOOSE_TOOL_PROMPT + END + START + "assistant"
    conversation = llm.encode(initial_prompt)[0].tolist()
    logits = llm.get_logits_from_input_ids(conversation)
    idx = _argmax(logits)
    conversation.append(idx)
    answer = []
    thought = []
    # Here we are letting the model think
    while idx != END_THINK_TOK:
        logits = llm.get_logits_from_input_ids(conversation)
        idx = _argmax(logits)
        # print(llm.decode([idx]), end="")
        # sys.stdout.flush()
        conversation.append(idx)
        thought.append(idx)
    conversation.extend(prefix)
    # Here for example this is inefficient because we can just store the len at this time and return the encoded_prompt[len:]
    answer.extend(prefix)
    # print("ALL THAT IS FIXED: ", llm.decode(answer))
    valid_token_ids = trie.get_valid_next_tokens(prefix)
    while len(valid_token_ids):
        valid_token_ids = trie.get_valid_next_tokens(prefix)
        logits = llm.get_logits_from_input_ids(conversation)
        if len(valid_token_ids):
            idx = max(valid_token_ids, key=lambda tid: logits[tid])
            prefix.append(idx)
        else:
            idx = _argmax(logits)
        # print(llm.decode([idx]), end="")
        # sys.stdout.flush()
        conversation.append(idx)
        answer.append(idx)
    return llm.decode(answer)


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

    vocab_size = len(llm.get_logits_from_input_ids(convo))
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
    """Index of the largest logit, optionally restricted to `candidate_ids`.

    Replaces the torch.argmax(...)/torch.softmax(...) calls this project
    used to prototype with: softmax is a strictly monotonic transform, so it
    never changes which index is largest and can simply be dropped rather
    than reimplemented.
    """
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

        logits = llm.get_logits_from_input_ids(convo)
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

    return llm.decode(value_ids).strip()


def _fill_bool_value(llm: Small_LLM_Model, convo: list[int]) -> str:
    """Trie-constrained decode restricted to the literals "true"/"false"."""
    trie = build_bool_trie(llm)
    prefix: list[int] = []
    valid_token_ids = trie.get_valid_next_tokens(prefix)
    while valid_token_ids:
        logits = llm.get_logits_from_input_ids(convo)
        idx = _argmax(logits, valid_token_ids)
        convo.append(idx)
        prefix.append(idx)
        valid_token_ids = trie.get_valid_next_tokens(prefix)
    return llm.decode(prefix).strip()


def _fill_string_value(llm: Small_LLM_Model, convo: list[int]) -> str:
    """Capped, stop-gated decode for free-form string parameters.

    String content isn't grammar-constrained the way numbers/booleans are,
    but generation is still bounded by MAX_STRING_TOKENS. The model very
    often opens a value with a JSON-style quote character (mimicking the
    double-quoted tool template already in context), so a bare opening
    quote is consumed as a delimiter rather than treated as a stop; once
    "inside" a quote, only the matching closing quote (or `\n`/END_TOKEN/the
    cap) ends the value. Without this, a value that legitimately starts with
    a quote character comes back empty instead of its real content.
    """
    value_ids: list[int] = []
    quote_char: str | None = None
    started = False
    for _ in range(MAX_STRING_TOKENS):
        logits = llm.get_logits_from_input_ids(convo)
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

    text = llm.decode(value_ids).strip()
    if quote_char is not None and text.endswith(quote_char):
        text = text[: -len(quote_char)].rstrip()
    return text


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
    convo.extend(llm.encode(f"{START_THINK} {hint}")[0].tolist())
    idx = -1
    for _ in range(max_tokens):
        logits = llm.get_logits_from_input_ids(convo)
        idx = _argmax(logits)
        convo.append(idx)
        if idx == END_THINK_TOK:
            break
    if idx != END_THINK_TOK:
        convo.append(END_THINK_TOK)


def _coerce_value(text: str, type_: str) -> Any:
    """Convert the decoded literal text to the type the schema expects."""
    if type_ == "number":
        try:
            return float(text) if "." in text else int(text)
        except ValueError:
            return text
    if type_ == "boolean":
        return text == "true"
    return text


def fill_in_parameters(
    llm: Small_LLM_Model, prompt: str, tools: dict[str, Tool], tool_name: str
) -> dict:
    SYSTEM = f"""
    system
    You are a helpful assistant, the user chose the "{tool_name}" tool to solve the following prompt:
    {prompt}

    You are now tasked to fill in the parameters of the function call.
    The function takes the following shape:
    {tools[tool_name].template}

    assistant
    """
    initial_prompt = START + SYSTEM + END + START + "assistant"
    fn = copy.deepcopy(tools[tool_name].template)
    params = fn["parameters"]
    convo = llm.encode(initial_prompt)[0].tolist()

    # Reason about every parameter together, once, before any value is
    # committed. This lets later fields build on how earlier ones were
    # derived (e.g. "replacement" can see what "regex" was reasoned to be)
    # instead of each field re-deriving everything from scratch in
    # isolation, which is what caused cross-field confusion before.
    param_list = ", ".join(f"{k} ({obj['type']})" for k, obj in params.items())
    plan_hint = (
        f"Let me work out the values for all parameters of {tool_name} "
        f"together: {param_list}. I will reason about how they relate to "
        f"each other before answering."
    )
    _let_model_think(llm, convo, plan_hint, max_tokens=MAX_PLAN_THINK_TOKENS)

    for k, obj in params.items():
        snapshot_len = len(convo)

        meta = f"{START_THINK} I am filling the parameter {k}, which must be of type {obj['type']} {END_THINK}"
        convo.extend(llm.encode(meta)[0].tolist())

        # Cue the model into "answer mode" right before constrained decoding
        # starts. String values in particular tend to drift into echoing a
        # JSON/dict-shaped draft (e.g. "{'name': 'Shrek'}") from the shared
        # reasoning above instead of the bare value; ruling that out
        # explicitly pulls the clean value out reliably.
        if obj["type"] == "string":
            cue = (
                f"\nBased on the reasoning above, writing only the bare "
                f"{k} value, with no braces, quotes, or key name: "
            )
        else:
            cue = f"\nBased on the reasoning above, {k} value: "
        convo.extend(llm.encode(cue)[0].tolist())

        if obj["type"] == "number":
            text = _fill_number_value(llm, convo)
        elif obj["type"] == "boolean":
            text = _fill_bool_value(llm, convo)
        else:
            text = _fill_string_value(llm, convo)

        value = _coerce_value(text, obj["type"])
        obj["value"] = value

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

    return params


def complete_fn_call(
    llm: Small_LLM_Model, prompt: str, tools: dict[str, Tool], trie: Trie, pref: list
) -> tuple[str, str]:
    # First we need the model to choose the tool name. We will do so using constrained decoding to force
    # a valid tool_name from our list using our trie.
    #
    # After that, we will create the tool.template that will be filled, we will then iterate on the parameters
    # as k, v and the llm will set the desired values to solve the prompt.
    #
    # Lastly, we will create a sub-dictionary with only the k, v expected in the output, and run a json.dumps()
    tool_name = choose_fn(llm, prompt, tools, trie, pref)
    SYSTEM = f"""
    system
    You are a helpful assistant, the user chose the "{tool_name}" tool to solve the following prompt:
    {prompt}

    You are now tasked to fill in the parameters of the function call.

    assistant
    """
    return ("a", "b")


def main() -> None:
    start = time.perf_counter()
    with open("./data/input/functions_definition.json", "r") as f:
        tools_list = json.load(f)
    llm = Small_LLM_Model()
    # print("Len of tools_list: ", len(tools_list))
    tools_dict = {}
    for tool in tools_list:
        tools_dict[tool["name"]] = Tool(tool)

    tool_names = [name for name in tools_dict.keys()]
    tokenized_tools = [llm.encode(tool).tolist()[0] for tool in tool_names]
    trie = Trie()
    for toktool in tokenized_tools:
        trie.insert(toktool)
    prefix = []
    node = trie.root
    while len(node.children) == 1:
        tid, _ = next(iter(node.children.items()))
        prefix.append(tid)
        node = node.children[tid]

    with open("./data/input/function_calling_tests.json", "r") as f:
        prompts = json.load(f)
    prompts_str = [p["prompt"] for p in prompts]
    print(
        f"\n\n-----------------------\nAnswering {len(prompts_str)} prompts\n--------------------------\n\n"
    )
    print("PREFIX:", llm.decode(prefix))
    print()
    answers = []
    for prompt in prompts_str:
        tool = choose_fn(llm, prompt, tools_dict, trie, prefix)
        params = fill_in_parameters(llm, prompt, tools_dict, tool)
        call = Util.build_fn_call(prompt, tool, params)
        answers.append(call)
        print(f"{call}")

    end = time.perf_counter()
    print(f"Elapsed: {end - start:.4f} seconds")
    return


if __name__ == "__main__":
    main()
