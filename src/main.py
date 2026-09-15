from llm_sdk import Small_LLM_Model
import torch
import sys
import json
from typing import Any
import time


START = "<|im_start|> "
END = " <|im_end|>"
START_THINK = "<think>"
END_THINK = "</think>"
START_TOKEN = 151644
END_TOKEN = 151645
START_THINK_TOK = 151667
END_THINK_TOK = 151668


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


class JSONUtil:
    @classmethod
    def get_name(cls, tool: dict[str, str]) -> str:
        return tool["name"]

    @classmethod
    def get_parameters(cls, tool: dict[str, Any]) -> dict:
        out = {}
        parameters_dict = tool["parameters"]
        for k, v in parameters_dict.items():
            if v["type"] == "number":
                typ = int
            elif v["type"] == "string":
                typ = str
            else:
                typ = v["type"]
            out[k] = typ
        return out


class Tool:
    name: str
    parameters: dict[str, type]

    def __init__(self, tool: dict[str, str]):
        self.name = JSONUtil.get_name(tool)
        self.parameters = JSONUtil.get_parameters(tool)

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


def get_tool_name(tool: dict) -> str:
    name = '"name": ' + '"' + tool["name"] + '",'
    return name


def solve_prompt(
    llm: Small_LLM_Model, prompt: str, tools: list, trie: Trie, pref: list
) -> tuple[str, str]:
    SYSTEM = f"""system You are a helpful assistant, you will answer user questions using the tools provided.
        tools_list: {tools}

        Examples of correctly filled in tool calls:
        {{
            "prompt": "What is the sum of 2 and 3?",
            "name": "fn_add_numbers",
            "parameters": {{"a": 1, "b": 3}}
        }}
        {{
            "prompt": "Reverse the string 'hello'",
            "name": "fn_reverse_string",
            "parameters": {{"s": "hello"}}
        }}
        {{
            "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
            "name": "fn_substitute_string_with_regex",
            "paramaters": {{"source_string": "Hello 34 I'm 233 years old", "regex": "\\\\d+", "replacement": "NUMBERS"}}
        }}

        IMPORTANT REMINDER:
        When a string value must contain a literal backslash (e.g. in a regex pattern), escape it as \\\\ so the JSON remains valid. 

        user 
    """
    initial_prompt = START + SYSTEM + prompt + END + START + "assistant"
    prefix = pref.copy()
    encoded_prompt = llm.encode(initial_prompt)[0].tolist()
    logits = torch.tensor(llm.get_logits_from_input_ids(encoded_prompt))
    probs = torch.softmax(logits, dim=-1)
    idx = probs.argmax(dim=-1).item()
    encoded_prompt.append(idx)
    answer = []
    thought = []
    # Here we are letting the model think
    while idx != END_THINK_TOK:
        logits = torch.tensor(llm.get_logits_from_input_ids(encoded_prompt))
        probs = torch.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1).item()
        # print(llm.decode([idx]), end="")
        # sys.stdout.flush()
        encoded_prompt.append(idx)
        thought.append(idx)
    op_brace_tok = llm.encode("{").tolist()[0][0]
    encoded_prompt.append(op_brace_tok)
    encoded_prompt.extend(prefix)
    # Here for example this is inefficient because we can just store the len at this time and return the encoded_prompt[len:]
    prompt_key = llm.encode(f'{{"prompt": {json.dumps(prompt)}, ').tolist()[0]
    answer.extend(prompt_key)
    answer.extend(prefix)
    # print("ALL THAT IS FIXED: ", llm.decode(answer))
    valid_token_ids = trie.get_valid_next_tokens(prefix)
    while len(valid_token_ids):
        valid_token_ids = trie.get_valid_next_tokens(prefix)
        logits = torch.tensor(llm.get_logits_from_input_ids(encoded_prompt))
        probs = torch.softmax(logits, dim=-1)
        if len(valid_token_ids):
            idx = max(valid_token_ids, key=lambda tid: probs[tid])
            prefix.append(idx)
        else:
            idx = probs.argmax(dim=-1).item()
        # print(llm.decode([idx]), end="")
        # sys.stdout.flush()
        encoded_prompt.append(idx)
        answer.append(idx)
    print("BEFORE APPENDING PARAMETERS:", llm.decode(answer))
    param_prefix = llm.encode('parameters": {').tolist()[0]
    encoded_prompt.extend(param_prefix)
    answer.extend(param_prefix)
    while idx != END_TOKEN:
        logits = torch.tensor(llm.get_logits_from_input_ids(encoded_prompt))
        probs = torch.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1).item()
        # print(llm.decode([idx]), end="")
        # sys.stdout.flush()
        encoded_prompt.append(idx)
        answer.append(idx)

    # print("\nFULL TEXT:")
    # print(llm.decode(encoded_prompt))
    ret_str = llm.decode(answer)
    thought_str = llm.decode(thought)
    return (thought_str, ret_str)


def main() -> None:
    start = time.perf_counter()
    with open("./data/input/functions_definition.json", "r") as f:
        tools_list = json.load(f)
    llm = Small_LLM_Model()
    # print("Len of tools_list: ", len(tools_list))
    tools_dict = {}
    for tool in tools_list:
        tools_dict[tool["name"]] = Tool(tool)

    # for _, v in tools_dict.items():
    #     print(v.as_string())
    tool_names = []
    for tool in tools_list:
        tool_names.append(get_tool_name(tool))
    tokenized_tools = [llm.encode(tool).tolist()[0] for tool in tool_names]
    # for name, tokens in zip(tool_names, tokenized_tools):
    #     print("tokenizing...", name, tokens)
    #     print()

    # Initializing and filling in the Trie
    trie = Trie()
    for toktool in tokenized_tools:
        trie.insert(toktool)
    prefix = []
    node = trie.root
    while len(node.children) == 1:
        # print(node.children.items())
        tid, _ = next(iter(node.children.items()))
        prefix.append(tid)
        node = node.children[tid]

    with open("./data/input/function_calling_tests.json", "r") as f:
        prompts = json.load(f)
    prompts_str = [p["prompt"] for p in prompts]
    print(
        f"\n\n-----------------------\nAnswering {len(prompts_str)} prompts\n--------------------------\n\n"
    )
    answers = []
    for prompt in prompts_str:
        answers.append(solve_prompt(llm, prompt, tools_list, trie, prefix))

    print(f"Found {len(answers)} answers\n\n")
    try:
        for q, a in zip(prompts_str, answers):
            print(q)
            # print("thought:", a[0])
            print("fn_call:", a[1])
            try:
                call = json.loads(a[1])
                print(
                    "calling tool validation for tool:", tools_dict[call["name"]].name
                )
                print(tools_dict[call["name"]].validate_tool_call(a[1]))
            except Exception:
                print(f"Failed to load string into json object for: \n{a[1]}")
            print()
    except Exception as e:
        print(e)
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.4f} seconds")
    return


if __name__ == "__main__":
    main()
