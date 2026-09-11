from llm_sdk import Small_LLM_Model
import torch
import sys
import json


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


def get_tool_name(tool: dict) -> str:
    name = '"name": ' + '"' + tool["name"] + '"'
    return name


# def solve_prompt(prompt: str, trie: Trie) -> str: ...
#


def main() -> None:
    with open("./data/input/functions_definition.json", "r") as f:
        tools_list = json.load(f)
    llm = Small_LLM_Model()
    print("Len of tools_list: ", len(tools_list))
    tools = json.dumps(tools_list)
    tool_names = []
    for tool in tools_list:
        tool_names.append(get_tool_name(tool))
    tokenized_tools = [llm.encode(tool).tolist()[0] for tool in tool_names]
    for name, tokens in zip(tool_names, tokenized_tools):
        print("tokenizing...", name, tokens)
        print()

    # Initializing and filling in the Trie
    trie = Trie()
    for toktool in tokenized_tools:
        trie.insert(toktool)
    print("successfully initialized the trie")
    trie.display()
    prefix = []
    node = trie.root
    while len(node.children) == 1:
        print(node.children.items())
        tid, _ = next(iter(node.children.items()))
        prefix.append(tid)
        node = node.children[tid]
    print("prefix:", prefix)
    SYSTEM = f"""system You are a helpful assistant, you will answer user questions using the tools provided.
        tools_list: {tools}

        An example filled in tool call:
        {{"name": "fn_add_numbers", "a": 1, "b": 3}}
        user 
    """
    PROMPT = "Reverse the following string: elloH"
    initial_prompt = START + SYSTEM + PROMPT + END + START + "assistant"
    print(initial_prompt)
    encoded_prompt = llm.encode(initial_prompt)[0].tolist()
    logits = torch.tensor(llm.get_logits_from_input_ids(encoded_prompt))
    probs = torch.softmax(logits, dim=-1)
    idx = probs.argmax(dim=-1).item()
    encoded_prompt.append(idx)
    print(PROMPT)
    print()
    while idx != END_THINK_TOK:
        logits = torch.tensor(llm.get_logits_from_input_ids(encoded_prompt))
        probs = torch.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1).item()
        print(llm.decode([idx]), end="")
        sys.stdout.flush()
        encoded_prompt.append(idx)
    encoded_prompt.append(llm.encode("{").tolist()[0][0])
    encoded_prompt.extend(prefix)
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
        print(llm.decode([idx]), end="")
        sys.stdout.flush()
        encoded_prompt.append(idx)
    while idx != END_TOKEN:
        logits = torch.tensor(llm.get_logits_from_input_ids(encoded_prompt))
        probs = torch.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1).item()
        print(llm.decode([idx]), end="")
        sys.stdout.flush()
        encoded_prompt.append(idx)

    print("\nFULL TEXT:")
    print(llm.decode(encoded_prompt))


if __name__ == "__main__":
    main()
