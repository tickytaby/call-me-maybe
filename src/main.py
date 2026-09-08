from llm_sdk import Small_LLM_Model
import torch
import sys
import json


START = "<|im_start|> "
END = " <|im_end|>"
START_TOKEN = 151644
END_TOKEN = 151645
START_THINK = 151667
END_THINK = 151668


def main() -> None:
    with open("./data/input/functions_definition.json", "r") as f:
        tools_list = json.load(f)
    SYSTEM = f"""system You are a helpful assistant, you will answer user questions using the tools provided below.
        tools_list: {json.dumps(tools_list)}
        You will only provide the function call in valid json.
        user 
    """
    PROMPT = "Reverse the following string: elloH"
    llm = Small_LLM_Model()
    encoded_prompt = llm.encode(START + SYSTEM + PROMPT + END + START + "assistant")[0].tolist()
    logits = torch.tensor(llm.get_logits_from_input_ids(encoded_prompt))
    probs = torch.softmax(logits, dim=-1)
    idx = probs.argmax(dim=-1).item()
    encoded_prompt.append(idx)
    print(PROMPT)
    print()
    while idx != END_TOKEN:
        logits = torch.tensor(llm.get_logits_from_input_ids(encoded_prompt))
        probs = torch.softmax(logits, dim=-1)
        if probs[END_THINK] > 0.8:
            idx = END_THINK
        else:
            idx = probs.argmax(dim=-1).item()
        print(llm.decode([idx]), end="")
        sys.stdout.flush()
        encoded_prompt.append(idx)


if __name__ == "__main__":
    main()
