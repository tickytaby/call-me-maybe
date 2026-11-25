"""Builds the chat prompt shown to the model before constrained decoding."""

from .models import ToolDefinition

SYSTEM_PROMPT = """You are a function-calling assistant. Pick the one \
function that fulfils the user's request and extract its arguments.
Answer with a single JSON object: {{"name": ..., "parameters": {{...}}}}.
Rules:
- Copy argument values exactly as written in the request. Do not compute \
or transform them yourself: the function does that.
- Strings are JSON strings: escape backslashes (a regex digit class is \
"\\\\d").
- When an argument describes a category or pattern instead of a literal \
value, write the value the function needs (e.g. a regular expression).

Available functions:
{tools}

Example (with other functions):
Request: Count the words in 'to be or not to be'
{{"name": "fn_count_words", "parameters": {{"text": "to be or not to be"}}}}
Request: Find every email address in "mail bob@x.org or amy@y.com"
{{"name": "fn_find_all", "parameters": {{"text": "mail bob@x.org or \
amy@y.com", "pattern": "[\\\\w.]+@[\\\\w.]+"}}}}"""


def describe_tool(tool: ToolDefinition) -> str:
    """Render one tool as a compact signature line.

    Args:
        tool: The tool to describe.

    The description comes first: the model then reads what the function
    does before seeing its name, which stops it from choosing by surface
    similarity (e.g. "Remove ..." picking ``fn_reverse_string``).

    Returns:
        A line like ``- Add two numbers. -> fn_add(a: number, b: number)``
    """
    params = ", ".join(f"{k}: {v.type}" for k, v in tool.parameters.items())
    signature = f"{tool.name}({params})"
    if not tool.description:
        return f"- {signature}"
    return f"- {tool.description} -> {signature}"


def sanitize(text: str) -> str:
    """Stop user text from injecting chat-template special tokens.

    Args:
        text: Raw user prompt.

    Returns:
        The prompt with ``<|`` and ``|>`` broken apart.
    """
    return text.replace("<|", "< |").replace("|>", "| >")


def build_chat(prompt: str, tools: list[ToolDefinition]) -> str:
    """Build the full Qwen chat-template text, thinking disabled.

    Args:
        prompt: The user's natural-language request.
        tools: All available tools.

    Returns:
        Text ending right where the assistant's JSON answer starts.
    """
    system = SYSTEM_PROMPT.format(
        tools="\n".join(describe_tool(t) for t in tools)
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{sanitize(prompt)}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
