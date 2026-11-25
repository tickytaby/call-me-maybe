"""Pydantic models for the input files, the output file and the name trie."""

from typing import Any

from pydantic import BaseModel, Field

SUPPORTED_TYPES = frozenset({"number", "integer", "string", "boolean"})


class ParameterSchema(BaseModel):
    """Schema of one function parameter, e.g. ``{"type": "number"}``.

    Attributes:
        type: JSON type name. Types outside ``SUPPORTED_TYPES`` are accepted
            and decoded as strings, with a warning printed at load time.
    """

    type: str = Field(min_length=1)


class ToolDefinition(BaseModel):
    """One entry of ``functions_definition.json``.

    Attributes:
        name: Function name the model must choose.
        description: Human-readable purpose, shown to the model.
        parameters: Parameter name -> schema, in declaration order.
        returns: Return schema (not used for decoding).
    """

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, ParameterSchema] = Field(default_factory=dict)
    returns: dict[str, Any] | None = None


class PromptItem(BaseModel):
    """One entry of the prompts file: ``{"prompt": "..."}``."""

    prompt: str


class FunctionCall(BaseModel):
    """One object of the output file.

    Attributes:
        prompt: The original natural-language request.
        name: The chosen function.
        parameters: Argument values, typed according to the schema.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]


class TrieNode(BaseModel):
    """A node of a token-id trie.

    Attributes:
        children: Next token id -> child node.
        is_end: True when a complete sequence ends at this node.
    """

    children: dict[int, "TrieNode"] = Field(default_factory=dict)
    is_end: bool = False


class Trie(BaseModel):
    """A set of token-id sequences, walked one token at a time."""

    root: TrieNode = Field(default_factory=TrieNode)

    def insert(self, token_ids: list[int]) -> None:
        """Add one token sequence to the trie.

        Args:
            token_ids: The sequence to add.
        """
        node = self.root
        for tid in token_ids:
            node = node.children.setdefault(tid, TrieNode())
        node.is_end = True
