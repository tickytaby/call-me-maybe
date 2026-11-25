"""Constrained decoding of a complete function call as JSON.

The model writes ``{"name": "...", "parameters": {...}}`` token by token, but
it only ever *chooses* at the open slots: the function name and each
argument value. Everything structural (braces, keys, quotes, commas) is known
in advance and appended directly, without running the model. At each slot
the logits are restricted to the tokens that keep the text valid JSON for
that slot's type, so the result always parses and always matches the schema.
"""

import json
import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from .models import SUPPORTED_TYPES, ToolDefinition, Trie, TrieNode
from .prompt import build_chat
from .vocab import IdArray, Vocabulary

MAX_STRING_TOKENS = 96
MAX_NUMBER_TOKENS = 48


class ModelGenerationError(RuntimeError):
    """A call into the model failed (e.g. out of memory)."""


class SlotResult(BaseModel):
    """JSON text produced for one slot.

    Attributes:
        text: The JSON literal written into the call (e.g. ``2.5``,
            ``"abc"``, ``true``), including quotes for strings.
        truncated: True when the token budget ran out before the value
            ended naturally.
    """

    text: str
    truncated: bool = False


class DecodedCall(BaseModel):
    """Result of decoding one prompt.

    Attributes:
        name: The chosen function.
        parameters: Parsed, schema-typed argument values.
        json_text: The exact JSON text the model generated.
        passes: Number of forward passes the model ran for this prompt.
        truncated: Parameters whose value hit the token budget.
    """

    name: str
    parameters: dict[str, Any]
    json_text: str
    passes: int
    truncated: list[str]


class ConstrainedDecoder(BaseModel):
    """Drives the model through one JSON function call per prompt.

    Attributes:
        llm: The ``Small_LLM_Model`` instance.
        vocab: Precomputed token tables (see ``vocab.py``).
        tools: Available tools, in file order.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Any
    vocab: Vocabulary
    tools: list[ToolDefinition]

    _passes: int = PrivateAttr(default=0)
    _encode_cache: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _name_trie: Trie = PrivateAttr(default_factory=Trie)
    _name_by_ids: dict[tuple[int, ...], str] = PrivateAttr(default_factory=dict)
    _bool_trie: Trie = PrivateAttr(default_factory=Trie)
    _closing_quote: IdArray = PrivateAttr()
    _first_digit: IdArray = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Build the name/boolean tries and derived token sets."""
        for tool in self.tools:
            ids = self.encode(tool.name)
            self._name_trie.insert(ids)
            self._name_by_ids[tuple(ids)] = tool.name
        for literal in ("true", "false"):
            self._bool_trie.insert(self.encode(" " + literal))
        v = self.vocab
        self._closing_quote = np.array(
            [t for t, q in v.quote_at_normal.items() if q == 0], dtype=np.int64
        )
        # A number may not start with "0" followed by more digits.
        self._first_digit = np.array(
            [t for t in v.digits.tolist()
             if not (v.token_bytes[t][:1] == b"0" and len(v.token_bytes[t]) > 1)],
            dtype=np.int64,
        )

    # ------------------------------------------------------------ primitives

    def encode(self, text: str) -> list[int]:
        """Tokenize ``text`` (cached, since forced pieces repeat a lot).

        Args:
            text: Text to tokenize.

        Returns:
            Its token ids.
        """
        if text not in self._encode_cache:
            ids = self.llm.encode(text)[0].tolist()
            self._encode_cache[text] = [int(i) for i in ids]
        return list(self._encode_cache[text])

    def force(self, convo: list[int], text: str) -> None:
        """Append known text to the conversation without calling the model.

        Args:
            convo: Conversation token ids, modified in place.
            text: Text to append.
        """
        convo.extend(self.encode(text))

    def choose(self, convo: list[int], allowed: IdArray) -> int:
        """Pick the most likely next token among ``allowed``.

        When only one token is allowed the model is not called at all.

        Args:
            convo: Conversation so far.
            allowed: Token ids that keep the output valid.

        Returns:
            The chosen token id.

        Raises:
            ModelGenerationError: If the model call fails.
        """
        if len(allowed) == 1:
            return int(allowed[0])
        try:
            logits = np.asarray(
                self.llm.get_logits_from_input_ids(convo), dtype=np.float32
            )
        except Exception as e:
            raise ModelGenerationError(
                f"model inference failed at {len(convo)} tokens: {e}"
            ) from e
        self._passes += 1
        # Same as setting every other logit to -inf, then taking the argmax.
        return int(allowed[int(np.argmax(logits[allowed]))])

    def walk_trie(
        self, convo: list[int], trie: Trie, end_ids: IdArray
    ) -> list[int]:
        """Decode one of the sequences stored in ``trie``.

        Args:
            convo: Conversation, extended in place with the chosen tokens.
            trie: The allowed sequences.
            end_ids: Tokens that mean "stop here" at a node where a
                sequence ends but longer ones continue (e.g. ``fn_add`` vs
                ``fn_add_numbers``). They are not appended.

        Returns:
            The chosen sequence of token ids.
        """
        node: TrieNode = trie.root
        chosen: list[int] = []
        while node.children:
            options = np.fromiter(node.children, dtype=np.int64)
            if node.is_end:
                options = np.concatenate([options, end_ids])
            tid = self.choose(convo, options)
            if tid not in node.children:
                break
            convo.append(tid)
            chosen.append(tid)
            node = node.children[tid]
        return chosen

    # ----------------------------------------------------------------- slots

    def decode_number(self, convo: list[int], integer: bool) -> SlotResult:
        """Decode a JSON number with a token-level state machine.

        Grammar: ``-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?``.
        ``integer`` removes the fraction and exponent parts. The caller
        forces ``"key":`` without the space: the first token is `` `` or
        `` -``, matching how Qwen tokenizes ``"key": -3`` (token healing).

        Args:
            convo: Conversation, extended in place.
            integer: True for the "integer" type.

        Returns:
            The number's JSON text.
        """
        v = self.vocab
        end = v.number_end
        frac_or_exp = (
            np.array([], dtype=np.int64) if integer
            else np.concatenate([v.dot, v.exponent])
        )
        table: dict[str, IdArray] = {
            "lead": np.concatenate([v.space, v.space_minus]),
            "start": np.concatenate([self._first_digit, v.minus]),
            "sign": self._first_digit,
            "zero": np.concatenate([frac_or_exp, end]),
            "int": np.concatenate([v.digits, frac_or_exp, end]),
            "dot": v.digits,
            "frac": np.concatenate([v.digits, v.exponent, end]),
            "exp": np.concatenate([v.digits, v.minus, v.plus]),
            "exp_sign": v.digits,
            "exp_digits": np.concatenate([v.digits, end]),
        }
        complete = {"zero", "int", "frac", "exp_digits"}
        end_set = set(end.tolist())
        state, text = "lead", ""
        for _ in range(MAX_NUMBER_TOKENS):
            tid = self.choose(convo, table[state])
            raw = v.token_bytes[tid]
            if state in complete and tid in end_set:
                return SlotResult(text=text)
            convo.append(tid)
            text += raw.decode("ascii").strip()
            state = self._next_number_state(state, raw)
        if state not in complete:  # budget ran out mid-number: close it
            text += "0"
        return SlotResult(text=text, truncated=True)

    @staticmethod
    def _next_number_state(state: str, raw: bytes) -> str:
        """Transition of the number state machine after token ``raw``."""
        if state == "lead":
            return "sign" if raw == b" -" else "start"
        if raw == b"-":
            return "sign" if state == "start" else "exp_sign"
        if raw == b"+":
            return "exp_sign"
        if raw == b".":
            return "dot"
        if raw in (b"e", b"E"):
            return "exp"
        if state in ("start", "sign"):
            return "zero" if raw == b"0" else "int"
        if state in ("dot", "frac"):
            return "frac"
        if state in ("exp", "exp_sign", "exp_digits"):
            return "exp_digits"
        return "int"

    def decode_bool(self, convo: list[int]) -> SlotResult:
        """Decode `` true`` or `` false`` (single tokens in Qwen).

        Args:
            convo: Conversation, extended in place.

        Returns:
            The literal's JSON text.
        """
        ids = self.walk_trie(convo, self._bool_trie, np.array([], np.int64))
        is_true = ids == self.encode(" true")
        return SlotResult(text="true" if is_true else "false")

    def decode_string(self, convo: list[int]) -> SlotResult:
        """Decode a JSON string, starting with its opening quote.

        The first token must start with `` "`` (space + quote, see
        ``Vocabulary.str_open``). Then the allowed tokens are content
        without control characters, valid escape sequences, or a token
        containing the closing quote. The closing quote ends the value;
        anything after it in the same token is dropped, because the caller
        forces the JSON punctuation that follows.

        Args:
            convo: Conversation, extended in place (up to, not including,
                the closing quote, which the caller forces).

        Returns:
            The string's JSON text, quotes included.
        """
        v = self.vocab
        content = bytearray()
        state = "open"
        tables = {
            "open": (v.str_open, v.quote_at_open, v.pending_open, 2),
            "normal": (v.str_normal, v.quote_at_normal, v.pending_normal, 0),
            "escaped": (
                v.str_escaped, v.quote_at_escaped, v.pending_escaped, 0
            ),
        }
        for _ in range(MAX_STRING_TOKENS):
            allowed, quote_at, pending, skip = tables[state]
            tid = self.choose(convo, allowed)
            raw = v.token_bytes[tid]
            q = quote_at.get(tid, -1)
            if q >= 0:
                content += raw[skip:q]
                self.force(convo, raw[:q].decode("utf-8", "replace"))
                return SlotResult(text=self._string_json(bytes(content)))
            content += raw[skip:]
            convo.append(tid)
            state = "escaped" if tid in pending else "normal"
        if state == "escaped":  # drop a dangling backslash: keep JSON valid
            content = content[:-1]
        return SlotResult(
            text=self._string_json(bytes(content)), truncated=True
        )

    @staticmethod
    def _string_json(content: bytes) -> str:
        """Wrap raw string content in quotes (content is already escaped)."""
        return '"' + content.decode("utf-8", "replace") + '"'

    # ------------------------------------------------------------- full call

    def decode_call(self, prompt: str) -> DecodedCall:
        """Generate the function call for one prompt.

        Args:
            prompt: The user's natural-language request.

        Returns:
            The decoded call, parsed from the JSON the model wrote.

        Raises:
            ModelGenerationError: If the model call fails.
            ValueError: If a number decodes to a non-finite value.
        """
        self._passes = 0
        convo = self.encode(build_chat(prompt, self.tools))
        pieces: list[str] = []

        def emit(text: str) -> None:
            pieces.append(text)
            self.force(convo, text)

        # Forced text is accumulated in ``pending`` and tokenized in one go
        # right before the next slot, so it gets its natural tokenization.
        emit('{"name": "')
        name_ids = self.walk_trie(convo, self._name_trie, self._closing_quote)
        name = self._name_by_ids[tuple(name_ids)]
        pieces.append(name)
        tool = next(t for t in self.tools if t.name == name)
        pending = '", "parameters": {'

        truncated: list[str] = []
        for key, schema in tool.parameters.items():
            kind = schema.type if schema.type in SUPPORTED_TYPES else "string"
            # Stop before the space: the slot's first token includes it.
            emit(pending + json.dumps(key) + ":")
            if kind in ("number", "integer"):
                slot = self.decode_number(convo, integer=kind == "integer")
            elif kind == "boolean":
                slot = self.decode_bool(convo)
            else:
                slot = self.decode_string(convo)
            pieces.append(" " + slot.text)
            if slot.truncated:
                truncated.append(key)
            pending = ", "
        pieces.append("}}")

        json_text = "".join(pieces)
        call = json.loads(json_text)  # cannot fail: every token was checked
        params: dict[str, Any] = call["parameters"]
        for key, schema in tool.parameters.items():
            if schema.type == "number":
                params[key] = float(params[key])
                if not math.isfinite(params[key]):
                    raise ValueError(f"parameter {key!r} is not finite")
            elif schema.type == "integer":
                params[key] = int(params[key])
        return DecodedCall(
            name=name,
            parameters=params,
            json_text=json_text,
            passes=self._passes,
            truncated=truncated,
        )
