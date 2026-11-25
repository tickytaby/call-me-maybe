"""Token vocabulary tables used to build constrained-decoding masks.

The model's ``vocab.json`` maps byte-level BPE token strings (GPT-2 style,
where e.g. ``Ġ`` stands for a space) to token ids. This module turns it into
the raw bytes of every token, then precomputes, once per run, which tokens
are legal in each grammar state (number digits, JSON string content, ...).
"""

import json
import re

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

IdArray = NDArray[np.int64]

# Characters that may follow a backslash inside a JSON string. ``u`` is left
# out on purpose: \uXXXX escapes would need a 4-hex-digit sub-grammar and are
# never needed for this project's prompts.
_JSON_ESCAPES = frozenset(b'"\\/bfnrt')
_DIGITS_RE = re.compile(rb"[0-9]+")


def bytes_to_unicode() -> dict[int, str]:
    """Return GPT-2's byte -> printable-character table used by byte BPE.

    Returns:
        A mapping from every byte value (0-255) to the unicode character that
        represents it inside ``vocab.json``.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = printable[:]
    extra = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(printable, chars)}


def scan_string_token(raw: bytes, escape_pending: bool) -> tuple[bool, int, bool]:
    """Simulate a JSON-string lexer over one token.

    Args:
        raw: The token's bytes.
        escape_pending: True when the previous token ended right after a
            backslash, so the first byte here must complete an escape.

    Returns:
        ``(valid, quote_at, escape_pending_after)``. ``valid`` is False when
        the token would make the string invalid JSON. ``quote_at`` is the
        index of an unescaped closing quote (-1 if none): everything before
        it is string content and the string ends there. The last value says
        whether the token ends in the middle of an escape sequence.
    """
    pending = escape_pending
    for i, byte in enumerate(raw):
        if pending:
            if byte not in _JSON_ESCAPES:
                return False, -1, False
            pending = False
        elif byte == 0x5C:  # backslash
            pending = True
        elif byte == 0x22:  # closing quote
            return True, i, False
        elif byte < 0x20:  # control characters are illegal in JSON strings
            return False, -1, False
    return True, -1, pending


class Vocabulary(BaseModel):
    """Byte strings of every token plus the precomputed masks.

    Attributes:
        token_bytes: ``token_bytes[i]`` is the raw byte string of token ``i``
            (empty for ids that are special/added tokens).
        digits: Tokens made only of ASCII digits.
        minus, plus, dot, exponent: Single-character number tokens.
        space, space_minus: The tokens `` `` and `` -``, which open a number
            right after ``"key":`` (Qwen merges the space with a minus sign,
            but never with a digit).
        number_end: Tokens that can end a number (start with ``,``, ``}``,
            or whitespace). They are never written into the output: the
            decoder forces the real JSON punctuation itself.
        str_normal: Tokens legal inside a JSON string in the normal state
            (content, escapes, or a closing quote).
        str_escaped: Same, when the previous token ended with a backslash.
        quote_at_normal, quote_at_escaped: For string tokens that contain
            an unescaped closing quote, the index of that quote, for each
            starting state.
        pending_normal, pending_escaped: String tokens that end in the
            middle of an escape sequence, for each starting state.
        str_open, quote_at_open, pending_open: Same three tables for the
            first token of a string, which must start with `` "`` (space +
            opening quote). Quote indices are relative to the full token.
            Letting the model pick the opening quote together with the
            first characters avoids "token healing" problems: Qwen merges
            `` "`` with what follows (e.g. `` "*``), so forcing a lone
            `` "`` token would push the model off its usual tokenization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    token_bytes: list[bytes]
    digits: IdArray
    minus: IdArray
    plus: IdArray
    dot: IdArray
    exponent: IdArray
    space: IdArray
    space_minus: IdArray
    number_end: IdArray
    str_normal: IdArray
    str_escaped: IdArray
    quote_at_normal: dict[int, int]
    quote_at_escaped: dict[int, int]
    pending_normal: frozenset[int]
    pending_escaped: frozenset[int]
    str_open: IdArray
    quote_at_open: dict[int, int]
    pending_open: frozenset[int]

    @classmethod
    def from_vocab_file(cls, path: str) -> "Vocabulary":
        """Build all tables from the model's ``vocab.json``.

        Args:
            path: Path returned by ``Small_LLM_Model.get_path_to_vocab_file``.

        Returns:
            A fully populated Vocabulary.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw_vocab: dict[str, int] = json.load(f)
        byte_decoder = {c: b for b, c in bytes_to_unicode().items()}
        size = max(raw_vocab.values()) + 1
        token_bytes = [b""] * size
        for text, tid in raw_vocab.items():
            token_bytes[tid] = bytes(byte_decoder[c] for c in text)
        return cls.from_token_bytes(token_bytes)

    @classmethod
    def from_token_bytes(cls, token_bytes: list[bytes]) -> "Vocabulary":
        """Classify every token for the number and string grammars.

        Args:
            token_bytes: Raw bytes of each token, indexed by token id.

        Returns:
            A Vocabulary with every mask precomputed.
        """
        buckets: dict[str, list[int]] = {
            k: []
            for k in (
                "digits", "minus", "plus", "dot", "exponent", "space",
                "space_minus",
                "number_end", "str_normal", "str_escaped", "str_open",
            )
        }
        quote_at: dict[str, dict[int, int]] = {
            "normal": {}, "escaped": {}, "open": {}}
        pending_ids: dict[str, set[int]] = {
            "normal": set(), "escaped": set(), "open": set()}
        singles = {b"-": "minus", b"+": "plus", b".": "dot",
                   b"e": "exponent", b"E": "exponent", b" ": "space",
                   b" -": "space_minus"}
        for tid, raw in enumerate(token_bytes):
            if not raw:
                continue
            if _DIGITS_RE.fullmatch(raw):
                buckets["digits"].append(tid)
            if raw in singles:
                buckets[singles[raw]].append(tid)
            if raw[:1] in (b",", b"}") or raw.isspace():
                buckets["number_end"].append(tid)
            scans = [("normal", raw, False, 0), ("escaped", raw, True, 0)]
            if raw.startswith(b' "'):
                scans.append(("open", raw[2:], False, 2))
            for state, body, escaped, offset in scans:
                valid, q, pending = scan_string_token(body, escaped)
                if not valid:
                    continue
                buckets["str_" + state].append(tid)
                if q >= 0:
                    quote_at[state][tid] = q + offset
                if pending:
                    pending_ids[state].add(tid)
        arrays = {k: np.array(v, dtype=np.int64) for k, v in buckets.items()}
        return cls(
            token_bytes=token_bytes,
            quote_at_normal=quote_at["normal"],
            quote_at_escaped=quote_at["escaped"],
            quote_at_open=quote_at["open"],
            pending_normal=frozenset(pending_ids["normal"]),
            pending_escaped=frozenset(pending_ids["escaped"]),
            pending_open=frozenset(pending_ids["open"]),
            **arrays,
        )
