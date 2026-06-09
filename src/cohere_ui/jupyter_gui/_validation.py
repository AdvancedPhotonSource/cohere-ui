"""Pure validation/parsing helpers used by jupyter_gui tabs.

This module MUST NOT import ipywidgets. Tests import it without a Jupyter kernel.
"""
from __future__ import annotations

import ast
import difflib
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence


KNOWN_ALGORITHMS: tuple[str, ...] = (
    "ER", "HIO", "RAAR", "SF",
    "ERpc", "HIOpc", "RAARpc", "SFpc",
    "modulus", "phase_only",
)


@dataclass(frozen=True)
class ValidationError:
    field: str
    message: str
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        base = f"{self.field}: {self.message}"
        return f"{base} ({self.suggestion})" if self.suggestion else base


class _ParseError(Exception):
    def __init__(self, message: str, suggestion: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion


class _AlgSeqParser:
    """Recursive-descent parser for the cohere algorithm_sequence grammar.

    Grammar:
        sum    := term ('+' term)*
        term   := integer '*' factor | factor
        factor := name | '(' sum ')'

    A multiplier in front of a bare algorithm name collapses to a single
    (name, count) pair; in front of a parenthesised sub-expression it
    repeats the list ``count`` times. Mirrors how cohere's Rec expands the
    sequence into per-iteration operations.
    """

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def at_end(self) -> bool:
        return self.pos >= len(self.text)

    def peek(self) -> str:
        return self.text[self.pos] if not self.at_end() else ""

    def consume(self, ch: str) -> bool:
        if self.peek() == ch:
            self.pos += 1
            return True
        return False

    def parse_sum(self) -> list[tuple[str, int]]:
        result = self.parse_term()
        while self.consume("+"):
            result.extend(self.parse_term())
        return result

    def parse_term(self) -> list[tuple[str, int]]:
        start = self.pos
        n = self._parse_integer()
        if n is not None and self.consume("*"):
            return self._parse_factor_with_count(n)
        if n is not None:
            raise _ParseError(
                f'expected "*" after multiplier {n} at position {self.pos}'
            )
        self.pos = start
        return self._parse_factor_with_count(1)

    def _parse_factor_with_count(self, n: int) -> list[tuple[str, int]]:
        if self.consume("("):
            inner = self.parse_sum()
            if not self.consume(")"):
                raise _ParseError(
                    f'unbalanced parentheses: expected ")" at position {self.pos}'
                )
            return inner * n
        name = self._parse_alg_name()
        return [(name, n)]

    def _parse_integer(self) -> Optional[int]:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        if self.pos == start:
            return None
        return int(self.text[start:self.pos])

    def _parse_alg_name(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and (
            self.text[self.pos].isalnum() or self.text[self.pos] == "_"
        ):
            self.pos += 1
        if self.pos == start:
            raise _ParseError(
                f"expected algorithm name at position {self.pos}, got "
                f"{self.peek()!r}"
            )
        name = self.text[start:self.pos]
        if name not in KNOWN_ALGORITHMS:
            # difflib is case-sensitive; try the case-matched variant first
            # so that 'er' suggests 'ER' rather than scoring it below cutoff.
            close = difflib.get_close_matches(name, KNOWN_ALGORITHMS, n=1)
            if not close:
                upper = name.upper()
                if upper in KNOWN_ALGORITHMS:
                    close = [upper]
                else:
                    close = difflib.get_close_matches(upper, KNOWN_ALGORITHMS, n=1)
            suggestion = f"did you mean {close[0]!r}?" if close else None
            raise _ParseError(f"unknown algorithm {name!r}", suggestion=suggestion)
        return name


def parse_algorithm_sequence(text: str) -> list[tuple[str, int]] | ValidationError:
    """Parse e.g. ``'3*(20*ER+180*HIO)+20*ER'`` into a list of ``(name, count)``.

    Returns a :class:`ValidationError` for unknown algorithm tokens,
    unbalanced parentheses, or malformed multipliers. Case-sensitive — ``'er'``
    is rejected with a suggestion of ``'ER'``.

    The sum of counts equals the total iteration count
    (``total_iters_from_alg_sequence`` from ``rec_subprocess/progress.py``).
    """
    text = (text or "").strip()
    if not text:
        return ValidationError("algorithm_sequence", "sequence is empty")
    text = text.replace(" ", "")
    parser = _AlgSeqParser(text)
    try:
        result = parser.parse_sum()
    except _ParseError as e:
        return ValidationError("algorithm_sequence", e.message, suggestion=e.suggestion)
    if not parser.at_end():
        return ValidationError(
            "algorithm_sequence",
            f"unexpected character {parser.peek()!r} at position {parser.pos}",
        )
    return result


def validate_beta(name: str, value: Any) -> ValidationError | None:
    """``beta`` must be a real number in the open interval (0, 1)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ValidationError(name, f"expected a number, got {value!r}")
    if not (0.0 < v < 1.0):
        return ValidationError(name, f"must be in (0, 1), got {v}")
    return None


def _coerce_sequence(value: Any) -> Optional[Sequence]:
    if isinstance(value, (list, tuple)):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, (list, tuple)):
            return parsed
    return None


def validate_support_area(value: Any) -> ValidationError | None:
    """``initial_support_area`` must be a length-3 sequence of floats in (0, 1]."""
    seq = _coerce_sequence(value)
    if seq is None:
        return ValidationError(
            "initial_support_area",
            f"expected a 3-element sequence of floats, got {value!r}",
        )
    if len(seq) != 3:
        return ValidationError(
            "initial_support_area",
            f"expected 3 elements, got {len(seq)}",
        )
    for i, x in enumerate(seq):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return ValidationError(
                "initial_support_area",
                f"element {i} is not a number: {x!r}",
            )
        if not (0.0 < xf <= 1.0):
            return ValidationError(
                "initial_support_area",
                f"element {i} must be in (0, 1], got {xf}",
            )
    return None


def validate_device_field(
    text: str,
    available: Optional[Sequence] = None,
) -> ValidationError | None:
    """Reject empty/whitespace device strings; if ``available`` is given,
    reject values not in the list. Empty ``available`` (or None) is lenient.
    """
    if text is None or not str(text).strip():
        return ValidationError("device", "device field is empty")
    text = str(text).strip()
    if available is None or len(available) == 0:
        return None
    if text in available:
        return None
    available_str = [str(a) for a in available]
    if text in available_str:
        return None
    close = difflib.get_close_matches(text, available_str, n=1)
    suggestion = f"available: {', '.join(available_str)}"
    if close:
        suggestion = f"did you mean {close[0]!r}? ({suggestion})"
    return ValidationError("device", f"unknown device {text!r}", suggestion=suggestion)


# Path-typed keys per config name. Empty/None values are skipped.
_PATH_FIELDS: dict[str, tuple[str, ...]] = {
    "config_rec": ("continue_dir", "AI_trained_model"),
    "config_data": ("alien_file",),
}


def validate_paths(conf_map: dict, conf_name: str) -> list[ValidationError]:
    """Check that path-typed fields exist on disk when set. Empty / missing
    keys are silently OK (the field is optional). Returns a possibly-empty
    list of :class:`ValidationError`.
    """
    if conf_map is None:
        return []
    errors: list[ValidationError] = []
    for key in _PATH_FIELDS.get(conf_name, ()):
        value = conf_map.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, str) and not os.path.exists(value):
            errors.append(
                ValidationError(key, f"path does not exist: {value}")
            )
    return errors


def validate_algorithm_sequence(value: Any) -> ValidationError | None:
    """Wrapper that adapts :func:`parse_algorithm_sequence` to the
    ``ValidationError | None`` shape FIELD_VALIDATORS expects."""
    if not isinstance(value, str):
        return ValidationError(
            "algorithm_sequence",
            f"expected a string, got {type(value).__name__}",
        )
    result = parse_algorithm_sequence(value)
    return result if isinstance(result, ValidationError) else None


FIELD_VALIDATORS: dict[str, Callable[[Any], "ValidationError | None"]] = {
    "hio_beta": lambda v: validate_beta("hio_beta", v),
    "raar_beta": lambda v: validate_beta("raar_beta", v),
    "initial_support_area": validate_support_area,
    "algorithm_sequence": validate_algorithm_sequence,
}


def validate_field(name: str, value: Any) -> ValidationError | None:
    """Look up ``name`` in :data:`FIELD_VALIDATORS`; unknown fields pass."""
    validator = FIELD_VALIDATORS.get(name)
    if validator is None:
        return None
    return validator(value)
