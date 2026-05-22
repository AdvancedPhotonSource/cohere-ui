"""Progress-line parsing and error-history bookkeeping.

Pure parsing + a small dataclass-style history container. 
Imported by both the listener (which feeds
records in) and the log-view module (which renders them out).
"""

import re
import sys

from cohere_ui.jupyter_gui.error_format import format_error_summary

PROGRESS_PATTERNS = (
    re.compile(r'^------iter\s+(\d+)\s+error\s+(\S+)\s*$'),
    re.compile(r'^\|\s+iter\s+(\d+)\s+\|.*\|\s+err\s+([0-9.eE+-]+)\s+\|'),
)


def total_iters_from_alg_sequence(seq: str) -> int:
    """Sum the iteration counts in a cohere algorithm_sequence string.

    Replaces every ``*<algname>`` token with ``*1`` so the remaining
    expression is pure arithmetic over integers and grouping parens, then
    evaluates it. Returns 0 on parse failure (progress bar stays in
    indeterminate visual state).
    """
    if not seq:
        return 0
    arithmetic = re.sub(r'\*[A-Za-z][A-Za-z0-9.]*', '*1', seq.replace(' ', ''))
    try:
        return int(eval(arithmetic, {'__builtins__': {}}, {}))
    except Exception as e:
        sys.stderr.write(
            f"total_iters_from_alg_sequence: could not parse {seq!r} "
            f"(reduced to {arithmetic!r}); progress bar will stay indeterminate. "
            f"{format_error_summary(e)}\n"
        )
        return 0


def parse_progress_line(line: str):
    """Return ``(iter, error)`` if the line matches a progress pattern, else None."""
    for pat in PROGRESS_PATTERNS:
        m = pat.match(line)
        if m:
            try:
                return int(m.group(1)), float(m.group(2))
            except ValueError:
                return None
    return None


class ErrorHistory:
    """Append-only (iter, error) log used to render the convergence plot.

    Skips the first reading (the pre-modulus random-init error would
    dominate the log scale).
    """

    def __init__(self):
        self._points = []

    def append(self, iteration, error) -> bool:
        """Add a point. Returns True when the caller should re-render."""
        if iteration is None or error is None:
            return False
        if self._points and self._points[-1][0] == iteration:
            return False
        self._points.append((iteration, error))
        return len(self._points) >= 2

    def points_for_plot(self):
        """Return points to plot, dropping the first reading."""
        return self._points[1:]

    def last_iter(self):
        return self._points[-1][0] if self._points else None

    def reset(self):
        self._points = []

    def __len__(self):
        return len(self._points)

    def __bool__(self):
        return bool(self._points)
