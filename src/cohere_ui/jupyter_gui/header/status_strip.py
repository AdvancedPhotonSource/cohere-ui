"""Slim status strip rendered above the tab strip.

One line: ``Current: <experiment_dir or "(none)">`` plus an optional
``multipeak`` suffix when the loaded experiment is configured for
multi-peak. Per-tab saved/modified markers live in the tab titles, not
here, so the strip and the tabs can't disagree about what a marker
refers to.
"""

from html import escape as _esc
from typing import Optional

import ipywidgets as widgets


class StatusStrip:
    """Renders the persistent header status line."""

    def __init__(self):
        self._exp_path: Optional[str] = None
        self._is_multipeak: bool = False
        self.widget = widgets.HTML(value=self._render())

    def set_state(self, *, experiment_dir: Optional[str],
                  is_multipeak: bool = False):
        self._exp_path = experiment_dir or None
        self._is_multipeak = bool(is_multipeak)
        self.widget.value = self._render()

    def _render(self) -> str:
        path_html = _esc(self._exp_path) if self._exp_path else '<i>(none)</i>'
        mp_suffix = (
            ' <span title="This experiment has multi-peak configuration." '
            'style="color:#1e7a1e; font-weight:600; margin-left:8px;">'
            '&middot; multipeak</span>'
            if self._is_multipeak else ''
        )
        return (
            '<div style="border:1px solid #ddd; border-radius:4px; '
            'background:#fafafa; margin:4px 0; '
            'font-family:Menlo,Consolas,monospace; font-size:12px; '
            'color:#222; padding:6px 10px;">'
            f'<b>Current:</b> {path_html}{mp_suffix}'
            '</div>'
        )
