"""Compare a tab's on-disk config to its current widget state.

Cheap mode (``deep=False``): existence-only. Deep mode (``deep=True``):
read the file and compare to ``tab.get_config()``. The deep path can be
expensive (``RecTab.get_config`` flattens every feature's widgets), so
``StateWatcher`` debounces change-burst notifications into one compute.
"""

import os
import threading
from typing import Callable, Literal, Optional

import cohere_core.utilities as ut

from cohere_ui.jupyter_gui.config import _strip_version

State = Literal['saved', 'modified', 'absent']


def compute(tab, config_manager, *, deep: bool = False) -> State:
    """Return ``'saved'``, ``'modified'``, or ``'absent'`` for ``tab``.

    Exceptions in the deep path fall back to ``'saved'`` so a broken
    widget can't poison the status strip.
    """
    if config_manager is None:
        return 'absent'
    conf_path = config_manager.conf_path(tab.conf_name)
    if conf_path is None or not os.path.isfile(conf_path):
        return 'absent'
    if not deep:
        return 'saved'
    try:
        widget_map = tab.get_config()
        # Ignore any legacy _schema_version stamp on disk: widget maps never
        # carry it, so leaving it in would make every saved tab look modified.
        disk_map = _strip_version(ut.read_config(conf_path)) or {}
    except Exception:
        return 'saved'
    return 'modified' if widget_map != disk_map else 'saved'


class StateWatcher:
    """Debounced watcher that recomputes a tab's state on widget changes."""

    DEBOUNCE_SEC = 0.15

    def __init__(self, tab, config_manager_getter: Callable,
                 on_change: Optional[Callable[[State], None]] = None):
        self._tab = tab
        self._cm_getter = config_manager_getter
        self._on_change = on_change
        self._timer: Optional[threading.Timer] = None
        self._paused = False
        self._last_state: Optional[State] = None
        self._lock = threading.Lock()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def _on_change_event(self, _change):
        if self._paused:
            return
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.DEBOUNCE_SEC, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self):
        state = compute(self._tab, self._cm_getter(), deep=True)
        if state != self._last_state:
            self._last_state = state
            if self._on_change is not None:
                try:
                    self._on_change(state)
                except Exception:
                    pass

    def recompute(self) -> State:
        """Force an immediate (non-debounced) recompute."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        state = compute(self._tab, self._cm_getter(), deep=True)
        if state != self._last_state:
            self._last_state = state
            if self._on_change is not None:
                try:
                    self._on_change(state)
                except Exception:
                    pass
        return state

    @property
    def last_state(self) -> Optional[State]:
        return self._last_state
