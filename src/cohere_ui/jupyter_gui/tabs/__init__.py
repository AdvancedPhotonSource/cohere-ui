"""Tab implementations + the tab registry for the Jupyter GUI.

The registry is a single source of truth for tab membership, display
order, visibility, and run-all participation. CoherenceGUI consults
``get_registered_tabs()`` at construction time and never hard-codes
the tab list.

Third-party code can extend the GUI by calling :func:`register_tab`
with a :class:`TabSpec` BEFORE constructing the GUI:

    from cohere_ui.jupyter_gui.tabs import TabSpec, register_tab
    from cohere_ui.jupyter_gui.tabs.base import BaseTab

    class QualityTab(BaseTab):
        name = 'Quality'
        conf_name = 'config_quality'
        # ... implement the abstractmethods ...

    register_tab(TabSpec(key='quality', factory=QualityTab))

Or, when the optional :mod:`importlib.metadata` entry-point group is
defined in ``pyproject.toml`` under ``cohere_ui.jupyter_gui.tabs``,
the GUI will discover and register them automatically the first time
:func:`get_registered_tabs` is called.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

from cohere_ui.jupyter_gui.tabs.base import BaseTab
from cohere_ui.jupyter_gui.tabs.data import DataTab
from cohere_ui.jupyter_gui.tabs.prep import PrepTab
from cohere_ui.jupyter_gui.tabs.rec import RecTab
from cohere_ui.jupyter_gui.tabs.disp import DispTab
from cohere_ui.jupyter_gui.tabs.instr import InstrTab
from cohere_ui.jupyter_gui.tabs.mp import MpTab


@dataclass(frozen=True)
class TabSpec:
    """Declarative description of a tab the GUI should mount.

    ``factory`` is a zero-argument callable that returns a fresh
    ``BaseTab`` instance. This is usually the tab class itself, or a
    ``lambda`` when the constructor needs arguments (e.g.
    ``InstrTab(beamline=None)``).

    Set ``optional=True`` together with a ``visible_when`` predicate to
    mount the tab only when the predicate returns True; the GUI
    re-evaluates it on every experiment load/create.

    Set ``skip_in_run_all=True`` for tabs that have no backend step
    (e.g. ``InstrTab``).
    """
    key: str
    factory: Callable[[], BaseTab]
    optional: bool = False
    visible_when: Optional[Callable[[object], bool]] = None
    skip_in_run_all: bool = False


DEFAULT_TABS: List[TabSpec] = [
    TabSpec(key='instr', factory=lambda: InstrTab(beamline=None),
            skip_in_run_all=True),
    TabSpec(key='prep',  factory=PrepTab),
    TabSpec(key='data',  factory=DataTab),
    TabSpec(key='rec',   factory=RecTab),
    TabSpec(key='disp',  factory=DispTab),
    TabSpec(key='mp',    factory=MpTab, optional=True,
            visible_when=lambda gui: gui._is_multipeak_experiment()),
]

_REGISTERED_TABS: List[TabSpec] = list(DEFAULT_TABS)
_ENTRY_POINTS_LOADED = False
_TAB_ENTRY_POINT_GROUP = 'cohere_ui.jupyter_gui.tabs'


def register_tab(spec: TabSpec, *, position: Optional[int] = None) -> None:
    """Register ``spec`` in the tab list.

    Rejects duplicate keys: re-registering the same key raises
    ``ValueError`` rather than silently shadowing the existing entry.
    Pass ``position`` to insert at a specific index (default: append).

    Only affects subsequently-constructed ``CoherenceGUI`` instances;
    live GUIs keep the spec list they were built with.
    """
    if any(s.key == spec.key for s in _REGISTERED_TABS):
        raise ValueError(f'Tab already registered: {spec.key!r}')
    if position is None:
        _REGISTERED_TABS.append(spec)
    else:
        _REGISTERED_TABS.insert(position, spec)


def unregister_tab(key: str) -> bool:
    """Remove the tab with ``key`` from the registry.

    Returns True if a tab was removed, False if no such key existed.
    Useful for tests and for downstream apps that want to hide a
    built-in tab.
    """
    for i, spec in enumerate(_REGISTERED_TABS):
        if spec.key == key:
            del _REGISTERED_TABS[i]
            return True
    return False


def _load_entry_point_tabs() -> None:
    """Discover ``cohere_ui.jupyter_gui.tabs`` entry points and register them.

    Each entry point's value must resolve to either a ``TabSpec``
    instance or a zero-argument callable that returns one. Failures
    are silent at import time but logged via the standard ``logging``
    module so a missing optional plugin never blocks the GUI from
    coming up.
    """
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return
    try:
        eps = entry_points(group=_TAB_ENTRY_POINT_GROUP)
    except TypeError:
        # importlib.metadata < 3.10 returns a dict-of-lists
        eps = entry_points().get(_TAB_ENTRY_POINT_GROUP, ())
    import logging
    log = logging.getLogger(__name__)
    for ep in eps:
        try:
            obj = ep.load()
            spec = obj() if callable(obj) and not isinstance(obj, TabSpec) else obj
            if not isinstance(spec, TabSpec):
                log.warning(
                    'Tab entry point %r resolved to %r, expected TabSpec',
                    ep.name, type(spec).__name__,
                )
                continue
            try:
                register_tab(spec)
            except ValueError as e:
                log.warning('Tab entry point %r skipped: %s', ep.name, e)
        except Exception as e:
            log.warning('Tab entry point %r failed to load: %s', ep.name, e)


def get_registered_tabs() -> List[TabSpec]:
    """Return the current tab spec list, in display order.

    Loads entry-point-defined tabs on first call (one-shot). Subsequent
    calls return the same list (mutable in place via
    ``register_tab`` / ``unregister_tab``).
    """
    _load_entry_point_tabs()
    return list(_REGISTERED_TABS)


__all__ = [
    'BaseTab', 'DataTab', 'PrepTab', 'RecTab', 'DispTab', 'InstrTab', 'MpTab',
    'TabSpec', 'DEFAULT_TABS', 'register_tab', 'unregister_tab',
    'get_registered_tabs',
]
