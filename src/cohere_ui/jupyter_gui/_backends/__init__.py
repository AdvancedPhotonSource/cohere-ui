"""Live-view backends used by jupyter_gui's reconstruction subprocess.

Each backend subclasses ``cohere_core.utilities.view_utils.LiveViewBackend``
and ships rendered output (PNG bytes or raw volume data) back to the parent
kernel via the ``multiprocessing.Queue`` it receives at construction.

The ``BACKEND_BUILDERS`` dict maps a ``kind`` string (the value the GUI
puts in the ``BackendConfig`` dict) to a callable that constructs the
backend. The subprocess runner (``rec_subprocess.runner``) consults
this dict instead of hard-coding the kind/class mapping, so a plugin can
register a new live-view backend with:

    from cohere_ui.jupyter_gui._backends import register_backend

    def _build_my_backend(msg_queue, cfg):
        return MyBackend(msg_queue, my_param=cfg.get('my_param'))

    register_backend('my_kind', _build_my_backend)

Entry-point group ``cohere_ui.jupyter_gui.backends`` is loaded on first
access via :func:`get_backend_builders`.
"""

from typing import Callable, Dict

from cohere_ui.jupyter_gui._backends.matplotlib_png import JupyterMatplotlibBackend
from cohere_ui.jupyter_gui._backends.pyvista_png import PyVistaBackend


def _build_matplotlib(msg_queue, cfg):
    return JupyterMatplotlibBackend(
        msg_queue,
        mode=cfg.get('mode', 'center_slice'),
        slice_axis=cfg.get('slice_axis', 2),
        slice_method=cfg.get('slice_method', 'center_of_mass'),
        stride=cfg.get('stride', 4),
        phase_cmap=cfg.get('phase_cmap', 'twilight'),
        apply_support_mask=cfg.get('apply_support_mask', True),
    )


def _build_pyvista(msg_queue, cfg):
    return PyVistaBackend(
        msg_queue,
        stride=cfg.get('stride', 4),
        iso_level=cfg.get('iso_level', 0.3),
    )


BACKEND_BUILDERS: Dict[str, Callable] = {
    'matplotlib': _build_matplotlib,
    'pyvista':    _build_pyvista,
}

_BACKEND_ENTRY_POINT_GROUP = 'cohere_ui.jupyter_gui.backends'
_BACKEND_EP_LOADED = False


def register_backend(kind: str, builder: Callable) -> None:
    """Register a live-view backend under ``kind``.

    ``builder`` is a callable ``(msg_queue, cfg_dict) -> LiveViewBackend``
    that the subprocess runner invokes when the GUI's BackendConfig
    payload has ``kind == <this kind>``. Re-registering the same kind
    raises ``ValueError``. Call :func:`unregister_backend` first to
    override a built-in.
    """
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError('backend kind must be a non-empty string')
    if not callable(builder):
        raise TypeError(f'builder must be callable, got {builder!r}')
    if kind in BACKEND_BUILDERS:
        raise ValueError(f'backend already registered: {kind!r}')
    BACKEND_BUILDERS[kind] = builder


def unregister_backend(kind: str) -> bool:
    """Remove a backend from the registry. Returns True if removed."""
    return BACKEND_BUILDERS.pop(kind, None) is not None


def _load_entry_point_backends() -> None:
    """Discover ``cohere_ui.jupyter_gui.backends`` entry points and register them."""
    global _BACKEND_EP_LOADED
    if _BACKEND_EP_LOADED:
        return
    _BACKEND_EP_LOADED = True
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return
    try:
        eps = entry_points(group=_BACKEND_ENTRY_POINT_GROUP)
    except TypeError:
        eps = entry_points().get(_BACKEND_ENTRY_POINT_GROUP, ())
    import logging
    log = logging.getLogger(__name__)
    for ep in eps:
        try:
            builder = ep.load()
            if not callable(builder):
                log.warning(
                    'Backend entry point %r resolved to non-callable %r',
                    ep.name, type(builder).__name__,
                )
                continue
            try:
                register_backend(ep.name, builder)
            except ValueError as e:
                log.warning('Backend entry point %r skipped: %s', ep.name, e)
        except Exception as e:
            log.warning('Backend entry point %r failed to load: %s', ep.name, e)


def get_backend_builders() -> Dict[str, Callable]:
    """Return the current map from ``kind`` to ``builder``, loading entry points once."""
    _load_entry_point_backends()
    return BACKEND_BUILDERS


__all__ = [
    'JupyterMatplotlibBackend', 'PyVistaBackend',
    'BACKEND_BUILDERS', 'register_backend', 'unregister_backend',
    'get_backend_builders',
]
