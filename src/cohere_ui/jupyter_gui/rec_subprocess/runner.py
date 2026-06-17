"""Subprocess entry point and helpers.

Pure subprocess plumbing: stdio redirection, LiveViewBackend
construction, the spawn-mode wrapper around ``manage_reconstruction``.
No widgets, no threads, no tab state.
"""

import os
import sys
import traceback


class QueueWriter:
    """sys.stdout/stderr replacement that ships each line through ``queue``."""

    def __init__(self, queue, stream):
        self._queue = queue
        self._stream = stream
        self._buf = ''

    def write(self, s):
        self._buf += s
        while '\n' in self._buf:
            line, _, self._buf = self._buf.partition('\n')
            try:
                self._queue.put({'kind': 'stdout', 'stream': self._stream, 'line': line})
            except Exception as e:
                # Use __stderr__ to avoid recursing into self (sys.stderr IS this QueueWriter).
                sys.__stderr__.write(
                    f"QueueWriter.write failed ({type(e).__name__}: {e}); "
                    f"dropped line on {self._stream}: {line}\n"
                )

    def flush(self):
        pass

    def isatty(self):
        return False


def build_live_backend(backend_cfg, msg_queue):
    """Construct a LiveViewBackend from the GUI's BackendConfig dict.

    Returns ``None`` when the user has no live view active, in which case
    ``Rec`` falls back to ``MatplotlibBackend`` (which would try to open a
    window from the subprocess; harmless under Agg, but pointless).

    Dispatches via ``_backends.get_backend_builders()`` so a plugin
    that called ``register_backend(kind, builder)`` in the parent
    process before the subprocess fork is picked up here.
    """
    if not backend_cfg:
        return None
    kind = backend_cfg.get('kind')
    if kind is None:
        return None
    from cohere_ui.jupyter_gui._backends import get_backend_builders
    builder = get_backend_builders().get(kind)
    if builder is None:
        return None
    return builder(msg_queue, backend_cfg)


def run_reconstruction(experiment_dir, msg_queue, backend_cfg, kwargs):
    """Spawn-mode subprocess entry. Must be importable from the wrapper context.

    Sets up the process group, redirects stdio, registers the live-view
    backend (if any), then invokes ``manage_reconstruction``. The parent
    listener parses progress lines from the redirected stdout (no callback
    in cohere_core).
    """
    try:
        os.setsid()
    except (OSError, AttributeError):
        pass

    sys.stdout = QueueWriter(msg_queue, 'stdout')
    sys.stderr = QueueWriter(msg_queue, 'stderr')

    from cohere_core.utilities import view_utils as view_ut

    backend = build_live_backend(backend_cfg, msg_queue)
    if backend is not None:
        view_ut.set_default_live_backend(backend)

    try:
        import cohere_ui.cohere_reconstruction as run_rc
        run_rc.manage_reconstruction(experiment_dir, **kwargs)
        msg_queue.put({'kind': 'message', 'level': 'info', 'text': 'reconstruction finished'})
    except Exception as e:
        from cohere_ui.jupyter_gui.utils.error_format import format_error_summary
        msg_queue.put({
            'kind': 'message', 'level': 'error',
            'text': format_error_summary(e, prefix='run_reconstruction'),
        })
        msg_queue.put({
            'kind': 'message', 'level': 'debug',
            'text': traceback.format_exc(),
        })
        raise
    finally:
        view_ut.set_default_live_backend(None)
