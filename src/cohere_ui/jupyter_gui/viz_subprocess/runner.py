"""Subprocess entry point for the visualization/postprocess step.

Pure subprocess plumbing: stdio redirection plus the spawn-mode wrapper
around ``beamline_postprocess.handle_visualization``. No widgets, no
threads, no tab state. Reuses :class:`QueueWriter` from the reconstruction
runner so each stdout/stderr line is shipped back through the queue.
"""

import os
import sys
import traceback

from cohere_ui.jupyter_gui.rec_subprocess.runner import QueueWriter


def run_visualization(experiment_dir, msg_queue, kwargs):
    """Spawn-mode subprocess entry. Must be importable from the wrapper context.

    Redirects stdio through the queue, then invokes ``handle_visualization``.
    The parent listener streams the redirected stdout (the backend's
    ``saved ...vts`` / progress prints) straight into the tab log.
    """
    try:
        os.setsid()
    except (OSError, AttributeError):
        pass

    # Force a non-interactive matplotlib backend before any cohere import
    # pulls pyplot in: the child never has a display.
    os.environ.setdefault('MPLBACKEND', 'Agg')

    sys.stdout = QueueWriter(msg_queue, 'stdout')
    sys.stderr = QueueWriter(msg_queue, 'stderr')

    try:
        import cohere_ui.beamline_postprocess as dp
        dp.handle_visualization(experiment_dir, **(kwargs or {}))
        msg_queue.put({
            'kind': 'message', 'level': 'success', 'text': 'visualization finished',
        })
    except Exception as e:
        from cohere_ui.jupyter_gui.utils.error_format import format_error_summary
        msg_queue.put({
            'kind': 'message', 'level': 'error',
            'text': format_error_summary(e, prefix='run_visualization'),
        })
        msg_queue.put({
            'kind': 'message', 'level': 'debug',
            'text': traceback.format_exc(),
        })
        raise
