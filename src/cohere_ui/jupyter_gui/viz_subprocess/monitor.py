"""VizMonitor: run the postprocess step in a subprocess with a listener thread.

The generic subset of :class:`RecMonitor` (spawn + queue drain + watchdog +
cleanup + stop), with none of the reconstruction-specific surface (no
progress-bar parsing, error history, snapshot images, or owned widgets).
It owns no UI: log lines are handed to a ``log`` callback so the caller
(DispTab) routes them into its existing LogPanel.
"""

import os
import signal
import threading

from cohere_ui.jupyter_gui.viz_subprocess.runner import run_visualization


class VizMonitor:
    """Manages a single visualization/postprocess subprocess.

    Callbacks:
      - ``on_running_changed(running: bool)`` - fired at start and at completion.
      - ``on_finished(exit_code: int)`` - fired after the child exits and the
        final queue records have been drained.
    """

    STOP_GRACE_SECONDS = 5.0

    def __init__(self, log=None):
        # log: callable(text: str, level: str) -> None. Defaults to a no-op.
        self._log_sink = log
        self.on_running_changed = None
        self.on_finished = None

        self._proc = None
        self._msg_queue = None
        self._listener_thread = None
        self._listener_stop = threading.Event()
        self._watchdog_thread = None
        self._stop_watchdog = threading.Event()
        self._last_exit_code = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def log(self, msg, level: str = 'info'):
        if self._log_sink is None:
            return
        try:
            self._log_sink(str(msg), level)
        except Exception:
            pass

    def start(self, experiment_dir, kwargs=None):
        """Spawn the subprocess and start the listener and watchdog threads."""
        import multiprocessing as mp

        if self.is_running:
            self.log('Visualization already running. Click Stop first.', 'warning')
            return

        self._last_exit_code = None

        ctx = mp.get_context('spawn')
        # macOS: Jupyter may launch via framework Python even inside a venv.
        # Pin the child to the venv's python so it sees the kernel's packages.
        venv_root = os.environ.get('VIRTUAL_ENV')
        if venv_root:
            venv_python = os.path.join(venv_root, 'bin', 'python')
            if os.path.isfile(venv_python):
                try:
                    ctx.set_executable(venv_python)
                except Exception as e:
                    self.log(f'could not pin spawn executable to {venv_python}: {e}',
                             'debug')

        self._msg_queue = ctx.Queue()
        self._proc = ctx.Process(
            target=run_visualization,
            args=(experiment_dir, self._msg_queue, dict(kwargs or {})),
            name='cohere-visualization',
        )
        self.log('Spawning visualization subprocess...')
        self._proc.start()
        self._set_running_state(True)

        self._listener_stop.clear()
        self._listener_thread = threading.Thread(
            target=self._run_listener, daemon=True, name='VizMonitor-listener',
        )
        self._listener_thread.start()

        self._stop_watchdog.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watch_subprocess, daemon=True, name='VizMonitor-watchdog',
        )
        self._watchdog_thread.start()

    def stop(self):
        """Graceful stop: SIGTERM, wait STOP_GRACE_SECONDS, then SIGKILL.

        SIGKILL is the fallback when the worker is wedged inside a C
        extension (e.g. a VTK write or xrayutilities call).
        """
        proc = self._proc
        if proc is None or not proc.is_alive():
            return

        self.log(f'Stop clicked: sending SIGTERM to process group of pid {proc.pid}')
        sent_term = False
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            sent_term = True
        except (AttributeError, ProcessLookupError, OSError) as e:
            self.log(f'killpg(SIGTERM) failed ({e}); falling back to proc.terminate()')
            try:
                proc.terminate()
                sent_term = True
            except Exception as e2:
                self.log(f'terminate fallback failed: {e2}')

        if not sent_term:
            return

        proc.join(timeout=self.STOP_GRACE_SECONDS)
        if proc.is_alive():
            self.log(
                f'Process still alive after {self.STOP_GRACE_SECONDS}s; '
                f'sending SIGKILL to process group of pid {proc.pid}'
            )
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (AttributeError, ProcessLookupError, OSError) as e:
                self.log(f'killpg(SIGKILL) failed ({e}); calling proc.kill()')
                try:
                    proc.kill()
                except Exception as e2:
                    self.log(f'proc.kill() failed: {e2}')

    def _set_running_state(self, running: bool):
        if self.on_running_changed is not None:
            try:
                self.on_running_changed(running)
            except Exception as e:
                self.log(f'on_running_changed callback error: {e}', 'debug')

    def _watch_subprocess(self):
        proc = self._proc
        while proc.is_alive() and not self._stop_watchdog.is_set():
            proc.join(timeout=0.5)
        proc.join(timeout=5)
        self._last_exit_code = proc.exitcode
        self.log(f'Subprocess exited with code {proc.exitcode}')
        self._cleanup_after_run()

    def _cleanup_after_run(self):
        # Stop the listener first so it stops trying to drain a dead Queue,
        # then drain any tail records (final "saved ...vts" / finished line).
        self._listener_stop.set()
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=2)
            self._listener_thread = None
        if self._msg_queue is not None:
            import queue as _queue
            for _ in range(200):
                try:
                    rec = self._msg_queue.get(timeout=0.05)
                except _queue.Empty:
                    break
                except (EOFError, BrokenPipeError, OSError):
                    break
                self._dispatch_record(rec)
            try:
                self._msg_queue.close()
            except Exception:
                pass
            self._msg_queue = None
        exit_code = self._last_exit_code
        self._proc = None
        self._set_running_state(False)
        if self.on_finished is not None:
            try:
                self.on_finished(exit_code)
            except Exception as e:
                self.log(f'on_finished callback error: {e}', 'debug')

    def _run_listener(self):
        """Drain the multiprocessing Queue, dispatch each record by ``kind``."""
        import queue as _queue
        q = self._msg_queue
        while not self._listener_stop.is_set():
            try:
                rec = q.get(timeout=0.1)
            except _queue.Empty:
                continue
            except (EOFError, BrokenPipeError, OSError):
                return
            self._dispatch_record(rec)

    def _dispatch_record(self, rec):
        kind = rec.get('kind')
        try:
            if kind == 'stdout':
                line = rec.get('line', '')
                if line:
                    self.log(line)
            elif kind == 'message':
                self.log(rec.get('text', ''), rec.get('level', 'info'))
        except Exception as e:
            self.log(f'listener dispatch error ({kind}): {e}', 'debug')
