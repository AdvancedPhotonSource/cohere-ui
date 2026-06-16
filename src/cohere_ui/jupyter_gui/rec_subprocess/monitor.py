"""RecMonitor: subprocess + listener thread + watchdog.

Lifecycle and threading only. The widget tree (progress label, live
view, error plot, log, image toolbar) lives in
:mod:`monitor_widgets`; the converged-image render lives in
:mod:`monitor_snapshot`. RecMonitor composes both and hoists the
widget attributes onto itself so RecTab and CoherenceGUI can keep
reaching them as ``monitor.progress_label`` /
``monitor.show_debug_checkbox`` etc.

The Reconstruction tab constructs one of these in its ``_build_ui``
and embeds ``monitor.widgets_box()`` in its layout.
"""

import os
import signal
import threading
import time

import ipywidgets as widgets

from cohere_ui.jupyter_gui.utils.error_format import format_error_summary
from cohere_ui.jupyter_gui.text import load_text
from cohere_ui.jupyter_gui.rec_subprocess.monitor_snapshot import (
    show_final_snapshot as _do_show_final_snapshot,
)
from cohere_ui.jupyter_gui.rec_subprocess.monitor_widgets import RecMonitorWidgets
from cohere_ui.jupyter_gui.rec_subprocess.progress import ErrorHistory, parse_progress_line
from cohere_ui.jupyter_gui.rec_subprocess.log_view import render_error_plot
from cohere_ui.jupyter_gui.rec_subprocess.runner import run_reconstruction

_UI = load_text('ui_strings')


# Names of widget attributes that external callers (RecTab, CoherenceGUI)
# reach into directly on the monitor instance. Each is hoisted from the
# embedded RecMonitorWidgets in ``__init__`` so existing access patterns
# keep working without going through a proxy.
_HOISTED_WIDGET_ATTRS = (
    'progress_label', 'progress_bar',
    'live_view', 'live_view_caption', 'error_plot',
    'show_log_checkbox', 'show_debug_checkbox', 'log_widget',
    'image_size_dropdown', 'copy_log_button',
    'image_toolbar', 'log_toolbar',
)


def _format_eta(seconds: float) -> str:
    """Format a remaining-time estimate as a short human string."""
    s = int(round(seconds))
    if s < 1:
        return '<1s'
    if s < 60:
        return f'{s}s'
    m, s = divmod(s, 60)
    if m < 60:
        return f'{m}m {s:02d}s'
    h, m = divmod(m, 60)
    return f'{h}h {m:02d}m {s:02d}s'


class RecMonitor:
    """Manages a single reconstruction subprocess and its UI surface."""

    PROGRESS_WARN_SECONDS = 30.0
    ITER_RATE_TICK_SECONDS = 0.2
    STOP_GRACE_SECONDS = 5.0
    STUCK_WARN_SECONDS = 300.0
    STUCK_POLL_SECONDS = 30.0

    def __init__(self):
        self._w = RecMonitorWidgets()
        # Expose the widget attributes that RecTab and CoherenceGUI
        # access directly on the monitor object. This keeps the original
        # access surface unchanged.
        for attr in _HOISTED_WIDGET_ATTRS:
            setattr(self, attr, getattr(self._w, attr))

        self._error_history = ErrorHistory()

        # Capture the kernel's IOLoop now (we're on the main thread
        # during CoherenceGUI construction). The watchdog thread uses
        # this to marshal main-thread-only renders (PyVista's OpenGL
        # context on Cocoa) back onto the kernel's main thread.
        try:
            import tornado.ioloop
            self._main_loop = tornado.ioloop.IOLoop.current(instance=False)
        except Exception:
            self._main_loop = None

        # Subprocess + thread bookkeeping.
        self._proc = None
        self._msg_queue = None
        self._listener_thread = None
        self._listener_stop = threading.Event()
        self._watchdog_thread = None
        self._stop_watchdog = threading.Event()
        self._progress_seen = False
        self._progress_warning_timer = None
        self._iter_rate = None
        self._last_iter_value = 0
        self._last_iter_wall = None
        self._iter_rate_ticker = None
        self._iter_rate_stop = threading.Event()
        self._last_exit_code = None
        self._stuck_thread = None
        self._stuck_stop = threading.Event()
        self._stuck_warned = False
        self._expected_total_iters = 0
        # Stashed at start() so show_final_snapshot honors the user's
        # selected renderer instead of forcing matplotlib center_slice.
        self._backend_cfg = None
        self._experiment_dir = None

        self.on_running_changed = None
        self.on_finished = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def widgets_box(self) -> widgets.VBox:
        """Return the monitor's widgets stacked in display order."""
        return self._w.widgets_box()

    def start(self, experiment_dir, backend_cfg, kwargs,
              total_iters: int, show_progress_bar: bool):
        """Spawn the subprocess and start the listener and watchdog threads."""
        import multiprocessing as mp

        if self.is_running:
            self.log('Reconstruction already running. Click Stop first.')
            return

        # Saved for show_final_snapshot to honor the user's renderer.
        self._backend_cfg = backend_cfg
        self._experiment_dir = experiment_dir

        self._reset_for_run(total_iters, show_progress_bar)

        ctx = mp.get_context('spawn')
        # macOS: Jupyter may launch via framework Python even inside a
        # venv. Force the child to the venv's python so it sees the
        # kernel's packages (otherwise the subprocess hits
        # ModuleNotFoundError).
        venv_root = os.environ.get('VIRTUAL_ENV')
        if venv_root:
            venv_python = os.path.join(venv_root, 'bin', 'python')
            if os.path.isfile(venv_python):
                try:
                    ctx.set_executable(venv_python)
                except Exception as e:
                    self.log(
                        f'could not pin spawn executable to {venv_python}: {e}',
                        level='debug',
                    )
        self._msg_queue = ctx.Queue()
        self._proc = ctx.Process(
            target=run_reconstruction,
            args=(experiment_dir, self._msg_queue, backend_cfg, kwargs),
            name='cohere-reconstruction',
        )
        self.log('Spawning reconstruction subprocess...')
        self._proc.start()
        self._set_running_state(True)

        self._listener_stop.clear()
        self._listener_thread = threading.Thread(
            target=self._run_listener, daemon=True, name='RecMonitor-listener',
        )
        self._listener_thread.start()

        # Silent-format-break detection: warn once if no progress lines
        # have been parsed within PROGRESS_WARN_SECONDS.
        if self._progress_warning_timer is not None:
            self._progress_warning_timer.cancel()
        self._progress_warning_timer = threading.Timer(
            self.PROGRESS_WARN_SECONDS, self._warn_if_no_progress,
        )
        self._progress_warning_timer.daemon = True
        self._progress_warning_timer.start()

        self._stop_watchdog.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watch_subprocess, daemon=True, name='RecMonitor-watchdog',
        )
        self._watchdog_thread.start()

        # Iter-rate ticker advances the bar between real iter events at
        # the observed rate.
        if self.progress_bar.layout.visibility == 'visible':
            self._iter_rate_stop.clear()
            self._iter_rate_ticker = threading.Thread(
                target=self._iter_rate_progress_loop, daemon=True,
                name='RecMonitor-iter-rate-progress',
            )
            self._iter_rate_ticker.start()

        # Stuck-process watchdog: warn once after STUCK_WARN_SECONDS of
        # silence. Never auto-kills.
        self._stuck_warned = False
        self._expected_total_iters = max(0, int(total_iters))
        if self._expected_total_iters > 0:
            self._stuck_stop.clear()
            self._stuck_thread = threading.Thread(
                target=self._stuck_watchdog_loop, daemon=True,
                name='RecMonitor-stuck-watchdog',
            )
            self._stuck_thread.start()

    def stop(self):
        """Graceful stop: SIGTERM, wait STOP_GRACE_SECONDS, then SIGKILL.

        The soft kill lets the reconstruction loop flush the last image
        and error history before exiting. SIGKILL is the fallback when
        the worker is wedged inside a C extension.
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
            # Windows or already-dead: fall back to multiprocessing
            # terminate (SIGTERM on Unix, CTRL_BREAK_EVENT on Windows).
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

    # log / debug forwarding to RecMonitorWidgets

    def log(self, msg, level: str = 'info'):
        """Append a line to the log widget at the given level."""
        self._w.append(msg, level=level)

    def clear_log(self):
        """Clear the log widget."""
        self._w.clear()

    def set_show_debug(self, value: bool):
        self._w.set_show_debug(value)

    @property
    def show_debug(self) -> bool:
        return self._w.show_debug

    # image setters (route through the widgets' visibility logic)

    def set_live_view(self, png):
        """Set the live-view image; revealed only when it has bytes."""
        self._w.set_live_view(png)

    def set_error_plot(self, png):
        """Set the error-history plot; revealed only when it has bytes."""
        self._w.set_error_plot(png)

    # final snapshot delegation

    def show_final_snapshot(self, image, support, errors, backend_cfg):
        """See :func:`monitor_snapshot.show_final_snapshot`."""
        _do_show_final_snapshot(self, image, support, errors, backend_cfg)

    # internal lifecycle

    def _reset_for_run(self, total_iters: int, show_progress_bar: bool):
        self.clear_log()
        # Empty + hide both images for the new run; they reveal again when
        # the first snapshot / error of this run arrives.
        self._w.set_live_view(b'')
        self.live_view_caption.value = _UI['snapshot_panel']['running']
        self._w.set_error_plot(b'')
        self._error_history.reset()
        self._progress_seen = False
        self.progress_label.value = '<i>Preparing...</i>'

        if total_iters > 0 and show_progress_bar:
            self.progress_bar.max = total_iters
            self.progress_bar.value = 0
            self.progress_bar.bar_style = 'info'
            self.progress_bar.layout.visibility = 'visible'
        else:
            self.progress_bar.layout.visibility = 'hidden'

        self._iter_rate = None
        self._last_iter_value = 0
        self._last_iter_wall = None

    def _set_running_state(self, running: bool):
        if not running:
            self.progress_label.value = '<i>Idle</i>'
            # Success when the worker exited cleanly; warning when killed
            # or errored. cohere reports iter 0..N-1, so the bar's last
            # observed value is max-1.
            if self._last_exit_code == 0 and self.progress_bar.max > 1:
                self.progress_bar.value = self.progress_bar.max
                self.progress_bar.bar_style = 'success'
            elif self.progress_bar.value > 0:
                self.progress_bar.bar_style = 'warning'
        if self.on_running_changed is not None:
            try:
                self.on_running_changed(running)
            except Exception as e:
                self.log(f'on_running_changed callback error: {e}')

    def _watch_subprocess(self):
        proc = self._proc
        while proc.is_alive() and not self._stop_watchdog.is_set():
            proc.join(timeout=0.5)
        proc.join(timeout=5)
        exit_code = proc.exitcode
        self.log(f'Subprocess exited with code {exit_code}')
        self._last_exit_code = exit_code
        self._cleanup_after_run()

    def _cleanup_after_run(self):
        # Stop listener first so it stops trying to drain a dead Queue.
        self._listener_stop.set()
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=2)
            self._listener_thread = None
        if self._progress_warning_timer is not None:
            self._progress_warning_timer.cancel()
            self._progress_warning_timer = None
        self._iter_rate_stop.set()
        if self._iter_rate_ticker is not None:
            self._iter_rate_ticker.join(timeout=1)
            self._iter_rate_ticker = None
        self._stuck_stop.set()
        if self._stuck_thread is not None:
            self._stuck_thread.join(timeout=1)
            self._stuck_thread = None
        # Drain remaining queue records before closing. Without this,
        # the listener can be stopped before the tail of the run's
        # output (final iter print, "iterate took", etc.) is
        # dispatched, dropping the last data point from the error plot
        # and the closing log lines.
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
            except Exception as e:
                self.log(format_error_summary(e, prefix='_cleanup_after_run'), level='debug')
            self._msg_queue = None
        exit_code = self._last_exit_code
        self._proc = None
        self._set_running_state(False)
        if self.on_finished is not None:
            try:
                self.on_finished(exit_code)
            except Exception as e:
                self.log(f'on_finished callback error: {e}')

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
            if kind == 'snapshot':
                self._on_snapshot(rec)
            elif kind == 'stdout':
                self._on_stdout(rec)
            elif kind == 'message':
                self._on_message(rec)
        except Exception as e:
            self.log(f'listener dispatch error ({kind}): {e}')

    def _on_iter(self, iteration, error):
        total = self.progress_bar.max
        label_total = f'/{total}' if total > 1 else ''
        if total > 1:
            now = time.monotonic()
            if self._last_iter_wall is not None:
                dt = now - self._last_iter_wall
                di = int(iteration) - self._last_iter_value
                if dt > 0 and di > 0:
                    instant_rate = di / dt
                    # EMA: weight 0.4 toward newest sample
                    if self._iter_rate is None:
                        self._iter_rate = instant_rate
                    else:
                        self._iter_rate = 0.6 * self._iter_rate + 0.4 * instant_rate
            self._last_iter_value = int(iteration)
            self._last_iter_wall = now
            self.progress_bar.value = min(int(iteration), total)
        parts = [
            f'<b>iter {iteration}{label_total}</b>',
            f'error <code>{error:.6g}</code>',
        ]
        rate = self._iter_rate
        if rate and rate > 0:
            parts.append(f'<code>{rate:.2f}</code> it/s')
            if total > 1:
                remaining = max(0, int(total) - int(iteration))
                if remaining > 0:
                    parts.append(f'ETA <code>{_format_eta(remaining / rate)}</code>')
        self.progress_label.value = ' &nbsp;|&nbsp; '.join(parts)
        self._record_error(iteration, error)

    def _on_stdout(self, record):
        line = record.get('line', '')
        if not line:
            return
        parsed = parse_progress_line(line)
        if parsed is not None:
            self._on_iter(*parsed)
            self._progress_seen = True
        self.log(line)

    def _on_snapshot(self, record):
        err = record.get('error')
        iter_n = record.get('iter')
        if err is not None:
            self._record_error(iter_n, err)
        png = record.get('image_bytes')
        if png:
            self._w.set_live_view(png)

    def _on_message(self, record):
        level = record.get('level', 'info')
        text = record.get('text', '')
        self.log(text, level=level)

    def _record_error(self, iteration, error):
        if self._error_history.append(iteration, error):
            try:
                self._w.set_error_plot(render_error_plot(
                    self._error_history.points_for_plot()))
            except Exception as e:
                self.log(format_error_summary(e, prefix='_record_error'),
                         level='debug')

    def _warn_if_no_progress(self):
        if not self._progress_seen and self.is_running:
            self.log(_UI['progress_warning']['no_progress_lines'])

    def _stuck_watchdog_loop(self):
        """Warn once if no progress line arrives within STUCK_WARN_SECONDS
        while the subprocess is alive. It never kills the subprocess
        automatically. The user makes that decision with the Stop button.
        """
        while not self._stuck_stop.is_set():
            self._stuck_stop.wait(self.STUCK_POLL_SECONDS)
            if self._stuck_stop.is_set():
                return
            if self._stuck_warned:
                continue
            if not self.is_running:
                return
            last = self._last_iter_wall
            if last is None:
                continue
            elapsed = time.monotonic() - last
            if elapsed >= self.STUCK_WARN_SECONDS:
                self.log(
                    f'No progress line in {int(elapsed)}s '
                    f'(last iter {self._last_iter_value}/'
                    f'{self._expected_total_iters}). Subprocess may be '
                    f'stuck; click Stop to abort if needed.',
                    level='warning',
                )
                self._stuck_warned = True

    def _iter_rate_progress_loop(self):
        # Tick ~5 Hz. Each tick, if we have a rate estimate and it's
        # been a while since the last real iter event, advance the bar
        # by rate * elapsed. Capped one short of max so the next real
        # fire can still snap to the truth without a regression.
        while not self._iter_rate_stop.is_set():
            self._iter_rate_stop.wait(self.ITER_RATE_TICK_SECONDS)
            if self._iter_rate_stop.is_set():
                break
            rate = self._iter_rate
            if rate is None or rate <= 0:
                continue
            last_wall = self._last_iter_wall
            if last_wall is None:
                continue
            elapsed = time.monotonic() - last_wall
            est_iter = self._last_iter_value + int(rate * elapsed)
            cap = max(self.progress_bar.value, self.progress_bar.max - 1)
            new_value = min(est_iter, cap)
            if new_value > self.progress_bar.value:
                try:
                    self.progress_bar.value = new_value
                except Exception as e:
                    self.log(format_error_summary(e, prefix='_iter_rate_progress_loop'),
                             level='debug')
