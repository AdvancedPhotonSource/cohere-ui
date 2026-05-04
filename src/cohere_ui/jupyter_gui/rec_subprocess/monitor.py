"""RecMonitor: subprocess + listener thread + watchdog + GUI widgets.

Owns the multiprocessing.Queue, the listener daemon that drains it, the
watchdog that detects subprocess exit, and the four widgets that surface
state to the user (progress label, progress bar, log HTML, snapshot
image, error-history plot). The Reconstruction tab constructs one of
these in its ``_build_ui`` and embeds ``monitor.widgets_box`` in its
layout.
"""

import os
import signal
import threading
import time

import ipywidgets as widgets

from ..text import load_text
from .progress import ErrorHistory, parse_progress_line
from .log_view import render_error_plot, render_log_html
from .runner import run_reconstruction

_UI = load_text('ui_strings')


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
    PSEUDO_TICK_SECONDS = 0.2
    LOG_MAX_LINES = 1000

    def __init__(self):
        self.progress_label = widgets.HTML(value='<i>Idle</i>')
        self.progress_bar = widgets.IntProgress(
            value=0, min=0, max=1,
            bar_style='info',
            layout=widgets.Layout(width='100%', margin='4px 0 4px 0'),
        )
        self.progress_bar.layout.visibility = 'hidden'
        self.live_view_caption = widgets.HTML(
            value=_UI['snapshot_panel']['idle'],
        )
        self.live_view = widgets.Image(
            format='png',
            layout=widgets.Layout(border='1px solid #ddd', min_height='280px',
                                  max_width='800px'),
        )
        self.error_plot = widgets.Image(
            format='png',
            layout=widgets.Layout(border='1px solid #ddd', min_height='200px',
                                  max_width='800px', margin='8px 0 0 0'),
        )
        self.log_widget = widgets.HTML(
            value=render_log_html([]),
            layout=widgets.Layout(border='1px solid #ccc', height='150px',
                                  margin='4px 0 0 0'),
        )

        self._log_lines = []
        self._error_history = ErrorHistory()

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
        self._pseudo_ticker = None
        self._pseudo_stop = threading.Event()
        self._last_exit_code = None

        self.on_running_changed = None
        self.on_finished = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def widgets_box(self) -> widgets.VBox:
        """Return the monitor's widgets stacked in display order."""
        return widgets.VBox([
            self.progress_label,
            self.progress_bar,
            self.live_view_caption,
            self.live_view,
            self.error_plot,
            self.log_widget,
        ])

    def start(self, experiment_dir, backend_cfg, kwargs,
              total_iters: int, show_progress_bar: bool):
        """Spawn the subprocess and start the listener and watchdog threads."""
        import multiprocessing as mp

        if self.is_running:
            self.log('Reconstruction already running. Click Stop first.')
            return

        self._reset_for_run(total_iters, show_progress_bar)

        ctx = mp.get_context('spawn')
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

        # Silent-format-break detection: warn once if no progress lines have
        # been parsed within 30 s.
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

        # Pseudo-progress ticker: when the bar is visible, interpolate
        # between real iter events using the observed rate so the bar moves
        # smoothly even when progress_trigger is sparse.
        if self.progress_bar.layout.visibility == 'visible':
            self._pseudo_stop.clear()
            self._pseudo_ticker = threading.Thread(
                target=self._pseudo_progress_loop, daemon=True,
                name='RecMonitor-pseudo-progress',
            )
            self._pseudo_ticker.start()

    def stop(self):
        """Kill the running subprocess group."""
        proc = self._proc
        if proc is None or not proc.is_alive():
            return
        self.log(f'Stop clicked: killing process group of pid {proc.pid}')
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, OSError):
            # Windows or already-dead: fall back to terminating just the wrapper.
            try:
                proc.terminate()
            except Exception as e:
                self.log(f'terminate fallback failed: {e}')

    def log(self, msg):
        """Append a line to the log widget."""
        self._log_lines.append(str(msg))
        if len(self._log_lines) > self.LOG_MAX_LINES:
            self._log_lines = self._log_lines[-self.LOG_MAX_LINES:]
        try:
            self.log_widget.value = render_log_html(self._log_lines)
        except Exception:
            pass

    def clear_log(self):
        """Clear the log widget."""
        self._log_lines = []
        self.log_widget.value = render_log_html([])

    def show_final_snapshot(self, image, support, errors, backend_cfg):
        """Render the converged image to the snapshot panel.

        The last live_trigger fire usually lags a few iterations behind
        the final state; this overrides it with what cohere actually
        wrote to disk.
        """
        if image is None:
            return
        try:
            from cohere_ui.jupyter_gui._backends import JupyterMatplotlibBackend
        except Exception as e:
            self.log(f'final snapshot: backend import failed ({e})')
            return
        cfg = backend_cfg or {}
        if cfg.get('kind') != 'matplotlib':
            cfg = {'kind': 'matplotlib', 'mode': 'center_slice'}
        captured = []

        class _Box:
            def put(self, rec, **_kw):
                captured.append(rec)

        backend = JupyterMatplotlibBackend(
            _Box(),
            mode=cfg.get('mode', 'center_slice'),
            slice_axis=cfg.get('slice_axis', 2),
            slice_method=cfg.get('slice_method', 'center_of_mass'),
            stride=cfg.get('stride', 4),
            phase_cmap=cfg.get('phase_cmap', 'twilight'),
            apply_support_mask=cfg.get('apply_support_mask', True),
        )
        errs_list = list(errors) if errors is not None else []
        title = f'Final result (iter {len(errs_list)})'
        try:
            backend.update_singlepeak(image, errs_list, support, title)
        except Exception as e:
            self.log(f'final snapshot render failed: {e}')
            return
        for rec in captured:
            if rec.get('kind') == 'snapshot' and rec.get('image_bytes'):
                self.live_view.value = rec['image_bytes']
                final_iter = len(errs_list)
                self.live_view_caption.value = (
                    _UI['snapshot_panel']['final_caption'].format(iter=final_iter)
                )
                self.log('Final result rendered.')
                # Push the closing error value if not already in the history.
                if errs_list and self._error_history.last_iter() not in (final_iter, None):
                    final_err = float(errs_list[-1])
                    if self._error_history.append(final_iter, final_err):
                        self.error_plot.value = render_error_plot(
                            self._error_history.points_for_plot())
                return

    def _reset_for_run(self, total_iters: int, show_progress_bar: bool):
        self.clear_log()
        self.live_view.value = b''
        self.live_view_caption.value = _UI['snapshot_panel']['running']
        self.error_plot.value = b''
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
            # Success when the worker exited cleanly; warning when killed/errored.
            # cohere reports iter 0..N-1, so the bar's last observed value is max-1.
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
        self._pseudo_stop.set()
        if self._pseudo_ticker is not None:
            self._pseudo_ticker.join(timeout=1)
            self._pseudo_ticker = None
        # Drain remaining queue records before closing. Without this, the
        # listener can be stopped before the tail of the run's output (final
        # iter print, "iterate took", etc.) is dispatched, dropping the
        # last data point from the error plot and the closing log lines.
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
                self.log(f'on_finished callback error: {e}')

    def _run_listener(self):
        """Drain the multiprocessing Queue, dispatch each record by `kind`."""
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
        self.progress_label.value = ' &nbsp;&middot;&nbsp; '.join(parts)
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
            self.live_view.value = png

    def _on_message(self, record):
        level = record.get('level', 'info')
        text = record.get('text', '')
        self.log(f'[{level}] {text}')

    def _record_error(self, iteration, error):
        if self._error_history.append(iteration, error):
            self.error_plot.value = render_error_plot(
                self._error_history.points_for_plot())

    def _warn_if_no_progress(self):
        if not self._progress_seen and self.is_running:
            self.log(_UI['progress_warning']['no_progress_lines'])

    def _pseudo_progress_loop(self):
        # Tick ~5 Hz. Each tick, if we have a rate estimate and it's been a
        # while since the last real iter event, advance the bar by
        # rate * elapsed. Capped one short of max so the next real fire can
        # still snap to the truth without a regression.
        while not self._pseudo_stop.is_set():
            self._pseudo_stop.wait(self.PSEUDO_TICK_SECONDS)
            if self._pseudo_stop.is_set():
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
                except Exception:
                    return
