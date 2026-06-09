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
import sys
import threading
import time
import traceback

import ipywidgets as widgets

from cohere_ui.jupyter_gui.utils.error_format import format_error_summary
from cohere_ui.jupyter_gui.text import load_text
from cohere_ui.jupyter_gui.rec_subprocess.progress import ErrorHistory, parse_progress_line
from cohere_ui.jupyter_gui.rec_subprocess.log_view import render_error_plot, render_log_html
from cohere_ui.jupyter_gui.rec_subprocess.runner import run_reconstruction

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
    ITER_RATE_TICK_SECONDS = 0.2
    LOG_MAX_LINES = 1000
    # Graceful-shutdown window (seconds) before force-kill.
    STOP_GRACE_SECONDS = 5.0
    # Stuck-process watchdog: warn once if no progress line arrives
    # within this window while the subprocess is alive. Never auto-kills
    # - the user owns that decision.
    STUCK_WARN_SECONDS = 300.0
    STUCK_POLL_SECONDS = 30.0

    # Small / medium / large in-tab image sizes selected from the
    # toolbar dropdown. Defaults keep the rec tab compact.
    _IMAGE_SIZE_OPTIONS = {
        # 'M' fits two images side-by-side with the 12 px container gap.
        # The calc() subtracts half the gap (6 px) plus a 2 px safety
        # margin: ipywidgets' `widgets.Image` has a `box-sizing: content-box`
        # default whose 1 px border isn't counted in max-width, so a bare
        # `50% - 6px` makes each rendered child 1 px too wide and the
        # second image wraps. The 2 px slack also covers sub-pixel
        # rounding in the flex container. `flex='1 1 0'` lets each image
        # actually grow to that half; without it the <img> stays at its
        # intrinsic PNG width.
        'S': {'min_height': '180px', 'max_width': '420px', 'flex': '0 1 auto'},
        'M': {'min_height': '260px', 'max_width': 'calc(50% - 8px)', 'flex': '1 1 0'},
        'L': {'min_height': '360px', 'max_width': '900px', 'flex': '0 1 auto'},
    }

    def __init__(self):
        self.progress_label = widgets.HTML(value=_UI['status']['idle'])
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
            layout=widgets.Layout(border='1px solid #ddd', min_height='180px',
                                  max_width='420px'),
        )
        # No top margin: the live_view + error_plot are siblings inside the
        # flex-wrap container `self._images_row`. Spacing between them comes
        # from the container's CSS gap (.jup-gui-rec-images), which handles
        # both side-by-side and wrapped-to-new-line layouts uniformly.
        self.error_plot = widgets.Image(
            format='png',
            layout=widgets.Layout(border='1px solid #ddd', min_height='160px',
                                  max_width='420px'),
        )
        # Pair the two image panels in a row that wraps to the next line
        # when the viewport is narrower than two image columns. Both child
        # images keep their own max_width so wide layouts use the space
        # efficiently and narrow layouts simply stack them.
        self._images_row = widgets.HBox(
            [self.live_view, self.error_plot],
            layout=widgets.Layout(
                width='100%',
                flex_flow='row wrap',
                align_items='flex-start',
                margin='4px 0 0 0',
            ),
        )
        self._images_row.add_class('jup-gui-rec-images')
        # Shared size selector above the live view and error plot.
        self.image_toolbar = self._make_shared_image_toolbar()

        self.log_widget = widgets.HTML(
            value=render_log_html([]),
            layout=widgets.Layout(border='1px solid #ccc', height='150px',
                                  margin='4px 0 0 0', overflow='hidden'),
        )
        # Reconstruction log mirrors LogPanel's show_log + show_debug +
        # copy controls. Plain HTML copy button (not ipywidgets) so the
        # browser keeps clipboard permission tied to the user's click.
        from cohere_ui.jupyter_gui.widgets import make_copy_to_clipboard_html
        self._make_copy_html = make_copy_to_clipboard_html

        self.show_log_checkbox = widgets.Checkbox(
            value=False, description=_UI['action_buttons']['log_show'], indent=False,
            tooltip=_UI['tooltips']['log_show_rec'],
            layout=widgets.Layout(margin='0 12px 0 0', width='auto'),
        )
        self.show_log_checkbox.observe(self._on_show_log_toggle, names='value')
        self.show_debug = False
        self.show_debug_checkbox = widgets.Checkbox(
            value=False, description=_UI['action_buttons']['log_debug'], indent=False,
            tooltip=_UI['tooltips']['log_debug'],
            layout=widgets.Layout(margin='0 12px 0 0', width='auto'),
        )
        self.show_debug_checkbox.observe(self._on_debug_toggle, names='value')
        # Icon-only FontAwesome fa-copy, right-aligned via margin:auto.
        snippet, self._copy_log_uid = self._make_copy_html(
            '', icon='copy', label='',
            tooltip=_UI['tooltips']['log_copy_rec'],
        )
        self.copy_log_button = widgets.HTML(
            value=snippet,
            layout=widgets.Layout(width='auto', margin='0 0 0 auto'),
        )
        self.log_toolbar = widgets.HBox(
            [
                self.show_log_checkbox,
                self.show_debug_checkbox,
                self.copy_log_button,
            ],
            layout=widgets.Layout(
                # overflow=visible avoids a stray scrollbar when toolbar
                # widths approach 100%.
                width='100%', align_items='center', margin='4px 0 0 0',
                overflow='visible',
            ),
        )
        # Log hidden by default; user toggles to reveal.
        self.log_widget.layout.display = 'none'

        self._log_lines = []
        self._error_history = ErrorHistory()

        # Capture the kernel's IOLoop now (we're on the main thread during
        # CoherenceGUI construction). The watchdog thread later uses this
        # handle to marshal main-thread-only renders (PyVista's OpenGL
        # context on macOS / Cocoa) back onto the kernel's main thread.
        # In non-Jupyter contexts (e.g. unit-test instantiation) no IOLoop
        # is current; we fall back to inline rendering, which is fine
        # since those contexts don't have Cocoa to crash into.
        try:
            import tornado.ioloop
            self._main_loop = tornado.ioloop.IOLoop.current(instance=False)
        except Exception:
            self._main_loop = None

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
        # Stashed at start() so show_final_snapshot can honor the user's
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
        return widgets.VBox([
            self.progress_label,
            self.progress_bar,
            self.live_view_caption,
            self.image_toolbar,
            self._images_row,
            self.log_toolbar,
            self.log_widget,
        ])

    def _make_shared_image_toolbar(self) -> widgets.HBox:
        """Shared image size selector above the live_view / error_plot stack."""
        self.image_size_dropdown = widgets.Dropdown(
            options=list(self._IMAGE_SIZE_OPTIONS),
            value='M',
            description='Image size:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='160px'),
        )
        self.image_size_dropdown.observe(self._apply_image_size, 'value')
        # Apply the initial size so the images render with the M layout
        # (half-width + flex-grow) on first display, not S's defaults
        # that the bare widgets.Image layouts inherit.
        self._apply_image_size({'new': self.image_size_dropdown.value})

        return widgets.HBox(
            [self.image_size_dropdown],
            layout=widgets.Layout(
                align_items='center', margin='4px 0 4px 0',
            ),
        )

    def _apply_image_size(self, change) -> None:
        opts = self._IMAGE_SIZE_OPTIONS[change['new']]
        for img in (self.live_view, self.error_plot):
            img.layout.min_height = opts['min_height']
            img.layout.max_width = opts['max_width']
            img.layout.flex = opts.get('flex', '')


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
        # macOS: Jupyter may launch via framework Python even inside a venv.
        # Force the child to the venv's python so it sees the kernel's
        # packages (otherwise the subprocess hits ModuleNotFoundError).
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

        # Iter-rate ticker advances the bar between real iter events at the observed rate.
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
        """Graceful stop: SIGTERM (terminate request), wait
        STOP_GRACE_SECONDS, then SIGKILL (force-kill).

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
            # Windows or already-dead: fall back to multiprocessing terminate
            # (SIGTERM on Unix, CTRL_BREAK_EVENT on Windows).
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

    def log(self, msg, level: str = 'info'):
        """Append a line to the log widget at the given level."""
        self._log_lines.append((level, str(msg)))
        if len(self._log_lines) > self.LOG_MAX_LINES:
            self._log_lines = self._log_lines[-self.LOG_MAX_LINES:]
        self._refresh_log()
        if level == 'error' and not self.show_log_checkbox.value:
            self.show_log_checkbox.value = True

    def clear_log(self):
        """Clear the log widget."""
        self._log_lines = []
        self._refresh_log()

    def set_show_debug(self, value: bool):
        value = bool(value)
        if self.show_debug == value:
            return
        self.show_debug = value
        if self.show_debug_checkbox.value != value:
            self.show_debug_checkbox.value = value
        self._refresh_log()

    def _on_debug_toggle(self, change):
        self.show_debug = bool(change['new'])
        self._refresh_log()

    def _on_show_log_toggle(self, change):
        """Reveal or hide the log panel without losing its contents."""
        self.log_widget.layout.display = '' if change['new'] else 'none'

    def _log_copy_text(self) -> str:
        """Plain-text log dump (one line per entry, respecting show_debug).

        Uses the same level prefixes as the rendered view so clipboard
        output matches what's on screen.
        """
        prefix = {
            'info': '', 'success': '[OK] ',
            'warning': '[WARN] ', 'error': '[ERROR] ',
            'debug': '[DEBUG] ',
        }
        out = []
        for level, msg in self._log_lines:
            if level == 'debug' and not self.show_debug:
                continue
            out.append(f'{prefix.get(level, "")}{msg}')
        return '\n'.join(out)

    def _refresh_copy_log_button(self) -> None:
        """Re-render the copy button HTML with the current log text."""
        snippet, _ = self._make_copy_html(
            self._log_copy_text(), icon='copy', label='',
            tooltip=_UI['tooltips']['log_copy_rec'],
        )
        self.copy_log_button.value = snippet

    def _refresh_log(self):
        try:
            self.log_widget.value = render_log_html(
                self._log_lines, show_debug=self.show_debug,
            )
        except Exception as e:
            sys.stderr.write(
                f"RecMonitor._refresh_log failed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )
        # Keep the copy button's embedded text in sync with the visible log.
        try:
            self._refresh_copy_log_button()
        except Exception as e:
            sys.stderr.write(
                f"RecMonitor._refresh_copy_log_button failed: "
                f"{type(e).__name__}: {e}\n"
            )

    def show_final_snapshot(self, image, support, errors, backend_cfg):
        """Render the converged image to the snapshot panel.

        The last live preview lags a few iterations behind the converged
        state, so this re-reads what cohere wrote to disk and uses the
        renderer the user picked in the Live feature for consistency
        with the in-flight previews.

        PyVista mode marshals the actual render to the kernel's main
        thread because PyVista's off-screen Plotter creates an OpenGL
        context (CGL on macOS) that must originate on the Cocoa main
        thread. The watchdog thread that drives _on_rec_finished isn't
        that thread, so calling pv.Plotter directly here crashes the
        kernel.
        """
        if image is None:
            return
        cfg = backend_cfg or self._backend_cfg or {}

        if cfg.get('kind') == 'pyvista' and self._main_loop is not None:
            # Reschedule the entire impl onto the kernel IOLoop. The
            # other branches (matplotlib Agg) are thread-safe and stay
            # on the watchdog thread to avoid unnecessary latency.
            self._main_loop.add_callback(
                self._show_final_snapshot_impl,
                image, support, errors, cfg,
            )
            return
        self._show_final_snapshot_impl(image, support, errors, cfg)

    def _show_final_snapshot_impl(self, image, support, errors, cfg):
        kind = cfg.get('kind')
        errs_list = list(errors) if errors is not None else []
        final_iter = len(errs_list)
        title = f'Final result (iter {final_iter})'

        if kind == 'pyvista':
            rendered = self._render_final_pyvista(image, support, errs_list, cfg, title)
        else:
            rendered = self._render_final_matplotlib(image, support, errs_list, cfg, title)

        if not rendered:
            return

        self.live_view_caption.value = (
            _UI['snapshot_panel']['final_caption'].format(iter=final_iter)
        )
        self.log('Final result rendered.')

        # Push the closing error value if not already in the history.
        if errs_list and self._error_history.last_iter() not in (final_iter, None):
            final_err = float(errs_list[-1])
            if self._error_history.append(final_iter, final_err):
                try:
                    self.error_plot.value = render_error_plot(
                        self._error_history.points_for_plot())
                except Exception as e:
                    self.log(format_error_summary(
                        e, prefix='show_final_snapshot'), level='debug')

    def _render_final_matplotlib(self, image, support, errs_list, cfg, title) -> bool:
        """Render the final snapshot via JupyterMatplotlibBackend in the
        mode the user picked in the Live feature."""
        try:
            from cohere_ui.jupyter_gui._backends import JupyterMatplotlibBackend
        except Exception as e:
            self.log(f'final snapshot: backend import failed ({e})')
            return False
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
        try:
            backend.update_singlepeak(image, errs_list, support, title)
        except Exception as e:
            self.log(f'final snapshot (matplotlib) render failed: {e}')
            return False
        for rec in captured:
            if rec.get('kind') == 'snapshot' and rec.get('image_bytes'):
                self.live_view.value = rec['image_bytes']
                return True
        return False

    def _render_final_pyvista(self, image, support, errs_list, cfg, title) -> bool:
        """Final-snapshot render in PyVista mode.

        Uses the same PyVistaBackend the live preview used in the
        subprocess so the final still matches the in-flight previews in
        style (isosurface, white background, etc.). Must run on the
        kernel's main thread because pv.Plotter's off-screen renderer
        creates an OpenGL context that on macOS is bound to Cocoa; the
        watchdog thread that triggers show_final_snapshot can't safely
        do that, which is why show_final_snapshot marshals here via the
        IOLoop captured at construction.

        Falls back to a matplotlib 3D mosaic when PyVista is unavailable
        or the render itself raises (e.g. no GPU, missing OSMesa on
        headless Linux). The fallback path keeps a still showing in the
        panel rather than leaving it blank.
        """
        try:
            from cohere_ui.jupyter_gui._backends import PyVistaBackend
        except Exception as e:
            self.log(
                f'final snapshot: PyVista backend import failed ({e}); '
                f'falling back to matplotlib 3D mosaic.',
                level='warning',
            )
            return self._render_final_pyvista_fallback(
                image, support, errs_list, cfg, title,
            )

        captured = []

        class _Box:
            def put(self, rec, **_kw):
                captured.append(rec)

        try:
            backend = PyVistaBackend(
                _Box(),
                stride=int(cfg.get('stride', 4)),
                iso_level=float(cfg.get('iso_level', 0.3)),
            )
            backend.update_singlepeak(image, errs_list, support, title)
        except Exception as e:
            self.log(
                f'final snapshot (PyVista) render failed: {e}; '
                f'falling back to matplotlib 3D mosaic.',
                level='warning',
            )
            return self._render_final_pyvista_fallback(
                image, support, errs_list, cfg, title,
            )

        for rec in captured:
            if rec.get('kind') == 'snapshot' and rec.get('image_bytes'):
                self.live_view.value = rec['image_bytes']
                return True
        # update_singlepeak returned without emitting a snapshot record
        # (e.g. it logged a warning instead). Fall back so the panel
        # isn't left empty.
        return self._render_final_pyvista_fallback(
            image, support, errs_list, cfg, title,
        )

    def _render_final_pyvista_fallback(self, image, support, errs_list, cfg, title) -> bool:
        """Matplotlib 3D mosaic as a last-resort final snapshot.

        Used when PyVista isn't importable or the off-screen plotter
        raises (no GPU, missing OSMesa, etc.). The mosaic mode + the
        user's selected slice method roughly mirrors the PyVista
        isosurface's intent (a 3D-aware overview rather than a single
        2D slice), so the panel still conveys converged 3D structure.
        """
        mp_cfg = {
            'kind': 'matplotlib',
            'mode': 'strided_3d',
            'stride': int(cfg.get('stride', 4)),
            'slice_method': cfg.get('slice_method', 'center_of_mass'),
            'phase_cmap': cfg.get('phase_cmap', 'twilight'),
            'apply_support_mask': bool(cfg.get('apply_support_mask', True)),
        }
        return self._render_final_matplotlib(
            image, support, errs_list, mp_cfg, title,
        )

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
        self._iter_rate_stop.set()
        if self._iter_rate_ticker is not None:
            self._iter_rate_ticker.join(timeout=1)
            self._iter_rate_ticker = None
        self._stuck_stop.set()
        if self._stuck_thread is not None:
            self._stuck_thread.join(timeout=1)
            self._stuck_thread = None
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
        self.log(text, level=level)

    def _record_error(self, iteration, error):
        if self._error_history.append(iteration, error):
            try:
                self.error_plot.value = render_error_plot(
                    self._error_history.points_for_plot())
            except Exception as e:
                self.log(format_error_summary(e, prefix='_record_error'),
                         level='debug')

    def _warn_if_no_progress(self):
        if not self._progress_seen and self.is_running:
            self.log(_UI['progress_warning']['no_progress_lines'])

    def _stuck_watchdog_loop(self):
        """Warn once if no progress line arrives within STUCK_WARN_SECONDS
        while the subprocess is alive. Never auto-kills - the user owns
        that decision via the Stop button.
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
        # Tick ~5 Hz. Each tick, if we have a rate estimate and it's been a
        # while since the last real iter event, advance the bar by
        # rate * elapsed. Capped one short of max so the next real fire can
        # still snap to the truth without a regression.
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
                    return
