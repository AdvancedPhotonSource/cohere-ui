"""Final-snapshot rendering for :class:`RecMonitor`.

The four ``_render_final_*`` helpers handle the post-run rendering paths.
They import the matplotlib and PyVista backend modules, touch OpenGL, and
handle their own fallback chain, kept separate from the subprocess
lifecycle code (threads, signals, the multiprocessing.Queue listener).

All entry points take a ``monitor`` instance and reach back into it
for the widgets they update (``live_view``, ``live_view_caption``,
``error_plot``) and for the thread-marshalling IOLoop. None mutate
monitor's lifecycle state.
"""

from cohere_ui.jupyter_gui.rec_subprocess.log_view import render_error_plot
from cohere_ui.jupyter_gui.text import load_text
from cohere_ui.jupyter_gui.utils.error_format import format_error_summary

_UI = load_text('ui_strings')


def show_final_snapshot(monitor, image, support, errors, backend_cfg):
    """Render the converged image to the snapshot panel.

    The last live preview lags a few iterations behind the converged
    state, so this re-reads what cohere wrote to disk and uses the
    renderer the user picked in the Live feature for consistency with
    the in-flight previews.

    PyVista mode marshals the actual render onto the kernel's main
    thread because PyVista's off-screen Plotter creates an OpenGL
    context (CGL on macOS) that must originate on the Cocoa main
    thread. The watchdog thread that drives ``_on_rec_finished`` is
    not that thread, so calling pv.Plotter directly here crashes the
    kernel.
    """
    if image is None:
        return
    cfg = backend_cfg or monitor._backend_cfg or {}

    if cfg.get('kind') == 'pyvista' and monitor._main_loop is not None:
        monitor._main_loop.add_callback(
            _show_final_snapshot_impl,
            monitor, image, support, errors, cfg,
        )
        return
    _show_final_snapshot_impl(monitor, image, support, errors, cfg)


def _show_final_snapshot_impl(monitor, image, support, errors, cfg):
    kind = cfg.get('kind')
    errs_list = list(errors) if errors is not None else []
    final_iter = len(errs_list)
    title = f'Final result (iter {final_iter})'

    if kind == 'pyvista':
        rendered = _render_final_pyvista(monitor, image, support, errs_list, cfg, title)
    else:
        rendered = _render_final_matplotlib(monitor, image, support, errs_list, cfg, title)

    if not rendered:
        return

    monitor.live_view_caption.value = (
        _UI['snapshot_panel']['final_caption'].format(iter=final_iter)
    )
    monitor.log('Final result rendered.')

    # Push the closing error value if not already in the history.
    if errs_list and monitor._error_history.last_iter() not in (final_iter, None):
        final_err = float(errs_list[-1])
        if monitor._error_history.append(final_iter, final_err):
            try:
                monitor.set_error_plot(render_error_plot(
                    monitor._error_history.points_for_plot()))
            except Exception as e:
                monitor.log(format_error_summary(
                    e, prefix='show_final_snapshot'), level='debug')


def _render_final_matplotlib(monitor, image, support, errs_list, cfg, title) -> bool:
    """Render the final snapshot via JupyterMatplotlibBackend in the
    mode the user picked in the Live feature."""
    try:
        from cohere_ui.jupyter_gui._backends import JupyterMatplotlibBackend
    except Exception as e:
        monitor.log(f'final snapshot: backend import failed ({e})')
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
        monitor.log(f'final snapshot (matplotlib) render failed: {e}')
        return False
    for rec in captured:
        if rec.get('kind') == 'snapshot' and rec.get('image_bytes'):
            monitor.set_live_view(rec['image_bytes'])
            return True
    return False


def _render_final_pyvista(monitor, image, support, errs_list, cfg, title) -> bool:
    """Final-snapshot render in PyVista mode.

    Uses the same PyVistaBackend the live preview used in the
    subprocess so the final still matches the in-flight previews in
    style (isosurface, white background, etc.). Must run on the
    kernel's main thread because pv.Plotter's off-screen renderer
    creates an OpenGL context that on macOS is bound to Cocoa; the
    watchdog thread that triggers show_final_snapshot can't safely do
    that, which is why show_final_snapshot marshals here via the
    IOLoop captured at construction.

    Falls back to a matplotlib 3D mosaic when PyVista is unavailable
    or the render itself raises (e.g. no GPU, missing OSMesa on
    headless Linux). The fallback path keeps a still showing in the
    panel rather than leaving it blank.
    """
    try:
        from cohere_ui.jupyter_gui._backends import PyVistaBackend
    except Exception as e:
        monitor.log(
            f'final snapshot: PyVista backend import failed ({e}); '
            f'falling back to matplotlib 3D mosaic.',
            level='warning',
        )
        return _render_final_pyvista_fallback(
            monitor, image, support, errs_list, cfg, title,
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
        monitor.log(
            f'final snapshot (PyVista) render failed: {e}; '
            f'falling back to matplotlib 3D mosaic.',
            level='warning',
        )
        return _render_final_pyvista_fallback(
            monitor, image, support, errs_list, cfg, title,
        )

    for rec in captured:
        if rec.get('kind') == 'snapshot' and rec.get('image_bytes'):
            monitor.set_live_view(rec['image_bytes'])
            return True
    # update_singlepeak returned without emitting a snapshot record
    # (e.g. it logged a warning instead). Fall back so the panel isn't
    # left empty.
    return _render_final_pyvista_fallback(
        monitor, image, support, errs_list, cfg, title,
    )


def _render_final_pyvista_fallback(monitor, image, support, errs_list, cfg, title) -> bool:
    """Matplotlib 3D mosaic as a last-resort final snapshot.

    Used when PyVista isn't importable or the off-screen plotter
    raises (no GPU, missing OSMesa, etc.). The mosaic mode + the
    user's selected slice method roughly mirrors the PyVista
    isosurface's intent (a 3D-aware overview rather than a single 2D
    slice), so the panel still conveys converged 3D structure.
    """
    mp_cfg = {
        'kind': 'matplotlib',
        'mode': 'strided_3d',
        'stride': int(cfg.get('stride', 4)),
        'slice_method': cfg.get('slice_method', 'center_of_mass'),
        'phase_cmap': cfg.get('phase_cmap', 'twilight'),
        'apply_support_mask': bool(cfg.get('apply_support_mask', True)),
    }
    return _render_final_matplotlib(
        monitor, image, support, errs_list, mp_cfg, title,
    )
