"""Two-pane TIFF stack viewer used by PrepTab and DataTab."""

import glob as _glob
import html as _html_mod
import io
import os
import subprocess
import traceback

import ipywidgets as widgets

try:
    from ipyfilechooser import FileChooser
except ImportError:
    FileChooser = None

from cohere_ui.jupyter_gui.text import load_text
from cohere_ui.jupyter_gui.utils.error_format import format_error_summary
from cohere_ui.jupyter_gui.viewers.imagej import resolve_imagej_path
from cohere_ui.jupyter_gui.widgets import text_field, button

_UI = load_text('ui_strings')


def _html_escape(text):
    return _html_mod.escape(str(text), quote=True)


class TiffViewer:
    """Two-pane TIFF stack viewer with shared scale controls and a Show/Hide toggle.

    Each pane is configured via a dict with these keys:
      - key: short id used as the pane handle (e.g. 'raw', 'prep', 'compare', 'data')
      - label: HTML header text shown above the pane
      - default_path: callable() -> Optional[str] resolving the canonical path
      - shortcuts: optional list of (button_label, resolver) tuples; when the user
        clicks the button the resolver's path is loaded into the pane
    """

    def __init__(self, panes, *, title='TIFF Viewer',
                 initial_visible=False, log_debug=None):
        assert len(panes) == 2, "TiffViewer requires exactly two panes"
        self._panes = {p['key']: p for p in panes}
        self._keys = [p['key'] for p in panes]
        self._log_debug = log_debug or (lambda msg: None)
        self._tiff_data = {k: None for k in self._keys}
        self._tiff_scale_cache = None
        self._tiff_lut_cache = {}
        self._build(title, initial_visible)

    def widget(self) -> widgets.Widget:
        return self._widget

    def set_visible(self, on: bool):
        self.show_toggle.value = bool(on)

    def _build(self, title, initial_visible):
        self.path = {}
        self.load_btn = {}
        self.imagej_btn = {}
        self.image = {}
        self.slider = {}
        self.status = {}
        self.chooser = {}
        self.dir_mode = {}
        self.vmin = {}
        self.vmax = {}
        self.shortcut_box = {}

        self.sync = widgets.Checkbox(
            value=True, description=_UI['feature_options']['sync_scroll'], indent=False,
        )
        self.log = widgets.Checkbox(
            value=False, description='Log scale', indent=False,
        )
        self.cmap = widgets.Dropdown(
            options=[
                'magma', 'viridis', 'plasma', 'inferno', 'cividis',
                'gray', 'hot', 'turbo', 'bone', 'cubehelix',
            ],
            value='magma',
            layout=widgets.Layout(width='130px'),
            description='Cmap',
        )
        self.invert = widgets.Checkbox(
            value=False, description='Invert cmap', indent=False,
        )
        self.zoom = widgets.Dropdown(
            options=[('1x (native)', 1), ('2x', 2), ('3x', 3), ('4x', 4), ('8x', 8)],
            value=1, description='Zoom',
            layout=widgets.Layout(width='130px'),
        )
        self.scale_mode = widgets.RadioButtons(
            options=[
                ('Auto (per slice)', 'auto'),
                ('Sync (shared across both panes)', 'sync'),
                ('Manual', 'manual'),
            ],
            value='auto',
            layout=widgets.Layout(width='280px'),
        )

        for key in self._keys:
            cfg = self._panes[key]
            self.path[key] = text_field(
                placeholder=cfg.get('placeholder', 'dir / file / glob'),
                width='340px',
            )
            self.load_btn[key] = button('Load', style='warning', width='70px')
            self.load_btn[key].on_click(
                lambda _b, k=key: self._safe('_tiff_load', self._load_path, k)
            )
            ij_tooltip = (
                'Open this TIFF in ImageJ for full-resolution inspection '
                '(orthoviews, line profiles, ROIs).'
            )
            self.imagej_btn[key] = button('ImageJ', style='', width='90px')
            self.imagej_btn[key].tooltip = ij_tooltip
            self.imagej_btn[key].on_click(
                lambda _b, k=key: self._safe('_tiff_open_imagej', self._open_imagej, k)
            )
            self.image[key] = widgets.Image(
                format='png',
                layout=widgets.Layout(border='1px solid #ddd'),
            )
            self.slider[key] = widgets.IntSlider(
                value=0, min=0, max=0, step=1, description='Slice',
                continuous_update=True, layout=widgets.Layout(width='95%'),
            )
            self.slider[key].observe(
                lambda c, k=key: self._safe('_tiff_on_slice', self._on_slice, k, c),
                names='value',
            )
            self.status[key] = widgets.HTML(value=_UI['status']['not_loaded'])
            self.vmin[key] = text_field(placeholder='auto', width='90px')
            self.vmax[key] = text_field(placeholder='auto', width='90px')
            self.chooser[key] = self._build_chooser(key) if FileChooser else None
            self.dir_mode[key] = widgets.Checkbox(
                value=False, description='Folder mode', indent=False,
                layout=widgets.Layout(width='130px'),
            )
            self.dir_mode[key].observe(
                lambda c, k=key: self._safe('_tiff_on_dir_mode', self._on_dir_mode, k, c),
                names='value',
            )
            # Shortcut buttons (above the path field).
            shortcuts = cfg.get('shortcuts') or []
            shortcut_buttons = []
            for label, resolver in shortcuts:
                btn = button(label, style='info', width='110px')
                btn.on_click(
                    lambda _b, k=key, r=resolver:
                        self._safe('_tiff_shortcut', self._load_resolved, k, r)
                )
                shortcut_buttons.append(btn)
            if shortcut_buttons:
                self.shortcut_box[key] = widgets.HBox(
                    [widgets.HTML('<small>Quick load:</small>'), *shortcut_buttons],
                    layout=widgets.Layout(margin='0 0 4px 0'),
                )
            else:
                # Filled in below if the other pane has shortcuts, to keep
                # image rows aligned.
                self.shortcut_box[key] = None

        # If ANY pane has shortcuts, give the others an empty same-height
        # placeholder so the image widgets in both panes line up vertically.
        any_shortcuts = any(self.shortcut_box[k] is not None for k in self._keys)
        if any_shortcuts:
            for key in self._keys:
                if self.shortcut_box[key] is None:
                    self.shortcut_box[key] = widgets.HBox(
                        [widgets.HTML('&nbsp;')],
                        layout=widgets.Layout(
                            margin='0 0 4px 0', height='28px', visibility='hidden',
                        ),
                    )

        manual_row_children = [widgets.HTML('<small>Manual:</small>')]
        for key in self._keys:
            manual_row_children.extend([
                widgets.HTML(f'<small>&nbsp;{self._panes[key]["label"]} '
                             '[vmin/vmax]</small>'),
                self.vmin[key], self.vmax[key],
            ])
        self._manual_row = widgets.HBox(manual_row_children)
        self._manual_row.layout.display = 'none'

        self.log.observe(lambda _c: self._safe('_render_all', self._render_all),
                         names='value')
        self.cmap.observe(lambda _c: self._safe('_render_all', self._render_all),
                          names='value')
        self.invert.observe(lambda _c: self._safe('_render_all', self._render_all),
                            names='value')
        self.zoom.observe(lambda _c: self._safe('_render_all', self._render_all),
                          names='value')
        self.scale_mode.observe(
            lambda c: self._safe('_on_scale_mode', self._on_scale_mode, c),
            names='value',
        )
        for key in self._keys:
            for w in (self.vmin[key], self.vmax[key]):
                w.observe(lambda _c: self._safe('_render_all', self._render_all),
                          names='value')

        panes = []
        for key in self._keys:
            cfg = self._panes[key]
            browse_row = (
                widgets.HBox([self.chooser[key], self.dir_mode[key]])
                if self.chooser[key] is not None
                else widgets.HTML('<small><i>install ipyfilechooser to browse</i></small>')
            )
            pane_children = [
                widgets.HTML(f'<b>{cfg["label"]}</b>'),
            ]
            if self.shortcut_box[key] is not None:
                pane_children.append(self.shortcut_box[key])
            pane_children.extend([
                widgets.HBox([self.path[key], self.load_btn[key], self.imagej_btn[key]]),
                browse_row,
                widgets.Box([self.image[key]],
                            layout=widgets.Layout(height='560px', overflow='auto',
                                                  align_items='center',
                                                  justify_content='center')),
                self.slider[key],
                self.status[key],
            ])
            # Left pane has right padding; right pane has left padding.
            is_left = (key == self._keys[0])
            padding = '0 6px 0 0' if is_left else '0 0 0 6px'
            panes.append(widgets.VBox(
                pane_children,
                layout=widgets.Layout(width='50%', padding=padding),
            ))

        self._body = widgets.VBox([
            widgets.HBox([self.sync, self.log,
                          self.cmap, self.invert, self.zoom]),
            widgets.HBox([widgets.HTML('<small>Intensity scale:</small>'),
                          self.scale_mode]),
            self._manual_row,
            widgets.HBox(panes, layout=widgets.Layout(width='100%')),
        ])
        if not initial_visible:
            self._body.layout.display = 'none'

        self.show_toggle = widgets.Checkbox(
            value=initial_visible, description='Show TIFF viewer', indent=False,
        )
        self.show_toggle.observe(self._on_show_toggle, names='value')

        self._widget = widgets.VBox([
            widgets.HBox([
                widgets.HTML(f'<h4 style="margin:6px 0;">{title}</h4>'),
                self.show_toggle,
            ], layout=widgets.Layout(align_items='center')),
            self._body,
        ])

    def _on_show_toggle(self, change):
        self._body.layout.display = '' if bool(change.get('new')) else 'none'

    def _safe(self, prefix, fn, *args, **kwargs):
        """Run fn, catching exceptions and logging detail at debug level."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            self._log_debug(format_error_summary(e, prefix=prefix))
            self._log_debug(traceback.format_exc())


    def _load_path(self, key):
        """Resolve the path text or default-path callback, then load."""
        path = self.path[key].value.strip() or self._default_path(key) or ''
        self._load_from(key, path)

    def _load_resolved(self, key, resolver):
        """Apply a shortcut resolver to a pane: set its path field, then load."""
        path = resolver() if callable(resolver) else (resolver or '')
        path = (path or '').strip()
        if not path:
            self.status[key].value = '<span style="color:#a00;">shortcut returned no path</span>'
            return
        self.path[key].value = path
        self._load_from(key, path)

    def _default_path(self, key):
        fn = self._panes[key].get('default_path')
        if fn is None:
            return None
        try:
            return fn()
        except Exception as e:
            self._log_debug(format_error_summary(e, prefix='_default_path'))
            return None


    def _build_chooser(self, key):
        assert FileChooser is not None
        fc = FileChooser(
            path=os.getcwd(),
            select_default=False,
            show_only_dirs=False,
            title='',
        )
        fc.register_callback(
            lambda c, k=key: self._safe('_tiff_on_chooser', self._on_chooser, k, c)
        )
        return fc

    def _on_chooser(self, key, chooser):
        sel = chooser.selected_path if chooser.show_only_dirs else chooser.selected
        if not sel:
            return
        self.path[key].value = sel
        self._load_from(key, sel)

    def _on_dir_mode(self, key, change):
        chooser = self.chooser[key]
        if chooser is None:
            return
        chooser.show_only_dirs = bool(change.get('new'))
        try:
            chooser.refresh()
        except Exception as e:
            self._log_debug(format_error_summary(e, prefix='_on_dir_mode'))


    def _on_scale_mode(self, change):
        self._manual_row.layout.display = (
            '' if change.get('new') == 'manual' else 'none'
        )
        self._render_all()


    def _stack_frames(self, files, status, tifffile, np):
        first = tifffile.imread(files[0])
        if first.ndim != 2:
            status.value = (f'<span style="color:#a00;">'
                            f'expected 2D frames, got ndim={first.ndim}</span>')
            return None
        arr = np.empty((len(files),) + first.shape, dtype=first.dtype)
        arr[0] = first
        for i, f in enumerate(files[1:], start=1):
            arr[i] = tifffile.imread(f)
        return arr

    def _load_from(self, key, path):
        status = self.status[key]
        slider = self.slider[key]
        if not path:
            status.value = '<span style="color:#a00;">no path provided</span>'
            return
        is_glob = '*' in path or '?' in path or '[' in path
        if not is_glob and not (os.path.isfile(path) or os.path.isdir(path)):
            status.value = f'<span style="color:#a00;">not found: {path}</span>'
            return
        try:
            import numpy as np
            import tifffile
            if is_glob:
                files = sorted(f for f in _glob.glob(path, recursive=True)
                               if os.path.isfile(f))
                if not files:
                    status.value = f'<span style="color:#a00;">no files match: {path}</span>'
                    return
                arr = self._stack_frames(files, status, tifffile, np)
                if arr is None:
                    return
                display_name = f'{path}  ({len(files)} files)'
            elif os.path.isdir(path):
                files = sorted(set(
                    _glob.glob(os.path.join(path, '**', '*.tif'),  recursive=True) +
                    _glob.glob(os.path.join(path, '**', '*.tiff'), recursive=True)
                ))
                if not files:
                    status.value = (f'<span style="color:#a00;">'
                                    f'no *.tif / *.tiff under {path}</span>')
                    return
                arr = self._stack_frames(files, status, tifffile, np)
                if arr is None:
                    return
                display_name = f'{os.path.basename(path) or path} ({len(files)} frames)'
            else:
                arr = tifffile.imread(path)
                if arr.ndim == 2:
                    arr = arr[np.newaxis, :, :]
                if arr.ndim != 3:
                    status.value = f'<span style="color:#a00;">unexpected ndim={arr.ndim} (need 2 or 3)</span>'
                    return
                display_name = path.rsplit('/', 1)[-1]
        except Exception as e:
            status.value = f'<span style="color:#a00;">load failed: {e}</span>'
            self._log_debug(format_error_summary(e, prefix='_load_from'))
            self._log_debug(traceback.format_exc())
            return
        self._tiff_data[key] = arr
        self._tiff_scale_cache = None
        self.path[key].value = path
        status.value = (
            f'<small style="color:#444;">{display_name} '
            f'shape={arr.shape} dtype={arr.dtype}</small>'
        )
        slider.max = max(0, arr.shape[0] - 1)
        slider.value = arr.shape[0] // 2
        self._render(key)


    def _on_slice(self, key, change):
        if self._tiff_data.get(key) is None:
            return
        self._render(key)
        if self.sync.value:
            other = next(k for k in self._keys if k != key)
            other_arr = self._tiff_data.get(other)
            if other_arr is None:
                return
            target = min(int(change.get('new', 0)), other_arr.shape[0] - 1)
            if self.slider[other].value != target:
                self.slider[other].value = target  # observer fires _render(other)

    def _render_all(self):
        self._tiff_scale_cache = None
        for key in self._keys:
            if self._tiff_data.get(key) is not None:
                self._render(key)

    def _compute_scale(self, key, scaled_slice):
        """vmin/vmax for the requested pane under the current scale mode."""
        import numpy as np
        mode = self.scale_mode.value
        if mode == 'manual':
            vmin_w = self.vmin[key]
            vmax_w = self.vmax[key]
            try:
                vmin = float(vmin_w.value) if vmin_w.value.strip() else float(scaled_slice.min())
            except ValueError:
                vmin = float(scaled_slice.min())
            try:
                vmax = float(vmax_w.value) if vmax_w.value.strip() else float(scaled_slice.max())
            except ValueError:
                vmax = float(scaled_slice.max())
            return vmin, vmax, 'manual'
        if mode == 'sync':
            cache = self._tiff_scale_cache
            if cache is None:
                bounds = []
                for k in self._keys:
                    arr = self._tiff_data.get(k)
                    if arr is None:
                        continue
                    a = arr.astype(np.float32, copy=False)
                    if self.log.value:
                        a = np.log10(np.maximum(a, 0) + 1.0)
                    bounds.append((float(a.min()), float(a.max())))
                if bounds:
                    self._tiff_scale_cache = (
                        min(b[0] for b in bounds),
                        max(b[1] for b in bounds),
                    )
                else:
                    self._tiff_scale_cache = (
                        float(scaled_slice.min()), float(scaled_slice.max()),
                    )
            vmin, vmax = self._tiff_scale_cache
            return vmin, vmax, 'sync'
        return float(scaled_slice.min()), float(scaled_slice.max()), 'auto'

    def _render(self, key):
        import numpy as np
        from PIL import Image
        import matplotlib

        arr = self._tiff_data[key]
        slider = self.slider[key]
        status = self.status[key]
        idx = min(slider.value, arr.shape[0] - 1)
        img_widget = self.image[key]
        path_widget = self.path[key]

        slice_2d = np.asarray(arr[idx])
        if self.log.value:
            scaled = np.log10(np.maximum(slice_2d, 0).astype(np.float32) + 1.0)
            label = 'log10(I+1)'
        else:
            scaled = slice_2d.astype(np.float32, copy=False)
            label = 'intensity'

        mn, mx, scale_label = self._compute_scale(key, scaled)
        if mx > mn:
            norm = np.clip((scaled - mn) / (mx - mn), 0.0, 1.0)
            norm = (norm * 255.0).astype(np.uint8)
        else:
            norm = np.zeros_like(scaled, dtype=np.uint8)

        cmap_name = self.cmap.value
        if self.invert.value:
            cmap_name = cmap_name + '_r'
        lut = self._tiff_lut_cache.get(cmap_name)
        if lut is None:
            cmap_obj = matplotlib.colormaps.get_cmap(cmap_name)
            lut = (cmap_obj(np.arange(256))[:, :3] * 255).astype(np.uint8)
            self._tiff_lut_cache[cmap_name] = lut
        rgb = lut[norm]
        pil_img = Image.fromarray(rgb, mode='RGB')
        zoom = max(1, int(self.zoom.value))
        if zoom > 1:
            h, w = rgb.shape[:2]
            pil_img = pil_img.resize((w * zoom, h * zoom), Image.NEAREST)
        png_buf = io.BytesIO()
        pil_img.save(
            png_buf, format='PNG', optimize=False, compress_level=1,
        )
        img_widget.value = png_buf.getvalue()

        raw_min = float(slice_2d.min())
        raw_max = float(slice_2d.max())
        raw_mean = float(slice_2d.mean())
        fname = os.path.basename(path_widget.value) or '(unnamed)'
        status.value = (
            f'<small style="color:#444;">'
            f'<b>{fname}</b> &nbsp; slice <code>{idx}/{arr.shape[0] - 1}</code> &nbsp; '
            f'shape <code>{arr.shape[1]}x{arr.shape[2]}</code> &nbsp; '
            f'dtype <code>{arr.dtype}</code> &nbsp; '
            f'min/max/mean <code>{raw_min:g}/{raw_max:g}/{raw_mean:.4g}</code> &nbsp; '
            f'display <i>{label}</i> &nbsp; '
            f'scale <i>{scale_label}</i> <code>[{mn:.4g}, {mx:.4g}]</code></small>'
        )


    def _open_imagej(self, key):
        path_widget = self.path[key]
        status = self.status[key]
        path = path_widget.value.strip()
        if not path or not os.path.isfile(path):
            status.value = (
                '<span style="color:#a00;">Open in ImageJ: load a file first.</span>'
            )
            return
        cmd_prefix, source, tried = resolve_imagej_path()
        if cmd_prefix is None:
            tried_html = '<br>'.join(
                f'&nbsp;&nbsp;<code>{_html_escape(t)}</code>' for t in tried
            ) or '&nbsp;&nbsp;(nothing)'
            status.value = (
                '<span style="color:#a00;">Open in ImageJ: install not '
                'detected. Set the path from a notebook cell:<br>'
                '&nbsp;&nbsp;<code>import cohere_ui.jupyter_gui as cgui</code><br>'
                "&nbsp;&nbsp;<code>cgui.IMAGEJ_PATH = '/path/to/ImageJ.app'</code>"
                ' &nbsp; (macOS .app bundle, Linux <code>ImageJ-linux64</code> '
                'binary, Windows <code>ImageJ-win64.exe</code>)<br>'
                f'<small>Searched paths:<br>{tried_html}</small></span>'
            )
            return
        try:
            subprocess.Popen(
                cmd_prefix + [path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            status.value = (
                f'<small style="color:#2a7a2a;">Opened in ImageJ via '
                f'<i>{source}</i>: <code>{cmd_prefix[-1]}</code></small>'
            )
        except Exception as e:
            status.value = (
                f'<span style="color:#a00;">Open in ImageJ failed: {e}</span>'
            )
            self._log_debug(format_error_summary(e, prefix='_open_imagej'))
            self._log_debug(traceback.format_exc())
