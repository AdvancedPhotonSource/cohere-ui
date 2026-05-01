"""PrepTab: beamline preprocessing configuration."""

import ast
import html as _html_mod
import os
import subprocess

import ipywidgets as widgets

try:
    from ipyfilechooser import FileChooser
except ImportError:
    FileChooser = None

from .base import BaseTab, _MSG
from ..widgets import form_row, text_field, dropdown, checkbox, button, LogPanel
from ..imagej import resolve_imagej_path


def _html_escape(text):
    return _html_mod.escape(str(text), quote=True)


class PrepTab(BaseTab):
    """Tab for beamline preprocessing configuration.

    Handles min_frames, roi, exclude_scans, outlier removal.
    """

    name = "Prep Data"
    conf_name = "config_prep"

    def _build_ui(self) -> widgets.Widget:
        self.min_frames = text_field(placeholder='e.g., 10')
        self.exclude_scans = text_field(placeholder='e.g., [1, 2, 3]')
        self.roi = text_field(placeholder='e.g., [x1, y1, x2, y2]')
        self.roi_format = dropdown(
            options=['', 'center_point_dist', 'start_point_end_point', 'start_point_dist'],
            value=''
        )
        self.max_crop = text_field(placeholder='e.g., [100, 100]')
        self.remove_outliers = checkbox('remove outliers')
        self.outliers_scans = text_field(placeholder='Auto-populated after prep')

        self.load_btn = button('Load Config', style='warning', width='120px', role='load')
        self.run_btn = button('Prepare', style='success', width='120px', role='run')
        self.load_btn.on_click(lambda b: self._load_config_dialog())
        self.run_btn.on_click(lambda b: self.run_tab())

        self.log_panel = LogPanel(height='150px')

        roi_tooltip = widgets.HTML(
            '<small style="color: #666;">center_point_dist: [cx, cy, dx, dy] | '
            'start_point_end_point: [x1, y1, x2, y2] | '
            'start_point_dist: [x1, dx, y1, dy]</small>'
        )

        layout = widgets.VBox([
            form_row('Min Frames', self.min_frames),
            form_row('Exclude Scans', self.exclude_scans),
            form_row('ROI', self.roi),
            form_row('ROI Format', self.roi_format),
            roi_tooltip,
            form_row('Max Crop', self.max_crop),
            self.remove_outliers,
            form_row('Outliers Scans', self.outliers_scans),
            widgets.HBox([self.load_btn, self.run_btn]),
            self.log_panel.widget,
            widgets.HTML('<hr style="margin:12px 0;">'),
            self._build_tiff_viewer(),
        ])

        return layout

    def _build_tiff_viewer(self) -> widgets.Widget:
        """Two-pane TIFF stack viewer.
        Left  = raw per-frame TIFFs from <data_dir> (config_instr).
        Right = beamline-preprocessed assembled stack (preprocessed_data/prep_data.tif).
        Each pane has its own slice slider and path field; shared toggles
        control sync-scroll and log scale. The path field accepts either a
        single TIFF (loaded as a 1- or N-slice stack) or a directory (every
        *.tif under it is loaded recursively, sorted by filename, and stacked).
        """
        self._tiff_data = {'raw': None, 'prep': None}

        self.tiff_sync = widgets.Checkbox(
            value=True, description='Sync scroll', indent=False,
        )
        self.tiff_log = widgets.Checkbox(
            value=False, description='Log scale', indent=False,
        )
        self.tiff_cmap = widgets.Dropdown(
            options=[
                'magma', 'viridis', 'plasma', 'inferno', 'cividis',
                'gray', 'hot', 'turbo', 'bone', 'cubehelix',
            ],
            value='magma',
            layout=widgets.Layout(width='130px'),
            description='Cmap',
        )
        self.tiff_invert = widgets.Checkbox(
            value=False, description='Invert cmap', indent=False,
        )
        self.tiff_zoom = widgets.Dropdown(
            options=[('1x (native)', 1), ('2x', 2), ('3x', 3), ('4x', 4), ('8x', 8)],
            value=1, description='Zoom',
            layout=widgets.Layout(width='130px'),
        )
        self.tiff_zoom.observe(lambda _c: self._tiff_render_all(), names='value')
        self.tiff_scale_mode = widgets.RadioButtons(
            options=[
                ('Auto (per slice)', 'auto'),
                ('Sync (shared across both panes)', 'sync'),
                ('Manual', 'manual'),
            ],
            value='auto',
            layout=widgets.Layout(width='280px'),
        )
        self.tiff_raw_vmin = text_field(placeholder='auto', width='90px')
        self.tiff_raw_vmax = text_field(placeholder='auto', width='90px')
        self.tiff_prep_vmin = text_field(placeholder='auto', width='90px')
        self.tiff_prep_vmax = text_field(placeholder='auto', width='90px')
        self._tiff_manual_row = widgets.HBox([
            widgets.HTML('<small>Manual: raw [vmin/vmax]</small>'),
            self.tiff_raw_vmin, self.tiff_raw_vmax,
            widgets.HTML('<small>&nbsp;&nbsp;prep [vmin/vmax]</small>'),
            self.tiff_prep_vmin, self.tiff_prep_vmax,
        ])
        self._tiff_manual_row.layout.display = 'none'
        self.tiff_log.observe(lambda _c: self._tiff_render_all(), names='value')
        self.tiff_cmap.observe(lambda _c: self._tiff_render_all(), names='value')
        self.tiff_invert.observe(lambda _c: self._tiff_render_all(), names='value')
        self.tiff_scale_mode.observe(self._tiff_on_scale_mode, names='value')
        for w in (self.tiff_raw_vmin, self.tiff_raw_vmax,
                  self.tiff_prep_vmin, self.tiff_prep_vmax):
            w.observe(lambda _c: self._tiff_render_all(), names='value')

        self.tiff_raw_path = text_field(
            placeholder='dir / file / glob (e.g. <data_dir>, frame.tif, scan_*.tif)',
            width='340px',
        )
        self.tiff_prep_path = text_field(
            placeholder='dir / file / glob (e.g. preprocessed_data/prep_data.tif)',
            width='340px',
        )
        self.tiff_raw_load = button('Load', style='warning', width='70px')
        self.tiff_prep_load = button('Load', style='warning', width='70px')
        self.tiff_raw_load.on_click(lambda _b: self._tiff_load('raw'))
        self.tiff_prep_load.on_click(lambda _b: self._tiff_load('prep'))
        ij_tooltip = (
            'Open this TIFF in ImageJ for full-resolution inspection '
            '(orthoviews, line profiles, ROIs).'
        )
        self.tiff_raw_imagej = button('ImageJ', style='', width='90px')
        self.tiff_prep_imagej = button('ImageJ', style='', width='90px')
        self.tiff_raw_imagej.tooltip = ij_tooltip
        self.tiff_prep_imagej.tooltip = ij_tooltip
        self.tiff_raw_imagej.on_click(lambda _b: self._tiff_open_imagej('raw'))
        self.tiff_prep_imagej.on_click(lambda _b: self._tiff_open_imagej('prep'))

        # Native pixel display: no width / height constraints, no
        # object-fit. Image shows at exactly the source TIFF resolution
        # (e.g. 256x256 for prep_data.tif). No browser smoothing, no PIL
        # upscale, no resampling artifacts. Container box scrolls if the
        # image is larger than the available space.
        self.tiff_raw_image = widgets.Image(
            format='png',
            layout=widgets.Layout(border='1px solid #ddd'),
        )
        self.tiff_prep_image = widgets.Image(
            format='png',
            layout=widgets.Layout(border='1px solid #ddd'),
        )
        self.tiff_raw_slider = widgets.IntSlider(
            value=0, min=0, max=0, step=1, description='Slice',
            continuous_update=True, layout=widgets.Layout(width='95%'),
        )
        self.tiff_prep_slider = widgets.IntSlider(
            value=0, min=0, max=0, step=1, description='Slice',
            continuous_update=True, layout=widgets.Layout(width='95%'),
        )
        self.tiff_raw_slider.observe(lambda c: self._tiff_on_slice('raw', c), names='value')
        self.tiff_prep_slider.observe(lambda c: self._tiff_on_slice('prep', c), names='value')

        self.tiff_raw_status = widgets.HTML(value='<i>not loaded</i>')
        self.tiff_prep_status = widgets.HTML(value='<i>not loaded</i>')

        # Per-pane file/folder chooser. Default: file mode. The "Folder mode"
        # checkbox flips show_only_dirs at runtime so the same chooser can
        # pick either a single TIFF or a directory of per-frame TIFFs.
        self.tiff_raw_chooser = self._build_chooser('raw') if FileChooser else None
        self.tiff_prep_chooser = self._build_chooser('prep') if FileChooser else None
        self.tiff_raw_dir_mode = widgets.Checkbox(
            value=False, description='Folder mode', indent=False,
            layout=widgets.Layout(width='130px'),
        )
        self.tiff_prep_dir_mode = widgets.Checkbox(
            value=False, description='Folder mode', indent=False,
            layout=widgets.Layout(width='130px'),
        )
        self.tiff_raw_dir_mode.observe(
            lambda c: self._tiff_on_dir_mode('raw', c), names='value',
        )
        self.tiff_prep_dir_mode.observe(
            lambda c: self._tiff_on_dir_mode('prep', c), names='value',
        )

        # Pin the image widget into a fixed-height container so the slice
        # metadata (which lives BELOW the image) doesn't shove the panel
        # contents up/down between renders. Both panes share the same
        # geometry so they stay aligned when sync-scrolling.
        # Fixed-height box keeps the panes aligned; scroll if the image
        # would otherwise overflow. Image displays centered at native res.
        raw_browse_row = (
            widgets.HBox([self.tiff_raw_chooser, self.tiff_raw_dir_mode])
            if self.tiff_raw_chooser is not None
            else widgets.HTML('<small><i>install ipyfilechooser to browse</i></small>')
        )
        prep_browse_row = (
            widgets.HBox([self.tiff_prep_chooser, self.tiff_prep_dir_mode])
            if self.tiff_prep_chooser is not None
            else widgets.HTML('<small><i>install ipyfilechooser to browse</i></small>')
        )
        raw_pane = widgets.VBox([
            widgets.HTML('<b>Raw frames (per-frame detector TIFFs from data_dir)</b>'),
            widgets.HBox([self.tiff_raw_path, self.tiff_raw_load, self.tiff_raw_imagej]),
            raw_browse_row,
            widgets.Box([self.tiff_raw_image],
                        layout=widgets.Layout(height='560px', overflow='auto',
                                              align_items='center', justify_content='center')),
            self.tiff_raw_slider,
            self.tiff_raw_status,
        ], layout=widgets.Layout(width='50%', padding='0 6px 0 0'))
        prep_pane = widgets.VBox([
            widgets.HTML('<b>Beamline-preprocessed (assembled stack)</b>'),
            widgets.HBox([self.tiff_prep_path, self.tiff_prep_load, self.tiff_prep_imagej]),
            prep_browse_row,
            widgets.Box([self.tiff_prep_image],
                        layout=widgets.Layout(height='560px', overflow='auto',
                                              align_items='center', justify_content='center')),
            self.tiff_prep_slider,
            self.tiff_prep_status,
        ], layout=widgets.Layout(width='50%', padding='0 0 0 6px'))

        return widgets.VBox([
            widgets.HTML('<h4 style="margin:6px 0;">TIFF Viewer</h4>'),
            widgets.HBox([self.tiff_sync, self.tiff_log,
                          self.tiff_cmap, self.tiff_invert, self.tiff_zoom]),
            widgets.HBox([widgets.HTML('<small>Intensity scale:</small>'),
                          self.tiff_scale_mode]),
            self._tiff_manual_row,
            widgets.HBox([raw_pane, prep_pane],
                         layout=widgets.Layout(width='100%')),
        ])

    def _tiff_on_scale_mode(self, change):
        self._tiff_manual_row.layout.display = (
            '' if change.get('new') == 'manual' else 'none'
        )
        self._tiff_render_all()

    def _stack_frames(self, files, status, tifffile, np):
        """Read a sorted list of per-frame TIFFs into a single 3D array.
        On error, write to ``status`` and return None."""
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

    def _build_chooser(self, kind):
        """Build an ipyfilechooser for one pane. Default: file mode.
        The Folder-mode checkbox flips show_only_dirs at runtime."""
        assert FileChooser is not None  # call sites guard with `if FileChooser`
        fc = FileChooser(
            path=os.getcwd(),
            select_default=False,
            show_only_dirs=False,
            title='',
        )
        fc.register_callback(lambda c: self._tiff_on_chooser(kind, c))
        return fc

    def _tiff_on_chooser(self, kind, chooser):
        """ipyfilechooser callback: copy selection to the path text and load."""
        sel = chooser.selected_path if chooser.show_only_dirs else chooser.selected
        if not sel:
            return
        path_widget = self.tiff_raw_path if kind == 'raw' else self.tiff_prep_path
        path_widget.value = sel
        self._tiff_load(kind)

    def _tiff_on_dir_mode(self, kind, change):
        """Toggle the chooser between file and folder mode at runtime."""
        chooser = self.tiff_raw_chooser if kind == 'raw' else self.tiff_prep_chooser
        if chooser is None:
            return
        chooser.show_only_dirs = bool(change.get('new'))
        try:
            chooser.refresh()
        except Exception:
            pass

    def _tiff_default_path(self, kind):
        try:
            base = self.main_gui.experiment_dir
        except Exception:
            return None
        if not base:
            return None
        if kind == 'prep':
            p = os.path.join(base, 'preprocessed_data', 'prep_data.tif')
            return p if os.path.isfile(p) else None
        # 'raw': pull data_dir from config_instr (per-frame TIFFs live there)
        try:
            instr = self.main_gui.config_manager.get_cached('config_instr') or {}
        except Exception:
            instr = {}
        data_dir = instr.get('data_dir')
        if data_dir and os.path.isdir(data_dir):
            return data_dir
        return None

    def _tiff_load(self, kind):
        path_widget = self.tiff_raw_path if kind == 'raw' else self.tiff_prep_path
        status = self.tiff_raw_status if kind == 'raw' else self.tiff_prep_status
        slider = self.tiff_raw_slider if kind == 'raw' else self.tiff_prep_slider
        path = path_widget.value.strip() or self._tiff_default_path(kind) or ''
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
            import glob as _glob
            if is_glob:
                # User-supplied glob (e.g. "<dir>/scan_54_*.tif" or
                # "<root>/**/*.tiff"). Honor the literal pattern; the user
                # is in charge of which extension(s) to match.
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
                # Per-frame TIFFs (e.g. <data_dir>/<scan>/<scan>_NNNNN.tif).
                # Recursive glob both .tif and .tiff, sort, stack.
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
            return
        self._tiff_data[kind] = arr
        self._tiff_scale_cache = None  # invalidate sync-mode bounds
        path_widget.value = path
        status.value = (
            f'<small style="color:#444;">{display_name} '
            f'shape={arr.shape} dtype={arr.dtype}</small>'
        )
        slider.max = max(0, arr.shape[0] - 1)
        slider.value = arr.shape[0] // 2
        self._tiff_render(kind)

    def _tiff_on_slice(self, kind, change):
        if self._tiff_data.get(kind) is None:
            return
        self._tiff_render(kind)
        if self.tiff_sync.value:
            other = 'prep' if kind == 'raw' else 'raw'
            other_arr = self._tiff_data.get(other)
            if other_arr is None:
                return
            other_slider = self.tiff_prep_slider if other == 'prep' else self.tiff_raw_slider
            target = min(int(change.get('new', 0)), other_arr.shape[0] - 1)
            if other_slider.value != target:
                other_slider.value = target  # observer fires _tiff_render(other)

    def _tiff_render_all(self):
        # Sync mode reads bounds from BOTH stacks; cache invalidates when
        # log toggle, manual fields, or the underlying data change.
        self._tiff_scale_cache = None
        for kind in ('raw', 'prep'):
            if self._tiff_data.get(kind) is not None:
                self._tiff_render(kind)

    def _tiff_compute_scale(self, kind, scaled_slice):
        """Return (vmin, vmax, label) for the requested kind.

        Auto: per-slice min/max (responsive but can't compare across slices).
        Sync: shared min/max across BOTH stacks' full data, post log scaling.
        Manual: user-entered vmin/vmax fields; falls back to per-slice if blank.
        """
        import numpy as np
        mode = self.tiff_scale_mode.value
        if mode == 'manual':
            vmin_w = self.tiff_raw_vmin if kind == 'raw' else self.tiff_prep_vmin
            vmax_w = self.tiff_raw_vmax if kind == 'raw' else self.tiff_prep_vmax
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
            cache = getattr(self, '_tiff_scale_cache', None)
            if cache is None:
                bounds = []
                for k in ('raw', 'prep'):
                    arr = self._tiff_data.get(k)
                    if arr is None:
                        continue
                    a = arr.astype(np.float32, copy=False)
                    if self.tiff_log.value:
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
        # auto (default)
        return float(scaled_slice.min()), float(scaled_slice.max()), 'auto'

    def _tiff_render(self, kind):
        # Direct Pillow + colormap LUT render (no matplotlib figure
        # overhead). Each pixel in the output PNG corresponds 1:1 to a
        # pixel in the underlying TIFF slice (no resampling, no axes,
        # no anti-aliasing). Far faster than matplotlib (we typically see
        # ~5-15 ms per slice for 512x512 instead of ~150 ms with figures).
        import io
        import numpy as np
        from PIL import Image
        import matplotlib

        arr = self._tiff_data[kind]
        slider = self.tiff_raw_slider if kind == 'raw' else self.tiff_prep_slider
        status = self.tiff_raw_status if kind == 'raw' else self.tiff_prep_status
        idx = min(slider.value, arr.shape[0] - 1)
        img_widget = self.tiff_raw_image if kind == 'raw' else self.tiff_prep_image
        path_widget = self.tiff_raw_path if kind == 'raw' else self.tiff_prep_path

        slice_2d = np.asarray(arr[idx])
        if self.tiff_log.value:
            scaled = np.log10(np.maximum(slice_2d, 0).astype(np.float32) + 1.0)
            label = 'log10(I+1)'
        else:
            scaled = slice_2d.astype(np.float32, copy=False)
            label = 'intensity'

        mn, mx, scale_label = self._tiff_compute_scale(kind, scaled)
        if mx > mn:
            norm = np.clip((scaled - mn) / (mx - mn), 0.0, 1.0)
            norm = (norm * 255.0).astype(np.uint8)
        else:
            norm = np.zeros_like(scaled, dtype=np.uint8)

        cmap_name = self.tiff_cmap.value
        if self.tiff_invert.value:
            cmap_name = cmap_name + '_r'
        if not hasattr(self, '_tiff_lut_cache'):
            self._tiff_lut_cache = {}
        lut = self._tiff_lut_cache.get(cmap_name)
        if lut is None:
            cmap_obj = matplotlib.colormaps.get_cmap(cmap_name)
            lut = (cmap_obj(np.arange(256))[:, :3] * 255).astype(np.uint8)
            self._tiff_lut_cache[cmap_name] = lut
        rgb = lut[norm]
        pil_img = Image.fromarray(rgb, mode='RGB')
        zoom = max(1, int(self.tiff_zoom.value))
        if zoom > 1:
            # NEAREST replicates each source pixel exactly. No smoothing,
            # no resampling, no new colors; every output pixel maps 1:1
            # to a source pixel, just at a larger display size.
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

    def _tiff_open_imagej(self, kind):
        path_widget = self.tiff_raw_path if kind == 'raw' else self.tiff_prep_path
        status = self.tiff_raw_status if kind == 'raw' else self.tiff_prep_status
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

    def load_tab(self, conf_map: dict):
        """Populate widgets from config dictionary."""
        if 'min_frames' in conf_map:
            self.min_frames.value = self._fmt_value(conf_map['min_frames'])
        if 'exclude_scans' in conf_map:
            self.exclude_scans.value = self._fmt_value(conf_map['exclude_scans'])
        if 'roi' in conf_map:
            self.roi.value = self._fmt_value(conf_map['roi'])
        if 'roi_format' in conf_map:
            self.roi_format.value = conf_map['roi_format']
        else:
            self.roi_format.value = ''
        if 'max_crop' in conf_map:
            self.max_crop.value = self._fmt_value(conf_map['max_crop'])
        self.remove_outliers.value = conf_map.get('remove_outliers', False)
        if 'outliers_scans' in conf_map:
            self.outliers_scans.value = self._fmt_value(conf_map['outliers_scans'])
        # Pre-fill TIFF viewer paths if the corresponding files exist.
        for kind in ('raw', 'prep'):
            default = self._tiff_default_path(kind)
            if default:
                widget = self.tiff_raw_path if kind == 'raw' else self.tiff_prep_path
                widget.value = default

    def get_config(self) -> dict:
        """Read current widget values into config dictionary."""
        conf_map = {}

        if self.min_frames.value:
            conf_map['min_frames'] = ast.literal_eval(self.min_frames.value)
        if self.exclude_scans.value:
            conf_map['exclude_scans'] = ast.literal_eval(self.exclude_scans.value)
        if self.roi.value:
            conf_map['roi'] = ast.literal_eval(self.roi.value)
        if self.roi_format.value:
            conf_map['roi_format'] = self.roi_format.value
        if self.max_crop.value:
            conf_map['max_crop'] = ast.literal_eval(self.max_crop.value)
        if self.remove_outliers.value:
            conf_map['remove_outliers'] = True

        return conf_map

    def clear_conf(self):
        """Reset all widgets to defaults."""
        self.min_frames.value = ''
        self.exclude_scans.value = ''
        self.roi.value = ''
        self.roi_format.value = ''
        self.max_crop.value = ''
        self.outliers_scans.value = ''
        self.remove_outliers.value = False

    def run_tab(self):
        """Execute beamline preprocessing."""
        import cohere_ui.beamline_preprocess as prep

        self.clear_output()

        err = self._validate_experiment()
        if err:
            self.log_error(err)
            return

        conf_map = self.get_config()
        err = self.main_gui.config_manager.verify(self.conf_name, conf_map)
        if err and not self.main_gui.no_verify:
            self.log_error(_MSG['tab']['config_error'].format(error=err))
            return

        # Preserve outliers_scans across runs when remove_outliers is on.
        if self.remove_outliers.value:
            current_prep = self.main_gui.config_manager.load_config(self.conf_name)
            if current_prep and 'outliers_scans' in current_prep:
                conf_map['outliers_scans'] = current_prep['outliers_scans']

        _, action = self.main_gui.config_manager.save_config(
            self.conf_name, conf_map, self.main_gui.no_verify)
        if action:
            self._log_config_action(action)

        before = self._snapshot_outputs()
        try:
            self.log_info(_MSG['prep']['running'])
            prep.handle_prep(self.main_gui.experiment_dir, no_verify=self.main_gui.no_verify)
            self.log_success(_MSG['prep']['complete'])

            updated_conf = self.main_gui.config_manager.load_config(self.conf_name)
            if updated_conf:
                self.clear_conf()
                self.load_tab(updated_conf)

        except (ValueError, FileNotFoundError, KeyError) as e:
            self.log_error(f"{type(e).__name__}: {e}")
        finally:
            self._log_file_changes(before)

