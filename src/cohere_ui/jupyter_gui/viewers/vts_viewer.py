"""In-tab interactive VTS/VTI viewer for the Postprocess tab.

Wraps a PyVista ``Plotter`` rendered through the trame Jupyter backend
into a collapsible ipywidget panel with a file picker, scalar selector,
isosurface threshold + opacity sliders, colormap dropdown, and reload /
dispose / "Open in ParaView" actions.

Lifecycle:

  * Lazy: the plotter and the trame widget are not constructed until
    the user expands the panel for the first time. PyVista's
    ``Plotter`` constructor creates a real VTK render window with an
    OpenGL context, which is non-trivial to set up; deferring it keeps
    the GUI's import + render cost low for users who only use the
    other tabs.
  * Reload in place: the same ``Plotter`` instance is kept across
    ``reload()`` calls; only the actors are swapped. Avoids the
    leak-prone "close + reconstruct" cycle and avoids the iframe
    refresh flicker the user would otherwise see.
  * Dispose: ``dispose()`` closes the ipywidget, the plotter (which
    tears down the VTK render window, scalar bars, and actor refs),
    and clears the iframe output. After dispose the panel can be
    expanded again to construct a fresh plotter.

Threading:

  Per-tab user interactions (click the chevron, drag a slider, change
  the file dropdown) fire on the kernel's main thread, which is where
  VTK + trame want to be. Two paths do NOT come from the main thread,
  though:

    * ``CoherenceGUI._on_run_all_clicked`` spawns a background
      ``CoherenceGUI-run-all`` thread that walks every tab's
      ``run_tab``; ``DispTab.run_tab``'s ``finally`` hook calls
      ``viewer.reload()`` (or ``_refresh_file_options``) from there.
    * Any future watchdog callback that wants to ping the viewer.

  ``reload()`` / ``_refresh_file_options()`` therefore self-marshal:
  if the call originates off the captured kernel IOLoop's thread,
  the work is scheduled via ``loop.add_callback`` and the off-thread
  call returns immediately. Inline execution still happens when the
  caller is already on the main thread (and in non-Jupyter contexts
  where no IOLoop was captured, e.g. unit tests).
"""

import glob
import html
import os
import threading
from contextlib import contextmanager

import ipywidgets as widgets
from IPython.display import display

from cohere_ui.jupyter_gui.viewers.paraview_launcher import open_in_paraview


# Default scalar to color by when present (amplitude is the canonical
# "did this reconstruction work" signal). When absent, the viewer falls
# back to the first non-VTK-internal array in the file.
_PREFERRED_SCALAR = 'imAmp'

# Filename glob patterns in priority order. The first pattern that
# matches anything under ``results_viz/`` provides the default-selected
# file; everything else still appears in the dropdown, ordered by this
# priority list.
_VTS_PRIORITY = (
    'direct_space_images_AmpPhase.vts',
    'direct_space_images_*.vts',
    'direct_space_images_interpolated_AmpPhase.vti',
    'direct_space_images_interpolated_*.vti',
    'twin_direct_space_images.vts',
    'reciprocal_space.vts',
    'resolution_direct.vts',
    'resolution_recip.vts',
    '*_coherence.vts',
    'full_data.vti',
    '*.vts',
    '*.vti',
)

# Colormaps offered in the dropdown. All are perceptually-uniform with
# the exception of turbo (kept because it's familiar to BCDI users
# coming from ParaView's default palette).
_CMAPS = ('viridis', 'plasma', 'magma', 'inferno', 'cividis', 'turbo', 'coolwarm', 'twilight')


class VtsViewer:
    """Collapsible interactive 3D viewer for postprocess VTS/VTI files."""

    # Match the FeaturePanel chevron convention so the toggle reads as
    # a heading instead of a button.
    _COLLAPSED_GLYPH = '▸'
    _EXPANDED_GLYPH = '▾'

    def __init__(self, main_gui):
        self.main_gui = main_gui

        # Render state. None until the user expands the panel for the
        # first time; recreated by _ensure_plotter() on next expand
        # after dispose().
        self._plotter = None
        self._viewer_widget = None
        self._mesh = None       # current pv dataset (StructuredGrid / ImageData)
        self._actor = None      # last actor added (multi-iso adds several)
        self._current_file = None

        # Camera preset chosen via the Display-options camera row.
        # None means "auto-fit on next render"; otherwise one of the
        # _CAMERA_PRESETS keys. Persists across mode switches so
        # rebuilding the actors doesn't yank the user's chosen view.
        # We cannot read back a user-orbited camera from vtk.js
        # client-rendering, so any in-browser orbit is lost on the
        # next re-render, acknowledged tradeoff for the safe
        # cross-platform render path.
        self._pending_camera = None

        # "User touched these knobs" flags. When True, mode / scalar
        # cascades that would otherwise apply BCDI defaults skip the
        # override so the user's explicit values survive. Cleared
        # on file load (fresh data warrants fresh defaults) and on
        # the explicit Auto button (which IS the user asking for the
        # default to be re-applied).
        self._user_clim_dirty = False
        self._user_cmap_dirty = False

        # Set during programmatic state swaps (file change cascades
        # into scalar repopulation + iso reset) so each observer fires
        # exactly once at the end. Without this, changing the file
        # would re-render twice: once when scalar_selector.options is
        # rewritten, again when scalar_selector.value lands.
        self._suppress_observers = False

        # Capture the kernel's tornado IOLoop on the main thread now
        # so reload() / _refresh_file_options() can self-marshal when
        # invoked from CoherenceGUI's Run All worker thread (see module
        # docstring). In non-Jupyter contexts this is None and the
        # methods run inline, safe there since there's no Cocoa/GL
        # context to crash into.
        try:
            import tornado.ioloop
            self._main_loop = tornado.ioloop.IOLoop.current(instance=False)
        except Exception:
            self._main_loop = None
        # Record the main-thread identity so we can compare cheaply
        # against threading.get_ident() in the marshal check (faster
        # than asking the IOLoop).
        self._main_thread_ident = threading.get_ident()

        self._build_ui()

    def _on_main_thread(self) -> bool:
        """True iff the calling thread is the one that built this viewer.

        Cheap call; used by reload() / _refresh_file_options() to
        decide whether to run inline or marshal via the IOLoop.
        """
        return threading.get_ident() == self._main_thread_ident

    # Process-wide flag so multiple VtsViewer instances (e.g., across
    # several CoherenceGUI cells) don't each try to re-patch the loop.
    _signal_patch_applied = False

    @staticmethod
    def _detect_trame_extension() -> bool:
        """True iff ``trame_jupyter_extension`` is importable.

        We only check Python-side presence (the package is the trame
        bridge that proxies its WSGI through the running Jupyter
        server). Server-side enablement is a Jupyter Server config
        concern; if the Python package is installed but the server
        extension wasn't enabled, show_trame's iframe URL will 404
        and the user gets a visible error, safer than silently
        falling back when we can't probe the server's config from
        the kernel side.
        """
        try:
            import trame_jupyter_extension  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def _patch_loop_signal_handlers_once(cls):
        """No-op asyncio signal handlers on the running event loop.

        Why this is needed:
          * trame's wslink/aiohttp backend calls
            ``loop.add_signal_handler(SIGINT, _raise_graceful_exit)``
            during server bootstrap (aiohttp/web_runner.py:283).
          * ``asyncio.unix_events.add_signal_handler`` requires the
            loop's thread be the OS main thread (CPython enforces
            this via ``signal.set_wakeup_fd``).
          * ipykernel >= 6 runs its tornado IOLoop on a worker thread;
            the OS main thread is reserved for native GUI integration
            (Cocoa on macOS, Qt on Linux). Result: trame never starts
            and the iframe stays blank.

        The clean fix is the ``trame-jupyter-extension`` Python package,
        which routes through the existing Jupyter server and never
        spins up a separate aiohttp instance. Until that's in the
        Jupyter extra of cohere-ui, this patch makes
        ``add_signal_handler`` / ``remove_signal_handler`` no-op on
        the active loop. Safe because:
          * The Jupyter kernel installs its own SIGINT handler at
            startup; trame doesn't need to.
          * No other component in this codebase registers asyncio
            signal handlers (kept stable across releases by the
            patch's idempotency).

        Patch is applied once per process. ``cls._signal_patch_applied``
        guards against re-patching across multiple viewer instances.
        """
        if cls._signal_patch_applied:
            return
        try:
            import asyncio
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except Exception:
            return
        if loop is None or not hasattr(loop, 'add_signal_handler'):
            return
        # Replace bound methods with no-op lambdas. We bind them as
        # attribute overrides on the instance so subsequent loop
        # lookups by other code still get the patched versions.
        loop.add_signal_handler = lambda *_a, **_kw: None
        loop.remove_signal_handler = lambda *_a, **_kw: True
        cls._signal_patch_applied = True

    @property
    def widget(self) -> widgets.Widget:
        """The viewer's root widget; embed this in the DispTab VBox."""
        return self._widget

    @property
    def is_visible(self) -> bool:
        """True iff the body is currently expanded.

        DispTab's ``run_tab`` checks this in its post-run hook to
        decide whether to auto-reload (an expanded viewer reflects the
        new run immediately; a collapsed viewer just refreshes its
        file picker so the next expand is current).
        """
        return self._body.layout.display != 'none'

    # ----- UI assembly -----

    def _build_ui(self):
        """Construct the collapsible title row + hidden body.

        Layout (v2):

          [title chevron / count]
          File [v]  Scalar [v]  Mode [v]
          Iso [slider]  Opacity [slider]  Cmap [v]      (each shown
                                                         based on mode)
          Color by [v]  Levels [v]  X(o)[off] Y(o)[off]  (conditional)
          [Reload] [Dispose] [Open in ParaView]
          v Display options                              (collapsible)
              Camera: [+X][-X][+Y][-Y][+Z][-Z][Iso][Fit]
              Bg [v]  Edges(o) Axes(x) Scalar bar(x)
              vmin [..] vmax [..] [Auto]
          [status line]
          [trame iframe]
        """
        # ----- title -----
        # Placeholder description; _refresh_title() at the end of
        # _build_ui rewrites it once file_picker exists.
        self._title_btn = widgets.Button(
            description='▸  3D Viewer',
            layout=widgets.Layout(
                width='100%', height='28px', padding='0',
                margin='8px 0 0 0',
            ),
        )
        self._title_btn.add_class('jup-gui-feature-title-toggle')
        self._title_btn.on_click(self._on_toggle)

        # ----- top row: file / scalar / mode -----
        self.file_picker = widgets.Dropdown(
            options=[], value=None,
            layout=widgets.Layout(width='300px'),
        )
        self.scalar_selector = widgets.Dropdown(
            options=[], value=None,
            layout=widgets.Layout(width='160px'),
            tooltip='Scalar used to build the iso surface',
        )
        self.mode_dd = widgets.Dropdown(
            options=[
                ('Single iso', 'single-iso'),
                ('Multi-iso',  'multi-iso'),
                ('Slice planes', 'slices'),
                ('Glyphs (vectors)', 'glyphs'),
                ('Outline only', 'outline'),
            ],
            value='single-iso',
            layout=widgets.Layout(width='160px'),
        )

        # ----- render row (iso / opacity / cmap) -----
        # continuous_update=False keeps slider drags from flooding the
        # browser with intermediate frames.
        self.iso_slider = widgets.FloatSlider(
            value=0.3, min=0.0, max=1.0, step=0.01,
            continuous_update=False, readout=True, readout_format='.3g',
            layout=widgets.Layout(width='300px'),
        )
        self.opacity_slider = widgets.FloatSlider(
            value=1.0, min=0.0, max=1.0, step=0.05,
            continuous_update=False, readout=True, readout_format='.2f',
            layout=widgets.Layout(width='200px'),
        )
        self.cmap_selector = widgets.Dropdown(
            options=list(_CMAPS), value='viridis',
            layout=widgets.Layout(width='140px'),
        )
        self._iso_box = self._labeled('Iso',  self.iso_slider, label_w='40px')
        self._opacity_box = self._labeled('Opacity', self.opacity_slider, label_w='60px')
        self._cmap_box = self._labeled('Cmap', self.cmap_selector, label_w='50px')

        # ----- conditional row -----
        self.color_dd = widgets.Dropdown(
            options=[], value=None,
            layout=widgets.Layout(width='160px'),
            tooltip='Scalar used to COLOR the iso / slice / glyph (may differ from the iso scalar)',
        )
        self.levels_dd = widgets.Dropdown(
            options=[('2', 2), ('3', 3), ('5', 5)],
            value=3,
            layout=widgets.Layout(width='60px'),
        )
        # Slice axis toggles (one per axis) + offset sliders (per axis,
        # normalized to [-1, +1] of the half-extent so values are
        # mesh-agnostic).
        self.slice_axes = {
            'x': widgets.Checkbox(value=True,  description='X', indent=False,
                                   layout=widgets.Layout(width='40px')),
            'y': widgets.Checkbox(value=True,  description='Y', indent=False,
                                   layout=widgets.Layout(width='40px')),
            'z': widgets.Checkbox(value=False, description='Z', indent=False,
                                   layout=widgets.Layout(width='40px')),
        }
        self.slice_offsets = {
            ax: widgets.FloatSlider(
                value=0.0, min=-1.0, max=1.0, step=0.05,
                continuous_update=False, readout=True, readout_format='.2f',
                layout=widgets.Layout(width='160px'),
            )
            for ax in ('x', 'y', 'z')
        }
        self.vec_dd = widgets.Dropdown(
            options=[], value=None,
            layout=widgets.Layout(width='160px'),
            tooltip='Vector array to orient glyphs (3-component only)',
        )
        self.stride_dd = widgets.Dropdown(
            options=[('4', 4), ('8', 8), ('16', 16), ('32', 32), ('64', 64)],
            value=16,
            layout=widgets.Layout(width='60px'),
            tooltip='Downsample factor before glyph generation (higher = fewer arrows, faster)',
        )

        self._color_box   = self._labeled('Color by', self.color_dd, label_w='70px')
        self._levels_box  = self._labeled('Levels',   self.levels_dd, label_w='60px')
        self._slices_box  = widgets.HBox(
            [self.slice_axes['x'], self.slice_offsets['x'],
             self.slice_axes['y'], self.slice_offsets['y'],
             self.slice_axes['z'], self.slice_offsets['z']],
            layout=widgets.Layout(
                align_items='center', flex_flow='row wrap', margin='0',
            ),
        )
        self._glyphs_box = widgets.HBox(
            [self._labeled('Vector', self.vec_dd, label_w='60px'),
             self._labeled('Stride', self.stride_dd, label_w='60px')],
            layout=widgets.Layout(align_items='center', flex_flow='row wrap', margin='0'),
        )

        # ----- actions -----
        self.reload_btn = widgets.Button(
            description='Reload',
            tooltip='Rescan results_viz/ and re-read the current file',
            layout=widgets.Layout(width='90px'),
        )
        self.dispose_btn = widgets.Button(
            description='Dispose',
            tooltip='Close the plotter and free GL resources (next expand reconstructs)',
            layout=widgets.Layout(width='90px'),
        )
        self.paraview_btn = widgets.Button(
            description='Open in ParaView',
            tooltip='Launch ParaView on the currently selected file',
            layout=widgets.Layout(width='160px'),
        )

        # ----- Display options collapsible -----
        # Chevron-toggle pattern copied from LiveFeature._benchmark_toggle
        # in features/rec_features.py, same idiom as the outer viewer
        # toggle so the visual language is consistent.
        self._disp_toggle_btn = widgets.Button(
            description='▸  Display options',
            layout=widgets.Layout(
                width='auto', height='24px', padding='0',
                margin='6px 0 0 0',
            ),
        )
        self._disp_toggle_btn.add_class('jup-gui-feature-title-toggle')
        self._disp_toggle_btn.on_click(self._on_disp_toggle)
        self._disp_open = False

        # Camera preset buttons (8 small buttons in one row).
        self._cam_buttons = {}
        cam_btn_widgets = []
        for label in ('+X', '-X', '+Y', '-Y', '+Z', '-Z', 'Iso', 'Fit'):
            btn = widgets.Button(
                description=label,
                layout=widgets.Layout(width='44px', margin='0 2px'),
                tooltip=f'Camera preset {label}',
            )
            # Capture preset name by default arg, not closure, so each
            # lambda sees its own key (closure-capture gotcha).
            btn.on_click(lambda _b, k=label: self._on_camera_preset(k))
            self._cam_buttons[label] = btn
            cam_btn_widgets.append(btn)
        cam_row = widgets.HBox(
            [widgets.Label('Camera', layout=widgets.Layout(width='60px')),
             *cam_btn_widgets],
            layout=widgets.Layout(align_items='center', flex_flow='row wrap', margin='2px 0'),
        )

        # Appearance row: background + edges/axes/scalar-bar toggles.
        self.bg_dd = widgets.Dropdown(
            options=[('white', 'white'), ('black', 'black'),
                     ('grey', '#bbbbbb'), ('paraview', '#444444')],
            value='white',
            layout=widgets.Layout(width='110px'),
        )
        self.edges_cb = widgets.Checkbox(value=False, description='Edges',
                                          indent=False,
                                          layout=widgets.Layout(width='90px'))
        self.axes_cb = widgets.Checkbox(value=True,  description='Axes',
                                         indent=False,
                                         layout=widgets.Layout(width='80px'))
        self.sbar_cb = widgets.Checkbox(value=True,  description='Scalar bar',
                                         indent=False,
                                         layout=widgets.Layout(width='120px'))
        appearance_row = widgets.HBox(
            [self._labeled('Bg', self.bg_dd, label_w='30px'),
             self.edges_cb, self.axes_cb, self.sbar_cb],
            layout=widgets.Layout(align_items='center', flex_flow='row wrap', margin='2px 0'),
        )

        # Scalar range row: vmin/vmax + Auto recompute.
        self.vmin_field = widgets.FloatText(
            value=0.0, step=0.01,
            layout=widgets.Layout(width='100px'),
        )
        self.vmax_field = widgets.FloatText(
            value=1.0, step=0.01,
            layout=widgets.Layout(width='100px'),
        )
        self.auto_clim_btn = widgets.Button(
            description='Auto',
            tooltip='Set vmin/vmax to the 1st/99th percentile of the current color scalar',
            layout=widgets.Layout(width='60px'),
        )
        self.auto_clim_btn.on_click(self._on_auto_clim)
        clim_row = widgets.HBox(
            [widgets.Label('vmin', layout=widgets.Layout(width='40px')),
             self.vmin_field,
             widgets.Label('vmax', layout=widgets.Layout(width='40px',
                                                          margin='0 0 0 8px')),
             self.vmax_field,
             self.auto_clim_btn],
            layout=widgets.Layout(align_items='center', flex_flow='row wrap', margin='2px 0'),
        )

        self._disp_body = widgets.VBox(
            [cam_row, appearance_row, clim_row],
            layout=widgets.Layout(
                display='none', padding='4px 8px 4px 16px',
                margin='0 0 4px 0',
                border_left='2px solid #d0d0d0',
            ),
        )

        # ----- status + viewer output -----
        self.status = widgets.HTML(
            value='', layout=widgets.Layout(margin='2px 0 2px 0'),
        )
        self.viewer_output = widgets.Output(
            layout=widgets.Layout(
                min_height='540px', width='100%',
                border='1px solid #ddd', margin='4px 0 0 0',
            ),
        )

        # ----- assemble rows -----
        controls_top = widgets.HBox(
            [self._labeled('File',   self.file_picker, label_w='40px'),
             self._labeled('Scalar', self.scalar_selector, label_w='55px'),
             self._labeled('Mode',   self.mode_dd, label_w='45px')],
            layout=widgets.Layout(align_items='center', flex_flow='row wrap'),
        )
        render_row = widgets.HBox(
            [self._iso_box, self._opacity_box, self._cmap_box],
            layout=widgets.Layout(align_items='center', flex_flow='row wrap',
                                   margin='4px 0 0 0'),
        )
        cond_row = widgets.HBox(
            [self._color_box, self._levels_box, self._slices_box, self._glyphs_box],
            layout=widgets.Layout(align_items='center', flex_flow='row wrap',
                                   margin='2px 0 0 0'),
        )
        actions_row = widgets.HBox(
            [self.reload_btn, self.dispose_btn, self.paraview_btn],
            layout=widgets.Layout(margin='6px 0 0 0'),
        )

        self._body = widgets.VBox(
            [controls_top, render_row, cond_row, actions_row,
             self._disp_toggle_btn, self._disp_body,
             self.status, self.viewer_output],
            layout=widgets.Layout(
                display='none', padding='10px 12px',
                margin='4px 0 0 0',
            ),
        )
        self._body.add_class('jup-gui-vts-viewer-frame')

        self._widget = widgets.VBox([self._title_btn, self._body])

        # ----- observers -----
        self.file_picker.observe(self._on_file_change, names='value')
        self.scalar_selector.observe(self._on_scalar_change, names='value')
        self.mode_dd.observe(self._on_mode_change, names='value')
        self.iso_slider.observe(self._on_iso_change, names='value')
        self.opacity_slider.observe(self._on_opacity_change, names='value')
        self.cmap_selector.observe(self._on_cmap_change, names='value')
        self.color_dd.observe(self._on_color_change, names='value')
        self.levels_dd.observe(self._on_levels_change, names='value')
        for ax in ('x', 'y', 'z'):
            self.slice_axes[ax].observe(self._on_slice_change, names='value')
            self.slice_offsets[ax].observe(self._on_slice_change, names='value')
        self.vec_dd.observe(self._on_vec_change, names='value')
        self.stride_dd.observe(self._on_stride_change, names='value')
        self.bg_dd.observe(self._on_appearance_change, names='value')
        self.edges_cb.observe(self._on_appearance_change, names='value')
        self.axes_cb.observe(self._on_appearance_change, names='value')
        self.sbar_cb.observe(self._on_appearance_change, names='value')
        self.vmin_field.observe(self._on_clim_change, names='value')
        self.vmax_field.observe(self._on_clim_change, names='value')
        self.reload_btn.on_click(self._on_reload)
        self.dispose_btn.on_click(self._on_dispose)
        self.paraview_btn.on_click(self._on_paraview)

        # Apply initial visibility for the default mode.
        self._apply_mode_visibility(self.mode_dd.value)

        # Populate the picker and title even before first expand so the
        # title count is accurate. No GL context created yet.
        self._refresh_file_options()
        self._refresh_title()

    @staticmethod
    def _labeled(label: str, widget, *, label_w: str = '60px') -> widgets.HBox:
        """Compact `[Label] [widget]` row used inside the toolbar HBoxes.

        Returns an HBox so the whole pair can be toggled hidden via
        ``.layout.display`` in one call.
        """
        return widgets.HBox(
            [widgets.Label(label, layout=widgets.Layout(width=label_w)),
             widget],
            layout=widgets.Layout(
                align_items='center', margin='0 12px 0 0',
            ),
        )

    # ----- collapse / expand -----

    def _on_toggle(self, _b):
        if self.is_visible:
            self._body.layout.display = 'none'
            self._refresh_title()
            return
        # Expand: refresh discovery first so a viz run done while the
        # panel was collapsed appears in the picker on open.
        self._body.layout.display = ''
        self._refresh_file_options()
        self._refresh_title()
        # Pre-flight hints so the user understands what (if anything)
        # they're about to see before paying the plotter init cost.
        if not self.main_gui.experiment_dir:
            self._set_status(
                'Load an experiment first to populate the file picker.',
                kind='info',
            )
            return
        if not self.file_picker.options:
            self._set_status(
                'No results_viz files yet for this experiment. Run '
                'Process Display, then click Reload (or expand again).',
                kind='info',
            )
            return
        # Lazy-construct the plotter on first expand. Failures are
        # surfaced in the status line (e.g. trame/pyvista not installed,
        # GL context creation fails on a headless host).
        if not self._ensure_plotter():
            return
        # Auto-load the first option if nothing is currently loaded.
        if self._current_file is None and self.file_picker.options:
            first_path = self.file_picker.options[0][1]
            self.file_picker.value = first_path  # triggers _on_file_change

    def _format_title(self, *, collapsed: bool) -> str:
        glyph = self._COLLAPSED_GLYPH if collapsed else self._EXPANDED_GLYPH
        return f'{glyph}  3D Viewer  {self._summary()}'

    def _refresh_title(self):
        # ipywidgets trait mutation; marshal to the main thread when
        # called from CoherenceGUI's Run All worker (DispTab.run_tab's
        # collapsed-viewer post-hook calls this directly).
        if not self._on_main_thread() and self._main_loop is not None:
            self._main_loop.add_callback(self._refresh_title)
            return
        self._title_btn.description = self._format_title(collapsed=not self.is_visible)

    def _summary(self) -> str:
        n = len(self.file_picker.options)
        if not self.main_gui.experiment_dir:
            return '(no experiment loaded)'
        if n == 0:
            return '(no results_viz files yet, run Process Display)'
        plural = 's' if n != 1 else ''
        return f'({n} file{plural})'

    # ----- file discovery -----

    def _refresh_file_options(self):
        """Re-glob ``results_viz*/`` and update the picker options.

        Preserves the current selection if it's still on disk.
        Triggered both on expand (main thread) and from
        ``DispTab.run_tab``'s post-hook (which may run on the Run All
        worker thread). When called off-thread we marshal back to the
        captured IOLoop so ipywidgets trait mutations don't race.

        Idempotent: if the discovered file set matches what the picker
        already has, the method returns without mutating anything.
        This is load-bearing for avoiding a frontend/backend race in
        the ipywidgets Selection widget, if the user clicks the
        dropdown while we're rewriting its options, the frontend's
        "set index N" message can arrive after the backend has fewer
        options and ``_Selection._validate_index`` raises ``TraitError:
        Invalid selection: index out of bounds``.
        """
        if not self._on_main_thread() and self._main_loop is not None:
            self._main_loop.add_callback(self._refresh_file_options)
            return
        exp_dir = self.main_gui.experiment_dir
        if not exp_dir:
            new_opts: list = []
        else:
            files = self._discover_vts(exp_dir)
            new_opts = [(os.path.basename(p), p) for p in files]

        # Bail when nothing changed, the common case during run_tab's
        # post-hook (files already in picker from prior expand). Tuple
        # compare because ipywidgets normalizes options to tuples.
        current_opts = list(self.file_picker.options)
        if [tuple(o) for o in current_opts] == [tuple(o) for o in new_opts]:
            return

        current = self._current_file
        with self._silence_observers():
            # Clear value to None FIRST so the new .options swap doesn't
            # try to validate the stale value against the new options,
            # and so the frontend's "select index N" messages in flight
            # land on a dropdown that explicitly has no selection.
            # Selection widgets accept None even when options are non-empty.
            try:
                self.file_picker.value = None
            except Exception:
                pass
            self.file_picker.options = new_opts
            if current and any(p == current for _, p in new_opts):
                self.file_picker.value = current
            elif new_opts:
                # Don't auto-load on a passive refresh; just show the
                # default highlighted so the user can click to load.
                # _on_toggle handles initial auto-load on first expand.
                self.file_picker.value = new_opts[0][1]

    @staticmethod
    def _discover_vts(exp_dir: str) -> list:
        """Recursive glob of ``results_viz*/`` under ``exp_dir``.

        Returns files ordered by ``_VTS_PRIORITY`` (canonical viz files
        first, generic catch-all last) with duplicates removed.

        Two subtleties:
          * ``glob.escape`` on ``exp_dir`` so user paths containing
            literal ``[``, ``?``, or ``*`` (e.g. ``scan[1].run/``)
            aren't interpreted as glob metacharacters and made to
            silently miss every file.
          * ``results_viz*`` (trailing wildcard) matches both
            single-peak ``results_viz/`` and multipeak
            ``results_viz_<hkl>/`` directory layouts.
        """
        seen = set()
        ordered = []
        safe_root = glob.escape(exp_dir)
        for pat in _VTS_PRIORITY:
            full_pat = os.path.join(safe_root, '**', 'results_viz*', pat)
            for path in sorted(glob.glob(full_pat, recursive=True)):
                if path in seen:
                    continue
                seen.add(path)
                ordered.append(path)
        return ordered

    # ----- plotter lifecycle -----

    def _ensure_plotter(self) -> bool:
        """Construct the Plotter + trame widget if not already up.

        Returns True iff the plotter is ready after the call.
        Failures (missing pyvista[jupyter], failed GL context) surface
        as a red status line; the panel remains expanded so the user
        can still use the ParaView fallback.
        """
        if self._plotter is not None and self._viewer_widget is not None:
            return True
        try:
            import pyvista
        except ImportError as e:
            self._set_status(
                f'PyVista not installed, install with '
                f'`pip install cohere-ui[jupyter]` (then restart kernel). '
                f'ParaView button still works. ({e})',
                kind='error',
            )
            return False

        # Prefer the trame-jupyter-extension routing path when the
        # package is installed (and Jupyter Server has it enabled).
        # That path proxies the trame WSGI through the existing Jupyter
        # server URL and never spawns a separate aiohttp instance, so
        # the wslink signal-handler crash is avoided entirely.
        # When the extension isn't available, fall back to the
        # signal-handler monkey-patch (the kernel still owns SIGINT).
        self._jupyter_extension_available = self._detect_trame_extension()
        if not self._jupyter_extension_available:
            self._patch_loop_signal_handlers_once()

        try:
            pyvista.set_jupyter_backend('trame')
        except Exception as e:
            self._set_status(
                f'Trame backend unavailable: {type(e).__name__}: {e}. '
                f'Use the ParaView button instead.',
                kind='error',
            )
            return False
        try:
            # off_screen=True keeps PyVista from allocating an
            # interactive NSWindow at construction time (which would
            # have to happen on the OS main thread on macOS, see the
            # mode='client' rationale below). notebook=True wires the
            # plotter to the trame embedding path so show_trame can
            # return an ipywidget. window_size matches the
            # viewer_output min_height roughly (4:3 aspect at 720 wide).
            self._plotter = pyvista.Plotter(
                notebook=True, off_screen=True, window_size=(720, 540),
            )
            self._plotter.background_color = 'white'
            self._plotter.add_axes()
        except Exception as e:
            self._set_status(
                f'Failed to construct PyVista plotter '
                f'({type(e).__name__}: {e}). On a headless Linux host '
                f'this usually means missing OSMesa/EGL. Use ParaView.',
                kind='error',
            )
            self._plotter = None
            return False
        try:
            from pyvista.trame.jupyter import show_trame
            # mode='client' renders entirely in the browser via vtk.js;
            # the kernel side only builds polydata (cheap, since we
            # contour to a thin iso surface before add_mesh) and ships
            # it to the browser. We MUST use client-side rendering on
            # macOS because the alternative ('server' / 'trame' modes)
            # fires a vtkWebApplication::StillRender call on a
            # trame worker thread, which calls vtkCocoaRenderWindow::
            # CreateAWindow() off the OS main thread and aborts with
            # SIGABRT (verified via crash report 2026-06-08-192148).
            # OSMesa would also avoid this, but it isn't bundled with
            # the PyPI vtk wheel; client mode is the cross-platform
            # fix for a notebook context.
            #
            # jupyter_extension_enabled routes the trame WSGI through
            # the existing Jupyter server when the extension package
            # is installed, cleaner URLs and no extra aiohttp server.
            # When not available, show_trame uses its built-in launch
            # path; the signal-handler monkey patch above covers that.
            self._viewer_widget = show_trame(
                self._plotter, mode='client',
                collapse_menu=True, default_server_rendering=False,
                jupyter_extension_enabled=self._jupyter_extension_available,
            )
        except Exception as e:
            self._set_status(
                f'Failed to embed trame viewer '
                f'({type(e).__name__}: {e}). Use ParaView.',
                kind='error',
            )
            try:
                self._plotter.close()
            except Exception:
                pass
            self._plotter = None
            return False
        # Mount the iframe into the Output slot. clear_output(wait=True)
        # absorbs the re-mount without flashing the prior content.
        with self.viewer_output:
            self.viewer_output.clear_output(wait=True)
            display(self._viewer_widget)
        return True

    # ----- file load / render -----

    def _load_file(self, path: str):
        if not self._ensure_plotter():
            return
        import pyvista
        try:
            new_mesh = pyvista.read(path)
        except Exception as e:
            self._set_status(
                f'Failed to read {os.path.basename(path)}: {type(e).__name__}: {e}',
                kind='error',
            )
            # Revert the picker to the prior good file so the user
            # isn't trapped on a broken selection (reload would just
            # re-attempt the same bad file otherwise). Suppress so
            # _on_file_change doesn't re-fire _load_file recursively.
            prior = self._current_file
            if prior and prior != path and os.path.isfile(prior):
                with self._silence_observers():
                    self.file_picker.value = prior
            return
        self._mesh = new_mesh
        self._current_file = path
        # Fresh data warrants fresh defaults; drop the user-touched
        # cmap / clim memory so the BCDI defaults (twilight for phase,
        # ±π clim, etc.) re-apply for the new file's scalars.
        self._user_cmap_dirty = False
        self._user_clim_dirty = False

        scalars = self._listable_scalars(self._mesh)
        iso_default = _PREFERRED_SCALAR if _PREFERRED_SCALAR in scalars else (
            scalars[0] if scalars else None)
        vectors = self._vector_arrays(self._mesh)
        mode = self.mode_dd.value
        color_default = self._default_color_for_mode(mode, scalars) or iso_default
        cmap_default = self._default_cmap_for_scalar(color_default)

        # Bulk-update controls inside a suppress block so the cascade
        # of observers fires only once at the end (via _render_current).
        with self._silence_observers():
            # Iso-from scalar (used to build the surface/slice). Setting
            # options before value avoids an index validation crash if
            # the prior value isn't in the new list.
            self.scalar_selector.options = scalars
            self.scalar_selector.value = iso_default
            # Color-by scalar (independent of iso scalar for single/multi
            # iso; same for slice; ignored for outline/glyphs).
            self.color_dd.options = scalars
            self.color_dd.value = color_default
            # Vector array dropdown for glyphs.
            self.vec_dd.options = vectors
            self.vec_dd.value = vectors[0] if vectors else None
            if cmap_default in self.cmap_selector.options:
                self.cmap_selector.value = cmap_default
            self._reset_iso_slider_for_scalar(iso_default, silent=True)
            self._reset_clim_for_scalar(color_default, silent=True)
        # If the file lacks any 3-component array, gray out glyph mode
        # to make the constraint obvious.
        self._refresh_mode_availability(vectors)
        # Reset camera preset on file load so a new geometry doesn't
        # inherit the prior file's framing.
        self._pending_camera = None
        self._render_current()
        if self._mesh is not None:
            dims = getattr(self._mesh, 'dimensions', None)
            shape = f' shape={dims}' if dims is not None else ''
            self._set_status(
                f'Loaded {os.path.basename(path)}{shape}  '
                f'iso={iso_default!r}  color={color_default!r}'
                f'{"  (vectors: " + ", ".join(vectors) + ")" if vectors else ""}',
                kind='success',
            )

    @staticmethod
    def _listable_scalars(mesh) -> list:
        """All point/cell scalars on ``mesh`` minus VTK-internal arrays.

        VTI files from ``InterpolationFeature`` carry ``vtkValidPointMask``
        and ``vtkGhostType`` housekeeping arrays that aren't meaningful
        to color by; hide them from the user. Also filters per-component
        suffixes (`Displacement_0`, `Displacement_1`, ...) that the
        Glyph filter sometimes adds, they're not meaningful by themselves
        and clutter the dropdown.
        """
        scalars = []
        if hasattr(mesh, 'point_data'):
            scalars += list(mesh.point_data.keys())
        if hasattr(mesh, 'cell_data'):
            scalars += list(mesh.cell_data.keys())
        # De-dup preserving order; drop vtk* internals.
        seen = set()
        ordered = []
        for s in scalars:
            if s in seen or s.startswith('vtk'):
                continue
            seen.add(s)
            ordered.append(s)
        return ordered

    @staticmethod
    def _vector_arrays(mesh) -> list:
        """3-component arrays suitable as glyph orientations.

        BCDI postprocess writes ``Displacement`` as a vector. Resolution
        and reciprocal-space files don't carry vectors; this returns []
        for them and the viewer disables the Glyph mode option in
        response (see _refresh_mode_availability).
        """
        import numpy as np
        out = []
        for ds in (getattr(mesh, 'point_data', None),
                   getattr(mesh, 'cell_data', None)):
            if ds is None:
                continue
            for name in ds.keys():
                if name.startswith('vtk') or name in out:
                    continue
                try:
                    arr = np.asarray(ds[name])
                except Exception:
                    continue
                if arr.ndim == 2 and arr.shape[1] == 3:
                    out.append(name)
        return out

    @staticmethod
    def _safe_data_range(mesh, scalar_name):
        """Return ``(min, max)`` of ``mesh[scalar_name]`` with NaN/Inf scrubbed.

        ``StructuredGrid.get_data_range`` raises on all-NaN or empty
        arrays; we want a graceful fallback so the multi-iso branch
        can show an outline + warning instead of crashing the render.
        """
        import numpy as np
        try:
            arr = np.asarray(mesh[scalar_name])
            if arr.dtype.kind == 'b':
                arr = arr.astype(float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return (0.0, 0.0)
            return (float(finite.min()), float(finite.max()))
        except Exception:
            return (0.0, 0.0)

    @staticmethod
    def _default_color_for_mode(mode: str, scalars: list):
        """Pick a sensible default color scalar per visualization mode.

        BCDI convention (Agent B): single/multi iso colors by amplitude
        (or phase if both present, since the iso is on amplitude and
        coloring by phase reveals dislocations). Slice mode defaults to
        phase because the wrapped phase is the standard 2D inspection
        signal. Glyphs default to None, they get colored by vector
        magnitude in _render_glyphs.
        """
        if not scalars:
            return None
        if mode == 'glyphs':
            return None
        # Slices: phase is the canonical slice color.
        if mode == 'slices' and 'imPh' in scalars:
            return 'imPh'
        # Single iso: prefer phase coloring if present (reveals
        # dislocations in the amplitude envelope); otherwise amplitude.
        if mode == 'single-iso' and 'imAmp' in scalars and 'imPh' in scalars:
            return 'imPh'
        if _PREFERRED_SCALAR in scalars:
            return _PREFERRED_SCALAR
        return scalars[0]

    def _outline_color(self) -> str:
        """Pick an outline color that contrasts with the current background.

        Black outline on a dark background is invisible, which left
        users staring at a blank iframe whenever a mode fell back to
        the outline (empty contour, no slice planes enabled, etc.).
        Bg dropdown values are CSS colors ('white', 'black', a hex);
        switch to white when the background looks dark.
        """
        bg = (self.bg_dd.value or 'white').lower()
        # Treat any of these as "dark enough that black would vanish".
        dark = bg in ('black', '#000', '#000000') or (
            bg.startswith('#') and len(bg) in (4, 7)
            and all(int(bg[i:i + (len(bg) // 3)], 16) < 96
                    for i in range(1, len(bg), len(bg) // 3))
        )
        return 'white' if dark else 'black'

    @staticmethod
    def _default_cmap_for_scalar(scalar) -> str:
        """Heuristic cmap default driven by scalar name.

        Phase-like arrays get the cyclic ``twilight`` colormap so the
        +pi/-pi wrap-around isn't visible as a hard discontinuity.
        Strain-like (and signed-residual-like) arrays get the diverging
        ``coolwarm``. Everything else gets the perceptually-uniform
        ``viridis``.
        """
        name = (scalar or '').lower()
        if 'ph' in name and 'amp' not in name:
            # 'imPh', 'recip_resPh', anything with 'phase' in it.
            return 'twilight'
        if 'strain' in name:
            return 'coolwarm'
        return 'viridis'

    def _resolved_clim(self):
        """Active (vmin, vmax) tuple from the Display-options fields.

        Returns ``None`` when fields are invalid (vmin >= vmax) so
        pyvista picks the data range itself.
        """
        try:
            vmin = float(self.vmin_field.value)
            vmax = float(self.vmax_field.value)
        except (TypeError, ValueError):
            return None
        if vmin < vmax:
            return (vmin, vmax)
        return None

    def _reset_clim_for_scalar(self, scalar, *, silent: bool = False):
        """Recompute vmin/vmax to ~q01/q99 of the active color scalar.

        Quantiles instead of full range so strain (and any other
        outlier-prone array) doesn't blow out the colormap. Called on
        file load + scalar change + Auto button. Phase arrays clamp to
        the standard symmetric range around 0 since the colormap is
        cyclic and quantiles would shift the wrap point.
        """
        if scalar is None or self._mesh is None:
            return
        import math
        import numpy as np
        try:
            arr = np.asarray(self._mesh[scalar])
            if arr.dtype.kind == 'b':
                arr = arr.astype(float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                lo, hi = 0.0, 1.0
            elif 'ph' in scalar.lower() and 'amp' not in scalar.lower():
                lo, hi = -math.pi, math.pi
            else:
                lo = float(np.quantile(finite, 0.01))
                hi = float(np.quantile(finite, 0.99))
                if hi <= lo:
                    lo = float(finite.min())
                    hi = float(finite.max())
                    if hi <= lo:
                        hi = lo + 1.0
        except Exception:
            lo, hi = 0.0, 1.0

        prior = self._suppress_observers
        if silent:
            self._suppress_observers = True
        try:
            self.vmin_field.value = lo
            self.vmax_field.value = hi
        finally:
            self._suppress_observers = prior

    @staticmethod
    def _auto_glyph_factor(mesh, vec_name) -> float:
        """Pick a glyph scale factor so the longest arrow is ~5 % of
        the smallest mesh extent.

        Glyph factor multiplies the vector before drawing the arrow,
        so if vectors are in voxel units (typical for Displacement in
        nm) and the mesh extent is hundreds of voxels, the right factor
        is around 0.05 * extent / max_magnitude. Falls back to 1.0 if
        anything goes wrong, safer than a black-screen overflow.
        """
        import numpy as np
        try:
            vec = np.asarray(mesh[vec_name])
            if vec.ndim != 2 or vec.shape[1] != 3:
                return 1.0
            mag = np.linalg.norm(vec, axis=1)
            mag_max = float(mag.max()) if mag.size else 0.0
            if mag_max <= 0:
                return 1.0
            b = np.asarray(mesh.bounds, dtype=float)
            extent = min(b[1] - b[0], b[3] - b[2], b[5] - b[4])
            if extent <= 0:
                return 1.0
            return float(0.05 * extent / mag_max)
        except Exception:
            return 1.0

    def _apply_camera(self, preset):
        """Apply a camera preset to the plotter, or auto-fit if None.

        Camera state lives only on the kernel side in client mode, we
        can't read back the user's manually-orbited camera from
        vtk.js. So every re-render reapplies whichever preset was
        last clicked (or auto-fit if no preset was set).
        """
        if self._plotter is None or self._mesh is None:
            return
        if preset is None or preset == 'Fit':
            try:
                self._plotter.reset_camera()
            except Exception:
                pass
            return
        import numpy as np
        center = np.asarray(self._mesh.center, dtype=float)
        b = np.asarray(self._mesh.bounds, dtype=float)
        extent = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
        if extent <= 0:
            extent = 1.0
        # 3x extent puts the camera safely outside the bounding box.
        dist = 3.0 * extent
        directions = {
            '+X': (1.0, 0.0, 0.0), '-X': (-1.0, 0.0, 0.0),
            '+Y': (0.0, 1.0, 0.0), '-Y': (0.0, -1.0, 0.0),
            '+Z': (0.0, 0.0, 1.0), '-Z': (0.0, 0.0, -1.0),
            'Iso': (1.0, 1.0, 1.0),
        }
        d = np.asarray(directions.get(preset, (1.0, 1.0, 1.0)), dtype=float)
        d = d / np.linalg.norm(d)
        # +Z is the conventional "up" except when the camera looks
        # straight down z, where +Y avoids a degenerate Up axis.
        up = (0.0, 0.0, 1.0) if abs(d[2]) < 0.99 else (0.0, 1.0, 0.0)
        pos = center + dist * d
        try:
            self._plotter.camera_position = [
                tuple(pos.tolist()),
                tuple(center.tolist()),
                up,
            ]
        except Exception:
            # Some pyvista builds expose the camera differently; let
            # reset_camera() pick a sane default rather than crashing.
            try:
                self._plotter.reset_camera()
            except Exception:
                pass

    def _reset_iso_slider_for_scalar(self, scalar, *, silent: bool = False):
        """Rescale the iso slider to the new scalar's [min, max] range.

        Initial value is 30% along the range, the standard BCDI
        default that shows the support envelope without the noise
        floor. ``silent`` skips re-rendering (caller does it once at
        the end of the bulk update).
        """
        if scalar is None or self._mesh is None:
            return
        try:
            import numpy as np
            arr = np.asarray(self._mesh[scalar])
            if arr.dtype.kind == 'b':
                arr = arr.astype(float)
            lo = float(arr.min())
            hi = float(arr.max())
        except Exception:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        step = max((hi - lo) / 200.0, 1e-9)
        prior_suppress = self._suppress_observers
        if silent:
            self._suppress_observers = True
        try:
            # Order matters: set min before max if new min > current max,
            # otherwise the trait validator clamps and we lose the range.
            self.iso_slider.max = max(hi, self.iso_slider.min)
            self.iso_slider.min = lo
            self.iso_slider.max = hi
            self.iso_slider.step = step
            self.iso_slider.value = lo + 0.3 * (hi - lo)
        finally:
            self._suppress_observers = prior_suppress

    def _render_current(self):
        """Mode-dispatched rebuild of the iframe view.

        Branches on ``self.mode_dd.value``: outline / single-iso /
        multi-iso / slices / glyphs. Each branch:
          1. Builds the appropriate polydata (contour / slice / glyph)
             from ``self._mesh`` (no GL render happens here, it's
             pure CPU/VTK pipeline work on the main thread).
          2. Adds it to the plotter with the shared color scalar,
             colormap, clim, and scalar-bar visibility.
        After dispatch, common appearance settings (background, axes,
        camera) are applied. Errors surface in the status line; the
        viewer never tears itself down on a per-render failure.

        ``clear_actors`` doesn't remove the orientation marker added
        by ``Plotter.show_axes()``, we re-call show_axes/hide_axes
        each render to honor the user's checkbox state.
        """
        if (self._plotter is None or self._mesh is None
                or self.scalar_selector.value is None):
            return

        mode = self.mode_dd.value
        iso_scalar = self.scalar_selector.value
        # Color-by may be None for glyphs (magnitude-coloured) or when
        # the color dropdown hasn't been populated yet; fall back to
        # the iso scalar so add_mesh has something to color by.
        # Also validate the chosen scalar still exists in the loaded
        # mesh, file swaps that share a scalar name with the prior
        # file are common, but unrelated files won't, and add_mesh
        # would raise a confusing "scalar not found" error otherwise.
        color_scalar = self.color_dd.value or iso_scalar
        if color_scalar is not None and color_scalar not in self._listable_scalars(self._mesh):
            color_scalar = iso_scalar
        cmap = self.cmap_selector.value
        clim = self._resolved_clim()
        opacity = float(self.opacity_slider.value)
        sbar = bool(self.sbar_cb.value)
        edges = bool(self.edges_cb.value)

        try:
            self._plotter.clear_actors()
            self._actor = None

            if mode == 'outline':
                self._actor = self._plotter.add_mesh(
                    self._mesh.outline(), color=self._outline_color(),
                )

            elif mode == 'single-iso':
                iso_level = float(self.iso_slider.value)
                contour = self._mesh.contour([iso_level], scalars=iso_scalar)
                if contour.n_points == 0:
                    self._actor = self._plotter.add_mesh(
                        self._mesh.outline(), color=self._outline_color(),
                    )
                    self._set_status(
                        f'Iso threshold {iso_level:.3g} produced an '
                        f'empty surface for {iso_scalar!r}; showing '
                        f'outline only.',
                        kind='warning',
                    )
                else:
                    self._actor = self._plotter.add_mesh(
                        contour, scalars=color_scalar,
                        cmap=cmap, clim=clim,
                        opacity=opacity, show_scalar_bar=sbar,
                        show_edges=edges,
                    )

            elif mode == 'multi-iso':
                import numpy as np
                rng = self._safe_data_range(self._mesh, iso_scalar)
                n = int(self.levels_dd.value)
                if rng[1] <= rng[0] or n <= 0:
                    self._actor = self._plotter.add_mesh(
                        self._mesh.outline(), color=self._outline_color(),
                    )
                    self._set_status(
                        f'Constant scalar {iso_scalar!r}; can\'t build '
                        f'multi-iso. Showing outline.',
                        kind='warning',
                    )
                else:
                    fractions = np.linspace(0.2, 0.8, n)
                    # Inner shell (high fraction = closer to peak amplitude)
                    # gets the higher opacity so the user can see the core;
                    # outer shells stay translucent so they don't occlude
                    # the inner core. The standard BCDI nested-iso look.
                    opacities = np.linspace(0.25, 0.85, n)
                    middle = n // 2
                    drew = False
                    for i, (f, op) in enumerate(zip(fractions, opacities)):
                        level = rng[0] + float(f) * (rng[1] - rng[0])
                        contour = self._mesh.contour([level], scalars=iso_scalar)
                        if contour.n_points == 0:
                            continue
                        actor = self._plotter.add_mesh(
                            contour, scalars=color_scalar,
                            cmap=cmap, clim=clim,
                            opacity=float(op),
                            show_scalar_bar=(sbar and i == middle),
                            show_edges=edges,
                        )
                        self._actor = actor
                        drew = True
                    if not drew:
                        self._actor = self._plotter.add_mesh(
                            self._mesh.outline(), color=self._outline_color(),
                        )
                        self._set_status(
                            'Multi-iso: every level produced empty '
                            'geometry; showing outline.', kind='warning',
                        )

            elif mode == 'slices':
                import numpy as np
                center = np.asarray(self._mesh.center, dtype=float)
                b = np.asarray(self._mesh.bounds, dtype=float)
                half = np.array([
                    (b[1] - b[0]) / 2.0,
                    (b[3] - b[2]) / 2.0,
                    (b[5] - b[4]) / 2.0,
                ])
                axis_normals = {'x': (1.0, 0.0, 0.0),
                                'y': (0.0, 1.0, 0.0),
                                'z': (0.0, 0.0, 1.0)}
                shown = []
                for i, ax in enumerate(('x', 'y', 'z')):
                    if not self.slice_axes[ax].value:
                        continue
                    offset = float(self.slice_offsets[ax].value)
                    origin = center + offset * half[i] * np.asarray(
                        axis_normals[ax], dtype=float)
                    try:
                        slc = self._mesh.slice(
                            normal=axis_normals[ax], origin=tuple(origin),
                        )
                    except Exception:
                        continue
                    if slc.n_points == 0:
                        continue
                    self._actor = self._plotter.add_mesh(
                        slc, scalars=color_scalar,
                        cmap=cmap, clim=clim,
                        show_scalar_bar=(sbar and not shown),
                        show_edges=edges,
                    )
                    shown.append(ax)
                if not shown:
                    self._actor = self._plotter.add_mesh(
                        self._mesh.outline(), color=self._outline_color(),
                    )
                    self._set_status(
                        'No slice planes enabled or all produced empty '
                        'geometry; showing outline.', kind='warning',
                    )

            elif mode == 'glyphs':
                vec_name = self.vec_dd.value
                if not vec_name:
                    self._actor = self._plotter.add_mesh(
                        self._mesh.outline(), color=self._outline_color(),
                    )
                    self._set_status(
                        'Glyph mode needs a 3-component vector array '
                        '(e.g., Displacement). None found in this file.',
                        kind='warning',
                    )
                else:
                    self._actor = self._render_glyphs(
                        vec_name, color_scalar, cmap, clim, sbar,
                    )

            # ----- common appearance + camera -----
            try:
                self._plotter.set_background(self.bg_dd.value)
            except Exception:
                pass
            try:
                if self.axes_cb.value:
                    self._plotter.show_axes()
                else:
                    self._plotter.hide_axes()
            except Exception:
                pass
            self._apply_camera(self._pending_camera)
            # Trigger a no-op render so the trame viewer sees the
            # actor list update. mode='client' makes this a Python-side
            # bookkeeping call, the actual frame is rendered in vtk.js
            # in the browser using the polydata add_mesh just shipped.
            try:
                self._plotter.render()
            except Exception:
                pass
        except Exception as e:
            import traceback
            self._set_status(
                f'Render failed: {type(e).__name__}: {e}',
                kind='error',
            )
            # Keep the panel functional; the failed mode can be swapped.

    def _render_glyphs(self, vec_name, color_scalar, cmap, clim, sbar):
        """Build a downsampled glyph field from the active vector array.

        Returns the actor (or None on failure). Refuses to render if
        the output would exceed ~200k cells, vtk.js's CPU rasterizer
        slows to a crawl past that point.
        """
        import numpy as np
        stride = max(int(self.stride_dd.value), 1)
        n_pts = self._mesh.n_points
        # Cap projected output: glyph factor expands each kept point
        # into an ~10-cell arrow, so refuse renders that would top 200k
        # cells. The user can lower the dropdown's stride.
        approx_kept = n_pts / stride
        if approx_kept * 10 > 200_000:
            self._set_status(
                f'Glyph mode: stride {stride} would produce '
                f'~{int(approx_kept * 10)} cells; raise the Stride '
                f'dropdown to keep the browser responsive.',
                kind='warning',
            )
            return self._plotter.add_mesh(
                self._mesh.outline(), color=self._outline_color(),
            )
        try:
            idx = np.arange(0, n_pts, stride, dtype=np.int64)
            sub = self._mesh.extract_points(idx, adjacent_cells=False,
                                             include_cells=False)
            if vec_name not in sub.array_names:
                # extract_points dropped the vector, sample it back.
                sub = sub.sample(self._mesh)
            factor = self._auto_glyph_factor(sub, vec_name)
            glyph = sub.glyph(orient=vec_name, scale=vec_name,
                              factor=factor, tolerance=None)
            if glyph.n_points == 0:
                self._set_status(
                    f'Glyph build for {vec_name!r} returned empty '
                    f'geometry; showing outline.', kind='warning',
                )
                return self._plotter.add_mesh(
                    self._mesh.outline(), color=self._outline_color(),
                )
            # Color the glyph by the magnitude of the vector array
            # (Agent B default). The glyph filter propagates the
            # vector; magnitude is just np.linalg.norm.
            try:
                vec = np.asarray(glyph[vec_name])
                if vec.ndim == 2 and vec.shape[1] == 3:
                    glyph['|vector|'] = np.linalg.norm(vec, axis=1)
                    return self._plotter.add_mesh(
                        glyph, scalars='|vector|',
                        cmap=cmap, clim=clim,
                        show_scalar_bar=sbar,
                    )
            except Exception:
                pass
            # Fallback: solid color glyphs (no scalar colouring).
            return self._plotter.add_mesh(glyph, color='steelblue')
        except Exception as e:
            self._set_status(
                f'Glyph build failed: {type(e).__name__}: {e}; '
                f'showing outline.', kind='warning',
            )
            return self._plotter.add_mesh(
                self._mesh.outline(), color=self._outline_color(),
            )

    # ----- mode + visibility helpers -----

    def _apply_mode_visibility(self, mode: str):
        """Hide/show per-mode control boxes based on the active mode.

        Outline shows none of the per-mode rows (it's a pure preview).
        Single-iso shows iso/opacity/cmap/color. Multi-iso replaces
        iso with levels. Slices replaces iso+opacity with the axis
        toggles. Glyphs replaces iso+opacity+color with vector+stride
        (glyphs are colored by vector magnitude).
        """
        def _show(box, on):
            box.layout.display = '' if on else 'none'

        is_iso = mode == 'single-iso'
        is_multi = mode == 'multi-iso'
        is_slices = mode == 'slices'
        is_glyphs = mode == 'glyphs'
        is_outline = mode == 'outline'

        # Render row pieces.
        _show(self._iso_box, is_iso)
        _show(self._opacity_box, is_iso)
        _show(self._cmap_box, not is_outline)
        # Conditional row pieces.
        _show(self._color_box, is_iso or is_multi or is_slices)
        _show(self._levels_box, is_multi)
        _show(self._slices_box, is_slices)
        _show(self._glyphs_box, is_glyphs)

    def _refresh_mode_availability(self, vectors: list):
        """Disable the Glyphs option when no vector arrays are present.

        ipywidgets Dropdown doesn't support per-option-disable, so we
        strip the option entirely when unsupported; restore it when
        a vector-bearing file loads.
        """
        # Capture user's current selection so we don't bump them off it.
        current_mode = self.mode_dd.value
        has_vectors = bool(vectors)
        # Rebuild options list. The full set is the same as in __init__.
        full = [
            ('Single iso', 'single-iso'),
            ('Multi-iso',  'multi-iso'),
            ('Slice planes', 'slices'),
            ('Glyphs (vectors)', 'glyphs'),
            ('Outline only', 'outline'),
        ]
        new_opts = [(l, v) for (l, v) in full
                    if v != 'glyphs' or has_vectors]
        # Bail when nothing changed, skip ipywidgets churn.
        if list(self.mode_dd.options) == [(l, v) for l, v in new_opts]:
            return
        # If current mode disappeared (glyphs without vectors), fall
        # back to single-iso.
        target = current_mode
        if not any(v == target for _, v in new_opts):
            target = 'single-iso'
        with self._silence_observers():
            self.mode_dd.value = None
            self.mode_dd.options = new_opts
            self.mode_dd.value = target
        if target != current_mode:
            self._apply_mode_visibility(target)

    # ----- observers / handlers -----

    def _on_file_change(self, change):
        if self._suppress_observers:
            return
        new = change.get('new')
        if new and new != self._current_file:
            self._load_file(new)

    def _on_scalar_change(self, change):
        if self._suppress_observers:
            return
        new = change.get('new')
        if new:
            self._reset_iso_slider_for_scalar(new, silent=True)
            self._render_current()

    def _on_iso_change(self, _change):
        if self._suppress_observers:
            return
        self._render_current()

    def _on_opacity_change(self, change):
        if self._suppress_observers:
            return
        # Single-iso has exactly one actor whose opacity we can poke
        # directly. Multi-iso / slices / glyphs / outline use multiple
        # actors (or fixed per-level opacity for multi-iso) so they
        # need a full rebuild to honor the slider.
        if (self.mode_dd.value == 'single-iso'
                and self._actor is not None and self._plotter is not None):
            try:
                self._actor.GetProperty().SetOpacity(float(change['new']))
                self._plotter.render()
                return
            except Exception:
                pass
        self._render_current()

    def _on_cmap_change(self, _change):
        if self._suppress_observers:
            return
        # Mark the cmap as user-touched so subsequent mode / color
        # cascades don't blow away the explicit choice. Cleared on
        # file load and on explicit Auto.
        self._user_cmap_dirty = True
        self._render_current()

    def _on_mode_change(self, change):
        """Mode change cascade: visibility update, default-color repick,
        optional cmap nudge, then a re-render.

        The default color / cmap repick on every mode change matches the
        BCDI convention (phase on slices, amplitude on iso, magnitude on
        glyphs). Skips when no file is loaded yet so the user's
        explicit pre-load selection isn't clobbered.
        """
        if self._suppress_observers:
            return
        new_mode = change.get('new')
        self._apply_mode_visibility(new_mode)
        if self._mesh is None:
            return
        scalars = self._listable_scalars(self._mesh)
        new_color = self._default_color_for_mode(new_mode, scalars)
        new_cmap = self._default_cmap_for_scalar(new_color)
        with self._silence_observers():
            if new_color and new_color in self.color_dd.options:
                self.color_dd.value = new_color
            # Honor user-touched cmap / clim: only nudge defaults when
            # the user hasn't explicitly overridden them. The user can
            # re-apply defaults with the Auto button or by loading a
            # new file.
            if not self._user_cmap_dirty and new_cmap in self.cmap_selector.options:
                self.cmap_selector.value = new_cmap
            if not self._user_clim_dirty:
                self._reset_clim_for_scalar(new_color, silent=True)
        # Surface that the camera will snap back if the user had
        # manually orbited in the browser (we can't read that back
        # from vtk.js client rendering).
        if self._pending_camera is not None:
            self._set_status(
                f'Camera held at preset {self._pending_camera!r} on mode swap '
                f'(manual browser orbit not preserved).',
                kind='info',
            )
        self._render_current()

    def _on_color_change(self, change):
        if self._suppress_observers:
            return
        new = change.get('new')
        if new:
            # Honor user-touched cmap / clim: only nudge defaults
            # when the user hasn't explicitly overridden them. The
            # Auto button and file load reset both dirty flags.
            new_cmap = self._default_cmap_for_scalar(new)
            with self._silence_observers():
                if not self._user_cmap_dirty and new_cmap in self.cmap_selector.options:
                    self.cmap_selector.value = new_cmap
                if not self._user_clim_dirty:
                    self._reset_clim_for_scalar(new, silent=True)
        self._render_current()

    def _on_levels_change(self, _change):
        if self._suppress_observers:
            return
        if self.mode_dd.value == 'multi-iso':
            self._render_current()

    def _on_slice_change(self, _change):
        if self._suppress_observers:
            return
        if self.mode_dd.value == 'slices':
            self._render_current()

    def _on_vec_change(self, _change):
        if self._suppress_observers:
            return
        if self.mode_dd.value == 'glyphs':
            self._render_current()

    def _on_stride_change(self, _change):
        if self._suppress_observers:
            return
        if self.mode_dd.value == 'glyphs':
            self._render_current()

    def _on_appearance_change(self, _change):
        """Background / edges / axes / scalar-bar toggles.

        Edges and scalar-bar visibility are baked into add_mesh, so
        any change requires a full re-render. Background and axes
        could be applied incrementally, but routing through
        _render_current keeps state in one place at a small cost
        (no polydata rebuild, just a swap of the existing actors'
        property bag).
        """
        if self._suppress_observers:
            return
        self._render_current()

    def _on_clim_change(self, _change):
        if self._suppress_observers:
            return
        # Mark clim as user-touched so subsequent mode / color cascades
        # don't recompute q01/q99 over the typed values. Cleared on
        # file load and on explicit Auto.
        self._user_clim_dirty = True
        self._render_current()

    def _on_auto_clim(self, _b):
        """Recompute vmin/vmax from the current color scalar (q01/q99)."""
        if self._mesh is None:
            self._set_status('Load a file before requesting Auto clim.',
                             kind='info')
            return
        scalar = self.color_dd.value or self.scalar_selector.value
        if scalar is None:
            self._set_status('No color scalar selected.', kind='warning')
            return
        # _reset_clim_for_scalar already silences observers internally
        # via silent=True (which sets _suppress_observers within the
        # helper). Explicit Auto click clears the user-dirty flag too,
        # this IS the user asking for the default to be re-applied.
        self._user_clim_dirty = False
        self._reset_clim_for_scalar(scalar, silent=True)
        self._render_current()
        self._set_status(
            f'Clim auto-set to q01/q99 of {scalar!r} '
            f'[{self.vmin_field.value:.3g}, {self.vmax_field.value:.3g}].',
            kind='info',
        )

    def _on_camera_preset(self, preset: str):
        """Camera preset button handler.

        Stores the preset name so subsequent re-renders (e.g. after a
        mode swap) keep the same view; calls _apply_camera + render
        immediately for instant feedback.
        """
        if self._plotter is None or self._mesh is None:
            self._set_status(
                'Camera presets require a loaded file.', kind='info',
            )
            return
        # 'Fit' resets to auto-fit so subsequent renders don't keep
        # snapping back to a preset view the user may have orbited away
        # from in the browser (even though we can't read that orbit).
        self._pending_camera = None if preset == 'Fit' else preset
        self._apply_camera(self._pending_camera)
        try:
            self._plotter.render()
        except Exception:
            pass
        self._set_status(f'Camera: {preset}', kind='info')

    def _on_disp_toggle(self, _b):
        """Show/hide the Display options sub-panel.

        Persists across re-renders (it's a UI state, not viewer state)
        but resets to collapsed on dispose, matches the outer viewer's
        idiom of starting collapsed and remembering subsequent toggles
        within the session.
        """
        self._disp_open = not self._disp_open
        self._disp_body.layout.display = '' if self._disp_open else 'none'
        glyph = self._EXPANDED_GLYPH if self._disp_open else self._COLLAPSED_GLYPH
        self._disp_toggle_btn.description = f'{glyph}  Display options'

    def _on_reload(self, _b):
        self.reload()

    def _on_dispose(self, _b):
        self.dispose()
        self._set_status('Disposed. Expand again to reload.', kind='info')

    def _on_paraview(self, _b):
        path = self._current_file or (self.file_picker.value or None)
        if not path:
            self._set_status('Pick a file first.', kind='warning')
            return
        ok, msg = open_in_paraview(path)
        self._set_status(msg, kind='success' if ok else 'error')

    # ----- public reload / dispose -----

    def reload(self):
        """Rescan ``results_viz*/`` and re-read the current file in place.

        Called by DispTab after a successful Process Display run (only
        when ``is_visible``) and by the Reload button. The plotter is
        kept across calls; only the dataset + actor are refreshed.

        ``DispTab.run_tab`` may invoke this from the CoherenceGUI
        Run All worker thread; touching VTK or ipywidgets traits
        off-thread is unsafe (the same constraint motivated the
        ``IOLoop.add_callback`` pattern in ``rec_subprocess/monitor.py``).
        Self-marshal to the captured kernel IOLoop in that case.
        """
        if not self._on_main_thread() and self._main_loop is not None:
            self._main_loop.add_callback(self.reload)
            return
        self._refresh_file_options()
        self._refresh_title()
        if self._current_file and os.path.isfile(self._current_file):
            self._load_file(self._current_file)
        elif self.file_picker.options:
            # Current file vanished (e.g., results_viz wiped + re-run);
            # auto-load the new default.
            self.file_picker.value = self.file_picker.options[0][1]

    def dispose(self):
        """Tear down the plotter and trame widget.

        Recipe for pyvista 0.46.x (verified with the trame_vtk/trame_vuetify
        versions in this venv):
          1. ``viewer_widget.close()``: releases the ipywidget comm.
          2. ``pyvista.trame.ui._VIEWERS.pop(plotter._id_name)``:
             removes the trame Viewer wrapper that ``show_trame`` /
             ``get_viewer`` register on construction. Without this pop
             the Viewer object outlives the Plotter and pins the
             render-window reference graph, so dispose+re-expand cycles
             grow the kernel by one Viewer per cycle.
          3. ``plotter.close()``: tears down the VTK render window,
             scalar bars, and actor references.
        Done early-out on each step so a partial construction (e.g.
        plotter created but show_trame failed) still cleans up.
        """
        if self._viewer_widget is not None:
            try:
                self._viewer_widget.close()
            except Exception:
                pass
            self._viewer_widget = None
        if self._plotter is not None:
            # Pop the trame Viewer wrapper BEFORE closing the plotter
            # so the pop key (plotter._id_name) is still readable.
            try:
                from pyvista.trame.ui import _VIEWERS
                _VIEWERS.pop(self._plotter._id_name, None)
            except Exception:
                # Registry not present in this pyvista release, or
                # _id_name attr removed, non-fatal.
                pass
            try:
                self._plotter.close()
            except Exception:
                pass
            self._plotter = None
        self._mesh = None
        self._actor = None
        self._current_file = None
        self._pending_camera = None
        # Collapse the Display-options sub-panel so the next expand
        # starts from a clean visual state (matches outer viewer
        # collapsing on dispose).
        self._disp_open = False
        self._disp_body.layout.display = 'none'
        self._disp_toggle_btn.description = f'{self._COLLAPSED_GLYPH}  Display options'
        # Clear the iframe slot so the user sees the panel is empty.
        with self.viewer_output:
            self.viewer_output.clear_output()

    # ----- helpers -----

    @contextmanager
    def _silence_observers(self):
        """Suppress observer cascades during programmatic bulk updates.

        Replacing ``dropdown.options`` then ``dropdown.value`` would
        fire the value observer twice (once with the stale value, once
        with the new). With suppression, the caller is responsible for
        triggering whatever follow-up render is needed once at the end.
        """
        prior = self._suppress_observers
        self._suppress_observers = True
        try:
            yield
        finally:
            self._suppress_observers = prior

    def _set_status(self, message: str, kind: str = 'info'):
        """Single-line status above the iframe.

        ``kind`` is one of ``info`` / ``success`` / ``warning`` /
        ``error`` and drives the color + prefix.
        """
        palette = {
            'info': '#444',
            'success': '#1e7a1e',
            'warning': '#a06000',
            'error': '#a02020',
        }
        prefix = {
            'success': '[OK] ',
            'warning': '[WARN] ',
            'error': '[ERROR] ',
        }.get(kind, '')
        color = palette.get(kind, '#444')
        weight = 'font-weight:600;' if kind in ('error', 'success') else ''
        self.status.value = (
            f'<span style="color:{color}; font-size:12px; {weight}'
            f'font-family:Menlo, Consolas, monospace;">'
            f'{html.escape(prefix + str(message))}'
            f'</span>'
        )
