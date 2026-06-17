"""RecTab: reconstruction configuration form.

Subprocess management, queue listening, progress / log / snapshot /
error-history rendering all live in ``..rec_subprocess``. This tab owns
the config form (algorithm sequence, betas, features) and wires the
Run / Stop buttons through ``RecMonitor``.
"""

import ast
import html as _html
import importlib.util
import os
import shutil
import sys
from contextlib import nullcontext as _nullcontext

import ipywidgets as widgets

import cohere_core.utilities as ut

from cohere_ui.jupyter_gui._validation import (
    ValidationError as _ValidationError,
    parse_algorithm_sequence,
    validate_device_field,
)
from cohere_ui.jupyter_gui.tabs.base import BaseTab, _MSG
from cohere_ui.jupyter_gui.widgets import (
    FeaturePanel, PathChooser, button, dropdown, form_row, text_field,
)
from cohere_ui.jupyter_gui.rec_subprocess import RecMonitor
from cohere_ui.jupyter_gui.rec_subprocess.progress import total_iters_from_alg_sequence
import traceback


_MAIN_REC_ID_LABEL = 'main'  # User-facing label for the default config_rec.

from cohere_ui.jupyter_gui.utils.device_info import (
    list_devices, format_devices, parse_device_field,
)
from cohere_ui.jupyter_gui.utils.error_format import format_error_summary
from cohere_ui.jupyter_gui.text import load_text

_UI = load_text('ui_strings')
_URLS = load_text('urls')


class RecTab(BaseTab):
    """Tab for reconstruction configuration.

    Supports named alternative rec configurations: a
    ``rec_id`` dropdown above the form selects which of
    ``config_rec`` / ``config_rec_<id>`` the widgets read from and
    write to. The "+" button creates a new id by copying the current
    config. The currently-selected id is also passed through to
    ``manage_reconstruction(rec_id=...)``, so the subprocess writes its
    output under ``results_phasing_<id>/`` instead of
    ``results_phasing/``.
    """

    name = "Reconstruction"
    # conf_name is a property below to support config_rec_<id> dynamically.
    # Flip to True once the backend honors the 'precision' config key.
    _PRECISION_ENABLED = False

    def __init__(self):
        super().__init__()
        # '' = the canonical 'config_rec' (UI label: 'main'); anything
        # else maps to 'config_rec_<id>'.
        self._rec_id: str = ''

    @property
    def conf_name(self) -> str:
        """Effective config file name based on the selected rec id."""
        return 'config_rec' if not self._rec_id else f'config_rec_{self._rec_id}'

    def _output_dirname(self) -> str:
        """Reconstruction output directory for the active rec id.

        'main' writes to results_phasing/, a named id to
        results_phasing_<id>/.
        """
        return 'results_phasing' if not self._rec_id else f'results_phasing_{self._rec_id}'

    def _refresh_rec_id_status(self) -> None:
        """Show which config file is active and where its output will land."""
        if not hasattr(self, 'rec_id_status'):
            return
        label = _MAIN_REC_ID_LABEL if not self._rec_id else self._rec_id
        self.rec_id_status.value = (
            f'<span style="font-size:12px;color:var(--jup-fg-muted);">Editing '
            f'<code>conf/{_html.escape(self.conf_name)}</code> '
            f'(<b>{_html.escape(label)}</b>) | output to '
            f'<code>{_html.escape(self._output_dirname())}/</code></span>'
        )

    def _build_ui(self) -> widgets.Widget:
        self.init_guess = dropdown(
            options=['random', 'continue', 'AI algorithm'],
            value='random'
        )
        # Colored-dot status beside Initial Guess; only shown for AI algorithm.
        self.ai_status = widgets.HTML(
            layout=widgets.Layout(width='180px', margin='0 0 0 8px')
        )
        self.init_guess_params = widgets.VBox()
        self.init_guess.observe(self._on_init_guess_change, 'value')

        # List all backends; the status dot shows which are available on this
        # machine. An unavailable pick is refused at Run time with a clear message.
        proc_options = ['auto', 'np', 'cp', 'torch']
        self.proc = dropdown(options=proc_options, value='auto')
        self.proc_status = widgets.HTML(
            layout=widgets.Layout(width='180px', margin='0 0 0 8px')
        )
        self.device = text_field(placeholder='e.g., [0] or "all"')
        self.device_hint = widgets.HTML(value='')
        self.reconstructions = text_field(placeholder='e.g., 1')

        # Precision: torch supports the full sweep (complex32/64/128); numpy.fft
        # has no complex32 and cupy.fft's float16 path is too restrictive, so
        # those backends only offer 'auto' / float64 / float32.
        # bf16/fp8 are intentionally NOT exposed: torch.fft has no complex
        # dtype paired to either, so the FFT path would silently upcast and
        # the only "savings" would be the static intensity tensor, not worth
        # the additional surface area.
        self._precision_full = ('auto', 'float64', 'float32', 'float16')
        self._precision_np_cp = ('auto', 'float64', 'float32')
        self.precision = dropdown(options=self._precision_full, value='auto')
        self.proc.observe(self._on_proc_change, 'value')

        self.alg_seq = text_field(placeholder='e.g., 3*(20*ER+180*HIO)+20*ER', width='300px')
        self.hio_beta = text_field(placeholder='0.9')
        self.raar_beta = text_field(placeholder='0.45')
        self.initial_support_area = text_field(placeholder='[0.5, 0.5, 0.5]')

        self.defaults_btn = button('Set Defaults', style='info', width='120px', role='info')
        self.defaults_btn.on_click(lambda b: self._set_defaults_guarded())

        from cohere_ui.jupyter_gui.features import get_rec_features
        self.features = {name: cls() for name, cls in get_rec_features().items()}
        self.feature_panel = FeaturePanel(self.features)

        self.action_row = self._build_action_row(run_label='Save and Run', run_width='150px')
        self.stop_btn = button('Stop', style='danger', width='80px')
        self.stop_btn.disabled = True
        self.stop_btn.on_click(lambda b: self._on_stop_guarded())

        self.monitor = RecMonitor()
        self.monitor.on_running_changed = self._on_running_changed
        self.monitor.on_finished = self._on_rec_finished
        # log_panel stays None; logging routes through the monitor instead.

        proc_row = widgets.HBox([self.proc, self.proc_status])
        init_guess_row = widgets.HBox([self.init_guess, self.ai_status])
        params_section = widgets.VBox([
            form_row('Initial Guess', init_guess_row),
            self.init_guess_params,
            form_row('Processor', proc_row),
            form_row('Device(s)', self.device),
            self.device_hint,
            *([form_row('Precision', self.precision)] if self._PRECISION_ENABLED else []),
            form_row('Reconstructions', self.reconstructions),
            form_row('Algorithm Sequence', self.alg_seq),
            form_row('HIO Beta', self.hio_beta),
            form_row('RAAR Beta', self.raar_beta),
            form_row('Initial Support Area', self.initial_support_area),
            self.defaults_btn,
        ])
        # Apply initial backend-precision restriction + render initial status.
        self._apply_precision_options(self.proc.value)
        self._update_proc_status(self.proc.value)
        # Probe devices once at tab load; live-highlight matches as the user
        # edits the Device(s) field (no refresh button per design).
        self._device_list = list_devices(
            self._resolve_backend(self.proc.value)[1]
        )
        self.device.observe(self._on_device_change, 'value')
        self._refresh_device_hint()
        self._apply_device_enabled(self.proc.value)

        # Named alternative rec configurations row. Hidden until the
        # user creates the first non-'main' id.
        self.rec_id_dropdown = dropdown(
            options=[_MAIN_REC_ID_LABEL], value=_MAIN_REC_ID_LABEL, width='180px',
        )
        self.rec_id_dropdown.observe(self._on_rec_id_change, 'value')
        self.new_rec_id_input = text_field(
            placeholder='new id (letters / digits / _)', width='220px',
        )
        self.add_rec_id_btn = button(
            '+ Add config', style='info', width='130px', role='info',
        )
        self.add_rec_id_btn.on_click(lambda b: self._on_add_rec_id())
        self.rec_id_row = widgets.HBox(
            [
                widgets.HTML(
                    '<span style="margin-right:6px;">Configuration</span>'
                ),
                self.rec_id_dropdown,
                widgets.HTML('<span style="margin:0 6px 0 12px;">|</span>'),
                self.new_rec_id_input,
                self.add_rec_id_btn,
            ],
            layout=widgets.Layout(align_items='center', margin='4px 0 4px 0'),
        )
        # The dropdown is meaningful even when only 'main' exists (it
        # advertises the feature); leave it visible.
        # Status line: which config file is being edited and where its
        # reconstruction output will land. Refreshed on every switch / load.
        self.rec_id_status = widgets.HTML(
            layout=widgets.Layout(margin='0 0 8px 0'),
        )
        self._refresh_rec_id_status()

        return widgets.VBox([
            self.rec_id_row,
            self.rec_id_status,
            params_section,
            # FeaturePanel draws its own "Features (k/N active)" header.
            self.feature_panel.widget,
            # width:100% so the flex-growing action row (with its status
            # strip) fills the gap and Stop stays pinned to the right.
            widgets.HBox(
                [self.action_row, self.stop_btn],
                layout=widgets.Layout(width='100%', align_items='center'),
            ),
            self.monitor.widgets_box(),
        ])

    # Named alternative rec configs

    def _discover_rec_ids(self) -> list:
        """Return the sorted list of '<id>' names found in ``conf/config_rec_*``.

        Delegates to :meth:`ConfigManager.discover_rec_ids` (which excludes
        the canonical ``config_rec`` and any ``*_backup`` copy). Returns an
        empty list when no experiment is loaded.
        """
        if self.main_gui is None:
            return []
        return self.main_gui.config_manager.discover_rec_ids()

    def refresh_rec_id_options(self) -> None:
        """Repopulate the rec_id dropdown from on-disk config_rec_* files.

        Preserves the current selection when the corresponding file
        still exists; falls back to 'main' otherwise.
        """
        if not hasattr(self, 'rec_id_dropdown'):
            return
        ids = self._discover_rec_ids()
        options = [_MAIN_REC_ID_LABEL] + ids
        # Re-assigning .options unobserved would still fire the
        # observer for the value transition; suppress with a manual
        # unobserve/reobserve so the switch handler doesn't think the
        # user picked something new during a programmatic refresh.
        self.rec_id_dropdown.unobserve(self._on_rec_id_change, 'value')
        try:
            self.rec_id_dropdown.options = options
            target = self._rec_id or _MAIN_REC_ID_LABEL
            self.rec_id_dropdown.value = (
                target if target in options else _MAIN_REC_ID_LABEL
            )
            if self.rec_id_dropdown.value == _MAIN_REC_ID_LABEL:
                self._rec_id = ''
        finally:
            self.rec_id_dropdown.observe(self._on_rec_id_change, 'value')

    @BaseTab._guard
    def _on_rec_id_change(self, change):
        """Save the previous selection's widget state, then load the new one.

        Best-effort save: a validation error on the OLD config still
        allows the switch, because the user wants to look at the NEW config.
        """
        if self.main_gui is None or not self.main_gui.experiment_exists():
            return
        new_label = change['new']
        # Persist the outgoing config under its old name first. _rec_id
        # still holds the previous selection here, so save_and_verify
        # writes to the right file; best-effort (a validation error on the
        # old config must not block looking at the new one).
        self.save_and_verify()
        # Now switch. Capture the target so we can restore it after
        # clear_conf() wipes _rec_id below.
        target = '' if new_label == _MAIN_REC_ID_LABEL else new_label
        self._rec_id = target
        # Load the destination config; if it doesn't exist (deleted on
        # disk between refresh and switch), fall back silently to main.
        conf_map = self.main_gui.config_manager.load_config(self.conf_name)
        if conf_map is None:
            self.monitor.log(
                f'config file for id {new_label!r} is missing; staying on main',
                level='warning',
            )
            self._rec_id = ''
            self._refresh_rec_id_status()
            return
        self.clear_conf()
        # clear_conf() reset _rec_id to '' and the dropdown to 'main';
        # restore the target so load_tab, which calls
        # refresh_rec_id_options, re-selects it instead of bouncing the
        # dropdown back to 'main'.
        self._rec_id = target
        self.load_tab(conf_map)
        self._notify_save()  # refresh the status strip / titles
        self._refresh_rec_id_status()
        self.monitor.log(
            f'Now editing conf/{self.conf_name}; output goes to '
            f'{self._output_dirname()}/.',
            level='info',
        )
        # Auto-follow: point the Postprocess tab at the same reconstruction.
        self._follow_disp(self._rec_id)

    def _follow_disp(self, rec_id: str) -> None:
        """Tell the Postprocess (Disp) tab to follow this reconstruction id.

        One-way notification (disp never calls back). Best-effort: a
        missing or broken disp tab is logged at debug, never raised.
        """
        main_gui = self.main_gui
        if main_gui is None:
            return
        disp = getattr(main_gui, '_tabs', {}).get('disp')
        if disp is None or not hasattr(disp, 'update_tab'):
            return
        try:
            disp.update_tab(rec_id=rec_id)
        except Exception as e:
            self.monitor.log(f'disp follow skipped: {e}', level='debug')

    @BaseTab._guard
    def _on_add_rec_id(self):
        """Create config_rec_<new_id> by copying the current config.

        New id is read from the inline text field; validated against
        existing names and the filesystem regex. On success the
        dropdown is refreshed and the new id becomes the active
        selection (so subsequent edits target the new file).
        """
        if self.main_gui is None or not self.main_gui.experiment_exists():
            self.monitor.log('No experiment loaded; cannot create a rec config.',
                             level='warning')
            return
        raw = (self.new_rec_id_input.value or '').strip()
        if not raw:
            self.monitor.log('Enter a new id in the text field first.',
                             level='warning')
            return
        # Restrict to filesystem-safe characters to avoid path tricks.
        if not all(ch.isalnum() or ch == '_' for ch in raw):
            self.monitor.log(
                f'New id {raw!r}: use only letters, digits, and underscores.',
                level='error',
            )
            return
        if raw == _MAIN_REC_ID_LABEL:
            self.monitor.log(f'{_MAIN_REC_ID_LABEL!r} is reserved.', level='error')
            return
        if raw in self._discover_rec_ids():
            self.monitor.log(f'rec id {raw!r} already exists.', level='error')
            return

        # Save the current widget state to its file first, so the copy
        # reflects the latest edits.
        prior = self.save_and_verify()
        if prior:
            self.monitor.log(
                'Validation failed; fix the current config before adding a new id.',
                level='error',
            )
            return

        conf_dir = self.main_gui.config_manager.conf_dir
        src = ut.join(conf_dir, 'config_rec')
        dst = ut.join(conf_dir, f'config_rec_{raw}')
        if not os.path.isfile(src):
            # No main config yet: write an empty placeholder so
            # subsequent loads succeed. The dropdown change will load
            # it back as an empty dict.
            ut.write_config({}, dst)
        else:
            shutil.copyfile(src, dst)
        self.new_rec_id_input.value = ''
        self.refresh_rec_id_options()
        # Switch to the new id (the observer was just re-attached by
        # refresh_rec_id_options; setting .value triggers it).
        self.rec_id_dropdown.value = raw
        self.monitor.log(f'Created config_rec_{raw} (active).', level='success')

    def _refresh_device_hint(self):
        selected = parse_device_field(self.device.value, self._device_list)
        self.device_hint.value = format_devices(self._device_list, selected=selected)
        # Flag typos / unknown device ids before Run so the user doesn't wait
        # on a subprocess that crashes immediately.
        text = (self.device.value or '').strip()
        if not text or text == 'all':
            return
        if selected:
            return
        err = validate_device_field(text, self._device_list or [])
        if err is not None:
            self.monitor.log(str(err), level='warning')

    def _apply_device_enabled(self, proc_value):
        """Grey out Device(s) when np is selected; field value is preserved
        so anything downstream that requires a 'device' key still gets one."""
        self.device.disabled = (proc_value == 'np')

    @BaseTab._guard
    def _on_device_change(self, _change):
        self._refresh_device_hint()

    @BaseTab._guard
    def _on_proc_change(self, change):
        self._apply_precision_options(change['new'])
        self._update_proc_status(change['new'])
        self._apply_device_enabled(change['new'])

    @staticmethod
    def _resolve_ai_runtime():
        """Return (available, reason) for the AI init_guess runtime.

        Checks tensorflow presence without importing it (importing tensorflow
        pulls in CUDA libs and is slow). Presence on sys.path does not
        guarantee a successful import, so _run_tab_impl re-checks at Run time.
        """
        try:
            present = importlib.util.find_spec('tensorflow') is not None
        except (ImportError, ValueError):
            present = False
        if present:
            return True, ''
        return False, (
            'tensorflow is not installed. Run '
            '`pip install tensorflow`, then restart the Jupyter kernel'
        )

    @staticmethod
    def _resolve_backend(name):
        """Resolve a Processor selection to (available, resolved_name, reason).

        Mirrors api.common.get_pkg without invoking it (avoids importing the
        cohere_ui workflow module just to render a UI hint). 'auto' tries the
        same chain get_pkg does: cupy, then torch, then numpy.
        """
        def _have(mod):
            # Probe without importing. On macOS, importing torch loads its
            # bundled libomp.dylib and later collides with xrayutilities'
            # libomp during postprocessing (Fatal Python error: Aborted).
            try:
                return importlib.util.find_spec(mod) is not None
            except (ImportError, ValueError):
                return False

        if name == 'np':
            return True, 'np', ''
        if name == 'cp':
            if sys.platform == 'darwin':
                return False, 'cp', 'cupy is not supported on macOS'
            if not _have('cupy'):
                return False, 'cp', 'cupy is not installed'
            return True, 'cp', ''
        if name == 'torch':
            if not _have('torch'):
                return False, 'torch', 'torch is not installed'
            return True, 'torch', ''
        if name == 'auto':
            if sys.platform != 'darwin' and _have('cupy'):
                return True, 'cp', ''
            if _have('torch'):
                return True, 'torch', ''
            return True, 'np', ''
        return False, name, 'unknown backend'

    def _update_proc_status(self, proc_value):
        available, resolved, reason = self._resolve_backend(proc_value)
        color = 'var(--jup-success)' if available else 'var(--jup-error)'
        if proc_value == 'auto':
            label = f'auto: {resolved}'
            tooltip = f'auto-resolves to {resolved} on this machine'
        elif available:
            label = resolved
            tooltip = f'{resolved} backend is available'
        else:
            label = f'{resolved} unavailable'
            tooltip = reason
        self.proc_status.value = (
            f'<span title="{tooltip}" style="line-height:24px;">'
            f'<span style="color:{color}; font-size:14px;">&#9679;</span>'
            f' {label}</span>'
        )

    def _apply_precision_options(self, proc_value):
        """Restrict the precision dropdown to options the chosen backend can run.

        numpy.fft does not support complex32 and cupy.fft's float16 path is too
        restrictive for general FFT shapes, so 'float16' is removed when the
        Processor is fixed to 'np' or 'cp'. 'auto' keeps the full set since the
        runtime backend isn't pinned yet.
        """
        if proc_value in ('np', 'cp'):
            allowed = self._precision_np_cp
        else:
            allowed = self._precision_full
        if self.precision.options != allowed:
            current = self.precision.value
            self.precision.options = allowed
            self.precision.value = current if current in allowed else 'auto'

    @BaseTab._guard
    def _on_init_guess_change(self, change):
        guess = change['new']
        self.init_guess_params.children = []
        # Only show the tensorflow status dot when the AI path is selected.
        if guess == 'AI algorithm':
            self._update_ai_status()
        else:
            self.ai_status.value = ''
        if guess == 'continue':
            self.cont_dir = PathChooser(
                kind='dir',
                placeholder=_UI['placeholders']['prev_recon_dir'],
                width='350px',
            )
            self.init_guess_params.children = [
                form_row('Continue Directory', self.cont_dir.widget),
            ]
        elif guess == 'AI algorithm':
            # Same PathChooser used by InstrTab's specfile / data_dir fields.
            self.ai_model = PathChooser(
                kind='file',
                placeholder=_UI['placeholders']['ai_model_path'],
                width='350px',
            )

            # Link to the pre-trained .hdf5 model on the Globus distribution.
            # FontAwesome `fa-download` is monochrome and inherits the
            # link color; the previous Unicode downward-arrow glyph rendered
            # as a color emoji on macOS/Windows and clashed with the rest of
            # the GUI's FontAwesome iconography (copy, folder, etc.).
            download_link = widgets.HTML(
                value=(
                    f'<a href="{_URLS["pretrained_ai"]}" '
                    f'target="_blank" rel="noopener" '
                    f'download="cohere-trained_model.hdf5" '
                    f'title="{_UI["tooltips"]["ai_model_download"]}" '
                    f'style="margin-left:8px; font-size:12px;">'
                    f'<i class="fa fa-download" '
                    f'style="margin-right:4px;"></i>'
                    f'Download .hdf5</a>'
                ),
                layout=widgets.Layout(margin='4px 0 0 0'),
            )

            self.init_guess_params.children = [
                form_row(
                    'AI Model File',
                    widgets.HBox(
                        [self.ai_model.widget, download_link],
                        layout=widgets.Layout(align_items='center'),
                    ),
                ),
            ]

    def _update_ai_status(self) -> None:
        """Render the tensorflow availability dot beside Initial Guess.

        Colored dot + label, with the full reason in the tooltip.
        """
        if not hasattr(self, 'ai_status'):
            return
        available, reason = self._resolve_ai_runtime()
        color = 'var(--jup-success)' if available else 'var(--jup-error)'
        if available:
            label = 'tensorflow'
            tooltip = (
                'tensorflow is importable. Import may still fail if '
                'CUDA/ROCm libraries are missing.'
            )
        else:
            label = 'tensorflow unavailable'
            tooltip = reason
        self.ai_status.value = (
            f'<span title="{tooltip}" style="line-height:24px;">'
            f'<span style="color:{color}; font-size:14px;">&#9679;</span>'
            f' {label}</span>'
        )

    @BaseTab._guard
    def _set_defaults_guarded(self):
        self._set_defaults()

    @BaseTab._guard
    def _on_stop_guarded(self):
        self._on_stop()

    def _set_defaults(self):
        self.reconstructions.value = '1'
        self.proc.value = 'auto'
        self.device.value = '[0]'
        self.precision.value = 'auto'
        self.alg_seq.value = '3*(20*ER+180*HIO)+20*ER'
        self.hio_beta.value = '.9'
        self.raar_beta.value = '.45'
        self.initial_support_area.value = '[0.5, 0.5, 0.5]'

    def load_tab(self, conf_map: dict):
        # Discover any named alternative configs sitting in conf/ so
        # the dropdown reflects what's actually on disk. The current
        # _rec_id is preserved when its file is still present.
        self.refresh_rec_id_options()
        init_guess = conf_map.get('init_guess', 'random')
        if init_guess == 'random':
            self.init_guess.value = 'random'
        elif init_guess == 'continue':
            self.init_guess.value = 'continue'
            if 'continue_dir' in conf_map and hasattr(self, 'cont_dir'):
                self.cont_dir.value = conf_map['continue_dir'].replace('\\', '/')
        elif init_guess == 'AI_guess':
            self.init_guess.value = 'AI algorithm'
            if 'AI_trained_model' in conf_map and hasattr(self, 'ai_model'):
                self.ai_model.value = conf_map['AI_trained_model'].replace('\\', '/')

        if 'processing' in conf_map:
            self.proc.value = conf_map['processing']
        if 'device' in conf_map:
            self.device.value = self._fmt_value(conf_map['device'])
        if 'precision' in conf_map and conf_map['precision'] in self.precision.options:
            self.precision.value = conf_map['precision']
        else:
            self.precision.value = 'auto'
        if 'reconstructions' in conf_map:
            self.reconstructions.value = str(conf_map['reconstructions'])
        if 'algorithm_sequence' in conf_map:
            self.alg_seq.value = str(conf_map['algorithm_sequence'])
        if 'hio_beta' in conf_map:
            self.hio_beta.value = str(conf_map['hio_beta'])
        if 'raar_beta' in conf_map:
            self.raar_beta.value = str(conf_map['raar_beta'])
        if 'initial_support_area' in conf_map:
            self.initial_support_area.value = self._fmt_value(conf_map['initial_support_area'])

        self.feature_panel.init_configs(conf_map)
        self._refresh_rec_id_status()

    def get_config(self) -> dict:
        conf_map = {}

        if self.init_guess.value == 'continue':
            conf_map['init_guess'] = 'continue'
            if hasattr(self, 'cont_dir') and self.cont_dir.value:
                conf_map['continue_dir'] = self.cont_dir.value
        elif self.init_guess.value == 'AI algorithm':
            conf_map['init_guess'] = 'AI_guess'
            if hasattr(self, 'ai_model') and self.ai_model.value:
                conf_map['AI_trained_model'] = self.ai_model.value

        if self.proc.value:
            conf_map['processing'] = self.proc.value
        if self._PRECISION_ENABLED and self.precision.value and self.precision.value != 'auto':
            conf_map['precision'] = self.precision.value
        if self.device.value:
            dev = self.device.value.strip()
            if dev == 'all':
                conf_map['device'] = dev
            else:
                conf_map['device'] = self._parse_field('device', dev)
        if self.reconstructions.value:
            conf_map['reconstructions'] = self._parse_field('reconstructions', self.reconstructions.value)
        if self.alg_seq.value:
            conf_map['algorithm_sequence'] = self.alg_seq.value.strip()
        if self.hio_beta.value:
            conf_map['hio_beta'] = self._parse_field('hio_beta', self.hio_beta.value)
        if self.raar_beta.value:
            conf_map['raar_beta'] = self._parse_field('raar_beta', self.raar_beta.value)
        if self.initial_support_area.value:
            conf_map['initial_support_area'] = self._parse_field('initial_support_area', self.initial_support_area.value)

        self.feature_panel.add_configs(conf_map)
        return conf_map

    def clear_conf(self):
        self.init_guess.value = 'random'
        self.init_guess_params.children = []
        self.proc.value = 'auto'
        self.device.value = ''
        self.precision.value = 'auto'
        self.reconstructions.value = ''
        self.alg_seq.value = ''
        self.hio_beta.value = ''
        self.raar_beta.value = ''
        self.initial_support_area.value = ''
        self.feature_panel.clear_all()
        # Reset rec-id state to 'main' on a full clear. The dropdown
        # options are refreshed on the next load_tab.
        self._rec_id = ''
        if hasattr(self, 'rec_id_dropdown'):
            self.rec_id_dropdown.unobserve(self._on_rec_id_change, 'value')
            try:
                self.rec_id_dropdown.options = [_MAIN_REC_ID_LABEL]
                self.rec_id_dropdown.value = _MAIN_REC_ID_LABEL
            finally:
                self.rec_id_dropdown.observe(self._on_rec_id_change, 'value')
        self._refresh_rec_id_status()

    def _wire_status_line(self):
        """RecTab logs through its monitor, not a LogPanel, so mirror the
        monitor's appended lines into the status strip."""
        if self.status_label is not None:
            self.monitor.on_append = self._set_status

    def clear_output(self):
        self.monitor.clear_log()
        self._reset_status()

    def log(self, message: str):
        self.monitor.log(message)

    # RecTab has no log_panel; route the level helpers through the
    # monitor so BaseTab.save_and_verify (and any inherited code that
    # calls self.log_error / log_warning / log_info / log_debug) lands
    # with the right severity in the monitor's log widget.
    def log_info(self, message: str):
        self.monitor.log(message, level='info')

    def log_success(self, message: str):
        self.monitor.log(message, level='success')

    def log_warning(self, message: str):
        self.monitor.log(message, level='warning')

    def log_error(self, message: str):
        self.monitor.log(message, level='error')

    def log_debug(self, message: str):
        self.monitor.log(message, level='debug')

    def run_tab(self, skip_save: bool = False):
        """Validate config, save, and ask the monitor to start the subprocess."""
        try:
            self._run_tab_impl(skip_save=skip_save)
        except Exception as e:
            self.monitor.log(format_error_summary(e, prefix='run_tab'), level='error')
            self.monitor.log(traceback.format_exc(), level='debug')
            self.monitor.progress_label.value = '<i>Idle (error)</i>'
            self._on_running_changed(False)

    def _run_tab_impl(self, *, skip_save: bool = False):
        if self.monitor.is_running:
            self.monitor.log(_MSG['rec']['already_running'])
            return

        # Refuse early if the selected Processor isn't available, to skip the
        # 30 s subprocess spin-up that would crash on the cupy/torch import.
        # Logging at 'error' triggers the auto-reveal so the user sees it.
        ok, _resolved, reason = self._resolve_backend(self.proc.value)
        if not ok:
            self.monitor.log(
                f"Cannot run: {reason}. Pick a different Processor.",
                level='error',
            )
            return

        # busy() blocks re-entry while the filesystem walk and feature checks run.
        busy_cm = self.split_run.busy('Validating...') if self.split_run else _nullcontext()
        with busy_cm:
            err = self._validate_experiment()
            if err:
                self.monitor.log(err)
                return

            # Parse the algorithm sequence here so a typo surfaces inline
            # ("did you mean 'ER'?") instead of crashing the subprocess later.
            if self.alg_seq.value:
                parsed = parse_algorithm_sequence(self.alg_seq.value)
                if isinstance(parsed, _ValidationError):
                    self.monitor.log(
                        _MSG['tab']['config_error'].format(error=str(parsed)),
                        level='error',
                    )
                    return

            # AI guess requires tensorflow (imported by cohere_core.controller.AI_guess).
            # Fail fast with an install hint rather than crashing 30 s into the subprocess.
            if self.init_guess.value == 'AI algorithm':
                if importlib.util.find_spec('tensorflow') is None:
                    self.monitor.log(
                        "init_guess='AI algorithm' selected but tensorflow "
                        "is not installed in this environment. Install it "
                        "first: `pip install tensorflow` (then restart the "
                        "Jupyter kernel).",
                        level='error',
                    )
                    return
                model_path = (
                    self.ai_model.value.strip()
                    if hasattr(self, 'ai_model') and self.ai_model.value
                    else ''
                )
                if not model_path or not os.path.isfile(model_path):
                    self.monitor.log(
                        f"init_guess='AI algorithm' requires AI_trained_model "
                        f"to point at an existing model file (.keras or .hdf5)"
                        f"{f' (got: {model_path!r})' if model_path else ''}. "
                        f"Use the Download .hdf5 link to grab the public "
                        f"trained model. On Python 3.11+ the .hdf5 file must "
                        f"first be converted with "
                        f"`python tools/convert_ai_model.py SRC.hdf5 DST.keras` "
                        f"(see tools/convert_ai_model.py for the rationale).",
                        level='error',
                    )
                    return

            found = any(
                'data.tif' in f or 'data.npy' in f
                for _, _, f in os.walk(self.main_gui.experiment_dir)
            )
            if not found:
                self.monitor.log(_MSG['rec']['no_input_data'])
                return

        if skip_save:
            self.monitor.log(_MSG['tab']['run_modified_warning'], level='warning')
        else:
            # save_and_verify (BaseTab) reads the config, validates the
            # fields, runs the cohere_core verifier, and saves. Errors are
            # already logged via self.log_error (routed to the monitor);
            # a non-empty return means we should abort.
            if self.save_and_verify():
                return

        progress_feature = self.features.get('progress')
        show_bar_widget = getattr(progress_feature, 'show_progress_bar', None) if progress_feature else None
        progress_active = bool(progress_feature.active.value) if progress_feature else False
        show_bar = bool(show_bar_widget.value) if show_bar_widget else True

        self._pre_run_snapshot = self._snapshot_outputs()

        run_kwargs = {
            'no_verify': self.main_gui.no_verify,
            'debug': True,
        }
        # Pass the selected named rec id through to manage_reconstruction
        # so output lands in results_phasing_<id>/ instead of
        # results_phasing/. None signals the canonical 'main' config.
        if self._rec_id:
            run_kwargs['rec_id'] = self._rec_id

        self.monitor.start(
            experiment_dir=self.main_gui.experiment_dir,
            backend_cfg=self._collect_backend_config(),
            kwargs=run_kwargs,
            total_iters=total_iters_from_alg_sequence(self.alg_seq.value),
            show_progress_bar=progress_active and show_bar,
        )

    def _on_stop(self):
        self.monitor.stop()

    def _on_running_changed(self, running: bool):
        if self.split_run is not None:
            self.split_run.set_enabled(not running)
        if self.save_button is not None:
            self.save_button.widget.disabled = running or self.save_button.widget.disabled
        if self.load_btn is not None:
            self.load_btn.disabled = running
        self.stop_btn.disabled = not running

    def _on_rec_finished(self, exit_code):
        try:
            self.main_gui.results.reload()
        except Exception as e:
            self.monitor.log(f'results reload failed: {e}')
        if exit_code == 0:
            try:
                results = self.main_gui.results
                self.monitor.show_final_snapshot(
                    results.image, results.support, results.errors,
                    self._collect_backend_config(),
                )
            except Exception as e:
                self.monitor.log(f'final snapshot: results unavailable ({e})')
        # Log files written during the run regardless of exit_code so the
        # user can see partial output from a killed/errored reconstruction.
        before = getattr(self, '_pre_run_snapshot', None)
        if before is not None:
            self._log_file_changes(before)
            self._pre_run_snapshot = None
        # Auto-follow: repoint the Postprocess tab at the reconstruction
        # that just produced output.
        self._follow_disp(self._rec_id)

    def _collect_backend_config(self):
        """Build the picklable BackendConfig dict for the wrapper subprocess.

        Reads from the LiveFeature widgets when 'live' is active; returns
        ``None`` when the user hasn't enabled live view (subprocess then runs
        without registering a backend, so cohere falls through to its default
        MatplotlibBackend; harmless under Agg in the subprocess).
        """
        live = self.features.get('live')
        if live is None or not getattr(live, 'active', None) or not live.active.value:
            return None
        renderer = getattr(live, 'renderer', None)
        renderer_value = renderer.value if renderer is not None else 'matplotlib_2d'
        if renderer_value == 'pyvista_static':
            return {
                'kind': 'pyvista',
                'stride': self._read_int(getattr(live, 'stride', None), 4),
                'iso_level': self._read_float(getattr(live, 'iso_level', None), 0.3),
            }
        mask_widget = getattr(live, 'apply_support_mask', None)
        cfg = {
            'kind': 'matplotlib',
            'phase_cmap': getattr(getattr(live, 'phase_cmap', None), 'value', 'twilight'),
            'apply_support_mask': bool(mask_widget.value) if mask_widget else True,
        }
        method_widget = getattr(live, 'slice_method', None)
        slice_method = method_widget.value if method_widget else 'center_of_mass'
        if renderer_value == 'matplotlib_3d_mosaic':
            cfg['mode'] = 'strided_3d'
            cfg['slice_method'] = slice_method
        else:
            cfg['mode'] = 'center_slice'
            axis_widget = getattr(live, 'slice_axis', None)
            cfg['slice_axis'] = self._read_int(axis_widget, 2) if axis_widget else 2
            cfg['slice_method'] = slice_method
        return cfg

    @staticmethod
    def _read_int(widget, default):
        if widget is None:
            return default
        try:
            return int(widget.value)
        except (TypeError, ValueError, AttributeError):
            return default

    @staticmethod
    def _read_float(widget, default):
        if widget is None:
            return default
        try:
            return float(widget.value)
        except (TypeError, ValueError, AttributeError):
            return default
