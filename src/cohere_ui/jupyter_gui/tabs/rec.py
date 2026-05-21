"""RecTab: reconstruction configuration form.

Subprocess management, queue listening, progress / log / snapshot /
error-history rendering all live in ``..rec_subprocess``. This tab owns
the config form (algorithm sequence, betas, features) and wires the
Run / Stop buttons through ``RecMonitor``.
"""

import ast
import os
import sys

import ipywidgets as widgets

from cohere_ui.jupyter_gui.tabs.base import BaseTab, _MSG
from cohere_ui.jupyter_gui.widgets import form_row, text_field, dropdown, button, FeaturePanel
from cohere_ui.jupyter_gui.rec_subprocess import RecMonitor
from cohere_ui.jupyter_gui.rec_subprocess.progress import total_iters_from_alg_sequence
import traceback

from cohere_ui.jupyter_gui.device_info import (
    list_devices, format_devices, parse_device_field,
)
from cohere_ui.jupyter_gui.error_format import format_error_summary


class RecTab(BaseTab):
    """Tab for reconstruction configuration."""

    name = "Reconstruction"
    conf_name = "config_rec"
    # Flip to True once the backend honors the 'precision' config key.
    _PRECISION_ENABLED = False

    def _build_ui(self) -> widgets.Widget:
        self.init_guess = dropdown(
            options=['random', 'continue', 'AI algorithm'],
            value='random'
        )
        self.init_guess_params = widgets.VBox()
        self.init_guess.observe(self._on_init_guess_change, 'value')

        proc_options = ['auto', 'np', 'torch']
        if sys.platform != 'darwin':
            proc_options.insert(1, 'cp')
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
        self.defaults_btn.on_click(lambda b: self._set_defaults())

        from cohere_ui.jupyter_gui.features import REC_FEATURES
        self.features = {name: cls() for name, cls in REC_FEATURES.items()}
        self.feature_panel = FeaturePanel(self.features)

        self.load_btn = button('Load Config', style='warning', width='120px', role='load')
        self.run_btn = button('Run Reconstruction', style='success', width='150px', role='run')
        self.stop_btn = button('Stop', style='danger', width='80px')
        self.stop_btn.disabled = True
        self.load_btn.on_click(lambda b: self._load_config_dialog())
        self.run_btn.on_click(lambda b: self.run_tab())
        self.stop_btn.on_click(lambda b: self._on_stop())

        self.monitor = RecMonitor()
        self.monitor.on_running_changed = self._on_running_changed
        self.monitor.on_finished = self._on_rec_finished
        # log_panel stays None; logging routes through the monitor instead.

        proc_row = widgets.HBox([self.proc, self.proc_status])
        params_section = widgets.VBox([
            form_row('Initial Guess', self.init_guess),
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

        return widgets.VBox([
            params_section,
            widgets.HTML('<h4>Features</h4>'),
            self.feature_panel.widget,
            widgets.HBox([self.load_btn, self.run_btn, self.stop_btn]),
            self.monitor.widgets_box(),
        ])

    def _refresh_device_hint(self):
        selected = parse_device_field(self.device.value, self._device_list)
        self.device_hint.value = format_devices(self._device_list, selected=selected)

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
    def _resolve_backend(name):
        """Resolve a Processor selection to (available, resolved_name, reason).

        Mirrors api.common.get_pkg without invoking it (avoids importing the
        cohere_ui workflow module just to render a UI hint). 'auto' tries the
        same chain get_pkg does: cupy, then torch, then numpy.
        """
        def _have(mod):
            try:
                __import__(mod)
                return True
            except Exception:
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
        color = '#2e7d32' if available else '#c62828'
        if proc_value == 'auto':
            label = f'auto &rarr; {resolved}'
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
        if guess == 'continue':
            self.cont_dir = text_field(placeholder='Path to continue directory')
            self.init_guess_params.children = [form_row('Continue Directory', self.cont_dir)]
        elif guess == 'AI algorithm':
            self.ai_model = text_field(placeholder='Path to trained model .hdf5')
            self.init_guess_params.children = [form_row('AI Model File', self.ai_model)]

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

    def clear_output(self):
        self.monitor.clear_log()

    def log(self, message: str):
        self.monitor.log(message)

    def run_tab(self):
        """Validate config, save, and ask the monitor to start the subprocess."""
        try:
            self._run_tab_impl()
        except Exception as e:
            self.monitor.log(format_error_summary(e, prefix='run_tab'), level='error')
            self.monitor.log(traceback.format_exc(), level='debug')
            self.monitor.progress_label.value = '<i>Idle (error)</i>'
            self._on_running_changed(False)

    def _run_tab_impl(self):
        if self.monitor.is_running:
            self.monitor.log(_MSG['rec']['already_running'])
            return

        err = self._validate_experiment()
        if err:
            self.monitor.log(err)
            return

        found = any(
            'data.tif' in f or 'data.npy' in f
            for _, _, f in os.walk(self.main_gui.experiment_dir)
        )
        if not found:
            self.monitor.log(_MSG['rec']['no_input_data'])
            return

        for feat in self.features.values():
            if not feat.active.value:
                continue
            err_msg = feat.verify_active()
            if err_msg:
                self.monitor.log(_MSG['rec']['feature_error'].format(error=err_msg))
                return

        conf_map = self.get_config()
        if not conf_map:
            return

        err = self.main_gui.config_manager.verify(self.conf_name, conf_map)
        if err and not self.main_gui.no_verify:
            self.monitor.log(_MSG['tab']['config_error'].format(error=err))
            return

        _, action = self.main_gui.config_manager.save_config(
            self.conf_name, conf_map, self.main_gui.no_verify)
        if action:
            self._log_config_action(action)

        progress_feature = self.features.get('progress')
        show_bar_widget = getattr(progress_feature, 'show_progress_bar', None) if progress_feature else None
        progress_active = bool(progress_feature.active.value) if progress_feature else False
        show_bar = bool(show_bar_widget.value) if show_bar_widget else True

        self._pre_run_snapshot = self._snapshot_outputs()

        self.monitor.start(
            experiment_dir=self.main_gui.experiment_dir,
            backend_cfg=self._collect_backend_config(),
            kwargs={
                'no_verify': self.main_gui.no_verify,
                'debug': True,
            },
            total_iters=total_iters_from_alg_sequence(self.alg_seq.value),
            show_progress_bar=progress_active and show_bar,
        )

    def _on_stop(self):
        self.monitor.stop()

    def _on_running_changed(self, running: bool):
        self.run_btn.disabled = running
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
