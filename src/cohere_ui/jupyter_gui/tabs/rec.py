"""RecTab: reconstruction configuration form.

Subprocess management, queue listening, progress / log / snapshot /
error-history rendering all live in ``..rec_subprocess``. This tab owns
the config form (algorithm sequence, betas, features) and wires the
Run / Stop buttons through ``RecMonitor``.
"""

import ast
import os
import sys
import traceback

import ipywidgets as widgets

from .base import BaseTab, _MSG
from ..widgets import form_row, text_field, dropdown, button, FeaturePanel
from ..rec_subprocess import RecMonitor
from ..rec_subprocess.progress import total_iters_from_alg_sequence


class RecTab(BaseTab):
    """Tab for reconstruction configuration."""

    name = "Reconstruction"
    conf_name = "config_rec"

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
        self.device = text_field(placeholder='e.g., [0] or "all"')
        self.reconstructions = text_field(placeholder='e.g., 1')

        self.alg_seq = text_field(placeholder='e.g., 3*(20*ER+180*HIO)+20*ER', width='300px')
        self.hio_beta = text_field(placeholder='0.9')
        self.raar_beta = text_field(placeholder='0.45')
        self.initial_support_area = text_field(placeholder='[0.5, 0.5, 0.5]')

        self.defaults_btn = button('Set Defaults', style='info', width='120px', role='info')
        self.defaults_btn.on_click(lambda b: self._set_defaults())

        from ..features import REC_FEATURES
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

        params_section = widgets.VBox([
            form_row('Initial Guess', self.init_guess),
            self.init_guess_params,
            form_row('Processor', self.proc),
            form_row('Device(s)', self.device),
            form_row('Reconstructions', self.reconstructions),
            form_row('Algorithm Sequence', self.alg_seq),
            form_row('HIO Beta', self.hio_beta),
            form_row('RAAR Beta', self.raar_beta),
            form_row('Initial Support Area', self.initial_support_area),
            self.defaults_btn,
        ])

        return widgets.VBox([
            params_section,
            widgets.HTML('<h4>Features</h4>'),
            self.feature_panel.widget,
            widgets.HBox([self.load_btn, self.run_btn, self.stop_btn]),
            self.monitor.widgets_box(),
        ])

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
        if self.device.value:
            dev = self.device.value.strip()
            if dev == 'all':
                conf_map['device'] = dev
            else:
                conf_map['device'] = ast.literal_eval(dev)
        if self.reconstructions.value:
            conf_map['reconstructions'] = ast.literal_eval(self.reconstructions.value)
        if self.alg_seq.value:
            conf_map['algorithm_sequence'] = self.alg_seq.value.strip()
        if self.hio_beta.value:
            conf_map['hio_beta'] = ast.literal_eval(self.hio_beta.value)
        if self.raar_beta.value:
            conf_map['raar_beta'] = ast.literal_eval(self.raar_beta.value)
        if self.initial_support_area.value:
            conf_map['initial_support_area'] = ast.literal_eval(self.initial_support_area.value)

        self.feature_panel.add_configs(conf_map)
        return conf_map

    def clear_conf(self):
        self.init_guess.value = 'random'
        self.init_guess_params.children = []
        self.proc.value = 'auto'
        self.device.value = ''
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
            self.monitor.log(f'run_tab error: {e}')
            self.monitor.log(traceback.format_exc())
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
                'debug': self.main_gui.debug,
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
