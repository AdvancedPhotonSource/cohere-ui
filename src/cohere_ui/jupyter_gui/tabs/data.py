"""DataTab: data formatting configuration and processing."""

import os
import traceback

import ipywidgets as widgets

from cohere_ui.jupyter_gui.tabs.base import BaseTab, _MSG
from cohere_ui.jupyter_gui.widgets import form_row, text_field, dropdown, checkbox, button, LogPanel
from cohere_ui.jupyter_gui.viewers.tiff_viewer import TiffViewer
from cohere_ui.jupyter_gui.utils.error_format import format_error_summary
from cohere_ui.jupyter_gui.text import load_text

_UI = load_text('ui_strings')


class DataTab(BaseTab):
    """Tab for data formatting configuration.

    Handles alien removal, intensity threshold, binning, shift, crop/pad.
    """

    name = "Standard Prep"
    conf_name = "config_data"

    # Keys the form exposes. Other keys loaded from disk round-trip
    # through _passthrough so save_conf keeps backend-set fields like ``pkg``.
    _KNOWN_KEYS = frozenset({
        'alien_alg', 'aliens', 'alien_file',
        'AA1_size_threshold', 'AA1_asym_threshold', 'AA1_min_pts',
        'AA1_eps', 'AA1_amp_threshold', 'AA1_save_arrs',
        'AA1_expandcleanedsigma',
        'auto_intensity_threshold', 'intensity_threshold',
        'binning', 'shift', 'crop_pad', 'no_center_max',
    })

    def __init__(self):
        super().__init__()
        self._passthrough: dict = {}

    def _build_ui(self) -> widgets.Widget:
        # Alien algorithm selection
        self.alien_alg = dropdown(
            options=['none', 'block aliens', 'alien file', 'AutoAlien1'],
            value='none'
        )
        self.alien_params_box = widgets.VBox()
        self.alien_alg.observe(self._on_alien_change, 'value')

        # Standard fields
        self.auto_intensity_threshold = checkbox('auto intensity threshold')
        self.intensity_threshold = text_field(placeholder='e.g., 2.5')
        self.shift = text_field(placeholder='e.g., [0, 0, 0]')
        self.crop_pad = text_field(placeholder='e.g., [0, 0, 0, 0, 0, 0]')
        self.binning = text_field(placeholder='e.g., [1, 1, 1]')
        self.no_center_max = checkbox('not center max')

        self.action_row = self._build_action_row(run_label='Format Data', run_width='140px')

        self.log_panel = LogPanel(height='150px')

        self.tiff_viewer = TiffViewer(
            panes=[
                {
                    'key': 'compare',
                    'label': 'Comparison source',
                    'placeholder': 'dir / file / glob -- or use Quick load below',
                    'default_path': lambda: self._prep_default_path(),
                    'shortcuts': [
                        ('Raw scan', lambda: self._raw_default_path() or ''),
                        ('Prep Data', lambda: self._prep_default_path() or ''),
                    ],
                },
                {
                    'key': 'data',
                    'label': 'Data tab output (phasing_data/data.tif)',
                    'placeholder': 'dir / file / glob (default: phasing_data/data.tif)',
                    'default_path': lambda: self._data_default_path(),
                },
            ],
            title='TIFF Viewer',
            initial_visible=False,
            log_debug=self.log_debug,
        )

        return widgets.VBox([
            form_row('Alien Algorithm', self.alien_alg),
            self.alien_params_box,
            self.auto_intensity_threshold,
            form_row('Intensity Threshold', self.intensity_threshold),
            form_row('Shift', self.shift),
            form_row('Crop/Pad', self.crop_pad),
            form_row('Binning', self.binning),
            self.no_center_max,
            self.action_row,
            self.log_panel.widget,
            widgets.HTML('<hr style="margin:12px 0;">'),
            self.tiff_viewer.widget(),
        ])

    def _raw_default_path(self):
        """Per-frame raw TIFFs from <data_dir> in config_instr."""
        if not self.main_gui or not self.main_gui.experiment_dir:
            return None
        try:
            instr = self.main_gui.config_manager.get_cached('config_instr') or {}
        except Exception as e:
            self.log_debug(format_error_summary(e, prefix='_raw_default_path'))
            return None
        data_dir = instr.get('data_dir')
        return data_dir if data_dir and os.path.isdir(data_dir) else None

    def _prep_default_path(self):
        """Assembled prep stack: <exp>/preprocessed_data/prep_data.tif."""
        if not self.main_gui or not self.main_gui.experiment_dir:
            return None
        p = os.path.join(
            self.main_gui.experiment_dir, 'preprocessed_data', 'prep_data.tif',
        )
        return p if os.path.isfile(p) else None

    def _data_default_path(self):
        """Data tab output: <exp>/phasing_data/data.tif."""
        if not self.main_gui or not self.main_gui.experiment_dir:
            return None
        p = os.path.join(
            self.main_gui.experiment_dir, 'phasing_data', 'data.tif',
        )
        return p if os.path.isfile(p) else None

    @BaseTab._guard
    def _on_alien_change(self, change):
        """Update alien parameters based on selected algorithm."""
        alg = change['new']
        self.alien_params_box.children = []

        if alg == 'block aliens':
            self.aliens = text_field(placeholder='e.g., [[x1,y1,z1,x2,y2,z2], ...]')
            self.alien_params_box.children = [form_row('Aliens', self.aliens)]

        elif alg == 'alien file':
            self.alien_file = text_field(placeholder=_UI['placeholders']['alien_file'])
            self.alien_params_box.children = [form_row('Alien File', self.alien_file)]

        elif alg == 'AutoAlien1':
            self.AA1_size_threshold = text_field(placeholder='0.01')
            self.AA1_asym_threshold = text_field(placeholder='1.75')
            self.AA1_min_pts = text_field(placeholder='5')
            self.AA1_eps = text_field(placeholder='1.1')
            self.AA1_amp_threshold = text_field(placeholder='6.0')
            self.AA1_save_arrs = checkbox('save analysis arrays')
            self.AA1_expandcleanedsigma = text_field(placeholder='')

            self.AA1_defaults_btn = button('Set AA1 Defaults', style='info', width='140px', role='info')
            self.AA1_defaults_btn.on_click(lambda b: self._set_AA1_defaults_guarded())

            self.alien_params_box.children = [
                form_row('Size Threshold', self.AA1_size_threshold),
                form_row('Asymmetry Threshold', self.AA1_asym_threshold),
                form_row('Min Points', self.AA1_min_pts),
                form_row('Cluster Eps', self.AA1_eps),
                form_row('Amp Threshold', self.AA1_amp_threshold),
                self.AA1_save_arrs,
                form_row('Expand Sigma', self.AA1_expandcleanedsigma),
                self.AA1_defaults_btn
            ]

    @BaseTab._guard
    def _set_AA1_defaults_guarded(self):
        self._set_AA1_defaults()

    def _set_AA1_defaults(self):
        """Set AutoAlien1 parameters to defaults."""
        self.AA1_size_threshold.value = '0.01'
        self.AA1_asym_threshold.value = '1.75'
        self.AA1_min_pts.value = '5'
        self.AA1_eps.value = '1.1'
        self.AA1_amp_threshold.value = '6.0'
        self.AA1_save_arrs.value = False

    def load_tab(self, conf_map: dict):
        """Populate widgets from config dictionary."""
        # Hold any keys the form has no widget for so save_conf
        # round-trips them instead of dropping them on disk.
        self._passthrough = {
            k: v for k, v in conf_map.items() if k not in self._KNOWN_KEYS
        }
        # Alien algorithm
        alg = conf_map.get('alien_alg', 'random')
        if alg == 'random' or alg not in ['block_aliens', 'alien_file', 'AutoAlien1']:
            self.alien_alg.value = 'none'
        elif alg == 'block_aliens':
            self.alien_alg.value = 'block aliens'
            if 'aliens' in conf_map:
                self.aliens.value = self._fmt_value(conf_map['aliens'])
        elif alg == 'alien_file':
            self.alien_alg.value = 'alien file'
            if 'alien_file' in conf_map:
                self.alien_file.value = str(conf_map['alien_file'])
        elif alg == 'AutoAlien1':
            self.alien_alg.value = 'AutoAlien1'
            if 'AA1_size_threshold' in conf_map:
                self.AA1_size_threshold.value = str(conf_map['AA1_size_threshold'])
            if 'AA1_asym_threshold' in conf_map:
                self.AA1_asym_threshold.value = str(conf_map['AA1_asym_threshold'])
            if 'AA1_min_pts' in conf_map:
                self.AA1_min_pts.value = str(conf_map['AA1_min_pts'])
            if 'AA1_eps' in conf_map:
                self.AA1_eps.value = str(conf_map['AA1_eps'])
            if 'AA1_amp_threshold' in conf_map:
                self.AA1_amp_threshold.value = str(conf_map['AA1_amp_threshold'])
            if 'AA1_save_arrs' in conf_map:
                self.AA1_save_arrs.value = conf_map['AA1_save_arrs']
            if 'AA1_expandcleanedsigma' in conf_map:
                self.AA1_expandcleanedsigma.value = str(conf_map['AA1_expandcleanedsigma'])

        # Standard fields
        self.auto_intensity_threshold.value = conf_map.get('auto_intensity_threshold', False)
        if 'intensity_threshold' in conf_map:
            self.intensity_threshold.value = str(conf_map['intensity_threshold'])
        if 'binning' in conf_map:
            self.binning.value = self._fmt_value(conf_map['binning'])
        if 'shift' in conf_map:
            self.shift.value = self._fmt_value(conf_map['shift'])
        if 'crop_pad' in conf_map:
            self.crop_pad.value = self._fmt_value(conf_map['crop_pad'])
        self.no_center_max.value = conf_map.get('no_center_max', False)
        # Pre-fill TIFF viewer paths if the corresponding files exist.
        for key, resolver in (('compare', self._prep_default_path),
                              ('data', self._data_default_path)):
            default = resolver()
            if default:
                self.tiff_viewer.path[key].value = default

    def get_config(self) -> dict:
        """Read current widget values into config dictionary."""
        conf_map = {}

        # Alien algorithm
        alg = self.alien_alg.value
        if alg == 'block aliens':
            conf_map['alien_alg'] = 'block_aliens'
            if hasattr(self, 'aliens') and self.aliens.value:
                conf_map['aliens'] = self.aliens.value
        elif alg == 'alien file':
            conf_map['alien_alg'] = 'alien_file'
            if hasattr(self, 'alien_file') and self.alien_file.value:
                conf_map['alien_file'] = self.alien_file.value
        elif alg == 'AutoAlien1':
            conf_map['alien_alg'] = 'AutoAlien1'
            if hasattr(self, 'AA1_size_threshold') and self.AA1_size_threshold.value:
                conf_map['AA1_size_threshold'] = self._parse_field('AA1_size_threshold', self.AA1_size_threshold.value)
            if hasattr(self, 'AA1_asym_threshold') and self.AA1_asym_threshold.value:
                conf_map['AA1_asym_threshold'] = self._parse_field('AA1_asym_threshold', self.AA1_asym_threshold.value)
            if hasattr(self, 'AA1_min_pts') and self.AA1_min_pts.value:
                conf_map['AA1_min_pts'] = self._parse_field('AA1_min_pts', self.AA1_min_pts.value)
            if hasattr(self, 'AA1_eps') and self.AA1_eps.value:
                conf_map['AA1_eps'] = self._parse_field('AA1_eps', self.AA1_eps.value)
            if hasattr(self, 'AA1_amp_threshold') and self.AA1_amp_threshold.value:
                conf_map['AA1_amp_threshold'] = self._parse_field('AA1_amp_threshold', self.AA1_amp_threshold.value)
            if hasattr(self, 'AA1_save_arrs') and self.AA1_save_arrs.value:
                conf_map['AA1_save_arrs'] = True
            if hasattr(self, 'AA1_expandcleanedsigma') and self.AA1_expandcleanedsigma.value:
                conf_map['AA1_expandcleanedsigma'] = self._parse_field('AA1_expandcleanedsigma', self.AA1_expandcleanedsigma.value)

        # Standard fields
        if self.intensity_threshold.value:
            conf_map['intensity_threshold'] = self._parse_field('intensity_threshold', self.intensity_threshold.value)
        if self.binning.value:
            conf_map['binning'] = self._parse_field('binning', self.binning.value)
        if self.shift.value:
            conf_map['shift'] = self._parse_field('shift', self.shift.value)
        if self.crop_pad.value:
            conf_map['crop_pad'] = self._parse_field('crop_pad', self.crop_pad.value)
        if self.auto_intensity_threshold.value:
            conf_map['auto_intensity_threshold'] = True
        if self.no_center_max.value:
            conf_map['no_center_max'] = True

        # Preserve backend-set keys the form doesn't expose (e.g. 'pkg').
        for k, v in self._passthrough.items():
            conf_map.setdefault(k, v)
        return conf_map

    def clear_conf(self):
        """Reset all widgets to defaults."""
        self.alien_alg.value = 'none'
        self.intensity_threshold.value = ''
        self.binning.value = ''
        self.shift.value = ''
        self.crop_pad.value = ''
        self.auto_intensity_threshold.value = False
        self.no_center_max.value = False

    def run_tab(self, skip_save: bool = False):
        """Execute data formatting."""
        import cohere_ui.standard_preprocess as run_dt

        self.clear_output()

        err = self._validate_experiment()
        if err:
            self.log_error(err)
            return

        found_file = any(
            'prep_data.tif' in f
            for _, _, f in os.walk(self.main_gui.experiment_dir)
        )
        if not found_file:
            self.log_error(_MSG['data']['no_prep_data'])
            return

        if skip_save:
            self.log_warning(_MSG['tab']['run_modified_warning'])
        else:
            if self.save_and_verify():
                return

        before = self._snapshot_outputs()
        try:
            self.log_info(_MSG['data']['running'])
            run_dt.format_data(self.main_gui.experiment_dir, no_verify=self.main_gui.no_verify)
            self.log_success(_MSG['data']['complete'])
        except Exception as e:
            self.log_error(format_error_summary(e))
            self.log_debug(traceback.format_exc())
        finally:
            self._log_file_changes(before)

