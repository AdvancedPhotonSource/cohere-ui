"""DispTab: display/visualization configuration and processing."""

import ast

import ipywidgets as widgets

from cohere_ui.jupyter_gui.tabs.base import BaseTab, _MSG
from cohere_ui.jupyter_gui.widgets import form_row, text_field, dropdown, checkbox, button, FeaturePanel, LogPanel
import traceback

from cohere_ui.jupyter_gui.error_format import format_error_summary


class DispTab(BaseTab):
    """Tab for visualization/postprocessing configuration.

    Handles result display, cropping, interpolation, strain visualization.
    """

    name = "Display"
    conf_name = "config_disp"

    def _build_ui(self) -> widgets.Widget:
        self.result_dir = text_field(placeholder='Path to phasing results', width='350px')
        self.make_twin = checkbox('make twin')
        self.unwrap = checkbox('include unwrapped phase')
        self.rampups = text_field(placeholder='e.g., 1')
        self.complex_mode = dropdown(options=['AmpPhase', 'ReIm'], value='AmpPhase')

        # Features panel
        from cohere_ui.jupyter_gui.features import DISP_FEATURES
        self.features = {name: cls() for name, cls in DISP_FEATURES.items()}
        self.feature_panel = FeaturePanel(self.features)

        self.load_btn = button('Load Config', style='warning', width='120px', role='load')
        self.run_btn = button('Process Display', style='success', width='140px', role='run')
        self.load_btn.on_click(lambda b: self._load_config_dialog())
        self.run_btn.on_click(lambda b: self.run_tab())

        self.log_panel = LogPanel(height='150px')

        params_section = widgets.VBox([
            form_row('Results Directory', self.result_dir),
            self.make_twin,
            self.unwrap,
            form_row('Ramp Upscale', self.rampups),
            form_row('Complex Mode', self.complex_mode),
        ])

        layout = widgets.VBox([
            params_section,
            widgets.HTML('<h4>Features</h4>'),
            self.feature_panel.widget,
            widgets.HBox([self.load_btn, self.run_btn]),
            self.log_panel.widget,
        ])

        return layout

    def load_tab(self, conf_map: dict):
        """Populate widgets from config dictionary."""
        if 'results_dir' in conf_map:
            self.result_dir.value = conf_map['results_dir'].replace('\\', '/')
        self.make_twin.value = conf_map.get('make_twin', False)
        self.unwrap.value = conf_map.get('unwrap', False)
        if 'rampups' in conf_map:
            self.rampups.value = str(conf_map['rampups'])
        if 'complex_mode' in conf_map:
            self.complex_mode.value = conf_map['complex_mode']

        # Features
        self.feature_panel.init_configs(conf_map)

    def get_config(self) -> dict:
        """Read current widget values into config dictionary."""
        conf_map = {}

        if self.result_dir.value:
            conf_map['results_dir'] = self.result_dir.value
        if self.make_twin.value:
            conf_map['make_twin'] = True
        if self.unwrap.value:
            conf_map['unwrap'] = True
        if self.rampups.value:
            conf_map['rampups'] = self._parse_field('rampups', self.rampups.value)
        conf_map['complex_mode'] = self.complex_mode.value

        # Features
        self.feature_panel.add_configs(conf_map)

        return conf_map

    def clear_conf(self):
        """Reset all widgets to defaults."""
        self.result_dir.value = ''
        self.make_twin.value = False
        self.unwrap.value = False
        self.rampups.value = ''
        self.complex_mode.value = 'AmpPhase'
        self.feature_panel.clear_all()

    def run_tab(self):
        """Execute visualization/postprocessing."""
        import cohere_ui.beamline_postprocess as dp

        self.clear_output()

        err = self._validate_experiment()
        if err:
            self.log_error(err)
            return

        if not self.result_dir.value:
            self.result_dir.value = self.main_gui.experiment_dir
            self.log_info(_MSG['disp']['set_results_dir'].format(path=self.result_dir.value))

        if self.save_and_verify():
            return

        before = self._snapshot_outputs()
        try:
            self.log_info(_MSG['disp']['running'])
            dp.handle_visualization(self.main_gui.experiment_dir, no_verify=self.main_gui.no_verify)
            self.log_success(_MSG['disp']['complete'])
        except Exception as e:
            self.log_error(format_error_summary(e))
            self.log_debug(traceback.format_exc())
        finally:
            self._log_file_changes(before)

    def update_tab(self, **kwargs):
        """Update tab from external notification (e.g., after reconstruction)."""
        if 'rec_id' in kwargs:
            import cohere_core.utilities as ut
            results_dir = ut.join(self.main_gui.experiment_dir, f'results_phasing_{kwargs["rec_id"]}')
            self.result_dir.value = results_dir
