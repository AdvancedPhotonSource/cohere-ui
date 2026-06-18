"""PrepTab: beamline preprocessing configuration."""

import os
import traceback

import ipywidgets as widgets

from cohere_ui.jupyter_gui.tabs.base import BaseTab, _MSG
from cohere_ui.jupyter_gui.widgets import form_row, text_field, dropdown, checkbox, button, LogPanel
from cohere_ui.jupyter_gui.viewers.tiff_viewer import TiffViewer
from cohere_ui.jupyter_gui.utils.error_format import format_error_summary
from cohere_ui.jupyter_gui.text import load_text

_UI = load_text('ui_strings')


class PrepTab(BaseTab):
    """Tab for beamline preprocessing configuration.

    Handles min_frames, roi, exclude_scans, outlier removal.
    """

    name = "Beamline Prep"
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
        self.remove_outliers = checkbox(
            'remove outliers', tooltip=_UI['tooltips']['remove_outliers']
        )
        self.outliers_scans = text_field(placeholder=_UI['placeholders']['auto_after_prep'])

        self.action_row = self._build_action_row(run_label='Save and Run', run_width='150px')

        self.log_panel = LogPanel()

        roi_tooltip = widgets.HTML(
            f'<small style="color: var(--jup-fg-muted);">{_MSG["prep"]["roi_formats"]}</small>'
        )

        self.tiff_viewer = TiffViewer(
            panes=[
                {
                    'key': 'raw',
                    'label': 'Raw frames (per-frame detector TIFFs from data_dir)',
                    'placeholder': 'dir / file / glob (e.g. <data_dir>, frame.tif, scan_*.tif)',
                    'default_path': lambda: self._raw_default_path(),
                },
                {
                    'key': 'prep',
                    'label': 'Beamline-preprocessed (assembled stack)',
                    'placeholder': 'dir / file / glob (e.g. preprocessed_data/prep_data.tif)',
                    'default_path': lambda: self._prep_default_path(),
                },
            ],
            title='TIFF Viewer',
            initial_visible=False,
            log_debug=self.log_debug,
        )

        return widgets.VBox([
            form_row('Min Frames', self.min_frames),
            form_row('Exclude Scans', self.exclude_scans),
            form_row('ROI', self.roi),
            form_row('ROI Format', self.roi_format),
            roi_tooltip,
            form_row('Max Crop', self.max_crop),
            self.remove_outliers,
            form_row('Outliers Scans', self.outliers_scans,
                     title=_UI['tooltips']['outliers_scans']),
            self.action_row,
            self.log_panel.widget,
            widgets.HTML('<hr style="margin:12px 0;">'),
            self.tiff_viewer.widget(),
        ])

    def _raw_default_path(self):
        """Path to per-frame raw TIFFs: <data_dir from config_instr>."""
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
        """Path to assembled prep stack: <exp>/preprocessed_data/prep_data.tif."""
        if not self.main_gui or not self.main_gui.experiment_dir:
            return None
        p = os.path.join(
            self.main_gui.experiment_dir, 'preprocessed_data', 'prep_data.tif',
        )
        return p if os.path.isfile(p) else None

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
        for key, resolver in (('raw', self._raw_default_path),
                              ('prep', self._prep_default_path)):
            default = resolver()
            if default:
                self.tiff_viewer.path[key].value = default

    def get_config(self) -> dict:
        """Read current widget values into config dictionary."""
        conf_map = {}

        if self.min_frames.value:
            conf_map['min_frames'] = self._parse_field('min_frames', self.min_frames.value)
        if self.exclude_scans.value:
            conf_map['exclude_scans'] = self._parse_field('exclude_scans', self.exclude_scans.value)
        if self.roi.value:
            conf_map['roi'] = self._parse_field('roi', self.roi.value)
        if self.roi_format.value:
            conf_map['roi_format'] = self.roi_format.value
        if self.max_crop.value:
            conf_map['max_crop'] = self._parse_field('max_crop', self.max_crop.value)
        if self.remove_outliers.value:
            conf_map['remove_outliers'] = True

        return conf_map

    def _pre_save_hook(self, conf_map: dict) -> None:
        """Preserve ``outliers_scans`` across save+run cycles.

        The prep backend writes the list back into ``config_prep`` after
        each run; the form has no widget for it (the user only chooses
        whether outlier removal is on). Without this hook, every save
        would drop the prior list, defeating the cache.
        """
        if not self.remove_outliers.value:
            return
        current = self.main_gui.config_manager.load_config(self.conf_name)
        if current and 'outliers_scans' in current:
            conf_map['outliers_scans'] = current['outliers_scans']

    def clear_conf(self):
        """Reset all widgets to defaults."""
        self.min_frames.value = ''
        self.exclude_scans.value = ''
        self.roi.value = ''
        self.roi_format.value = ''
        self.max_crop.value = ''
        self.outliers_scans.value = ''
        self.remove_outliers.value = False

    def run_tab(self, skip_save: bool = False):
        """Execute beamline preprocessing."""
        import cohere_ui.beamline_preprocess as prep

        self.clear_output()

        err = self._validate_experiment()
        if err:
            self.log_error(err)
            return

        if skip_save:
            self.log_warning(_MSG['tab']['run_modified_warning'])
        else:
            if self.save_and_verify():
                return

        before = self._snapshot_outputs()
        try:
            self.log_info(_MSG['prep']['running'])
            prep.handle_prep(self.main_gui.experiment_dir, no_verify=self.main_gui.no_verify)
            self.log_success(_MSG['prep']['complete'])

            updated_conf = self.main_gui.config_manager.load_config(self.conf_name)
            if updated_conf:
                self.clear_conf()
                self.load_tab(updated_conf)

        except Exception as e:
            self.log_error(format_error_summary(e))
            self.log_debug(traceback.format_exc())
        finally:
            self._log_file_changes(before)
