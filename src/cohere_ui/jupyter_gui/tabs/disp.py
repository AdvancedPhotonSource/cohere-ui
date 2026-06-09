"""DispTab: display/visualization configuration and processing."""

import ast
import os

import ipywidgets as widgets

from cohere_ui.jupyter_gui._validation import ValidationError
from cohere_ui.jupyter_gui.tabs.base import BaseTab, _MSG
from cohere_ui.jupyter_gui.text import load_text
from cohere_ui.jupyter_gui.widgets import (
    FeaturePanel, LogPanel, PathChooser, button, checkbox, dropdown,
    form_row, text_field,
)
import traceback

from cohere_ui.jupyter_gui.utils.error_format import format_error_summary
from cohere_ui.jupyter_gui.viewers.vts_viewer import VtsViewer

_UI = load_text('ui_strings')


class DispTab(BaseTab):
    """Tab for visualization/postprocessing configuration.

    Handles result display, cropping, interpolation, strain visualization.
    """

    name = "Postprocess"
    conf_name = "config_disp"

    def _build_ui(self) -> widgets.Widget:
        # Text input plus Browse popup with full-path tooltip,
        # matching the InstrTab specfile/data_dir and RecTab AI-model fields.
        self.result_dir = PathChooser(
            kind='dir',
            placeholder=_UI['placeholders']['phasing_results'],
            width='350px',
        )
        self.make_twin = checkbox(
            description=_UI['feature_options']['make_twin'],
            tooltip=_UI['tooltips']['make_twin'],
        )
        self.unwrap = checkbox(
            description=_UI['feature_options']['include_unwrap'],
            tooltip=_UI['tooltips']['unwrap'],
        )
        self.rampups = text_field(
            placeholder='e.g., 1 (no smoothing) or 3', width='100px',
        )
        self.complex_mode = dropdown(
            options=['AmpPhase', 'ReIm'], value='AmpPhase', width='140px',
        )

        # Features panel
        from cohere_ui.jupyter_gui.features import DISP_FEATURES
        self.features = {name: cls() for name, cls in DISP_FEATURES.items()}
        self.feature_panel = FeaturePanel(self.features)

        self.action_row = self._build_action_row(run_label='Process Display', run_width='160px')

        self.log_panel = LogPanel(height='150px')

        # Collapsible 3D viewer for results_viz/*.vts and *.vti files.
        # Construction is lazy (no GL context until the user expands it
        # for the first time), so this is cheap to instantiate eagerly.
        self.viewer = VtsViewer(main_gui=self.main_gui)

        # Group the two boolean toggles under one labelled "Options" row;
        # inline left margin spaces the second checkbox.
        self.unwrap.layout = widgets.Layout(margin='0 0 0 16px')
        options_row = form_row(
            'Options', widgets.HBox([self.make_twin, self.unwrap])
        )

        params_section = widgets.VBox([
            form_row('Results directory', self.result_dir.widget),
            form_row('Complex mode', self.complex_mode),
            form_row('Rampups (smoothing passes)', self.rampups),
            options_row,
        ])

        layout = widgets.VBox([
            params_section,
            # FeaturePanel renders its own "Features (k/N active)" header.
            self.feature_panel.widget,
            self.action_row,
            self.log_panel.widget,
            # 3D viewer lives at the bottom: collapsed by default, only
            # meaningful once Process Display has written results_viz/.
            self.viewer.widget,
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

        # Repoint the viewer's file picker at the freshly-loaded
        # experiment. Cheap (no GL work) - just re-globs results_viz/.
        self._refresh_viewer_metadata()

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
        # Always write complex_mode: the postprocessor branches on it for
        # interpolation, so saving it explicitly pins behaviour even if the
        # upstream default changes.
        conf_map['complex_mode'] = self.complex_mode.value

        # Features
        self.feature_panel.add_configs(conf_map)

        return conf_map

    def _validate_fields(self, conf_map: dict) -> list:
        """Add per-feature ``verify_active`` results and cross-feature
        dependency checks to the base field validation, so every
        feature-level error appears in the LogPanel before the backend runs.
        """
        errors = super()._validate_fields(conf_map)
        if conf_map is None or getattr(self.main_gui, 'no_verify', False):
            return errors
        errors.extend(self._collect_feature_errors())
        errors.extend(self._validate_feature_dependencies(conf_map))
        return errors

    def _collect_feature_errors(self) -> list:
        """Run each feature's ``verify_active`` and wrap the messages."""
        errs = []
        for name, feat in self.features.items():
            msg = (feat.verify_active() or "").strip()
            if msg:
                errs.append(ValidationError(name, msg))
        return errs

    def _validate_feature_dependencies(self, conf_map: dict) -> list:
        """Catch cross-feature dependencies the backend would otherwise raise on:

        * Strain (``compute_strain=True``) requires the Displacement feature
          (``Bragg_displacement`` key set).
        * Interpolation at ``min_deconv_res`` requires the Resolution feature
          (``determine_resolution_type`` set).
        """
        errs = []
        if conf_map.get('compute_strain') and 'Bragg_displacement' not in conf_map:
            errs.append(ValidationError(
                'compute_strain',
                "Strain requires the Displacement feature active "
                "(Bragg_displacement key missing from config_disp)",
            ))
        if conf_map.get('interpolation_resolution') == 'min_deconv_res' \
                and 'determine_resolution_type' not in conf_map:
            errs.append(ValidationError(
                'interpolation_resolution',
                "Setting interpolation_resolution='min_deconv_res' "
                "requires the Resolution feature active "
                "(determine_resolution_type key missing from config_disp)",
            ))
        return errs

    def clear_conf(self):
        """Reset all widgets to defaults."""
        self.result_dir.value = ''
        self.make_twin.value = False
        self.unwrap.value = False
        self.rampups.value = ''
        self.complex_mode.value = 'AmpPhase'
        self.feature_panel.clear_all()

    def run_tab(self, skip_save: bool = False):
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

        # Surface "no reconstruction output yet" up front rather than
        # letting the backend fail with a FileNotFoundError.
        missing = self._missing_required_inputs()
        if missing:
            self.log_error(
                "Cannot run postprocessing - "
                + "; ".join(missing)
                + ". Run Reconstruction first."
            )
            return

        if skip_save:
            self.log_warning(_MSG['tab']['run_modified_warning'])
        else:
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
            # Refresh the viewer if it's currently visible so a re-run
            # is reflected without a manual Reload click. When collapsed
            # we still refresh its file picker so the next expand is
            # current, but skip the expensive re-read of the iframe.
            try:
                if self.viewer.is_visible:
                    self.viewer.reload()
                else:
                    self.viewer._refresh_file_options()
                    self.viewer._refresh_title()
            except Exception as e:
                self.log_debug(f'viewer post-run refresh skipped: {e}')

    def _missing_required_inputs(self) -> list:
        """Return descriptions of any required input file that's absent.

        Mirrors ``handle_visualization``'s search: at least one
        ``results_phasing/<*>/image.npy`` must exist under the experiment dir.
        """
        if not self.main_gui or not self.main_gui.experiment_dir:
            return ["experiment directory not set"]
        exp_dir = self.main_gui.experiment_dir
        found_image = False
        for root, _, files in os.walk(exp_dir):
            if 'image.npy' in files and 'results_phasing' in root:
                found_image = True
                break
        if not found_image:
            return [f"no results_phasing/image.npy under {exp_dir}"]
        return []

    def update_tab(self, **kwargs):
        """Update tab from external notification (e.g., after reconstruction)."""
        if 'rec_id' in kwargs:
            import cohere_core.utilities as ut
            results_dir = ut.join(self.main_gui.experiment_dir, f'results_phasing_{kwargs["rec_id"]}')
            self.result_dir.value = results_dir
        # Reconstruction wrote new image.npy files but didn't touch
        # results_viz/. Refresh anyway so the viewer's count and title
        # reflect whatever's on disk now (handles the case where the
        # user manually re-runs viz between reconstructions).
        self._refresh_viewer_metadata()

    def _refresh_viewer_metadata(self):
        """Update the viewer's file picker + title without re-rendering.

        Called from load_tab and update_tab so the picker reflects the
        current experiment's results_viz/ content. Cheap; no GL work.
        """
        try:
            self.viewer._refresh_file_options()
            self.viewer._refresh_title()
        except Exception as e:
            # Don't let a viewer hiccup block the tab's load/update.
            self.log_debug(f'viewer metadata refresh skipped: {e}')
