"""DispTab: display/visualization configuration and processing."""

import html as _html
import os

import ipywidgets as widgets

import cohere_core.utilities as ut

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

# User-facing label for the "main" reconstruction output.
_MAIN_DISP_ID_LABEL = 'main'


class DispTab(BaseTab):
    """Tab for visualization/postprocessing configuration.

    Handles result display, cropping, interpolation, strain visualization.
    """

    name = "Postprocess"
    conf_name = "config_disp"

    def __init__(self):
        super().__init__()
        # '' == 'main' == results_phasing/. A named id maps to
        # results_phasing_<id>/. Selects WHICH reconstruction's output
        # this tab post-processes; config_disp itself stays one shared
        # file whose results_dir pointer encodes the choice.
        self._rec_id: str = ''

    # reconstruction-id helpers

    def _output_dirname(self) -> str:
        """Input reconstruction dir for the active id."""
        return 'results_phasing' if not self._rec_id else f'results_phasing_{self._rec_id}'

    def _viz_dirname(self) -> str:
        """Visualization output dir for the active id (backend derives this
        from the input dir by replacing _phasing with _viz)."""
        return 'results_viz' if not self._rec_id else f'results_viz_{self._rec_id}'

    def _result_path(self) -> str:
        """Absolute path of the active reconstruction's results dir, or ''."""
        if not self.main_gui or not self.main_gui.experiment_dir:
            return ''
        return ut.join(self.main_gui.experiment_dir, self._output_dirname())

    def _build_ui(self) -> widgets.Widget:
        # Text input plus Browse popup with full-path tooltip,
        # matching the InstrTab specfile/data_dir and RecTab AI-model fields.
        self.result_dir = PathChooser(
            kind='dir',
            placeholder=_UI['placeholders']['phasing_results'],
            width='350px',
        )
        # A manual edit of the path should best-effort re-select the
        # matching reconstruction in the dropdown (or mark it custom).
        self.result_dir.observe(self._on_result_dir_change, 'value')
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
        from cohere_ui.jupyter_gui.features import get_disp_features
        self.features = {name: cls() for name, cls in get_disp_features().items()}
        self.feature_panel = FeaturePanel(self.features)

        self.action_row = self._build_action_row(run_label='Process Display', run_width='160px')

        self.log_panel = LogPanel()

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

        # Reconstruction selector: quick-switch which reconstruction this
        # tab post-processes. Selecting an id points result_dir at
        # results_phasing[_<id>]/ so output lands in results_viz[_<id>]/
        # (backend-derived).
        self.disp_rec_id_dropdown = dropdown(
            options=[_MAIN_DISP_ID_LABEL], value=_MAIN_DISP_ID_LABEL, width='180px',
        )
        self.disp_rec_id_dropdown.observe(self._on_disp_rec_id_change, 'value')
        self.disp_rec_id_row = widgets.HBox(
            [
                widgets.HTML('<span style="margin-right:6px;">Reconstruction</span>'),
                self.disp_rec_id_dropdown,
            ],
            layout=widgets.Layout(align_items='center', margin='4px 0 4px 0'),
        )
        self.disp_rec_id_status = widgets.HTML(
            layout=widgets.Layout(margin='0 0 8px 0'),
        )

        layout = widgets.VBox([
            self.disp_rec_id_row,
            self.disp_rec_id_status,
            params_section,
            # FeaturePanel renders its own "Features (k/N active)" header.
            self.feature_panel.widget,
            self.action_row,
            self.log_panel.widget,
            # 3D viewer lives at the bottom: collapsed by default, only
            # meaningful once Process Display has written results_viz/.
            self.viewer.widget,
        ])

        # Multipeak ignores rec_id (mp.process_dir hardcodes
        # results_phasing/results_viz), so hide the selector there.
        self._apply_multipeak_visibility()
        self._refresh_disp_status()

        return layout

    # reconstruction selector machinery

    def _apply_multipeak_visibility(self) -> None:
        """Hide the reconstruction selector in multipeak experiments.

        The multipeak postprocess path (mp.process_dir) hardcodes
        results_phasing/results_viz and ignores rec_id, so the selector
        would be misleading; the viewer is left unscoped to show every
        results_viz_<hkl>/.
        """
        if not hasattr(self, 'disp_rec_id_row'):
            return
        try:
            is_mp = bool(self.main_gui and self.main_gui._is_multipeak_experiment())
        except Exception:
            is_mp = False
        display = 'none' if is_mp else ''
        self.disp_rec_id_row.layout.display = display
        self.disp_rec_id_status.layout.display = display
        if is_mp:
            self._rec_id = ''
            if hasattr(self, 'viewer'):
                self.viewer.set_scope(None)

    def refresh_disp_rec_id_options(self) -> None:
        """Repopulate the selector: 'main' + union of reconstruction configs
        and results_phasing_* dirs that hold output (excluding *_backup)."""
        if not hasattr(self, 'disp_rec_id_dropdown'):
            return
        ids = []
        if self.main_gui is not None:
            cm = self.main_gui.config_manager
            ids = sorted(set(cm.discover_rec_ids()) | set(cm.discover_result_ids()))
        options = [_MAIN_DISP_ID_LABEL] + ids
        self.disp_rec_id_dropdown.unobserve(self._on_disp_rec_id_change, 'value')
        try:
            self.disp_rec_id_dropdown.options = options
            target = self._rec_id or _MAIN_DISP_ID_LABEL
            self.disp_rec_id_dropdown.value = (
                target if target in options else _MAIN_DISP_ID_LABEL
            )
            if self.disp_rec_id_dropdown.value == _MAIN_DISP_ID_LABEL:
                self._rec_id = ''
        finally:
            self.disp_rec_id_dropdown.observe(self._on_disp_rec_id_change, 'value')

    def _set_dropdown_value(self, label: str) -> None:
        """Set the selector value without firing the switch handler."""
        self.disp_rec_id_dropdown.unobserve(self._on_disp_rec_id_change, 'value')
        try:
            if label in list(self.disp_rec_id_dropdown.options):
                self.disp_rec_id_dropdown.value = label
        finally:
            self.disp_rec_id_dropdown.observe(self._on_disp_rec_id_change, 'value')

    def _set_result_dir(self, path: str) -> None:
        """Set result_dir without re-triggering the reverse-map observer."""
        self.result_dir.unobserve(self._on_result_dir_change, 'value')
        try:
            self.result_dir.value = path
        finally:
            self.result_dir.observe(self._on_result_dir_change, 'value')

    @BaseTab._guard
    def _on_disp_rec_id_change(self, change):
        """Switch which reconstruction this tab post-processes.

        Points result_dir at the selected reconstruction's results dir (so
        the backend writes viz output to results_viz[_<id>]/), scopes the
        3D viewer, and refreshes the status line. Does not persist
        config_disp (that happens on Save / Run) and never notifies rec.
        """
        if self.main_gui is None or not self.main_gui.experiment_exists():
            return
        new_label = change['new']
        self._rec_id = '' if new_label == _MAIN_DISP_ID_LABEL else new_label
        self._set_result_dir(self._result_path())
        self.viewer.set_scope(self._viz_dirname())
        self._refresh_viewer_metadata()
        self._refresh_disp_status()
        self.log_info(
            f'Post-processing reconstruction {new_label!r}: input from '
            f'{self._output_dirname()}/, output to {self._viz_dirname()}/.'
        )

    @BaseTab._guard
    def _on_result_dir_change(self, _change):
        """A manual results-dir edit best-effort re-selects the matching id."""
        self._sync_id_from_result_dir()

    def _sync_id_from_result_dir(self) -> None:
        """Map the current result_dir basename back to a selector id.

        results_phasing maps to main, and results_phasing_<id> maps to
        <id> when that id is in the dropdown. An unrecognized path leaves
        the dropdown alone (the status line then flags it as a custom path).
        """
        if not hasattr(self, 'disp_rec_id_dropdown'):
            return
        val = (self.result_dir.value or '').replace('\\', '/').rstrip('/')
        base = os.path.basename(val) if val else ''
        if not base or base == 'results_phasing':
            new_id = ''
        elif base.startswith('results_phasing_'):
            new_id = base[len('results_phasing_'):]
        else:
            new_id = None  # custom / unrecognized
        if new_id is not None:
            label = _MAIN_DISP_ID_LABEL if not new_id else new_id
            if label in list(self.disp_rec_id_dropdown.options):
                self._rec_id = new_id
                self._set_dropdown_value(label)
                if hasattr(self, 'viewer'):
                    self.viewer.set_scope(self._viz_dirname())
        self._refresh_disp_status()

    def _refresh_disp_status(self) -> None:
        """Show active reconstruction, input/output dirs, and whether the
        reconstruction output exists yet."""
        if not hasattr(self, 'disp_rec_id_status'):
            return
        label = _MAIN_DISP_ID_LABEL if not self._rec_id else self._rec_id
        in_dir = self._output_dirname()
        out_dir = self._viz_dirname()
        rp = self._result_path()
        has_input = bool(rp) and os.path.isdir(rp) and any(
            'image.npy' in files for _, _, files in os.walk(rp)
        )
        note = ('input results present' if has_input
                else 'no results yet, run Reconstruction first')
        custom = ''
        cur = (self.result_dir.value or '').replace('\\', '/').rstrip('/')
        if rp and cur and os.path.normpath(cur) != os.path.normpath(rp):
            custom = ' | <i>(custom path)</i>'
        self.disp_rec_id_status.value = (
            f'<span style="font-size:12px;color:var(--jup-fg-muted);">Post-processing '
            f'<b>{_html.escape(label)}</b> | input '
            f'<code>{_html.escape(in_dir)}/</code> to output '
            f'<code>{_html.escape(out_dir)}/</code> | {note}{custom}</span>'
        )

    def load_tab(self, conf_map: dict):
        """Populate widgets from config dictionary."""
        # Discover available reconstructions for the selector first.
        self.refresh_disp_rec_id_options()
        if 'results_dir' in conf_map:
            self._set_result_dir(conf_map['results_dir'].replace('\\', '/'))
        self.make_twin.value = conf_map.get('make_twin', False)
        self.unwrap.value = conf_map.get('unwrap', False)
        if 'rampups' in conf_map:
            self.rampups.value = str(conf_map['rampups'])
        if 'complex_mode' in conf_map:
            self.complex_mode.value = conf_map['complex_mode']

        # Features
        self.feature_panel.init_configs(conf_map)

        # Reverse-map the loaded results_dir to a selector id, then apply
        # multipeak visibility (which may override the viewer scope).
        self._sync_id_from_result_dir()
        self._apply_multipeak_visibility()
        # Repoint the viewer's file picker at the freshly-loaded
        # experiment. Cheap (no GL work). Just re-globs results_viz/.
        self._refresh_viewer_metadata()
        self._refresh_disp_status()

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
        """Add cross-feature dependency checks to the base field
        validation. Per-feature ``verify_active`` is already collected
        by ``BaseTab._validate_fields`` via ``_collect_feature_errors``.
        """
        errors = super()._validate_fields(conf_map)
        if conf_map is None or getattr(self.main_gui, 'no_verify', False):
            return errors
        errors.extend(self._validate_feature_dependencies(conf_map))
        return errors

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
        self._set_result_dir('')
        self.make_twin.value = False
        self.unwrap.value = False
        self.rampups.value = ''
        self.complex_mode.value = 'AmpPhase'
        self.feature_panel.clear_all()
        # Reset the reconstruction selector to 'main'.
        self._rec_id = ''
        if hasattr(self, 'disp_rec_id_dropdown'):
            self.disp_rec_id_dropdown.unobserve(self._on_disp_rec_id_change, 'value')
            try:
                self.disp_rec_id_dropdown.options = [_MAIN_DISP_ID_LABEL]
                self.disp_rec_id_dropdown.value = _MAIN_DISP_ID_LABEL
            finally:
                self.disp_rec_id_dropdown.observe(self._on_disp_rec_id_change, 'value')
        if hasattr(self, 'viewer'):
            self.viewer.set_scope(None)
        self._refresh_disp_status()

    def run_tab(self, skip_save: bool = False):
        """Execute visualization/postprocessing."""
        import cohere_ui.beamline_postprocess as dp

        self.clear_output()

        err = self._validate_experiment()
        if err:
            self.log_error(err)
            return

        if not self.result_dir.value:
            # Default to the selected reconstruction's results dir (main =
            # results_phasing/), NOT the whole experiment tree.
            self._set_result_dir(self._result_path())
            self.log_info(_MSG['disp']['set_results_dir'].format(path=self.result_dir.value))

        # Surface "no reconstruction output yet" up front rather than
        # letting the backend fail with a FileNotFoundError.
        missing = self._missing_required_inputs()
        if missing:
            self.log_error(
                "Cannot run postprocessing. Missing: "
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

        At least one ``results_phasing/<*>/image.npy`` must exist under
        the experiment dir.
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
        """Auto-follow entry point: point this tab at a reconstruction id.

        Called by RecTab when the user switches the rec config dropdown or
        a reconstruction finishes. Normalizes ``rec_id`` (None, '', and
        'main' all mean the canonical main output), drives the selector,
        repoints result_dir, scopes the viewer, and refreshes the status.
        It never calls back into the rec tab.
        """
        if 'rec_id' in kwargs:
            rid = kwargs['rec_id']
            target = '' if rid in (None, '', _MAIN_DISP_ID_LABEL) else str(rid)
            self._rec_id = target
            if hasattr(self, 'disp_rec_id_dropdown'):
                # refresh_disp_rec_id_options may reset _rec_id if the id
                # isn't on disk yet; restore the intended target after.
                self.refresh_disp_rec_id_options()
                self._rec_id = target
                self._set_dropdown_value(_MAIN_DISP_ID_LABEL if not target else target)
                self._set_result_dir(self._result_path())
                if hasattr(self, 'viewer'):
                    self.viewer.set_scope(self._viz_dirname())
                self._refresh_disp_status()
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
