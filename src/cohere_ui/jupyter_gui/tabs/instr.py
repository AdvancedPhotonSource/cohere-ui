"""InstrTab: instrument/beamline-specific configuration."""

import ast
import os

import ipywidgets as widgets

from .base import BaseTab, _MSG
from ..widgets import form_row, text_field, button, LogPanel


# Per-beamline field schema. General fields are always shown; spec fields
# are visually grouped under a header.
# Tuple shape: (config_key, display_label, placeholder).
_BEAMLINE_SCHEMA = {
    'aps_34idc': {
        'general': [
            ('diffractometer', 'diffractometer', 'e.g., 34idc'),
            ('specfile', 'spec file', 'Path to spec file'),
            ('data_dir', 'data directory', 'Data directory'),
            ('darkfield_filename', 'darkfield file', 'Dark field file'),
            ('whitefield_filename', 'whitefield file', 'White field file'),
            ('Imult', 'Imult', ''),
            ('det_roi', 'detector area (det_roi)', 'e.g., [y1, y2, x1, x2]'),
        ],
        'spec': [
            ('energy', 'energy', 'keV'),
            ('delta', 'delta (deg)', ''),
            ('gamma', 'gamma (deg)', ''),
            ('detdist', 'detdist (mm)', ''),
            ('th', 'th (deg)', ''),
            ('chi', 'chi (deg)', ''),
            ('phi', 'phi (deg)', ''),
            ('scanmot', 'scan motor', ''),
            ('scanmot_del', 'scan motor delta', ''),
            ('detector', 'detector', ''),
        ],
    },
    'aps_1ide': {
        'general': [
            ('diffractometer', 'diffractometer', 'e.g., 1ide'),
            ('data_dir', 'data directory', 'Data directory'),
            ('whitefield_filename', 'whitefield file', 'White field file'),
            ('roi', 'roi', 'e.g., [y1, y2, x1, x2]'),
            ('energy', 'energy', 'keV'),
        ],
        'spec': [],
    },
    'esrf_id01': {
        'general': [
            ('detector', 'detector', 'Detector name'),
            ('diffractometer', 'diffractometer', 'Diffractometer type'),
            ('h5file', 'h5 file', 'HDF5 file path'),
            ('roi', 'roi', 'e.g., [y1, y2, x1, x2]'),
        ],
        'spec': [],
    },
    'Petra3_P10': {
        'general': [
            ('diffractometer', 'diffractometer', 'Diffractometer type'),
            ('data_dir', 'data directory', 'Data directory'),
            ('sample', 'sample', 'Sample name'),
            ('darkfield_filename', 'darkfield file', 'Dark field file'),
            ('detector_module', 'detector module', ''),
            ('detector', 'detector', 'Detector name'),
        ],
        'spec': [
            ('energy', 'energy', 'keV'),
            ('del', 'del', 'Delta angle'),
            ('gam', 'gam', 'Gamma angle'),
            ('mu', 'mu', 'Mu angle'),
            ('om', 'om', 'Omega angle'),
            ('chi', 'chi', 'Chi angle'),
            ('phi', 'phi', 'Phi angle'),
            ('detdist', 'detdist', 'Detector distance'),
            ('scanmot', 'scan motor', ''),
        ],
    },
}

_DEFAULT_SCHEMA = {
    'general': [
        ('diffractometer', 'diffractometer', 'Diffractometer type'),
        ('data_dir', 'data directory', 'Data directory'),
    ],
    'spec': [],
}


class InstrTab(BaseTab):
    """Tab for beamline instrument configuration.

    Supports: aps_34idc, aps_1ide, esrf_id01, Petra3_P10.
    """

    name = "Instrument"
    conf_name = "config_instr"

    def __init__(self, beamline: str = None):
        super().__init__()
        self.beamline = beamline
        self._fields = {}

    def _build_ui(self) -> widgets.Widget:
        self.beamline_header = widgets.HTML(
            f'<b>Beamline: {self.beamline or "Not set"}</b>'
        )
        self.params_box = widgets.VBox()
        self._build_beamline_fields()

        self.load_btn = button('Load Config', style='warning',
                               width='180px', role='load')
        self.save_btn = button('Save Config', style='success',
                               width='150px', role='run')
        self.load_btn.on_click(lambda b: self._load_config_dialog())
        self.save_btn.on_click(lambda b: self.save_conf())

        self.log_panel = LogPanel(height='100px')

        return widgets.VBox([
            self.beamline_header,
            self.params_box,
            widgets.HBox([self.load_btn, self.save_btn],
                         layout=widgets.Layout(margin='8px 0')),
            self.log_panel.widget,
        ])

    def set_beamline(self, beamline: str):
        """Set beamline and rebuild UI fields."""
        self.beamline = beamline
        self.beamline_header.value = f'<b>Beamline: {self.beamline or "Not set"}</b>'
        self._build_beamline_fields()

    def _build_beamline_fields(self):
        """Build fields based on beamline type."""
        self._fields = {}
        schema = _BEAMLINE_SCHEMA.get(self.beamline, _DEFAULT_SCHEMA)

        sections = []
        general_rows = [self._make_row(spec) for spec in schema['general']]
        sections.append(widgets.VBox(general_rows))

        if schema['spec']:
            sections.append(widgets.HTML(
                '<div style="margin:8px 0 4px 0; padding-top:4px;'
                ' border-top:1px solid #ddd;"><b>Spec parameters</b></div>'
            ))
            spec_rows = [self._make_row(spec) for spec in schema['spec']]
            sections.append(widgets.VBox(spec_rows))

        self.params_box.children = sections

    def _make_row(self, spec):
        field_key, label, placeholder = spec
        # continuous_update=False so observers fire on blur/Enter, not every keystroke.
        widget = text_field(placeholder=placeholder, width='480px')
        widget.continuous_update = False
        self._fields[field_key] = widget
        # Re-parse the spec file when the user changes the inputs that drive parsing.
        if field_key in ('specfile', 'diffractometer'):
            widget.observe(self._on_spec_input_change, names='value')
        return form_row(label, widget, label_width='200px', right_align=True)

    def _on_spec_input_change(self, _change):
        # When the user changes specfile/diffractometer/scan they want fresh spec
        # values, so only protect general (non-spec) fields from being overwritten.
        schema = _BEAMLINE_SCHEMA.get(self.beamline, _DEFAULT_SCHEMA)
        skip = {key for key, _, _ in schema['general']}
        # Clear current spec values so re-parse fully repopulates from the new source.
        for key, _, _ in schema['spec']:
            widget = self._fields.get(key)
            if widget is not None:
                widget.value = ''
        self.parse_spec(skip_keys=skip)

    def init(self, main_gui):
        super().init(main_gui)
        # main_gui's scan field also drives spec parsing; observe it once main_gui is set.
        try:
            main_gui.scan.observe(self._on_spec_input_change, names='value')
        except AttributeError:
            pass

    def load_tab(self, conf_map: dict):
        """Populate widgets from config dictionary, then fill spec fields from the spec file."""
        if not self._fields:
            self._build_beamline_fields()

        for field_name, widget in self._fields.items():
            if field_name in conf_map:
                value = conf_map[field_name]
                if isinstance(value, (list, dict)):
                    widget.value = self._fmt_value(value)
                else:
                    widget.value = str(value)
            else:
                widget.value = ''

        # Parse the spec file to fill any spec fields the saved config did not override.
        self.parse_spec(skip_keys=set(conf_map.keys()))

    def parse_spec(self, skip_keys=None):
        """Parse spec file (if any) and populate matching empty fields.

        Only updates fields whose keys are absent from ``skip_keys`` so saved
        config values win over spec readings.
        """
        if skip_keys is None:
            skip_keys = set()
        if not self.beamline:
            return
        specfile = self._fields.get('specfile')
        diffractometer = self._fields.get('diffractometer')
        if specfile is None or diffractometer is None:
            return
        if not specfile.value or not diffractometer.value:
            return
        # Avoid noisy error spam if the user is mid-edit and the path doesn't exist yet.
        if not os.path.isfile(specfile.value):
            return
        scan_value = ''
        if self.main_gui is not None:
            scan_value = (self.main_gui.scan.value or '').strip()
        if not scan_value:
            return
        try:
            last_scan = int(scan_value.split('-')[-1].split(',')[-1])
        except ValueError:
            return

        try:
            import importlib
            diff_mod = importlib.import_module(
                f'cohere_beamlines.{self.beamline}.diffractometers'
            )
            diff_obj = diff_mod.create_diffractometer(
                diffractometer.value, {'specfile': specfile.value}
            )
            spec_dict = diff_obj.parse_metadata(last_scan)
        except Exception as e:
            self.log_error(_MSG['instr']['parse_spec_failed'].format(error=e))
            return
        if not spec_dict:
            return

        for key, value in spec_dict.items():
            if key in skip_keys:
                continue
            widget = self._fields.get(key)
            if widget is not None and not widget.value:
                widget.value = str(value)

    def get_config(self) -> dict:
        """Read current widget values into config dictionary."""
        conf_map = {}

        for field_name, widget in self._fields.items():
            if widget.value:
                # Try to parse as Python literal, otherwise keep as string
                try:
                    conf_map[field_name] = ast.literal_eval(widget.value)
                except (ValueError, SyntaxError):
                    conf_map[field_name] = widget.value

        return conf_map

    def clear_conf(self):
        """Reset all widgets to defaults."""
        for widget in self._fields.values():
            widget.value = ''

    def run_tab(self):
        """Instrument tab has no run action; it just saves config."""
        self.save_conf()
        self.log_success(_MSG['instr']['saved'])
