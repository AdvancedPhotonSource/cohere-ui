"""InstrTab: instrument/beamline-specific configuration.

Per-beamline field lists are loaded from each beamline's sibling
``instr_schema.py`` via ``header.beamlines.load_instr_schema``.

To add a new beamline:
  1. Add the beamline module under ``cohere_beamlines``.
  2. Place an ``instr_schema.py`` next to it that declares
     ``INSTR_FIELDS`` (general + spec fields) and ``SPEC_DRIVERS``
     (the field keys whose changes trigger a spec re-parse).
"""

import ast
import html as _html
import os
import traceback

import ipywidgets as widgets

from cohere_ui.jupyter_gui.header.beamlines import (
    load_instr_schema, normalize_field, resolve_choices,
)
from cohere_ui.jupyter_gui.tabs.base import BaseTab, _MSG
from cohere_ui.jupyter_gui.utils.error_format import format_error_summary
from cohere_ui.jupyter_gui.widgets import (
    ChoiceInput, LogPanel, PathChooser, checkbox, form_row, text_field,
)


class InstrTab(BaseTab):
    """Tab for beamline instrument configuration."""

    name = "Instrument"
    conf_name = "config_instr"

    def __init__(self, beamline: str = None):
        super().__init__()
        self.beamline = beamline
        self._fields = {}
        self._schema_fields: dict = {'general': [], 'spec': []}
        self._spec_drivers: tuple = ()

    _PLACEHOLDER_HTML = (
        f'<div style="padding:18px; color:#666; font-size:13px;">'
        f'{_MSG["instr"]["placeholder"]}</div>'
    )

    def _build_ui(self) -> widgets.Widget:
        self.beamline_header = widgets.HTML(
            f'<b>Beamline: {self.beamline or "Not set"}</b>'
        )
        self.params_box = widgets.VBox()
        self._render_params_section()

        self.action_row = self._build_action_row(run_label=None)
        self.log_panel = LogPanel(height='100px')

        return widgets.VBox([
            self.beamline_header,
            self.params_box,
            widgets.Box([self.action_row], layout=widgets.Layout(margin='8px 0')),
            self.log_panel.widget,
        ])

    def _render_params_section(self):
        """Build either the per-beamline form or the no-beamline placeholder."""
        if self.beamline:
            self._build_beamline_fields()
        else:
            self.params_box.children = [widgets.HTML(self._PLACEHOLDER_HTML)]

    def set_beamline(self, beamline: str):
        """Set beamline and rebuild UI fields. Safe to call repeatedly."""
        self.beamline = beamline
        self.beamline_header.value = f'<b>Beamline: {self.beamline or "Not set"}</b>'
        self._render_params_section()
        if self.main_gui is not None:
            self._install_state_observers()

    def _build_beamline_fields(self):
        """Build fields from the beamline's ``instr_schema``."""
        self._fields = {}
        fields, drivers = load_instr_schema(self.beamline)
        # Normalise every field spec (tuple or dict) to a canonical dict.
        self._schema_fields = {
            section: [normalize_field(s) for s in fields.get(section, [])]
            for section in ('general', 'spec')
        }
        self._spec_drivers = tuple(drivers)

        sections = []
        general_rows = [self._make_row(s) for s in self._schema_fields['general']]
        sections.append(widgets.VBox(general_rows))

        if self._schema_fields['spec']:
            sections.append(widgets.HTML(
                '<div style="margin:8px 0 4px 0; padding-top:4px;'
                ' border-top:1px solid #ddd;"><b>Spec parameters</b></div>'
            ))
            spec_rows = [self._make_row(s) for s in self._schema_fields['spec']]
            sections.append(widgets.VBox(spec_rows))

        self.params_box.children = sections

    def _make_row(self, spec: dict):
        """Build a form row from a normalised field spec.

        ``spec['type']`` selects the widget:
        text/float/int -> text_field, bool -> checkbox,
        choice -> ChoiceInput (pick or type), dir/file -> PathChooser.
        """
        field_key = spec['key']
        field_type = spec.get('type', 'text')
        placeholder = spec.get('placeholder', '')

        if field_type == 'bool':
            widget = checkbox('', value=False)
        elif field_type == 'choice':
            choices = resolve_choices(self.beamline, spec)
            # ChoiceInput: pick from the list or enter a custom value.
            # Choices may be plain strings or (display, key) tuples; the
            # saved value is always the key. Fall back to a plain text
            # field when no choices are defined.
            if choices:
                widget = ChoiceInput(choices=choices, width='240px')
            else:
                widget = text_field(placeholder=placeholder, width='480px')
                # Fire observers on blur/Enter, not every keystroke
                # (spec re-parse is expensive).
                widget.continuous_update = False
        elif field_type in ('dir', 'file'):
            widget = PathChooser(
                kind=field_type, placeholder=placeholder, width='500px',
            )
        else:  # text, float, int (text-input backed)
            widget = text_field(placeholder=placeholder, width='480px')
            # Fire observers on blur/Enter, not every keystroke
            # (spec re-parse is expensive).
            widget.continuous_update = False

        # Re-parse the spec source when a driver field changes.
        if field_key in self._spec_drivers:
            widget.observe(self._on_spec_input_change, names='value')

        self._fields[field_key] = widget
        # PathChooser and similar wrappers expose the displayable
        # widget via ``.widget``; raw ipywidgets are used as-is.
        display_widget = (
            widget.widget
            if not isinstance(widget, widgets.Widget) and hasattr(widget, 'widget')
            else widget
        )
        label_html = self._render_label(spec)
        return form_row(
            label_html, display_widget,
            label_width='220px', right_align=True,
        )

    @staticmethod
    def _set_widget_value(widget, rendered: str):
        """Assign ``rendered`` to ``widget``.

        For a Dropdown, an unknown value falls back to the first option
        (avoids an error). Text-like widgets accept any string.
        """
        if isinstance(widget, widgets.Dropdown):
            if rendered in widget.options:
                widget.value = rendered
            else:
                widget.value = '' if '' in widget.options else widget.options[0]
        else:
            widget.value = rendered

    @staticmethod
    def _render_label(spec: dict) -> str:
        """Compose the HTML label: name + optional units + tooltip."""
        name = _html.escape(spec.get('label') or spec['key'])
        unit = spec.get('unit') or ''
        desc = spec.get('description') or ''
        # Unit appended in small grey after the name.
        unit_html = (
            f' <small style="color:#888;">({_html.escape(unit)})</small>'
            if unit else ''
        )
        if desc:
            return (
                f'<span title="{_html.escape(desc)}" '
                f'style="cursor:help; text-decoration:underline dotted;">'
                f'{name}</span>{unit_html}'
            )
        return f'{name}{unit_html}'

    @BaseTab._guard
    def _on_spec_input_change(self, _change):
        # General fields are user-managed: never overwrite them on
        # re-parse. Spec fields are re-pulled (force=True); if the
        # parse fails or returns nothing, existing values are kept.
        general_keys = {s['key'] for s in self._schema_fields.get('general', [])}
        self.parse_spec(skip_keys=general_keys, force=True)

    def init(self, main_gui):
        super().init(main_gui)
        # The main scan field also drives spec parsing.
        try:
            main_gui.scan.observe(self._on_spec_input_change, names='value')
        except AttributeError:
            pass

    def load_tab(self, conf_map: dict):
        """Populate widgets from config dictionary, then fill spec fields from the spec source."""
        if not self._fields:
            self._build_beamline_fields()

        for field_name, widget in self._fields.items():
            if isinstance(widget, widgets.Checkbox):
                widget.value = bool(conf_map.get(field_name, False))
                continue
            if field_name in conf_map:
                value = conf_map[field_name]
                if isinstance(value, (list, dict)):
                    rendered = self._fmt_value(value)
                else:
                    rendered = str(value)
                self._set_widget_value(widget, rendered)
            else:
                self._set_widget_value(widget, '')

        # Saved config values win over spec readings.
        self.parse_spec(skip_keys=set(conf_map.keys()))

        # parse_spec may have populated spec fields (chi, delta,
        # det_roi, ...) that aren't in the saved config_instr.
        # Persist them so the tab doesn't show as 'modified' on load.
        if self.main_gui is not None:
            new_conf = self.get_config()
            if new_conf != conf_map:
                try:
                    self.main_gui.config_manager.save_config(
                        self.conf_name, new_conf, no_verify=True,
                    )
                except Exception as e:
                    self.log_debug(
                        f'config_instr autosave after parse_spec failed: {e}'
                    )

    def parse_spec(self, skip_keys=None, *, force: bool = False):
        """Parse the beamline's spec source and populate spec fields.

        Parameters:
          skip_keys: keys to leave untouched (user-managed general
            fields, or keys already present in a saved config).
          force: when False, only empty spec widgets are filled; when
            True, every parsed key overwrites its widget (used after a
            driver field changes so the latest spec values are pulled).

        Any failure (missing driver value, malformed scan, parse error,
        empty result) returns silently and leaves widgets unchanged.
        """
        if skip_keys is None:
            skip_keys = set()
        if not self.beamline:
            return
        params = {}
        # Silent on incomplete driver fields - avoids parse-error spam
        # while the user is still typing.
        for key in self._spec_drivers:
            w = self._fields.get(key)
            if w is None:
                continue
            val = getattr(w, 'value', '')
            if isinstance(val, str):
                val = val.strip()
            if not val:
                return
            params[key] = val

        scan_value = ''
        if self.main_gui is not None:
            scan_value = (getattr(self.main_gui.scan, 'value', '') or '').strip()
        if not scan_value:
            return
        try:
            first_scan = int(scan_value.split(',')[0].split('-')[0])
        except ValueError:
            return

        # The instrument rework moved spec/metadata parsing off the
        # diffractometer onto the beamline Instrument. Build the per-beamline
        # Instrument subclass without a detector (parse_metadata never touches
        # det_obj) and read the metadata from it; the driver values (specfile /
        # h5file / data_dir) are passed in via config_instr. Fall back to the
        # legacy create_diffractometer API for any beamline not yet migrated.
        try:
            import importlib
            import inspect
            instr_mod = importlib.import_module(
                f'cohere_beamlines.{self.beamline}.instrument'
            )
            diff_mod = importlib.import_module(
                f'cohere_beamlines.{self.beamline}.diffractometers'
            )
            try:
                from cohere_beamlines.common.instr import Instrument as _InstrBase
                instr_cls = next(
                    (c for _, c in inspect.getmembers(instr_mod, inspect.isclass)
                     if issubclass(c, _InstrBase) and c is not _InstrBase),
                    None,
                )
            except ImportError:
                instr_cls = None

            if instr_cls is not None:
                instr_obj = instr_cls(
                    None, diff_mod.Diffractometer(),
                    {'config': {}, 'config_instr': dict(params)},
                )
                spec_dict = instr_obj.parse_metadata(first_scan)
            elif hasattr(diff_mod, 'create_diffractometer'):
                diff_field = self._fields.get('diffractometer')
                diff_name = getattr(diff_field, 'value', None)
                spec_dict = diff_mod.create_diffractometer(
                    diff_name, params).parse_metadata(first_scan)
            else:
                return
        except Exception as e:
            self.log_error(_MSG['instr']['parse_spec_failed'].format(
                error=format_error_summary(e)))
            self.log_debug(traceback.format_exc())
            return
        if not spec_dict:
            return

        for key, value in spec_dict.items():
            if key in skip_keys:
                continue
            widget = self._fields.get(key)
            if widget is None:
                continue
            if isinstance(widget, widgets.Checkbox):
                widget.value = bool(value)
            elif force or not widget.value:
                self._set_widget_value(widget, str(value))

    def get_config(self) -> dict:
        """Read current widget values into config dictionary."""
        conf_map = {}
        for field_name, widget in self._fields.items():
            if isinstance(widget, widgets.Checkbox):
                if widget.value:
                    conf_map[field_name] = True
                continue
            if widget.value:
                try:
                    conf_map[field_name] = ast.literal_eval(widget.value)
                except (ValueError, SyntaxError):
                    conf_map[field_name] = widget.value
        return conf_map

    def clear_conf(self):
        """Reset all widgets to defaults."""
        for widget in self._fields.values():
            if isinstance(widget, widgets.Checkbox):
                widget.value = False
            else:
                self._set_widget_value(widget, '')

    def run_tab(self, skip_save: bool = False):
        """Save the instrument config; this tab has no run step."""
        if skip_save:
            self.log_warning(_MSG['instr']['nothing_to_run'])
            return
        if not self.beamline:
            self.log_warning(_MSG['instr']['no_beamline_save'])
            return
        self.save_conf()
        self.log_success(_MSG['instr']['saved'])
