"""MpTab: Multi-peak configuration tab.

Conditionally visible: appears only when the experiment is configured
for multi-peak (``config['multipeak']`` flag, ``config_mp`` file
presence, or any ``mp_*/`` subdir; see
``CoherenceGUI._is_multipeak_experiment``).

Values are stored verbatim in ``conf/config_mp`` and parsed by
``cohere_ui.api.multipeak.preprocess`` and
``cohere_core.controller.phasing.CoupledRec``. Save-only action row:
multi-peak prep runs from the Beamline Preprocess tab and multi-peak
rec from the Reconstruction tab.
"""

import traceback

import ipywidgets as widgets

from cohere_ui.jupyter_gui.tabs.base import BaseTab, _MSG
from cohere_ui.jupyter_gui.utils.error_format import format_error_summary
from cohere_ui.jupyter_gui.widgets import LogPanel, form_row, text_field


_GEOMETRY_FIELDS = [
    ('scan',                 'scan(s) per peak',     'e.g., 898-913,919-934,940-955,961-976'),
    ('orientations',         'peak orientations',    'e.g., [[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1]]'),
    ('hkl_in',               'hkl in',               'e.g., [1,1,1]'),
    ('hkl_out',              'hkl out',              'e.g., [1,1,-1]'),
    ('twin_plane',           'twin plane',           'e.g., [1,1,1]'),
    ('sample_axis',          'sample axis',          'e.g., [0,0,1]'),
    ('lattice_size',         'lattice size',         'e.g., 4.0786'),
    ('final_size',           'final size',           'e.g., 256'),
    ('switch_peak_trigger',  'switch peak trigger',  'e.g., [0, 5]'),
    ('adapt_trigger',        'adapt trigger',        'e.g., [0, 10]'),
]

_REC_CONTROL_FIELDS = [
    ('adapt_threshold_init',  'adapt threshold init',  ''),
    ('adapt_threshold_iters', 'adapt threshold iters', ''),
    ('adapt_threshold_vals',  'adapt threshold vals',  ''),
    ('weight_init',           'initial weight',        ''),
    ('weight_iters',          'weight iters',          ''),
    ('weight_vals',           'weight vals',           ''),
    ('adapt_alien_start',     'adapt alien start',     ''),
    ('adapt_alien_threshold', 'adapt alien threshold', ''),
    ('adapt_power',           'adapt power',           ''),
]

# Per-field type expectations enforced in get_config(). 'scan' stays str
# (the backend parses it). Anything not listed is accepted as a string.
_LIST_FIELDS = frozenset({
    'orientations', 'hkl_in', 'hkl_out', 'twin_plane', 'sample_axis',
    'switch_peak_trigger', 'adapt_trigger',
    'adapt_threshold_iters', 'adapt_threshold_vals',
    'weight_iters', 'weight_vals',
})
_INT_FIELDS = frozenset({'final_size'})
_FLOAT_FIELDS = frozenset({
    'lattice_size', 'adapt_threshold_init', 'weight_init',
    'adapt_alien_start', 'adapt_alien_threshold', 'adapt_power',
})


class MpTab(BaseTab):
    """Tab for multi-peak (``config_mp``) configuration."""

    name = "Multi-peak"
    conf_name = "config_mp"

    def __init__(self):
        super().__init__()
        self._fields: dict[str, widgets.Text] = {}

    def _build_ui(self) -> widgets.Widget:
        self._fields = {}
        sections = [
            widgets.HTML(
                f'<small style="color:var(--jup-fg-faint);">{_MSG["mp"]["intro"]}</small>'
            ),
            widgets.HTML('<b>Geometry</b>'),
            widgets.VBox([self._make_row(spec) for spec in _GEOMETRY_FIELDS]),
            widgets.HTML(
                '<div style="margin:8px 0 4px 0; padding-top:4px;'
                ' border-top:1px solid var(--jup-border);"><b>Reconstruction control</b></div>'
            ),
            widgets.VBox([self._make_row(spec) for spec in _REC_CONTROL_FIELDS]),
        ]

        self.action_row = self._build_action_row(run_label=None)
        self.log_panel = LogPanel()

        return widgets.VBox([
            *sections,
            widgets.Box([self.action_row], layout=widgets.Layout(margin='8px 0')),
            self.log_panel.widget,
        ])

    def _make_row(self, spec):
        key, label, placeholder = spec
        w = text_field(placeholder=placeholder, width='420px')
        w.continuous_update = False
        self._fields[key] = w
        return form_row(label, w, label_width='200px', right_align=True)

    def load_tab(self, conf_map: dict):
        """Populate widgets from ``config_mp`` dict."""
        if not self._fields:
            return
        for key, widget in self._fields.items():
            if key in conf_map:
                value = conf_map[key]
                if isinstance(value, (list, dict)):
                    widget.value = self._fmt_value(value)
                else:
                    widget.value = str(value)
            else:
                widget.value = ''

    def get_config(self) -> dict:
        """Read widget values into a config_mp dict.

        Empty fields are omitted. Typed fields (`_LIST_FIELDS`,
        `_INT_FIELDS`, `_FLOAT_FIELDS`) are parsed via
        ``BaseTab._parse_field``; on parse or type-check failure the
        field is dropped from the returned dict and a warning is logged
        via ``mp.invalid_field``. ``scan`` stays a string verbatim.
        """
        conf_map = {}
        for key, widget in self._fields.items():
            text = (widget.value or '').strip()
            if not text:
                continue
            if key in _LIST_FIELDS:
                value = self._parse_field(key, text)
                if not isinstance(value, list):
                    self._log_invalid(key, text, 'list')
                    continue
                conf_map[key] = value
            elif key in _INT_FIELDS:
                value = self._parse_field(key, text)
                if not isinstance(value, int) or isinstance(value, bool):
                    self._log_invalid(key, text, 'int')
                    continue
                conf_map[key] = value
            elif key in _FLOAT_FIELDS:
                value = self._parse_field(key, text)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    self._log_invalid(key, text, 'number')
                    continue
                conf_map[key] = value
            else:
                conf_map[key] = text
        return conf_map

    def _log_invalid(self, name: str, value: str, expected: str):
        self.log_error(_MSG['mp']['invalid_field'].format(
            name=name, value=value, expected=expected,
        ))

    def clear_conf(self):
        for widget in self._fields.values():
            widget.value = ''

    def run_tab(self, skip_save: bool = False):
        """No backend; Save is the only meaningful action."""
        if skip_save:
            self.log_warning(_MSG['mp']['nothing_to_run'])
            return
        try:
            self.save_conf()
        except Exception as e:
            self.log_error(format_error_summary(e, prefix='MpTab.run_tab'))
            self.log_debug(traceback.format_exc())
