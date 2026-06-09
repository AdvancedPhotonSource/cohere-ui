"""Display features for the Postprocess tab.

Each feature corresponds to one config_disp key consumed by
``cohere_ui.api.postprocessor.make_image_viz``.

To add a feature:
  1. Subclass Feature and set the relevant config_disp keys.
  2. Return parameter widgets from fill_active. Boolean-only features
     return an empty list, the active checkbox alone toggles them,
     and the on-disk key is written exactly when active is True.
  3. Parse numeric/list text inputs through the inherited parse_value
     so the config file stores real int/float/list values that the
     backend pattern-matches on.
  4. Register the subclass in DISP_FEATURES so the tab shows it.

Cross-feature dependencies (Strain needs Displacement;
``interpolation_resolution='min_deconv_res'`` needs Resolution) are
validated at the DispTab level via _validate_feature_dependencies.
"""

import ast

import ipywidgets as widgets

from cohere_ui.jupyter_gui.features.base import Feature
from cohere_ui.jupyter_gui.text import load_text
from cohere_ui.jupyter_gui.widgets import (
    FEATURE_INPUT_WIDTH, ChoiceInput, checkbox, dropdown, grid_field, text_field,
)

_TEXT = load_text('features')


class CropFeature(Feature):
    """Post-reconstruction crop of the visualization grid.

    Two strategies (``crop_type``):
      * ``"tight"``: bounding box of pixels above ``crop_thresh``
        (fraction of max amplitude), extended by ``crop_margin`` pixels.
      * ``"fraction"``: keep a fixed ``crop_fraction`` of each
        dimension around the center.
    """

    name = _TEXT['crop']['name']
    description = _TEXT['crop']['description']

    def fill_active(self) -> list:
        # Build all parameter rows once and toggle visibility per-row,
        # rather than rebuilding on every type switch. Each row is a
        # top-level grid_field so the parent grid sees them as cells,
        # the previous nested VBox grouping broke alignment because the
        # VBox itself became a single grid item that didn't participate
        # in the label/value tracks.
        self.crop_type = dropdown(options=['tight', 'fraction'], value='fraction',
                                  width=FEATURE_INPUT_WIDTH)
        self.margin = text_field(placeholder='10', width=FEATURE_INPUT_WIDTH)
        self.thresh = text_field(placeholder='0.5', width=FEATURE_INPUT_WIDTH)
        self.fraction = text_field(placeholder='[0.5, 0.5, 0.5]', width=FEATURE_INPUT_WIDTH)

        self._margin_row = grid_field('Margin (pixels)', self.margin)
        self._thresh_row = grid_field('Threshold (fraction of max)', self.thresh)
        self._fraction_row = grid_field('Fraction per axis', self.fraction)

        self.crop_type.observe(self._on_type_change, 'value')
        self._apply_type_visibility(self.crop_type.value)

        return [
            grid_field('Crop type', self.crop_type),
            self._margin_row,
            self._thresh_row,
            self._fraction_row,
        ]

    def _on_type_change(self, change):
        self._apply_type_visibility(change['new'])

    def _apply_type_visibility(self, kind: str) -> None:
        # Empty string clears the inline display style, restoring the
        # `display: contents` class rule that promotes the row's label
        # and widget into the parent grid.
        self._margin_row.layout.display = '' if kind == 'tight' else 'none'
        self._thresh_row.layout.display = '' if kind == 'tight' else 'none'
        self._fraction_row.layout.display = '' if kind == 'fraction' else 'none'

    def set_defaults(self):
        self.crop_type.value = 'fraction'
        self.margin.value = '10'
        self.thresh.value = '0.5'
        self.fraction.value = '[0.5, 0.5, 0.5]'
        self._apply_type_visibility(self.crop_type.value)

    def init_config(self, conf_map: dict):
        if 'crop_type' not in conf_map:
            self.active.value = False
            return
        self.active.value = True
        kind = conf_map['crop_type']
        self.crop_type.value = kind
        if 'crop_margin' in conf_map:
            self.margin.value = str(conf_map['crop_margin'])
        if 'crop_thresh' in conf_map:
            self.thresh.value = str(conf_map['crop_thresh'])
        if 'crop_fraction' in conf_map:
            self.fraction.value = self.format_value(conf_map['crop_fraction'])
        self._apply_type_visibility(kind)

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        kind = self.crop_type.value
        conf_map['crop_type'] = kind
        if kind == 'tight':
            if self.margin.value:
                conf_map['crop_margin'] = self.parse_value(self.margin.value)
            if self.thresh.value:
                conf_map['crop_thresh'] = self.parse_value(self.thresh.value)
        else:  # fraction
            if self.fraction.value:
                conf_map['crop_fraction'] = self.parse_value(self.fraction.value)

    def verify_active(self) -> str:
        if not self.active.value:
            return ""
        kind = self.crop_type.value
        if kind == 'tight':
            if not self.margin.value or not self.thresh.value:
                return f"{self.name}: tight crop requires both margin and threshold"
        elif kind == 'fraction':
            if not self.fraction.value:
                return f"{self.name}: fraction crop requires crop_fraction"
        return ""


class InterpolationFeature(Feature):
    """Interpolate the direct-space reconstruction onto a uniform grid.

    Output: ``direct_space_images_interpolated_<mode>.vti``.

    Valid ``interpolation_resolution`` values:
      * ``int`` / ``float``: uniform spacing in nm
      * ``list[float]``: per-axis spacing in nm
      * ``"min_deconv_res"``: derive from the Resolution feature
        (which must also be active; checked at the tab level)
    """

    name = _TEXT['interpolation']['name']
    description = _TEXT['interpolation']['description']

    def fill_active(self) -> list:
        self.mode = dropdown(options=['AmpPhase', 'ReIm'], value='AmpPhase',
                             width=FEATURE_INPUT_WIDTH)
        self.resolution = text_field(
            placeholder='float, [x,y,z], or "min_deconv_res"',
            width=FEATURE_INPUT_WIDTH,
        )
        return [
            grid_field('Mode', self.mode),
            grid_field('Resolution', self.resolution),
        ]

    def set_defaults(self):
        self.mode.value = 'AmpPhase'
        self.resolution.value = 'min_deconv_res'

    def init_config(self, conf_map: dict):
        if 'interpolation_mode' not in conf_map:
            self.active.value = False
            return
        self.active.value = True
        self.mode.value = conf_map['interpolation_mode']
        if 'interpolation_resolution' in conf_map:
            self.resolution.value = self.format_value(
                conf_map['interpolation_resolution']
            )

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        conf_map['interpolation_mode'] = self.mode.value
        if self.resolution.value:
            # parse_value yields int/float/list for literals, or the
            # raw string for named keys like "min_deconv_res".
            conf_map['interpolation_resolution'] = self.parse_value(
                self.resolution.value
            )

    def verify_active(self) -> str:
        if not self.active.value:
            return ""
        if not self.resolution.value:
            return f"{self.name}: interpolation_resolution is required"
        # Must parse to a number, list, or "min_deconv_res".
        parsed = self.parse_value(self.resolution.value)
        if parsed == 'min_deconv_res':
            return ""
        if isinstance(parsed, (int, float)):
            return ""
        if isinstance(parsed, (list, tuple)):
            return ""
        return (
            f"{self.name}: interpolation_resolution must be a number, "
            f"a list, or 'min_deconv_res', got {self.resolution.value!r}"
        )


class ResolutionFeature(Feature):
    """Compute reconstruction resolution via deconvolution analysis.

    Writes ``resolution_direct.vts`` and ``resolution_recip.vts`` to the
    viz directory. The backend supports only the "deconv" method, so
    activating this feature pins ``determine_resolution_type = 'deconv'``.
    """

    name = _TEXT['resolution']['name']
    description = _TEXT['resolution']['description']

    # Backend's only supported method.
    _METHOD = 'deconv'

    def fill_active(self) -> list:
        self.deconv_contrast = text_field(placeholder='0.25', width=FEATURE_INPUT_WIDTH)
        return [
            grid_field('Deconvolution contrast', self.deconv_contrast),
        ]

    def set_defaults(self):
        self.deconv_contrast.value = '0.25'

    def init_config(self, conf_map: dict):
        if 'determine_resolution_type' not in conf_map:
            self.active.value = False
            return
        self.active.value = True
        if 'resolution_deconv_contrast' in conf_map:
            self.deconv_contrast.value = str(conf_map['resolution_deconv_contrast'])

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        conf_map['determine_resolution_type'] = self._METHOD
        if self.deconv_contrast.value:
            conf_map['resolution_deconv_contrast'] = self.parse_value(
                self.deconv_contrast.value
            )

    def verify_active(self) -> str:
        if not self.active.value:
            return ""
        if not self.deconv_contrast.value:
            return f"{self.name}: deconv_contrast is required"
        try:
            v = float(self.parse_value(self.deconv_contrast.value))
        except (TypeError, ValueError):
            return f"{self.name}: deconv_contrast must be a number"
        if not (0.0 < v < 1.0):
            return f"{self.name}: deconv_contrast must be in (0, 1), got {v}"
        return ""


class ReciprocalFeature(Feature):
    """Also write the reciprocal-space VTS file.

    Activating writes ``write_recip = True``; deactivating omits the
    key. The active checkbox is the only control.
    """

    name = _TEXT['reciprocal']['name']
    description = _TEXT['reciprocal']['description']

    def fill_active(self) -> list:
        return []

    def set_defaults(self):
        pass

    def init_config(self, conf_map: dict):
        self.active.value = bool(conf_map.get('write_recip', False))

    def add_config(self, conf_map: dict):
        if self.active.value:
            conf_map['write_recip'] = True


class StrainFeature(Feature):
    """Compute strain from the displacement gradient along Q.

    Requires the Displacement feature to also be active so
    ``Bragg_displacement`` is set; checked at the tab level by
    :meth:`DispTab._validate_feature_dependencies`.
    """

    name = _TEXT['strain']['name']
    description = _TEXT['strain']['description']

    def fill_active(self) -> list:
        return []

    def set_defaults(self):
        pass

    def init_config(self, conf_map: dict):
        self.active.value = bool(conf_map.get('compute_strain', False))

    def add_config(self, conf_map: dict):
        if self.active.value:
            conf_map['compute_strain'] = True


class DisplacementFeature(Feature):
    """Convert reconstruction phase into a real-space displacement field (nm).

    ``Bragg_displacement`` accepts:

    * ``"Q"``: d-spacing derived geometrically from the q-vector
      (uses energy, detector position, and sample orientation from
      ``config_instr``). Usual choice.
    * ``<float>``: explicit d-spacing in Å; bypasses the geometric
      derivation. Use when geometry metadata is incomplete or to lock
      the conversion factor.

    ``make_image_viz`` computes ``displacement = phase * d_spacing / 10``
    (Å -> nm).
    """

    name = _TEXT['displacement']['name']
    description = _TEXT['displacement']['description']

    def fill_active(self) -> list:
        # One preset plus ChoiceInput's (custom...) escape so users
        # can type a numeric d-spacing. ChoiceInput wraps a widget:
        # use .widget when laying out, .value when reading.
        self.bragg_disp = ChoiceInput(
            choices=[('Q (geometric)', 'Q')],
            value='Q',
        )
        return [
            grid_field('Bragg displacement', self.bragg_disp.widget),
        ]

    def set_defaults(self):
        self.bragg_disp.value = 'Q'

    def init_config(self, conf_map: dict):
        if 'Bragg_displacement' not in conf_map:
            self.active.value = False
            return
        self.active.value = True
        value = conf_map['Bragg_displacement']
        # Legacy configs wrote True instead of "Q"; coerce to the
        # geometric default so old configs keep working.
        if value is True:
            value = 'Q'
        self.bragg_disp.value = str(value)

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        raw = self.bragg_disp.value
        if not raw:
            return
        # literal_eval for numeric d-spacing; fall back to the raw
        # string for named keys like "Q".
        try:
            conf_map['Bragg_displacement'] = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            conf_map['Bragg_displacement'] = str(raw)

    def verify_active(self) -> str:
        if not self.active.value:
            return ""
        raw = self.bragg_disp.value
        if not raw or not str(raw).strip():
            return f"{self.name}: Bragg_displacement value is required"
        if str(raw).strip() == 'Q':
            return ""
        try:
            float(ast.literal_eval(str(raw)))
            return ""
        except (TypeError, ValueError, SyntaxError):
            return (
                f"{self.name}: Bragg_displacement must be 'Q' or a "
                f"number, got {raw!r}"
            )
