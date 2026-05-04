"""Base class for toggleable features in reconstruction and display tabs."""

from abc import ABC, abstractmethod
from typing import Optional
import ipywidgets as widgets
import ast

from ..widgets import checkbox, button


class Feature(ABC):
    """Base class for toggleable configuration features.

    Features have an 'active' checkbox that shows/hides parameter fields.
    Subclasses implement fill_active() to define their parameters.

    Class-level metadata (set by subclasses):
    - `name`: short identifier shown in the FeaturePanel selector.
    - `description`: one-paragraph explanation shown to the user when the
      feature is selected. Markdown links are not rendered, but inline HTML is.
    - `disabled_reason`: if non-empty, the active checkbox is disabled and a
      red banner with this text is shown to the user. Used for features that
      exist in the schema but are not viable in the current GUI (e.g. `live`
      under Jupyter).
    """

    name: str = "Feature"
    description: str = ""
    disabled_reason: str = ""

    def __init__(self):
        self.active = checkbox(description='active')
        self.params_box = widgets.VBox()
        self.defaults_btn = button('Set Defaults', style='info', width='120px')

        if self.disabled_reason:
            self.active.disabled = True

        self.active.observe(self._on_active_change, 'value')
        self.defaults_btn.on_click(lambda b: self.set_defaults())

        self._widget = widgets.VBox([
            self.active,
            self.params_box
        ])

    @property
    def widget(self) -> widgets.Widget:
        """The feature's root widget."""
        return self._widget

    def _on_active_change(self, change):
        if change['new']:
            self._show_params()
        else:
            self._hide_params()

    def _show_params(self):
        """Show parameter widgets when feature is activated. Auto-applies
        set_defaults() so the feature is immediately usable — without it,
        fields stay empty and add_config silently skips them, dropping the
        feature from the saved config with no error."""
        param_widgets = self.fill_active()
        self.set_defaults()
        param_widgets.append(self.defaults_btn)
        self.params_box.children = param_widgets

    def _hide_params(self):
        """Hide parameter widgets when feature is deactivated."""
        self.params_box.children = []

    @abstractmethod
    def fill_active(self) -> list:
        """Create and return parameter widgets when feature becomes active.

        Returns:
            List of widgets to display
        """
        pass

    @abstractmethod
    def set_defaults(self):
        """Set all parameters to their default values."""
        pass

    @abstractmethod
    def init_config(self, conf_map: dict):
        """Initialize parameters from configuration dictionary.

        Args:
            conf_map: Configuration dictionary
        """
        pass

    @abstractmethod
    def add_config(self, conf_map: dict):
        """Add feature parameters to configuration dictionary.

        Only called when feature is active.

        Args:
            conf_map: Configuration dictionary to modify
        """
        pass

    def verify_active(self) -> str:
        """Verify that feature configuration is valid.

        Returns:
            Error message, or empty string if valid
        """
        return ""

    def _require_field(self, attr_name: str, label: Optional[str] = None) -> str:
        """Verify a parameter widget exists and is non-empty when the feature
        is active. Pass the attribute *name* (string), not the widget itself,
        so the lookup is deferred until after the inactive-skip check —
        parameter widgets are created lazily in fill_active(), which only
        runs when the user toggles the feature active. Calling this with
        self.foo when fill_active() has never run would AttributeError before
        we get a chance to short-circuit on inactive.
        """
        if not self.active.value:
            return ""
        field = getattr(self, attr_name, None)
        if field is None or not field.value:
            return f"{self.name} is active but {label or attr_name} is not configured"
        return ""

    def clear(self):
        """Reset feature to inactive state."""
        self.active.value = False
        self._hide_params()

    @staticmethod
    def parse_value(text: str):
        """Parse a string value into Python object using ast.literal_eval."""
        text = text.strip()
        if not text:
            return None
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text

    @staticmethod
    def format_value(value) -> str:
        """Format a Python value as string for display."""
        if value is None:
            return ""
        return str(value).replace(" ", "")
