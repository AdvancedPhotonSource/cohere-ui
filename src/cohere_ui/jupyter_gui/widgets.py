"""
Widget factory functions and helpers for building ipywidgets-based forms.
"""

import os
import sys
import traceback
import ipywidgets as widgets
from IPython.display import display

from cohere_ui.jupyter_gui.styles import apply_button_role

try:
    from ipyfilechooser import FileChooser
    HAS_FILECHOOSER = True
except ImportError:
    HAS_FILECHOOSER = False


def form_row(label: str, widget, label_width: str = '180px',
             right_align: bool = False) -> widgets.HBox:
    """Create a horizontal label-widget pair.

    Set right_align=True to right-align the label inside its column.
    """
    label_html = (
        f'<div style="text-align:right; padding-right:8px;">{label}</div>'
        if right_align else label
    )
    label_widget = (
        widgets.HTML(value=label_html, layout=widgets.Layout(width=label_width))
        if right_align else
        widgets.Label(value=label, layout=widgets.Layout(width=label_width))
    )
    return widgets.HBox(
        [label_widget, widget],
        layout=widgets.Layout(margin='1px 0')
    )


def text_field(value: str = '', placeholder: str = '', width: str = '200px') -> widgets.Text:
    """Create a text input field."""
    return widgets.Text(value=value, placeholder=placeholder, layout=widgets.Layout(width=width))


def int_field(value: int = 0, width: str = '100px') -> widgets.IntText:
    """Create an integer input field."""
    return widgets.IntText(value=value, layout=widgets.Layout(width=width))


def float_field(value: float = 0.0, width: str = '100px') -> widgets.FloatText:
    """Create a float input field."""
    return widgets.FloatText(value=value, layout=widgets.Layout(width=width))


def dropdown(options: list, value=None, width: str = '200px') -> widgets.Dropdown:
    """Create a dropdown selection widget."""
    return widgets.Dropdown(
        options=options,
        value=value if value is not None else (options[0] if options else None),
        layout=widgets.Layout(width=width)
    )


def checkbox(description: str = '', value: bool = False) -> widgets.Checkbox:
    """Create a checkbox widget."""
    return widgets.Checkbox(value=value, description=description, indent=False)


def button(description: str, style: str = 'primary', width: str = '150px',
           role: str = None) -> widgets.Button:
    """Create a button widget.

    Args:
        description: Button text
        style: ipywidgets button_style ('primary', 'success', 'warning', 'danger', 'info')
        width: CSS width string
        role: Optional color role ('load', 'set', 'run', 'info') applied via CSS class.
    """
    btn = widgets.Button(
        description=description,
        button_style=style,
        layout=widgets.Layout(width=width)
    )
    if role:
        apply_button_role(btn, role)
    return btn


class DirChooser:
    """Directory chooser widget.

    Uses ipyfilechooser if available, otherwise falls back to button + text display.
    """

    def __init__(self, start_path: str = None, title: str = 'Select Directory'):
        self.title = title
        self._value = ''
        self._callbacks = []

        start_path = start_path or os.getcwd()

        if HAS_FILECHOOSER:
            self._fc = FileChooser(
                path=start_path,
                select_default=True,
                show_only_dirs=True,
                title=title
            )
            self._fc.register_callback(self._on_select)
            self.widget = self._fc
        else:
            # Fallback: button + path display + manual entry
            self._path_display = widgets.Text(
                value='',
                placeholder='Click Browse or enter path...',
                layout=widgets.Layout(width='350px')
            )
            self._browse_btn = widgets.Button(
                description='Browse...',
                button_style='info',
                layout=widgets.Layout(width='80px')
            )
            self._browse_btn.on_click(self._show_browser)
            self._browser_output = widgets.Output()

            self.widget = widgets.VBox([
                widgets.HBox([self._path_display, self._browse_btn]),
                self._browser_output
            ])

    def _on_select(self, chooser):
        """Callback when directory is selected via ipyfilechooser."""
        self._value = chooser.selected_path or ''
        for cb in self._callbacks:
            cb(self._value)

    def _show_browser(self, b):
        """Show a simple directory browser in fallback mode."""
        with self._browser_output:
            self._browser_output.clear_output()
            current = self._path_display.value or os.getcwd()
            if not os.path.isdir(current):
                current = os.getcwd()

            print(f"Current: {current}")
            print("Subdirectories:")

            # Show parent
            parent = os.path.dirname(current)
            if parent and parent != current:
                print(f"  [..] (parent)")

            # Show subdirectories
            try:
                for item in sorted(os.listdir(current)):
                    full_path = os.path.join(current, item)
                    if os.path.isdir(full_path) and not item.startswith('.'):
                        print(f"  [{item}]")
            except PermissionError:
                print("  (permission denied)")

            print("\nEnter path in text field above and press Enter")

    @property
    def value(self) -> str:
        if HAS_FILECHOOSER:
            return self._fc.selected_path or ''
        return self._path_display.value

    @value.setter
    def value(self, val: str):
        self._value = val
        if HAS_FILECHOOSER:
            if val and os.path.isdir(val):
                self._fc.reset(path=val)
        else:
            self._path_display.value = val

    def register_callback(self, callback):
        """Register a callback for when selection changes."""
        self._callbacks.append(callback)
        if HAS_FILECHOOSER:
            pass  # Already registered in __init__
        else:
            self._path_display.observe(lambda c: callback(c['new']), 'value')


class FileField:
    """File chooser widget."""

    def __init__(self, start_path: str = None, filter_pattern: str = '*', title: str = 'Select File'):
        self.title = title
        self._value = ''
        self._callbacks = []

        start_path = start_path or os.getcwd()

        if HAS_FILECHOOSER:
            self._fc = FileChooser(
                path=start_path,
                filter_pattern=filter_pattern,
                select_default=False,
                title=title
            )
            self._fc.register_callback(self._on_select)
            self.widget = self._fc
        else:
            self._path_display = widgets.Text(
                value='',
                placeholder='Enter file path...',
                layout=widgets.Layout(width='400px')
            )
            self.widget = self._path_display

    def _on_select(self, chooser):
        self._value = chooser.selected or ''
        for cb in self._callbacks:
            cb(self._value)

    @property
    def value(self) -> str:
        if HAS_FILECHOOSER:
            return self._fc.selected or ''
        return self._path_display.value

    @value.setter
    def value(self, val: str):
        self._value = val
        if HAS_FILECHOOSER:
            if val and os.path.exists(val):
                self._fc.reset(path=os.path.dirname(val), filename=os.path.basename(val))
        else:
            self._path_display.value = val

    def register_callback(self, callback):
        self._callbacks.append(callback)


def dir_chooser(start_path: str = None, title: str = 'Select Directory') -> DirChooser:
    """Create a directory chooser widget."""
    return DirChooser(start_path=start_path, title=title)


def file_chooser(start_path: str = None, filter_pattern: str = '*', title: str = 'Select File') -> FileField:
    """Create a file chooser widget."""
    return FileField(start_path=start_path, filter_pattern=filter_pattern, title=title)


def output_area(height: str = '200px') -> widgets.Output:
    """Create an output area for messages and logs."""
    return widgets.Output(layout=widgets.Layout(
        border='1px solid #ccc',
        height=height,
        overflow='auto'
    ))


def section_header(text: str) -> widgets.HTML:
    """Create a section header."""
    return widgets.HTML(f'<h4 style="margin: 10px 0 5px 0; color: #333;">{text}</h4>')


class LogPanel:
    """Auto-scrolling, level-styled log panel (info/success/warning/error/debug)."""

    _LEVEL_STYLE = {
        'info':    'color:#222;',
        'success': 'color:#1e7a1e; font-weight:600;',
        'warning': 'color:#a06000;',
        'error':   'color:#a02020; font-weight:600;',
        'debug':   'color:#777; font-style:italic;',
    }
    _LEVEL_PREFIX = {
        'info':    '',
        'success': '[OK] ',
        'warning': '[WARN] ',
        'error':   '[ERROR] ',
        'debug':   '[DEBUG] ',
    }

    def __init__(self, height: str = '150px', max_lines: int = 500):
        self._lines = []
        self._max_lines = max_lines
        self.show_debug = False
        self._html = widgets.HTML(
            value=self._render(),
            layout=widgets.Layout(border='1px solid #ccc', height=height,
                                  margin='4px 0 0 0'),
        )
        self.show_debug_checkbox = widgets.Checkbox(
            value=False, description='show debug', indent=False,
            layout=widgets.Layout(margin='2px 0 0 0', width='auto'),
        )
        self.show_debug_checkbox.observe(self._on_debug_toggle, names='value')
        self.widget = widgets.VBox([self._html, self.show_debug_checkbox])

    def info(self, msg):
        self._append('info', msg)

    def success(self, msg):
        self._append('success', msg)

    def warning(self, msg):
        self._append('warning', msg)

    def error(self, msg):
        self._append('error', msg)

    def debug(self, msg):
        self._append('debug', msg)

    def clear(self):
        self._lines = []
        self._refresh()

    def set_show_debug(self, value: bool):
        value = bool(value)
        if self.show_debug == value:
            return
        self.show_debug = value
        if self.show_debug_checkbox.value != value:
            self.show_debug_checkbox.value = value
        self._refresh()

    def _on_debug_toggle(self, change):
        self.show_debug = bool(change['new'])
        self._refresh()

    def _append(self, level: str, msg):
        self._lines.append((level, str(msg)))
        if len(self._lines) > self._max_lines:
            self._lines = self._lines[-self._max_lines:]
        self._refresh()

    def _refresh(self):
        try:
            self._html.value = self._render()
        except Exception as e:
            sys.stderr.write(
                f"LogPanel._refresh failed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )

    def _render(self) -> str:
        from html import escape
        rows = []
        visible_idx = 0
        for level, msg in self._lines:
            if level == 'debug' and not self.show_debug:
                continue
            style = self._LEVEL_STYLE.get(level, '')
            prefix = self._LEVEL_PREFIX.get(level, '')
            rows.append(
                f'<div style="order:{-visible_idx};{style}">{prefix}{escape(msg)}</div>'
            )
            visible_idx += 1
        return (
            '<div style="height:100%;overflow-y:auto;'
            'display:flex;flex-direction:column-reverse;'
            'font-family:Menlo,Consolas,monospace;font-size:11px;'
            'line-height:1.35;padding:6px;background:#fafafa;'
            'white-space:pre-wrap;word-break:break-word;">'
            + ''.join(rows) + '</div>'
        )


class FeaturePanel:
    """Selector + stacked-parameter view for toggleable features."""

    def __init__(self, features: dict):
        """
        Args:
            features: Dict mapping feature names to Feature objects
        """
        self.features = features
        self.feature_names = list(features.keys())

        self.selector = widgets.Select(
            options=self._build_options(),
            value=self.feature_names[0] if self.feature_names else None,
            layout=widgets.Layout(width='180px', height='220px')
        )

        for feature in self.features.values():
            feature.active.observe(
                lambda _change: self._refresh_selector_options(),
                names='value',
            )

        self._info_area = widgets.HTML()
        self._params_holder = widgets.VBox()

        self.params_area = widgets.VBox(
            children=[self._info_area, self._params_holder],
            layout=widgets.Layout(
                width='460px',
                min_height='220px',
                padding='10px',
                border='1px solid #ddd',
            ),
        )

        self.selector.observe(self._on_select, 'value')
        self._update_params_display()

        self.widget = widgets.HBox([self.selector, self.params_area])

    _ACTIVE_PREFIX = '● '   # filled circle + space
    _INACTIVE_PREFIX = '○ '  # hollow circle + space

    def _build_options(self):
        """Return [(display_label, value)] tuples with ●/○ prefix per feature."""
        return [
            (
                (self._ACTIVE_PREFIX if feature.active.value else self._INACTIVE_PREFIX)
                + name,
                name,
            )
            for name, feature in self.features.items()
        ]

    def _refresh_selector_options(self):
        """Rebuild active/inactive label prefixes; suppress _on_select during the swap."""
        current = self.selector.value
        try:
            self.selector.unobserve(self._on_select, 'value')
        except (ValueError, RuntimeError):
            pass
        try:
            self.selector.options = self._build_options()
            if current is not None and current in self.feature_names:
                self.selector.value = current
        except Exception as e:
            sys.stderr.write(
                f"FeaturePanel._refresh_selector_options failed: "
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
        finally:
            self.selector.observe(self._on_select, 'value')

    def _on_select(self, change):
        self._update_params_display()

    def _update_params_display(self):
        name = self.selector.value
        if name and name in self.features:
            feature = self.features[name]
            self._info_area.value = self._render_info(feature)
            self._params_holder.children = [feature.widget]
        else:
            self._info_area.value = ''
            self._params_holder.children = []

    @staticmethod
    def _render_info(feature) -> str:
        title = (
            f'<div style="font-weight:600; font-size:1.05em; margin-bottom:4px;">'
            f'{feature.name}</div>'
        )
        desc = (
            f'<div style="color:#555; font-size:0.9em; margin-bottom:8px;">'
            f'{feature.description}</div>'
        ) if feature.description else ''
        banner = ''
        if feature.disabled_reason:
            banner = (
                '<div style="background:#fdecea; color:#a02020; '
                'border:1px solid #f5c2c0; padding:6px 8px; border-radius:4px; '
                'margin-bottom:8px; font-size:0.85em;">'
                f'<b>Disabled:</b> {feature.disabled_reason}</div>'
            )
        return title + desc + banner

    def init_configs(self, conf_map: dict):
        """Initialize all features from config dictionary."""
        for feature in self.features.values():
            feature.init_config(conf_map)
        self._update_params_display()

    def add_configs(self, conf_map: dict):
        """Add all active feature configs to dictionary."""
        for feature in self.features.values():
            feature.add_config(conf_map)

    def clear_all(self):
        """Clear all feature configurations."""
        for feature in self.features.values():
            feature.clear()
        self._update_params_display()
