"""
Widget factory functions and helpers for building ipywidgets-based forms.
"""

import html as _html
import itertools as _itertools
import os
import sys
import traceback
import ipywidgets as widgets
from IPython.display import display

from cohere_ui.jupyter_gui.styles import apply_button_role
from cohere_ui.jupyter_gui.text import load_text

_BTN = load_text('messages')['buttons']
_UI = load_text('ui_strings')

# Unique DOM id per clipboard helper so multiple log panels in the
# same notebook don't share a hidden textarea.
_copy_button_uid = _itertools.count()


def make_copy_to_clipboard_html(text: str, icon: str = 'copy',
                                label: str = '',
                                tooltip: str = 'Copy to clipboard') -> tuple[str, str]:
    """HTML+JS snippet whose embedded ``<button>`` copies ``text`` to the clipboard.

    Args:
        text: payload to copy.
        icon: FontAwesome name without the ``fa-`` prefix (e.g. ``"copy"``);
            empty string skips the icon.
        label: optional text next to the icon; empty keeps it icon-only.
        tooltip: hover title.

    Returns ``(html_snippet, dom_id)``. Re-render with the same dom_id when
    the text changes. Uses ``navigator.clipboard.writeText`` where available
    and falls back to ``document.execCommand('copy')``.

    Plain HTML button (not ipywidgets) so the browser keeps clipboard
    permission tied to the user's click.
    """
    uid = f'cohere-copy-{next(_copy_button_uid)}'
    safe_textarea = _html.escape(text or '', quote=True)
    icon_html = f'<i class="fa fa-{_html.escape(icon)}"></i>' if icon else ''
    label_html = (
        f'<span style="margin-left:6px;">{_html.escape(label)}</span>'
        if label else ''
    )
    button_inner = icon_html + label_html
    # Flash fa-check / fa-times for 1.5 s then restore the original
    # innerHTML so the FontAwesome <i> tag survives the swap.
    #
    # Use &quot; (not \\") for inner attribute quotes: the HTML parser
    # decodes &quot; to " inside the attribute without terminating it,
    # whereas \\" would end the onclick attribute early and break the
    # handler.
    ok_html = (
        '<i class=&quot;fa fa-check&quot; '
        'style=&quot;color:#1e7a1e;&quot;></i>'
    )
    fail_html = (
        '<i class=&quot;fa fa-times&quot; '
        'style=&quot;color:#a02020;&quot;></i>'
    )
    onclick = (
        "(function(btn){"
        f"  var ta = document.getElementById('{uid}-ta');"
        "  if (!ta) { return; }"
        "  var original = btn.innerHTML;"
        "  function flash(html){ btn.innerHTML = html; "
        "    setTimeout(function(){ btn.innerHTML = original; }, 1500); }"
        "  try {"
        "    if (navigator.clipboard && navigator.clipboard.writeText) {"
        "      navigator.clipboard.writeText(ta.value).then("
        f"        function(){{ flash('{ok_html}'); }},"
        f"        function(){{ ta.select(); document.execCommand('copy'); "
        f"                       flash('{ok_html}'); }});"
        "    } else {"
        "      ta.select(); document.execCommand('copy');"
        f"      flash('{ok_html}');"
        "    }"
        "  } catch (e) {"
        f"    flash('{fail_html}');"
        "  }"
        "})(this);"
    )
    html_snippet = (
        f'<span style="display:inline-flex; align-items:center;">'
        # Off-screen textarea holds the current text.
        f'<textarea id="{uid}-ta" readonly aria-hidden="true" '
        f'style="position:absolute; left:-9999px; top:0; width:1px; '
        f'height:1px; opacity:0;">{safe_textarea}</textarea>'
        f'<button type="button" title="{_html.escape(tooltip)}" '
        f'onclick="{onclick}" '
        # min-width prevents icon clipping on narrow toolbars;
        # line-height:1 keeps the bounding box snug around the glyph.
        f'style="min-width:30px; padding:4px 8px; border:1px solid #bbb; '
        f'border-radius:4px; background:#f7f7f7; cursor:pointer; '
        f'font-size:13px; line-height:1; color:#444;">'
        f'{button_inner}</button>'
        f'</span>'
    )
    return html_snippet, uid

try:
    from ipyfilechooser import FileChooser
    HAS_FILECHOOSER = True
except ImportError:
    HAS_FILECHOOSER = False


# One width token shared by every text/dropdown input inside a Feature
# subclass. Path/file inputs and the ChoiceInput composite keep their own
# content-aware sizing because they wrap multi-element rows. Tab-level
# widgets (under tabs/*.py) are unaffected; they still use form_row.
FEATURE_INPUT_WIDTH = '260px'


def grid_field(label: str, widget, *, title: str = '') -> widgets.HBox:
    """Emit one labeled row that participates in the parent feature grid.

    The returned HBox carries the `jup-gui-feature-row` class whose CSS
    rule is `display: contents`, promoting the label and widget to direct
    cells of the surrounding `.jup-gui-feature-grid`. Use inside a
    Feature's `fill_active()` instead of `form_row` so labels and inputs
    auto-align across every feature without per-row pixel agreement.

    Args:
        label: column-1 text. Plain string, rendered as HTML so a tooltip
            can be attached.
        widget: column-2 widget.
        title: optional hover tooltip; gives the label a dotted underline
            and help cursor to advertise it (matches form_row's behavior).
    """
    if title:
        title_attr = f' title="{_html.escape(title)}"'
        label_html = (
            f'<div{title_attr} '
            f'style="cursor:help; text-decoration:underline dotted;">'
            f'{_html.escape(label)}</div>'
        )
    else:
        label_html = f'<div>{_html.escape(label)}</div>'
    label_widget = widgets.HTML(value=label_html)
    row = widgets.HBox([label_widget, widget])
    row.add_class('jup-gui-feature-row')
    return row


def grid_full(widget, *, justify: str = 'start') -> widgets.Box:
    """Wrap a widget so it spans both columns of the parent feature grid.

    Use for bare checkboxes, HR separators, sub-section toggles, and the
    Set Defaults action button - anything without a label column.

    Args:
        widget: the widget to host.
        justify: CSS `justify-self` for the row. 'start' (default) hugs the
            label column's left edge; 'end' right-aligns within the row
            (used by the Set Defaults button).
    """
    box = widgets.Box([widget])
    box.add_class('jup-gui-feature-row-full')
    # justify-self is grid-cell-level - inline via the layout so each call
    # site can pick its own alignment without exploding the CSS class set.
    box.layout.justify_self = justify
    return box


def form_row(label: str, widget, label_width: str = '180px',
             right_align: bool = False, title: str = '') -> widgets.HBox:
    """Create a horizontal label-widget pair.

    Set right_align=True to right-align the label inside its column.
    Set title to attach a hover tooltip; the label gets a dotted
    underline and help cursor to advertise it.
    """
    if title or right_align:
        align = 'text-align:right; padding-right:8px;' if right_align else ''
        hover = (
            'cursor:help; text-decoration:underline dotted;' if title else ''
        )
        title_attr = f' title="{_html.escape(title)}"' if title else ''
        label_html = (
            f'<div{title_attr} style="{align} {hover}">{label}</div>'
        )
        label_widget = widgets.HTML(
            value=label_html, layout=widgets.Layout(width=label_width),
        )
    else:
        label_widget = widgets.Label(
            value=label, layout=widgets.Layout(width=label_width),
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


def checkbox(description: str = '', value: bool = False,
             tooltip: str = '') -> widgets.Checkbox:
    """Create a checkbox widget. ``tooltip`` shows on hover over the description."""
    cb = widgets.Checkbox(value=value, description=description, indent=False)
    if tooltip:
        cb.tooltip = tooltip
    return cb


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


class SaveButton:
    """Save button whose enabled state mirrors the tab's saved/modified/absent badge.

    The owner tab calls ``set_state('saved'|'modified'|'absent')`` after
    each StatusStrip refresh. ``'saved'`` disables the button; any other
    state enables it. Clicking always invokes ``on_save``.
    """

    TOOLTIPS = {
        'saved': _BTN['save_saved'],
        'modified': _BTN['save_modified'],
        'absent': _BTN['save_absent'],
    }

    def __init__(self, on_save, width: str = '90px'):
        self._on_save = on_save
        self.btn = widgets.Button(
            description=_UI['action_buttons']['save'],
            tooltip=self.TOOLTIPS['saved'],
            layout=widgets.Layout(width=width),
        )
        apply_button_role(self.btn, 'set')
        self.btn.disabled = True
        self.btn.on_click(lambda b: self._on_save())

    @property
    def widget(self):
        return self.btn

    def set_state(self, state: str):
        self.btn.disabled = (state == 'saved')
        self.btn.tooltip = self.TOOLTIPS.get(state, '')


class SplitRunButton:
    """Run button with an inline 3-way menu for unsaved-changes choices.

    When the tab is clean, the main face triggers a plain run. When
    modified or absent, the row swaps to ``[Save & Run] [Run anyway]
    [Cancel]`` and collapses back after any choice.
    """

    def __init__(self, description: str, *, on_save_and_run, on_run_only,
                 width: str = '160px'):
        self._description = description
        self._on_save_and_run = on_save_and_run
        self._on_run_only = on_run_only
        self._state = 'saved'

        self.main_btn = widgets.Button(
            description=description,
            layout=widgets.Layout(width=width),
        )
        apply_button_role(self.main_btn, 'run')
        self.caret_btn = widgets.Button(
            description='\u25bc',
            tooltip=_BTN['run_options'],
            layout=widgets.Layout(width='28px'),
        )
        apply_button_role(self.caret_btn, 'run')

        self.save_and_run_btn = widgets.Button(
            description=_UI['action_buttons']['save_and_run'],
            layout=widgets.Layout(width='130px'),
        )
        apply_button_role(self.save_and_run_btn, 'run')
        self.run_only_btn = widgets.Button(
            description=_UI['action_buttons']['run_anyway'],
            tooltip=_BTN['run_only'],
            layout=widgets.Layout(width='130px'),
        )
        apply_button_role(self.run_only_btn, 'info')
        self.cancel_btn = widgets.Button(
            description=_UI['action_buttons']['cancel'],
            layout=widgets.Layout(width='90px'),
        )

        self.main_btn.on_click(self._on_main_click)
        self.caret_btn.on_click(lambda b: self._open_menu())
        self.save_and_run_btn.on_click(lambda b: self._choose(self._on_save_and_run))
        self.run_only_btn.on_click(lambda b: self._choose(self._on_run_only))
        self.cancel_btn.on_click(lambda b: self._close_menu())

        self._collapsed_row = widgets.HBox([self.main_btn, self.caret_btn])
        self._expanded_row = widgets.HBox(
            [self.save_and_run_btn, self.run_only_btn, self.cancel_btn]
        )
        self._container = widgets.HBox([self._collapsed_row])
        self._update_visuals()

    @property
    def widget(self):
        return self._container

    def set_state(self, state: str):
        if state == self._state:
            return
        self._state = state
        self._update_visuals()

    def set_enabled(self, enabled: bool):
        self.main_btn.disabled = not enabled
        self.caret_btn.disabled = not enabled

    def busy(self, message: str = 'Working...'):
        """Context manager: disable the button group and show a busy label.

        Use for the synchronous prep phase (config validation, file
        checks). The subprocess lifecycle is toggled separately by the
        monitor's running-state callback.
        """
        outer = self

        class _Busy:
            def __enter__(self_inner):
                self_inner._prior_desc = outer.main_btn.description
                self_inner._prior_disabled = outer.main_btn.disabled
                self_inner._prior_caret_disabled = outer.caret_btn.disabled
                outer.main_btn.disabled = True
                outer.caret_btn.disabled = True
                outer.main_btn.description = message
                return outer

            def __exit__(self_inner, exc_type, exc, tb):
                outer.main_btn.description = self_inner._prior_desc
                outer.main_btn.disabled = self_inner._prior_disabled
                outer.caret_btn.disabled = self_inner._prior_caret_disabled
                return False
        return _Busy()

    def _update_visuals(self):
        if self._state == 'saved':
            self.main_btn.description = self._description
            self.main_btn.tooltip = _BTN['run_saved']
        elif self._state == 'modified':
            self.main_btn.description = f'{self._description} *'
            self.main_btn.tooltip = _BTN['run_modified']
        else:  # absent
            self.main_btn.description = self._description
            self.main_btn.tooltip = _BTN['run_absent']

    def _on_main_click(self, _b):
        if self._state == 'modified':
            self._open_menu()
        else:
            # saved -> no-op save; absent -> first-write save.
            self._choose(self._on_save_and_run)

    def _open_menu(self):
        self._container.children = [self._expanded_row]

    def _close_menu(self):
        self._container.children = [self._collapsed_row]

    def _choose(self, handler):
        self._close_menu()
        handler()


class ChoiceInput:
    """Strict dropdown of ``(display, key)`` pairs with a ``(custom...)`` escape.

    The dropdown shows the human-readable display name; a small grey
    label echoes the key that will be written to the config. Picking
    ``(custom...)`` swaps the dropdown for a free-form text input plus
    a back arrow. ``.value`` is the key (or the typed custom string).

    Choices may be plain strings (treated as ``(s, s)``) or
    ``(display, key)`` tuples. Exposes ``.value`` and
    ``observe('value', ...)`` so it drops in where ``widgets.Text`` was used.
    """

    CUSTOM_SENTINEL = '__choice_custom__'

    def __init__(self, choices=None, value: str = '', width: str = '240px'):
        self._pairs: list[tuple[str, str]] = []
        for entry in (choices or []):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                disp, key = str(entry[0]), str(entry[1])
            else:
                disp = key = str(entry)
            self._pairs.append((disp, key))

        opts = [('--', '')] + [(d, k) for d, k in self._pairs]
        opts.append(('(custom...)', self.CUSTOM_SENTINEL))

        self.dropdown = widgets.Dropdown(
            options=opts, value='',
            layout=widgets.Layout(width=width, flex='0 0 auto'),
        )
        self.key_label = widgets.HTML(
            value='', layout=widgets.Layout(margin='0 0 0 8px'),
        )
        self.dropdown.observe(self._on_dropdown_change, 'value')

        self.text = widgets.Text(
            placeholder=_UI['placeholders']['custom_value'],
            layout=widgets.Layout(width=width, flex='0 0 auto'),
        )
        self.text.continuous_update = False
        self.back_btn = widgets.Button(
            description='', icon='arrow-left',
            tooltip=_UI['tooltips']['back_to_picker'],
            layout=widgets.Layout(width='38px'),
        )
        self.back_btn.on_click(self._exit_custom)

        self._picker_row = widgets.HBox(
            [self.dropdown, self.key_label],
            layout=widgets.Layout(align_items='center'),
        )
        self._custom_row = widgets.HBox(
            [self.text, self.back_btn],
            layout=widgets.Layout(align_items='center'),
        )

        # External observers added via .observe(); paused during mode swaps
        # so a programmatic reset doesn't fire downstream handlers on the
        # transient empty value.
        self._external_handlers: list[tuple] = []
        self._in_custom = False
        self.widget = widgets.HBox(
            [self._picker_row],
            layout=widgets.Layout(margin='0'),
        )
        if value:
            self.value = value

    @property
    def value(self) -> str:
        if self._in_custom:
            return self.text.value or ''
        v = self.dropdown.value
        if v == self.CUSTOM_SENTINEL or not v:
            return ''
        return v

    @value.setter
    def value(self, v: str):
        v = v or ''
        keys = {k for _, k in self._pairs}
        if v == '' or v in keys:
            self._pause_external_observers()
            try:
                self._in_custom = False
                self.widget.children = [self._picker_row]
                self.dropdown.value = v if v in keys else ''
                self._refresh_key_label()
            finally:
                self._resume_external_observers()
            self._fire_external_observers(self.value)
        else:
            self._enter_custom(initial=v)

    def observe(self, handler, names='value'):
        self._external_handlers.append((handler, names))
        self.dropdown.observe(handler, names=names)
        self.text.observe(handler, names=names)

    def unobserve(self, handler, names='value'):
        try:
            self._external_handlers.remove((handler, names))
        except ValueError:
            pass
        self.dropdown.unobserve(handler, names=names)
        self.text.unobserve(handler, names=names)

    def _pause_external_observers(self):
        # Unobserve may raise if the handler isn't attached on that
        # widget (e.g. paused twice). Log to stderr rather than swallow
        # so a real bug still leaves a trail.
        for h, n in self._external_handlers:
            try:
                self.dropdown.unobserve(h, names=n)
            except (ValueError, RuntimeError) as e:
                sys.stderr.write(
                    f"ChoiceInput._pause_external_observers (dropdown): "
                    f"{type(e).__name__}: {e}\n"
                )
            try:
                self.text.unobserve(h, names=n)
            except (ValueError, RuntimeError) as e:
                sys.stderr.write(
                    f"ChoiceInput._pause_external_observers (text): "
                    f"{type(e).__name__}: {e}\n"
                )

    def _resume_external_observers(self):
        for h, n in self._external_handlers:
            self.dropdown.observe(h, names=n)
            self.text.observe(h, names=n)

    def _fire_external_observers(self, new_value: str):
        # Handler exceptions must not abort the swap (the widget needs
        # to land in a consistent state), so log to stderr instead of
        # re-raising.
        change = {'name': 'value', 'new': new_value, 'old': new_value,
                  'owner': self, 'type': 'change'}
        for h, _n in self._external_handlers:
            try:
                h(change)
            except Exception as e:
                sys.stderr.write(
                    f"ChoiceInput._fire_external_observers handler "
                    f"{getattr(h, '__name__', repr(h))} raised "
                    f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                )

    def _on_dropdown_change(self, change):
        new = change.get('new')
        if new == self.CUSTOM_SENTINEL:
            self._enter_custom()
        else:
            self._refresh_key_label()

    def _refresh_key_label(self):
        v = self.dropdown.value or ''
        if v and v != self.CUSTOM_SENTINEL:
            # Suppress the shadow when display equals key (bare strings).
            display = next((d for d, k in self._pairs if k == v), v)
            if display == v:
                self.key_label.value = ''
                return
            self.key_label.value = (
                f'<small style="color:#888; '
                f'font-family:Menlo,Consolas,monospace;">'
                f'&rarr; {_html.escape(v)}</small>'
            )
        else:
            self.key_label.value = ''

    def _enter_custom(self, initial: str = ''):
        self._in_custom = True
        # Pause external observers across the reset + row swap so they
        # don't see a spurious value='' from the programmatic clear;
        # fire once at the end with the real value.
        self._pause_external_observers()
        try:
            self.dropdown.unobserve(self._on_dropdown_change, 'value')
            try:
                self.dropdown.value = ''
            finally:
                self.dropdown.observe(self._on_dropdown_change, 'value')
            self.key_label.value = ''
            self.text.value = initial
            self.widget.children = [self._custom_row]
        finally:
            self._resume_external_observers()
        self._fire_external_observers(self.value)

    def _exit_custom(self, _b=None):
        self._in_custom = False
        current = self.text.value
        keys = {k for _, k in self._pairs}
        self._pause_external_observers()
        try:
            if current in keys:
                self.dropdown.value = current
            else:
                self.dropdown.value = ''
            self._refresh_key_label()
            self.widget.children = [self._picker_row]
        finally:
            self._resume_external_observers()
        self._fire_external_observers(self.value)


class PathChooser:
    """Compact text input + collapsible browse popup for dir/file paths.

    Long paths show their tail (the informative end) via
    ``direction:rtl`` styling; the full path is mirrored to the input's
    tooltip on hover. The Browse button toggles an inline
    ``ipyfilechooser`` rooted at the current value's parent.

    Exposes ``.value`` and ``.observe('value', ...)`` so it drops in
    where ``widgets.Text`` was used.
    """

    def __init__(self, kind: str = 'dir', placeholder: str = '',
                 width: str = '420px'):
        if kind not in ('dir', 'file'):
            raise ValueError(f'kind must be "dir" or "file", got {kind!r}')
        self.kind = kind

        self.text = widgets.Text(
            placeholder=placeholder,
            layout=widgets.Layout(width=width, flex='0 0 auto'),
        )
        self.text.continuous_update = False
        self.text.add_class('jup-gui-path-input')
        self.text.observe(self._on_text_change, 'value')
        self.text.tooltip = ''

        self.browse_btn = widgets.Button(
            description='', icon='folder-open',
            tooltip=f'Browse for {kind}',
            layout=widgets.Layout(width='38px'),
        )
        self.browse_btn.on_click(self._toggle_chooser)

        self._chooser_box = widgets.VBox(
            [], layout=widgets.Layout(margin='4px 0 4px 0'),
        )
        self._chooser_open = False

        self.widget = widgets.VBox([
            widgets.HBox([self.text, self.browse_btn]),
            self._chooser_box,
        ])

    @property
    def value(self) -> str:
        return self.text.value

    @value.setter
    def value(self, v: str):
        self.text.value = v or ''

    def observe(self, handler, names='value'):
        """Forward observers to the underlying text widget."""
        self.text.observe(handler, names=names)

    def unobserve(self, handler, names='value'):
        self.text.unobserve(handler, names=names)

    def _on_text_change(self, change):
        # Mirror the full path into the tooltip so it stays readable
        # when the visible window clips.
        self.text.tooltip = change.get('new') or ''

    def _toggle_chooser(self, _b):
        if self._chooser_open:
            self._close_chooser()
            return
        if not HAS_FILECHOOSER:
            self._chooser_box.children = [widgets.HTML(
                '<small><i>install ipyfilechooser to browse</i></small>'
            )]
            self._chooser_open = True
            return
        start = os.path.dirname(self.text.value) if self.text.value else os.getcwd()
        if not os.path.isdir(start):
            start = os.getcwd()
        fc = FileChooser(
            path=start,
            show_only_dirs=(self.kind == 'dir'),
            select_default=False,
            title='',
        )
        fc.register_callback(self._on_pick)
        close_btn = widgets.Button(
            description=_UI['action_buttons']['close'],
            layout=widgets.Layout(width='80px'),
        )
        close_btn.on_click(lambda _: self._close_chooser())
        self._chooser_box.children = [fc, close_btn]
        self._chooser_open = True

    def _on_pick(self, chooser):
        picked = (
            chooser.selected_path if self.kind == 'dir' else chooser.selected
        )
        if picked:
            self.text.value = picked
        self._close_chooser()

    def _close_chooser(self):
        self._chooser_box.children = []
        self._chooser_open = False


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
                placeholder=_UI['placeholders']['browse_or_path'],
                layout=widgets.Layout(width='350px')
            )
            self._browse_btn = widgets.Button(
                description=_UI['action_buttons']['browse'],
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
                placeholder=_UI['placeholders']['enter_path'],
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

    def __init__(self, height: str = '150px', max_lines: int = 500,
                 show_by_default: bool = False):
        self._lines = []
        self._max_lines = max_lines
        self.show_debug = False
        self._html = widgets.HTML(
            value=self._render(),
            layout=widgets.Layout(border='1px solid #ccc', height=height,
                                  margin='4px 0 0 0', overflow='auto'),
        )
        # Master "show log" toggle. Log panels are hidden by default
        # to keep tabs uncluttered until the user wants the chatter.
        self.show_log_checkbox = widgets.Checkbox(
            value=show_by_default, description=_UI['action_buttons']['log_show'], indent=False,
            tooltip=_UI['tooltips']['log_show'],
            layout=widgets.Layout(margin='0 12px 0 0', width='auto'),
        )
        self.show_log_checkbox.observe(self._on_show_log_toggle, names='value')

        self.show_debug_checkbox = widgets.Checkbox(
            value=False, description=_UI['action_buttons']['log_debug'], indent=False,
            tooltip=_UI['tooltips']['log_debug'],
            layout=widgets.Layout(margin='0 12px 0 0', width='auto'),
        )
        self.show_debug_checkbox.observe(self._on_debug_toggle, names='value')

        # Plain HTML button (not ipywidgets) so the browser keeps
        # clipboard permission tied to the user's click. Icon-only
        # (fa-copy) so it stays compact on narrow toolbars.
        snippet, self._copy_uid = make_copy_to_clipboard_html(
            '', icon='copy', label='',
            tooltip=_UI['tooltips']['log_copy'],
        )
        self.copy_button = widgets.HTML(
            value=snippet,
            # margin-left:auto pushes the button flush right, opposite
            # the checkboxes on the left.
            layout=widgets.Layout(width='auto', margin='0 0 0 auto'),
        )

        toolbar = widgets.HBox(
            [
                self.show_log_checkbox,
                self.show_debug_checkbox,
                self.copy_button,
            ],
            layout=widgets.Layout(
                # overflow=visible prevents ipywidgets from wrapping
                # the toolbar in a scroll container when item widths
                # near the parent width.
                width='100%', align_items='center', margin='0',
                overflow='visible',
            ),
        )

        self.widget = widgets.VBox(
            [toolbar, self._html],
            layout=widgets.Layout(overflow='visible', width='100%'),
        )
        self._apply_show_log(show_by_default)

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
        self._refresh_copy_button()

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
        self._refresh_copy_button()

    def _on_show_log_toggle(self, change):
        self._apply_show_log(bool(change['new']))

    def _apply_show_log(self, show: bool) -> None:
        """Show/hide the bordered log area without removing it from the
        widget tree, preserving panel size and accumulated lines."""
        self._html.layout.display = '' if show else 'none'

    def set_show_log(self, value: bool) -> None:
        """Programmatic equivalent of clicking the show log checkbox."""
        value = bool(value)
        if self.show_log_checkbox.value != value:
            self.show_log_checkbox.value = value  # fires _on_show_log_toggle
        else:
            self._apply_show_log(value)

    def copy_text(self) -> str:
        """Return the log as plain text, one line per entry, honoring
        the current show_debug state."""
        out = []
        for level, msg in self._lines:
            if level == 'debug' and not self.show_debug:
                continue
            prefix = self._LEVEL_PREFIX.get(level, '')
            out.append(f'{prefix}{msg}')
        return '\n'.join(out)

    def _refresh_copy_button(self) -> None:
        """Re-render the copy button with the current log text;
        called after every append and after show_debug toggles."""
        snippet, _ = make_copy_to_clipboard_html(
            self.copy_text(), icon='copy', label='',
            tooltip=_UI['tooltips']['log_copy'],
        )
        self.copy_button.value = snippet

    def _append(self, level: str, msg):
        self._lines.append((level, str(msg)))
        if len(self._lines) > self._max_lines:
            self._lines = self._lines[-self._max_lines:]
        self._refresh()
        self._refresh_copy_button()
        if level == 'error':
            self.set_show_log(True)

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
    """Selector + stacked-parameter view for toggleable features.

    Selector on the left, parameter editor on the right, sharing one
    bordered card. The header above the selector reports how many
    features are active. The params area expands vertically for large
    features but never collapses below the selector height.
    """

    # Symmetric pre-padded glyphs so active/inactive rows align in monospace.
    _ACTIVE_PREFIX = '\u25cf  '   # filled circle + two spaces
    _INACTIVE_PREFIX = '\u25cb  '  # open circle + two spaces
    # Per-option height in the rendered <select> listbox, used to size the
    # panel so every feature is visible without scrolling. ~36 px matches
    # JupyterLab's default chrome at the .jup-gui-feature-select font size.
    _PER_OPTION_PX = 36
    # Lower bound on the panel height so a tiny feature roster doesn't
    # collapse to a sliver; also keeps the params column tall enough for
    # the title row + a short feature description on inactive features.
    _MIN_PANEL_PX = 180
    _SELECTOR_WIDTH = '170px'

    def __init__(self, features: dict):
        """
        Args:
            features: Dict mapping feature names to Feature objects
        """
        self.features = features
        self.feature_names = list(features.keys())

        # Size the panel to the actual feature roster so the listbox shows
        # every option without scrolling, and Disp's shorter roster doesn't
        # leave a long tail of empty selector space below the last item.
        # The +16 absorbs the listbox's top/bottom inner padding.
        height_px = max(
            self._MIN_PANEL_PX,
            len(features) * self._PER_OPTION_PX + 16,
        )
        self._panel_height = f'{height_px}px'

        # Compact "Features (k/N active)" header with a usage hint.
        self._header = widgets.HTML(
            value=self._render_header(),
            layout=widgets.Layout(margin='0 0 4px 2px'),
        )

        self.selector = widgets.Select(
            options=self._build_options(),
            value=self.feature_names[0] if self.feature_names else None,
            layout=widgets.Layout(
                width=self._SELECTOR_WIDTH, height=self._panel_height,
            ),
        )
        self.selector.add_class('jup-gui-feature-select')

        for name, feature in self.features.items():
            feature.active.observe(
                lambda change, n=name: self._on_feature_active_changed(n, change),
                names='value',
            )

        # Info area: clickable title (collapses description), always-visible
        # disabled banner, and the description+howto HTML that hides when
        # collapsed. Defaults flip on feature switch; activation auto-collapses.
        self._desc_collapsed: bool = False
        self._title_btn = widgets.Button(
            description='',
            layout=widgets.Layout(
                flex='1 1 auto', height='28px', padding='0',
            ),
        )
        self._title_btn.add_class('jup-gui-feature-title-toggle')
        self._title_btn.on_click(self._on_title_click)
        # The active checkbox lives here so its vertical position is fixed
        # (top-right of the params column, attached to the title row) instead
        # of moving with the description's collapse state.
        self._active_slot = widgets.HBox(
            layout=widgets.Layout(flex='0 0 auto', margin='0 4px 0 0'),
        )
        self._title_row = widgets.HBox(
            [self._title_btn, self._active_slot],
            layout=widgets.Layout(
                align_items='center', width='100%',
            ),
        )
        self._banner_html = widgets.HTML()
        self._desc_html = widgets.HTML()
        self._info_box = widgets.VBox(
            [self._title_row, self._banner_html, self._desc_html],
            layout=widgets.Layout(margin='0 0 8px 0'),
        )

        self._params_holder = widgets.VBox()

        self.params_area = widgets.VBox(
            children=[self._info_box, self._params_holder],
            layout=widgets.Layout(
                # The .jup-gui-feature-body grid already gives this column
                # a 1fr track, so flex sizing is no longer meaningful;
                # min_width=0 keeps text fields inside the grid track
                # instead of forcing the column wider.
                min_width='0',
                min_height=self._panel_height,
                padding='8px 12px',
                # No margin: selector and params share one card. The
                # divider is the params column's border-left in styles.py.
                margin='0',
            ),
        )
        self.params_area.add_class('jup-gui-feature-params')

        self.selector.observe(self._on_select, 'value')
        self._update_params_display()

        # Wrap selector + params in one bordered card.
        body = widgets.HBox(
            [self.selector, self.params_area],
            layout=widgets.Layout(
                align_items='stretch', width='100%',
            ),
        )
        body.add_class('jup-gui-feature-body')

        self.widget = widgets.VBox(
            [self._header, body],
            layout=widgets.Layout(
                # 6 px top groups the panel under the section above;
                # matches form_row's vertical rhythm below.
                margin='6px 0 4px 0',
                width='100%',
            ),
        )

    def _render_header(self) -> str:
        active = sum(1 for f in self.features.values() if f.active.value)
        total = len(self.features)
        return (
            f'<div style="font-size:12px; color:#444;">'
            f'<b>Features</b> '
            f'<span style="color:#666;">({active}/{total} active)</span> '
            f'<span style="color:#888; font-size:11px;">'
            f'&nbsp;- click a row to edit; '
            f'toggle <i>active</i> on the right to enable</span>'
            f'</div>'
        )

    def _on_feature_active_changed(self, name, change) -> None:
        """Refresh selector prefix glyphs and the header active count when
        any feature's active checkbox flips. If the toggled feature is the
        currently selected one, also flip the description collapse default
        (activate -> collapse to reclaim space for parameters; deactivate
        -> expand so the explanation re-appears).
        """
        self._refresh_selector_options()
        self._header.value = self._render_header()
        if self.selector.value == name:
            self._desc_collapsed = bool(change['new'])
            self._apply_collapse_state(self.features[name])

    def _build_options(self):
        """Return [(display_label, value)] tuples with active/inactive prefix."""
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
        if not name or name not in self.features:
            self._title_btn.description = ''
            self._banner_html.value = ''
            self._desc_html.value = ''
            self._active_slot.children = []
            self._params_holder.children = []
            return
        feature = self.features[name]
        self._banner_html.value = self._render_banner_html(feature)
        self._desc_html.value = self._render_desc_html(feature)
        # Reset on switch: default expanded when inactive (you want the
        # explanation before deciding to enable), collapsed when active
        # (you've already enabled it; reclaim the vertical space for
        # parameters). User can override either way by clicking the title.
        self._desc_collapsed = bool(feature.active.value)
        self._apply_collapse_state(feature)
        # Render the active checkbox in the title row (fixed position);
        # render only the params box below so the checkbox doesn't appear
        # twice.
        self._active_slot.children = [feature.active]
        self._params_holder.children = [feature.params_box]

    @staticmethod
    def _format_title(feature, collapsed: bool) -> str:
        chevron = '\u25b8' if collapsed else '\u25be'  # right-pointing / down-pointing triangle
        return f'{chevron}  {feature.name}'

    @staticmethod
    def _render_banner_html(feature) -> str:
        if not feature.disabled_reason:
            return ''
        return (
            '<div style="background:#fdecea; color:#a02020; '
            'border:1px solid #f5c2c0; padding:6px 8px; border-radius:4px; '
            'margin-bottom:8px; font-size:0.85em;">'
            f'<b>Disabled:</b> {feature.disabled_reason}</div>'
        )

    @staticmethod
    def _render_desc_html(feature) -> str:
        if not feature.description:
            return ''
        return (
            '<div style="color:#555; font-size:0.9em; margin-bottom:8px;">'
            f'{feature.description}</div>'
        )

    def _apply_collapse_state(self, feature):
        self._desc_html.layout.display = 'none' if self._desc_collapsed else ''
        self._title_btn.description = self._format_title(feature, self._desc_collapsed)

    def _on_title_click(self, _btn):
        name = self.selector.value
        if not name or name not in self.features:
            return
        self._desc_collapsed = not self._desc_collapsed
        self._apply_collapse_state(self.features[name])

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
