"""Single-parent-directory experiment picker.

Top-level parent-dir chooser plus a Load/Create mode toggle that swaps
between ``ExperimentList`` (searchable subdirs with ``conf/config``) and
``ExperimentWizard`` (Template + Root/Serial/Beamline + mode radio +
live folder-name preview + project-structure preview + Create button).
The template string is the single source of truth for the rendered
folder name; it persists per parent dir via
``naming.write_parent_template``.
"""

import html
import os
import re
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

import ipywidgets as widgets

from cohere_ui.jupyter_gui.header import layout, naming
from cohere_ui.jupyter_gui.header.beamlines import list_beamlines
from cohere_ui.jupyter_gui.text import load_text
from cohere_ui.jupyter_gui.widgets import (
    checkbox, dir_chooser, dropdown, form_row, text_field,
)


_MSG = load_text('messages')
_TIP = _MSG['tooltip']
_WIZ = _MSG['wizard']
_UI = load_text('ui_strings')
_PIK = _MSG['picker']

Mode = Literal['single', 'separate_scans', 'separate_scan_ranges', 'multipeak']


def _plain(s: str) -> str:
    """Strip HTML tags + collapse whitespace for use in a title= tooltip."""
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', '', s))).strip()


def _mode_tip(key: str) -> str:
    """Resolve a per-mode tooltip into plain text for title= attributes."""
    return _plain(_TIP[key])


def _input_row(label_html: str, widget, *, tooltip: str = '',
               label_width: str = '130px') -> widgets.HBox:
    """Compact label+widget row. Matches the Load section's rhythm.

    ``label_html`` is embedded verbatim (callers can pass <b>/<small> markup).
    ``tooltip`` adds a hover tooltip + dotted underline + help cursor.
    """
    title_attr = f' title="{html.escape(tooltip)}"' if tooltip else ''
    hover = (
        'cursor:help; text-decoration:underline dotted;' if tooltip else ''
    )
    label = widgets.HTML(
        value=(
            f'<span{title_attr} style="{hover} line-height:28px; '
            f'padding-right:8px; white-space:nowrap; display:inline-block;">'
            f'{label_html}</span>'
        ),
        layout=widgets.Layout(width=label_width),
    )
    return widgets.HBox(
        [label, widget],
        layout=widgets.Layout(margin='0 0 4px 0', align_items='center'),
    )


@dataclass
class WizardResult:
    """Returned from the Create wizard to ``core._create_from_wizard``."""
    parent_dir: str
    folder_name: str
    template: str
    scan: str
    beamline: str
    mode: Mode


class ExperimentList:
    """Searchable list of subdirs under a parent dir.

    Owns the editable Template field surfaced in Load mode; persisted
    per parent dir via ``naming.write_parent_template``.
    """

    SORT_OPTIONS = ('Name', 'Newest first', 'Oldest first')

    def __init__(self, on_load: Callable[[str], None]):
        self._on_load = on_load
        self._parent_dir: Optional[str] = None
        self._entries: list[tuple[str, bool, float]] = []

        # Pre-populated from <parent>/.cohere_template.json if present.
        self.template = text_field(value=naming.DEFAULT_TEMPLATE, width='340px')
        self.template.observe(self._on_template_edit, 'value')

        self._filter = widgets.Combobox(
            options=[], placeholder='Filter...',
            ensure_option=False,
            layout=widgets.Layout(width='260px'),
        )
        self._sort = dropdown(options=list(self.SORT_OPTIONS), value='Name', width='120px')
        self._show_all = checkbox('Show non-experiment subdirs', value=False)
        self._refresh_btn = widgets.Button(
            description='', icon='refresh', tooltip=_UI['tooltips']['rescan_parent'],
            layout=widgets.Layout(width='38px'),
        )
        self._select = widgets.Select(
            options=[], rows=8,
            layout=widgets.Layout(width='560px'),
        )
        self._load_btn = widgets.Button(
            description='Load experiment', button_style='warning',
            layout=widgets.Layout(width='160px'),
        )
        from cohere_ui.jupyter_gui.styles import apply_button_role
        apply_button_role(self._load_btn, 'load')
        self._load_btn.disabled = True
        self._empty_hint = widgets.HTML(value='')

        self._filter.observe(self._on_filter_change, 'value')
        self._sort.observe(self._on_filter_change, 'value')
        self._show_all.observe(self._on_filter_change, 'value')
        self._refresh_btn.on_click(lambda b: self.refresh())
        self._select.observe(self._on_select_change, 'value')
        self._load_btn.on_click(self._on_load_click)

        # template_hint is a plain-text block in YAML; keep its line
        # breaks + column alignment so the tooltip stays readable.
        template_tip = (
            f'{_TIP["template_hint"]}\n\n'
            f'{_PIK["template_saved_hint"]}'
        )
        template_row = _input_row(
            f'<b>{_PIK["template_section"]}</b>',
            self.template,
            tooltip=template_tip,
            label_width='170px',
        )

        # Filter / sort / show-all / refresh on one tight row.
        filter_row = widgets.HBox(
            [self._filter, self._sort, self._show_all, self._refresh_btn],
            layout=widgets.Layout(
                margin='0 0 2px 0', align_items='center',
                justify_content='flex-start',
            ),
        )

        action_row = widgets.HBox(
            [self._load_btn, self._empty_hint],
            layout=widgets.Layout(margin='4px 0 0 0', align_items='center'),
        )

        self.widget = widgets.VBox(
            [template_row, filter_row, self._select, action_row],
            layout=widgets.Layout(margin='0'),
        )

    def set_parent_dir(self, parent_dir: Optional[str]):
        self._parent_dir = parent_dir
        if parent_dir:
            stored = naming.read_parent_template(parent_dir)
            new_value = stored if stored else naming.DEFAULT_TEMPLATE
        else:
            new_value = naming.DEFAULT_TEMPLATE
        if self.template.value != new_value:
            try:
                self.template.unobserve(self._on_template_edit, 'value')
            except (ValueError, RuntimeError):
                pass
            self.template.value = new_value
            self.template.observe(self._on_template_edit, 'value')
        self.refresh()

    def _on_template_edit(self, change):
        if self._parent_dir and change.get('new'):
            naming.write_parent_template(self._parent_dir, change['new'])

    @property
    def template_value(self) -> str:
        return self.template.value or naming.DEFAULT_TEMPLATE

    def refresh(self):
        self._entries = []
        if self._parent_dir and os.path.isdir(self._parent_dir):
            try:
                for entry in os.listdir(self._parent_dir):
                    full = os.path.join(self._parent_dir, entry)
                    if not os.path.isdir(full) or entry.startswith('.'):
                        continue
                    has_conf = os.path.isfile(os.path.join(full, 'conf', 'config'))
                    try:
                        mtime = os.path.getmtime(full)
                    except OSError:
                        mtime = 0.0
                    self._entries.append((entry, has_conf, mtime))
            except OSError:
                pass
        self._reapply_filter()

    def _on_filter_change(self, _change):
        self._reapply_filter()

    def _reapply_filter(self):
        filter_text = (self._filter.value or '').lower()
        sort = self._sort.value
        show_all = self._show_all.value

        items = [
            e for e in self._entries
            if (show_all or e[1]) and filter_text in e[0].lower()
        ]
        if sort == 'Newest first':
            items.sort(key=lambda e: -e[2])
        elif sort == 'Oldest first':
            items.sort(key=lambda e: e[2])
        else:
            items.sort(key=lambda e: e[0].lower())

        options = [(self._format_option(name, has_conf), name) for name, has_conf, _ in items]
        try:
            self._select.unobserve(self._on_select_change, 'value')
        except (ValueError, RuntimeError):
            pass
        self._select.options = options
        if options:
            self._select.value = options[0][1]
        self._select.observe(self._on_select_change, 'value')
        self._filter.options = [name for name, _, _ in self._entries]
        self._update_hint(items)
        self._on_select_change({'new': self._select.value})

    @staticmethod
    def _format_option(name: str, has_conf: bool) -> str:
        return f'{"●" if has_conf else "○"} {name}'

    def _update_hint(self, items: list):
        if not self._parent_dir:
            key = 'empty_no_parent'
        elif not self._entries:
            key = 'empty_no_subdirs'
        elif not items:
            key = 'empty_no_matches'
        else:
            self._empty_hint.value = ''
            return
        self._empty_hint.value = f'<i style="color:#888;">{_PIK[key]}</i>'

    def _on_select_change(self, change):
        name = change.get('new') if isinstance(change, dict) else None
        if not name or not self._parent_dir:
            self._load_btn.disabled = True
            return
        full = os.path.join(self._parent_dir, name)
        self._load_btn.disabled = not os.path.isfile(os.path.join(full, 'conf', 'config'))

    def _on_load_click(self, _b):
        name = self._select.value
        if not name or not self._parent_dir:
            return
        full = os.path.join(self._parent_dir, name)
        self._on_load(full)

    def select(self, folder_name: str):
        """Pre-select ``folder_name`` to mirror state after a load."""
        for label, value in self._select.options:
            if value == folder_name:
                self._select.value = value
                return


class ExperimentWizard:
    """Create-new wizard: template + value inputs + mode radio.

    The template string is the single source of truth (edit it directly
    to drop a token or change its ``:fmt``). The ``Serial`` field
    accepts either an integer (auto-bumps for uniqueness when the
    template uses an int format spec) or a scan-range string like
    ``"2-7,10-15"`` (used verbatim, also written to ``config.scan``).
    """

    _MODE_OPTIONS = [
        ('Single experiment', 'single'),
        ('Separate scans', 'separate_scans'),
        ('Separate scan ranges', 'separate_scan_ranges'),
        ('Multi-peak', 'multipeak'),
    ]

    def __init__(self, on_create: Callable[['WizardResult'], None]):
        self._on_create = on_create
        self._parent_dir: Optional[str] = None
        # Bound externally to ExperimentList.template by attach_template()
        # so Load-mode edits flow into Create-mode.
        self._template_source: Optional[widgets.Text] = None

        self.root = text_field(placeholder='e.g., cdi', width='260px')
        self.serial = text_field(
            placeholder='e.g., 1 or 54 or 2-7,10-15', width='260px',
        )
        # InstrTab still observes main_gui.scan; alias kept for back-compat.
        self.scan = self.serial
        self.beamline = widgets.Combobox(
            options=list_beamlines(),
            placeholder='Search beamlines...',
            ensure_option=True,
            layout=widgets.Layout(width='260px'),
        )
        self.mode = widgets.ToggleButtons(
            options=self._MODE_OPTIONS,
            value='single',
            tooltips=[
                _mode_tip('single'),
                _mode_tip('separate_scans'),
                _mode_tip('separate_scan_ranges'),
                _mode_tip('multipeak'),
            ],
            layout=widgets.Layout(margin='2px 0'),
        )

        self.preview = widgets.HTML(value='')
        self.layout_preview = widgets.HTML(value='')
        # Layout preview is collapsed by default to keep the wizard
        # compact. Show-preview checkbox toggles `layout.display`.
        self.layout_preview.layout.display = 'none'
        self.show_layout_preview = widgets.Checkbox(
            value=False, description='Show files & folders preview',
            indent=False,
            layout=widgets.Layout(margin='4px 0 0 0', width='auto'),
        )
        self.show_layout_preview.observe(
            self._on_show_layout_preview_change, 'value'
        )
        self.create_btn = widgets.Button(
            description='Create experiment', button_style='success',
            layout=widgets.Layout(width='180px'),
        )
        from cohere_ui.jupyter_gui.styles import apply_button_role
        apply_button_role(self.create_btn, 'set')
        self.create_btn.disabled = True

        for w in (self.root, self.serial):
            w.observe(self._refresh_preview, 'value')
        self.mode.observe(self._refresh_preview, 'value')
        self.create_btn.on_click(self._on_create_click)

        # Layout is filled in by attach_template() once the shared
        # template source is bound (called immediately after construction).
        self.widget = widgets.VBox([])

    def attach_template(self, template_widget: widgets.Text):
        """Bind the wizard to a shared template Text widget so Load-mode
        edits flow into Create-mode."""
        self._template_source = template_widget
        template_widget.observe(self._refresh_preview, 'value')

        # template_hint is a plain-text block in YAML; keep its line
        # breaks + column alignment so the tooltip stays readable.
        template_tip = (
            f'{_TIP["template_hint"]}\n\n'
            f'{_PIK["template_saved_hint"]}'
        )
        heading_html = (
            f'<span title="{html.escape(_plain(_WIZ["heading_note"]))}" '
            f'style="cursor:help; text-decoration:underline dotted;">'
            f'<b>{_WIZ["heading"]}</b></span>'
        )

        label_w = '170px'
        self.widget.children = (
            widgets.HTML(
                heading_html,
                layout=widgets.Layout(margin='0 0 6px 0'),
            ),
            _input_row(
                f'<b>{_PIK["template_section"]}</b>',
                template_widget,
                tooltip=template_tip,
                label_width=label_w,
            ),
            _input_row('Root', self.root, label_width=label_w),
            _input_row(
                'Serial / scan', self.serial,
                tooltip=_WIZ['serial_hint'],
                label_width=label_w,
            ),
            _input_row('Beamline', self.beamline, label_width=label_w),
            _input_row(
                f'<b>{_WIZ["mode_section"]}</b>', self.mode,
                label_width=label_w,
            ),
            self.preview,
            self.show_layout_preview,
            self.layout_preview,
            widgets.HBox(
                [self.create_btn],
                layout=widgets.Layout(margin='6px 0 0 0'),
            ),
        )
        self._refresh_preview(None)

    def _on_show_layout_preview_change(self, change):
        """Reveal/hide the files-and-folders preview on toggle."""
        self.layout_preview.layout.display = '' if change['new'] else 'none'

    def set_parent_dir(self, parent_dir: Optional[str]):
        self._parent_dir = parent_dir
        self._refresh_preview(None)

    def prefill(self, *, serial='', beamline='', mode: Mode = 'single'):
        """Pre-fill wizard fields after a load (for sibling creation)."""
        self.serial.value = serial or ''
        if beamline:
            if beamline not in self.beamline.options:
                self.beamline.options = list_beamlines()
            if beamline in self.beamline.options:
                self.beamline.value = beamline
        if mode not in {opt[1] for opt in self._MODE_OPTIONS}:
            mode = 'single'
        self.mode.value = mode
        # Force the folder-name + layout preview to reflect the loaded
        # state in case the observers didn't fire for unchanged values.
        self._refresh_preview(None)

    @property
    def template_value(self) -> str:
        if self._template_source is None:
            return naming.DEFAULT_TEMPLATE
        return self._template_source.value or naming.DEFAULT_TEMPLATE

    def _compute_folder_name(self) -> tuple[str, bool, Union[int, str]]:
        """Return (folder_name, collides, resolved_serial).

        ``resolved_serial`` is the integer or string ``next_serial``
        resolved to, suitable for substituting back into ``{serial}``
        and for writing to ``config.scan``.
        """
        tmpl = self.template_value
        root = self.root.value or ''
        user_serial = (self.serial.value or '').strip()
        if self._parent_dir:
            resolved = naming.next_serial(
                self._parent_dir, tmpl, root=root, serial=user_serial,
            )
        else:
            resolved = user_serial or 1
            if isinstance(resolved, str) and resolved.isdigit():
                resolved = int(resolved)
        try:
            rendered = naming.render(tmpl, root=root, serial=resolved)
        except (KeyError, IndexError, ValueError) as e:
            return f'<error: {e}>', False, resolved
        collides = (
            self._parent_dir is not None
            and naming.folder_exists(self._parent_dir, rendered)
        )
        return rendered, collides, resolved

    def _refresh_preview(self, _change):
        if not self._parent_dir:
            self.preview.value = (
                f'<i style="color:#888;">{_WIZ["parent_dir_prompt"]}</i>'
            )
            self.layout_preview.value = ''
            self.create_btn.disabled = True
            return
        rendered, collides, resolved = self._compute_folder_name()

        warn_html = ''
        unknown = naming.unknown_tokens(self.template_value)
        if unknown:
            warn_html = (
                f'<div style="color:#a06000; font-family:Menlo,Consolas,monospace;">'
                f'{_WIZ["unknown_tokens"].format(names=", ".join(unknown))}</div>'
            )

        bump_note = ''
        # Auto-bumped integer serial above 1 means we overrode the user.
        user_serial = (self.serial.value or '').strip()
        if isinstance(resolved, int) and resolved > 1 and (
            not user_serial or (user_serial.isdigit() and int(user_serial) < resolved)
        ):
            bump_note = (
                f' <span style="color:#888;">'
                f'{_WIZ["serial_bumped"].format(next=resolved)}</span>'
            )

        if collides:
            self.preview.value = (
                f'{warn_html}'
                f'<div style="font-family:Menlo,Consolas,monospace;">'
                f'<b>Folder name:</b> {rendered}{bump_note}<br>'
                f'<span style="color:#a02020;">{_WIZ["collision"]}</span>'
                f'</div>'
            )
            self.create_btn.disabled = True
        else:
            self.preview.value = (
                f'{warn_html}'
                f'<div style="font-family:Menlo,Consolas,monospace;">'
                f'<b>Folder name:</b> {rendered}{bump_note}'
                f'</div>'
            )
            self.create_btn.disabled = not (rendered and self._parent_dir)

        self.layout_preview.value = self._render_layout_preview(rendered, resolved)

    def _render_layout_preview(self, folder_name: str, resolved_serial) -> str:
        """Render the dir-tree sketch from ``layout.project_layout``."""
        mode = self.mode.value
        scan_str = (
            self.serial.value.strip()
            if isinstance(resolved_serial, str)
            else str(resolved_serial)
        )
        # Single mode doesn't expand by scan; safe to pass an empty string.
        if mode == 'single':
            scan_str = ''
        try:
            entries = layout.project_layout(
                folder_name=folder_name, scan=scan_str, mode=mode,
            )
        except ValueError:
            return (
                f'<div style="color:#a06000;">{_WIZ["scan_unparseable"]}</div>'
            )
        tree_text = layout.format_tree(entries, root=folder_name)
        return (
            f'<div style="background:#fafafa; padding:6px 10px; '
            f'border:1px solid #eee; border-radius:4px; margin:6px 0;">'
            f'<div style="color:#666; font-style:italic; font-size:11px; '
            f'margin-bottom:4px;">{_WIZ["layout_preview_heading"]}</div>'
            f'<pre style="margin:0; padding:0; max-height:300px; overflow:auto; '
            f'font-family:Menlo,Consolas,monospace; font-size:11px; '
            f'line-height:1.35; color:#444;">'
            f'{html.escape(tree_text)}</pre>'
            f'</div>'
        )

    def _on_create_click(self, _b):
        if not self._parent_dir:
            return
        rendered, collides, resolved = self._compute_folder_name()
        if collides or not rendered:
            return
        # Backend reads config.scan; populate from the user's serial input
        # when it's a scan-range string, otherwise from the resolved int.
        user_serial = (self.serial.value or '').strip()
        scan_value = user_serial if user_serial else str(resolved)
        result = WizardResult(
            parent_dir=self._parent_dir,
            folder_name=rendered,
            template=self.template_value,
            scan=scan_value,
            beamline=self.beamline.value.strip(),
            mode=self.mode.value,
        )
        self._on_create(result)


class ExperimentPicker:
    """Parent-dir chooser + Load/Create mode toggle + Stack of subpanels.

    Owns a right-anchored path-mirror HTML label below the chooser
    (CSS ``direction: rtl`` to keep the tail visible on deep paths,
    plus ``title=`` for the full path on hover).
    """

    def __init__(
        self,
        *,
        start_path: str,
        on_load: Callable[[str], None],
        on_create: Callable[['WizardResult'], None],
        log_panel=None,
    ):
        self._on_load = on_load
        self._on_create = on_create
        self._log_panel = log_panel

        # No `title=` on the chooser; the section's bold header labels it.
        self.parent_chooser = dir_chooser(
            start_path=start_path or os.getcwd(),
            title='',
        )
        # Hide ipyfilechooser's own selected-path label; the RTL path
        # mirror below the chooser shows the same info with better
        # tail-visibility on deep paths.
        try:
            self.parent_chooser._fc._label.layout.display = 'none'
        except AttributeError:
            pass
        self._path_mirror = widgets.HTML(value=self._render_path_mirror(''))

        self.mode_toggle = widgets.RadioButtons(
            options=[('Load existing', 'load'), ('Create new', 'create')],
            value='load', orientation='horizontal',
            layout=widgets.Layout(margin='4px 0'),
        )

        self.list = ExperimentList(on_load=self._handle_load)
        self.wizard = ExperimentWizard(on_create=self._handle_create)
        # Share the list's template widget so Load and Create render
        # from the same source.
        self.wizard.attach_template(self.list.template)

        self._stack = widgets.Stack(
            children=[self.list.widget, self.wizard.widget],
            selected_index=0,
            layout=widgets.Layout(margin='4px 0'),
        )

        self.mode_toggle.observe(self._on_mode_change, 'value')
        self.parent_chooser.register_callback(self._on_parent_change)

        initial = self.parent_chooser.value or start_path or os.getcwd()
        self._propagate_parent(initial)

        self.widget = widgets.VBox([
            widgets.HTML(
                f'<b>{_PIK["parent_section"]}</b> '
                f'<small style="color:#888;">{_PIK["parent_section_note"]}</small>'
            ),
            self.parent_chooser.widget,
            self._path_mirror,
            self.mode_toggle,
            self._stack,
        ])

    @staticmethod
    def _render_path_mirror(path: str) -> str:
        """Right-anchored display of ``path`` (tail visible, head ellipsed)."""
        if not path:
            return (
                f'<div style="color:#888; font-size:11px; padding:0 4px 4px 4px;">'
                f'<i>{_PIK["path_mirror_empty"]}</i></div>'
            )
        escaped = html.escape(path)
        # direction:rtl + text-align:left puts the ellipsis on the head
        # instead of the tail.
        return (
            f'<div title="{escaped}" '
            'style="font-family:Menlo,Consolas,monospace; font-size:11px; '
            'padding:0 4px 4px 4px; color:#555; '
            'white-space:nowrap; overflow:hidden; text-overflow:ellipsis; '
            'direction:rtl; text-align:left; max-width:520px;">'
            f'{escaped}</div>'
        )

    @property
    def parent_dir(self) -> str:
        return self.parent_chooser.value or ''

    def _propagate_parent(self, parent_dir: str):
        self.list.set_parent_dir(parent_dir)
        self.wizard.set_parent_dir(parent_dir)
        self._path_mirror.value = self._render_path_mirror(parent_dir or '')

    def _on_parent_change(self, new_value):
        self._propagate_parent(new_value)

    def _on_mode_change(self, change):
        self._stack.selected_index = 0 if change['new'] == 'load' else 1

    def _handle_load(self, full_path: str):
        self._on_load(full_path)

    def _handle_create(self, result: WizardResult):
        # Persist the template the user just used for this parent dir.
        if result.parent_dir and result.template:
            naming.write_parent_template(result.parent_dir, result.template)
        self._on_create(result)

    def set_loaded(self, *, parent_dir: str, folder_name: str,
                   serial: str = '', beamline: str = '',
                   mode: Mode = 'single'):
        """Reflect a freshly-loaded experiment in the picker state."""
        if parent_dir and parent_dir != self.parent_dir:
            self.parent_chooser.value = parent_dir
            self._propagate_parent(parent_dir)
        self.mode_toggle.value = 'load'
        self.list.select(folder_name)
        self.wizard.prefill(serial=serial, beamline=beamline, mode=mode)

    def reset(self):
        """Clear loaded state. Parent dir is preserved."""
        self.wizard.prefill()
        self.list.refresh()
