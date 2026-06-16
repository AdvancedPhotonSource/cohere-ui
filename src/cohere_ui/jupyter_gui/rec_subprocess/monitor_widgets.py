"""Widget tree, log and image rendering, and image-size toolbar for RecMonitor.

This module holds no lifecycle state. ``RecMonitor`` owns the subprocess
handle, queues, threads, and exit-code bookkeeping, while
``RecMonitorWidgets`` owns only what is on screen.

The widget attributes are deliberately public (no underscore) because
:class:`RecMonitor` hoists them onto itself, so the existing
``monitor.progress_label`` and ``monitor.show_debug_checkbox`` access
patterns from RecTab and CoherenceGUI keep working unchanged.
"""

import sys
import traceback

import ipywidgets as widgets

from cohere_ui.jupyter_gui.rec_subprocess.log_view import render_log_html
from cohere_ui.jupyter_gui.text import load_text
from cohere_ui.jupyter_gui.widgets import make_copy_to_clipboard_html, log_box_height

_UI = load_text('ui_strings')


class RecMonitorWidgets:
    """Owns every ipywidget RecMonitor displays. Pure UI state."""

    LOG_MAX_LINES = 1000

    # Small / Medium / Large in-tab image sizes selectable from the
    # shared toolbar dropdown. M is the default and the only one that
    # uses flex-grow so two images sit side-by-side; S / L pin
    # max-width so they keep their intrinsic proportions.
    _IMAGE_SIZE_OPTIONS = {
        # M's calc() subtracts half the container gap (6 px) plus a 2 px
        # safety margin to avoid wrapping to a second row.
        'S': {'min_height': '180px', 'max_width': '420px', 'flex': '0 1 auto'},
        'M': {'min_height': '260px', 'max_width': 'calc(50% - 10px)', 'flex': '1 1 0'},
        'L': {'min_height': '360px', 'max_width': '900px', 'flex': '0 1 auto'},
    }

    def __init__(self):
        # progress label + bar
        self.progress_label = widgets.HTML(value=_UI['status']['idle'])
        self.progress_bar = widgets.IntProgress(
            value=0, min=0, max=1,
            bar_style='info',
            layout=widgets.Layout(width='100%', margin='4px 0 4px 0'),
        )
        self.progress_bar.layout.visibility = 'hidden'

        # live view + error plot
        self.live_view_caption = widgets.HTML(
            value=_UI['snapshot_panel']['idle'],
        )
        self.live_view = widgets.Image(
            format='png',
            layout=widgets.Layout(border='1px solid var(--jup-border)', min_height='180px',
                                  max_width='420px'),
        )
        self.error_plot = widgets.Image(
            format='png',
            layout=widgets.Layout(border='1px solid var(--jup-border)', min_height='160px',
                                  max_width='420px'),
        )
        # Hidden until they hold real bytes: an empty widgets.Image renders
        # as an empty bordered box, so a never-used live view / error plot
        # would otherwise sit on screen as a placeholder. set_live_view /
        # set_error_plot reveal each one when (and only when) it gets content.
        self.live_view.layout.display = 'none'
        self.error_plot.layout.display = 'none'
        self.images_row = widgets.HBox(
            [self.live_view, self.error_plot],
            layout=widgets.Layout(
                width='100%',
                flex_flow='row wrap',
                align_items='flex-start',
                margin='4px 0 0 0',
            ),
        )
        self.images_row.add_class('jup-gui-rec-images')
        self.image_toolbar = self._make_shared_image_toolbar()
        # The size selector is only meaningful once an image is showing.
        self.image_toolbar.layout.display = 'none'

        # log widget + toolbar
        # Compact by default (~5 lines), grows to ~10 when debug is on
        # (see _on_debug_toggle), and is drag-resizable via .jup-gui-log-box.
        self._log_h_normal = log_box_height(5)
        self._log_h_debug = log_box_height(10)
        self.log_widget = widgets.HTML(
            value=render_log_html([]),
            layout=widgets.Layout(border='1px solid var(--jup-border)', height=self._log_h_normal,
                                  margin='4px 0 0 0', overflow='hidden'),
        )
        self.log_widget.add_class('jup-gui-log-box')
        self._make_copy_html = make_copy_to_clipboard_html
        self.show_log_checkbox = widgets.Checkbox(
            value=False, description=_UI['action_buttons']['log_show'], indent=False,
            tooltip=_UI['tooltips']['log_show_rec'],
            layout=widgets.Layout(margin='0 12px 0 0', width='auto'),
        )
        self.show_log_checkbox.observe(self._on_show_log_toggle, names='value')
        self.show_debug = False
        self.show_debug_checkbox = widgets.Checkbox(
            value=False, description=_UI['action_buttons']['log_debug'], indent=False,
            tooltip=_UI['tooltips']['log_debug'],
            layout=widgets.Layout(margin='0 12px 0 0', width='auto'),
        )
        self.show_debug_checkbox.observe(self._on_debug_toggle, names='value')
        snippet, self._copy_log_uid = self._make_copy_html(
            '', icon='copy', label='',
            tooltip=_UI['tooltips']['log_copy_rec'],
        )
        self.copy_log_button = widgets.HTML(
            value=snippet,
            layout=widgets.Layout(width='auto', margin='0 0 0 auto'),
        )
        self.log_toolbar = widgets.HBox(
            [
                self.show_log_checkbox,
                self.show_debug_checkbox,
                self.copy_log_button,
            ],
            layout=widgets.Layout(
                width='100%', align_items='center', margin='4px 0 0 0',
                overflow='visible',
            ),
        )
        self.log_widget.layout.display = 'none'

        self._log_lines: list = []

    # public stack

    def widgets_box(self) -> widgets.VBox:
        """Return the monitor's widgets stacked in display order."""
        return widgets.VBox([
            self.progress_label,
            self.progress_bar,
            self.live_view_caption,
            self.image_toolbar,
            self.images_row,
            self.log_toolbar,
            self.log_widget,
        ])

    # log API used by RecMonitor

    def append(self, msg, level: str = 'info') -> None:
        """Append a line to the log widget at ``level``.

        Auto-reveals the log panel on the first ``error`` entry so the
        user sees the failure even if they hadn't expanded the log yet.
        """
        self._log_lines.append((level, str(msg)))
        if len(self._log_lines) > self.LOG_MAX_LINES:
            self._log_lines = self._log_lines[-self.LOG_MAX_LINES:]
        self._refresh_log()
        if level == 'error' and not self.show_log_checkbox.value:
            self.show_log_checkbox.value = True

    def clear(self) -> None:
        """Clear the log widget."""
        self._log_lines = []
        self._refresh_log()

    def set_show_debug(self, value: bool) -> None:
        value = bool(value)
        if self.show_debug == value:
            return
        self.show_debug = value
        if self.show_debug_checkbox.value != value:
            self.show_debug_checkbox.value = value
        self._apply_log_height()
        self._refresh_log()

    def _apply_log_height(self) -> None:
        """Grow the log box while debug output is shown, shrink back otherwise."""
        self.log_widget.layout.height = (
            self._log_h_debug if self.show_debug else self._log_h_normal
        )

    # live view / error plot visibility

    def set_live_view(self, png) -> None:
        """Set the live-view image and reveal it only when it has bytes."""
        self.live_view.value = png or b''
        self._sync_image_visibility()

    def set_error_plot(self, png) -> None:
        """Set the error-history plot and reveal it only when it has bytes."""
        self.error_plot.value = png or b''
        self._sync_image_visibility()

    def _sync_image_visibility(self) -> None:
        """Show each image (and the shared size toolbar) only when populated.

        Per-image rather than per-row: the error plot fills every accepted
        iteration while the live view only fills on snapshot events, so each
        appears independently and an unused one never shows an empty box.
        """
        has_live = bool(self.live_view.value)
        has_err = bool(self.error_plot.value)
        self.live_view.layout.display = '' if has_live else 'none'
        self.error_plot.layout.display = '' if has_err else 'none'
        self.image_toolbar.layout.display = '' if (has_live or has_err) else 'none'

    # internal handlers

    def _make_shared_image_toolbar(self) -> widgets.HBox:
        """Shared image-size selector above the live_view + error_plot stack."""
        self.image_size_dropdown = widgets.Dropdown(
            options=list(self._IMAGE_SIZE_OPTIONS),
            value='M',
            description='Image size:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='160px'),
        )
        self.image_size_dropdown.observe(self._apply_image_size, 'value')
        # Apply M immediately so the images render with the half-width
        # layout on first display, not the bare widget defaults.
        self._apply_image_size({'new': self.image_size_dropdown.value})

        return widgets.HBox(
            [self.image_size_dropdown],
            layout=widgets.Layout(
                align_items='center', margin='4px 0 4px 0',
            ),
        )

    def _apply_image_size(self, change) -> None:
        opts = self._IMAGE_SIZE_OPTIONS[change['new']]
        for img in (self.live_view, self.error_plot):
            img.layout.min_height = opts['min_height']
            img.layout.max_width = opts['max_width']
            img.layout.flex = opts.get('flex', '')

    def _on_debug_toggle(self, change) -> None:
        # Fires for both direct clicks and the linked master toggle
        # (core._sync_debug_panels), so the height swap lives here.
        self.show_debug = bool(change['new'])
        self._apply_log_height()
        self._refresh_log()

    def _on_show_log_toggle(self, change) -> None:
        """Reveal or hide the log panel without losing its contents."""
        self.log_widget.layout.display = '' if change['new'] else 'none'

    def _log_copy_text(self) -> str:
        """Plain-text log dump (one line per entry, respecting show_debug).

        Uses the same level prefixes as the rendered view so clipboard
        output matches what's on screen.
        """
        prefix = {
            'info': '', 'success': '[OK] ',
            'warning': '[WARN] ', 'error': '[ERROR] ',
            'debug': '[DEBUG] ',
        }
        out = []
        for level, msg in self._log_lines:
            if level == 'debug' and not self.show_debug:
                continue
            out.append(f'{prefix.get(level, "")}{msg}')
        return '\n'.join(out)

    def _refresh_copy_log_button(self) -> None:
        """Re-render the copy button HTML with the current log text."""
        snippet, _ = self._make_copy_html(
            self._log_copy_text(), icon='copy', label='',
            tooltip=_UI['tooltips']['log_copy_rec'],
        )
        self.copy_log_button.value = snippet

    def _refresh_log(self) -> None:
        try:
            self.log_widget.value = render_log_html(
                self._log_lines, show_debug=self.show_debug,
            )
        except Exception as e:
            sys.stderr.write(
                f"RecMonitorWidgets._refresh_log failed: "
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
        try:
            self._refresh_copy_log_button()
        except Exception as e:
            sys.stderr.write(
                f"RecMonitorWidgets._refresh_copy_log_button failed: "
                f"{type(e).__name__}: {e}\n"
            )
