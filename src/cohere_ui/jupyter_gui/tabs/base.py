"""Base class for all tab implementations."""

import functools
import os
import traceback
from abc import ABC, abstractmethod
from typing import Optional

import ipywidgets as widgets

from cohere_ui.jupyter_gui._validation import (
    ValidationError, validate_field, validate_paths,
)
from cohere_ui.jupyter_gui.utils.file_events import diff as file_diff, snapshot as file_snapshot
from cohere_ui.jupyter_gui.utils.error_format import format_error_summary, safe_parse
from cohere_ui.jupyter_gui.text import load_text
from cohere_ui.jupyter_gui.widgets import SaveButton, SplitRunButton, button

_MSG = load_text('messages')


_FORM_WIDGET_TYPES = (
    widgets.Text, widgets.Checkbox, widgets.Dropdown,
    widgets.IntText, widgets.FloatText, widgets.Combobox,
    widgets.RadioButtons, widgets.SelectMultiple, widgets.Textarea,
)


def _walk_widgets(w):
    """Yield the widget and every descendant, depth-first."""
    yield w
    for child in getattr(w, 'children', ()) or ():
        yield from _walk_widgets(child)


class BaseTab(ABC):
    """Standard tab interface.

    Extending: to add a tab, subclass BaseTab and implement
      - name: display name shown on the tab header
      - conf_name: config file name (e.g., 'config_rec')
      - _build_ui(): build the widget tree
      - load_tab(conf_map): populate widgets from a config dict
      - get_config(): read widgets back into a config dict
      - clear_conf(): reset widgets to defaults
      - run_tab(): launch the tab's backend step

    For a styled, auto-scrolling status log, create a ``LogPanel`` in
    ``_build_ui`` and assign it to ``self.log_panel``; the
    ``log_info`` / ``log_success`` / ``log_warning`` / ``log_error``
    helpers route to it. Tabs with a different log surface (RecTab,
    via its monitor) override ``log()`` and ``clear_output()`` instead;
    the level helpers fall back to plain ``log()`` in that case.
    """

    name: str = "Tab"
    conf_name: str = "config"

    def __init__(self):
        self.main_gui = None
        self.log_panel = None
        self._widget = None
        # Populated by _build_action_row() for the standard [Load][Save][Run]
        # row; refresh_action_state silently no-ops if a tab skips the helper.
        self.save_button: Optional[SaveButton] = None
        self.split_run: Optional[SplitRunButton] = None
        self.load_btn: Optional[widgets.Button] = None

    def init(self, main_gui):
        """Initialize the tab with reference to main GUI."""
        self.main_gui = main_gui
        self._widget = self._build_ui()
        self._install_state_observers()

    @staticmethod
    def _guard(fn):
        """Route widget-callback exceptions to the tab's log panel
        instead of letting them disappear into the widget event loop.

        Apply to every ``observe`` / ``on_click`` handler so a missing
        file or parser error becomes a visible ``[ERROR]`` + ``[DEBUG]``
        pair rather than a frozen UI.
        """
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            try:
                return fn(self, *args, **kwargs)
            except Exception as e:
                self.log_error(format_error_summary(e, prefix=fn.__name__))
                self.log_debug(traceback.format_exc())
        return wrapper

    @staticmethod
    def _guard(fn):
        """Decorator that surfaces widget-callback exceptions to the tab's
        log panel instead of letting them die silently in ipywidgets'
        event loop. Wrap ``observe`` / ``on_click`` handlers with this so
        a missing file or a parser error becomes a visible ``[ERROR]``
        plus ``[DEBUG]`` pair rather than a frozen UI."""
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            try:
                return fn(self, *args, **kwargs)
            except Exception as e:
                self.log_error(format_error_summary(e, prefix=fn.__name__))
                self.log_debug(traceback.format_exc())
        return wrapper

    @property
    def widget(self) -> widgets.Widget:
        """The tab's root widget."""
        if self._widget is None:
            self._widget = self._build_ui()
        return self._widget

    @abstractmethod
    def _build_ui(self) -> widgets.Widget:
        """Build and return the tab's widget tree."""
        pass

    @abstractmethod
    def load_tab(self, conf_map: dict):
        """Populate widgets from configuration dictionary."""
        pass

    @abstractmethod
    def get_config(self) -> dict:
        """Read current widget values into a configuration dictionary."""
        pass

    @abstractmethod
    def clear_conf(self):
        """Reset all widgets to their default/empty state."""
        pass

    def save_conf(self) -> str:
        """Save current configuration to file. Returns error string (empty when successful).

        Reports the create/update action through the tab's logging.
        """
        if not self.main_gui or not self.main_gui.experiment_exists():
            return _MSG['tab']['experiment_not_set']
        try:
            conf_map = self.get_config()
            err, action = self.main_gui.config_manager.save_config(
                self.conf_name,
                conf_map,
                no_verify=self.main_gui.no_verify
            )
            if action:
                self._log_config_action(action)
                self._notify_save()
            return err
        except Exception as e:
            self.log_error(format_error_summary(e, prefix='save_conf'))
            self.log_debug(traceback.format_exc())
            return str(e) or type(e).__name__

    def save_and_verify(self) -> str:
        """Get config, validate, verify, save. Returns error string ('' on success).

        Runs the local field validators and path-existence check before
        the cohere_core verifier so a bad entry surfaces with its field
        name and a suggestion, not as a generic backend error later.

        Subclasses that need to mutate the config map between validation
        and write (e.g. PrepTab preserving ``outliers_scans`` from disk)
        override ``_pre_save_hook(conf_map)`` rather than reimplementing
        this method.
        """
        try:
            conf_map = self.get_config()
            field_errors = self._validate_fields(conf_map)
            if field_errors:
                for e in field_errors:
                    self.log_error(str(e))
                return field_errors[0].message
            err = self.main_gui.config_manager.verify(self.conf_name, conf_map)
            if err and not self.main_gui.no_verify:
                self.log_error(_MSG['tab']['config_error'].format(error=err))
                return err
            self._pre_save_hook(conf_map)
            _, action = self.main_gui.config_manager.save_config(
                self.conf_name, conf_map, self.main_gui.no_verify)
            if action:
                self._log_config_action(action)
                self._notify_save()
            return ""
        except Exception as e:
            self.log_error(format_error_summary(e, prefix='save_and_verify'))
            self.log_debug(traceback.format_exc())
            return str(e) or type(e).__name__

    def _pre_save_hook(self, conf_map: dict) -> None:
        """Mutate ``conf_map`` in place just before the on-disk save.

        Default no-op. Override in tabs that need to preserve or inject
        fields the form does not expose (e.g. PrepTab keeps the
        ``outliers_scans`` list the prep backend writes back).
        """
        return None

    def _validate_fields(self, conf_map: dict) -> list:
        """Run field validators, path-existence checks, and per-feature
        ``verify_active`` (when the tab owns a ``self.features`` dict).

        Returns a list of ValidationError (possibly empty). Skipped when
        the main GUI has ``no_verify`` set.
        """
        if conf_map is None or getattr(self.main_gui, 'no_verify', False):
            return []
        errors = []
        for key, value in conf_map.items():
            err = validate_field(key, value)
            if err is not None:
                errors.append(err)
        errors.extend(validate_paths(conf_map, self.conf_name))
        errors.extend(self._collect_feature_errors())
        return errors

    def _collect_feature_errors(self) -> list:
        """Run ``verify_active`` on every feature the tab owns.

        No-op for tabs without a ``self.features`` dict. Inactive
        features return ``""`` from the base ``verify_active`` and are
        filtered out here.
        """
        features = getattr(self, 'features', None)
        if not features:
            return []
        errs = []
        for name, feat in features.items():
            msg = (feat.verify_active() or "").strip()
            if msg:
                errs.append(ValidationError(name, msg))
        return errs

    def _notify_save(self):
        """Tell the main GUI that this tab's on-disk config changed.

        Safe whether or not the main GUI exposes a status strip.
        """
        if self.main_gui is None:
            return
        notify = getattr(self.main_gui, '_notify_save_complete', None)
        if notify is not None:
            try:
                notify(self.conf_name)
            except Exception as e:
                # Save succeeded; only the status-strip refresh failed.
                self.log_debug(
                    f'_notify_save_complete failed: {format_error_summary(e)}'
                )

    def _log_config_action(self, action: str):
        """Log a config save (action = 'created' or 'updated')."""
        path = self.main_gui.config_manager.conf_path(self.conf_name) or self.conf_name
        key = 'config_created' if action == 'created' else 'config_updated'
        self.log_info(_MSG['tab'][key].format(name=self.conf_name, path=path))

    # Tabs that run a backend: call _snapshot_outputs() before the run
    # and _log_file_changes(snap) after. New paths log as "created"
    # (success), changed paths as "updated" (info), relative to
    # experiment_dir.

    def _snapshot_outputs(self) -> dict:
        """Snapshot the experiment dir's output files (TIFF/npy/vts/...)."""
        if not self.main_gui or not self.main_gui.experiment_dir:
            return {}
        return file_snapshot(self.main_gui.experiment_dir)

    def _log_file_changes(self, before: dict):
        """Log files created or updated since ``before`` was taken."""
        after = self._snapshot_outputs()
        created, updated = file_diff(before, after)
        if not created and not updated:
            self.log_info(_MSG['files']['no_changes'])
            return
        exp_dir = self.main_gui.experiment_dir
        for p in created:
            rel = os.path.relpath(p, exp_dir)
            self.log_success(_MSG['files']['created'].format(path=rel))
        for p in updated:
            rel = os.path.relpath(p, exp_dir)
            self.log_info(_MSG['files']['updated'].format(path=rel))

    @staticmethod
    def _fmt_value(value) -> str:
        """Format a config value for display (strips whitespace from collection literals)."""
        return str(value).replace(' ', '')

    @abstractmethod
    def run_tab(self):
        """Execute the tab's backend processing function."""
        pass

    def _load_config_dialog(self):
        """Load this tab's config from the current experiment's conf directory."""
        self.clear_output()
        if not self.main_gui or not self.main_gui.experiment_exists():
            self.log_error(_MSG['tab']['experiment_not_set'])
            return
        conf_map = self.main_gui.config_manager.load_config(self.conf_name)
        if conf_map is None:
            self.log_warning(_MSG['tab']['config_not_found'].format(
                conf_name=self.conf_name,
                conf_dir=self.main_gui.config_manager.conf_dir,
            ))
            return
        self.load_tab(conf_map)
        self.log_success(_MSG['tab']['config_loaded'].format(conf_name=self.conf_name))
        self._notify_save()  # widgets now match disk; refresh the status strip

    def _build_action_row(self, run_label: Optional[str] = None,
                          run_width: str = '160px') -> widgets.HBox:
        """Return the standard [Load][Save] (+ optional [SplitRun]) HBox.

        Pass ``run_label=None`` for tabs with no backend step to omit
        SplitRun. Sets ``self.load_btn``, ``self.save_button``, and
        ``self.split_run``.
        """
        self.load_btn = button('Load Config', style='warning', width='120px', role='load')
        self.load_btn.on_click(lambda b: self._load_config_dialog())

        self.save_button = SaveButton(on_save=self._on_save_clicked, width='90px')

        children = [self.load_btn, self.save_button.widget]

        if run_label is not None:
            self.split_run = SplitRunButton(
                description=run_label,
                on_save_and_run=lambda: self._dispatch_run(skip_save=False),
                on_run_only=lambda: self._dispatch_run(skip_save=True),
                width=run_width,
            )
            children.append(self.split_run.widget)
        else:
            self.split_run = None
        return widgets.HBox(children)

    def _on_save_clicked(self):
        """Standalone Save button handler (no run)."""
        if not self.main_gui or not self.main_gui.experiment_exists():
            self.log_error(_MSG['tab']['experiment_not_set'])
            return
        err = self.save_conf()
        if err:
            self.log_error(err)

    def _dispatch_run(self, *, skip_save: bool):
        """SplitRun entry point; calls ``run_tab(skip_save=...)``."""
        try:
            self.run_tab(skip_save=skip_save)
        except TypeError:
            self.run_tab()

    def refresh_action_state(self, state: str):
        """Show ``'saved'`` / ``'modified'`` / ``'absent'`` on the action buttons."""
        if self.save_button is not None:
            self.save_button.set_state(state)
        if self.split_run is not None:
            self.split_run.set_state(state)

    def _install_state_observers(self):
        """Watch every form input so the status strip refreshes on each edit."""
        if self.main_gui is None or self._widget is None:
            return
        for w in _walk_widgets(self._widget):
            if isinstance(w, _FORM_WIDGET_TYPES):
                try:
                    w.observe(self._on_field_change, 'value')
                except (TypeError, ValueError, RuntimeError) as e:
                    # Some custom widgets reject observe-by-name; only
                    # the modified-state badge for that field is lost.
                    self.log_debug(
                        f'_install_state_observers skipped a widget: '
                        f'{type(e).__name__}: {e}'
                    )

    def _on_field_change(self, _change):
        if self.main_gui is None:
            return
        notify = getattr(self.main_gui, '_notify_field_change', None)
        if notify is not None:
            try:
                notify(self.conf_name)
            except Exception as e:
                self.log_debug(
                    f'_notify_field_change failed: {format_error_summary(e)}'
                )

    def log(self, message: str):
        """Log a message at info level (or default sink if no log_panel)."""
        if self.log_panel is not None:
            self.log_panel.info(message)

    def log_info(self, message: str):
        if self.log_panel is not None:
            self.log_panel.info(message)
        else:
            self.log(message)

    def log_success(self, message: str):
        if self.log_panel is not None:
            self.log_panel.success(message)
        else:
            self.log(message)

    def log_warning(self, message: str):
        if self.log_panel is not None:
            self.log_panel.warning(message)
        else:
            self.log(message)

    def log_error(self, message: str):
        if self.log_panel is not None:
            self.log_panel.error(message)
        else:
            self.log(message)

    def log_debug(self, message: str):
        if self.log_panel is not None:
            self.log_panel.debug(message)

    def _parse_field(self, name: str, value):
        """Parse a text-field value, logging field-named errors to the tab.

        Wraps ``safe_parse``: malformed entries (unclosed bracket, wrong
        type) log as ``[ERROR] <name>: invalid ...`` plus ``[DEBUG]``
        traceback rather than raising. Returns the raw string on failure
        so the downstream verifier can still flag it by field name.
        """
        return safe_parse(
            name, value,
            log_error=self.log_error,
            log_debug=self.log_debug,
        )

    def clear_output(self):
        """Clear the log panel. Subclasses with a different sink override this."""
        if self.log_panel is not None:
            self.log_panel.clear()

    def _validate_experiment(self) -> Optional[str]:
        """Return an error message if the experiment is not ready, else None."""
        if not self.main_gui:
            return _MSG['tab']['not_initialized']
        if not self.main_gui.experiment_exists():
            return _MSG['tab']['experiment_missing']
        if not self.main_gui.experiment_unchanged():
            return _MSG['tab']['experiment_changed']
        return None
