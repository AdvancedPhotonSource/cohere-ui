"""Base class for all tab implementations."""

import os
from abc import ABC, abstractmethod
from typing import Optional

import ipywidgets as widgets

from ..file_events import diff as file_diff, snapshot as file_snapshot
from ..text import load_text

_MSG = load_text('messages')


class BaseTab(ABC):
    """Abstract base class defining the standard tab interface.

    Each tab must implement:
      - name: Display name for the tab
      - conf_name: Configuration file name (e.g., 'config_rec')
      - _build_ui(): Create the widget tree
      - load_tab(conf_map): Populate widgets from config dict
      - get_config(): Read widgets into config dict
      - clear_conf(): Reset all widgets to defaults
      - run_tab(): Execute the tab's backend function

    Tabs that want a styled, auto-scrolling status log create a
    ``LogPanel`` and assign it to ``self.log_panel`` in ``_build_ui``;
    the convenience methods ``log_info`` / ``log_success`` /
    ``log_warning`` / ``log_error`` route to it. Tabs that ship their
    own log surface (RecTab, via its monitor) override ``log()`` and
    ``clear_output()`` instead.
    """

    name: str = "Tab"
    conf_name: str = "config"

    def __init__(self):
        self.main_gui = None
        self.log_panel = None
        self._widget = None

    def init(self, main_gui):
        """Initialize the tab with reference to main GUI."""
        self.main_gui = main_gui
        self._widget = self._build_ui()

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
        conf_map = self.get_config()
        err, action = self.main_gui.config_manager.save_config(
            self.conf_name,
            conf_map,
            no_verify=self.main_gui.no_verify
        )
        if action:
            self._log_config_action(action)
        return err

    def save_and_verify(self) -> str:
        """Get config, verify, save. Returns error string (empty when successful).

        Replaces the duplicated verify-then-save block in every tab's run_tab.
        Logs the create/update action through the tab's logging.
        """
        conf_map = self.get_config()
        err = self.main_gui.config_manager.verify(self.conf_name, conf_map)
        if err and not self.main_gui.no_verify:
            self.log_error(_MSG['tab']['config_error'].format(error=err))
            return err
        _, action = self.main_gui.config_manager.save_config(
            self.conf_name, conf_map, self.main_gui.no_verify)
        if action:
            self._log_config_action(action)
        return ""

    def _log_config_action(self, action: str):
        """Log a config save (action = 'created' or 'updated')."""
        path = self.main_gui.config_manager.conf_path(self.conf_name) or self.conf_name
        key = 'config_created' if action == 'created' else 'config_updated'
        self.log_info(_MSG['tab'][key].format(name=self.conf_name, path=path))

    # --- File-event reporting ---
    #
    # Tabs that run a backend snapshot the experiment directory's output
    # files before the run and call _log_file_changes(snap) afterwards;
    # the difference is logged as created (success level) or updated
    # (info level), with paths relative to the experiment directory for
    # readability.

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

    # --- Logging API ---
    #
    # Subclasses should populate self.log_panel in _build_ui to receive
    # styled, auto-scrolling output. Tabs with a different log surface
    # (RecTab) override log() and clear_output() to route elsewhere; in
    # that case the level helpers fall back to plain log().

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
