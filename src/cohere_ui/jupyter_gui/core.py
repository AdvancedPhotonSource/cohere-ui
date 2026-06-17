"""Main CoherenceGUI class for the Jupyter notebook GUI."""

import os
import threading
import time
import traceback

import ipywidgets as widgets
from IPython.display import display

import cohere_core.utilities as ut

from cohere_ui.jupyter_gui._env_check import check_libomp_consistency
from cohere_ui.jupyter_gui.config import ConfigManager
from cohere_ui.jupyter_gui.header.experiment_picker import (
    ExperimentPicker, WizardResult,
)
from cohere_ui.jupyter_gui.header.status_strip import StatusStrip
from cohere_ui.jupyter_gui.results import ResultsContainer
from cohere_ui.jupyter_gui.styles import apply_button_role, inject_custom_css
from cohere_ui.jupyter_gui.text import load_text
from cohere_ui.jupyter_gui.utils.error_format import format_error_summary
from cohere_ui.jupyter_gui.widgets import LogPanel

_MSG = load_text('messages')
_UI = load_text('ui_strings')
_URLS = load_text('urls')


def _mode_from_conf(conf_map: dict) -> str:
    """Derive the wizard mode from a loaded ``conf/config``.

    Priority matches ``cohere_ui/beamline_preprocess.py``:
    multipeak > separate_scan_ranges > separate_scans > single.
    """
    if conf_map.get('multipeak'):
        return 'multipeak'
    if conf_map.get('separate_scan_ranges'):
        return 'separate_scan_ranges'
    if conf_map.get('separate_scans'):
        return 'separate_scans'
    return 'single'


def _find_default_parent_dir() -> str:
    """Default parent dir for the picker.

    Walk up from cwd looking for ``cohere_examples/example_workspace`` so
    notebooks shipped in the repo land in a useful folder. Fall back to cwd
    if not found.
    """
    marker = os.path.join('cohere_examples', 'example_workspace')
    here = os.getcwd()
    cur = here
    while True:
        cand = os.path.join(cur, marker)
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            return here
        cur = parent


class CoherenceGUI:
    """Tab-based interface for Bragg CDI reconstruction.

    The tab list is built from :func:`cohere_ui.jupyter_gui.tabs.get_registered_tabs`
    once per GUI instance and frozen on ``self._tab_specs``. Mutating the
    registry (via ``register_tab`` / ``unregister_tab``) AFTER a GUI is
    constructed has no effect on that GUI. Re-instantiate to pick up
    new tabs.
    """

    def __init__(self):
        self.config_manager = ConfigManager()
        self.results = ResultsContainer(self.config_manager)

        self.no_verify = False
        self.debug = False

        self._folder_name: str | None = None

        # Freeze the registry snapshot for this GUI instance so a later
        # register_tab call cannot half-mount a tab into an existing GUI.
        from cohere_ui.jupyter_gui.tabs import get_registered_tabs
        self._tab_specs = list(get_registered_tabs())
        self._tab_specs_by_key = {s.key: s for s in self._tab_specs}

        self._tabs: dict = {}
        self._widget = None
        self._linked_debug_panels: set = set()

        # Coalesce rapid widget edits into a single status-strip refresh.
        self._refresh_timer: threading.Timer | None = None
        self._refresh_lock = threading.Lock()

        self._build_ui()
        self.init_tabs()
        # Route ResultsContainer load failures to the in-tab log panel
        # (notebook users rarely watch kernel stderr).
        self.results._log_error = self.log_panel.error
        self.results._log_debug = self.log_panel.debug
        self._announce_env_check()

    def _announce_env_check(self):
        """Surface the libomp / torch / xrayutilities consistency check.

        Silent on a healthy install. On macOS with torch's bundled libomp
        in place, log the error and remediation steps so the user sees the
        fix before the Postprocess pipeline crashes the kernel.
        """
        result = check_libomp_consistency()
        if result.ok:
            return
        if result.severity == 'error':
            self.log_panel.error(result.message)
        else:
            self.log_panel.warning(result.message)
        if result.remediation:
            for line in result.remediation.splitlines():
                self.log_panel.info(line)

    def _build_ui(self):
        """Build the main GUI widget tree."""
        self.log_panel = LogPanel()
        self.status_strip = StatusStrip()
        self.tab_widget = widgets.Tab()

        self.experiment_picker = ExperimentPicker(
            start_path=_find_default_parent_dir(),
            on_load=self._handle_picker_load,
            on_create=self._create_from_wizard,
            log_panel=self.log_panel,
        )

        # InstrTab reads main_gui.scan.value during parse_spec;
        # alias the wizard's scan field so it stays in sync.
        self.scan = self.experiment_picker.wizard.scan

        self.run_all_btn = widgets.Button(
            description=_UI['action_buttons']['run_all'],
            tooltip=_UI['tooltips']['run_all'],
            layout=widgets.Layout(width='110px', margin='0 0 0 8px'),
        )
        apply_button_role(self.run_all_btn, 'run')
        self.run_all_btn.on_click(self._on_run_all_clicked)
        self._run_all_thread: threading.Thread | None = None

        # Global Safe mode: when checked, each tab's Save and Run saves the
        # form (with validation) before running. When unchecked, it runs
        # against the on-disk config and ignores unsaved form changes.
        self.safe_mode_checkbox = widgets.Checkbox(
            value=True,
            description=_UI['action_buttons']['safe_mode'],
            indent=False,
            tooltip=_UI['tooltips']['safe_mode'],
            layout=widgets.Layout(width='auto', margin='0 8px 0 0'),
        )
        self.safe_mode_checkbox.observe(self._on_safe_mode_change, 'value')

        # Header bar: title on the left, docs link on the right.
        # The link opens in a new tab so the notebook session is not disturbed.
        header_bar = widgets.HBox(
            [
                widgets.HTML(
                    '<h3 style="margin:6px 0 4px 0;">Cohere Reconstruction</h3>',
                    layout=widgets.Layout(flex='1 1 auto'),
                ),
                widgets.HTML(
                    f'<a href="{_URLS["github"]}" target="_blank" rel="noopener" '
                    f'title="{_UI["tooltips"]["github_open"]}" '
                    f'style="text-decoration:none; font-size:13px; '
                    f'padding:4px 10px; border:1px solid var(--jup-border-2); '
                    f'border-radius:4px; background:var(--jup-card-bg-2);">'
                    f'<i class="fa fa-github"></i> GitHub \u2197</a>',
                    layout=widgets.Layout(flex='0 0 auto', margin='6px 6px 0 0'),
                ),
                widgets.HTML(
                    f'<a href="{_URLS["docs"]}" target="_blank" rel="noopener" '
                    f'title="{_UI["tooltips"]["docs_open"]}" '
                    f'style="text-decoration:none; font-size:13px; '
                    f'padding:4px 10px; border:1px solid var(--jup-border-2); '
                    f'border-radius:4px; background:var(--jup-card-bg-2);">'
                    f'\U0001f4d6 Docs \u2197</a>',
                    layout=widgets.Layout(flex='0 0 auto', margin='6px 4px 0 0'),
                ),
            ],
            layout=widgets.Layout(
                width='100%', justify_content='space-between',
                align_items='center',
            ),
        )

        # Status strip + Run All button on the same baseline.
        # The strip expands; Run All sits flush right and centred.
        status_row = widgets.HBox(
            [
                widgets.Box(
                    [self.status_strip.widget],
                    layout=widgets.Layout(flex='1 1 auto'),
                ),
                self.safe_mode_checkbox,
                self.run_all_btn,
            ],
            layout=widgets.Layout(
                align_items='center', width='100%',
            ),
        )

        self._widget = widgets.VBox([
            header_bar,
            self.experiment_picker.widget,
            self.log_panel.widget,
            widgets.HTML('<hr style="margin:8px 0;">'),
            status_row,
            self.tab_widget,
        ])

    @property
    def safe_mode(self) -> bool:
        """When True, tab Save and Run buttons save (and validate) before running."""
        cb = getattr(self, 'safe_mode_checkbox', None)
        return True if cb is None else bool(cb.value)

    def _on_safe_mode_change(self, change):
        self._apply_safe_mode(bool(change['new']))

    def _apply_safe_mode(self, safe: bool | None = None):
        """Push the Safe mode flag to every tab's run button.

        Safe -> plain Save and Run; unsafe -> the split caret + menu. Run
        live from the header checkbox and after tabs (re)mount.
        """
        if safe is None:
            safe = self.safe_mode
        for tab in self._tabs.values():
            btn = getattr(tab, 'split_run', None)
            if btn is not None:
                btn.set_safe(safe)

    def init_tabs(self):
        """Build the always-visible tabs from the registry.

        Optional tabs (e.g. ``MpTab``) are mounted later by
        :meth:`_refresh_optional_tabs` when their ``visible_when``
        predicate first turns True.
        """
        self._tabs = {}
        for spec in self._tab_specs:
            if spec.optional:
                continue
            self._tabs[spec.key] = spec.factory()
        self._refresh_tab_widget()
        self._refresh_status()
        self._apply_safe_mode()

    def _refresh_optional_tabs(self):
        """Mount or unmount every optional tab based on its predicate.

        Replaces the old single-tab ``_set_multipeak_visibility``: any
        future optional tab plugs in through ``TabSpec(optional=True,
        visible_when=...)`` and gets the same lifecycle.
        """
        changed = False
        for spec in self._tab_specs:
            if not spec.optional:
                continue
            should_show = bool(spec.visible_when and spec.visible_when(self))
            has_tab = spec.key in self._tabs
            if should_show and not has_tab:
                self._tabs[spec.key] = spec.factory()
                changed = True
            elif not should_show and has_tab:
                self._tabs.pop(spec.key, None)
                changed = True
        if changed:
            self._refresh_tab_widget()
            self._apply_safe_mode()

    def _set_multipeak_visibility(self, visible: bool):
        """Back-compat shim: forces an optional-tab refresh.

        Older entry points call this with an explicit boolean. The
        boolean is ignored because the registry's ``visible_when``
        predicate is the single source of truth, but the side effect
        (mounting or unmounting the multi-peak tab) still happens.
        """
        del visible  # predicate decides
        self._refresh_optional_tabs()

    def _refresh_tab_widget(self):
        tab_children = []
        tab_titles = []
        for spec in self._tab_specs:
            tab = self._tabs.get(spec.key)
            if tab is None:
                continue
            tab.init(self)
            tab_children.append(tab.widget)
            tab_titles.append(tab.name)
        self.tab_widget.children = tab_children
        for i, title in enumerate(tab_titles):
            self.tab_widget.set_title(i, title)
        self._sync_debug_panels()

    def _sync_debug_panels(self):
        """Link every log panel's show-debug checkbox to the main panel's,
        so toggling any one reveals/hides debug lines everywhere."""
        master = self.log_panel.show_debug_checkbox
        candidates = []
        for tab in self._tabs.values():
            log_panel = getattr(tab, 'log_panel', None)
            if log_panel is not None:
                cb = getattr(log_panel, 'show_debug_checkbox', None)
                if cb is not None:
                    candidates.append(cb)
            monitor = getattr(tab, 'monitor', None)
            if monitor is not None:
                cb = getattr(monitor, 'show_debug_checkbox', None)
                if cb is not None:
                    candidates.append(cb)
        for cb in candidates:
            key = id(cb)
            if key in self._linked_debug_panels:
                continue
            widgets.link((master, 'value'), (cb, 'value'))
            self._linked_debug_panels.add(key)

    def add_instr_tab(self, beamline: str):
        """Switch the InstrTab content to the given beamline."""
        instr = self._tabs.get('instr')
        if instr is None:
            return
        instr.set_beamline(beamline)
        instr._widget = None
        new_widget = instr.widget
        children = list(self.tab_widget.children)
        # Match the rendered tab strip's order, which equals the spec
        # order minus optional tabs whose predicate is currently False.
        rendered_keys = [s.key for s in self._tab_specs if s.key in self._tabs]
        try:
            idx = rendered_keys.index('instr')
            if 0 <= idx < len(children):
                children[idx] = new_widget
                self.tab_widget.children = tuple(children)
        except ValueError:
            pass
        self._sync_debug_panels()
        self._refresh_status(compare_modified=True)

    @property
    def widget(self) -> widgets.Widget:
        return self._widget

    @property
    def experiment_dir(self) -> str | None:
        return self.config_manager.experiment_dir

    def display(self):
        """Display the GUI in the notebook."""
        inject_custom_css()
        display(self._widget)

    def log(self, message: str):
        """Log an info-level message to the main output panel."""
        self.log_panel.info(message)

    def clear_log(self):
        """Clear the main output panel."""
        self.log_panel.clear()

    def experiment_exists(self) -> bool:
        exp_dir = self.config_manager.experiment_dir
        return bool(exp_dir and os.path.isdir(exp_dir))

    def experiment_unchanged(self) -> bool:
        """True if the current folder basename matches the cached folder name.

        Always true within one load/create cycle: the folder name cannot
        be edited after an experiment is set.
        """
        exp_dir = self.config_manager.experiment_dir
        if not exp_dir or not self._folder_name:
            return False
        return os.path.basename(exp_dir) == self._folder_name

    def reset_window(self):
        """Reset GUI state to initial. The picker's parent dir is preserved."""
        self._folder_name = None
        self.config_manager.set_experiment_dir(None)
        self.experiment_picker.reset()
        for tab in self._tabs.values():
            tab.clear_conf()
        self._refresh_status(compare_modified=False)

    def _handle_picker_load(self, full_path: str):
        """Forward the picker's Load button to load_experiment."""
        self.load_experiment(full_path)

    def load_experiment(self, path: str | None = None):
        """Load an existing experiment from directory."""
        self.clear_log()
        if not path:
            self.log_panel.error(
                _MSG['main'].get('parent_dir_missing',
                                 'Please select an experiment in the picker.')
            )
            return
        load_dir = path if os.path.isabs(path) else os.path.abspath(path)

        config_path = ut.join(load_dir, 'conf', 'config')
        if not os.path.isfile(config_path):
            self.log_panel.error(_MSG['main']['config_missing'].format(path=load_dir))
            return

        try:
            self.config_manager.set_experiment_dir(load_dir)
            conf_list = ['config_prep', 'config_data', 'config_rec', 'config_disp',
                         'config_instr', 'config_mp']
            conf_dicts, missing = self.config_manager.load_configs(conf_list, no_verify=True)
        except Exception as e:
            self.log_panel.error(_MSG['main']['config_load_error'].format(
                error=format_error_summary(e)))
            self.log_panel.debug(traceback.format_exc())
            return

        main_conf = conf_dicts.get('config', {})
        self._load_main(main_conf)

        beamline = main_conf.get('beamline')
        if beamline:
            self.add_instr_tab(beamline)

        # Set multipeak visibility before loading per-tab configs so
        # MpTab is present when its config_mp is loaded into it below.
        self._set_multipeak_visibility(self._is_multipeak_experiment())

        for tab in self._tabs.values():
            if tab.conf_name in conf_dicts:
                tab.load_tab(conf_dicts[tab.conf_name])

        self.results.set_config_manager(self.config_manager)
        self.log_panel.success(_MSG['main']['experiment_loaded'].format(path=load_dir))
        if conf_dicts:
            self.log_panel.info(_MSG['main']['configs_loaded'].format(
                count=len(conf_dicts),
                names=', '.join(sorted(conf_dicts.keys())),
            ))
        if missing:
            self.log_panel.info(_MSG['main']['configs_missing'].format(
                names=', '.join(missing),
            ))
        self._refresh_status(compare_modified=True)

    def _load_main(self, conf_map: dict):
        """Populate the picker with the loaded experiment's metadata.

        The folder basename is the authoritative in-memory ``experiment_id``.
        If the on-disk ``experiment_id`` differs, the basename wins and
        ``config`` is left untouched until the user saves.
        """
        exp_dir = self.config_manager.experiment_dir
        self._folder_name = os.path.basename(exp_dir) if exp_dir else None
        parent_dir = os.path.dirname(exp_dir) if exp_dir else ''
        self.experiment_picker.set_loaded(
            parent_dir=parent_dir,
            folder_name=self._folder_name or '',
            serial=str(conf_map.get('scan', '')).replace(' ', ''),
            beamline=conf_map.get('beamline', ''),
            mode=_mode_from_conf(conf_map),
        )

    def _create_from_wizard(self, result: WizardResult):
        """Handler for the ExperimentWizard Create button."""
        self.set_experiment(
            parent_dir=result.parent_dir,
            folder_name=result.folder_name,
            scan=result.scan,
            beamline=result.beamline,
            mode=result.mode,
        )

    def set_experiment(
        self,
        *,
        parent_dir: str,
        folder_name: str,
        scan: str = '',
        beamline: str = '',
        mode: str = 'single',
    ):
        """Create or set the experiment directory and write its ``config``.

        Does NOT auto-save tab configs. Use each tab's Save button or
        SplitRun's "Save & Run" to write them.
        """
        self.clear_log()

        if not parent_dir:
            self.log_panel.error(_MSG['main'].get(
                'parent_dir_missing', 'Pick a parent directory first.'))
            return
        if not os.path.isdir(parent_dir):
            self.log_panel.error(_MSG['main']['working_dir_missing'].format(path=parent_dir))
            return
        if not folder_name.strip():
            self.log_panel.error(_MSG['main']['no_experiment_id'])
            return

        exp_dir = ut.join(parent_dir, folder_name)
        if os.path.isfile(ut.join(exp_dir, 'conf', 'config')):
            self.log_panel.error(_MSG['main'].get(
                'collision', f'Experiment already exists at {exp_dir}.'
                             ' Pick a different folder name or use Load.'))
            return

        self.config_manager.set_experiment_dir(exp_dir)
        exp_created, conf_created = self.config_manager.ensure_experiment_dir()
        if exp_created:
            self.log_panel.success(
                _MSG['main']['experiment_dir_created'].format(path=exp_dir))
        if conf_created and not exp_created:
            self.log_panel.info(_MSG['main']['conf_dir_created'].format(
                path=self.config_manager.conf_dir))

        self._folder_name = folder_name
        self._save_main(scan=scan, beamline=beamline, mode=mode)

        if beamline:
            self.add_instr_tab(beamline)

        # Show the Multi-peak tab if the wizard set the flag.
        # config_mp still must be filled in before running Prepare.
        self._set_multipeak_visibility(self._is_multipeak_experiment())

        self.results.set_config_manager(self.config_manager)
        self.log_panel.success(_MSG['main']['experiment_set'].format(path=exp_dir))
        self._refresh_status(compare_modified=False)

    def _save_main(self, *, scan: str, beamline: str, mode: str):
        """Write the top-level ``conf/config`` from the given arguments."""
        conf_map = {
            'working_dir': os.path.dirname(self.config_manager.experiment_dir or ''),
            'experiment_id': self._folder_name,
        }
        if scan.strip():
            conf_map['scan'] = scan.strip()
        if beamline.strip():
            conf_map['beamline'] = beamline.strip()
        if mode == 'multipeak':
            conf_map['multipeak'] = True
        elif mode == 'separate_scans':
            conf_map['separate_scans'] = True
        elif mode == 'separate_scan_ranges':
            conf_map['separate_scan_ranges'] = True

        _, action = self.config_manager.save_config('config', conf_map, no_verify=self.no_verify)
        if action:
            path = self.config_manager.conf_path('config')
            key = 'config_created' if action == 'created' else 'config_updated'
            self.log_panel.info(_MSG['tab'][key].format(name='config', path=path))

    def _notify_save_complete(self, conf_name: str):
        self._refresh_status(compare_modified=True)

    def _notify_field_change(self, conf_name: str):
        # Many widgets fire during load_tab; coalesce into one refresh.
        with self._refresh_lock:
            if self._refresh_timer is not None:
                self._refresh_timer.cancel()
            self._refresh_timer = threading.Timer(
                0.15, lambda: self._refresh_status(compare_modified=True),
            )
            self._refresh_timer.daemon = True
            self._refresh_timer.start()

    def _refresh_status(self, *, compare_modified: bool = False):
        """Recompute per-tab state, update tab-title status prefixes,
        and refresh the header strip. Save/Run buttons are disabled when
        no experiment is loaded.
        """
        from cohere_ui.jupyter_gui.header.tab_state import compute as _compute_state

        has_exp = self.experiment_exists()
        try:
            self.status_strip.set_state(
                experiment_dir=self.config_manager.experiment_dir,
                is_multipeak=self._is_multipeak_experiment(),
            )
        except Exception as e:
            # Log the outer error; the inner try guards against a broken
            # log panel. If debug logging also fails, there is nowhere
            # left to route the error - fall through.
            try:
                self.log_panel.debug(
                    f'status_strip set_state failed: {format_error_summary(e)}'
                )
            except Exception:
                pass

        # Iterate the rendered tab strip in the same order as the spec
        # list so the tab-title status prefix lines up with the tab idx.
        rendered = [(s.key, self._tabs[s.key])
                    for s in self._tab_specs if s.key in self._tabs]
        for idx, (key, tab) in enumerate(rendered):
            try:
                if has_exp:
                    state = _compute_state(
                        tab, self.config_manager, deep=compare_modified,
                    )
                else:
                    state = 'absent'

                prefix = {'saved': '\u25cf', 'modified': '\u25d0', 'absent': '\u25cb'}.get(state, '\u25cb')
                if idx < len(self.tab_widget.children):
                    self.tab_widget.set_title(idx, f'{prefix} {tab.name}')

                if not has_exp:
                    if tab.save_button is not None:
                        tab.save_button.btn.disabled = True
                    if tab.split_run is not None:
                        tab.split_run.set_enabled(False)
                    if tab.load_btn is not None:
                        tab.load_btn.disabled = True
                else:
                    if tab.load_btn is not None:
                        tab.load_btn.disabled = False
                    if tab.split_run is not None:
                        tab.split_run.set_enabled(True)
                    tab.refresh_action_state(state)
            except Exception as e:
                # A broken tab must not break the whole status refresh;
                # send the trace to the debug log so it is not lost.
                try:
                    self.log_panel.debug(
                        f'_refresh_status[{key}] failed: '
                        f'{format_error_summary(e)}'
                    )
                except Exception:
                    pass

    def _is_multipeak_experiment(self) -> bool:
        """Detect multipeak via flag, config_mp presence, or mp_* subdirs."""
        exp_dir = self.config_manager.experiment_dir
        if not exp_dir or not os.path.isdir(exp_dir):
            return False
        main_conf = self.config_manager.get_cached('config') or {}
        if main_conf.get('multipeak'):
            return True
        if self.config_manager.conf_dir and os.path.isfile(
                ut.join(self.config_manager.conf_dir, 'config_mp')):
            return True
        try:
            for entry in os.listdir(exp_dir):
                if entry.startswith('mp_'):
                    full = os.path.join(exp_dir, entry)
                    if os.path.isdir(full):
                        return True
        except OSError:
            pass
        return False

    def run_all(self):
        """Run all tabs in sequence. Skips InstrTab (no backend).

        Tabs that own a subprocess monitor (RecTab) start an async
        worker; the chain waits for the monitor to go idle before moving
        on so Disp does not render against partial Rec output. Aborts on
        the first tab that raises.
        """
        if not self.experiment_exists():
            self.log_panel.error(_MSG['tab']['experiment_missing'])
            return
        if not self.experiment_unchanged():
            self.log_panel.error(_MSG['tab']['experiment_changed'])
            return
        for spec in self._tab_specs:
            if spec.skip_in_run_all:
                continue
            tab = self._tabs.get(spec.key)
            if tab is None:
                continue
            key = spec.key
            try:
                tab.run_tab()
            except Exception as e:
                self.log_panel.error(
                    f'run_all aborted in {key}: {format_error_summary(e)}'
                )
                self.log_panel.debug(traceback.format_exc())
                return
            # If this tab launched a subprocess (RecTab), wait for it
            # to finish before advancing to Disp.
            monitor = getattr(tab, 'monitor', None)
            if monitor is None:
                continue
            self._wait_for_monitor_idle(monitor)
            # Stop if the subprocess exited non-zero - do not run Disp
            # against a half-written results_phasing/.
            exit_code = getattr(monitor, '_last_exit_code', None)
            if exit_code not in (None, 0):
                self.log_panel.error(
                    f'run_all aborted: {key} subprocess exited with '
                    f'code {exit_code}'
                )
                return

    @staticmethod
    def _wait_for_monitor_idle(monitor, poll_interval: float = 0.5,
                               max_wait: float = 24 * 3600.0):
        """Block until ``monitor.is_running`` reports False.

        Bounded by ``max_wait`` (default 24 h) as a safety net; the
        Stop button is the normal way out of a long run.
        """
        deadline = time.monotonic() + max_wait
        while monitor.is_running and time.monotonic() < deadline:
            time.sleep(poll_interval)

    def _on_run_all_clicked(self, _b):
        """Run All button handler. Runs run_all in a background thread
        so the UI stays responsive and Rec's Stop button keeps working."""
        if self._run_all_thread is not None and self._run_all_thread.is_alive():
            self.log_panel.warning('Run All is already in progress.')
            return
        self.run_all_btn.disabled = True

        def _runner():
            try:
                self.run_all()
            finally:
                self.run_all_btn.disabled = False
                self._run_all_thread = None

        self._run_all_thread = threading.Thread(
            target=_runner, daemon=True, name='CoherenceGUI-run-all',
        )
        self._run_all_thread.start()
