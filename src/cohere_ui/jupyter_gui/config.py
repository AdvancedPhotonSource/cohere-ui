"""Configuration management wrapping cohere_core.utilities and cohere_ui.api.common."""

import os
from typing import Optional

import cohere_core.utilities as ut


def _get_common():
    """Lazy import of cohere_ui.api.common to allow path setup first."""
    import cohere_ui.api.common as com
    return com


class ConfigManager:
    """Manages experiment configuration loading and saving.

    Mutating methods report what they did so callers can log it:
      - ``save_config`` returns ``(error, action)`` where action is
        ``'created'`` or ``'updated'`` on success.
      - ``ensure_experiment_dir`` / ``ensure_conf_dir`` return ``True``
        when they had to create the directory.
      - ``load_configs`` returns ``(loaded_maps, missing_names)``.
    """

    def __init__(self, experiment_dir: Optional[str] = None):
        self.experiment_dir = experiment_dir
        self._config_maps = {}

    @property
    def conf_dir(self) -> Optional[str]:
        if self.experiment_dir:
            return ut.join(self.experiment_dir, 'conf')
        return None

    def set_experiment_dir(self, experiment_dir: str):
        """Set or change the experiment directory."""
        self.experiment_dir = experiment_dir
        self._config_maps = {}

    def load_configs(self, conf_list: list, no_verify: bool = False) -> tuple[dict, list]:
        """Load multiple configuration files at once.

        Returns ``(loaded_maps, missing_names)``: the dict of names that
        existed and parsed, and the list of names that were absent.
        """
        if not self.experiment_dir:
            raise ValueError("Experiment directory not set")

        com = _get_common()
        maps, _ = com.get_config_maps(
            self.experiment_dir,
            conf_list,
            no_verify=no_verify
        )
        self._config_maps.update(maps)
        missing = [n for n in conf_list if n not in maps]
        return maps, missing

    def load_config(self, conf_name: str) -> Optional[dict]:
        """Load a single configuration file. Returns None if it doesn't exist."""
        if not self.conf_dir:
            return None

        conf_path = ut.join(self.conf_dir, conf_name)
        if not os.path.isfile(conf_path):
            return None

        conf_map = ut.read_config(conf_path)
        if conf_map:
            self._config_maps[conf_name] = conf_map
        return conf_map

    def save_config(self, conf_name: str, conf_map: dict,
                    no_verify: bool = False) -> tuple[str, str]:
        """Save configuration to file. Returns ``(error, action)``.

        ``action`` is ``'created'`` if the file was new, ``'updated'``
        if it already existed, or ``''`` when an error prevented the save.
        """
        if not self.conf_dir:
            return ("Experiment directory not set", "")

        err = self.verify(conf_name, conf_map)
        if err and not no_verify:
            return (err, "")

        conf_path = ut.join(self.conf_dir, conf_name)
        action = 'updated' if os.path.isfile(conf_path) else 'created'
        ut.write_config(conf_map, conf_path)
        self._config_maps[conf_name] = conf_map
        return ("", action)

    def conf_path(self, conf_name: str) -> Optional[str]:
        """Return the absolute path to ``conf_name`` (whether or not it exists)."""
        if not self.conf_dir:
            return None
        return ut.join(self.conf_dir, conf_name)

    def verify(self, conf_name: str, conf_map: dict) -> str:
        """Verify configuration dictionary. Returns error string (empty if valid).

        ``cohere_core.utilities.verify`` returns a tuple when no schema
        is registered (e.g. ``config_mp``); treat that as "no verifier"
        and pass rather than blocking the save.
        """
        result = ut.verify(conf_name, conf_map)
        if isinstance(result, tuple):
            return ''
        return result

    def get_cached(self, conf_name: str) -> Optional[dict]:
        """Get a previously loaded config from cache."""
        return self._config_maps.get(conf_name)

    def disk_mtime(self, conf_name: str) -> Optional[float]:
        """Return the on-disk mtime of ``conf_name``, or None if absent
        or the experiment dir isn't set. Used to detect edits made
        outside the GUI between load and the next save.
        """
        path = self.conf_path(conf_name)
        if path is None or not os.path.isfile(path):
            return None
        try:
            return os.path.getmtime(path)
        except OSError:
            return None

    def is_stale_vs_widgets(self, conf_name: str, widget_map: dict) -> bool:
        """Return True when on-disk ``conf_name`` differs from ``widget_map``.

        The disk copy is read fresh (NOT from cache) so external edits
        are visible. Returns False when the file is absent or parses
        equal to ``widget_map``. Unparseable files return True so the
        caller can surface the problem.
        """
        path = self.conf_path(conf_name)
        if path is None or not os.path.isfile(path):
            return False
        try:
            on_disk = ut.read_config(path) or {}
        except Exception:
            return True
        return dict(on_disk) != dict(widget_map or {})

    @property
    def all_configs(self) -> dict:
        """Return all cached configuration maps."""
        return self._config_maps.copy()

    def ensure_conf_dir(self) -> bool:
        """Create the conf directory if it doesn't exist. Returns True when created."""
        if self.conf_dir and not os.path.exists(self.conf_dir):
            os.makedirs(self.conf_dir)
            return True
        return False

    def ensure_experiment_dir(self) -> tuple[bool, bool]:
        """Create the experiment dir (and conf dir) if needed.

        Returns ``(experiment_dir_created, conf_dir_created)``.
        """
        exp_created = False
        if self.experiment_dir and not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
            exp_created = True
        conf_created = self.ensure_conf_dir()
        return exp_created, conf_created
