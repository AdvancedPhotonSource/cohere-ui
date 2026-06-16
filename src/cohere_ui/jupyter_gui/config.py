"""Configuration management wrapping cohere_core.utilities and cohere_ui.api.common.

Schema versioning
-----------------
Every config file written by :class:`ConfigManager` is stamped with a
``_schema_version`` key. On read, the manager detects the on-disk
version, runs any registered migrations to bring it up to
:data:`SCHEMA_CURRENT_VERSIONS` for that file, then strips the key
before handing the dict to the tab. This means:

* Existing pre-versioned files load cleanly: a missing
  ``_schema_version`` is treated as "current" and stamped on next save.
* Future structural changes to a config_* file register a migration
  in :data:`MIGRATIONS` and bump the entry in
  :data:`SCHEMA_CURRENT_VERSIONS`. Old files are upgraded on first read
  and the upgraded form is persisted in place (best-effort, silent on
  write failure so a read-only checkout still works).
* The ``_schema_version`` key never reaches the cohere_core verifier or
  the tab's ``load_tab`` because verifier schemas reject unknown keys.
"""

import glob
import logging
import os
from typing import Callable, Dict, Optional, Tuple

import cohere_core.utilities as ut


_log = logging.getLogger(__name__)


def _get_common():
    """Lazy import of cohere_ui.api.common to allow path setup first."""
    import cohere_ui.api.common as com
    return com


# Internal marker key. Underscore prefix flags it as metadata and keeps
# it sorted to the top by python-config writers that sort alphabetically.
SCHEMA_VERSION_KEY = '_schema_version'


# Current version for each config_* file. Looked up by conf_name; absent
# entries default to 1. Bump this when a structural change to a config
# requires migrating pre-existing files on disk, and register the
# transform in :data:`MIGRATIONS`.
SCHEMA_CURRENT_VERSIONS: Dict[str, int] = {
    'config':       1,
    'config_prep':  1,
    'config_data':  1,
    'config_rec':   1,
    'config_disp':  1,
    'config_instr': 1,
    'config_mp':    1,
}


# MIGRATIONS maps ``(conf_name, from_version)`` to a migration callable.
# Each migration takes a dict already at version ``from_version`` and
# returns the same dict at ``from_version + 1``. The loader walks the
# chain (1 to 2 to 3 and so on) until it reaches the current target.
MIGRATIONS: Dict[Tuple[str, int], Callable[[dict], dict]] = {
    # Template for the first real migration:
    # ('config_rec', 1): _migrate_config_rec_1_to_2,
}


def register_migration(conf_name: str, from_version: int,
                       func: Callable[[dict], dict]) -> None:
    """Register a config-schema migration.

    ``func`` must take a dict at ``from_version`` and return a dict at
    ``from_version + 1``. Bump :data:`SCHEMA_CURRENT_VERSIONS` for
    ``conf_name`` to the new target in the same change, or the
    migration will not fire (the loader stops once on-disk version
    matches the target).
    """
    key = (conf_name, from_version)
    if key in MIGRATIONS:
        raise ValueError(f'migration already registered: {key}')
    MIGRATIONS[key] = func


def _strip_version(conf_map: Optional[dict]) -> Optional[dict]:
    """Return a copy of ``conf_map`` without the schema-version key."""
    if not conf_map or SCHEMA_VERSION_KEY not in conf_map:
        return conf_map
    out = dict(conf_map)
    out.pop(SCHEMA_VERSION_KEY, None)
    return out


def _stamp_version(conf_name: str, conf_map: Optional[dict]) -> dict:
    """Return a copy of ``conf_map`` with the current schema_version inserted."""
    target = SCHEMA_CURRENT_VERSIONS.get(conf_name, 1)
    out = dict(conf_map or {})
    out[SCHEMA_VERSION_KEY] = target
    return out


def _migrate(conf_name: str, conf_map: Optional[dict]) -> Tuple[Optional[dict], bool]:
    """Walk the migration chain for ``conf_name``.

    Returns ``(migrated, did_migrate)``. ``did_migrate`` is True iff at
    least one registered migration step actually ran. When it does, the
    caller is expected to persist the upgraded dict back to disk.
    """
    if not conf_map:
        return conf_map, False
    target = SCHEMA_CURRENT_VERSIONS.get(conf_name, 1)
    current = conf_map.get(SCHEMA_VERSION_KEY, target)
    if current >= target:
        return conf_map, False
    did_migrate = False
    while current < target:
        step = MIGRATIONS.get((conf_name, current))
        if step is None:
            # No registered transition, so leave the dict at the highest
            # version the chain reached so the next read picks up where
            # this one stopped. Logged so a gap in the chain is visible.
            _log.warning(
                'No migration registered for %s v%d to v%d; leaving as v%d',
                conf_name, current, current + 1, current,
            )
            break
        try:
            conf_map = step(conf_map)
        except Exception as e:
            _log.error(
                'Migration %s v%d to v%d raised %s; aborting chain at v%d',
                conf_name, current, current + 1, type(e).__name__, current,
            )
            return conf_map, did_migrate
        current += 1
        did_migrate = True
    return conf_map, did_migrate


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

    def discover_rec_ids(self) -> list:
        """Return sorted '<id>' names found in ``conf/config_rec_<id>``.

        Excludes the canonical ``config_rec`` (the 'main' file) and any
        ``*_backup`` copy that ``convertconfig`` writes next to a config
        (otherwise 'backup' would surface as a fake reconstruction id).
        Returns an empty list when no experiment is loaded.
        """
        if not self.conf_dir:
            return []
        out = []
        for path in glob.glob(os.path.join(self.conf_dir, 'config_rec_*')):
            base = os.path.basename(path)
            # Skip convert's backups: 'config_rec_backup' (of the main
            # file) and 'config_rec_<id>_backup' both end in '_backup'.
            if base.endswith('_backup'):
                continue
            suffix = base[len('config_rec_'):]
            if suffix:
                out.append(suffix)
        return sorted(out)

    def discover_result_ids(self) -> list:
        """Return sorted '<id>' names of ``results_phasing_<id>/`` dirs that
        actually contain reconstruction output (an ``image.npy`` anywhere
        in their tree).

        Excludes the unsuffixed ``results_phasing/`` (that is 'main') and
        any ``*_backup`` copy. Returns an empty list when no experiment is
        loaded. Mirrors the input requirement of ``handle_visualization``.
        """
        if not self.experiment_dir:
            return []
        out = []
        for path in glob.glob(os.path.join(self.experiment_dir, 'results_phasing_*')):
            if not os.path.isdir(path):
                continue
            base = os.path.basename(path)
            if base.endswith('_backup'):
                continue
            suffix = base[len('results_phasing_'):]
            if not suffix:
                continue
            if any('image.npy' in files for _, _, files in os.walk(path)):
                out.append(suffix)
        return sorted(out)

    def load_configs(self, conf_list: list, no_verify: bool = False) -> tuple[dict, list]:
        """Load multiple configuration files at once.

        Returns ``(loaded_maps, missing_names)``: the dict of names that
        existed and parsed (with any schema migrations applied and the
        version key stripped), and the list of names that were absent.
        """
        if not self.experiment_dir:
            raise ValueError("Experiment directory not set")

        com = _get_common()
        raw_maps, _ = com.get_config_maps(
            self.experiment_dir,
            conf_list,
            no_verify=no_verify,
        )
        cleaned: dict = {}
        for name, raw in raw_maps.items():
            migrated, did_migrate = _migrate(name, raw)
            stripped = _strip_version(migrated)
            if did_migrate:
                self._persist_migration(name, stripped)
            cleaned[name] = stripped
        self._config_maps.update(cleaned)
        missing = [n for n in conf_list if n not in cleaned]
        return cleaned, missing

    def load_config(self, conf_name: str) -> Optional[dict]:
        """Load a single configuration file. Returns None if it doesn't exist.

        The on-disk file is migrated to :data:`SCHEMA_CURRENT_VERSIONS`
        for its name (best-effort write-back) and the
        ``_schema_version`` key is stripped before returning.
        """
        if not self.conf_dir:
            return None

        conf_path = ut.join(self.conf_dir, conf_name)
        if not os.path.isfile(conf_path):
            return None

        raw = ut.read_config(conf_path)
        if not raw:
            return raw
        migrated, did_migrate = _migrate(conf_name, raw)
        stripped = _strip_version(migrated)
        if did_migrate:
            self._persist_migration(conf_name, stripped)
        if stripped:
            self._config_maps[conf_name] = stripped
        return stripped

    def save_config(self, conf_name: str, conf_map: dict,
                    no_verify: bool = False) -> tuple[str, str]:
        """Save configuration to file. Returns ``(error, action)``.

        The on-disk file always includes a ``_schema_version`` stamp;
        the in-memory cache stores the dict without it, matching what
        tabs read via ``get_config()``.

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
        ut.write_config(_stamp_version(conf_name, conf_map), conf_path)
        self._config_maps[conf_name] = conf_map
        return ("", action)

    def _persist_migration(self, conf_name: str, migrated: Optional[dict]) -> None:
        """Write the upgraded dict back to disk after a migration.

        Best-effort: silently no-ops on a read-only checkout or any
        write failure. The next save will overwrite anyway.
        """
        if not self.conf_dir or not migrated:
            return
        try:
            path = ut.join(self.conf_dir, conf_name)
            ut.write_config(_stamp_version(conf_name, migrated), path)
        except Exception as e:
            _log.debug('persist migration for %s skipped: %s', conf_name, e)

    def conf_path(self, conf_name: str) -> Optional[str]:
        """Return the absolute path to ``conf_name`` (whether or not it exists)."""
        if not self.conf_dir:
            return None
        return ut.join(self.conf_dir, conf_name)

    def verify(self, conf_name: str, conf_map: dict) -> str:
        """Verify configuration dictionary. Returns error string (empty if valid).

        Strips ``_schema_version`` before delegating to cohere_core: the
        version key is jupyter_gui metadata, not part of any schema the
        core verifier knows about.

        ``cohere_core.utilities.verify`` returns a tuple when no schema
        is registered (e.g. ``config_mp``); treat that as "no verifier"
        and pass rather than blocking the save.
        """
        clean = _strip_version(conf_map) if conf_map else conf_map
        result = ut.verify(conf_name, clean)
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
        # Compare apples to apples: widget_map never carries the schema
        # version key (tabs.get_config doesn't know about it), so the
        # version stamp would otherwise always look "stale" after a save.
        on_disk = _strip_version(on_disk) or {}
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
