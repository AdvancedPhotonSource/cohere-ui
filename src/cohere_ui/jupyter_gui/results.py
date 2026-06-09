"""
ResultsContainer for exposing analysis variables to notebook users.
"""

import os
import sys
from typing import Callable, Optional
import numpy as np

import cohere_core.utilities as ut

import traceback

from cohere_ui.jupyter_gui.utils.error_format import format_error_summary


# Files larger than this fall back to memory-mapped loading by default.
# 2 GB is a deliberate floor, well below the 4 GB phasing volumes that
# come out of binned 3D BCDI scans, but high enough that small inspection
# loads stay fully in RAM where slicing is cheap.
DEFAULT_MMAP_THRESHOLD_BYTES = 2 * 1024 * 1024 * 1024


class ResultsContainer:
    """Container for lazy-loaded reconstruction results.

    Exposes experiment data and results as properties for interactive analysis.
    """

    def __init__(self, config_manager=None,
                 mmap_threshold_bytes: int = DEFAULT_MMAP_THRESHOLD_BYTES,
                 force_mmap: bool = False):
        self._config_manager = config_manager
        self._data = None
        self._image = None
        self._support = None
        self._coherence = None
        self._errors = None
        # Files at/above the threshold load read-only via memory map
        # so multi-GB phasing volumes don't exhaust kernel memory.
        # force_mmap bypasses the threshold for slicing small files.
        self._mmap_threshold_bytes = int(mmap_threshold_bytes)
        self._force_mmap = bool(force_mmap)
        # Default sink is stderr; CoherenceGUI rewires to log_panel.error after _build_ui.
        self._log_error: Callable[[str], object] = lambda msg: sys.stderr.write(msg + "\n")
        self._log_debug: Callable[[str], object] = lambda msg: sys.stderr.write(msg + "\n")

    def set_config_manager(self, config_manager):
        """Set or update the config manager reference."""
        self._config_manager = config_manager
        self.reload()

    @property
    def experiment_dir(self) -> Optional[str]:
        """Current experiment directory path."""
        if self._config_manager:
            return self._config_manager.experiment_dir
        return None

    @property
    def config(self) -> dict:
        """All loaded configuration maps."""
        if self._config_manager:
            return self._config_manager.all_configs
        return {}

    @property
    def data(self) -> Optional[np.ndarray]:
        """Diffraction data from phasing_data/data.tif or data.npy."""
        if self._data is None:
            self._data = self._load_data()
        return self._data

    @property
    def image(self) -> Optional[np.ndarray]:
        """Reconstructed image from results_phasing/image.npy."""
        if self._image is None:
            self._image = self._load_result('image.npy')
        return self._image

    @property
    def support(self) -> Optional[np.ndarray]:
        """Support array from results_phasing/support.npy."""
        if self._support is None:
            self._support = self._load_result('support.npy')
        return self._support

    @property
    def coherence(self) -> Optional[np.ndarray]:
        """Coherence function from results_phasing/coherence.npy (if PCDI used)."""
        if self._coherence is None:
            self._coherence = self._load_result('coherence.npy')
        return self._coherence

    @property
    def errors(self) -> Optional[np.ndarray]:
        """Error metrics from results_phasing/errors.npy."""
        if self._errors is None:
            self._errors = self._load_result('errors.npy')
        return self._errors

    def _load_data(self) -> Optional[np.ndarray]:
        """Load diffraction data."""
        if not self.experiment_dir:
            return None

        data_dir = ut.join(self.experiment_dir, 'phasing_data')

        for filename in ['data.npy', 'data.tif']:
            filepath = ut.join(data_dir, filename)
            if os.path.isfile(filepath):
                return self._load_array(filepath)

        return None

    def _load_result(self, filename: str) -> Optional[np.ndarray]:
        """Load a result file from results_phasing directory."""
        if not self.experiment_dir:
            return None

        results_dir = ut.join(self.experiment_dir, 'results_phasing')
        filepath = ut.join(results_dir, filename)

        if os.path.isfile(filepath):
            return self._load_array(filepath)

        return None

    def _load_array(self, filepath: str) -> Optional[np.ndarray]:
        """Load an array from ``.npy`` or ``.tif``.

        Files at or above :attr:`_mmap_threshold_bytes` (or whenever
        ``force_mmap`` is set) load via read-only memory map so
        multi-GB phasing volumes do not exhaust kernel memory.
        Slicing and viewing work on the mmap; writing does not.
        """
        try:
            try:
                size = os.path.getsize(filepath)
            except OSError:
                size = -1
            use_mmap = self._force_mmap or (
                size >= 0 and size >= self._mmap_threshold_bytes
            )
            if use_mmap:
                self._log_debug(
                    f'_load_array: mmap loading {filepath} '
                    f'({size / (1024 * 1024):.0f} MB)'
                )
            if filepath.endswith('.npy'):
                return np.load(filepath, mmap_mode='r') if use_mmap else np.load(filepath)
            elif filepath.endswith('.tif'):
                import tifffile as tf
                return tf.memmap(filepath) if use_mmap else tf.imread(filepath)
        except MemoryError as e:
            self._log_error(
                f'_load_array: out of memory loading {filepath}; '
                f'try ResultsContainer(force_mmap=True) to memory-map instead. '
                f'{format_error_summary(e)}'
            )
            self._log_debug(traceback.format_exc())
        except Exception as e:
            self._log_error(
                format_error_summary(e, prefix='_load_array'))
            self._log_debug(traceback.format_exc())
        return None

    def reload(self):
        """Force reload all cached data."""
        self._data = None
        self._image = None
        self._support = None
        self._coherence = None
        self._errors = None

    def list_results(self) -> list:
        """List available result files in results_phasing directory."""
        if not self.experiment_dir:
            return []

        results_dir = ut.join(self.experiment_dir, 'results_phasing')
        if not os.path.isdir(results_dir):
            return []

        return [f for f in os.listdir(results_dir) if f.endswith(('.npy', '.tif'))]

    def __repr__(self):
        exp = self.experiment_dir or "(not set)"
        return f"ResultsContainer(experiment_dir={exp})"
