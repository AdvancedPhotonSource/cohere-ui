"""Live-view backends used by jupyter_gui's reconstruction subprocess.

Each backend subclasses ``cohere_core.utilities.view_utils.LiveViewBackend``
and ships rendered output (PNG bytes or raw volume data) back to the parent
kernel via the ``multiprocessing.Queue`` it receives at construction.
"""

from .matplotlib_png import JupyterMatplotlibBackend
from .pyvista_png import PyVistaBackend

__all__ = ['JupyterMatplotlibBackend', 'PyVistaBackend']
