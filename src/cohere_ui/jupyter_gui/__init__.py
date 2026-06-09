"""Jupyter Notebook GUI for Cohere Bragg CDI reconstruction.

Install with the Jupyter extra (recommended for notebook users):

    pip install cohere-ui[jupyter]

Non-notebook installs (headless / CLI / desktop GUI) can install plain
``pip install cohere-ui``.

Use:

    from cohere_ui.jupyter_gui import CoherenceGUI
    gui = CoherenceGUI()
    gui.display()

After loading/running an experiment:

    gui.results.image    # numpy array
    gui.results.support
    gui.results.config   # all config dicts
"""

try:
    from cohere_core.utilities.view_utils import (  # noqa: F401
        LiveViewBackend,
        set_default_live_backend,
    )
except ImportError as e:
    raise ImportError(
        "cohere_ui.jupyter_gui requires a cohere_core release that exposes "
        "LiveViewBackend and set_default_live_backend in "
        "cohere_core.utilities.view_utils. Upgrade with: "
        "pip install --upgrade cohere_core"
    ) from e

from cohere_ui.jupyter_gui.core import CoherenceGUI
from cohere_ui.jupyter_gui.config import ConfigManager
from cohere_ui.jupyter_gui.results import ResultsContainer
from cohere_ui.jupyter_gui.header.layout import (
    format_tree, parse_scan, project_layout,
)

# User-overridable path to an ImageJ install (the resolver also finds
# ImageJ-distributions like Fiji automatically). Set this from a notebook
# cell BEFORE clicking the "ImageJ" button if auto-detection misses your
# install:
#
#     import cohere_ui.jupyter_gui as cgui
#     cgui.IMAGEJ_PATH = '/Users/me/Apps/ImageJ.app'   # macOS .app bundle
#     cgui.IMAGEJ_PATH = '/opt/imagej/ImageJ-linux64'  # Linux binary
#     cgui.IMAGEJ_PATH = r'D:\tools\ImageJ.app\ImageJ-win64.exe'  # Windows .exe
#
# Checked BEFORE the IMAGEJ env var and BEFORE any auto-detection,
# so this always wins.
IMAGEJ_PATH: str | None = None

# User-overridable path to a ParaView install. Same precedence rule as
# IMAGEJ_PATH: checked first by the Postprocess tab's "Open in ParaView"
# button, ahead of the PARAVIEW env var and the standard install paths.
#
#     import cohere_ui.jupyter_gui as cgui
#     cgui.PARAVIEW_PATH = '/Applications/ParaView.app'              # macOS .app
#     cgui.PARAVIEW_PATH = '/opt/paraview/bin/paraview'              # Linux binary
#     cgui.PARAVIEW_PATH = r'C:\Program Files\ParaView 5.11\bin\paraview.exe'
PARAVIEW_PATH: str | None = None

__all__ = [
    'CoherenceGUI', 'ConfigManager', 'ResultsContainer',
    'parse_scan', 'project_layout', 'format_tree',
    'IMAGEJ_PATH', 'PARAVIEW_PATH',
]
__version__ = '0.2.0'
