"""Subprocess plumbing for the visualization/postprocess step.

Mirrors :mod:`cohere_ui.jupyter_gui.rec_subprocess` but for
``beamline_postprocess.handle_visualization``: it runs off the kernel
thread so the notebook stays responsive and the backend's stdout streams
live into the Postprocess tab's log. A child crash (e.g. the macOS
libomp/xrayutilities abort noted in CLAUDE.md) is reported via the exit
code instead of restarting the kernel.
"""

from cohere_ui.jupyter_gui.viz_subprocess.monitor import VizMonitor
from cohere_ui.jupyter_gui.viz_subprocess.runner import run_visualization

__all__ = ['VizMonitor', 'run_visualization']
