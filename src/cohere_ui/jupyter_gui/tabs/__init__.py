"""Tab implementations for the Jupyter GUI."""

from cohere_ui.jupyter_gui.tabs.base import BaseTab
from cohere_ui.jupyter_gui.tabs.data import DataTab
from cohere_ui.jupyter_gui.tabs.prep import PrepTab
from cohere_ui.jupyter_gui.tabs.rec import RecTab
from cohere_ui.jupyter_gui.tabs.disp import DispTab
from cohere_ui.jupyter_gui.tabs.instr import InstrTab
from cohere_ui.jupyter_gui.tabs.mp import MpTab

__all__ = ['BaseTab', 'DataTab', 'PrepTab', 'RecTab', 'DispTab', 'InstrTab', 'MpTab']
