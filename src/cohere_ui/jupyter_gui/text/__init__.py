"""User-facing text bundles for the Jupyter GUI.

YAML files in this directory hold short HTML strings rendered to users
(feature descriptions, info banners, log messages). Use ``load_text``
to get a parsed dict; results are cached.

ipywidgets 8's ``HTMLMath`` is broken in JupyterLab 4 (calls a MathJax
v2 API that no longer exists), so LaTeX in descriptions wouldn't
typeset. The loader runs a substitution pass that converts the handful
of TeX commands we use into Unicode glyphs; extend the ``_TEX`` table
below if you add more.
"""

import re
from functools import lru_cache
from pathlib import Path

import yaml

_TEXT_DIR = Path(__file__).parent

# Strip math-mode delimiters; the content inside is then substituted.
_DELIMS = re.compile(r'\\[\(\)\[\]]')

# \text{x} -> x.
_TEXT_CMD = re.compile(r'\\text\{([^}]*)\}')

_TEX = {
    'pi': chr(0x03c0), 'sigma': chr(0x03c3), 'sum': chr(0x03a3),
    'geq': chr(0x2265), 'approx': chr(0x2248), 'pm': chr(0x00b1), 'times': chr(0x00d7),
}
_CMD_PATTERN = re.compile(r'\\(' + '|'.join(_TEX) + r')(?![A-Za-z])')

# Escaped specials we use: \% only, for now.
_ESCAPED = re.compile(r'\\([%])')


def _tex_to_unicode(s: str) -> str:
    if not isinstance(s, str) or '\\' not in s:
        return s
    s = _DELIMS.sub('', s)
    s = _TEXT_CMD.sub(r'\1', s)
    s = _CMD_PATTERN.sub(lambda m: _TEX[m.group(1)], s)
    s = _ESCAPED.sub(r'\1', s)
    return s


def _convert_tree(obj):
    if isinstance(obj, str):
        return _tex_to_unicode(obj)
    if isinstance(obj, dict):
        return {k: _convert_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_tree(v) for v in obj]
    return obj


@lru_cache(maxsize=None)
def load_text(name: str) -> dict:
    """Load and cache a text bundle by name (without the .yaml extension)."""
    path = _TEXT_DIR / f'{name}.yaml'
    return _convert_tree(yaml.safe_load(path.read_text()))
