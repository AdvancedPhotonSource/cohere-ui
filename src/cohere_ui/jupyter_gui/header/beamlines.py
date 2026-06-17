"""Beamline discovery + per-beamline InstrTab schema loader.

``list_beamlines()`` scans the installed ``cohere_beamlines`` package
for subdirs that look like beamline modules.
``load_instr_schema(beamline)`` imports the beamline's
``instr_schema`` sibling module, falling back to a minimal generic
schema for beamlines that don't ship one.
``list_diffractometers(beamline)`` / ``list_detectors(beamline)``
introspect the beamline's ``diffractometers.py`` / ``detectors.py`` for
class-level ``name`` attributes, used to populate dropdowns from schema
fields declared with ``auto_choices='diffractometer'`` (or
``'detector'``). Results are cached; beamline modules are discovered at
install time, not edited at runtime.
"""

import importlib
import importlib.util
import inspect
import os
from typing import Optional

_EXCLUDED = frozenset({'common'})

# A beamline module is identified by the presence of diffractometers.py.
_MARKER = 'diffractometers.py'

_list_cache: Optional[list[str]] = None
_schema_cache: dict[str, tuple[dict, tuple]] = {}
_choices_cache: dict[tuple[str, str], list[str]] = {}


def _package_dir() -> Optional[str]:
    spec = importlib.util.find_spec('cohere_beamlines')
    if spec is None or spec.origin is None:
        return None
    return os.path.dirname(spec.origin)


def list_beamlines() -> list[str]:
    """Return sorted list of selectable beamline names (case-insensitive sort)."""
    global _list_cache
    if _list_cache is not None:
        return list(_list_cache)
    pkg_dir = _package_dir()
    if pkg_dir is None:
        _list_cache = []
        return []
    found = []
    try:
        entries = os.listdir(pkg_dir)
    except OSError:
        _list_cache = []
        return []
    for entry in entries:
        if entry.startswith('_') or entry.startswith('.') or entry in _EXCLUDED:
            continue
        full = os.path.join(pkg_dir, entry)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, _MARKER)):
            found.append(entry)
    _list_cache = sorted(found, key=str.lower)
    return list(_list_cache)


def clear_caches():
    """Reset list and schema caches. Test-only; production paths don't call this."""
    global _list_cache
    _list_cache = None
    _schema_cache.clear()
    _choices_cache.clear()


def _list_named_classes(beamline: str, module_name: str) -> list[str]:
    """Return sorted unique ``name`` attributes from classes defined in
    ``cohere_beamlines.<beamline>.<module_name>``.

    Used to populate diffractometer / detector dropdowns. Returns an
    empty list when the module can't be imported (missing optional dep,
    syntax error) or has no named classes. Successful results cache;
    failures don't, so a maintainer can fix the beamline module and
    re-trigger the lookup without restarting the kernel.
    """
    key = (beamline, module_name)
    if key in _choices_cache:
        return list(_choices_cache[key])
    try:
        mod = importlib.import_module(f'cohere_beamlines.{beamline}.{module_name}')
    except Exception:
        return []
    names: list[str] = []
    for _, cls in inspect.getmembers(mod, inspect.isclass):
        if getattr(cls, '__module__', '') != mod.__name__:
            continue
        name_attr = getattr(cls, 'name', None)
        if isinstance(name_attr, str) and name_attr:
            names.append(name_attr)
    out = sorted(set(names))
    _choices_cache[key] = out
    return list(out)


def list_diffractometers(beamline: str) -> list[str]:
    """Names of diffractometers registered in the beamline's diffractometers.py."""
    return _list_named_classes(beamline, 'diffractometers')


def list_detectors(beamline: str) -> list[str]:
    """Names of detectors registered in the beamline's detectors.py."""
    return _list_named_classes(beamline, 'detectors')


def list_motors(beamline: str) -> list[str]:
    """Display names of the diffractometer's sample + detector axes.

    Pulled from the classes' ``sampleaxes_name`` / ``detectoraxes_name``
    tuples. These are display names, not necessarily spec-file keys, but
    schemas should declare (display, key) pairs in their scanmot
    ``choices`` rather than relying on these.
    """
    key = (beamline, 'motors')
    if key in _choices_cache:
        return list(_choices_cache[key])
    try:
        mod = importlib.import_module(f'cohere_beamlines.{beamline}.diffractometers')
    except Exception:
        return []
    motors: set = set()
    for _, cls in inspect.getmembers(mod, inspect.isclass):
        if getattr(cls, '__module__', '') != mod.__name__:
            continue
        for attr in ('sampleaxes_name', 'detectoraxes_name'):
            vals = getattr(cls, attr, None)
            if isinstance(vals, (tuple, list)):
                motors.update(str(v) for v in vals if v)
    out = sorted(motors, key=str.lower)
    _choices_cache[key] = out
    return list(out)


def normalize_field(spec) -> dict:
    """Convert a field spec (tuple or dict) into a canonical dict.

    Tuple shape (backward compat):
      ``(key, label, placeholder)`` -> text
      ``(key, label, placeholder, 'bool')`` -> checkbox

    Dict shape (preferred):
      ``{'key', 'label', 'placeholder', 'type', 'unit', 'description',
        'choices', 'auto_choices'}``, all optional except ``key`` + ``label``.

    Recognised ``type`` values:
      ``'text'`` (default), ``'bool'``, ``'choice'``, ``'dir'``,
      ``'file'``, ``'float'``, ``'int'``.
    """
    if isinstance(spec, dict):
        out = dict(spec)
    elif isinstance(spec, (list, tuple)):
        out = {
            'key': spec[0],
            'label': spec[1],
            'placeholder': spec[2] if len(spec) > 2 else '',
        }
        if len(spec) > 3:
            out['type'] = spec[3]
    else:
        raise TypeError(f'Unexpected schema field spec: {spec!r}')
    out.setdefault('type', 'text')
    out.setdefault('placeholder', '')
    out.setdefault('unit', '')
    out.setdefault('description', '')
    return out


def resolve_choices(beamline: str, spec: dict) -> list[str]:
    """Resolve a normalised spec's ``choices`` / ``auto_choices`` to a list."""
    choices = spec.get('choices')
    if choices:
        return list(choices)
    auto = spec.get('auto_choices')
    if auto == 'diffractometer':
        return list_diffractometers(beamline)
    if auto == 'detector':
        return list_detectors(beamline)
    if auto == 'scan_motor':
        return list_motors(beamline)
    return []


_GENERIC_FALLBACK_FIELDS = {
    'general': [
        ('diffractometer', 'diffractometer', ''),
        ('data_dir',       'data directory', ''),
    ],
    'spec': [],
}


def load_instr_schema(beamline: str) -> tuple[dict, tuple]:
    """Return ``(INSTR_FIELDS, SPEC_DRIVERS)`` for ``beamline``.

    On ImportError (no ``instr_schema`` for the beamline), returns a
    minimal generic fallback so the InstrTab still renders.
    """
    if beamline in _schema_cache:
        return _schema_cache[beamline]
    fields: Optional[dict] = None
    drivers: tuple = ('specfile', 'diffractometer')
    try:
        mod = importlib.import_module(f'cohere_beamlines.{beamline}.instr_schema')
        fields = getattr(mod, 'INSTR_FIELDS', None)
        drivers = getattr(mod, 'SPEC_DRIVERS', drivers)
    except ImportError:
        pass
    if fields is None:
        fields = _GENERIC_FALLBACK_FIELDS
    _schema_cache[beamline] = (fields, drivers)
    return fields, drivers


if __name__ == '__main__':
    names = list_beamlines()
    print(f'Detected {len(names)} beamline(s): {names}')
    for name in names:
        fields, drivers = load_instr_schema(name)
        n_gen = len(fields.get('general', []))
        n_spec = len(fields.get('spec', []))
        diffs = list_diffractometers(name)
        dets = list_detectors(name)
        print(f'  {name}: general={n_gen}, spec={n_spec}, drivers={drivers}')
        print(f'    diffractometers={diffs}, detectors={dets}')

    # normalize_field smoke tests.
    assert normalize_field(('k', 'L', 'p')) == {
        'key': 'k', 'label': 'L', 'placeholder': 'p',
        'type': 'text', 'unit': '', 'description': '',
    }
    assert normalize_field(('k', 'L', '', 'bool')) == {
        'key': 'k', 'label': 'L', 'placeholder': '',
        'type': 'bool', 'unit': '', 'description': '',
    }
    n = normalize_field({'key': 'x', 'label': 'X', 'type': 'choice',
                         'choices': ['a', 'b']})
    assert n['choices'] == ['a', 'b'] and n['type'] == 'choice'
    print('beamlines.py self-tests OK')
