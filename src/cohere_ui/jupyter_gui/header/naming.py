"""Token-based folder-name template for the Create-experiment wizard.

Pure stdlib. ``render()`` formats the template; ``next_serial()`` picks
a non-colliding serial under a parent directory.

Tokens:
  ``{root}``       free-text user input
  ``{date:fmt}``   ``strftime`` format, default ``%Y%m%d``
  ``{serial:fmt}`` integer (auto-bumped) OR scan-range string
                   (e.g. ``"2-7,10-15"``, used verbatim). When the
                   value is a non-integer string and the template
                   spec is an integer format, the spec is stripped
                   before substitution.

A disabled token's placeholder is removed along with one adjacent
underscore so the result has no orphan separators.

``write_parent_template`` persists the picked template at
``<parent>/.cohere_template.json``; ``read_parent_template`` reads it
back so sibling experiments share a template.
"""

import json
import os
import re
from datetime import datetime
from typing import Iterable, Optional, Union

SUPPORTED_TOKENS = ('root', 'date', 'serial')
DEFAULT_TEMPLATE = '{root}_{date:%Y%m%d}_{serial:03d}'
TEMPLATE_FILE = '.cohere_template.json'

_ANY_TOKEN_RE = re.compile(r'\{([A-Za-z_]\w*)(?::[^}]*)?\}')
_SERIAL_TOKEN_RE = re.compile(r'\{serial(?::[^}]*)?\}')

# Match `{token}` or `{token:format-spec}` and eat one adjacent
# underscore (leading if present, else trailing) so removing a token
# doesn't leave an orphan separator.
_token_re_cache: dict[str, re.Pattern] = {}


def _token_pattern(token: str) -> re.Pattern:
    if token not in _token_re_cache:
        _token_re_cache[token] = re.compile(
            rf'(?:_\{{{token}(?::[^}}]*)?\}})|'
            rf'(?:\{{{token}(?::[^}}]*)?\}}_?)'
        )
    return _token_re_cache[token]


def strip_disabled_tokens(template: str, enabled: Iterable[str]) -> str:
    """Remove placeholders for tokens not in ``enabled`` (with one adjacent ``_``)."""
    enabled_set = set(enabled)
    for token in SUPPORTED_TOKENS:
        if token not in enabled_set:
            template = _token_pattern(token).sub('', template)
    return template


def render(
    template: str = DEFAULT_TEMPLATE,
    *,
    root: str = '',
    date: Optional[datetime] = None,
    serial: Union[int, str] = 1,
    enabled: Optional[Iterable[str]] = None,
) -> str:
    """Render ``template`` with the given values.

    ``serial`` may be an int or a string. Non-integer strings cause any
    int-format spec on the ``{serial}`` token to be stripped before
    substitution (so ``{serial:03d}`` with ``serial='2-7'`` renders as
    ``'2-7'`` rather than raising ValueError).
    """
    if enabled is None:
        enabled = SUPPORTED_TOKENS
    tmpl = strip_disabled_tokens(template, enabled)
    if date is None:
        date = datetime.now()
    if isinstance(serial, str):
        if serial.isdigit():
            serial = int(serial)
        else:
            tmpl = _SERIAL_TOKEN_RE.sub('{serial}', tmpl)
    return tmpl.format_map({'root': root, 'date': date, 'serial': serial})


def next_serial(
    parent_dir: str,
    template: str = DEFAULT_TEMPLATE,
    *,
    root: str = '',
    serial: Union[int, str, None] = None,
    date: Optional[datetime] = None,
    enabled: Optional[Iterable[str]] = None,
) -> Union[int, str]:
    """Pick the serial value for the next rendered folder name.

    Caller supplies the user's input as ``serial``:
      - empty / None / integer string  -> auto-bump scan of ``parent_dir``
        returning the next unused integer
      - non-empty non-numeric string (e.g. ``"2-7,10-15"``) -> returned
        verbatim, no scan; caller checks collision via ``folder_exists``

    Returns 1 when the parent dir is missing or when ``serial`` is not
    an enabled token.
    """
    if isinstance(serial, str):
        s = serial.strip()
        if s and not s.isdigit():
            return s
        if s.isdigit():
            serial = int(s)
        else:
            serial = None

    if enabled is None:
        enabled = SUPPORTED_TOKENS
    enabled_set = set(enabled)
    if 'serial' not in enabled_set or not os.path.isdir(parent_dir):
        return serial if isinstance(serial, int) else 1
    m = _SERIAL_TOKEN_RE.search(template)
    if not m:
        return serial if isinstance(serial, int) else 1
    marker = '\x00SERIAL\x00'
    template_with_marker = template.replace(m.group(0), marker)
    template_clean = strip_disabled_tokens(
        template_with_marker, enabled_set | {'serial'},
    )
    if date is None:
        date = datetime.now()
    try:
        rendered = template_clean.format_map(
            {'root': root, 'date': date},
        )
    except (KeyError, IndexError, ValueError):
        return 1
    pattern = re.escape(rendered).replace(re.escape(marker), r'(\d+)')
    regex = re.compile(f'^{pattern}$')
    max_n = 0
    try:
        entries = os.listdir(parent_dir)
    except OSError:
        return 1
    for entry in entries:
        m2 = regex.match(entry)
        if m2:
            try:
                max_n = max(max_n, int(m2.group(1)))
            except ValueError:
                pass
    floor = serial if isinstance(serial, int) and serial > 0 else 1
    return max(max_n + 1, floor)


def folder_exists(parent_dir: str, folder_name: str) -> bool:
    return bool(folder_name) and os.path.isdir(os.path.join(parent_dir, folder_name))


def unknown_tokens(template: str) -> list[str]:
    """Return any ``{name}`` tokens in ``template`` not in SUPPORTED_TOKENS."""
    names = _ANY_TOKEN_RE.findall(template)
    return sorted({n for n in names if n not in SUPPORTED_TOKENS})


def read_parent_template(parent_dir: str) -> Optional[str]:
    """Return the template persisted at ``<parent>/.cohere_template.json``, or None."""
    if not parent_dir or not os.path.isdir(parent_dir):
        return None
    path = os.path.join(parent_dir, TEMPLATE_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        tmpl = data.get('template')
        return tmpl if isinstance(tmpl, str) and tmpl else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def write_parent_template(parent_dir: str, template: str) -> bool:
    """Persist ``template`` to ``<parent>/.cohere_template.json``."""
    if not parent_dir or not os.path.isdir(parent_dir) or not template:
        return False
    path = os.path.join(parent_dir, TEMPLATE_FILE)
    try:
        with open(path, 'w') as f:
            json.dump({'template': template, 'schema_version': 1}, f, indent=2)
        return True
    except OSError:
        return False


if __name__ == '__main__':
    import tempfile
    from datetime import datetime as _dt
    d = _dt(2026, 5, 27)

    # Original int-serial cases.
    assert render(root='cdi', date=d, serial=1) == 'cdi_20260527_001'
    assert render(root='cdi', date=d, serial=42, enabled={'root', 'serial'}) == 'cdi_042'
    assert render(root='cdi', date=d, enabled={'root'}) == 'cdi'
    assert render('{serial:03d}_{root}', root='cdi', serial=7) == '007_cdi'
    assert strip_disabled_tokens(DEFAULT_TEMPLATE, {'root'}) == '{root}'
    assert strip_disabled_tokens(DEFAULT_TEMPLATE, {'root', 'date'}) == '{root}_{date:%Y%m%d}'

    # String-serial cases: int format spec is stripped, value used verbatim.
    assert render(root='cdi', date=d, serial='2-7') == 'cdi_20260527_2-7'
    assert render(root='cdi', date=d, serial='2-7,10-15') == 'cdi_20260527_2-7,10-15'
    assert render(root='aps1ide', date=d, serial='2075-2108',
                  enabled={'root', 'serial'}) == 'aps1ide_2075-2108'
    # Digit-string still uses the int format.
    assert render(root='cdi', date=d, serial='42',
                  enabled={'root', 'serial'}) == 'cdi_042'

    # unknown_tokens detection.
    assert unknown_tokens(DEFAULT_TEMPLATE) == []
    assert unknown_tokens('{root}_{scan}_{date:%Y}') == ['scan']
    assert unknown_tokens('{foo}_{bar}_{root}') == ['bar', 'foo']

    # next_serial with string serial returns the string verbatim.
    with tempfile.TemporaryDirectory() as tmp:
        assert next_serial(tmp, DEFAULT_TEMPLATE, root='cdi', serial='2-7') == '2-7'
        assert next_serial(tmp, DEFAULT_TEMPLATE, root='cdi', serial='') == 1
        assert next_serial(tmp, DEFAULT_TEMPLATE, root='cdi', serial=None) == 1
        # Digit-string is converted and bumped against the parent dir.
        assert next_serial(tmp, DEFAULT_TEMPLATE, root='cdi', serial='7', date=d) == 7

    # write/read parent template round-trip.
    with tempfile.TemporaryDirectory() as tmp:
        assert read_parent_template(tmp) is None
        tmpl = '{root}_{date:%Y-%m-%d}_{serial:04d}'
        assert write_parent_template(tmp, tmpl) is True
        assert read_parent_template(tmp) == tmpl

    print('naming.py self-tests OK')
