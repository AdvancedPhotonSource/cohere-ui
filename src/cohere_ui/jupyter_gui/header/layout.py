"""Project-structure preview for the Create-experiment wizard.

Given a folder name, a scan string, and a mode, sketches the dirs the
cohere_ui backend will create. Pure stdlib; never touches the filesystem.

Modes mirror the four-way radio in the wizard:
  - ``'single'``               -> top-level preprocessed_data / phasing_data
                                  / results_phasing / results_viz
  - ``'separate_scans'``       -> one ``scan_N/`` subdir per individual scan
                                  in the parsed scan ranges
  - ``'separate_scan_ranges'`` -> one ``scan_<a>-<b>/`` per comma-separated
                                  range (intra-range scans are aligned +
                                  summed by the backend)
  - ``'multipeak'``            -> one ``mp_<range>_<hkl>/`` per
                                  (range, orientation) pair; shared
                                  ``results_phasing/`` at the experiment root
"""

import re
from typing import Literal, Optional

Mode = Literal['single', 'separate_scans', 'separate_scan_ranges', 'multipeak']
Role = Literal['conf', 'prep', 'phasing', 'results', 'viz', 'mp', 'overflow']

DEFAULT_OVERFLOW = 500
DEFAULT_COLLAPSE_THRESHOLD = 8

_SCAN_TOKEN_RE = re.compile(r'^(\d+)(?:-(\d+))?$')


def parse_scan(s: str) -> list[tuple[int, int]]:
    """Parse a scan string into a list of ``(start, end)`` integer ranges.

    Examples:
      ``"2-7,10-15"`` -> ``[(2, 7), (10, 15)]``
      ``"54"``        -> ``[(54, 54)]``
      ``"2,3,4"``     -> ``[(2, 2), (3, 3), (4, 4)]``
      ``""``          -> ``[]``

    Raises ``ValueError`` on any malformed token.
    """
    text = (s or '').replace(' ', '')
    if not text:
        return []
    out: list[tuple[int, int]] = []
    for unit in text.split(','):
        if not unit:
            raise ValueError(f'empty scan token in {s!r}')
        m = _SCAN_TOKEN_RE.match(unit)
        if not m:
            raise ValueError(f'unparseable scan token {unit!r} in {s!r}')
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) is not None else a
        if b < a:
            raise ValueError(f'reversed range {unit!r} in {s!r}')
        out.append((a, b))
    return out


def _hkl_label(orientation: list) -> str:
    """Render an orientation list as the backend's mp_<scan>_<hkl> tail.

    Matches the convention in cohere_ui/api/multipeak.py: each integer
    contributes its decimal form, with sign preserved (e.g. ``[1,1,-1]``
    -> ``'11-1'``).
    """
    return ''.join(str(int(x)) for x in orientation)


def project_layout(
    *,
    folder_name: str,
    scan: str,
    mode: Mode,
    orientations: Optional[list] = None,
    max_entries: int = DEFAULT_OVERFLOW,
) -> list[tuple[str, Role]]:
    """Return ``[(relpath, role), ...]`` sketching the dirs to create.

    Relative paths use forward slashes and end with ``'/'``. Order is
    top-level first, then per-subdir groups. Output is capped at
    ``max_entries`` with a trailing ``('...', 'overflow')`` line.
    """
    out: list[tuple[str, Role]] = [('conf/', 'conf')]
    if mode == 'single':
        out.extend([
            ('preprocessed_data/', 'prep'),
            ('phasing_data/', 'phasing'),
            ('results_phasing/', 'results'),
            ('results_viz/', 'viz'),
        ])
        return _cap(out, max_entries)

    ranges = parse_scan(scan)
    if mode == 'separate_scans':
        for a, b in ranges:
            for n in range(a, b + 1):
                base = f'scan_{n}/'
                out.extend([
                    (base + 'preprocessed_data/', 'prep'),
                    (base + 'phasing_data/', 'phasing'),
                    (base + 'results_phasing/', 'results'),
                    (base + 'results_viz/', 'viz'),
                ])
                if len(out) >= max_entries:
                    return _cap(out, max_entries)
        return _cap(out, max_entries)

    if mode == 'separate_scan_ranges':
        # Keep the user's literal token (a==b -> "N", else "a-b") so the
        # preview matches what the backend actually writes.
        units = (s for s in scan.replace(' ', '').split(',') if s)
        for unit in units:
            base = f'scan_{unit}/'
            out.extend([
                (base + 'preprocessed_data/', 'prep'),
                (base + 'phasing_data/', 'phasing'),
                (base + 'results_phasing/', 'results'),
                (base + 'results_viz/', 'viz'),
            ])
            if len(out) >= max_entries:
                return _cap(out, max_entries)
        return _cap(out, max_entries)

    if mode == 'multipeak':
        units = [u for u in scan.replace(' ', '').split(',') if u]
        orients = orientations or []
        if not orients:
            # Placeholder slot per range so the user sees the shape even
            # before populating MpTab orientations.
            for unit in units:
                out.append((f'mp_{unit}_<hkl>/', 'mp'))
                if len(out) >= max_entries:
                    return _cap(out, max_entries)
        else:
            for unit, orient in zip(units, orients):
                out.append((f'mp_{unit}_{_hkl_label(orient)}/', 'mp'))
                if len(out) >= max_entries:
                    return _cap(out, max_entries)
        # Combined result at the experiment root.
        out.append(('results_phasing/', 'results'))
        return _cap(out, max_entries)

    raise ValueError(f'unknown mode {mode!r}')


def _cap(items: list[tuple[str, Role]], n: int) -> list[tuple[str, Role]]:
    if len(items) <= n:
        return items
    return items[:n] + [('...', 'overflow')]


def format_tree(
    entries: list[tuple[str, Role]],
    root: str,
    *,
    collapse_threshold: int = DEFAULT_COLLAPSE_THRESHOLD,
) -> str:
    """Render ``entries`` as a UTF-8 box-drawing tree rooted at ``root``.

    Entries are grouped by their first path segment so each parent dir
    appears once with its children nested under box-drawing connectors.

    When the number of top-level groups that have children exceeds
    ``collapse_threshold``, the standard boilerplate children
    (preprocessed_data/, phasing_data/, results_phasing/, results_viz/)
    are dropped for ALL such groups and only the parent dir lines
    remain. This keeps the preview scannable on huge separate_scans
    runs without losing the structure.

    Returns one string with ``\\n`` separators; suitable for ``print()``
    or for embedding in an HTML ``<pre>`` block.
    """
    groups: list[list] = []  # [name, children_or_None]
    by_top: dict[str, int] = {}
    for relpath, _role in entries:
        if relpath == '...':
            groups.append(['...', None])
            continue
        rstripped = relpath.rstrip('/')
        if '/' in rstripped:
            top_seg, rest = rstripped.split('/', 1)
            top = top_seg + '/'
            child = rest + '/'
            if top not in by_top:
                by_top[top] = len(groups)
                groups.append([top, []])
            groups[by_top[top]][1].append(child)
        else:
            top = rstripped + '/'
            if top not in by_top:
                by_top[top] = len(groups)
                groups.append([top, None])

    n_with_children = sum(1 for _, ch in groups if ch)
    collapse = n_with_children > collapse_threshold

    lines = [f'{root.rstrip("/")}/']
    last_idx = len(groups) - 1
    for i, (name, children) in enumerate(groups):
        is_last = (i == last_idx)
        connector = '└── ' if is_last else '├── '
        lines.append(f'{connector}{name}')
        if not children or collapse:
            continue
        child_prefix = '    ' if is_last else '│   '
        child_last = len(children) - 1
        for j, child in enumerate(children):
            cc = '└── ' if j == child_last else '├── '
            lines.append(f'{child_prefix}{cc}{child}')
    return '\n'.join(lines)


if __name__ == '__main__':
    # Single mode.
    s = project_layout(folder_name='cdi_001', scan='', mode='single')
    assert s == [
        ('conf/', 'conf'),
        ('preprocessed_data/', 'prep'),
        ('phasing_data/', 'phasing'),
        ('results_phasing/', 'results'),
        ('results_viz/', 'viz'),
    ], s

    # parse_scan cases.
    assert parse_scan('2-7,10-15') == [(2, 7), (10, 15)]
    assert parse_scan('54') == [(54, 54)]
    assert parse_scan('2,3,4') == [(2, 2), (3, 3), (4, 4)]
    assert parse_scan('') == []
    try:
        parse_scan('abc')
        raise AssertionError('expected ValueError')
    except ValueError:
        pass
    try:
        parse_scan('7-2')
        raise AssertionError('expected ValueError')
    except ValueError:
        pass

    # separate_scans expansion.
    s = project_layout(folder_name='cdi_x', scan='2-3,5', mode='separate_scans')
    bases = {p.split('/', 1)[0] + '/' for p, _ in s if p != 'conf/'}
    assert bases == {'scan_2/', 'scan_3/', 'scan_5/'}, bases

    # separate_scan_ranges keeps literal tokens.
    s = project_layout(folder_name='cdi_x', scan='2-7,10-15',
                       mode='separate_scan_ranges')
    bases = {p.split('/', 1)[0] for p, _ in s if p != 'conf/'}
    assert bases == {'scan_2-7', 'scan_10-15'}, bases

    # multipeak with no orientations -> placeholder per range.
    s = project_layout(folder_name='cdi_x', scan='898-913,919-934',
                       mode='multipeak')
    assert ('mp_898-913_<hkl>/', 'mp') in s
    assert ('mp_919-934_<hkl>/', 'mp') in s
    assert ('results_phasing/', 'results') in s

    # multipeak with orientations.
    s = project_layout(folder_name='cdi_x', scan='898-913,919-934',
                       mode='multipeak',
                       orientations=[[1, 1, 1], [1, 1, -1]])
    paths = [p for p, _ in s]
    assert 'mp_898-913_111/' in paths
    assert 'mp_919-934_11-1/' in paths
    assert 'results_phasing/' in paths

    # Overflow cap.
    s = project_layout(folder_name='cdi_x', scan='1-100',
                       mode='separate_scans', max_entries=10)
    assert s[-1] == ('...', 'overflow')
    assert len(s) == 11

    # format_tree: small case keeps children expanded.
    entries = project_layout(folder_name='exp', scan='54-59,9',
                             mode='separate_scan_ranges')
    tree = format_tree(entries, root='exp')
    expected = (
        'exp/\n'
        '├── conf/\n'
        '├── scan_54-59/\n'
        '│   ├── preprocessed_data/\n'
        '│   ├── phasing_data/\n'
        '│   ├── results_phasing/\n'
        '│   └── results_viz/\n'
        '└── scan_9/\n'
        '    ├── preprocessed_data/\n'
        '    ├── phasing_data/\n'
        '    ├── results_phasing/\n'
        '    └── results_viz/'
    )
    assert tree == expected, repr(tree)

    # format_tree: many groups -> children collapsed.
    big = project_layout(folder_name='exp', scan='1-12',
                         mode='separate_scans')
    big_tree = format_tree(big, root='exp')
    assert '├── scan_1/' in big_tree
    assert '└── scan_12/' in big_tree
    assert 'preprocessed_data/' not in big_tree, big_tree

    # format_tree: single mode (no scan groups).
    s = project_layout(folder_name='cdi_001', scan='', mode='single')
    t = format_tree(s, root='cdi_001')
    assert t.splitlines()[0] == 'cdi_001/'
    assert '├── conf/' in t
    assert '└── results_viz/' in t

    # format_tree: multipeak with orientations.
    mp = project_layout(folder_name='cdi_x', scan='898-913,919-934',
                        mode='multipeak',
                        orientations=[[1, 1, 1], [1, 1, -1]])
    mp_tree = format_tree(mp, root='cdi_x')
    assert '├── mp_898-913_111/' in mp_tree
    assert '├── mp_919-934_11-1/' in mp_tree
    assert '└── results_phasing/' in mp_tree

    # format_tree: overflow marker shown.
    cap = project_layout(folder_name='cdi_x', scan='1-100',
                         mode='separate_scans', max_entries=12)
    cap_tree = format_tree(cap, root='cdi_x')
    assert '...' in cap_tree

    print('layout.py self-tests OK')
