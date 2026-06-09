"""ParaView install resolution + launch.

Mirrors ``viewers/imagej.py`` exactly for the VTS/VTI files written by
the Postprocess pipeline. Two public functions:

  * ``resolve_paraview_path()`` -> ``(cmd_prefix, source_label, tried)``
  * ``open_in_paraview(path)`` -> ``(ok, message)``

The two helpers are kept separate so the launcher button in the Disp
tab can render an informative "ParaView not found - tried these paths"
status without actually trying to launch.

Resolution order (highest precedence first):

  1. ``cohere_ui.jupyter_gui.PARAVIEW_PATH`` (Python module variable;
     users override this from a notebook cell BEFORE clicking).
  2. ``PARAVIEW`` env var (also accepted: ``PARAVIEW_PATH``).
  3. macOS standard install paths + ``mdfind`` Spotlight query for
     ``ParaView*.app``.
  4. Linux standard install paths (``/opt/paraview/bin/paraview``,
     ``/snap/bin/paraview``, etc.).
  5. Windows standard install paths
     (``C:\\Program Files\\ParaView*\\bin\\paraview.exe``).
  6. ``shutil.which('paraview')`` (PATH-based last resort).

ParaView accepts a ``.vts`` or ``.vti`` file as a positional argument
and opens it directly, so ``open_in_paraview(path)`` simply appends the
file path to the resolved command prefix.
"""

import glob
import os
import shutil
import subprocess
import sys


def resolve_paraview_path():
    """Return ``(cmd_prefix, source_label, tried_paths)``.

    ``cmd_prefix`` is the list to pass to ``subprocess.Popen`` BEFORE
    appending the file path. Returns ``(None, '', tried)`` when nothing
    is found; ``tried`` is the full list of candidates checked so the
    caller can surface it in the UI.
    """
    tried = []

    def _accept(path_or_app):
        """Convert any candidate path into a launch command, or None.

        On macOS, ``ParaView.app`` bundles are opened via ``open -a``
        so the system event dispatch wires the .vts argument up to the
        running ParaView instance (single-instance behavior). Anywhere
        else, the path must be an executable file.
        """
        if not path_or_app:
            return None
        tried.append(path_or_app)
        if sys.platform == 'darwin' and path_or_app.endswith('.app'):
            if os.path.isdir(path_or_app):
                # `open -a APP --args FILE` lets ParaView receive FILE
                # as a CLI arg. Without --args, `open` passes it as a
                # file-open event which ParaView also honors, but --args
                # is more deterministic across ParaView versions.
                return ['open', '-a', path_or_app, '--args']
        if os.path.isfile(path_or_app) and os.access(path_or_app, os.X_OK):
            return [path_or_app]
        return None

    # 1. Module-level user override (set from notebook).
    try:
        from cohere_ui import jupyter_gui as _jg
        override = getattr(_jg, 'PARAVIEW_PATH', None)
    except Exception:
        override = None
    if override:
        cmd = _accept(override)
        if cmd:
            return cmd, 'cohere_ui.jupyter_gui.PARAVIEW_PATH', tried

    # 2. Environment override. PARAVIEW_PATH accepted as a courtesy alias
    # because some build instructions document that name.
    for env_name in ('PARAVIEW', 'PARAVIEW_PATH'):
        val = os.environ.get(env_name)
        if val:
            cmd = _accept(val)
            if cmd:
                return cmd, f'${env_name}', tried

    # 3. Per-OS standard install locations.
    home = os.path.expanduser('~')
    if sys.platform == 'darwin':
        # ParaView installs as a versioned .app: ParaView-5.11.2.app.
        # Glob each common parent for both the unversioned and the
        # newest versioned bundle so we don't miss installs in /Downloads
        # that the user hasn't moved to /Applications yet.
        mac_parents = ('/Applications', f'{home}/Applications',
                       f'{home}/Downloads', f'{home}/Desktop')
        mac_apps = []
        for parent in mac_parents:
            mac_apps.append(f'{parent}/ParaView.app')
            # Sort by name descending so the highest version wins
            # (lexicographic order is good enough for ParaView-MAJOR.MINOR.PATCH).
            versioned = sorted(
                glob.glob(f'{parent}/ParaView-*.app'),
                reverse=True,
            )
            mac_apps.extend(versioned)
        for cand in mac_apps:
            cmd = _accept(cand)
            if cmd:
                return cmd, 'standard install path', tried
        # mdfind Spotlight fallback: finds ParaView wherever the user
        # installed it (external volumes, Dropbox, etc.).
        try:
            out = subprocess.run(
                ['mdfind', 'kMDItemKind == "Application" && kMDItemFSName == "ParaView*.app"'],
                capture_output=True, text=True, timeout=5,
            )
            # Newer-named bundles first.
            for line in sorted(out.stdout.splitlines(), reverse=True):
                line = line.strip()
                if not line:
                    continue
                cmd = _accept(line)
                if cmd:
                    return cmd, 'mdfind (Spotlight)', tried
        except Exception:
            pass
    elif sys.platform.startswith('linux'):
        # Glob for versioned ParaView-MAJOR.MINOR.PATCH/bin/paraview
        # inside each parent in case the user extracted a release tarball
        # without symlinking; sorted descending so newest version wins.
        linux_parents = (
            '/opt', '/usr/local', f'{home}', f'{home}/Apps',
            f'{home}/Applications', f'{home}/Downloads',
        )
        linux_cands = [
            '/usr/bin/paraview',
            '/usr/local/bin/paraview',
            '/opt/paraview/bin/paraview',
            '/snap/bin/paraview',
            f'{home}/bin/paraview',
        ]
        for parent in linux_parents:
            for hit in sorted(
                glob.glob(f'{parent}/ParaView-*/bin/paraview'),
                reverse=True,
            ):
                linux_cands.append(hit)
            for hit in sorted(
                glob.glob(f'{parent}/paraview-*/bin/paraview'),
                reverse=True,
            ):
                linux_cands.append(hit)
        for cand in linux_cands:
            cmd = _accept(cand)
            if cmd:
                return cmd, 'standard install path', tried
    elif sys.platform.startswith('win'):
        user_profile = os.environ.get('USERPROFILE', '')
        local_appdata = os.environ.get('LOCALAPPDATA', '')
        program_files = os.environ.get('ProgramFiles', r'C:\Program Files')
        program_files_x86 = os.environ.get(
            'ProgramFiles(x86)', r'C:\Program Files (x86)')
        win_parents = (
            program_files, program_files_x86,
            rf'{local_appdata}\Programs', user_profile,
            rf'{user_profile}\Desktop', rf'{user_profile}\Downloads',
        )
        win_cands = []
        for parent in win_parents:
            if not parent:
                continue
            # Versioned: "ParaView 5.11.2\bin\paraview.exe"
            for hit in sorted(
                glob.glob(rf'{parent}\ParaView *\bin\paraview.exe'),
                reverse=True,
            ):
                win_cands.append(hit)
            for hit in sorted(
                glob.glob(rf'{parent}\ParaView-*\bin\paraview.exe'),
                reverse=True,
            ):
                win_cands.append(hit)
            # Unversioned fallback.
            win_cands.append(rf'{parent}\ParaView\bin\paraview.exe')
        for cand in win_cands:
            cmd = _accept(cand)
            if cmd:
                return cmd, 'standard install path', tried

    # 6. PATH-based fallback. shutil.which honors PATHEXT on Windows.
    for binname in ('paraview', 'paraview.exe'):
        found = shutil.which(binname)
        tried.append(f'$PATH/{binname}')
        if found:
            return [found], f'$PATH ({binname})', tried

    return None, '', tried


def open_in_paraview(path: str) -> tuple[bool, str]:
    """Launch ParaView with ``path`` as the initial dataset.

    Returns ``(ok, message)``. On success, ``ok=True`` and ``message``
    is the resolved command for the log. On failure, ``ok=False`` and
    ``message`` explains why and lists the paths that were tried so
    the user can pick the right override.
    """
    if not path or not os.path.exists(path):
        return False, f'ParaView: file not found: {path!r}'
    cmd_prefix, source, tried = resolve_paraview_path()
    if cmd_prefix is None:
        bullet_list = '\n  - '.join(tried) if tried else '(no candidates)'
        return False, (
            'ParaView not found. Set the install path with:\n'
            '    import cohere_ui.jupyter_gui as cgui\n'
            "    cgui.PARAVIEW_PATH = '/Applications/ParaView.app'  # macOS\n"
            "    cgui.PARAVIEW_PATH = '/opt/paraview/bin/paraview'  # Linux\n"
            "    cgui.PARAVIEW_PATH = r'C:\\Program Files\\ParaView 5.11\\bin\\paraview.exe'\n"
            'or export $PARAVIEW before launching Jupyter.\n'
            f'Tried:\n  - {bullet_list}'
        )
    cmd = cmd_prefix + [path]
    try:
        # Detached: ParaView is a long-lived GUI; we don't want the
        # kernel to wait on it or to inherit its stderr.
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        return False, (
            f'ParaView launch failed: {type(e).__name__}: {e}\n'
            f'  command: {cmd!r}\n'
            f'  source:  {source}'
        )
    return True, f'Launched ParaView ({source}) with {os.path.basename(path)}'
