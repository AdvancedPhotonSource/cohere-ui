"""ImageJ install resolution.

Single function: ``resolve_imagej_path()``. Returns a launch command
prefix (a list, ready to be appended with the file path and passed to
``subprocess.Popen``), the source label that produced it, and the list
of paths checked along the way.

Resolution order:
  1. ``cohere_ui.jupyter_gui.IMAGEJ_PATH`` (Python module variable)
  2. ``IMAGEJ`` env var (with ``FIJI`` accepted as a backward-compat alias)
  3. macOS standard install paths + ``mdfind`` Spotlight query
  4. Linux standard install paths
  5. Windows standard install paths
  6. ``shutil.which`` (PATH-based lookup, last resort)

Recognizes both ImageJ and ImageJ-derived distributions like Fiji
(Fiji is ImageJ-2 with extra plugins).
"""

import os
import shutil
import sys


def resolve_imagej_path():
    """Return ``(cmd_prefix, source_label, tried_paths)``.

    Returns ``(None, '', tried)`` when nothing is found.
    """
    tried = []

    def _accept(path_or_app):
        """Convert any candidate path into a launch command, or None."""
        if not path_or_app:
            return None
        tried.append(path_or_app)
        if sys.platform == 'darwin' and path_or_app.endswith('.app'):
            if os.path.isdir(path_or_app):
                return ['open', '-a', path_or_app]
        if os.path.isfile(path_or_app) and os.access(path_or_app, os.X_OK):
            return [path_or_app]
        return None

    # 1. Module-level user override (set from notebook).
    try:
        from cohere_ui import jupyter_gui as _jg
        override = getattr(_jg, 'IMAGEJ_PATH', None)
    except Exception:
        override = None
    if override:
        cmd = _accept(override)
        if cmd:
            return cmd, 'cohere_ui.jupyter_gui.IMAGEJ_PATH', tried

    # 2. Environment override. Accept FIJI as a quiet alias.
    for env_name in ('IMAGEJ', 'FIJI'):
        val = os.environ.get(env_name)
        if val:
            cmd = _accept(val)
            if cmd:
                return cmd, f'${env_name}', tried

    # 3. Per-OS standard install locations.
    home = os.path.expanduser('~')
    if sys.platform == 'darwin':
        mac_apps = []
        for parent in ('/Applications', f'{home}/Applications',
                       f'{home}/Downloads', f'{home}/Desktop'):
            for app in ('Fiji.app', 'ImageJ.app'):
                mac_apps.append(f'{parent}/{app}')
        for cand in mac_apps:
            cmd = _accept(cand)
            if cmd:
                return cmd, 'standard install path', tried
        # mdfind Spotlight fallback: finds a Fiji/ImageJ .app anywhere
        # the user has installed it (even on external volumes / Dropbox).
        try:
            import subprocess as _sp
            for query in ('-name Fiji.app', '-name ImageJ.app'):
                out = _sp.run(
                    ['mdfind'] + query.split(),
                    capture_output=True, text=True, timeout=5,
                )
                for line in out.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    cmd = _accept(line)
                    if cmd:
                        return cmd, 'mdfind (Spotlight)', tried
        except Exception:
            pass
    elif sys.platform.startswith('linux'):
        linux_cands = [
            f'{home}/Fiji.app/ImageJ-linux64',
            f'{home}/Applications/Fiji.app/ImageJ-linux64',
            f'{home}/Downloads/Fiji.app/ImageJ-linux64',
            f'{home}/.local/share/Fiji.app/ImageJ-linux64',
            f'{home}/bin/Fiji.app/ImageJ-linux64',
            '/opt/fiji/ImageJ-linux64',
            '/opt/Fiji.app/ImageJ-linux64',
            '/usr/local/Fiji.app/ImageJ-linux64',
            '/usr/local/fiji/ImageJ-linux64',
            '/snap/fiji/current/ImageJ-linux64',
            f'{home}/Fiji.app/ImageJ-linux32',
            '/opt/Fiji.app/ImageJ-linux32',
        ]
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
        win_cands = [
            rf'{program_files}\Fiji.app\ImageJ-win64.exe',
            rf'{program_files_x86}\Fiji.app\ImageJ-win32.exe',
            rf'{local_appdata}\Programs\Fiji.app\ImageJ-win64.exe',
            r'C:\Fiji.app\ImageJ-win64.exe',
            r'C:\Fiji.app\ImageJ-win32.exe',
            rf'{user_profile}\Fiji.app\ImageJ-win64.exe',
            rf'{user_profile}\Desktop\Fiji.app\ImageJ-win64.exe',
            rf'{user_profile}\Downloads\Fiji.app\ImageJ-win64.exe',
        ]
        for cand in win_cands:
            cmd = _accept(cand)
            if cmd:
                return cmd, 'standard install path', tried

    # 6. PATH-based fallback. shutil.which honors PATHEXT on Windows.
    for binname in ('ImageJ-linux64', 'ImageJ-win64.exe',
                    'imagej', 'fiji', 'ImageJ', 'Fiji'):
        found = shutil.which(binname)
        tried.append(f'$PATH/{binname}')
        if found:
            return [found], f'$PATH ({binname})', tried

    return None, '', tried
