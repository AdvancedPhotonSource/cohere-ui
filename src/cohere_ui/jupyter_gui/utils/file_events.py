"""File-modification snapshots used to report what a backend produced.

Each tab's ``run_tab`` snapshots the experiment directory before running a
backend (``snapshot``), then diffs against a fresh snapshot afterwards
(``diff``) to log which files were created or updated. The extension
allowlist filters out caches, lock files, and other noise so the log
stays focused on user-relevant output (TIFFs, npy arrays, VTS volumes,
text).
"""

import os

# Extensions worth surfacing to the user. Backend functions write these;
# everything else is internal scratch.
DEFAULT_EXTS = ('.tif', '.tiff', '.npy', '.vts', '.txt', '.h5', '.hdf5')


def snapshot(root: str, exts=DEFAULT_EXTS) -> dict:
    """Return ``{absolute_path: mtime}`` for every file under ``root`` whose
    extension is in ``exts``. Missing root or unreadable files are skipped.
    """
    out = {}
    if not root or not os.path.isdir(root):
        return out
    for d, _, fs in os.walk(root):
        for f in fs:
            if exts and not f.lower().endswith(exts):
                continue
            p = os.path.join(d, f)
            try:
                out[p] = os.path.getmtime(p)
            except OSError:
                pass
    return out


def diff(before: dict, after: dict) -> tuple[list, list]:
    """Return ``(created, updated)`` lists of absolute paths.

    A path is *created* if it appears only in ``after``; *updated* if its
    mtime differs between the two snapshots. Deleted files are not
    reported (the GUI doesn't currently care about them).
    """
    created = sorted(p for p in after if p not in before)
    updated = sorted(p for p in after if p in before and after[p] != before[p])
    return created, updated
