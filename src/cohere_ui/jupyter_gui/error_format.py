"""Helpers for surfacing exceptions to the GUI log panels."""

import ast
import os
import traceback

_PROJECT_ROOTS = ('cohere_core', 'cohere_ui', 'cohere_beamlines')


def _project_frames(exc: BaseException) -> list:
    """Traceback frames whose filename has a project-root segment, outer-to-inner."""
    tb = exc.__traceback__
    if tb is None:
        return []
    return [
        f for f in traceback.extract_tb(tb)
        if any(seg in _PROJECT_ROOTS for seg in f.filename.split(os.sep))
    ]


def _format_frame(frame) -> str:
    """Last two path segments: '<parent>/<basename>:<line> in <func>()'."""
    parts = frame.filename.split(os.sep)
    short = '/'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return f"{short}:{frame.lineno} in {frame.name}()"


def frame_location(exc: BaseException) -> str:
    """Deepest project frame label, or '' when none. Backward-compat shim."""
    frames = _project_frames(exc)
    return _format_frame(frames[-1]) if frames else ''


def format_error_summary(exc: BaseException, prefix: str = '') -> str:
    """One-line summary: trigger-arrow-cause frame tag, [prefix: ]ExcType: msg."""
    frames = _project_frames(exc)
    body = f"{type(exc).__name__}: {exc}"
    if prefix:
        body = f"{prefix}: {body}"
    if not frames:
        return body
    trigger = _format_frame(frames[0])
    cause = _format_frame(frames[-1])
    loc = trigger if trigger == cause else f"{trigger} → {cause}"
    return f"{loc} — {body}"


def safe_parse(field_name: str, value, *, log_error=None, log_debug=None):
    """ast.literal_eval a text-field string; on failure log + return raw value."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError) as e:
        if log_error is not None:
            log_error(
                f"{field_name}: invalid value {value!r} "
                f"- {type(e).__name__}: {e}"
            )
        if log_debug is not None:
            log_debug(traceback.format_exc())
        return value
