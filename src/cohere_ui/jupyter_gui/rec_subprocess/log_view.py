"""HTML log rendering and error-history plotting.

Pure rendering helpers: log lines in -> HTML out, error points in ->
PNG bytes out. No subprocess, no threads, no widgets owned here (the
caller owns the widgets that receive these outputs).
"""

import html as _html
import io


def render_log_html(lines):
    """Render log lines as a colored HTML block pinned to the bottom.

    Uses the chat-app CSS trick: each line is its own flex child; outer
    is column-reverse with overflow-y:auto. DOM order = newest first.
    column-reverse positions the FIRST DOM child at the visual bottom,
    so newest appears at the bottom of the box. The scroll position in
    column-reverse anchors at the start of the DOM (= visual bottom),
    so the box stays pinned to the newest line as more arrive (no JS
    needed).
    """
    rows = ''.join(
        f'<div>{_html.escape(ln)}</div>' for ln in reversed(lines)
    )
    return (
        '<div style="height:100%;overflow-y:auto;'
        'display:flex;flex-direction:column-reverse;'
        'font-family:Menlo,Consolas,monospace;font-size:11px;'
        'line-height:1.35;padding:6px;background:#fafafa;'
        'color:#222;white-space:pre-wrap;word-break:break-word;">'
        f'{rows}</div>'
    )


def render_error_plot(points) -> bytes:
    """Render the convergence plot as PNG bytes. Returns b'' on failure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return b''
    try:
        iters = [p[0] for p in points]
        errs = [p[1] for p in points]
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.semilogy(iters, errs)
        ax.set_xlabel('iteration')
        ax.set_ylabel('error (log)')
        ax.grid(True, which='both', alpha=0.3)
        ax.set_title(f'Error history ({len(points)} points)')
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return b''
