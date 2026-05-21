"""HTML log rendering and error-history plotting.

Pure rendering helpers: log lines in -> HTML out, error points in ->
PNG bytes out. No subprocess, no threads, no widgets owned here (the
caller owns the widgets that receive these outputs).
"""

import html as _html
import io


_LEVEL_STYLE = {
    'info':    'color:#222;',
    'success': 'color:#1e7a1e;font-weight:600;',
    'warning': 'color:#a06000;',
    'error':   'color:#a02020;font-weight:600;',
    'debug':   'color:#777;font-style:italic;',
}
_LEVEL_PREFIX = {
    'info':    '',
    'success': '[OK] ',
    'warning': '[WARN] ',
    'error':   '[ERROR] ',
    'debug':   '[DEBUG] ',
}


def render_log_html(lines, show_debug: bool = False):
    """Render (level, text) lines as colored HTML pinned to the bottom; bare strings -> info."""
    rows_html = []
    visible_idx = 0
    for entry in lines:
        if isinstance(entry, tuple):
            level, text = entry
        else:
            level, text = 'info', entry
        if level == 'debug' and not show_debug:
            continue
        style = _LEVEL_STYLE.get(level, '')
        prefix = _LEVEL_PREFIX.get(level, '')
        rows_html.append(
            f'<div style="order:{-visible_idx};{style}">'
            f'{prefix}{_html.escape(text)}</div>'
        )
        visible_idx += 1
    return (
        '<div style="height:100%;overflow-y:auto;'
        'display:flex;flex-direction:column-reverse;'
        'font-family:Menlo,Consolas,monospace;font-size:11px;'
        'line-height:1.35;padding:6px;background:#fafafa;'
        'color:#222;white-space:pre-wrap;word-break:break-word;">'
        + ''.join(rows_html) + '</div>'
    )


def render_error_plot(points) -> bytes:
    """PNG of the convergence plot; returns b'' only if matplotlib is missing, else raises."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return b''
    iters = [p[0] for p in points]
    errs = [p[1] for p in points]
    fig, ax = plt.subplots(figsize=(8, 2.5))
    try:
        ax.semilogy(iters, errs)
        ax.set_xlabel('iteration')
        ax.set_ylabel('error (log)')
        ax.grid(True, which='both', alpha=0.3)
        ax.set_title(f'Error history ({len(points)} points)')
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80)
        return buf.getvalue()
    finally:
        plt.close(fig)
