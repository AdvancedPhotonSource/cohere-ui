"""Button color palette and CSS injection for the Jupyter GUI.

Roles:
  load - gold/tan, used for "load config" actions
  set  - light blue, used for "set/create" actions
  run  - light green, used for "run/execute" actions
  info - teal, used for "set defaults" / informational actions
"""

from IPython.display import display, HTML

BUTTON_PALETTE = {
    'load': 'rgb(205, 178, 102)',
    'set': 'rgb(120, 180, 220)',
    'run': 'rgb(175, 208, 156)',
    'info': 'rgb(0, 151, 167)',
}

CUSTOM_CSS = """
<style>
/* Button color palette */

/* Load buttons - gold/tan rgb(205, 178, 102) */
.jup-gui-load {
    background-color: rgb(205, 178, 102) !important;
    border-color: rgb(185, 158, 82) !important;
    color: #000 !important;
}
.jup-gui-load:hover {
    background-color: rgb(185, 158, 82) !important;
}

/* Set/Create buttons - light blue rgb(120, 180, 220) */
.jup-gui-set {
    background-color: rgb(120, 180, 220) !important;
    border-color: rgb(100, 160, 200) !important;
    color: #000 !important;
}
.jup-gui-set:hover {
    background-color: rgb(100, 160, 200) !important;
}

/* Run buttons - light green rgb(175, 208, 156) */
.jup-gui-run {
    background-color: rgb(175, 208, 156) !important;
    border-color: rgb(155, 188, 136) !important;
    color: #000 !important;
}
.jup-gui-run:hover {
    background-color: rgb(155, 188, 136) !important;
}

/* Info/defaults buttons - keep teal */
.jup-gui-info {
    background-color: rgb(0, 151, 167) !important;
    border-color: rgb(0, 131, 147) !important;
    color: #fff !important;
}
.jup-gui-info:hover {
    background-color: rgb(0, 131, 147) !important;
}
</style>
"""

_css_injected = False


def inject_custom_css():
    """Inject the GUI's button-palette CSS into the notebook."""
    global _css_injected
    if not _css_injected:
        display(HTML(CUSTOM_CSS))
        _css_injected = True


def apply_button_role(btn, role: str):
    """Apply a role-based color class to a button widget.

    Args:
        btn: ipywidgets Button
        role: One of 'load', 'set', 'run', 'info'
    """
    class_name = f'jup-gui-{role}'
    if hasattr(btn, 'add_class'):
        btn.add_class(class_name)
    else:
        current = btn._dom_classes if hasattr(btn, '_dom_classes') else ()
        btn._dom_classes = tuple(set(current) | {class_name})
