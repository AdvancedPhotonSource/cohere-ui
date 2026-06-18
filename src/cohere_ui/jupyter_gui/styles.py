"""Button color palette and CSS injection for the Jupyter GUI.

Roles:
  load: gold/tan, used for "load config" actions
  set : light blue, used for "set/create" actions
  run : light green, used for "run/execute" actions
  info: teal, used for "set defaults" / informational actions
"""

import json

from IPython.display import display, HTML, Javascript

BUTTON_PALETTE = {
    'load': 'rgb(205, 178, 102)',
    'set': 'rgb(120, 180, 220)',
    'run': 'rgb(175, 208, 156)',
    'info': 'rgb(0, 151, 167)',
}

CUSTOM_CSS = """
<style>
/* Single source of truth for the GUI's design tokens: the selector/label
   column widths (so the divider, the panel grid template, and any future
   overrides stay in lockstep) AND the color/theme tokens consumed via
   var(--jup-*) throughout the widget tree.

   Each color token prefers the host (JupyterLab / VS Code) --jp-* variable
   and falls back to a literal default, so the GUI follows the host's chosen
   theme automatically (and updates live when the host theme changes,
   since the var() indirection is re-resolved by the browser and nothing
   is baked in).
   The @media (prefers-color-scheme: dark) block below swaps only the
   *fallback* half to dark values for hosts that don't define --jp-* (classic
   notebook, some VS Code states, bare contexts); when the host does define
   --jp-*, that still wins.

   Tokens are declared on :root AND body so the var(--jp-*, ...) indirection
   resolves whether the host defines its --jp-* palette on :root or on body
   (a custom property computes against the scope it's declared in, so a
   :root-only declaration would miss host vars defined lower down). */
:root, body {
    --jup-feature-selector-w: 170px;
    --jup-feature-label-w: 180px;

    --jup-fg:           var(--jp-ui-font-color1, #222);
    --jup-fg-muted:     var(--jp-ui-font-color2, #555);
    --jup-fg-faint:     var(--jp-ui-font-color3, #888);
    --jup-card-bg:      var(--jp-layout-color1, #fafafa);
    --jup-card-bg-2:    var(--jp-layout-color2, #f7f7f7);
    --jup-border:       var(--jp-border-color1, #d0d0d0);
    --jup-border-2:     var(--jp-border-color2, #bbb);
    --jup-accent:       var(--jp-brand-color1, #1670a8);
    --jup-select-bg:    var(--jp-brand-color3, #e0eef9);
    --jup-error:        var(--jp-error-color1, #a02020);
    --jup-error-bg:     var(--jp-error-color3, #fdecea);
    --jup-error-border: var(--jp-error-color2, #f5c2c0);
    --jup-success:      var(--jp-success-color1, #1e7a1e);
    --jup-warn:         var(--jp-warn-color1, #a06000);
}

@media (prefers-color-scheme: dark) {
  :root, body {
    --jup-fg:           var(--jp-ui-font-color1, #e6e6e6);
    --jup-fg-muted:     var(--jp-ui-font-color2, #b8b8b8);
    --jup-fg-faint:     var(--jp-ui-font-color3, #8a8a8a);
    --jup-card-bg:      var(--jp-layout-color1, #2b2b2b);
    --jup-card-bg-2:    var(--jp-layout-color2, #333333);
    --jup-border:       var(--jp-border-color1, #555555);
    --jup-border-2:     var(--jp-border-color2, #666666);
    --jup-select-bg:    var(--jp-brand-color3, #1a3a52);
    --jup-error-bg:     var(--jp-error-color3, #3a1f1f);
    --jup-error-border: var(--jp-error-color2, #5a2a2a);
  }
}

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

/* Set/Create buttons, light blue rgb(120, 180, 220) */
.jup-gui-set {
    background-color: rgb(120, 180, 220) !important;
    border-color: rgb(100, 160, 200) !important;
    color: #000 !important;
}
.jup-gui-set:hover {
    background-color: rgb(100, 160, 200) !important;
}

/* Run buttons, light green rgb(175, 208, 156) */
.jup-gui-run {
    background-color: rgb(175, 208, 156) !important;
    border-color: rgb(155, 188, 136) !important;
    color: #000 !important;
}
.jup-gui-run:hover {
    background-color: rgb(155, 188, 136) !important;
}

/* Split-run caret: a one-glyph down-caret (U+25BC) button. ipywidgets'
   default ~10px side padding leaves a ~28px button too little room for the
   glyph, so it collapses to an ellipsis; zero the padding so it shows.
   Double selector covers both ipywidgets 7 and 8 DOM trees. */
.jup-gui-caret button,
.jup-gui-caret.widget-button {
    padding: 0 !important;
}

/* Info/defaults buttons, keep teal */
.jup-gui-info {
    background-color: rgb(0, 151, 167) !important;
    border-color: rgb(0, 131, 147) !important;
    color: #fff !important;
}
.jup-gui-info:hover {
    background-color: rgb(0, 131, 147) !important;
}

/* Path inputs: show the tail of overflowing paths (filename, not the
   long parent dirs). RTL scroll-anchors the end; text-align:left keeps
   typing left-to-right. */
.jup-gui-path-input input {
    direction: rtl !important;
    text-align: left !important;
}

/* FeaturePanel: rounded card wrapping selector + params as one section.
   The body is a real two-column grid so the divider (border-left on the
   params column) tracks the selector edge automatically, the previous
   ::after pseudo-element was pinned to a hard-coded `left: 170px` that
   drifted whenever the selector's rendered width changed. */
.jup-gui-feature-body {
    display: grid !important;
    grid-template-columns: var(--jup-feature-selector-w) minmax(0, 1fr) !important;
    align-items: stretch;
    border: 1px solid var(--jup-border) !important;
    border-radius: 4px !important;
    background: var(--jup-card-bg) !important;
    overflow: hidden;   /* clip selector edge to the rounded corners */
}
/* The .jup-gui-feature-params rule below sets `border-left` directly with
   !important so it always wins against ipywidgets' .widget-box defaults
   and any leftover `border: none` shorthand. */

/* Feature selector: monospace, tight spacing, subtle highlight. */
.jup-gui-feature-select select {
    font-family: Menlo, Consolas, monospace !important;
    font-size: 12px !important;
    padding: 2px 4px !important;
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
}
.jup-gui-feature-select select option {
    padding: 4px 6px;
}
.jup-gui-feature-select select option:checked {
    background: var(--jup-select-bg) !important;
    color: #000 !important;
    font-weight: 600;
}

/* Params area: borderless on three sides; the left border is the
   divider between selector and params columns of the feature body grid
   (matches the body card's outline via var(--jup-border)). */
.jup-gui-feature-params {
    background: transparent !important;
    border-top: none !important;
    border-right: none !important;
    border-bottom: none !important;
    border-left: 1px solid var(--jup-border) !important;
}

/* Feature title acts as a clickable heading that toggles the description.
   Strip the underlying ipywidgets Button chrome (border, background, shadow,
   padding) and left-align the label. Hover hints at the affordance.
   The double selector covers both ipywidgets 7 and 8 DOM trees. */
.jup-gui-feature-title-toggle button,
.jup-gui-feature-title-toggle.widget-button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--jup-fg) !important;
    font-weight: 600 !important;
    font-size: 1.05em !important;
    text-align: left !important;
    padding: 0 !important;
    cursor: pointer;
}
.jup-gui-feature-title-toggle:hover button,
.jup-gui-feature-title-toggle.widget-button:hover {
    color: var(--jup-accent) !important;
}

/* Per-feature params box: a two-column grid (label | value). Override
   ipywidgets' default `display: flex; flex-direction: column` on .widget-vbox
   with !important. Rows are emitted via the `grid_field` / `grid_full`
   helpers in widgets.py; both render direct children of params_box so the
   grid sees them as cells. */
.jup-gui-feature-grid {
    display: grid !important;
    grid-template-columns: var(--jup-feature-label-w) minmax(0, 1fr);
    column-gap: 12px;
    row-gap: 4px;
    align-items: center;
}

/* A normal labeled row: the HBox itself has no box, its label and widget
   are promoted to direct grid cells of the parent .jup-gui-feature-grid.
   `display: contents` is supported in every Chromium/Firefox-based
   JupyterLab front-end. */
.jup-gui-feature-row {
    display: contents !important;
}

/* Hide a feature row. The `display: contents !important` above outranks an
   inline display:none, so conditional rows toggle this higher-specificity
   class instead (see widgets.set_row_visible). */
.jup-gui-feature-row.jup-gui-row-hidden {
    display: none !important;
}

/* Postprocess tab: collapsible VTS/VTI viewer panel. Bordered card so
   it reads as a distinct section under the action row + log panel.
   Border/background use the same tokens as the FeaturePanel card so the
   two visually pair (and theme together in dark mode). */
.jup-gui-vts-viewer-frame {
    border: 1px solid var(--jup-border);
    border-radius: 4px;
    background: var(--jup-card-bg);
}

/* Reconstruction tab: live-view snapshot + error-history plot row.
   The HBox already has flex_flow='row wrap' via ipywidgets layout; this
   rule adds the inter-image gap (which ipywidgets Layout doesn't expose
   as a traitlet) so spacing is identical whether the two images sit
   side-by-side on a wide viewport or stacked on a narrow one.
   `box-sizing: border-box` on each image makes its 1 px border count
   inside max-width, so the calc() in `_IMAGE_SIZE_OPTIONS['M']` does
   not under-budget by the border width (which would otherwise force
   the second image onto a new line). */
.jup-gui-rec-images {
    gap: 12px;
}
.jup-gui-rec-images > .widget-image {
    box-sizing: border-box;
    /* Flex items default to min-width:auto (their intrinsic content
       width). The rendered PNGs are wider than half the row, so without
       this they refuse to shrink below their natural width and wrap onto
       a second line even though max-width caps them at calc(50% - 8px).
       min-width:0 lets flex-grow honor the cap so the M layout keeps both
       images side-by-side. (.widget-image is the <img> itself; ipywidgets
       already ships max-width:100% for it, so no extra img rule is needed.) */
    min-width: 0;
}

/* Log output boxes: user-resizable height via the browser's resize grip.
   The grip needs a non-visible overflow, which the log widgets set inline
   (auto / hidden). min-height keeps the box from collapsing to nothing when
   dragged all the way up. The grip lives on the outer fixed-height widget;
   the inner render div is height:100% and follows. */
.jup-gui-log-box {
    resize: vertical !important;
    min-height: 40px;
}

/* A row that spans both grid columns: bare checkboxes, separators,
   action buttons, and collapsible sub-sections. The display rule
   intentionally omits !important so that callers can hide the row via
   `widget.layout.display = 'none'` (inline style beats unimportant
   author CSS). The default for ipywidgets .widget-box is already
   display:flex, so this rule mainly ensures cell positioning. The
   justify-self is set per-instance via the widget's inline layout
   (default 'start'; 'end' for action buttons like Set Defaults). */
.jup-gui-feature-row-full {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
}
</style>
"""

# DOM id used to dedupe the <style> tag we append to document.head.
# Stable so repeat injections replace the prior copy instead of stacking.
_STYLE_DOM_ID = 'cohere-jup-gui-styles'


def _css_body() -> str:
    """Return CUSTOM_CSS without the wrapping <style>...</style> tags so it
    can be assigned to a <style>.textContent in JS."""
    body = CUSTOM_CSS.strip()
    if body.startswith('<style>'):
        body = body[len('<style>'):]
    if body.endswith('</style>'):
        body = body[:-len('</style>')]
    return body


def inject_custom_css():
    """Install the GUI's stylesheet in two places:

    1. As a ``<style>`` element in the current cell's output area
       (``display(HTML(...))``). This copy is saved to the .ipynb on
       disk, so opening the notebook in a fresh tab shows the prior
       run's widgets already styled before the kernel runs anything.

    2. As a ``<style id=cohere-jup-gui-styles>`` element appended to
       ``document.head`` via a small JS snippet. This copy lives outside
       any cell's output, so it survives:
         - re-executing ``gui.display()`` (clears the cell's output area)
         - ``Clear Outputs`` / ``Clear All Outputs``
         - ``Restart Kernel & Clear Output``

    Without (2), the prior single-call-and-flag implementation silently
    became a no-op after the user cleared output, and widgets reverted
    to ipywidgets defaults (wrong button colors, ungrided features)
    until a full page reload restored the saved HTML output. The JS
    snippet first removes any existing element with the same id so
    repeat calls refresh rather than stacking duplicates in head.
    """
    display(HTML(CUSTOM_CSS))
    js = (
        "(function() {"
        f" var ID = {json.dumps(_STYLE_DOM_ID)};"
        " var prior = document.getElementById(ID);"
        " if (prior) prior.remove();"
        " var style = document.createElement('style');"
        " style.id = ID;"
        f" style.textContent = {json.dumps(_css_body())};"
        " document.head.appendChild(style);"
        "})();"
    )
    display(Javascript(js))


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
