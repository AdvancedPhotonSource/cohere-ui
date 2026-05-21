"""PyVista off-screen renderer for the Jupyter GUI's live view.

Renders an isosurface of |ds_image| (colored by phase) to a PNG via
``pyvista.Plotter(off_screen=True)`` and ships the bytes back to the parent
kernel via the multiprocessing Queue, same protocol as the matplotlib
backend. Suitable for 3D context that a 2D matplotlib slice can't show.

An interactive trame-backed mode is planned for a follow-up; it requires
the trame widget to live in the parent kernel's notebook session, which
this subprocess-only backend doesn't address.
"""

from cohere_core.utilities.view_utils import LiveViewBackend


class PyVistaBackend(LiveViewBackend):
    """3D isosurface live view rendered as a PNG per fire.

    :param msg_queue: ``multiprocessing.Queue`` for shipping snapshot
        records back to the parent kernel.
    :param stride: integer >= 1; spatial downsample applied on-device before
        the volume crosses the GPU-to-host boundary. Higher = faster transfer
        and faster pyvista mesh extraction at the cost of resolution.
    :param iso_level: amplitude threshold (relative to max) used to
        extract the isosurface.
    :param window_size: pyvista off-screen window size in pixels.
    """

    def __init__(self, msg_queue, stride=4, iso_level=0.3, window_size=(640, 480)):
        self._queue = msg_queue
        self.stride = max(1, int(stride))
        self.iso_level = float(iso_level)
        self.window_size = tuple(window_size)

    def select_singlepeak_data(self, ds_image, support, devlib):
        s = self.stride
        ds = ds_image[::s, ::s, ::s]
        sup = support[::s, ::s, ::s] if support is not None else None
        return ds, sup

    def update_singlepeak(self, ds_image, errors, support, title=""):
        try:
            png = self._render(ds_image, support, title)
        except Exception as e:
            self._send({'kind': 'message', 'level': 'warning',
                        'text': f'pyvista render failed: {e}'})
            return
        last_err = float(errors[-1]) if len(errors) else None
        iter_n = self._iter_from_title(title)
        self._send({
            'kind': 'snapshot',
            'iter': iter_n,
            'error': last_err,
            'image_bytes': png,
        })

    def update_multipeak_fourier(self, proj, mask, meas, data, title=""):
        pass

    def update_multipeak_direct(self, rho, u0, u1, u2, title=""):
        pass

    def save(self, save_as):
        pass

    def block(self):
        pass

    def _send(self, record):
        try:
            self._queue.put(record, block=False)
        except Exception as e:
            import sys
            sys.__stderr__.write(
                f"PyVistaBackend._send: dropped {record.get('kind', '?')} "
                f"record ({type(e).__name__}: {e})\n"
            )

    @staticmethod
    def _iter_from_title(title):
        try:
            head = title.split('\n', 1)[0]
            n_part = head.split(':', 1)[1].strip().split('/')[0]
            return int(n_part)
        except Exception:
            return None

    def _render(self, image, support, title):
        import io
        import numpy as np
        import pyvista as pv

        amp = np.abs(image).astype(np.float32)
        peak = float(amp.max()) if amp.size else 0.0
        threshold = peak * self.iso_level if peak > 0 else 0.0

        # Build a UniformGrid carrying amplitude and phase as point arrays.
        grid = pv.ImageData()
        grid.dimensions = np.array(amp.shape) + 1  # cell-data shape vs dim convention
        grid.spacing = (1.0, 1.0, 1.0)
        grid.cell_data['amplitude'] = amp.flatten(order='F')
        if support is not None:
            grid.cell_data['support'] = support.astype(np.float32).flatten(order='F')
        # Convert cell-data to point-data for contouring
        grid = grid.cell_data_to_point_data()

        plotter = pv.Plotter(off_screen=True, window_size=self.window_size)
        plotter.set_background('white')
        if peak > 0:
            iso = grid.contour([threshold], scalars='amplitude')
            plotter.add_mesh(iso, color='steelblue', opacity=0.7,
                             show_scalar_bar=False)
        plotter.add_text(title.replace('\n', ' '), font_size=10, color='black')
        plotter.add_axes()
        png = plotter.screenshot(return_img=True)
        plotter.close()

        # plotter.screenshot returns an HxWx(3 or 4) numpy array; encode as PNG.
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(png).save(buf, format='PNG')
        return buf.getvalue()
