"""Matplotlib-PNG backend for the Jupyter GUI's live view.

Renders each ``update_singlepeak`` call to a PNG using matplotlib's headless
``Agg`` backend, then ships the bytes back to the parent kernel via the
multiprocessing Queue. The parent displays them in an ``ipywidgets.Image``.
"""

from cohere_core.utilities.view_utils import LiveViewBackend


class JupyterMatplotlibBackend(LiveViewBackend):
    """Snapshot-style live view rendered as a PNG per fire.

    Constructor params: ``msg_queue`` (parent-kernel Queue), ``mode``
    ('center_slice' or 'strided_3d'), ``slice_axis``, ``slice_method``,
    ``phase_cmap``, ``apply_support_mask``. See the GUI's Live feature
    description in ``text/features.yaml`` for user-facing guidance.
    """

    def __init__(self, msg_queue, mode='center_slice', slice_axis=2,
                 slice_method='center_of_mass', stride=4, phase_cmap='twilight',
                 apply_support_mask=True):
        self._queue = msg_queue
        self.mode = mode
        self.slice_axis = int(slice_axis)
        self.slice_method = slice_method
        self.stride = max(1, int(stride))
        self.phase_cmap = phase_cmap
        self.apply_support_mask = bool(apply_support_mask)

    def select_singlepeak_data(self, ds_image, support, devlib):
        if self.mode == 'strided_3d':
            return ds_image, support
        # center_slice: one 2D slice along self.slice_axis. Compute the index
        # device-side (cheap) so only the slice crosses the @use_numpy boundary.
        axis = self.slice_axis
        if self.slice_method == 'center_of_array':
            idx = ds_image.shape[axis] // 2
        else:  # center_of_mass
            com = devlib.center_of_mass(devlib.absolute(ds_image))
            idx = int(com[axis])
        sl = [slice(None)] * 3
        sl[axis] = idx
        sl = tuple(sl)
        ds = ds_image[sl]
        sup = support[sl] if support is not None else None
        return ds, sup

    def update_singlepeak(self, ds_image, errors, support, title=""):
        try:
            png = self._render(ds_image, support, errors, title)
        except Exception as e:
            self._send({'kind': 'message', 'level': 'warning',
                        'text': f'snapshot render failed: {e}'})
            return
        last_err = float(errors[-1]) if len(errors) else None
        # iter is parsed from cohere's title (the title is "Iteration: N/M\nError: ...")
        iter_n = self._iter_from_title(title)
        self._send({
            'kind': 'snapshot',
            'iter': iter_n,
            'error': last_err,
            'image_bytes': png,
        })

    def update_multipeak_fourier(self, proj, mask, meas, data, title=""):
        # Multipeak rendering not exposed in the GUI yet; accept the call so
        # cohere doesn't blow up, but skip rendering.
        pass

    def update_multipeak_direct(self, rho, u0, u1, u2, title=""):
        pass

    def save(self, save_as):
        # Each fire is shipped to the parent; the parent decides whether to
        # save. Subprocess save is intentionally a no-op.
        pass

    def block(self):
        pass

    def _send(self, record):
        try:
            self._queue.put(record, block=False)
        except Exception:
            pass

    @staticmethod
    def _iter_from_title(title):
        # cohere's live_operation title: "Iteration: N/M\nError: ..."
        try:
            head = title.split('\n', 1)[0]
            n_part = head.split(':', 1)[1].strip().split('/')[0]
            return int(n_part)
        except Exception:
            return None

    def _render(self, image, support, errors, title):
        import io
        import numpy as np
        import matplotlib
        matplotlib.use('Agg', force=False)
        import matplotlib.pyplot as plt

        if image.ndim == 3 and self.mode == 'strided_3d':
            return self._render_mosaic(image, support, errors, plt, np, io)
        return self._render_single(image, support, errors, plt, np, io)

    def _render_single(self, image, support, errors, plt, np, io):
        if image.ndim == 3:
            mid = image.shape[0] // 2
            image = image[mid]
            if support is not None:
                support = support[mid]
        amp = np.abs(image)
        phase = np.angle(image)
        masked = support is not None and self.apply_support_mask
        if masked:
            phase = np.where(support > 0.5, phase, np.nan)
        ncols = 3 if support is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        if ncols == 1:
            axes = [axes]
        im0 = axes[0].imshow(amp, cmap='gray')
        axes[0].set_title('Amplitude'); axes[0].set_xticks([]); axes[0].set_yticks([])
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cmap = plt.get_cmap(self.phase_cmap).copy()
        cmap.set_bad(color=(0, 0, 0, 0))
        im1 = axes[1].imshow(phase, cmap=cmap, vmin=-np.pi, vmax=np.pi)
        axes[1].set_title('Phase' + (' (support-masked)' if masked else ''))
        axes[1].set_xticks([]); axes[1].set_yticks([])
        cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04,
                           ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cb1.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        if support is not None:
            im2 = axes[2].imshow(support, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('Support'); axes[2].set_xticks([]); axes[2].set_yticks([])
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80)
        plt.close(fig)
        return buf.getvalue()

    def _render_mosaic(self, image, support, errors, plt, np, io):
        # 3 rows (one per axis) x 3 columns (amp / phase / support). Slice
        # index per axis follows slice_method, matching the 2D-slice mode.
        if self.slice_method == 'center_of_mass':
            from scipy.ndimage import center_of_mass
            com = center_of_mass(np.absolute(image))
            slice_idx = [int(c) for c in com]
        else:
            slice_idx = [s // 2 for s in image.shape]
        ncols = 3 if support is not None else 2
        fig, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 12))
        cmap = plt.get_cmap(self.phase_cmap).copy()
        cmap.set_bad(color=(0, 0, 0, 0))
        method_label = 'CoM' if self.slice_method == 'center_of_mass' else 'mid'
        for row, axis in enumerate((0, 1, 2)):
            sl = [slice(None)] * 3
            sl[axis] = slice_idx[axis]
            sl = tuple(sl)
            sub_img = image[sl]
            sub_sup = support[sl] if support is not None else None
            amp = np.abs(sub_img)
            phase = np.angle(sub_img)
            masked = sub_sup is not None and self.apply_support_mask
            if masked:
                phase = np.where(sub_sup > 0.5, phase, np.nan)
            axes[row, 0].imshow(amp, cmap='gray')
            axes[row, 0].set_title(f'Amplitude (axis {axis}, {method_label}={slice_idx[axis]})')
            axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
            axes[row, 1].imshow(phase, cmap=cmap, vmin=-np.pi, vmax=np.pi)
            phase_title = f'Phase (axis {axis}, {method_label}={slice_idx[axis]})'
            if masked:
                phase_title += ' [masked]'
            axes[row, 1].set_title(phase_title)
            axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])
            if sub_sup is not None:
                axes[row, 2].imshow(sub_sup, cmap='gray', vmin=0, vmax=1)
                axes[row, 2].set_title(f'Support (axis {axis}, {method_label}={slice_idx[axis]})')
                axes[row, 2].set_xticks([]); axes[row, 2].set_yticks([])
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=70)
        plt.close(fig)
        return buf.getvalue()
