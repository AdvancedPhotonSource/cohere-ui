"""Reconstruction features: GA, shrink_wrap, pcdi, twin, etc."""

import math

import ipywidgets as widgets

from cohere_ui.jupyter_gui.features.base import Feature
from cohere_ui.jupyter_gui.text import load_text
from cohere_ui.jupyter_gui.widgets import form_row, text_field, checkbox

_TEXT = load_text('features')


class GAFeature(Feature):
    """Genetic Algorithm feature."""

    name = _TEXT['GA']['name']
    description = _TEXT['GA']['description']

    def fill_active(self) -> list:
        self.ga_fast = checkbox('fast processing (size limited)')
        self.generations = text_field(placeholder='e.g., 3')
        self.metrics = text_field(placeholder='["chi"]')
        self.breed_modes = text_field(placeholder='["sqrt_ab"]')
        self.cullings = text_field(placeholder='e.g., [2]')
        self.sw_thresholds = text_field(placeholder='[0.1]')
        self.sw_gauss_sigmas = text_field(placeholder='[1.0]')
        self.lpf_sigmas = text_field(placeholder='')
        self.gen_pc_start = text_field(placeholder='3')

        return [
            self.ga_fast,
            form_row('Generations', self.generations),
            form_row('Fitness Metrics', self.metrics),
            form_row('Breed Modes', self.breed_modes),
            form_row('Cullings', self.cullings),
            form_row('SW Thresholds', self.sw_thresholds),
            form_row('SW Gauss Sigmas', self.sw_gauss_sigmas),
            form_row('LPF Sigmas', self.lpf_sigmas),
            form_row('Gen PC Start', self.gen_pc_start),
        ]

    def set_defaults(self):
        self.generations.value = '3'
        self.metrics.value = '["chi"]'
        self.breed_modes.value = '["sqrt_ab"]'
        self.sw_thresholds.value = '[.1]'
        self.sw_gauss_sigmas.value = '[1.0]'
        self.gen_pc_start.value = '3'
        self.ga_fast.value = False

    def init_config(self, conf_map: dict):
        if 'ga_generations' in conf_map:
            self.active.value = True
            self.generations.value = str(conf_map['ga_generations'])
            if 'ga_fast' in conf_map:
                self.ga_fast.value = conf_map['ga_fast']
            if 'ga_metrics' in conf_map:
                self.metrics.value = self.format_value(conf_map['ga_metrics'])
            if 'ga_breed_modes' in conf_map:
                self.breed_modes.value = self.format_value(conf_map['ga_breed_modes'])
            if 'ga_cullings' in conf_map:
                self.cullings.value = self.format_value(conf_map['ga_cullings'])
            if 'ga_sw_thresholds' in conf_map:
                self.sw_thresholds.value = self.format_value(conf_map['ga_sw_thresholds'])
            if 'ga_sw_gauss_sigmas' in conf_map:
                self.sw_gauss_sigmas.value = self.format_value(conf_map['ga_sw_gauss_sigmas'])
            if 'ga_lpf_sigmas' in conf_map:
                self.lpf_sigmas.value = self.format_value(conf_map['ga_lpf_sigmas'])
            if 'ga_gen_pc_start' in conf_map:
                self.gen_pc_start.value = self.format_value(conf_map['ga_gen_pc_start'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.ga_fast.value:
            conf_map['ga_fast'] = True
        if self.generations.value:
            conf_map['ga_generations'] = self.parse_value(self.generations.value)
        if self.metrics.value:
            conf_map['ga_metrics'] = self.parse_value(self.metrics.value)
        if self.breed_modes.value:
            conf_map['ga_breed_modes'] = self.parse_value(self.breed_modes.value)
        if self.cullings.value:
            conf_map['ga_cullings'] = self.parse_value(self.cullings.value)
        if self.sw_thresholds.value:
            conf_map['ga_sw_thresholds'] = self.parse_value(self.sw_thresholds.value)
        if self.sw_gauss_sigmas.value:
            conf_map['ga_sw_gauss_sigmas'] = self.parse_value(self.sw_gauss_sigmas.value)
        if self.lpf_sigmas.value:
            conf_map['ga_lpf_sigmas'] = self.parse_value(self.lpf_sigmas.value)
        if self.gen_pc_start.value:
            conf_map['ga_gen_pc_start'] = self.parse_value(self.gen_pc_start.value)

    def verify_active(self) -> str:
        return self._require_field('generations')


class LowResolutionFeature(Feature):
    """Low pass filter / low resolution feature."""

    name = _TEXT['low_resolution']['name']
    description = _TEXT['low_resolution']['description']

    def fill_active(self) -> list:
        self.trigger = text_field(placeholder='[iter_start, iter_step]')
        self.range = text_field(placeholder='[sigma_start, sigma_end]')

        return [
            form_row('Trigger', self.trigger),
            form_row('Range', self.range),
        ]

    def set_defaults(self):
        self.trigger.value = '[0, 1]'
        self.range.value = '[2.0, 0.5]'

    def init_config(self, conf_map: dict):
        if 'lowpass_filter_trigger' in conf_map:
            self.active.value = True
            self.trigger.value = self.format_value(conf_map['lowpass_filter_trigger'])
            if 'lowpass_filter_range' in conf_map:
                self.range.value = self.format_value(conf_map['lowpass_filter_range'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['lowpass_filter_trigger'] = self.parse_value(self.trigger.value)
        if self.range.value:
            conf_map['lowpass_filter_range'] = self.parse_value(self.range.value)

    def verify_active(self) -> str:
        return self._require_field('trigger')


class ShrinkWrapFeature(Feature):
    """Shrink wrap support feature."""

    name = _TEXT['shrink_wrap']['name']
    description = _TEXT['shrink_wrap']['description']

    def fill_active(self) -> list:
        self.trigger = text_field(placeholder='[iter_start, iter_step]')
        self.sw_type = text_field(placeholder='GAUSS')
        self.threshold = text_field(placeholder='0.1')
        self.gauss_sigma = text_field(placeholder='1.0')

        return [
            form_row('Trigger', self.trigger),
            form_row('Type', self.sw_type),
            form_row('Threshold', self.threshold),
            form_row('Gauss Sigma', self.gauss_sigma),
        ]

    def set_defaults(self):
        self.trigger.value = '[1, 1]'
        self.sw_type.value = 'GAUSS'
        self.threshold.value = '0.1'
        self.gauss_sigma.value = '1.0'

    def init_config(self, conf_map: dict):
        if 'shrink_wrap_trigger' in conf_map:
            self.active.value = True
            self.trigger.value = self.format_value(conf_map['shrink_wrap_trigger'])
            if 'shrink_wrap_type' in conf_map:
                self.sw_type.value = conf_map['shrink_wrap_type']
            if 'shrink_wrap_threshold' in conf_map:
                self.threshold.value = str(conf_map['shrink_wrap_threshold'])
            if 'shrink_wrap_gauss_sigma' in conf_map:
                self.gauss_sigma.value = str(conf_map['shrink_wrap_gauss_sigma'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['shrink_wrap_trigger'] = self.parse_value(self.trigger.value)
        if self.sw_type.value:
            conf_map['shrink_wrap_type'] = self.sw_type.value
        if self.threshold.value:
            conf_map['shrink_wrap_threshold'] = self.parse_value(self.threshold.value)
        if self.gauss_sigma.value:
            conf_map['shrink_wrap_gauss_sigma'] = self.parse_value(self.gauss_sigma.value)

    def verify_active(self) -> str:
        return self._require_field('trigger')


class PhaseConstrainFeature(Feature):
    """Phase constrain feature."""

    name = _TEXT['phase_constrain']['name']
    description = _TEXT['phase_constrain']['description']

    def fill_active(self) -> list:
        # phc_trigger MUST be 3-element [start, step, stop]; op_flow.py:308
        # does params['phc_trigger'][2] unconditionally (IndexError otherwise).
        self.trigger = text_field(placeholder='[start, step, stop]')
        self.phase_min = text_field(placeholder='-1.57')
        self.phase_max = text_field(placeholder='1.57')
        self.pi_prefactor = checkbox(description='bounds in units of π')
        self._phase_min_mul = widgets.HTML(
            value='<small style="color:#666;">× π</small>',
            layout=widgets.Layout(display='none', margin='0 0 0 6px'),
        )
        self._phase_max_mul = widgets.HTML(
            value='<small style="color:#666;">× π</small>',
            layout=widgets.Layout(display='none', margin='0 0 0 6px'),
        )
        self.pi_prefactor.observe(self._on_pi_toggle, 'value')
        return [
            form_row('Trigger', self.trigger),
            self.pi_prefactor,
            form_row('Phase Min', widgets.HBox([self.phase_min, self._phase_min_mul])),
            form_row('Phase Max', widgets.HBox([self.phase_max, self._phase_max_mul])),
        ]

    def set_defaults(self):
        # Reset checkbox first so the observer (if it fires) operates on
        # the OLD field values; the immediate writes below overwrite anyway.
        self.pi_prefactor.value = False
        # Default trigger MUST be 3-element; see fill_active() note.
        self.trigger.value = '[0, 1, -1]'
        self.phase_min.value = '-1.57'
        self.phase_max.value = '1.57'

    def _on_pi_toggle(self, change):
        """Convert Phase Min/Max between radians and π-prefactors when the
        checkbox flips, and show/hide the '× π' adornment to match."""
        if change['new']:
            op = lambda v: v / math.pi
            display = ''
        else:
            op = lambda v: v * math.pi
            display = 'none'
        self._convert_phase_field(self.phase_min, op)
        self._convert_phase_field(self.phase_max, op)
        self._phase_min_mul.layout.display = display
        self._phase_max_mul.layout.display = display

    @staticmethod
    def _convert_phase_field(field, op):
        """Apply ``op`` to the field value, supporting single floats and lists."""
        if not field.value.strip():
            return
        parsed = PhaseConstrainFeature.parse_value(field.value)
        if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
            field.value = str(round(op(parsed), 4))
        elif isinstance(parsed, list):
            converted = [
                round(op(x), 4) if isinstance(x, (int, float)) and not isinstance(x, bool) else x
                for x in parsed
            ]
            field.value = str(converted).replace(' ', '')
        # Anything else: leave the field alone.

    def init_config(self, conf_map: dict):
        if 'phc_trigger' in conf_map:
            self.active.value = True
            # Saved config is always in radians; reset checkbox before writing
            # field values so the prefactor-conversion observer doesn't munge them.
            self.pi_prefactor.value = False
            self.trigger.value = self.format_value(conf_map['phc_trigger'])
            if 'phc_phase_min' in conf_map:
                self.phase_min.value = str(conf_map['phc_phase_min'])
            if 'phc_phase_max' in conf_map:
                self.phase_max.value = str(conf_map['phc_phase_max'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['phc_trigger'] = self.parse_value(self.trigger.value)
        if self.phase_min.value:
            conf_map['phc_phase_min'] = self._maybe_to_radians(
                self.parse_value(self.phase_min.value)
            )
        if self.phase_max.value:
            conf_map['phc_phase_max'] = self._maybe_to_radians(
                self.parse_value(self.phase_max.value)
            )

    def _maybe_to_radians(self, value):
        """Multiply by π when the checkbox is on; element-wise for lists."""
        if not self.pi_prefactor.value:
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value * math.pi
        if isinstance(value, list):
            return [
                x * math.pi if isinstance(x, (int, float)) and not isinstance(x, bool) else x
                for x in value
            ]
        return value

    def verify_active(self) -> str:
        return self._require_field('trigger')


class PCDIFeature(Feature):
    """Partial coherence (PCDI) feature."""

    name = _TEXT['pcdi']['name']
    description = _TEXT['pcdi']['description']

    def fill_active(self) -> list:
        self.trigger = text_field(placeholder='[iter_start, iter_step]')
        self.pc_type = text_field(placeholder='LUCY')
        self.lucy_iterations = text_field(placeholder='20')
        self.normalize = checkbox('normalize')
        self.lucy_kernel = text_field(placeholder='[16, 16, 16]')

        return [
            form_row('Trigger', self.trigger),
            form_row('Type', self.pc_type),
            form_row('LUCY Iterations', self.lucy_iterations),
            self.normalize,
            form_row('LUCY Kernel', self.lucy_kernel),
        ]

    def set_defaults(self):
        self.trigger.value = '[50, 50]'
        self.pc_type.value = 'LUCY'
        self.lucy_iterations.value = '20'
        self.normalize.value = True
        self.lucy_kernel.value = '[16, 16, 16]'

    def init_config(self, conf_map: dict):
        if 'pc_interval' in conf_map:
            self.active.value = True
            self.trigger.value = self.format_value(conf_map['pc_interval'])
            if 'pc_type' in conf_map:
                self.pc_type.value = conf_map['pc_type']
            if 'pc_LUCY_iterations' in conf_map:
                self.lucy_iterations.value = str(conf_map['pc_LUCY_iterations'])
            if 'pc_normalize' in conf_map:
                self.normalize.value = conf_map['pc_normalize']
            if 'pc_LUCY_kernel' in conf_map:
                self.lucy_kernel.value = self.format_value(conf_map['pc_LUCY_kernel'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['pc_interval'] = self.parse_value(self.trigger.value)
        if self.pc_type.value:
            conf_map['pc_type'] = self.pc_type.value
        if self.lucy_iterations.value:
            conf_map['pc_LUCY_iterations'] = self.parse_value(self.lucy_iterations.value)
        if self.normalize.value:
            conf_map['pc_normalize'] = True
        if self.lucy_kernel.value:
            conf_map['pc_LUCY_kernel'] = self.parse_value(self.lucy_kernel.value)

    def verify_active(self) -> str:
        return self._require_field('trigger')


class TwinFeature(Feature):
    """Twin removal feature."""

    name = _TEXT['twin']['name']
    description = _TEXT['twin']['description']

    def fill_active(self) -> list:
        self.trigger = text_field(placeholder='[2]')
        self.halves = text_field(placeholder='[0, 0]')

        return [
            form_row('Trigger', self.trigger),
            form_row('Halves', self.halves),
        ]

    def set_defaults(self):
        self.trigger.value = '[2]'
        self.halves.value = '[0, 0]'

    def init_config(self, conf_map: dict):
        if 'twin_trigger' in conf_map:
            self.active.value = True
            self.trigger.value = self.format_value(conf_map['twin_trigger'])
            if 'twin_halves' in conf_map:
                self.halves.value = self.format_value(conf_map['twin_halves'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['twin_trigger'] = self.parse_value(self.trigger.value)
        if self.halves.value:
            conf_map['twin_halves'] = self.parse_value(self.halves.value)

    def verify_active(self) -> str:
        return self._require_field('trigger')


class AverageFeature(Feature):
    """Amplitude averaging feature."""

    name = _TEXT['average']['name']
    description = _TEXT['average']['description']

    def fill_active(self) -> list:
        self.trigger = text_field(placeholder='[-1, 1]')

        return [
            form_row('Trigger', self.trigger),
        ]

    def set_defaults(self):
        self.trigger.value = '[-1, 1]'

    def init_config(self, conf_map: dict):
        if 'average_trigger' in conf_map:
            self.active.value = True
            self.trigger.value = self.format_value(conf_map['average_trigger'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['average_trigger'] = self.parse_value(self.trigger.value)

    def verify_active(self) -> str:
        return self._require_field('trigger')


class ProgressFeature(Feature):
    """Progress reporting feature."""

    name = _TEXT['progress']['name']
    description = _TEXT['progress']['description']

    def fill_active(self) -> list:
        self.trigger = text_field(placeholder='[0, 20]')
        self.show_progress_bar = widgets.Checkbox(
            value=True, description='Show progress bar (with rate-based interpolation)',
            indent=False,
        )
        return [
            form_row('Trigger', self.trigger),
            form_row('', self.show_progress_bar),
        ]

    def set_defaults(self):
        self.trigger.value = '[0, 20]'
        self.show_progress_bar.value = True

    def init_config(self, conf_map: dict):
        if 'progress_trigger' in conf_map:
            self.active.value = True
            self.trigger.value = self.format_value(conf_map['progress_trigger'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['progress_trigger'] = self.parse_value(self.trigger.value)


class LiveFeature(Feature):
    """Live in-notebook view driven by cohere's ``live_trigger`` mechanism.

    Writes ``live_trigger`` into ``config_rec``. Renderer / slicing options
    are GUI-only (passed to the wrapper subprocess as a BackendConfig) and
    don't appear in ``config_rec``; cohere itself doesn't need to know which
    backend the GUI registered.
    """

    name = _TEXT['live']['name']
    description = _TEXT['live']['description']

    def fill_active(self) -> list:
        self.trigger = text_field(placeholder='[0, 20]')

        self.renderer = widgets.Dropdown(
            options=[
                ('matplotlib (2D slice)', 'matplotlib_2d'),
                ('matplotlib (3D mosaic)', 'matplotlib_3d_mosaic'),
                ('PyVista (3D static)', 'pyvista_static'),
            ],
            value='matplotlib_2d',
            layout=widgets.Layout(width='260px'),
        )

        self.slice_axis = widgets.Dropdown(
            options=[('z', 2), ('y', 1), ('x', 0)],
            value=2,
            layout=widgets.Layout(width='80px'),
        )

        self.slice_method = widgets.RadioButtons(
            options=[
                ('Center of mass (tracks the object)', 'center_of_mass'),
                ('Center of array (fixed mid-index)', 'center_of_array'),
            ],
            value='center_of_mass',
        )

        self.stride = text_field(placeholder='4')

        self.iso_level = text_field(placeholder='0.3')

        self.phase_cmap = widgets.Dropdown(
            options=[
                ('twilight (cyclic)', 'twilight'),
                ('twilight_shifted (cyclic)', 'twilight_shifted'),
                ('hsv (cyclic)', 'hsv'),
                ('viridis (non-cyclic)', 'viridis'),
                ('plasma (non-cyclic)', 'plasma'),
            ],
            value='twilight',
            layout=widgets.Layout(width='220px'),
        )

        self.apply_support_mask = widgets.Checkbox(
            value=True, description='Apply support mask to phase',
            indent=False,
        )

        # Toggle visibility of the per-renderer options as the user picks one.
        self._slice_row = form_row('Slice axis', self.slice_axis)
        self._method_row = form_row('Slice method', self.slice_method)
        self._stride_row = form_row('Stride', self.stride)
        self._iso_row = form_row('Iso level', self.iso_level)
        self._cmap_row = form_row('Phase cmap', self.phase_cmap)
        self._mask_row = form_row('', self.apply_support_mask)

        self.renderer.observe(self._on_renderer_change, names='value')
        self._on_renderer_change({'new': self.renderer.value})

        # Renderer-overhead benchmark. Synthetic 256^3 array, runs every
        # registered backend N times, reports mean ms/call so the user can
        # pick a renderer with the cost in plain sight.
        self.benchmark_btn = widgets.Button(
            description='Benchmark renderers',
            tooltip='Times each backend on a synthetic 256^3 reconstruction',
            layout=widgets.Layout(width='220px'),
        )
        self.benchmark_size = widgets.Dropdown(
            options=[('128^3', 128), ('192^3', 192), ('256^3 (default)', 256), ('384^3', 384)],
            value=256,
            layout=widgets.Layout(width='180px'),
        )
        self.benchmark_iters = widgets.Dropdown(
            options=[('3', 3), ('5 (default)', 5), ('10', 10)],
            value=5,
            layout=widgets.Layout(width='100px'),
        )
        self.benchmark_output = widgets.HTML(value='')
        self.benchmark_btn.on_click(lambda _b: self._run_benchmark())

        return [
            form_row('Trigger', self.trigger),
            form_row('Renderer', self.renderer),
            self._slice_row,
            self._method_row,
            self._stride_row,
            self._iso_row,
            self._cmap_row,
            self._mask_row,
            widgets.HTML(
                f'<small style="color:#888;">{_TEXT["live"]["renderer_note"]}</small>'
            ),
            widgets.HTML('<hr style="margin:8px 0;">'),
            widgets.HBox([
                self.benchmark_btn,
                form_row('Volume', self.benchmark_size),
                form_row('Iters', self.benchmark_iters),
            ]),
            self.benchmark_output,
        ]

    def _on_renderer_change(self, change):
        value = change['new']
        is_2d = (value == 'matplotlib_2d')
        is_mosaic = (value == 'matplotlib_3d_mosaic')
        is_pyvista = (value == 'pyvista_static')
        self._slice_row.layout.display = '' if is_2d else 'none'
        self._method_row.layout.display = '' if (is_2d or is_mosaic) else 'none'
        self._stride_row.layout.display = '' if is_pyvista else 'none'
        self._iso_row.layout.display = '' if is_pyvista else 'none'
        self._cmap_row.layout.display = '' if (is_2d or is_mosaic) else 'none'
        self._mask_row.layout.display = '' if (is_2d or is_mosaic) else 'none'

    def _run_benchmark(self):
        import time
        import numpy as np
        from cohere_ui.jupyter_gui._backends import JupyterMatplotlibBackend
        try:
            from cohere_ui.jupyter_gui._backends import PyVistaBackend
        except Exception:
            PyVistaBackend = None

        n = int(self.benchmark_size.value)
        iters = int(self.benchmark_iters.value)
        self.benchmark_btn.disabled = True
        self.benchmark_output.value = (
            f'<i>Benchmarking on {n}^3 volume, {iters} iters per renderer...</i>'
        )

        rng = np.random.default_rng(0)
        ds = (rng.random((n, n, n), dtype=np.float32) +
              1j * rng.random((n, n, n), dtype=np.float32))
        sup = (rng.random((n, n, n), dtype=np.float32) > 0.5)
        errs = list(rng.random(20, dtype=np.float32) + 0.01)
        title = f'Iteration: 1/{iters}\nError: 0.5'

        class _Stub:
            def put(self, rec, **kw):
                pass

        configs = [
            ('matplotlib (2D slice, CoM)', JupyterMatplotlibBackend, dict(
                msg_queue=_Stub(), mode='center_slice', slice_axis=2,
                slice_method='center_of_mass',
            )),
            ('matplotlib (3D mosaic, CoM)', JupyterMatplotlibBackend, dict(
                msg_queue=_Stub(), mode='strided_3d',
                slice_method='center_of_mass',
            )),
        ]
        if PyVistaBackend is not None:
            configs.append((
                'PyVista (3D static)', PyVistaBackend, dict(
                    msg_queue=_Stub(), stride=4, iso_level=0.3,
                ),
            ))

        rows = []
        for name, cls, kwargs in configs:
            try:
                backend = cls(**kwargs)
            except Exception as e:
                rows.append((name, 'init failed', str(e)))
                continue
            # Warmup (matplotlib import / pyvista init paid once)
            try:
                backend.update_singlepeak(ds, errs, sup, title)
            except Exception as e:
                rows.append((name, 'warmup failed', str(e)))
                continue
            t0 = time.perf_counter()
            ok = 0
            for _ in range(iters):
                try:
                    backend.update_singlepeak(ds, errs, sup, title)
                    ok += 1
                except Exception as e:
                    rows.append((name, f'failed iter {ok}', str(e)))
                    break
            else:
                elapsed = time.perf_counter() - t0
                ms = (elapsed / iters) * 1000.0
                rows.append((name, f'{ms:.1f} ms/call', f'{1.0/ms*1000:.1f} fps' if ms > 0 else ''))

        html_rows = ''.join(
            f'<tr><td style="padding:2px 12px 2px 0;">{name}</td>'
            f'<td style="padding:2px 12px 2px 0;font-family:monospace;">{cost}</td>'
            f'<td style="padding:2px;color:#888;font-family:monospace;">{note}</td></tr>'
            for name, cost, note in rows
        )
        self.benchmark_output.value = (
            f'<table style="border-collapse:collapse;margin-top:6px;">'
            f'<thead><tr style="border-bottom:1px solid #ccc;">'
            f'<th style="text-align:left;padding:2px 12px 2px 0;">Renderer</th>'
            f'<th style="text-align:left;padding:2px 12px 2px 0;">Cost</th>'
            f'<th style="text-align:left;padding:2px;">Notes</th></tr></thead>'
            f'<tbody>{html_rows}</tbody></table>'
            f'<small style="color:#888;">Synthetic {n}^3 complex array, '
            f'{iters} update_singlepeak calls per renderer (warmup discarded). '
            f'Multiply by your reconstruction\'s live_trigger fire count to '
            f'estimate total snapshot overhead.</small>'
        )
        self.benchmark_btn.disabled = False

    def set_defaults(self):
        self.trigger.value = '[0, 20]'
        self.renderer.value = 'matplotlib_2d'
        self.slice_axis.value = 2
        self.slice_method.value = 'center_of_mass'
        self.stride.value = '4'
        self.iso_level.value = '0.3'
        self.phase_cmap.value = 'twilight'
        self.apply_support_mask.value = True

    def init_config(self, conf_map: dict):
        if 'live_trigger' in conf_map:
            self.active.value = True
            self.trigger.value = self.format_value(conf_map['live_trigger'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['live_trigger'] = self.parse_value(self.trigger.value)
        if 'progress_trigger' not in conf_map:
            conf_map['progress_trigger'] = [0, 1]
