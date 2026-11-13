# tests/test_view_reconstruction.py
import importlib
import sys
import types
from unittest.mock import Mock

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest


# Mayavi is only needed for rendering; provide a stub for unit tests if absent.
try:
    import mayavi  # noqa: F401
except ModuleNotFoundError:
    mayavi_stub = types.ModuleType("mayavi")
    mayavi_stub.mlab = Mock()
    sys.modules["mayavi"] = mayavi_stub

from cohere_ui.api import view_reconstruction as view  # Adjust if needed


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    yield
    plt.close("all")


def make_test_arrays(size=6):
    """Return nonzero 3D arrays suitable for xsection's quantile calculations."""
    x, y, z = np.mgrid[:size, :size, :size]
    rho = 1.0 + x + y + z
    ux = 0.1 + x * 0.2 - y * 0.1 + z * 0.05
    uy = 0.2 - x * 0.05 + y * 0.15 + z * 0.1
    uz = 0.3 + x * 0.1 - y * 0.05 + z * 0.2
    return rho.astype(float), ux.astype(float), uy.astype(float), uz.astype(float)


def test_xsection_plots_center_slices():
    rho, ux, uy, uz = make_test_arrays()

    view.xsection(rho, ux, uy, uz)

    # Four image axes are created; the colorbars add four more axes.
    fig = plt.gcf()
    image_axes = [ax for ax in fig.axes if ax.images]
    assert len(image_axes) == 4

    center = rho.shape[0] // 2
    expected_density_slice = rho[center] / np.max(rho[center])
    np.testing.assert_allclose(
        image_axes[0].images[0].get_array(),
        expected_density_slice,
    )

    for ax in image_axes:
        assert ax.images[0].get_array().shape == rho[center].shape


def test_xsection_moves_axis_for_nonzero_axis_argument():
    rho, ux, uy, uz = make_test_arrays()
    axis = 1

    view.xsection(rho, ux, uy, uz, axis=axis)

    image_axes = [ax for ax in plt.gcf().axes if ax.images]
    center = rho.shape[0] // 2
    moved_density = np.moveaxis(rho, 0, -1)[center]
    expected_density_slice = moved_density / np.max(moved_density)

    np.testing.assert_allclose(
        image_axes[0].images[0].get_array(),
        expected_density_slice,
    )


def test_show_reconstruction_loads_image_and_calls_rendering_functions(
    monkeypatch,
):
    rho, ux, uy, uz = make_test_arrays()
    reconstruction = np.stack([rho, ux, uy, uz])

    load = Mock(return_value=reconstruction)
    monkeypatch.setattr(view.np, "load", load)

    figure = Mock()
    contour3d = Mock()
    show_3d = Mock()
    monkeypatch.setattr(view.mlab, "figure", figure)
    monkeypatch.setattr(view.mlab, "contour3d", contour3d)
    monkeypatch.setattr(view.mlab, "show", show_3d)

    xsection = Mock()
    monkeypatch.setattr(view, "xsection", xsection)
    show_2d = Mock()
    monkeypatch.setattr(view.plt, "show", show_2d)

    view.show_reconstruction("/experiment", contour=0.4)

    load.assert_called_once_with(
        "/experiment/results_phasing/image_nostrain.npy"
    )
    figure.assert_called_once_with()
    contour3d.assert_called_once_with(
        rho,
        color=(1.0, 0.98, 0.94),
        contours=[0.4 * np.max(rho)],
    )
    show_3d.assert_called_once_with()

    assert xsection.call_count == 3
    for axis, call in enumerate(xsection.call_args_list):
        assert call.args[:4] == (rho, ux, uy, uz)
        assert call.args[4] == axis
        assert call.kwargs == {"contour": 0.4}

    show_2d.assert_called_once_with()
