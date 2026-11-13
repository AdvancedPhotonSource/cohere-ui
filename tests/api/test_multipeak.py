# tests/test_multipeak.py
import os
from unittest.mock import Mock

import numpy as np
import pytest

from cohere_ui.api import multipeak  # Adjust to your project's import path


def test_calc_geometry_swaps_rows_and_applies_twin_matrix():
    reciprocal = np.array(
        [
            [1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0],
            [5.0, 0.0, 6.0],
        ]
    )
    twin = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    instr = Mock()
    instr.get_geometry.return_value = (reciprocal, None)

    result, voxel_size = multipeak.calc_geometry(
        instr, [1, 2, 3], 4, {}, twin
    )

    row_swapped = reciprocal[[1, 0, 2], :]
    np.testing.assert_allclose(result, twin @ row_swapped)
    expected_size = max(np.linalg.norm(row_swapped[:, i]) for i in range(3))
    assert voxel_size == pytest.approx(expected_size)
    instr.get_geometry.assert_called_once_with( [1, 2, 3], 4, {}, xtal=True)


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4, 4),
        (3, 5, 4),
        (2, 3, 6),
    ],
)
def test_pad_to_cube_returns_cube_and_preserves_input(shape):
    arr = np.ones(shape)

    result = multipeak.pad_to_cube(arr)

    assert result.ndim == 3
    assert len(set(result.shape)) == 1
    assert np.count_nonzero(result) == arr.size


def test_center_mp_moves_density_peak_to_center_and_zeros_reference_displacements():
    image = np.zeros((10, 7, 7, 7))
    image[0, 1, 2, 3] = 1
    image[1] = 2
    image[2] = 3
    image[3] = 4
    support = np.ones((7, 7, 7))

    centered_image, centered_support = multipeak.center_mp(image, support)

    center = (3, 3, 3)
    assert np.unravel_index(np.argmax(centered_image[0]), centered_image[0].shape) == center
    assert centered_image[0][center] == 1
    assert centered_support.shape == support.shape
    np.testing.assert_allclose(centered_image[1:4], 0)


def test_twin_matrix_returns_a_rotation_matrix():
    hklin = np.array([1.0, 0.0, 0.0])
    hklout = np.array([0.0, 1.0, 0.0])
    twin_plane = np.array([1.0, 0.0, 0.0])
    sample_axis = np.array([0.0, 1.0, 0.0])

    matrix = multipeak.twin_matrix(hklin, hklout, twin_plane, sample_axis)

    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_write_vti_saves_named_arrays_and_uses_twin_filename(
    tmp_path, monkeypatch
):
    class FakeImageData:
        def __init__(self):
            self.point_data = {}
            self.saved_path = None

        def save(self, path):
            self.saved_path = path

    fake_grid = FakeImageData()
    monkeypatch.setattr(multipeak.pv, "ImageData", lambda: fake_grid)

    data = [np.full((2, 3, 4), i) for i in range(11)]
    conf_maps = {"config_mp": {"ds_voxel_size": 0.25}}

    multipeak.write_vti(data, conf_maps, tmp_path, is_twin=True)

    assert fake_grid.dimensions == (2, 3, 4)
    assert fake_grid.spacing == (0.25, 0.25, 0.25)
    assert fake_grid.origin == (0.0, 0.0, 0.0)
    assert os.path.basename(fake_grid.saved_path) == "twin_full_data.vti"

    expected_names = [
        "density", "u_x", "u_y", "u_z", "s_xx", "s_yy",
        "s_zz", "s_xy", "s_yz", "s_zx", "support",
    ]
    assert list(fake_grid.point_data) == expected_names
    np.testing.assert_array_equal(
        fake_grid.point_data["density"], data[0].ravel(order="F")
    )


def test_reconstruction_returns_minus_one_for_ai_guess(monkeypatch):
    get_one_dev = Mock(return_value=0)
    monkeypatch.setattr(multipeak.balancer, "get_one_dev", get_one_dev)
    create_rec = Mock()
    monkeypatch.setattr(multipeak.calc, "create_rec", create_rec)

    result = multipeak.reconstruction(
        "np", {"init_guess": "AI_guess"}, ["/data/peak"], 0
    )

    assert result == -1
    get_one_dev.assert_not_called()
    create_rec.assert_not_called()


def test_reconstruction_saves_successful_worker_result(monkeypatch, tmp_path):
    worker = Mock()
    worker.iterate.return_value = 1
    monkeypatch.setattr(multipeak.balancer, "get_one_dev", lambda _: 2)
    create_rec = Mock(return_value=worker)
    monkeypatch.setattr(multipeak.calc, "create_rec", create_rec)

    pars = {"device": [2], "save_dir": str(tmp_path)}
    result = multipeak.reconstruction(
        "np", pars, ["/data/peak"], 0, debug=True
    )

    assert result is None
    assert pars["init_guess"] == "random"
    create_rec.assert_called_once_with(
        pars, ["/data/peak"], "np", 2, debug=True, rec_type="mp"
    )
    worker.save_res.assert_called_once_with(str(tmp_path))