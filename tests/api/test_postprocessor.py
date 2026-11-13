# tests/test_post_processor.py
import numpy as np
import pytest

from cohere_ui.api import post_processor as pp


class FakeStructuredGrid:
    def __init__(self):
        self.point_data = {}
        self.points = None
        self.dimensions = None
        self.saved_filename = None
        self.extracted_voi = None

    def extract_subset(self, voi):
        self.extracted_voi = voi
        return self

    def save(self, filename):
        self.saved_filename = filename


@pytest.fixture
def fake_grid(monkeypatch):
    grid = FakeStructuredGrid()
    monkeypatch.setattr(pp.pv, "StructuredGrid", lambda: grid)
    return grid


def test_dir_viz_initializes_grid_and_coordinates(fake_grid):
    viz = pp.Dir_viz((np.eye(3), np.eye(3)))
    data = np.zeros((2, 3, 4))

    viz.add_array("density", data)

    assert viz.shape == data.shape
    assert fake_grid.dimensions == (4, 3, 2)
    assert fake_grid.points.shape == (data.size, 3)


def test_add_array_accepts_flat_array_with_matching_element_count(fake_grid):
    viz = pp.Recip_viz((np.eye(3), np.eye(3)))
    viz.add_array("first", np.zeros((2, 3, 4)))

    flat = np.arange(24)
    viz.add_array("flat", flat)

    np.testing.assert_array_equal(viz.arrs["flat"], flat)


def test_add_array_rejects_shape_mismatch(fake_grid):
    viz = pp.Recip_viz((np.eye(3), np.eye(3)))
    viz.add_array("first", np.zeros((2, 3, 4)))

    with pytest.raises(ValueError, match="Shape mismatch"):
        viz.add_array("wrong", np.zeros((3, 3, 3)))


def test_get_structured_grid_splits_complex_array_into_amplitude_and_phase(
    fake_grid,
):
    viz = pp.Recip_viz((np.eye(3), np.eye(3)))
    data = np.array(
        [
            [[1 + 0j, 0 + 1j]],
            [[-1 + 0j, 0 - 1j]],
        ]
    )
    viz.add_array("image", data)

    result = viz.get_structured_grid(complex_mode="AmpPhase")

    assert result is fake_grid
    np.testing.assert_allclose(
        fake_grid.point_data["imageAmp"], np.abs(data).ravel()
    )
    np.testing.assert_allclose(
        fake_grid.point_data["imagePh"], np.angle(data).ravel()
    )


def test_get_structured_grid_supports_reim_mode(fake_grid):
    viz = pp.Recip_viz((np.eye(3), np.eye(3)))
    data = np.array([[[1 + 2j, 3 + 4j]]])
    viz.add_array("image", data)

    viz.get_structured_grid(complex_mode="ReIm")

    np.testing.assert_array_equal(
        fake_grid.point_data["imageRe"], data.real.ravel()
    )
    np.testing.assert_array_equal(
        fake_grid.point_data["imageImag"], data.imag.ravel()
    )


def test_get_structured_grid_can_select_arrays_and_apply_voi(fake_grid):
    viz = pp.Recip_viz((np.eye(3), np.eye(3)))
    viz.add_array("density", np.ones((2, 2, 2)))
    viz.add_array("support", np.zeros((2, 2, 2)))
    viz.voi = [0, 1, 0, 1, 0, 1]

    result = viz.get_structured_grid(arrstoinclude=["support"])

    assert result is fake_grid
    assert "support" in fake_grid.point_data
    assert "density" not in fake_grid.point_data
    assert fake_grid.extracted_voi == viz.voi


def test_get_structured_grid_raises_for_unknown_array(fake_grid):
    viz = pp.Recip_viz((np.eye(3), np.eye(3)))
    viz.add_array("density", np.ones((2, 2, 2)))

    with pytest.raises(ValueError, match="Array missing not found"):
        viz.get_structured_grid(arrstoinclude=["missing"])


def test_write_saves_structured_grid(fake_grid, tmp_path):
    viz = pp.Recip_viz((np.eye(3), np.eye(3)))
    viz.add_array("density", np.ones((2, 2, 2)))

    filename = tmp_path / "output.vts"
    viz.write(str(filename))

    assert fake_grid.saved_filename == str(filename)
    assert "density" in fake_grid.point_data


def test_get_com_calculates_absolute_value_weighted_center(fake_grid):
    viz = pp.Recip_viz((np.eye(3), np.eye(3)))
    data = np.zeros((2, 2, 2))
    data[0, 0, 0] = -1
    data[1, 1, 1] = 3
    viz.add_array("image", data)

    # Recip_viz with identity geometry has coordinates matching the index grid.
    np.testing.assert_allclose(viz.get_COM("image"), [0.75, 0.75, 0.75])


def test_get_com_raises_for_missing_array(fake_grid):
    viz = pp.Recip_viz((np.eye(3), np.eye(3)))

    with pytest.raises(ValueError, match="Array missing not found"):
        viz.get_COM("missing")


def test_pad_and_pad_infinite():
    assert list(pp.pad([1, 2], 5, 0)) == [1, 2, 0, 0, 0]
    assert list(pp.pad_infinite([1, 2], padding="x"))[:5] == [
        1, 2, "x", "x", "x"
    ]


def test_find_datarange_finds_centered_support_with_margin():
    data = np.zeros((9, 7, 5))
    data[4, 3, 2] = 10

    result = pp.find_datarange(
        data, plot_margin=1, amplitude_threshold=0.1
    )

    assert result == [3, 3, 3]


def test_find_datarange_keep_size_returns_original_shape():
    data = np.zeros((4, 5, 6))

    assert pp.find_datarange(data, keep_size=True) == [4, 5, 6]


def test_find_datarange_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="Empty array"):
        pp.find_datarange(np.array(1))

    with pytest.raises(TypeError, match="plot_margin"):
        pp.find_datarange(np.ones((3, 3)), plot_margin=(1, 1))

    with pytest.raises(ValueError, match="lenght"):
        pp.find_datarange(np.ones((3, 3)), plot_margin=[1])

    with pytest.raises(ValueError, match="No voxel above"):
        pp.find_datarange(
            np.zeros((3, 3, 3)), amplitude_threshold=0.1
        )


def test_interpolate_builds_image_data_and_samples_grid(monkeypatch):
    class FakeImageData:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def sample(self, grid):
            return ("sampled", grid, self.kwargs)

    monkeypatch.setattr(pp.pv, "ImageData", FakeImageData)

    grid = type(
        "Grid",
        (),
        {
            "dimensions": (5, 5, 5),
            "bounds": (0, 4, 0, 4, 0, 4),
        },
    )()

    result = pp.interpolate(grid, 2)

    assert result[0] == "sampled"
    assert result[1] is grid
    assert result[2] == {
        "dimensions": [2, 2, 2],
        "origin": [0, 0, 0],
        "spacing": [2, 2, 2],
    }


def test_interpolate_rejects_resolution_that_would_create_huge_grid():
    grid = type(
        "Grid",
        (),
        {
            "dimensions": (5, 5, 5),
            "bounds": (0, 4, 0, 4, 0, 4),
        },
    )()

    with pytest.raises(ValueError, match="does not match the grid scale"):
        pp.interpolate(grid, 0.001)


def test_get_centered_voisize():
    assert pp.get_centered_voisize([10, 8, 6], [4, 2, 6]) == [
        3, 7, 3, 5, 0, 6
    ]


def test_extent_to_slice():
    result = pp.extent_to_slice([1, 4, 2, 6, 0, 3])

    assert result == (slice(1, 4), slice(2, 6), slice(0, 3))