import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import cohere_ui.beamline_postprocess as bp


@pytest.fixture
def processing_case(tmp_path, monkeypatch):
    """Create reconstruction files and mock instrument/visualization services."""
    experiment_dir = tmp_path / "experiment"
    results_dir = experiment_dir / "results_phasing"
    results_dir.mkdir(parents=True)

    image = (
        np.arange(8).reshape(2, 2, 2)
        + 1j * np.ones((2, 2, 2))
    )
    support = np.ones(image.shape)

    np.save(results_dir / "image.npy", image)
    np.save(results_dir / "support.npy", support)

    config_maps = {
        "config": {"beamline": "test_beamline", "scan": "1-7"},
        "config_disp": {},
        "config_instr": {},
    }

    geometry = (
        np.eye(3),
        np.eye(3),
        np.array([1.0, 2.0, 3.0]),  # Q
        np.array([0.0, 0.0, 1.0]),  # Ki
        np.array([1.0, 2.0, 4.0]),  # Kf
    )

    instrument = Mock()
    instrument.get_geometry.return_value = geometry

    instrument_module = Mock()
    instrument_module.create_instr.return_value = instrument

    import_module = Mock(return_value=instrument_module)
    monkeypatch.setattr(bp.importlib, "import_module", import_module)

    direct_viz = Mock()
    direct_viz.get_COM.return_value = (0.5, 0.5, 0.5)

    vectors_viz = Mock()
    make_image_viz = Mock(return_value=direct_viz)
    make_vectors_viz = Mock(return_value=vectors_viz)

    monkeypatch.setattr(bp.pu, "make_image_viz", make_image_viz)
    monkeypatch.setattr(
        bp.pu, "make_geometry_vectors_viz", make_vectors_viz
    )

    return SimpleNamespace(
        experiment_dir=str(experiment_dir),
        results_dir=str(results_dir),
        save_dir=str(experiment_dir / "results_viz"),
        image=image,
        support=support,
        config_maps=config_maps,
        geometry=geometry,
        instrument=instrument,
        instrument_module=instrument_module,
        import_module=import_module,
        direct_viz=direct_viz,
        vectors_viz=vectors_viz,
        make_image_viz=make_image_viz,
        make_vectors_viz=make_vectors_viz,
    )


def run_processing(case):
    return bp.process_dir(
        case.experiment_dir,
        case.config_maps,
        (7, case.results_dir),
    )


def test_process_dir_default_outputs(processing_case):
    case = processing_case

    result = run_processing(case)

    assert bp.os.path.isdir(case.save_dir)
    case.import_module.assert_called_once_with(
        "cohere_beamlines.test_beamline.instrument"
    )
    case.instrument_module.create_instr.assert_called_once_with(
        case.config_maps
    )
    case.instrument.get_geometry.assert_called_once_with(
        None, 7, case.config_maps
    )

    args = case.make_image_viz.call_args.args
    assert args[0] is case.geometry
    np.testing.assert_array_equal(args[1], case.image)
    np.testing.assert_array_equal(args[2], case.support)
    assert args[3] is case.config_maps
    assert args[4] is result

    case.direct_viz.write.assert_called_once_with(
        bp.ut.join(
            case.save_dir, "direct_space_images_AmpPhase.vts"
        ),
        complex_mode="AmpPhase",
    )
    case.make_vectors_viz.assert_called_once_with(
        case.geometry, position=(0.5, 0.5, 0.5)
    )
    case.vectors_viz.save.assert_called_once_with(
        bp.ut.join(case.save_dir, "KiKfQ.vtu"),
        binary=False,
    )

    assert result["res_dir"] == case.results_dir
    assert result["image com"] == (0.5, 0.5, 0.5)
    np.testing.assert_array_equal(result["Q"], case.geometry[2])
    np.testing.assert_array_equal(result["Ki"], case.geometry[3])
    np.testing.assert_array_equal(result["Kf"], case.geometry[4])


def test_process_dir_missing_support(processing_case, capsys):
    case = processing_case
    bp.os.remove(bp.ut.join(case.results_dir, "support.npy"))

    run_processing(case)

    assert case.make_image_viz.call_args.args[2] is None
    assert "support file is missing" in capsys.readouterr().out


@pytest.mark.parametrize("separate", [False, True])
def test_process_dir_reads_preprocess_metadata(
    processing_case, monkeypatch, separate
):
    case = processing_case
    case.config_maps["config"]["separate_scans"] = separate

    # For this fixture, results_phasing is immediately under experiment_dir,
    # so both lookup modes resolve to this directory.
    prep_dir = bp.Path(case.experiment_dir) / "preprocessed_data"
    prep_dir.mkdir()
    (prep_dir / "preprocess.xlsx").touch()

    read_excel = Mock(
        return_value=pd.DataFrame({
            "max ind (x)": [10],
            "max ind (y)": [20],
            "max ind frame": [30],
        })
    )
    monkeypatch.setattr(bp.pd, "read_excel", read_excel)

    run_processing(case)

    read_excel.assert_called_once_with(
        bp.ut.join(prep_dir, "preprocess.xlsx"), nrows=2
    )
    case.instrument.get_geometry.assert_called_once_with(
        [10, 20, 30], 7, case.config_maps
    )


@pytest.mark.parametrize("mode", ["AmpPhase", "ReIm"])
def test_process_dir_interpolation(
    processing_case, monkeypatch, mode
):
    case = processing_case
    case.config_maps["config_disp"].update({
        "interpolation_mode": mode,
        "interpolation_resolution": 2.5,
    })

    interpolated = Mock()
    interpolated.point_data = {
        "imRe": np.array([3.0, 0.0]),
        "imImag": np.array([4.0, 1.0]),
    }
    interpolate = Mock(return_value=interpolated)
    monkeypatch.setattr(bp.pu, "interpolate", interpolate)

    run_processing(case)

    case.direct_viz.get_structured_grid.assert_called_once_with(
        complex_mode=mode
    )
    interpolate.assert_called_once_with(
        case.direct_viz.get_structured_grid.return_value, 2.5
    )
    interpolated.save.assert_called_once_with(
        bp.ut.join(
            case.save_dir,
            f"direct_space_images_interpolated_{mode}.vti",
        )
    )

    if mode == "ReIm":
        expected = np.array([3 + 4j, 1j])
        np.testing.assert_allclose(
            interpolated.point_data["imAmp"], np.abs(expected)
        )
        np.testing.assert_allclose(
            interpolated.point_data["imPh"], np.angle(expected)
        )


def test_process_dir_skips_interpolation_without_resolution(
    processing_case, monkeypatch, capsys
):
    case = processing_case
    case.config_maps["config_disp"]["interpolation_mode"] = "AmpPhase"
    interpolate = Mock()
    monkeypatch.setattr(bp.pu, "interpolate", interpolate)

    run_processing(case)

    interpolate.assert_not_called()
    assert (
        "interpolation_resolution parameter not configured"
        in capsys.readouterr().out
    )


def test_process_dir_min_deconv_requires_resolution_setting(
    processing_case
):
    case = processing_case
    case.config_maps["config_disp"].update({
        "interpolation_mode": "AmpPhase",
        "interpolation_resolution": "min_deconv_res",
    })

    with pytest.raises(ValueError, match="determine_resolution_type"):
        run_processing(case)


def test_process_dir_writes_resolution(processing_case, monkeypatch):
    case = processing_case
    case.config_maps["config_disp"]["determine_resolution_type"] = "test"

    direct_resolution = Mock()
    reciprocal_resolution = Mock()
    make_resolution = Mock(
        return_value=(direct_resolution, reciprocal_resolution)
    )
    monkeypatch.setattr(bp.pu, "make_resolution_viz", make_resolution)

    run_processing(case)

    args = make_resolution.call_args.args
    assert args[0] is case.geometry
    np.testing.assert_allclose(args[1], np.abs(case.image))
    assert args[2] is case.config_maps

    direct_resolution.write.assert_called_once_with(
        bp.ut.join(case.save_dir, "resolution_direct.vts")
    )
    reciprocal_resolution.write.assert_called_once_with(
        bp.ut.join(case.save_dir, "resolution_recip.vts")
    )


def test_process_dir_creates_twin(processing_case):
    case = processing_case
    case.config_maps["config_disp"]["make_twin"] = True

    run_processing(case)

    assert case.make_image_viz.call_count == 2
    twin_args = case.make_image_viz.call_args_list[1].args
    np.testing.assert_array_equal(
        twin_args[1], np.conjugate(np.flip(case.image))
    )
    np.testing.assert_array_equal(
        twin_args[2], np.flip(case.support)
    )
    case.direct_viz.write.assert_any_call(
        bp.ut.join(case.save_dir, "twin_direct_space_images.vts")
    )


def test_process_dir_writes_coherence(processing_case, monkeypatch):
    case = processing_case
    coherence = np.full(case.image.shape, 0.5)
    np.save(
        bp.ut.join(case.results_dir, "coherence.npy"), coherence
    )

    direct_coherence = Mock()
    reciprocal_coherence = Mock()
    make_coherence = Mock(
        return_value=(direct_coherence, reciprocal_coherence)
    )
    monkeypatch.setattr(bp.pu, "make_coherence_viz", make_coherence)

    run_processing(case)

    args = make_coherence.call_args.args
    assert args[0] is case.geometry
    np.testing.assert_array_equal(args[1], coherence)
    assert args[2] == case.image.shape

    direct_coherence.write.assert_called_once_with(
        bp.ut.join(case.save_dir, "direct_space_coherence.vts")
    )
    reciprocal_coherence.write.assert_called_once_with(
        bp.ut.join(case.save_dir, "recip_space_coherence.vts")
    )


@pytest.fixture
def visualization_case(tmp_path, monkeypatch):
    config_maps = {
        "config": {"beamline": "test_beamline", "scan": "1-7"},
        "config_disp": {},
        "config_instr": {},
    }
    errors = {name: [] for name in config_maps}

    get_config_maps = Mock(
        return_value=(config_maps, False, errors)
    )
    monkeypatch.setattr(bp.com, "get_config_maps", get_config_maps)

    process_dir = Mock(return_value={"res_dir": "mock-results"})
    monkeypatch.setattr(bp, "process_dir", process_dir)

    # Avoid writing a spreadsheet or requiring an Excel writer engine.
    to_excel = Mock()
    monkeypatch.setattr(bp.pd.DataFrame, "to_excel", to_excel)

    return SimpleNamespace(
        experiment_dir=str(tmp_path),
        config_maps=config_maps,
        errors=errors,
        process_dir=process_dir,
        to_excel=to_excel,
    )


def test_handle_visualization_processes_single_directory(
    visualization_case
):
    case = visualization_case
    results_dir = bp.Path(case.experiment_dir) / "results_phasing"
    results_dir.mkdir()
    np.save(results_dir / "image.npy", np.ones((2, 2, 2)))

    result = bp.handle_visualization(case.experiment_dir)

    assert result == ""
    case.process_dir.assert_called_once_with(
        case.experiment_dir,
        case.config_maps,
        [7, str(results_dir).replace(bp.os.sep, "/")],
    )
    case.to_excel.assert_called_once_with(
        bp.ut.join(case.experiment_dir, "visualization_results.xlsx"),
        index=False,
    )


@pytest.mark.parametrize("missing", ["config_disp", "config_instr"])
def test_handle_visualization_requires_configuration(
    visualization_case, missing
):
    case = visualization_case
    del case.config_maps[missing]

    with pytest.raises(FileNotFoundError, match=f"missing {missing}"):
        bp.handle_visualization(case.experiment_dir)

    case.process_dir.assert_not_called()


def test_handle_visualization_rejects_config_errors(visualization_case):
    case = visualization_case
    case.errors["config_disp"] = ["invalid complex_mode"]

    with pytest.raises(ValueError, match="invalid complex_mode"):
        bp.handle_visualization(case.experiment_dir)

    case.process_dir.assert_not_called()


def test_handle_visualization_no_images(visualization_case):
    case = visualization_case

    with pytest.raises(FileNotFoundError, match="no image.npy files"):
        bp.handle_visualization(case.experiment_dir)


def test_handle_visualization_multipeak(
    visualization_case, monkeypatch
):
    case = visualization_case
    case.config_maps["config"]["multipeak"] = True
    case.config_maps["config_mp"] = {}
    case.errors["config_mp"] = []

    multipeak_process = Mock()
    monkeypatch.setattr(bp.mp, "process_dir", multipeak_process)

    assert bp.handle_visualization(case.experiment_dir) is None

    multipeak_process.assert_called_once_with(
        case.experiment_dir, case.config_maps
    )
    case.process_dir.assert_not_called()
    case.to_excel.assert_not_called()


def test_main_forwards_command_line_arguments(monkeypatch):
    handle_visualization = Mock()
    monkeypatch.setattr(bp, "handle_visualization", handle_visualization)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "beamline_postprocess.py",
            "/experiment",
            "--rec_id",
            "reconstruction_2",
            "--no_verify",
            "--debug",
        ],
    )

    bp.main()

    handle_visualization.assert_called_once_with(
        "/experiment",
        rec_id="reconstruction_2",
        no_verify=True,
        debug=True,
    )