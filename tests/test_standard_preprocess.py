import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import standard_preprocess as sp  # Adjust to your package path.


@pytest.fixture
def preprocessing_case(tmp_path, monkeypatch):
    """Provide configuration and mock external preprocessing services."""
    config_maps = {
        "config": {},
        "config_data": {"intensity_threshold": 5.0},
        "config_rec": {"processing": "np"},
    }
    errors = {name: [] for name in config_maps}

    get_config_maps = Mock(
        return_value=(config_maps, False, errors)
    )
    monkeypatch.setattr(sp.com, "get_config_maps", get_config_maps)

    get_pkg = Mock(return_value="np")
    monkeypatch.setattr(sp.com, "get_pkg", get_pkg)

    # Return a new configuration dictionary, as the preprocessing API does.
    prep = Mock(side_effect=lambda datafile, **params: dict(params))
    monkeypatch.setattr(sp.fd, "prep", prep)

    write_config = Mock()
    monkeypatch.setattr(sp.ut, "write_config", write_config)

    chrono_format = Mock()
    monkeypatch.setattr(sp.te, "format_data", chrono_format)

    # Make traversal deterministic and ignore directories created during a run.
    entries = []
    monkeypatch.setattr(
        sp.os, "listdir", Mock(side_effect=lambda path: list(entries))
    )

    return SimpleNamespace(
        root=tmp_path,
        experiment_dir=str(tmp_path),
        config_maps=config_maps,
        errors=errors,
        entries=entries,
        get_config_maps=get_config_maps,
        get_pkg=get_pkg,
        prep=prep,
        write_config=write_config,
        chrono_format=chrono_format,
    )


def add_entry(case, name):
    """Create an experiment subdirectory and include it in traversal."""
    path = case.root / name
    path.mkdir()
    case.entries.append(name)
    return path


def test_format_data_single_experiment(preprocessing_case, capsys):
    case = preprocessing_case
    add_entry(case, "preprocessed_data")

    result = sp.format_data(case.experiment_dir, debug=True)

    assert result is None
    case.get_config_maps.assert_called_once_with(
        case.experiment_dir,
        ["config_data", "config_rec"],
        debug=True,
    )
    case.get_pkg.assert_called_once_with("np", [0])

    data_dir = sp.ut.join(case.experiment_dir, "phasing_data")
    assert (case.root / "phasing_data").is_dir()

    case.prep.assert_called_once_with(
        sp.ut.join(
            case.experiment_dir,
            "preprocessed_data",
            "prep_data.tif",
        ),
        intensity_threshold=5.0,
        data_dir=data_dir,
        pkg="np",
    )
    case.write_config.assert_called_once_with(
        {"intensity_threshold": 5.0, "pkg": "np"},
        sp.ut.join(case.experiment_dir, "conf", "config_data"),
    )
    case.chrono_format.assert_not_called()
    assert "formatting data" in capsys.readouterr().out


def test_format_data_custom_output_directory(preprocessing_case):
    case = preprocessing_case
    add_entry(case, "preprocessed_data")
    custom_dir = case.root / "custom_phasing"
    case.config_maps["config_data"]["data_dir"] = str(custom_dir)

    sp.format_data(case.experiment_dir)

    assert custom_dir.is_dir()
    assert case.prep.call_args.kwargs["data_dir"] == str(custom_dir)

    saved_config = case.write_config.call_args.args[0]
    assert "data_dir" not in saved_config


@pytest.mark.parametrize("entry", ["scan_1", "scan_1-3", "mp_1"])
def test_format_data_scan_and_multipeak_directories(
    preprocessing_case, entry
):
    case = preprocessing_case
    proc_dir = add_entry(case, entry)

    sp.format_data(case.experiment_dir)

    assert (proc_dir / "phasing_data").is_dir()
    case.prep.assert_called_once_with(
        sp.ut.join(str(proc_dir), "preprocessed_data", "prep_data.tif"),
        intensity_threshold=5.0,
        data_dir=sp.ut.join(str(proc_dir), "phasing_data"),
        pkg="np",
    )


def test_format_data_ignores_unrelated_directories(preprocessing_case):
    case = preprocessing_case
    add_entry(case, "conf")
    add_entry(case, "results_phasing")
    add_entry(case, "preprocessed_data")

    sp.format_data(case.experiment_dir)

    case.prep.assert_called_once()
    assert not (case.root / "conf" / "phasing_data").exists()
    assert not (case.root / "results_phasing" / "phasing_data").exists()


def test_format_data_existing_output_directory(preprocessing_case):
    case = preprocessing_case
    add_entry(case, "preprocessed_data")
    output_dir = case.root / "phasing_data"
    output_dir.mkdir()
    marker = output_dir / "existing.txt"
    marker.write_text("preserve me")

    sp.format_data(case.experiment_dir)

    assert marker.read_text() == "preserve me"
    case.prep.assert_called_once()


def test_format_data_saves_last_updated_configuration(preprocessing_case):
    case = preprocessing_case
    scan1 = add_entry(case, "scan_1")
    scan2 = add_entry(case, "scan_2")

    def update_threshold(datafile, **params):
        params["intensity_threshold"] += 1
        return params

    case.prep.side_effect = update_threshold

    sp.format_data(case.experiment_dir)

    assert case.prep.call_count == 2
    first_call, second_call = case.prep.call_args_list

    assert first_call.args == (
        sp.ut.join(str(scan1), "preprocessed_data", "prep_data.tif"),
    )
    assert second_call.args == (
        sp.ut.join(str(scan2), "preprocessed_data", "prep_data.tif"),
    )
    assert first_call.kwargs["intensity_threshold"] == 5.0
    assert second_call.kwargs["intensity_threshold"] == 6.0
    assert second_call.kwargs["data_dir"] == sp.ut.join(
        str(scan2), "phasing_data"
    )

    case.write_config.assert_called_once_with(
        {"intensity_threshold": 7.0, "pkg": "np"},
        sp.ut.join(case.experiment_dir, "conf", "config_data"),
    )


def test_format_data_defaults_to_auto_package(preprocessing_case):
    case = preprocessing_case
    add_entry(case, "preprocessed_data")
    del case.config_maps["config_rec"]

    sp.format_data(case.experiment_dir)

    case.get_pkg.assert_called_once_with("auto", [0])


def test_format_data_uses_selected_package(preprocessing_case):
    case = preprocessing_case
    add_entry(case, "preprocessed_data")
    case.config_maps["config_rec"]["processing"] = "cp"
    case.get_pkg.return_value = "cp"

    sp.format_data(case.experiment_dir)

    case.get_pkg.assert_called_once_with("cp", [0])
    assert case.prep.call_args.kwargs["pkg"] == "cp"


def test_format_data_package_selection_falls_back_to_numpy(
    preprocessing_case
):
    case = preprocessing_case
    add_entry(case, "preprocessed_data")
    case.get_pkg.side_effect = RuntimeError("GPU package unavailable")

    sp.format_data(case.experiment_dir)

    assert case.prep.call_args.kwargs["pkg"] == "np"
    assert case.write_config.call_args.args[0]["pkg"] == "np"


@pytest.mark.parametrize("config_name", ["config", "config_data"])
def test_format_data_rejects_configuration_errors(
    preprocessing_case, config_name
):
    case = preprocessing_case
    case.errors[config_name] = ["invalid configuration"]

    with pytest.raises(ValueError, match="invalid configuration"):
        sp.format_data(case.experiment_dir)

    case.prep.assert_not_called()
    case.write_config.assert_not_called()
    case.chrono_format.assert_not_called()


def test_format_data_requires_data_configuration(preprocessing_case):
    case = preprocessing_case
    del case.config_maps["config_data"]

    with pytest.raises(FileNotFoundError, match="Missing config_data"):
        sp.format_data(case.experiment_dir)

    case.prep.assert_not_called()
    case.write_config.assert_not_called()


def test_format_data_rejects_processing_configuration_error(
    preprocessing_case
):
    case = preprocessing_case
    # The implementation checks exact membership of "processing".
    case.errors["config_rec"] = ["processing"]

    with pytest.raises(ValueError):
        sp.format_data(case.experiment_dir)

    case.prep.assert_not_called()
    case.write_config.assert_not_called()


def test_format_data_ignores_unrelated_reconstruction_errors(
    preprocessing_case
):
    case = preprocessing_case
    add_entry(case, "preprocessed_data")
    case.errors["config_rec"] = ["algorithm_sequence"]

    sp.format_data(case.experiment_dir)

    case.prep.assert_called_once()
    case.write_config.assert_called_once()


def test_format_data_no_verify_prints_errors_and_continues(
    preprocessing_case, capsys
):
    case = preprocessing_case
    add_entry(case, "preprocessed_data")
    case.errors["config_data"] = ["invalid intensity threshold"]

    sp.format_data(case.experiment_dir, no_verify=True)

    case.get_config_maps.assert_called_once_with(
        case.experiment_dir,
        ["config_data", "config_rec"],
        no_verify=True,
    )
    case.prep.assert_called_once()
    assert "invalid intensity threshold" in capsys.readouterr().out


def test_format_data_chrono_delegates(preprocessing_case):
    case = preprocessing_case
    case.config_maps["config"]["chrono"] = True

    assert sp.format_data(case.experiment_dir) is None

    case.chrono_format.assert_called_once_with(
        case.experiment_dir, case.config_maps
    )
    case.get_pkg.assert_not_called()
    case.prep.assert_not_called()
    case.write_config.assert_not_called()


def test_format_data_propagates_preprocessing_failure(preprocessing_case):
    case = preprocessing_case
    add_entry(case, "preprocessed_data")
    error = RuntimeError("preprocessing failed")
    case.prep.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        sp.format_data(case.experiment_dir)

    assert exc_info.value is error
    case.write_config.assert_not_called()


@pytest.mark.parametrize(
    "flags, expected_no_verify, expected_debug",
    [
        ([], False, False),
        (["--no_verify"], True, False),
        (["--debug"], False, True),
        (["--no_verify", "--debug"], True, True),
    ],
)
def test_main_forwards_cli_arguments(
    monkeypatch, tmp_path, flags, expected_no_verify, expected_debug
):
    format_data = Mock()
    monkeypatch.setattr(sp, "format_data", format_data)
    monkeypatch.setattr(
        sys,
        "argv",
        ["standard_preprocess.py", str(tmp_path), *flags],
    )

    sp.main()

    format_data.assert_called_once_with(
        str(tmp_path),
        no_verify=expected_no_verify,
        debug=expected_debug,
    )


def test_main_requires_experiment_directory(monkeypatch):
    format_data = Mock()
    monkeypatch.setattr(sp, "format_data", format_data)
    monkeypatch.setattr(sys, "argv", ["standard_preprocess.py"])

    with pytest.raises(SystemExit) as exc_info:
        sp.main()

    assert exc_info.value.code == 2
    format_data.assert_not_called()


def test_format_data_no_matching_directories(preprocessing_case):
    case = preprocessing_case
    add_entry(case, "conf")

    assert sp.format_data(case.experiment_dir) is None

    case.prep.assert_not_called()
    case.write_config.assert_called_once_with(
        {"intensity_threshold": 5.0},
        sp.ut.join(case.experiment_dir, "conf", "config_data"),
    )