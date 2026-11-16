import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import cohere_ui.beamline_preprocess as bp


@pytest.fixture
def prep_case(tmp_path, monkeypatch):
    """Provide configuration and mock all external processing services."""
    config_maps = {
        "config": {
            "beamline": "test_beamline",
            "scan": "1-2,5",
        },
        "config_instr": {},
        "config_prep": {
            "remove_outliers": True,
            "do_RSM": True,
        },
    }
    errors = {name: [] for name in config_maps}

    get_config_maps = Mock(
        return_value=(config_maps, False, errors)
    )
    monkeypatch.setattr(bp.com, "get_config_maps", get_config_maps)

    batches = [
        [(1, "data_1"), (2, "data_2")],
        [(5, "data_5")],
    ]
    instrument = Mock()
    instrument.datainfo4scans.return_value = batches

    instrument_module = Mock()
    instrument_module.create_instr.return_value = instrument

    import_module = Mock(return_value=instrument_module)
    monkeypatch.setattr(bp.importlib, "import_module", import_module)

    process_batch = Mock(return_value=[])
    process_separate_scans = Mock()
    multipeak_preprocess = Mock(return_value=[])

    monkeypatch.setattr(
        bp.preprocessor, "process_batch", process_batch
    )
    monkeypatch.setattr(
        bp.preprocessor,
        "process_separate_scans",
        process_separate_scans,
    )
    monkeypatch.setattr(
        bp.mp, "preprocess", multipeak_preprocess
    )

    write_config = Mock()
    monkeypatch.setattr(bp.ut, "write_config", write_config)

    return SimpleNamespace(
        experiment_dir=str(tmp_path),
        config_maps=config_maps,
        errors=errors,
        batches=batches,
        instrument=instrument,
        instrument_module=instrument_module,
        import_module=import_module,
        get_config_maps=get_config_maps,
        process_batch=process_batch,
        process_separate_scans=process_separate_scans,
        multipeak_preprocess=multipeak_preprocess,
        write_config=write_config,
    )


def test_handle_prep_combines_all_scans(prep_case, capsys):
    case = prep_case
    case.process_batch.return_value = [2]

    result = bp.handle_prep(case.experiment_dir)

    assert result == ""

    case.get_config_maps.assert_called_once_with(
        case.experiment_dir,
        ["config_prep", "config_instr", "config_mp"],
    )
    case.import_module.assert_called_once_with(
        "cohere_beamlines.test_beamline.instrument"
    )
    case.instrument_module.create_instr.assert_called_once_with(
        case.config_maps, need_det=True
    )
    case.instrument.datainfo4scans.assert_called_once_with()

    case.process_batch.assert_called_once_with(
        [(1, "data_1"), (2, "data_2"), (5, "data_5")],
        case.experiment_dir,
        False,  # separate_scan_ranges
        True,   # remove_outliers
        case.instrument,
        True,   # do_RSM
    )
    case.process_separate_scans.assert_not_called()
    case.multipeak_preprocess.assert_not_called()

    assert case.config_maps["config_prep"]["outliers_scans"] == [2]
    case.write_config.assert_called_once_with(
        case.config_maps["config_prep"],
        bp.ut.join(case.experiment_dir, "conf", "config_prep"),
    )

    output = capsys.readouterr().out
    assert "pre-processing data" in output
    assert "finished beamline preprocessing" in output


def test_handle_prep_separate_scans(prep_case):
    case = prep_case
    case.config_maps["config"]["separate_scans"] = True

    assert bp.handle_prep(case.experiment_dir) == ""

    case.process_separate_scans.assert_called_once_with(
        [(1, "data_1"), (2, "data_2"), (5, "data_5")],
        case.experiment_dir,
        case.instrument,
        True,
    )
    case.process_batch.assert_not_called()
    case.multipeak_preprocess.assert_not_called()
    case.write_config.assert_called_once()


def test_handle_prep_separate_scan_ranges(prep_case):
    case = prep_case
    case.config_maps["config"]["separate_scan_ranges"] = True
    case.process_batch.side_effect = [[2], [5]]

    assert bp.handle_prep(case.experiment_dir) == ""

    assert case.process_batch.call_count == 2

    for call, batch in zip(
        case.process_batch.call_args_list, case.batches
    ):
        assert call.args == (
            batch,
            case.experiment_dir,
            True,  # separate_scan_ranges
            True,  # remove_outliers
            case.instrument,
            True,  # do_RSM
        )

    assert case.config_maps["config_prep"]["outliers_scans"] == [2, 5]
    case.process_separate_scans.assert_not_called()
    case.multipeak_preprocess.assert_not_called()


def test_handle_prep_multipeak(prep_case):
    case = prep_case
    case.config_maps["config"]["multipeak"] = True
    case.config_maps["config_mp"] = {}
    case.errors["config_mp"] = []
    case.multipeak_preprocess.return_value = [5]

    assert bp.handle_prep(case.experiment_dir) == ""

    case.multipeak_preprocess.assert_called_once_with(
        case.instrument,
        case.batches,
        case.experiment_dir,
        case.config_maps,
    )
    assert case.config_maps["config_prep"]["outliers_scans"] == [5]
    case.process_batch.assert_not_called()
    case.process_separate_scans.assert_not_called()


def test_handle_prep_default_processing_options(prep_case):
    case = prep_case
    case.config_maps["config_prep"].clear()

    bp.handle_prep(case.experiment_dir)

    case.process_batch.assert_called_once_with(
        [(1, "data_1"), (2, "data_2"), (5, "data_5")],
        case.experiment_dir,
        False,
        False,  # remove_outliers defaults to False
        case.instrument,
        None,   # do_RSM defaults to None
    )


def test_handle_prep_removes_stale_outliers(prep_case):
    case = prep_case
    case.config_maps["config_prep"]["outliers_scans"] = [99]
    case.process_batch.return_value = []

    bp.handle_prep(case.experiment_dir)

    assert "outliers_scans" not in case.config_maps["config_prep"]
    case.write_config.assert_called_once_with(
        case.config_maps["config_prep"],
        bp.ut.join(case.experiment_dir, "conf", "config_prep"),
    )


@pytest.mark.parametrize("batches", [[], [[]], [[], []]])
def test_handle_prep_no_data(prep_case, batches, capsys):
    case = prep_case
    case.instrument.datainfo4scans.return_value = batches

    result = bp.handle_prep(case.experiment_dir)

    assert result == "no data found for scans, exiting"
    assert result in capsys.readouterr().out
    case.process_batch.assert_not_called()
    case.process_separate_scans.assert_not_called()
    case.multipeak_preprocess.assert_not_called()
    case.write_config.assert_not_called()


@pytest.mark.parametrize(
    "config_name", ["config", "config_instr", "config_prep"]
)
def test_handle_prep_rejects_configuration_errors(
    prep_case, config_name
):
    case = prep_case
    case.errors[config_name] = ["invalid configuration"]

    with pytest.raises(ValueError, match="invalid configuration"):
        bp.handle_prep(case.experiment_dir)

    case.import_module.assert_not_called()
    case.process_batch.assert_not_called()
    case.write_config.assert_not_called()


def test_handle_prep_requires_instrument_configuration(prep_case):
    case = prep_case
    del case.config_maps["config_instr"]

    with pytest.raises(
        FileNotFoundError, match="missing config_instr file"
    ):
        bp.handle_prep(case.experiment_dir)

    case.import_module.assert_not_called()
    case.write_config.assert_not_called()


def test_handle_prep_no_verify_prints_errors_and_continues(
    prep_case, capsys
):
    case = prep_case
    case.errors["config_prep"] = ["invalid prep setting"]

    assert bp.handle_prep(
        case.experiment_dir, no_verify=True
    ) == ""

    case.get_config_maps.assert_called_once_with(
        case.experiment_dir,
        ["config_prep", "config_instr", "config_mp"],
        no_verify=True,
    )
    case.process_batch.assert_called_once()
    assert "invalid prep setting" in capsys.readouterr().out


@pytest.mark.parametrize("failure_stage", ["import", "create"])
def test_handle_prep_propagates_instrument_errors(
    prep_case, failure_stage, capsys
):
    case = prep_case
    error = RuntimeError("instrument initialization failed")

    if failure_stage == "import":
        case.import_module.side_effect = error
    else:
        case.instrument_module.create_instr.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        bp.handle_prep(case.experiment_dir)

    assert exc_info.value is error
    assert "exiting pre-processing" in capsys.readouterr().out
    case.process_batch.assert_not_called()
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
    experiment_dir = str(tmp_path)
    handle_prep = Mock()
    monkeypatch.setattr(bp, "handle_prep", handle_prep)
    monkeypatch.setattr(
        sys,
        "argv",
        ["beamline_preprocess.py", experiment_dir, *flags],
    )

    bp.main()

    handle_prep.assert_called_once_with(
        experiment_dir,
        no_verify=expected_no_verify,
        debug=expected_debug,
    )


def test_main_requires_experiment_directory(monkeypatch):
    handle_prep = Mock()
    monkeypatch.setattr(bp, "handle_prep", handle_prep)
    monkeypatch.setattr(sys, "argv", ["beamline_preprocess.py"])

    with pytest.raises(SystemExit) as exc_info:
        bp.main()

    assert exc_info.value.code == 2
    handle_prep.assert_not_called()


def test_handle_prep_allows_missing_prep_configuration(
    prep_case, capsys
):
    case = prep_case
    del case.config_maps["config_prep"]
    del case.errors["config_prep"]

    assert bp.handle_prep(case.experiment_dir) == ""

    case.process_batch.assert_called_once_with(
        [(1, "data_1"), (2, "data_2"), (5, "data_5")],
        case.experiment_dir,
        False,
        False,
        case.instrument,
        None,
    )
    case.write_config.assert_not_called()
    assert "no config_prep file, continuing" in capsys.readouterr().out