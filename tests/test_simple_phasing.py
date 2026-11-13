import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import simple_phasing as sp  # Adjust to your package path.


@pytest.fixture
def phasing_case(tmp_path, monkeypatch):
    """Create a placeholder data file and mock the reconstruction worker."""
    datafile = tmp_path / "data.tif"
    datafile.touch()

    worker = Mock()
    worker.iterate.return_value = 0

    create_rec = Mock(return_value=worker)
    monkeypatch.setattr(sp.rec, "create_rec", create_rec)

    return SimpleNamespace(
        datafile=str(datafile),
        parent_dir=str(tmp_path),
        worker=worker,
        create_rec=create_rec,
    )


def test_reconstruction_success_uses_expected_parameters(phasing_case):
    case = phasing_case

    result = sp.reconstruction(case.datafile, debug=True)

    assert result is None
    case.create_rec.assert_called_once_with(
        {
            "algorithm_sequence": "3*(20*ER+180*HIO)+20*ER",
            "shrink_wrap_trigger": [1, 1],
            "shrink_wrap_type": "GAUSS",
            "shrink_wrap_threshold": 0.1,
            "shrink_wrap_gauss_sigma": 1.0,
            "twin_trigger": [2],
            "progress_trigger": [0, 20],
            "save_dir": "results",
        },
        case.datafile.replace(sp.os.sep, "/"),
        "cp",
        0,
        debug=True,
    )
    case.worker.iterate.assert_called_once_with()
    case.worker.save_res.assert_called_once_with(
        case.parent_dir.replace(sp.os.sep, "/")
    )


def test_reconstruction_custom_save_directory(phasing_case):
    case = phasing_case
    save_dir = sp.os.path.join(case.parent_dir, "custom_results")

    sp.reconstruction(case.datafile, save_dir=save_dir, debug=False)

    assert case.create_rec.call_args.kwargs == {
        "save_dir": save_dir,
        "debug": False,
    }
    case.worker.save_res.assert_called_once_with(save_dir)


def test_reconstruction_missing_datafile(phasing_case, capsys):
    case = phasing_case
    missing_file = sp.os.path.join(case.parent_dir, "missing.npy")

    result = sp.reconstruction(missing_file)

    assert result is None
    assert (
        f"no file found {missing_file.replace(sp.os.sep, '/')}"
        in capsys.readouterr().out
    )
    case.create_rec.assert_not_called()
    case.worker.iterate.assert_not_called()
    case.worker.save_res.assert_not_called()


def test_reconstruction_worker_creation_failure(phasing_case):
    case = phasing_case
    case.create_rec.return_value = None

    assert sp.reconstruction(case.datafile) is None

    case.create_rec.assert_called_once()
    case.worker.iterate.assert_not_called()
    case.worker.save_res.assert_not_called()


@pytest.mark.parametrize("return_code", [-1, -2])
def test_reconstruction_iteration_failure(phasing_case, return_code):
    case = phasing_case
    case.worker.iterate.return_value = return_code

    assert sp.reconstruction(case.datafile) is None

    case.worker.iterate.assert_called_once_with()
    case.worker.save_res.assert_not_called()


@pytest.mark.parametrize("return_code", [0, 1])
def test_reconstruction_saves_for_nonnegative_return_code(
    phasing_case, return_code
):
    case = phasing_case
    case.worker.iterate.return_value = return_code

    sp.reconstruction(case.datafile)

    case.worker.save_res.assert_called_once_with(
        case.parent_dir.replace(sp.os.sep, "/")
    )


def test_reconstruction_relative_datafile(phasing_case, monkeypatch):
    case = phasing_case
    monkeypatch.chdir(case.parent_dir)

    sp.reconstruction("data.tif")

    assert case.create_rec.call_args.args[1] == "data.tif"
    # os.path.split("data.tif") returns ("", "data.tif").
    case.worker.save_res.assert_called_once_with("")


@pytest.mark.parametrize("stage", ["create", "iterate", "save"])
def test_reconstruction_propagates_exceptions(phasing_case, stage):
    case = phasing_case
    error = RuntimeError(f"{stage} failed")

    if stage == "create":
        case.create_rec.side_effect = error
    elif stage == "iterate":
        case.worker.iterate.side_effect = error
    else:
        case.worker.save_res.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        sp.reconstruction(case.datafile, debug=True)

    assert exc_info.value is error

    if stage == "create":
        case.worker.iterate.assert_not_called()
    if stage in ("create", "iterate"):
        case.worker.save_res.assert_not_called()


@pytest.mark.parametrize(
    "flags, expected_debug",
    [
        ([], False),
        (["--debug"], True),
    ],
)
def test_main_forwards_cli_arguments(
    monkeypatch, tmp_path, flags, expected_debug
):
    datafile = str(tmp_path / "data.npy")
    reconstruction = Mock()
    monkeypatch.setattr(sp, "reconstruction", reconstruction)
    monkeypatch.setattr(
        sys,
        "argv",
        ["simple_phasing.py", datafile, *flags],
    )

    sp.main()

    reconstruction.assert_called_once_with(
        datafile, debug=expected_debug
    )


def test_main_requires_datafile(monkeypatch):
    reconstruction = Mock()
    monkeypatch.setattr(sp, "reconstruction", reconstruction)
    monkeypatch.setattr(sys, "argv", ["simple_phasing.py"])

    with pytest.raises(SystemExit) as exc_info:
        sp.main()

    assert exc_info.value.code == 2
    reconstruction.assert_not_called()