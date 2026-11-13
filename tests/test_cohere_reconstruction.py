import sys
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path

import pytest

import cohere_reconstruction as cr  # Adjust to your package path.


# ---------------------------------------------------------------------------
# Memory estimates and hostfile allocation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, factor, constant",
    [
        (None, 170, 100),
        ("ga_fast", 184, 428),
        ("populous", 250, 0),
    ],
)
@pytest.mark.parametrize("pc_in_use", [False, True])
def test_get_job_size(method, factor, constant, pc_in_use):
    size = 2_000_000
    expected = size * factor / 1_000_000 + constant
    if pc_in_use:
        expected *= 2

    assert cr.get_job_size(size, method, pc_in_use) == pytest.approx(
        expected
    )


def test_split_resources_distributes_devices_across_hosts(tmp_path):
    hostfile = tmp_path / "hosts"
    hostfile.write_text("host_a:3\nhost_b:3\n")

    result = cr.split_resources(str(hostfile), devs=2, no_scans=3)

    assert result == [
        str(hostfile) + "0",
        str(hostfile) + "1",
        str(hostfile) + "2",
    ]
    assert cr.os.path.isfile(result[0])
    assert Path(result[0]).read_text() == "host_a:2\n"
    assert Path(result[1]).read_text() == "host_a:1\nhost_b:1\n"
    assert Path(result[2]).read_text() == "host_b:2\n"
    assert hostfile.read_text() == "host_a:3\nhost_b:3\n"


# ---------------------------------------------------------------------------
# Single reconstruction
# ---------------------------------------------------------------------------

@pytest.fixture
def single_case(tmp_path, monkeypatch):
    params = {}
    read_config = Mock(return_value=params)
    monkeypatch.setattr(cr.ut, "read_config", read_config)

    worker = Mock()
    worker.iterate.return_value = 0

    create_rec = Mock(return_value=worker)
    monkeypatch.setattr(cr.rec, "create_rec", create_rec)

    return SimpleNamespace(
        params=params,
        read_config=read_config,
        worker=worker,
        create_rec=create_rec,
        conf_file=str(tmp_path / "conf" / "config_rec_trial"),
        datafile=str(tmp_path / "phasing_data" / "data.tif"),
        experiment_dir=str(tmp_path),
    )


def run_single(case, **kwargs):
    return cr.reconstruction_single(
        "np",
        case.conf_file,
        case.datafile,
        case.experiment_dir,
        -1,
        **kwargs,
    )


def test_reconstruction_single_success(single_case):
    case = single_case

    assert run_single(case, debug=True) == ""

    case.read_config.assert_called_once_with(case.conf_file)
    assert case.params["init_guess"] == "random"
    case.create_rec.assert_called_once_with(
        case.params,
        case.datafile,
        "np",
        -1,
        debug=True,
    )
    case.worker.iterate.assert_called_once_with()
    case.worker.save_res.assert_called_once_with(
        cr.ut.join(case.experiment_dir, "results_phasing_trial")
    )


def test_reconstruction_single_custom_save_directory(single_case):
    case = single_case
    case.params["save_dir"] = "/custom/results"
    case.params["init_guess"] = "continue"

    assert run_single(case) == ""

    assert case.params["init_guess"] == "continue"
    case.worker.save_res.assert_called_once_with("/custom/results")


def test_reconstruction_single_worker_creation_failure(single_case):
    case = single_case
    case.create_rec.return_value = None

    assert run_single(case) == (
        "Could not create Rec object, check config_rec parameters."
    )
    case.worker.iterate.assert_not_called()
    case.worker.save_res.assert_not_called()


def test_reconstruction_single_iteration_failure(single_case):
    case = single_case
    case.worker.iterate.return_value = -1

    assert run_single(case) == "reconstruction failed during iterations"
    case.worker.save_res.assert_not_called()


# ---------------------------------------------------------------------------
# Reconstruction dispatch
# ---------------------------------------------------------------------------

def test_process_scan_range_single_device(monkeypatch):
    single = Mock(return_value="single result")
    monkeypatch.setattr(cr, "reconstruction_single", single)

    result = cr.process_scan_range(
        None,
        "np",
        "config_rec",
        "data.tif",
        "/experiment",
        [-1],
        debug=True,
    )

    assert result == "single result"
    single.assert_called_once_with(
        "np",
        "config_rec",
        "data.tif",
        "/experiment",
        -1,
        debug=True,
    )


@pytest.mark.parametrize(
    "rec_type, pkg, backend, expected_devices",
    [
        (None, "cp", "populous", [0, 1]),
        ("ga_fast", "cp", "mpi", [0, 1]),
        ("ga_fast", "torch", "mpi", [-1, -1]),
        ("populous", "cp", "ga", [0, 1]),
        ("populous", "torch", "ga", [-1, -1]),
    ],
)
def test_process_scan_range_dispatch(
    monkeypatch, rec_type, pkg, backend, expected_devices
):
    populous = Mock()
    mpi = Mock()
    ga = Mock()
    monkeypatch.setattr(
        cr.reconstruction_populous, "reconstruction", populous
    )
    monkeypatch.setattr(cr.mpi_cmd, "run_with_mpi", mpi)
    monkeypatch.setattr(cr.ga, "reconstruction", ga)

    queue = Mock()
    monkeypatch.setattr(cr.os, "getpid", lambda: 1234)

    assert cr.process_scan_range(
        rec_type,
        pkg,
        "config_rec",
        "data.tif",
        "/experiment",
        [0, 1],
        hostfile="hosts",
        q=queue,
    ) == ""

    expected_args = (
        pkg,
        "config_rec",
        "data.tif",
        "/experiment",
        expected_devices,
    )
    mocks = {"populous": populous, "mpi": mpi, "ga": ga}

    if backend == "mpi":
        mpi.assert_called_once_with(*expected_args, "hosts")
    else:
        mocks[backend].assert_called_once_with(*expected_args)

    for name, mock in mocks.items():
        if name != backend:
            mock.assert_not_called()

    queue.put.assert_called_once_with(
        (1234, expected_devices, "hosts")
    )


# ---------------------------------------------------------------------------
# Reconstruction management
# ---------------------------------------------------------------------------

@pytest.fixture
def management_case(tmp_path, monkeypatch):
    config_maps = {
        "config": {"beamline": "test_beamline"},
        "config_rec": {
            "processing": "np",
            "device": [-1],
            "reconstructions": 1,
            "algorithm_sequence": "1*ER",
        },
    }
    errors = {name: [] for name in config_maps}

    get_config_maps = Mock(return_value=(config_maps, False, errors))
    get_pkg = Mock(return_value="np")
    monkeypatch.setattr(cr.com, "get_config_maps", get_config_maps)
    monkeypatch.setattr(cr.com, "get_pkg", get_pkg)

    single = Mock(return_value="")
    scan_range = Mock(return_value="")
    get_one_dev = Mock(return_value=2)
    get_gpu_use = Mock(return_value=([0, 1], 2, None))

    monkeypatch.setattr(cr, "reconstruction_single", single)
    monkeypatch.setattr(cr, "process_scan_range", scan_range)
    monkeypatch.setattr(cr.balancer, "get_one_dev", get_one_dev)
    monkeypatch.setattr(cr.balancer, "get_gpu_use", get_gpu_use)

    # Keep device-selection tests independent of the host OS.
    monkeypatch.setattr(cr.sys, "platform", "linux")

    data_dir = tmp_path / "phasing_data"
    data_dir.mkdir()
    datafile = data_dir / "data.tif"
    datafile.touch()  # Contents are never read unless explicitly mocked.

    return SimpleNamespace(
        experiment_dir=str(tmp_path),
        datafile=str(datafile),
        config_maps=config_maps,
        errors=errors,
        get_config_maps=get_config_maps,
        get_pkg=get_pkg,
        single=single,
        scan_range=scan_range,
        get_one_dev=get_one_dev,
        get_gpu_use=get_gpu_use,
    )


def test_manage_reconstruction_single_cpu(management_case, capsys):
    case = management_case

    assert cr.manage_reconstruction(case.experiment_dir) is None

    case.get_config_maps.assert_called_once_with(
        case.experiment_dir, ["config_rec", "config_mp"]
    )
    case.single.assert_called_once_with(
        "np",
        cr.ut.join(case.experiment_dir, "conf", "config_rec"),
        case.datafile,
        case.experiment_dir,
        -1,
    )
    case.get_one_dev.assert_not_called()
    case.get_gpu_use.assert_not_called()
    assert "finished reconstruction" in capsys.readouterr().out


@pytest.mark.parametrize("hpc, expected_device", [(False, 2), (True, 0)])
def test_manage_reconstruction_single_gpu(
    management_case, hpc, expected_device
):
    case = management_case
    case.get_pkg.return_value = "cp"
    case.config_maps["config_rec"]["device"] = [0, 2]

    cr.manage_reconstruction(case.experiment_dir, hpc=hpc)

    assert case.single.call_args.args[-1] == expected_device
    assert case.single.call_args.kwargs == {"hpc": hpc}

    if hpc:
        case.get_one_dev.assert_not_called()
    else:
        case.get_one_dev.assert_called_once_with([0, 2])


def test_manage_reconstruction_alternate_config(management_case):
    case = management_case

    cr.manage_reconstruction(
        case.experiment_dir, rec_id="trial", debug=True
    )

    case.get_config_maps.assert_called_once_with(
        case.experiment_dir,
        ["config_rec", "config_mp"],
        rec_id="trial",
        debug=True,
    )
    case.single.assert_called_once_with(
        "np",
        cr.ut.join(case.experiment_dir, "conf", "config_rec_trial"),
        case.datafile,
        case.experiment_dir,
        -1,
        debug=True,
    )


def test_manage_reconstruction_reports_worker_failure(
    management_case, capsys
):
    case = management_case
    case.single.return_value = "reconstruction failed during iterations"

    cr.manage_reconstruction(case.experiment_dir)

    assert (
        "exiting reconstruction failed during iterations"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize("config_name", ["config", "config_rec"])
def test_manage_reconstruction_rejects_config_errors(
    management_case, config_name
):
    case = management_case
    case.errors[config_name] = ["invalid configuration"]

    with pytest.raises(ValueError, match="invalid configuration"):
        cr.manage_reconstruction(case.experiment_dir)

    case.get_pkg.assert_not_called()
    case.single.assert_not_called()


@pytest.mark.parametrize("rec_id", [None, "trial"])
def test_manage_reconstruction_missing_rec_config(
    management_case, rec_id
):
    case = management_case
    del case.config_maps["config_rec"]
    expected_name = "config_rec" if rec_id is None else "config_rec_trial"

    with pytest.raises(
        FileNotFoundError, match=f"missing {expected_name} file"
    ):
        cr.manage_reconstruction(case.experiment_dir, rec_id=rec_id)


def test_manage_reconstruction_no_verify(management_case, capsys):
    case = management_case
    case.errors["config_rec"] = ["invalid reconstruction setting"]

    cr.manage_reconstruction(case.experiment_dir, no_verify=True)

    case.single.assert_called_once()
    assert "invalid reconstruction setting" in capsys.readouterr().out


def test_manage_reconstruction_missing_data(management_case):
    case = management_case
    cr.os.remove(case.datafile)

    with pytest.raises(FileNotFoundError, match="did not find data.tif"):
        cr.manage_reconstruction(case.experiment_dir)

    case.single.assert_not_called()
    case.scan_range.assert_not_called()


@pytest.mark.parametrize(
    "ga_settings, expected_type",
    [
        ({}, None),
        ({"ga_generations": 3, "ga_fast": True}, "ga_fast"),
        ({"ga_generations": 3, "ga_fast": False}, "populous"),
    ],
)
def test_manage_reconstruction_multiple_cpu_reconstructions(
    management_case, ga_settings, expected_type
):
    case = management_case
    case.config_maps["config_rec"].update(
        {"reconstructions": 2, **ga_settings}
    )

    cr.manage_reconstruction(case.experiment_dir, debug=True)

    case.scan_range.assert_called_once_with(
        expected_type,
        "np",
        cr.ut.join(case.experiment_dir, "conf", "config_rec"),
        case.datafile,
        case.experiment_dir,
        [-1, -1],
        None,
        None,
        True,
    )
    case.single.assert_not_called()
    case.get_gpu_use.assert_not_called()


def test_manage_reconstruction_gpu_memory_allocation(
    management_case, monkeypatch
):
    case = management_case
    case.get_pkg.return_value = "cp"
    case.config_maps["config_rec"].update({
        "device": [0, 1],
        "reconstructions": 2,
        "algorithm_sequence": "1*ER+1*pc",
    })

    read_tif = Mock(return_value=SimpleNamespace(size=1000))
    job_size = Mock(return_value=123.0)
    monkeypatch.setattr(cr.ut, "read_tif", read_tif)
    monkeypatch.setattr(cr, "get_job_size", job_size)

    cr.manage_reconstruction(case.experiment_dir)

    read_tif.assert_called_once_with(case.datafile)
    job_size.assert_called_once_with(1000, None, True)
    case.get_gpu_use.assert_called_once_with([0, 1], 2, 123.0)
    assert case.scan_range.call_args.args[5] == [0, 1]


def test_manage_reconstruction_fast_ga_insufficient_resources(
    management_case, monkeypatch
):
    case = management_case
    case.get_pkg.return_value = "cp"
    case.config_maps["config_rec"].update({
        "device": [0],
        "reconstructions": 2,
        "ga_generations": 3,
        "ga_fast": True,
    })
    case.get_gpu_use.return_value = ([0], 1, None)
    monkeypatch.setattr(
        cr.ut, "read_tif", Mock(return_value=SimpleNamespace(size=1000))
    )

    with pytest.raises(
        ValueError, match="requested 2 reconstructions but only 1"
    ):
        cr.manage_reconstruction(case.experiment_dir)

    case.scan_range.assert_not_called()


def test_manage_reconstruction_launches_separate_scans(
    management_case, monkeypatch
):
    case = management_case
    case.config_maps["config"]["separate_scans"] = True

    root = cr.os.path.dirname(case.datafile)
    experiment_path = cr.os.path.dirname(root)

    for scan in ["scan_1", "scan_2"]:
        data_dir = cr.os.path.join(experiment_path, scan, "phasing_data")
        cr.os.makedirs(data_dir)
        with open(cr.os.path.join(data_dir, "data.tif"), "w"):
            pass

    # Mock processes rather than launching child interpreters.
    processes = [Mock(pid=101), Mock(pid=102)]
    process_factory = Mock(side_effect=processes)
    monkeypatch.setattr(cr, "Process", process_factory)

    cr.manage_reconstruction(case.experiment_dir)

    assert process_factory.call_count == 2
    launched_dirs = set()

    for call in process_factory.call_args_list:
        assert call.kwargs["target"] is case.scan_range
        args = call.kwargs["args"]
        assert args[:3] == (
            None,
            "np",
            cr.ut.join(case.experiment_dir, "conf", "config_rec"),
        )
        assert args[3] == cr.ut.join(args[4], "phasing_data", "data.tif")
        assert args[5:] == ([-1], None, None)
        launched_dirs.add(cr.os.path.basename(args[4]))

    assert launched_dirs == {"scan_1", "scan_2"}
    for process in processes:
        process.start.assert_called_once_with()
        process.join.assert_called_once_with()

    case.single.assert_not_called()


def test_manage_reconstruction_chrono_requires_one_reconstruction(
    management_case
):
    case = management_case
    case.config_maps["config"]["chrono"] = True
    case.config_maps["config_rec"]["reconstructions"] = 2

    with pytest.raises(ValueError, match="configure reconstructions to 1"):
        cr.manage_reconstruction(case.experiment_dir)


def test_manage_reconstruction_multipeak(management_case, monkeypatch):
    case = management_case
    case.config_maps["config_mp"] = {"peak_parameter": 42}

    cr.os.mkdir(cr.ut.join(case.experiment_dir, "mp_1"))
    cr.os.mkdir(cr.ut.join(case.experiment_dir, "mp_2"))

    reconstruction = Mock()
    monkeypatch.setattr(cr.multipeak, "reconstruction", reconstruction)

    cr.manage_reconstruction(case.experiment_dir, debug=True)

    args = reconstruction.call_args.args
    assert args[0] == "np"
    assert args[1]["peak_parameter"] == 42
    assert args[1]["beamline"] == "test_beamline"
    assert args[1]["save_dir"] == (
        f"{case.experiment_dir}/results_phasing"
    )
    assert set(args[2]) == {
        cr.ut.join(case.experiment_dir, "mp_1"),
        cr.ut.join(case.experiment_dir, "mp_2"),
    }
    assert args[3] == [-1]
    assert reconstruction.call_args.kwargs == {"debug": True}
    case.single.assert_not_called()


def test_manage_reconstruction_multipeak_rejects_torch(
    management_case
):
    case = management_case
    case.get_pkg.return_value = "torch"
    case.config_maps["config_mp"] = {}

    with pytest.raises(ValueError, match="not supported on torch"):
        cr.manage_reconstruction(case.experiment_dir)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "flags, expected",
    [
        (
            [],
            {
                "rec_id": None,
                "no_verify": False,
                "debug": False,
                "hpc": False,
            },
        ),
        (
            ["--rec_id", "trial", "--no_verify", "--debug", "--hpc"],
            {
                "rec_id": "trial",
                "no_verify": True,
                "debug": True,
                "hpc": True,
            },
        ),
    ],
)
def test_main_forwards_arguments(monkeypatch, tmp_path, flags, expected):
    manage = Mock()
    monkeypatch.setattr(cr, "manage_reconstruction", manage)
    monkeypatch.setattr(
        sys,
        "argv",
        ["cohere_reconstruction.py", str(tmp_path), *flags],
    )

    cr.main()

    manage.assert_called_once_with(str(tmp_path), **expected)


def test_main_requires_experiment_directory(monkeypatch):
    manage = Mock()
    monkeypatch.setattr(cr, "manage_reconstruction", manage)
    monkeypatch.setattr(sys, "argv", ["cohere_reconstruction.py"])

    with pytest.raises(SystemExit) as exc_info:
        cr.main()

    assert exc_info.value.code == 2
    manage.assert_not_called()