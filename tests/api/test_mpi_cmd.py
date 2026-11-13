# tests/test_mpi_cmd.py
import os
import subprocess
import sys
from unittest.mock import Mock

import pytest

from cohere_ui.api import mpi_cmd  # Adjust to your project's import path


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    [
        ("Open MPI 4.1.5", "", True),
        ("", "OpenRTE 4.1.5", True),
        ("MPICH Version 4.0", "", False),
    ],
)
def test_is_open_mpi_detects_launcher(stdout, stderr, expected, monkeypatch):
    monkeypatch.setattr(
        mpi_cmd.subprocess,
        "run",
        Mock(
            return_value=subprocess.CompletedProcess(
                args=["mpiexec", "--version"],
                returncode=0,
                stdout=stdout,
                stderr=stderr,
            )
        ),
    )

    assert mpi_cmd.is_open_mpi() is expected


def test_is_open_mpi_returns_false_for_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        mpi_cmd.subprocess,
        "run",
        Mock(
            return_value=subprocess.CompletedProcess(
                args=["mpiexec", "--version"],
                returncode=1,
                stdout="Open MPI",
                stderr="",
            )
        ),
    )

    assert mpi_cmd.is_open_mpi() is False


def test_is_open_mpi_raises_when_launcher_is_missing(monkeypatch):
    monkeypatch.setattr(
        mpi_cmd.subprocess,
        "run",
        Mock(side_effect=FileNotFoundError),
    )

    with pytest.raises(RuntimeError, match="MPI launcher not found: mpiexec"):
        mpi_cmd.is_open_mpi()


@pytest.mark.parametrize(
    "error",
    [
        OSError("launcher error"),
        subprocess.TimeoutExpired(cmd="mpiexec", timeout=10),
    ],
)
def test_is_open_mpi_returns_false_for_os_error_or_timeout(monkeypatch, error):
    monkeypatch.setattr(
        mpi_cmd.subprocess,
        "run",
        Mock(side_effect=error),
    )

    assert mpi_cmd.is_open_mpi() is False


def test_run_with_mpi_builds_command_and_cleans_up(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mpi_cmd, "is_open_mpi", lambda _: True)

    process = Mock()
    process.is_alive.return_value = False
    process_class = Mock(return_value=process)
    monkeypatch.setattr(mpi_cmd, "Process", process_class)

    run = Mock(
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
    )
    monkeypatch.setattr(mpi_cmd.subprocess, "run", run)
    monkeypatch.setattr(mpi_cmd.time, "time", Mock(side_effect=[10.0, 12.5]))

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")

    devices = [0, 1]
    hostfile = "/tmp/hosts"
    mpi_cmd.run_with_mpi(
        "my-lib", "config.yaml", "data.h5", "/data/run", devices, hostfile
    )

    log_file = f"ga_{os.getpid()}.log"
    script = (
        os.path.realpath(os.path.dirname(mpi_cmd.__file__)).replace(os.sep, "/")
        + "/reconstruction_ga.py"
    )
    expected_command = [
        "mpiexec",
        "--oversubscribe",
        "-np",
        "2",
        "--hostfile",
        hostfile,
        "python",
        script,
        "my-lib",
        "config.yaml",
        "data.h5",
        "/data/run",
        str(devices),
        log_file,
    ]

    run.assert_called_once_with(
        expected_command, check=True, capture_output=True, text=True
    )
    process_class.assert_called_once_with(
        target=mpi_cmd.show_log, args=(log_file,)
    )
    process.start.assert_called_once()
    process.terminate.assert_called_once()
    process.close.assert_called_once()
    assert not (tmp_path / log_file).exists()
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert "GA reconstruction took 2.5 seconds" in capsys.readouterr().out


def test_run_with_mpi_propagates_subprocess_error(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mpi_cmd, "is_open_mpi", lambda _: False)

    process = Mock()
    process_class = Mock(return_value=process)
    monkeypatch.setattr(mpi_cmd, "Process", process_class)

    error = subprocess.CalledProcessError(
        returncode=2, cmd=["mpiexec"], stderr="MPI failed"
    )
    monkeypatch.setattr(
        mpi_cmd.subprocess, "run", Mock(side_effect=error)
    )

    with pytest.raises(subprocess.CalledProcessError):
        mpi_cmd.run_with_mpi("lib", "conf", "data", "dir", [0])

    output = capsys.readouterr().out
    assert "Error Message: MPI failed" in output
    assert "Exit Code: 2" in output

    # The current implementation does not clean up the log or monitor process
    # when subprocess.run raises.
    (tmp_path / f"ga_{os.getpid()}.log").unlink(missing_ok=True)


def test_te_rec_builds_command_and_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mpi_cmd, "is_open_mpi", lambda _: True)
    run = Mock()
    monkeypatch.setattr(mpi_cmd.subprocess, "run", run)
    monkeypatch.setattr(mpi_cmd.time, "time", Mock(side_effect=[5.0, 6.0]))

    devices = [1, 3]
    mpi_cmd.te_rec("/experiment", "/tmp/hosts", devices)

    script = (
        os.path.realpath(os.path.dirname(mpi_cmd.__file__)).replace(os.sep, "/")
        + "/te_rec.py"
    )
    run.assert_called_once_with(
        [
            "mpiexec",
            "--oversubscribe",
            "-np",
            "2",
            "python",
            script,
            "/experiment",
            str(devices),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "chrono reconstruction took 1.0 seconds" in capsys.readouterr().out


def test_main_parses_arguments_and_calls_run_with_mpi(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mpi_cmd",
            "my-lib",
            "config.yaml",
            "data.h5",
            "/data/run",
            "[0, 2]",
            "/tmp/hosts",
        ],
    )
    run_with_mpi = Mock()
    monkeypatch.setattr(mpi_cmd, "run_with_mpi", run_with_mpi)

    mpi_cmd.main()

    run_with_mpi.assert_called_once_with(
        "my-lib", "config.yaml", "data.h5", "/data/run", [0, 2], "/tmp/hosts"
    )


def test_run_with_mpi_cleans_up_when_subprocess_fails(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mpi_cmd, "is_open_mpi", lambda _: False)

    # Simulate the monitor process started by run_with_mpi.
    process = Mock()
    process.is_alive.return_value = False
    monkeypatch.setattr(
        mpi_cmd, "Process", Mock(return_value=process)
    )

    error = subprocess.CalledProcessError(
        returncode=2,
        cmd=["mpiexec"],
        stderr="MPI failed",
    )
    monkeypatch.setattr(
        mpi_cmd.subprocess, "run", Mock(side_effect=error)
    )

    log_file = f"ga_{os.getpid()}.log"

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        mpi_cmd.run_with_mpi(
            "my-lib",
            "config.yaml",
            "data.h5",
            "/data/run",
            [0, 1],
        )

    assert exc_info.value is error
    output = capsys.readouterr().out
    assert "Error Message: MPI failed" in output
    assert "Exit Code: 2" in output

    process.terminate.assert_called_once()
    process.is_alive.assert_called_once()
    process.close.assert_called_once()
    assert not (tmp_path / log_file).exists()