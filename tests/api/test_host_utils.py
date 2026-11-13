# tests/test_host_utils.py
import sys
from unittest.mock import Mock

from cohere_ui.api import host_utils


def set_cli_args(monkeypatch, devices, run_size):
    monkeypatch.setattr(
        sys,
        "argv",
        ["host_utils", repr(devices), repr(run_size)],
    )


def test_main_uses_full_hostname_when_configured(monkeypatch, capsys):
    devices = {
        "gpu-node.example.org": {"gpu": 2},
        "gpu-node": {"gpu": 9},
    }
    run_size = 1024
    get_avail_gpu_runs = Mock(return_value={"0": 2})

    set_cli_args(monkeypatch, devices, run_size)
    monkeypatch.setattr(
        host_utils.socket, "gethostname", lambda: "gpu-node.example.org"
    )
    monkeypatch.setattr(
        host_utils.balancer, "get_avail_gpu_runs", get_avail_gpu_runs
    )

    host_utils.main()

    get_avail_gpu_runs.assert_called_once_with(
        devices["gpu-node.example.org"], run_size
    )
    assert capsys.readouterr().out == (
        "['gpu-node.example.org', {'0': 2}]\n"
    )


def test_main_falls_back_to_short_hostname(monkeypatch, capsys):
    devices = {"gpu-node": {"gpu": 4}}
    run_size = 2048
    get_avail_gpu_runs = Mock(return_value={"1": 3})

    set_cli_args(monkeypatch, devices, run_size)
    monkeypatch.setattr(
        host_utils.socket, "gethostname", lambda: "gpu-node.example.org"
    )
    monkeypatch.setattr(
        host_utils.balancer, "get_avail_gpu_runs", get_avail_gpu_runs
    )

    host_utils.main()

    get_avail_gpu_runs.assert_called_once_with(devices["gpu-node"], run_size)
    assert capsys.readouterr().out == "['gpu-node', {'1': 3}]\n"


def test_main_does_nothing_when_hostname_is_not_configured(monkeypatch, capsys):
    devices = {"another-node": {"gpu": 1}}
    get_avail_gpu_runs = Mock()

    set_cli_args(monkeypatch, devices, 1024)
    monkeypatch.setattr(
        host_utils.socket, "gethostname", lambda: "gpu-node.example.org"
    )
    monkeypatch.setattr(
        host_utils.balancer, "get_avail_gpu_runs", get_avail_gpu_runs
    )

    host_utils.main()

    get_avail_gpu_runs.assert_not_called()
    assert capsys.readouterr().out == ""
