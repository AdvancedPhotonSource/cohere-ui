import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import cohere_ui.api.balancer as balancer


# ---------------------------------------------------------------------------
# CPU and memory-based process estimates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "available_memory, expected",
    [
        (10_000, 8),  # CPU-limited: memory allows 50 processes.
        (900, 4),     # Memory-limited: 4.5 processes rounds down.
        (1_600, 8),   # Exactly enough memory for all CPUs.
        (100, 0),     # Insufficient memory for even one process.
    ],
)
def test_estimate_no_proc(monkeypatch, available_memory, expected):
    monkeypatch.setattr(balancer, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        balancer.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=available_memory),
    )

    assert balancer.estimate_no_proc(
        arr_size=100, factor=2
    ) == expected


# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------

@pytest.fixture
def gpu_case(monkeypatch):
    """Mock GPUtil, including its imports inside balancer functions."""
    gpus = [
        SimpleNamespace(id=0, memoryFree=1_000),
        SimpleNamespace(id=1, memoryFree=2_500),
        SimpleNamespace(id=2, memoryFree=400),
    ]
    get_gpus = Mock(return_value=gpus)

    monkeypatch.setitem(
        sys.modules,
        "GPUtil",
        SimpleNamespace(getGPUs=get_gpus),
    )

    # Restore this environment variable automatically after each test.
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)

    return SimpleNamespace(gpus=gpus, get_gpus=get_gpus)


@pytest.mark.parametrize(
    "devices, expected",
    [
        ("all", {0: 2, 1: 5, 2: 0}),
        ([0, 2], {0: 2, 2: 0}),
        ([1], {1: 5}),
        ([], {}),
        ([99], {}),
    ],
)
def test_get_avail_gpu_runs(gpu_case, devices, expected):
    result = balancer.get_avail_gpu_runs(devices, run_mem=500)

    assert result == expected
    gpu_case.get_gpus.assert_called_once_with()
    assert balancer.os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


@pytest.mark.parametrize(
    "ids, expected",
    [
        ("all", 1),
        ([0, 2], 0),
        ([2], 2),
        ([], -1),
        ([99], -1),
    ],
)
def test_get_one_dev(gpu_case, ids, expected):
    assert balancer.get_one_dev(ids) == expected

    gpu_case.get_gpus.assert_called_once_with()
    assert balancer.os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_get_one_dev_cluster_uses_local_host(gpu_case, monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "local_host")

    devices = {
        "local_host": [0, 2],
        "remote_host": [1],
    }

    assert balancer.get_one_dev(devices) == 0


def test_get_one_dev_keeps_first_device_on_memory_tie(gpu_case):
    gpu_case.gpus[0].memoryFree = 2_500

    assert balancer.get_one_dev("all") == 0


def test_get_one_dev_returns_minus_one_when_no_memory(gpu_case):
    for gpu in gpu_case.gpus:
        gpu.memoryFree = 0

    assert balancer.get_one_dev("all") == -1


# ---------------------------------------------------------------------------
# Cluster GPU discovery
# ---------------------------------------------------------------------------

def test_get_avail_hosts_gpu_runs(monkeypatch):
    devices = {
        "host_a": [0, 1],
        "host_b": "all",
    }
    run = Mock(
        return_value=SimpleNamespace(
            stdout=(
                "('host_a', {0: 2, 1: 1})\n"
                "('host_b', {2: 4})\n"
            )
        )
    )
    monkeypatch.setattr(balancer.subprocess, "run", run)

    result = balancer.get_avail_hosts_gpu_runs(devices, run_mem=500)

    assert result == {
        "host_a": {0: 2, 1: 1},
        "host_b": {2: 4},
    }

    script = balancer.ut.join(
        balancer.os.path.realpath(
            balancer.os.path.dirname(balancer.__file__)
        ),
        "host_utils.py",
    )
    run.assert_called_once_with(
        [
            "mpiexec",
            "-n",
            "2",
            "--host",
            "host_a,host_b",
            "python",
            script,
            str(devices),
            "500",
        ],
        stdout=balancer.subprocess.PIPE,
        text=True,
    )


def test_get_avail_hosts_gpu_runs_empty_output(monkeypatch):
    monkeypatch.setattr(
        balancer.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(stdout="")),
    )

    assert balancer.get_avail_hosts_gpu_runs(
        {"host_a": "all"}, 500
    ) == {}


# ---------------------------------------------------------------------------
# Load balancing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "availability, requested, expected_load, expected_count",
    [
        ({0: 2, 1: 1}, 5, {0: 2, 1: 1}, 3),
        ({0: 2, 1: 1}, 3, {0: 2, 1: 1}, 3),
        ({0: 6, 1: 3}, 6, {0: 4, 1: 2}, 6),
        ({0: 4, 1: 2, 2: 0}, 4, {0: 3, 1: 1}, 4),
        ({0: 3, 1: 3, 2: 3}, 2, {0: 1, 1: 1}, 2),
        ({0: 0, 1: 0}, 2, {0: 0, 1: 0}, 0),
    ],
)
def test_get_balanced_load(
    availability, requested, expected_load, expected_count
):
    # The implementation mutates availability when balancing is required.
    original = availability.copy()

    load, count = balancer.get_balanced_load(
        availability.copy(), requested
    )

    assert load == expected_load
    assert count == expected_count
    assert sum(load.values()) == count

    for device, allocated in load.items():
        assert 0 <= allocated <= original[device]


# ---------------------------------------------------------------------------
# Device allocation and hostfile generation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "requested, expected_devices, expected_count",
    [
        (4, [0, 0, 0, 1], 4),
        (10, [0, 0, 0, 0, 1, 1], 6),
    ],
)
def test_get_gpu_use_local(
    monkeypatch, requested, expected_devices, expected_count
):
    get_available = Mock(return_value={0: 4, 1: 2})
    monkeypatch.setattr(
        balancer, "get_avail_gpu_runs", get_available
    )

    result = balancer.get_gpu_use(
        devices=[0, 1],
        no_jobs=requested,
        job_size=500,
    )

    assert result == (expected_devices, expected_count, None)
    get_available.assert_called_once_with([0, 1], 500)


def test_get_gpu_use_local_all_devices(monkeypatch):
    get_available = Mock(return_value={2: 2})
    monkeypatch.setattr(
        balancer, "get_avail_gpu_runs", get_available
    )

    assert balancer.get_gpu_use("all", 2, 500) == (
        [2, 2],
        2,
        None,
    )
    get_available.assert_called_once_with("all", 500)


def test_get_gpu_use_cluster_creates_hostfile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(balancer.os, "getpid", lambda: 1234)

    devices = {
        "host_a": [0, 1],
        "host_b": [2],
    }
    get_available = Mock(
        return_value={
            "host_a": {0: 3, 1: 1},
            "host_b": {2: 2},
        }
    )
    monkeypatch.setattr(
        balancer,
        "get_avail_hosts_gpu_runs",
        get_available,
    )

    picked_devices, allocated_jobs, hostfile = balancer.get_gpu_use(
        devices, no_jobs=6, job_size=500
    )

    assert picked_devices == [[0, 0, 0, 1], [2, 2]]
    assert allocated_jobs == 6
    assert hostfile == "hostfile_1234"
    assert Path(hostfile).read_text().splitlines() == [
        "host_a:4",
        "host_b:2",
    ]
    get_available.assert_called_once_with(devices, 500)


def test_get_gpu_use_cluster_balances_across_hosts(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(balancer.os, "getpid", lambda: 5678)

    monkeypatch.setattr(
        balancer,
        "get_avail_hosts_gpu_runs",
        Mock(return_value={
            "host_a": {0: 6},
            "host_b": {1: 3},
        }),
    )

    picked_devices, allocated_jobs, hostfile = balancer.get_gpu_use(
        {"host_a": [0], "host_b": [1]},
        no_jobs=6,
        job_size=500,
    )

    assert picked_devices == [[0, 0, 0, 0], [1, 1]]
    assert allocated_jobs == 6
    assert Path(hostfile).read_text().splitlines() == [
        "host_a:4",
        "host_b:2",
    ]

@pytest.mark.parametrize(
    "availability, requested",
    [
        ({}, 3),
        ({0: 4, 1: 2}, 0),
    ],
)
def test_get_balanced_load_empty_or_zero_request(
    availability, requested
):
    assert balancer.get_balanced_load(
        availability, requested
    ) == ({}, 0)


def test_get_gpu_use_no_gpus(monkeypatch):
    monkeypatch.setattr(
        balancer,
        "get_avail_gpu_runs",
        Mock(return_value={}),
    )

    assert balancer.get_gpu_use([0], 3, 500) == ([], 0, None)