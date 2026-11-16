# tests/test_reconstruction_populous.py
import os
import sys
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from cohere_ui.api import reconstruction_populous as rp


def test_devices_assign_gpu_to_current_thread():
    devices = rp.Devices([2, 5])

    devices.assign_gpu()
    assert threading.current_thread().gpu == 2

    devices.assign_gpu()
    assert threading.current_thread().gpu == 5


def test_single_rec_process_saves_successful_reconstruction(monkeypatch):
    thread = threading.current_thread()
    thread.gpu = 3

    worker = Mock()
    worker.init_dev.return_value = 0
    worker.init_iter_loop.return_value = 0
    worker.iterate.return_value = 0
    worker.get_metric.return_value = {"chi": 0.25}

    monkeypatch.setattr(rp.dvut, "set_lib_from_pkg", Mock())
    monkeypatch.setattr(rp.ut, "get_lib", Mock(return_value=np))
    monkeypatch.setattr(rp.calc, "Rec", Mock(return_value=worker))

    result = rp.single_rec_process(
        "np",
        {"some": "parameters"},
        "data.tif",
        0,
        None,
        (None, "/output/0"),
    )

    assert result == [{"chi": 0.25}, "/output/0"]
    worker.init_dev.assert_called_once_with(3)
    worker.init_iter_loop.assert_called_once_with(None, 0)
    worker.iterate.assert_called_once()
    worker.save_res.assert_called_once_with("/output/0")


def test_single_rec_process_returns_none_metric_if_device_init_fails(
    monkeypatch, capsys
):
    threading.current_thread().gpu = 4

    worker = Mock()
    worker.init_dev.return_value = -1

    monkeypatch.setattr(rp.dvut, "set_lib_from_pkg", Mock())
    monkeypatch.setattr(rp.ut, "get_lib", Mock(return_value=np))
    monkeypatch.setattr(rp.calc, "Rec", Mock(return_value=worker))

    result = rp.single_rec_process(
        "np", {}, "data.tif", 0, None, (None, "/output/0")
    )

    assert result == [None, "/output/0"]
    worker.init_iter_loop.assert_not_called()
    worker.iterate.assert_not_called()
    worker.save_res.assert_not_called()
    assert "device not initialized" in capsys.readouterr().out


def test_single_rec_process_breeds_from_alpha_for_later_generation(
    monkeypatch,
):
    threading.current_thread().gpu = 1
    worker = Mock()
    worker.dev = "worker-device"
    worker.init_dev.return_value = 0
    worker.init_iter_loop.return_value = 0
    worker.iterate.return_value = 0
    worker.get_metric.return_value = {"chi": 0.1}

    alpha = np.ones((2, 2))
    loaded = Mock(return_value=alpha)
    breed = Mock(return_value=np.full((2, 2), 2.0))
    shrink_wrap = Mock(return_value=np.ones((2, 2), dtype=bool))

    monkeypatch.setattr(rp.dvut, "set_lib_from_pkg", Mock())
    monkeypatch.setattr(rp.ut, "get_lib", Mock(return_value=np))
    monkeypatch.setattr(rp.ut, "join", os.path.join)
    monkeypatch.setattr(rp.calc, "Rec", Mock(return_value=worker))
    monkeypatch.setattr(rp.dvut, "breed", breed)
    monkeypatch.setattr(rp.dvut, "shrink_wrap", shrink_wrap)
    monkeypatch.setattr(np, "load", loaded)

    pars = {
        "ga_sw_thresholds": [0.1, 0.2],
        "ga_sw_gauss_sigmas": [1.0, 2.0],
        "ga_breed_modes": ["mode0", "mode1"],
    }

    result = rp.single_rec_process(
        "np",
        pars,
        "data.tif",
        1,
        "/alpha",
        ("/previous", "/output/1"),
    )

    assert result == [{"chi": 0.1}, "/output/1"]
    loaded.assert_called_once_with(
        os.path.join("/alpha", "image.npy"), device="worker-device"
    )
    breed.assert_called_once_with("mode1", alpha, worker.ds_image)
    shrink_wrap.assert_called_once_with(worker.ds_image, 0.2, 2.0)
    worker.save_res.assert_called_once_with("/output/1")


class FakePool:
    """A synchronous stand-in for multiprocessing.pool.ThreadPool."""

    def __init__(self, processes, initializer=None, initargs=()):
        self.processes = processes
        self.initializer = initializer
        self.initargs = initargs

    def __enter__(self):
        if self.initializer is not None:
            self.initializer(*self.initargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def map_async(self, func, iterable, callback=None):
        results = [func(item) for item in iterable]
        if callback is not None:
            callback(results)

    def close(self):
        pass

    def join(self):
        pass

    def terminate(self):
        pass


def test_multi_rec_collects_results_and_puts_them_on_queue(monkeypatch):
    monkeypatch.setattr(rp, "Pool", FakePool)
    monkeypatch.setattr(rp.ut, "join", os.path.join)

    calls = []

    def fake_single_rec(pkg, pars, datafile, gen, alpha_dir, rec_attrs):
        calls.append((pkg, rec_attrs))
        return [{"metric": rec_attrs[1]}, rec_attrs[1]]

    monkeypatch.setattr(rp, "single_rec_process", fake_single_rec)
    queue = Mock()

    rp.multi_rec(
        "np",
        "/results",
        [0, 1],
        2,
        {"parameter": True},
        "data.tif",
        [None, None],
        q=queue,
    )

    expected = [
        [{"metric": "/results/0"}, "/results/0"],
        [{"metric": "/results/1"}, "/results/1"],
    ]
    queue.put.assert_called_once_with(expected)
    assert calls == [
        ("np", (None, "/results/0")),
        ("np", (None, "/results/1")),
    ]


def test_multi_rec_reports_failure_when_no_results(monkeypatch):
    monkeypatch.setattr(rp, "Pool", FakePool)
    queue = Mock()

    rp.multi_rec(
        "np",
        "/results",
        [0],
        0,
        {},
        "data.tif",
        [],
        q=queue,
    )

    queue.put.assert_called_once_with("failed")


def test_reconstruction_reads_config_and_starts_reconstructions(monkeypatch):
    pars = {"reconstructions": 2}
    read_config = Mock(return_value=pars)
    multi_rec = Mock()

    monkeypatch.setattr(rp.ut, "read_config", read_config)
    monkeypatch.setattr(rp.ut, "join", os.path.join)
    monkeypatch.setattr(rp, "multi_rec", multi_rec)

    rp.reconstruction(
        "np", "/experiment/conf/config_rec", "data.tif", "/experiment", [0, 1]
    )

    read_config.assert_called_once_with("/experiment/conf/config_rec")
    multi_rec.assert_called_once_with(
        "np",
        os.path.join("/experiment", "results_phasing"),
        [0, 1],
        2,
        pars,
        "data.tif",
        [None, None],
    )


def test_main_parses_device_list(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconstruction_populous",
            "np",
            "config_rec",
            "data.tif",
            "/experiment",
            "[0, 2]",
        ],
    )
    reconstruction = Mock()
    monkeypatch.setattr(rp, "reconstruction", reconstruction)

    rp.main()

    reconstruction.assert_called_once_with(
        "np", "config_rec", "data.tif", "/experiment", [0, 2]
    )

# pytest -q tests/test_reconstruction_populous.py
# The FakePool runs tasks synchronously, avoiding real threads while still exercising multi_rec’s mapping and callback behavior. Note that the current Devices.assign_gpu implementation does not wrap around if the pool initializes more workers than there are devices; these tests use matching worker/device counts.