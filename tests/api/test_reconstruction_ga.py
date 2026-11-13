# tests/test_reconstruction_ga.py
import os
import sys
from unittest.mock import Mock

import pytest

from cohere_core import reconstruction_GA as ga  # Adjust import if needed


def test_order_ranks_sorts_ascending_metrics_and_detects_better_previous_alpha():
    class FakeTracing:
        def __init__(self):
            self.generations = []

        def append_gen(self, generation):
            self.generations.append(generation)

    tracing = FakeTracing()
    metrics = {
        0: {"chi": 3.0},
        1: {"chi": 1.0},
        2: {"chi": 2.0},
    }

    ranked, best_metrics, last_alpha_best = ga.order_ranks(
        tracing,
        metrics,
        "chi",
        {"chi": 0.5},
    )

    assert ranked == [(1, 1.0), (2, 2.0), (0, 3.0)]
    assert best_metrics == {"chi": 1.0}
    assert last_alpha_best is True
    assert tracing.generations == [
        {
            1: (0, {"chi": 1.0}),
            2: (1, {"chi": 2.0}),
            0: (2, {"chi": 3.0}),
        }
    ]


def test_order_ranks_sorts_summed_phase_descending():
    class FakeTracing:
        def append_gen(self, generation):
            self.generation = generation

    tracing = FakeTracing()
    metrics = {
        0: {"summed_phase": 2.0},
        1: {"summed_phase": 8.0},
        2: {"summed_phase": 5.0},
    }

    ranked, best_metrics, last_alpha_best = ga.order_ranks(
        tracing,
        metrics,
        "summed_phase",
        {"summed_phase": 9.0},
    )

    assert ranked == [(1, 8.0), (2, 5.0), (0, 2.0)]
    assert best_metrics == {"summed_phase": 8.0}
    assert last_alpha_best is True


def test_cull_limits_list_length():
    assert ga.cull([3, 1, 2], 2) == [3, 1]
    assert ga.cull([3, 1], 5) == [3, 1]


def test_write_log_appends_line(tmp_path):
    logfile = tmp_path / "ga.log"

    ga.write_log(str(logfile), "starting generation 0")
    ga.write_log(str(logfile), "starting generation 1")

    assert logfile.read_text() == (
        "starting generation 0\n"
        "starting generation 1\n"
    )


def test_reconstruction_runs_single_rank_generation_and_saves_results(
    tmp_path, monkeypatch
):
    save_dir = tmp_path / "results"
    log_file = tmp_path / "ga.log"

    pars = {
        "save_dir": str(save_dir),
        "ga_generations": 1,
        "ga_metrics": ["chi"],
        "ga_reconstructions": [1],
        "init_guess": "random",
    }

    class FakeComm:
        def Get_size(self):
            return 1

        def Get_rank(self):
            return 0

        def Barrier(self):
            pass

    class FakeTracing:
        def __init__(self, *args):
            self.args = args
            self.generations = []
            self.saved_to = None
            self.rank_map = None

        def set_map(self, rank_map):
            self.rank_map = rank_map

        def append_gen(self, generation):
            self.generations.append(generation)

        def save(self, directory):
            self.saved_to = directory

    worker = Mock()
    worker.init_dev.return_value = 0
    worker.init_iter_loop.return_value = 0
    worker.iterate.return_value = 0
    worker.get_metric.return_value = {"chi": 0.25}
    worker.ds_image = "image-data"

    tracing = FakeTracing()

    monkeypatch.setattr(ga.MPI, "COMM_WORLD", FakeComm())
    monkeypatch.setattr(ga.dvut, "set_lib_from_pkg", Mock())
    monkeypatch.setattr(ga.ut, "read_config", lambda _: pars)
    monkeypatch.setattr(ga.gaut, "set_ga_defaults", lambda config: config)
    monkeypatch.setattr(ga.gaut, "Tracing", lambda *args: tracing)
    monkeypatch.setattr(ga.calc, "Rec", lambda *args: worker)

    ga.reconstruction(
        "np",
        str(tmp_path / "config_rec"),
        "data.tif",
        str(tmp_path),
        [4],
        str(log_file),
    )

    assert save_dir.is_dir()
    assert log_file.read_text() == "starting generation 0\n"
    worker.init_dev.assert_called_once_with(4)
    worker.init_iter_loop.assert_called_once_with(None, 0)
    worker.iterate.assert_called_once()
    worker.save_res.assert_called_once_with(str(save_dir))
    assert tracing.rank_map == {0: 0}
    assert tracing.generations == [{0: (0, {"chi": 0.25})}]
    assert tracing.saved_to == str(save_dir)


def test_reconstruction_main_parses_devices(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconstruction_GA",
            "np",
            "config_rec",
            "data.tif",
            "/experiment",
            "[0, 2]",
            "ga.log",
        ],
    )
    reconstruction = Mock()
    monkeypatch.setattr(ga, "reconstruction", reconstruction)

    ga.main()

    reconstruction.assert_called_once_with(
        "np",
        "config_rec",
        "data.tif",
        "/experiment",
        [0, 2],
        "ga.log",
    )