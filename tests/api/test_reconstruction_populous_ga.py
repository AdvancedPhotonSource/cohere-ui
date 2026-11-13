# tests/test_reconstruction_populous_ga.py
import os
import shutil
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from cohere_ui.api import reconstruction_populous_ga as ga  # Adjust if needed


class FakeTracing:
    def __init__(self, *args, init_dirs=None):
        self.args = args
        self.init_dirs = init_dirs or []
        self.map = {}
        self.generations = []
        self.saved_to = None

    def append_gen(self, generation):
        self.generations.append(generation)

    def set_map(self, mapping):
        self.map = dict(mapping)

    def save(self, save_dir):
        self.saved_to = save_dir


def test_order_dirs_sorts_ascending_and_renumbers_directories(tmp_path):
    gen_dir = tmp_path / "g_0"
    gen_dir.mkdir()

    dirs = []
    for index, marker in enumerate(("three", "one", "two")):
        result_dir = gen_dir / str(index)
        result_dir.mkdir()
        (result_dir / "marker.txt").write_text(marker)
        dirs.append(str(result_dir))

    tracing = FakeTracing()
    tracing.map = {0: 100, 1: 200, 2: 300}
    evals = [{"chi": 3.0}, {"chi": 1.0}, {"chi": 2.0}]

    rank_dirs, best_metrics = ga.order_dirs(
        tracing, dirs, evals, "chi"
    )

    assert rank_dirs == [
        str(gen_dir / "0"),
        str(gen_dir / "1"),
        str(gen_dir / "2"),
    ]
    assert best_metrics == {"chi": 1.0}
    assert tracing.generations == [
        {
            1: (0, {"chi": 1.0}),
            2: (1, {"chi": 2.0}),
            0: (2, {"chi": 3.0}),
        }
    ]
    assert tracing.map == {0: 200, 1: 300, 2: 100}

    # The contents follow the ranking: original directory 1 is now rank 0.
    assert (gen_dir / "0" / "marker.txt").read_text() == "one"
    assert (gen_dir / "1" / "marker.txt").read_text() == "two"
    assert (gen_dir / "2" / "marker.txt").read_text() == "three"


def test_order_dirs_ranks_summed_phase_descending(tmp_path):
    gen_dir = tmp_path / "g_0"
    gen_dir.mkdir()
    dirs = []

    for index in range(3):
        result_dir = gen_dir / str(index)
        result_dir.mkdir()
        dirs.append(str(result_dir))

    tracing = FakeTracing()
    tracing.map = {0: 0, 1: 1, 2: 2}
    evals = [
        {"summed_phase": 2.0},
        {"summed_phase": 8.0},
        {"summed_phase": 5.0},
    ]

    rank_dirs, best_metrics = ga.order_dirs(
        tracing, dirs, evals, "summed_phase"
    )

    assert rank_dirs == [
        str(gen_dir / "1"),
        str(gen_dir / "2"),
        str(gen_dir / "0"),
    ]
    assert best_metrics == {"summed_phase": 8.0}


def test_cull_limits_list_length():
    assert ga.cull(["a", "b", "c"], 2) == ["a", "b"]
    assert ga.cull(["a", "b"], 5) == ["a", "b"]


@pytest.mark.parametrize(
    ("config_updates", "expected_message"),
    [
        ({"ga_generations": 1}, "number of generations"),
        ({"reconstructions": 1}, "single reconstruction"),
    ],
)
def test_reconstruction_returns_early_for_invalid_configuration(
    monkeypatch, capsys, config_updates, expected_message
):
    pars = {
        "ga_generations": 2,
        "reconstructions": 2,
        "ga_fast": False,
        **config_updates,
    }
    monkeypatch.setattr(ga.ut, "read_config", lambda _: pars)
    monkeypatch.setattr(ga.gaut, "set_ga_defaults", lambda config: config)

    # The function should return before creating directories, a queue, or a process.
    monkeypatch.setattr(
        ga, "Process", Mock(side_effect=AssertionError("Process should not start"))
    )

    ga.reconstruction("np", "config_rec", "data.tif", "/experiment", [0, 1])

    assert expected_message in capsys.readouterr().out


def test_reconstruction_rejects_culling_that_leaves_fewer_than_two(
    monkeypatch,
):
    pars = {
        "ga_generations": 2,
        "reconstructions": 3,
        "ga_fast": False,
        "ga_cullings": [1, 1],
    }
    monkeypatch.setattr(ga.ut, "read_config", lambda _: pars)
    monkeypatch.setattr(ga.gaut, "set_ga_defaults", lambda config: config)

    with pytest.raises(ValueError, match="At least two reconstructions"):
        ga.reconstruction("np", "config_rec", "data.tif", "/experiment", [0, 1])


def test_reconstruction_runs_two_generations_and_moves_best_result(
    tmp_path, monkeypatch
):
    save_dir = tmp_path / "results"
    pars = {
        "ga_generations": 2,
        "reconstructions": 2,
        "ga_fast": False,
        "ga_metrics": ["chi", "chi"],
        "ga_reconstructions": [2, 2],
        "init_guess": "random",
        "ga_breed_modes": ["mode0", "mode1"],
    }
    tracing = FakeTracing(init_dirs=["previous-0", "previous-1"])

    class FakeQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

        def get(self):
            return self.items.pop(0)

    class FakeProcess:
        def __init__(self, target, args):
            self.target = target
            self.args = args

        def start(self):
            # multi_rec's arguments:
            # pkg, generation_dir, devices, reconstructions, pars, datafile,
            # prev_dirs, generation, alpha_dir, queue
            gen_dir = Path(self.args[1])
            generation = self.args[7]
            queue = self.args[9]

            result_dir = gen_dir / "0"
            result_dir.mkdir(parents=True)
            image_path = result_dir / "image.npy"
            np.save(image_path, np.array([generation]))

            queue.put([
                [{"chi": float(2 - generation)}, str(result_dir)]
            ])

        def join(self):
            pass

    monkeypatch.setattr(ga.ut, "read_config", lambda _: pars)
    monkeypatch.setattr(ga.ut, "join", os.path.join)
    monkeypatch.setattr(ga.gaut, "set_ga_defaults", lambda config: config)
    monkeypatch.setattr(ga.gaut, "Tracing", lambda *args: tracing)
    monkeypatch.setattr(ga, "Queue", FakeQueue)
    monkeypatch.setattr(ga, "Process", FakeProcess)

    def order_dirs_without_moving(tracing_obj, dirs, evals, metric_type):
        # The test focuses on reconstruction orchestration. Keep generation
        # directories in place and return the single result directory.
        return dirs, evals[0]

    monkeypatch.setattr(ga, "order_dirs", order_dirs_without_moving)

    ga.reconstruction(
        "np", "config_rec", "data.tif", str(tmp_path), [0, 1]
    )

    # The best result from the final generation is moved to save_dir.
    assert (save_dir / "image.npy").is_file()
    np.testing.assert_array_equal(
        np.load(save_dir / "image.npy"), np.array([1])
    )

    # Temporary generation and alpha directories are cleaned up.
    assert not (save_dir / "g_0").exists()
    assert not (save_dir / "g_1").exists()
    assert not (save_dir / "alpha").exists()

    assert tracing.saved_to == str(save_dir)