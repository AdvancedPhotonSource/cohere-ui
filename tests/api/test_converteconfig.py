import copy
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import cohere_ui.api.convertconfig as conv


@pytest.fixture(autouse=True)
def isolate_move_dict(monkeypatch):
    """Prevent convert()'s mutation of move_dict from leaking between tests."""
    monkeypatch.setattr(conv, "move_dict", copy.deepcopy(conv.move_dict))


# ---------------------------------------------------------------------------
# Version and key mappings
# ---------------------------------------------------------------------------

def test_get_version():
    assert conv.get_version() == 5


@pytest.mark.parametrize(
    "config_name, old_key, new_key",
    [
        (name, old, new)
        for name, mapping in conv.config_maps.items()
        for old, new in mapping.items()
    ],
)
def test_replace_keys(config_name, old_key, new_key):
    value = [1, 2, 3]
    params = {old_key: value, "unrelated_setting": True}

    result = conv.replace_keys(params, config_name)

    assert result is params
    assert result[new_key] is value
    assert result["unrelated_setting"] is True
    if old_key != new_key:
        assert old_key not in result


def test_replace_keys_overwrites_existing_destination():
    params = {"samples": 3, "reconstructions": 1}

    result = conv.replace_keys(params, "config_rec")

    assert result == {"reconstructions": 3}


def test_replace_keys_leaves_unmapped_keys_unchanged():
    params = {"custom_setting": 42}

    assert conv.replace_keys(params, "config_rec") == {
        "custom_setting": 42
    }


# ---------------------------------------------------------------------------
# Legacy dictionary conversions
# ---------------------------------------------------------------------------

def test_convert_dict_defaults_beamline_and_updates_version(capsys):
    configs = {"config": {}}

    result = conv.convert_dict(configs)

    assert result is configs
    assert result["config"] == {
        "beamline": "aps_34idc",
        "converter_ver": conv.get_version(),
    }
    assert 'setting beamline to "aps_34idc"' in capsys.readouterr().out


def test_convert_dict_preserves_existing_beamline():
    configs = {
        "config": {
            "beamline": "esrf_id01",
            "converter_ver": 1,
        }
    }

    conv.convert_dict(configs)

    assert configs["config"]["beamline"] == "esrf_id01"
    assert configs["config"]["converter_ver"] == conv.get_version()


@pytest.mark.parametrize(
    "detector, expected",
    [
        ("detector_name:", "detector_name"),
        ("detector_name", "detector_name"),
    ],
)
def test_convert_dict_detector_name(detector, expected):
    configs = {"config_instr": {"detector": detector}}

    conv.convert_dict(configs)

    assert configs["config_instr"]["detector"] == expected


@pytest.mark.parametrize(
    "aliens, expected",
    [
        (
            "((0, 0, 0), (2, 2, 2))",
            {
                "alien_alg": ' "block_aliens"',
                "aliens": "((0, 0, 0), (2, 2, 2))",
            },
        ),
        (
            "/data/alien_mask.npy",
            {
                "alien_alg": ' "alien_file"',
                "alien_file": "/data/alien_mask.npy",
            },
        ),
    ],
)
def test_convert_dict_legacy_aliens(aliens, expected):
    configs = {"config_data": {"aliens": aliens}}

    conv.convert_dict(configs)

    # These strings include the spaces/quotes used by the implementation.
    assert configs["config_data"] == expected


def test_convert_dict_preserves_current_alien_configuration():
    params = {
        "alien_alg": "block_aliens",
        "aliens": [[0, 0, 0, 2, 2, 2]],
    }
    original = copy.deepcopy(params)

    conv.convert_dict({"config_data": params})

    assert params == original


@pytest.mark.parametrize(
    "sequence, expected",
    [
        (
            [[3, ("ER", 20), ("HIO", 180)], [1, ("ER", 20)]],
            "3*(20*ER+180*HIO)+20*ER",
        ),
        ([[2, ("ER", 10)]], "2*(10*ER)"),
        ([[1, ("ER", 10)]], "10*ER"),
        ("20*ER+180*HIO", "20*ER+180*HIO"),
        ([], ""),
    ],
)
def test_convert_dict_algorithm_sequence(sequence, expected):
    configs = {"config_rec": {"algorithm_sequence": sequence}}

    conv.convert_dict(configs)

    assert configs["config_rec"]["algorithm_sequence"] == expected


@pytest.mark.parametrize(
    "interval, expected",
    [
        (20, "20"),
        ("20", "20"),
        ([0, 20], "20"),
        ("[0, 20]", "20"),
        ((0, 20), "20"),
    ],
)
def test_convert_dict_pc_interval(interval, expected):
    configs = {"config_rec": {"pc_interval": interval}}

    conv.convert_dict(configs)

    assert configs["config_rec"]["pc_interval"] == expected


def test_convert_dict_empty_input():
    assert conv.convert_dict({}) == {}


# ---------------------------------------------------------------------------
# File conversion
# ---------------------------------------------------------------------------

@pytest.fixture
def conversion_case(tmp_path, monkeypatch):
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()

    stored_configs = {}

    def add_config(name, params):
        path = conf_dir / name
        path.write_text(f"original contents of {name}\n")
        stored_configs[str(path)] = copy.deepcopy(params)
        return path

    add_config("config", {"beamline": "test_beamline"})

    def read_config(path):
        params = stored_configs.get(str(path))
        return copy.deepcopy(params)

    read = Mock(side_effect=read_config)
    write = Mock()
    monkeypatch.setattr(conv.ut, "read_config", read)
    monkeypatch.setattr(conv.ut, "write_config", write)
    monkeypatch.setattr(conv.os, "access", Mock(return_value=True))

    return SimpleNamespace(
        conf_dir=conf_dir,
        stored_configs=stored_configs,
        add_config=add_config,
        read=read,
        write=write,
    )


def saved_configs(case):
    """Get dictionaries supplied to write_config, indexed by filename."""
    return {
        call.args[1].replace("\\", "/").split("/")[-1]: call.args[0]
        for call in case.write.call_args_list
    }


def test_convert_legacy_files_and_creates_backups(conversion_case):
    case = conversion_case
    case.add_config(
        "config",
        {
            "specfile": "/data/scans.spec",
            "auto_data": True,
        },
    )
    case.add_config(
        "config_prep",
        {
            "separate_scans": True,
            "roi": [1, 2, 3, 4],
            "whitefield_filename": "/data/white.tif",
        },
    )
    case.add_config(
        "config_rec",
        {
            "samples": 2,
            "beta": 0.9,
            "algorithm_sequence": [[1, ("ER", 20)]],
            "pcdi_trigger": [0, 10],
        },
    )
    case.add_config(
        "config_disp",
        {"energy": 9.0, "crop": [0.5, 0.5, 0.5]},
    )
    case.add_config("config_data", {"amp_threshold": 5.0})

    assert conv.convert(str(case.conf_dir)) is None

    saved = saved_configs(case)

    assert saved["config"] == {
        "beamline": "aps_34idc",
        "converter_ver": 5,
        "separate_scans": True,
    }
    assert saved["config_instr"] == {
        "specfile": "/data/scans.spec",
        "whitefield_filename": "/data/white.tif",
        "det_roi": [1, 2, 3, 4],
        "energy": 9.0,
    }
    assert saved["config_prep"] == {"remove_outliers": True}
    assert saved["config_data"] == {
        "intensity_threshold": 5.0,
        "auto_intensity_threshold": True,
    }
    assert saved["config_rec"] == {
        "reconstructions": 2,
        "hio_beta": 0.9,
        "algorithm_sequence": "20*ER",
        "pc_interval": "10",
    }
    assert saved["config_disp"] == {
        "crop_fraction": [0.5, 0.5, 0.5],
        "crop_type": "fraction",
    }

    for original in case.stored_configs:
        original_path = case.conf_dir / original.split("/")[-1]
        backup = original_path.with_name(original_path.name + "_backup")
        assert backup.read_text() == original_path.read_text()


def test_convert_current_version_is_noop(conversion_case):
    case = conversion_case
    main_path = case.add_config(
        "config",
        {"converter_ver": conv.get_version()},
    )

    conv.convert(str(case.conf_dir))

    case.read.assert_called_once_with(str(main_path))
    case.write.assert_not_called()
    assert not list(case.conf_dir.glob("*_backup"))


def test_convert_versioned_config_skips_legacy_normalization(
    conversion_case
):
    case = conversion_case
    case.add_config(
        "config",
        {"beamline": "test_beamline", "converter_ver": 4},
    )
    sequence = [[1, ("ER", 20)]]
    case.add_config(
        "config_rec",
        {"samples": 2, "algorithm_sequence": sequence},
    )
    case.add_config("config_prep", {"roi": [0, 10, 0, 10]})

    conv.convert(str(case.conf_dir))

    saved = saved_configs(case)
    assert saved["config"]["converter_ver"] == 5
    assert saved["config_rec"]["reconstructions"] == 2
    # convert_dict() only runs when converter_ver was absent.
    assert saved["config_rec"]["algorithm_sequence"] == sequence
    assert saved["config_instr"]["det_roi"] == [0, 10, 0, 10]
    assert "roi" not in saved["config_prep"]


def test_convert_missing_directory(conversion_case, capsys):
    case = conversion_case
    missing_dir = case.conf_dir / "missing"

    conv.convert(str(missing_dir))

    assert "does not exist" in capsys.readouterr().out
    case.read.assert_not_called()
    case.write.assert_not_called()


def test_convert_missing_main_config(conversion_case, capsys):
    case = conversion_case
    case.read.return_value = None
    case.read.side_effect = None

    conv.convert(str(case.conf_dir))

    assert "main configuration file" in capsys.readouterr().out
    case.write.assert_not_called()


def test_convert_without_write_permission(conversion_case, monkeypatch):
    case = conversion_case
    monkeypatch.setattr(conv.os, "access", Mock(return_value=False))

    conv.convert(str(case.conf_dir))

    case.write.assert_not_called()
    assert not list(case.conf_dir.glob("*_backup"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_main_forwards_directory(monkeypatch, tmp_path):
    convert = Mock()
    monkeypatch.setattr(conv, "convert", convert)
    monkeypatch.setattr(
        sys,
        "argv",
        ["converterconfig.py", str(tmp_path)],
    )

    conv.main()

    convert.assert_called_once_with(str(tmp_path))


def test_main_requires_directory(monkeypatch):
    convert = Mock()
    monkeypatch.setattr(conv, "convert", convert)
    monkeypatch.setattr(sys, "argv", ["converterconfig.py"])

    with pytest.raises(SystemExit) as exc_info:
        conv.main()

    assert exc_info.value.code == 2
    convert.assert_not_called()