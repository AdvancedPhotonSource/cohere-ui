import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest

import cohere_ui.api.common as common
import cohere_core.utilities.dvc_utils as dvut


# ---------------------------------------------------------------------------
# preserve_devlib
# ---------------------------------------------------------------------------

def test_preserve_devlib_restores_existing_library(monkeypatch):
    original = object()
    replacement = object()
    monkeypatch.setattr(dvut, "devlib", original, raising=False)

    with common.preserve_devlib():
        dvut.devlib = replacement
        assert dvut.devlib is replacement

    assert dvut.devlib is original


def test_preserve_devlib_removes_library_if_originally_unset(monkeypatch):
    monkeypatch.delattr(dvut, "devlib", raising=False)

    with common.preserve_devlib():
        # Use monkeypatch so cleanup also occurs if the test fails.
        monkeypatch.setattr(dvut, "devlib", object(), raising=False)

    assert not hasattr(dvut, "devlib")


def test_preserve_devlib_restores_after_exception(monkeypatch):
    original = object()
    monkeypatch.setattr(dvut, "devlib", original, raising=False)

    with pytest.raises(RuntimeError, match="operation failed"):
        with common.preserve_devlib():
            dvut.devlib = object()
            raise RuntimeError("operation failed")

    assert dvut.devlib is original


def test_preserve_devlib_as_decorator(monkeypatch):
    original = object()
    monkeypatch.setattr(dvut, "devlib", original, raising=False)

    @common.preserve_devlib()
    def operation():
        dvut.devlib = object()
        return "done"

    assert operation() == "done"
    assert dvut.devlib is original


def test_preserve_devlib_nested_contexts(monkeypatch):
    original = object()
    outer = object()
    inner = object()
    monkeypatch.setattr(dvut, "devlib", original, raising=False)

    with common.preserve_devlib():
        dvut.devlib = outer
        with common.preserve_devlib():
            dvut.devlib = inner
        assert dvut.devlib is outer

    assert dvut.devlib is original


# ---------------------------------------------------------------------------
# get_config_maps
# ---------------------------------------------------------------------------

@pytest.fixture
def config_case(tmp_path, monkeypatch):
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()

    main_config = {
        "beamline": "test_beamline",
        "converter_ver": 2,
    }
    file_maps = {}

    def add_config(name, values):
        path = conf_dir / name
        path.touch()
        file_maps[str(path)] = values
        return str(path)

    main_path = add_config("config", main_config)

    read_config = Mock(side_effect=lambda path: file_maps[str(path)])
    verify_types = Mock(return_value="")
    verify_params = Mock(return_value="")

    monkeypatch.setattr(common.ut, "read_config", read_config)
    monkeypatch.setattr(common.ut, "verify_types", verify_types)
    monkeypatch.setattr(common.ut, "verify_params", verify_params)

    get_version = Mock(return_value=2)
    convert = Mock()
    monkeypatch.setattr(common.conv, "get_version", get_version)
    monkeypatch.setattr(common.conv, "convert", convert)

    schema_modules = {
        "config": common.exp_schema,
        "config_prep": common.prep_schema,
        "config_disp": common.post_schema,
        "config_rec": common.recon_schema,
        "config_data": common.st_prep_schema,
        "config_mp": common.mp_schema,
    }
    schemas = {}
    for name, module in schema_modules.items():
        schemas[name] = object()
        monkeypatch.setattr(
            module,
            "get_config_schema",
            Mock(return_value=schemas[name]),
        )

    return SimpleNamespace(
        experiment_dir=str(tmp_path),
        conf_dir=str(conf_dir),
        main_path=main_path,
        main_config=main_config,
        file_maps=file_maps,
        add_config=add_config,
        read_config=read_config,
        verify_types=verify_types,
        verify_params=verify_params,
        get_version=get_version,
        convert=convert,
        schemas=schemas,
    )


def test_get_config_maps_loads_and_verifies_configs(config_case):
    case = config_case
    data_config = {"intensity_threshold": 5.0}
    rec_config = {"processing": "np"}

    data_path = case.add_config("config_data", data_config)
    rec_path = case.add_config("config_rec", rec_config)

    maps, converted, errors = common.get_config_maps(
        case.experiment_dir, ["config_data", "config_rec"]
    )

    assert maps == {
        "config": case.main_config,
        "config_data": data_config,
        "config_rec": rec_config,
    }
    assert converted is False
    assert errors == {
        "config": "",
        "config_data": "",
        "config_rec": "",
    }

    assert case.read_config.call_args_list == [
        call(case.main_path),
        call(data_path),
        call(rec_path),
    ]
    assert case.verify_types.call_args_list == [
        call(case.schemas["config"], case.main_config),
        call(case.schemas["config_data"], data_config),
        call(case.schemas["config_rec"], rec_config),
    ]
    assert case.verify_params.call_args_list == [
        call("config", case.main_config),
        call("config_data", data_config),
        call("config_rec", rec_config),
    ]
    case.convert.assert_not_called()


def test_get_config_maps_missing_main_config(config_case):
    case = config_case
    common.os.remove(case.main_path)

    with pytest.raises(ValueError, match="no main config"):
        common.get_config_maps(case.experiment_dir, [])

    case.read_config.assert_not_called()
    case.convert.assert_not_called()


def test_get_config_maps_skips_missing_optional_config(config_case):
    maps, converted, errors = common.get_config_maps(
        config_case.experiment_dir, ["config_data"]
    )

    assert maps == {"config": config_case.main_config}
    assert errors == {"config": ""}
    assert converted is False


@pytest.mark.parametrize(
    "stored_version, current_version, expected_conversion",
    [
        (None, 2, True),  # No converter_ver in the main config.
        (1, 2, True),
        (2, 2, False),
        (3, 2, False),
        (2, None, True),
    ],
)
def test_get_config_maps_conversion(
    config_case, stored_version, current_version, expected_conversion
):
    case = config_case
    if stored_version is None:
        case.main_config.pop("converter_ver")
    else:
        case.main_config["converter_ver"] = stored_version

    case.get_version.return_value = current_version

    updated_config = {
        "beamline": "test_beamline",
        "converter_ver": 99,
    }

    def convert_config(conf_dir):
        case.file_maps[case.main_path] = updated_config

    case.convert.side_effect = convert_config

    maps, converted, _ = common.get_config_maps(
        case.experiment_dir, []
    )

    assert converted is expected_conversion

    if expected_conversion:
        case.convert.assert_called_once_with(case.conf_dir)
        assert maps["config"] is updated_config
        assert case.read_config.call_count == 2
    else:
        case.convert.assert_not_called()
        assert maps["config"] is case.main_config
        case.read_config.assert_called_once_with(case.main_path)


def test_get_config_maps_collects_validation_errors(config_case):
    case = config_case
    case.add_config("config_data", {})
    case.verify_types.side_effect = ["main type error", "data type error"]
    case.verify_params.side_effect = ["main parameter error", ""]

    _, _, errors = common.get_config_maps(
        case.experiment_dir, ["config_data"]
    )

    assert errors == {
        "config": "main type error\nmain parameter error\n",
        "config_data": "data type error\n",
    }


def test_get_config_maps_no_verify_still_collects_errors(config_case):
    case = config_case
    case.verify_types.return_value = "type error"

    _, _, errors = common.get_config_maps(
        case.experiment_dir, [], no_verify=True
    )

    # no_verify is handled by callers, not by get_config_maps itself.
    assert errors["config"] == "type error\n"
    case.verify_types.assert_called_once()


def test_get_config_maps_alternate_reconstruction_id(config_case):
    case = config_case
    alternate_config = {"processing": "cp"}
    alternate_path = case.add_config("config_rec_trial", alternate_config)
    case.add_config("config_rec", {"processing": "np"})

    maps, _, errors = common.get_config_maps(
        case.experiment_dir,
        ["config_rec"],
        rec_id="trial",
    )

    assert maps["config_rec"] is alternate_config
    assert errors["config_rec"] == ""
    assert case.read_config.call_args_list[-1] == call(alternate_path)
    case.verify_params.assert_called_with(
        "config_rec", alternate_config
    )


@pytest.mark.parametrize("multipeak", [False, True])
def test_get_config_maps_multipeak_handling(config_case, multipeak):
    case = config_case
    case.main_config["multipeak"] = multipeak
    mp_config = {"peaks": [1, 2]}
    case.add_config("config_mp", mp_config)
    requested = ["config_mp"]

    maps, _, errors = common.get_config_maps(
        case.experiment_dir, requested
    )

    if multipeak:
        assert maps["config_mp"] is mp_config
        assert errors["config_mp"] == ""
        assert requested == ["config_mp"]
    else:
        assert "config_mp" not in maps
        assert "config_mp" not in errors
        # The current implementation mutates the caller's list.
        assert requested == []


def test_get_config_maps_instrument_schema(config_case, monkeypatch):
    case = config_case
    instrument_config = {"detector": "test_detector"}
    case.add_config("config_instr", instrument_config)

    schema = object()
    schema_module = SimpleNamespace(
        get_config_schema=Mock(return_value=schema)
    )
    import_module = Mock(return_value=schema_module)
    monkeypatch.setattr(common.importlib, "import_module", import_module)

    maps, _, errors = common.get_config_maps(
        case.experiment_dir, ["config_instr"]
    )

    import_module.assert_called_once_with(
        "cohere_beamlines.test_beamline.instr_schema"
    )
    case.verify_types.assert_called_with(schema, instrument_config)
    assert maps["config_instr"] is instrument_config
    assert errors["config_instr"] == ""


def test_get_config_maps_instrument_requires_beamline(config_case):
    case = config_case
    del case.main_config["beamline"]

    with pytest.raises(ValueError, match="cannot import"):
        common.get_config_maps(
            case.experiment_dir, ["config_instr"]
        )


# ---------------------------------------------------------------------------
# get_pkg
# ---------------------------------------------------------------------------

@pytest.fixture
def package_imports(monkeypatch):
    """Control optional imports without loading real CuPy or Torch."""
    monkeypatch.setattr(common.sys, "platform", "linux")

    def configure(cupy=False, torch=False):
        # None in sys.modules makes an import raise ModuleNotFoundError.
        monkeypatch.setitem(
            sys.modules, "cupy", ModuleType("cupy") if cupy else None
        )
        monkeypatch.setitem(
            sys.modules, "torch", ModuleType("torch") if torch else None
        )

    return configure


@pytest.mark.parametrize(
    "cupy, torch, device, hpc, expected",
    [
        (True, True, [0], False, "cp"),
        (False, True, [0], False, "torch"),
        (False, False, [0], False, "np"),
        (True, True, [-1], False, "torch"),
        (True, False, [-1], False, "np"),
        (True, False, [-1], True, "cp"),
    ],
)
def test_get_pkg_auto(
    package_imports, cupy, torch, device, hpc, expected
):
    package_imports(cupy=cupy, torch=torch)

    assert common.get_pkg("auto", device, hpc=hpc) == expected


def test_get_pkg_numpy(package_imports):
    package_imports(cupy=False, torch=False)

    assert common.get_pkg("np", [-1]) == "np"


def test_get_pkg_explicit_cupy(package_imports):
    package_imports(cupy=True)

    assert common.get_pkg("cp", [0]) == "cp"


def test_get_pkg_cupy_requires_device(package_imports):
    package_imports(cupy=True)

    with pytest.raises(ValueError, match="define a valid device"):
        common.get_pkg("cp", [-1])


def test_get_pkg_cupy_hpc_allows_default_device(package_imports):
    package_imports(cupy=True)

    assert common.get_pkg("cp", [-1], hpc=True) == "cp"


def test_get_pkg_cupy_missing(package_imports):
    package_imports(cupy=False)

    with pytest.raises(ValueError, match="cupy is not installed"):
        common.get_pkg("cp", [0])


def test_get_pkg_cupy_unsupported_on_mac(package_imports, monkeypatch):
    package_imports(cupy=True)
    monkeypatch.setattr(common.sys, "platform", "darwin")

    with pytest.raises(ValueError, match="cupy is not supported by Mac"):
        common.get_pkg("cp", [0])


def test_get_pkg_explicit_torch(package_imports):
    package_imports(torch=True)

    assert common.get_pkg("torch", [-1]) == "torch"


def test_get_pkg_torch_missing(package_imports):
    package_imports(torch=False)

    with pytest.raises(ValueError, match="torch is not installed"):
        common.get_pkg("torch", [0])


def test_get_pkg_invalid_processing(package_imports):
    package_imports()

    with pytest.raises(ValueError, match='invalid "processing" value'):
        common.get_pkg("unsupported", [0])