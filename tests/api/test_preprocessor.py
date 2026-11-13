# tests/test_preprocessor.py
import os
from unittest.mock import Mock

import numpy as np
import pytest

from cohere_ui.api import preprocessor as prep  # Adjust if needed


def test_get_corr_stores_result_under_scan_pair(monkeypatch):
    scans = [101, 102]
    arrays = [np.array([1, 2]), np.array([3, 4])]
    shifts = {101: {}, 102: {}}
    expected = ((1, -2, 0), 0.95)

    get_ccamax_cc = Mock(return_value=expected)
    monkeypatch.setattr(prep.dvut, "get_ccamax_cc", get_ccamax_cc)

    prep.get_corr(arrays, shifts, scans, (0, 1))

    get_ccamax_cc.assert_called_once_with(arrays[0], arrays[1])
    assert shifts[101][102] == expected


def test_process_batch_single_scan_delegates_to_save_results(monkeypatch):
    scans_infos = [(42, {"scan_info": "value"})]
    save_results = Mock()
    monkeypatch.setattr(prep, "save_results4scan", save_results)

    result = prep.process_batch(
        scans_infos,
        "/experiment",
        separate_scan_ranges=True,
        remove_outliers=False,
        instrument=object(),
        do_RSM=True,
    )

    assert result == []
    save_results.assert_called_once_with(
        42,
        {"scan_info": "value"},
        save_results.call_args.args[2],
        os.path.join(
            "/experiment", "scan_42", "preprocessed_data"
        ),
        True,
    )


def test_process_batch_returns_empty_for_mismatched_array_shapes(
    monkeypatch, capsys
):
    arrays = {
        "first": (np.ones((3, 3, 3)), (0, 0)),
        "second": (np.ones((4, 3, 3)), (0, 0)),
    }
    instrument = Mock()
    instrument.get_scan_array.side_effect = lambda info: arrays[info]

    monkeypatch.setattr(prep, "set_lib_from_pkg", lambda _: None)
    monkeypatch.setattr(prep, "devlib", np, raising=False)

    result = prep.process_batch(
        [(1, "first"), (2, "second")],
        "/experiment",
        separate_scan_ranges=False,
        remove_outliers=False,
        instrument=instrument,
        do_RSM=False,
    )

    assert result == []
    assert "Array shape mismatch" in capsys.readouterr().out


def test_save_results4scan_applies_offset_and_saves_metadata(
    tmp_path, monkeypatch
):
    data = np.zeros((3, 4, 2))
    data[1, 2, 0] = 10
    offset = (5, 7)

    instrument = Mock()
    instrument.get_scan_array.return_value = (data, offset)
    instrument.get_pixelQ.return_value = np.array([1.0, 2.0, 2.0])

    monkeypatch.setattr(prep, "set_lib_from_pkg", lambda _: None)
    monkeypatch.setattr(prep, "devlib", np, raising=False)
    monkeypatch.setattr(prep.ut, "join", os.path.join)
    save_tif = Mock()
    monkeypatch.setattr(prep.ut, "save_tif", save_tif)

    class FakeExcelWriter:
        def __init__(self, filename, engine):
            self.filename = filename
            self.engine = engine

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    writer_instances = []

    def fake_excel_writer(filename, engine):
        writer = FakeExcelWriter(filename, engine)
        writer_instances.append(writer)
        return writer

    monkeypatch.setattr(prep.pd, "ExcelWriter", fake_excel_writer)

    written_frames = []

    def fake_to_excel(frame, writer, **kwargs):
        written_frames.append((frame.copy(), writer, kwargs))

    monkeypatch.setattr(prep.pd.DataFrame, "to_excel", fake_to_excel)

    save_dir = tmp_path / "scan"
    prep.save_results4scan(
        scan=12,
        info="scan-info",
        instrument=instrument,
        save_dir=str(save_dir),
        do_RSM=False,
    )

    instrument.get_scan_array.assert_called_once_with("scan-info")
    instrument.get_pixelQ.assert_called_once_with([6, 9, 0], 12)

    save_tif.assert_called_once()
    saved_array, saved_path = save_tif.call_args.args
    assert saved_array is data
    assert saved_path == os.path.join(str(save_dir), "prep_data.tif")

    assert len(writer_instances) == 1
    assert writer_instances[0].filename == os.path.join(
        str(save_dir), "preprocess.xlsx"
    )
    assert writer_instances[0].engine == "xlsxwriter"

    frame, writer, kwargs = written_frames[0]
    assert writer is writer_instances[0]
    assert kwargs == {
        "sheet_name": "testSheetJ",
        "startrow": 0,
        "startcol": 0,
        "index": False,
    }
    assert frame.loc[0, "scan"] == 12
    assert frame.loc[0, "max ind (x)"] == 6
    assert frame.loc[0, "max ind (y)"] == 9
    assert frame.loc[0, "max ind frame"] == 0
    assert frame.loc[0, "max intensity"] == 10
    assert frame.loc[0, "Q_mag"] == pytest.approx(3.0)
    assert frame.loc[0, "d_spacing"] == pytest.approx(2 * np.pi / 3)