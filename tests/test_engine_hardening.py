"""
Regression tests for the 2.1.9 hardening pass:
  - _compute_report_data no longer divides by a zero audio delta.
  - _write_failure_sidecar persists the failure diagnosis and is best-effort.
"""
import json

import describealaign as da


def test_compute_report_data_skips_zero_audio_delta():
    """A non-monotonic alignment pair (equal audio timestamps) used to raise
    ZeroDivisionError and crash the whole report write. The degenerate
    segment must now be skipped, not fatal."""
    video_times = [0.0, 10.0, 20.0, 30.0]
    audio_times = [0.0, 10.0, 10.0, 20.0]  # index 1->2 has a zero delta
    data = da._compute_report_data(audio_times, video_times, median_slope=1.0)
    # Segments 0->1 and 2->3 are valid; the zero-delta middle one is dropped.
    assert len(data["segment_records"]) == 2
    assert all(abs(s["rate_pct"]) < 1e-6 for s in data["segment_records"])


def test_compute_report_data_skips_negative_audio_delta():
    video_times = [0.0, 10.0, 20.0]
    audio_times = [0.0, 15.0, 5.0]  # index 1->2 goes backwards
    data = da._compute_report_data(audio_times, video_times, median_slope=1.0)
    assert len(data["segment_records"]) == 1


def test_write_failure_sidecar_writes_expected_schema(tmp_path):
    diag = {
        "summary": "AD audio is 0.20× the video duration — likely wrong episode",
        "duration_ratio": 0.2,
        "audio_quiet_fraction": 0.1,
    }
    da._write_failure_sidecar(str(tmp_path), "Some Show S01E01.mkv", diag)

    sidecar = tmp_path / "Some Show S01E01.fail.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["error"] == "alignment_mismatch"
    assert "wrong episode" in payload["summary"]
    assert payload["diagnostic"]["duration_ratio"] == 0.2
    assert payload["version"] == da.__version__


def test_write_failure_sidecar_tolerates_non_dict_diagnostic(tmp_path):
    da._write_failure_sidecar(str(tmp_path), "x.mkv", None)
    payload = json.loads((tmp_path / "x.fail.json").read_text())
    assert payload["summary"] == ""
    assert payload["diagnostic"] == {}


def test_write_failure_sidecar_swallows_io_errors(tmp_path):
    """It must never raise — a sidecar problem cannot be allowed to mask the
    real AlignmentMismatchError the caller is about to re-raise."""
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    # alignment_dir points at a regular file; folder creation/write fails.
    da._write_failure_sidecar(str(blocker), "x.mkv", {"summary": "s"})
