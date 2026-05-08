"""
Property tests on the describealaign report format.

Two consumers (the .txt report and the JSON sibling) must agree on every
numeric value, regardless of which file a downstream tool picks up.

We also pin the JSON schema: the keys describarr (and any future tooling)
relies on must keep their names and types.
"""
import json as _json
import os
import tempfile

import numpy as np
import pytest

import describealaign as da


def _synthesize_alignment(median_slope=1.0427, n_segments=11):
    """
    Build a fake but realistic alignment artifact set: a stable trunk at
    the given slope plus short artifact jumps between segments.

    Returns the inputs that ``plot_alignment`` expects.
    """
    seg_dur = 200.0  # seconds per stable trunk
    artifact_dur = 0.04  # short seam between trunks

    video_times = [0.0]
    audio_times = [0.0]
    for i in range(n_segments):
        # Stable trunk
        v_next = video_times[-1] + seg_dur
        a_next = audio_times[-1] + (seg_dur / median_slope)
        video_times.append(v_next)
        audio_times.append(a_next)
        # Artifact seam (only between trunks, not after the last)
        if i < n_segments - 1:
            v_next = video_times[-1] + artifact_dur
            a_next = audio_times[-1] + 0.0001  # almost-zero audio jump
            video_times.append(v_next)
            audio_times.append(a_next)

    video_times = np.array(video_times)
    audio_times = np.array(audio_times)

    # Bare-bones path (only x/y matter for plotting; quals are unused by
    # report-writing). Other columns set to zero to satisfy the unpack.
    n = len(video_times)
    path = np.zeros((n, 5))
    path[:, 0] = video_times * 210
    path[:, 1] = audio_times * 210
    return audio_times, video_times, path, median_slope


def _write_reports(tmpdir, similarity, median_slope=1.0427):
    audio_times, video_times, path, median = _synthesize_alignment(median_slope=median_slope)
    plot_filename_no_ext = os.path.join(tmpdir, "alignment_test")
    # plot_alignment writes both .txt and .json (and a .png). We only need
    # the data files; the PNG is ignorable.
    da.plot_alignment(
        plot_filename_no_ext, path, audio_times, video_times,
        similarity_percent=similarity, median_slope=median,
        stretch_audio=True, no_pitch_correction=False,
    )
    return plot_filename_no_ext


def test_json_and_text_reports_agree_on_similarity():
    with tempfile.TemporaryDirectory() as tmp:
        base = _write_reports(tmp, similarity=72.34)
        with open(base + ".txt") as f:
            text = f.read()
        with open(base + ".json") as f:
            data = _json.load(f)
        # Both should encode the same similarity, parsed with identical
        # rounding (2 decimals).
        assert "Input file similarity: 72.34%" in text
        assert data["similarity_pct"] == pytest.approx(72.34, abs=0.005)


def test_json_and_text_reports_agree_on_median_rate():
    with tempfile.TemporaryDirectory() as tmp:
        base = _write_reports(tmp, similarity=70.0, median_slope=1.0427)
        with open(base + ".txt") as f:
            text = f.read()
        with open(base + ".json") as f:
            data = _json.load(f)
        assert "Median Rate Change: 4.27%" in text
        assert data["median_rate_pct"] == pytest.approx(4.27, abs=0.005)


def test_json_segment_count_matches_text():
    with tempfile.TemporaryDirectory() as tmp:
        base = _write_reports(tmp, similarity=70.0)
        with open(base + ".txt") as f:
            text_segs = sum(1 for line in f if "Rate change of" in line)
        with open(base + ".json") as f:
            data = _json.load(f)
        assert text_segs == len(data["segments"])
        assert text_segs > 0


def test_stable_trunk_fraction_recognises_pal_ntsc_drift():
    """A clean PAL→NTSC alignment with short seam artifacts should land
    well above 90% stable-trunk."""
    with tempfile.TemporaryDirectory() as tmp:
        base = _write_reports(tmp, similarity=63.0, median_slope=1.0427)
        with open(base + ".json") as f:
            data = _json.load(f)
        # Stable trunks dominate runtime; the artifact spikes are < 0.05 s each.
        assert data["stable_trunk_fraction_pct"] >= 95.0


def test_json_schema_keys_are_stable():
    """describarr (and any downstream consumer) reads these specific keys.
    Adding keys is fine; renaming or removing them is a breaking change."""
    with tempfile.TemporaryDirectory() as tmp:
        base = _write_reports(tmp, similarity=70.0)
        with open(base + ".json") as f:
            data = _json.load(f)
        for key in (
            "version", "version_hash", "parameters",
            "similarity_pct", "stable_trunk_fraction_pct",
            "median_rate_pct", "start_offset_sec", "segments",
        ):
            assert key in data, f"required JSON key {key!r} missing"
        for seg_key in (
            "rate_pct",
            "video_start_sec", "video_end_sec",
            "audio_start_sec", "audio_end_sec",
        ):
            assert seg_key in data["segments"][0], \
                f"required segment key {seg_key!r} missing"
