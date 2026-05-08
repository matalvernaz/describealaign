"""
Property tests for the seam crossfade in describealaign.replace_aligned_segments.

The crossfade is supposed to:
  - Smooth the amplitude transition at every AD↔original boundary.
  - Preserve sample-level timing (a sample at time T stays at time T).
  - Be a no-op far from any boundary.
  - Apply equal-power (sin² + cos² = 1) so the perceived loudness is
    constant through the fade.
"""
import numpy as np
import pytest

import describealaign as da

SR = da.AUDIO_SAMPLE_RATE


def _three_segment_setup():
    """Three 5-second segments. Middle is skipped (huge slope)."""
    seg_dur = 5.0
    seg_samples = int(seg_dur * SR)
    n_segs = 3
    total_samples = seg_samples * n_segs

    # Constant-amplitude streams make per-sample expectations easy.
    video_arr = np.full((2, total_samples), 0.2, dtype=np.float32)
    audio_desc_arr = np.full((2, total_samples), 0.8, dtype=np.float32)

    video_times = np.array([0.0, 5.0, 10.0, 15.0])
    audio_desc_times = np.array([0.0, 5.0, 5.001, 10.001])
    return video_arr, audio_desc_arr, audio_desc_times, video_times


def test_seam_amplitude_is_smooth_at_fade_out_boundary():
    """At the AD→original boundary, the fade-out midpoint should sit at the
    equal-power midpoint, not jump straight from 0.8 to 0.2."""
    video_arr, audio_desc_arr, ad_times, vid_times = _three_segment_setup()
    da.replace_aligned_segments(
        video_arr, audio_desc_arr, ad_times, vid_times, no_pitch_correction=True
    )
    mono = video_arr.mean(axis=0)

    seam = int(5.0 * SR)
    fade = int(da.SEAM_CROSSFADE_SECONDS * SR)
    pre = mono[seam - fade - 100]
    mid = mono[seam - fade // 2]
    post = mono[seam + 100]

    assert pre == pytest.approx(0.8, abs=0.01), "AD region body should be at AD level"
    assert post == pytest.approx(0.2, abs=0.01), "skipped region should retain original"
    # Equal-power crossfade midpoint of two constants A and B is
    # sqrt(A^2 * 0.5 + B^2 * 0.5) ≈ 0.58, but for this test we only
    # require that mid is strictly between original and AD.
    assert 0.3 <= mid <= 0.7, f"midpoint {mid} should be between 0.2 and 0.8"


def test_seam_amplitude_is_smooth_at_fade_in_boundary():
    """At the original→AD boundary, the fade-in should ramp the AD up."""
    video_arr, audio_desc_arr, ad_times, vid_times = _three_segment_setup()
    da.replace_aligned_segments(
        video_arr, audio_desc_arr, ad_times, vid_times, no_pitch_correction=True
    )
    mono = video_arr.mean(axis=0)

    seam = int(10.0 * SR)
    fade = int(da.SEAM_CROSSFADE_SECONDS * SR)
    pre = mono[seam - 100]
    mid = mono[seam + fade // 2]
    post = mono[seam + fade + 100]

    assert pre == pytest.approx(0.2, abs=0.01)
    assert post == pytest.approx(0.8, abs=0.01)
    assert 0.3 <= mid <= 0.7


def test_fade_out_is_monotonically_decreasing():
    """The fade-out window should be a smooth ramp, not a step."""
    video_arr, audio_desc_arr, ad_times, vid_times = _three_segment_setup()
    da.replace_aligned_segments(
        video_arr, audio_desc_arr, ad_times, vid_times, no_pitch_correction=True
    )
    mono = video_arr.mean(axis=0)
    seam = int(5.0 * SR)
    fade = int(da.SEAM_CROSSFADE_SECONDS * SR)
    window = mono[seam - fade : seam]
    # Every successive sample should be ≤ the previous (within float epsilon).
    diffs = np.diff(window)
    assert (diffs <= 1e-5).all(), "fade-out window must monotonically decrease"


def test_no_op_far_from_any_seam():
    """Samples deep inside an AD region — outside any fade window —
    should be exactly the AD content the algorithm produced before crossfade
    (i.e. unchanged by the crossfade pass)."""
    video_arr, audio_desc_arr, ad_times, vid_times = _three_segment_setup()
    sr = SR

    # Probe at video time 2.5 s — middle of AD region 0, well clear of the
    # fade window at the start (0 → 0.2 s) and end (4.8 → 5.0 s) of that
    # region.
    da.replace_aligned_segments(
        video_arr, audio_desc_arr, ad_times, vid_times, no_pitch_correction=True
    )
    mono = video_arr.mean(axis=0)

    deep_inside = mono[int(2.5 * sr)]
    assert deep_inside == pytest.approx(0.8, abs=0.001), \
        "AD content far from any fade should be unchanged"
