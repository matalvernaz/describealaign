"""
A replaced segment too short for the pitch-preserving stretch.

The gap-fill pass marks a short skipped segment inside continuous narration
for replacement. When that segment is a few milliseconds long but its slope
is off by more than the just-noticeable ratio, it used to be sent to
stretch(), which needs at least three analysis windows of input. Heartland
S10E14 (2026-09-24) had a 462-sample connector at slope 0.909 between two
long segments: stretch() found no windows, backtracked an empty jump list
and crashed with "too many indices for array".
"""
import numpy as np
import pytest

import describealaign as da

RATE = da.AUDIO_SAMPLE_RATE
LONG = 3 * RATE


def _run(short_ad_samples, short_video_samples):
    """Drive replace_aligned_segments with long / short / long segments."""
    x_samples = np.array([0, LONG, LONG + short_ad_samples,
                          2 * LONG + short_ad_samples])
    y_samples = np.array([0, LONG, LONG + short_video_samples,
                          2 * LONG + short_video_samples])
    # Half a sample of headroom so the engine's float->int truncation lands
    # on exactly these sample indices.
    audio_desc_times = (x_samples + 0.5) / RATE
    video_times = (y_samples + 0.5) / RATE
    assert np.array_equal((audio_desc_times * RATE).astype(int), x_samples)
    assert np.array_equal((video_times * RATE).astype(int), y_samples)

    rng = np.random.default_rng(3)
    ad = (rng.standard_normal((2, int(x_samples[-1]) + RATE)) * 1000).astype(np.float32)
    video = np.zeros((2, int(y_samples[-1]) + RATE), dtype=np.float32)
    da.replace_aligned_segments(video, ad, audio_desc_times, video_times,
                                no_pitch_correction=False)
    return video, int(y_samples[1]), int(y_samples[2])


@pytest.mark.parametrize("short_ad, short_video", [
    (462, 508),    # the Heartland S10E14 segment: zero stretch windows
    (1000, 1100),  # one window
    (1400, 1540),  # two windows, still under the correlation minimum
])
def test_short_filled_segment_does_not_crash(short_ad, short_video):
    slope = short_ad / short_video
    assert da.JUST_NOTICEABLE_DIFF_IN_FREQ_RATIO < abs(1 - slope) <= da.MAX_RATE_RATIO_DIFF_ALIGN
    assert short_video - short_ad >= da.MIN_STRETCH_OFFSET

    video, start, end = _run(short_ad, short_video)

    # The gap was filled with narration, not left as a hole.
    assert np.all(np.isfinite(video))
    assert np.abs(video[:, start:end]).max() > 0
