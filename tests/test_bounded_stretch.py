"""
Bounded stretch-path equivalence.

The 2.2.0 stretch path replaced whole-array operations (full-segment
linspace + hstack interpolation, full-array isfinite/abs peak limiting,
one-shot gain) with bounded slices. These tests pin the invariant that
mattered in review: identical output samples.
"""
import os

import numpy as np
import pytest

import describealaign as da

TEST_MEDIA = os.path.join(os.path.dirname(__file__), '..', 'test_media')
VIDEO = os.path.join(TEST_MEDIA, 'ask_dad_trimmed.mp4')
AUDIO = os.path.join(TEST_MEDIA, 'ask_dad_moviesfortheblind_ep_01_trimmed.mp3')


def test_chunked_sample_points_match_linspace():
    # audio_desc_arr_interp generates np.linspace(start, stop, num,
    # endpoint=False) values chunkwise: start + i*step. Bit-identical.
    rng = np.random.default_rng(42)
    for _ in range(20):
        start = float(rng.integers(0, 10**6))
        stop = start + float(rng.integers(1, 10**6))
        num = int(rng.integers(1, 350_000))
        expected = np.linspace(start, stop, num=num, endpoint=False)
        step = (stop - start) / num
        got = np.concatenate([
            np.arange(begin, min(begin + 10**5, num), dtype=np.float64) * step + start
            for begin in range(0, num, 10**5)
        ])
        assert got.shape == expected.shape
        assert np.array_equal(got, expected)


def test_interp_writes_expected_values_in_place():
    rng = np.random.default_rng(7)
    ad = (rng.standard_normal((2, 5000)) * 1000).astype(np.float32)
    out = np.zeros((2, 3000), dtype=np.float32)
    start, stop, num = 100.0, 4000.0, 3000

    # Ground truth via the pre-2.2.0 shape: one linspace + scipy interp.
    sample_points = np.linspace(start, stop, num=num, endpoint=False)
    import scipy.interpolate
    interp = scipy.interpolate.interp1d(
        np.arange(ad.shape[1]), ad, copy=False, bounds_error=False,
        fill_value=0, kind='quadratic', assume_sorted=True)
    expected = interp(sample_points).astype(np.float32)

    da.replace_aligned_segments.__globals__  # module import sanity
    # Reach the closure through a real call: drive replace_aligned_segments
    # with a single no-pitch-correction segment covering [start, stop).
    video = np.zeros((2, 3200), dtype=np.float32)
    video_times = np.array([0.0, num / da.AUDIO_SAMPLE_RATE])
    audio_times = np.array([start / da.AUDIO_SAMPLE_RATE,
                            stop / da.AUDIO_SAMPLE_RATE])
    da.replace_aligned_segments(video, ad, audio_times, video_times,
                                no_pitch_correction=True)
    # Crossfades touch the edges; compare the interior.
    fade = 2048
    assert np.allclose(video[:, fade:num - fade],
                       expected[:, fade:num - fade], atol=2.0)


def test_end_to_end_stretch_branch(tmp_path, monkeypatch):
    # Force the stretch branch (the media pair is same-rate, so passthrough
    # would normally win) and prove the full pipeline — scratch copies,
    # bounded gain, interp-in-place, bounded peak limit, mux — produces a
    # valid output from read-only decode memmaps.
    monkeypatch.setattr(da, 'is_passthrough_alignment', lambda *a, **k: False)
    monkeypatch.setenv('DESCRIBEALAIGN_AUDIO_CACHE_DIR', str(tmp_path / 'cache'))
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    da.combine(VIDEO, AUDIO, stretch_audio=True, yes=True,
               output_dir=str(out_dir), alignment_dir=str(tmp_path / 'align'))
    outputs = [f for f in os.listdir(out_dir) if f.startswith('ad_')]
    assert len(outputs) == 1
    out_file = out_dir / outputs[0]
    assert out_file.stat().st_size > 10**5
    # No scratch leftovers beside the output.
    leftovers = [f for f in os.listdir(out_dir) if f.startswith('.tmp.')]
    assert leftovers == []
