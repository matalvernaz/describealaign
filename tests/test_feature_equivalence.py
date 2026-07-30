"""
Chunked feature-extractor equivalence.

get_zero_crossings was rewritten to process bounded chunks with a seam
carry. Outputs must be bit-identical to the original whole-array
formulation, reconstructed here as ground truth.
"""
import numpy as np
import pytest
import scipy.signal

import describealaign as da


def _original_zero_crossings(arr):
    xings = np.diff(np.signbit(arr), prepend=False, axis=-1)
    xings_clip = xings[:, :(xings.shape[1] - (xings.shape[1] % 210))] \
        .reshape(xings.shape[0], -1, 210)
    zero_crossings = np.sum(np.abs(xings_clip), axis=(0, 2)).astype(np.float32)
    if xings.shape[0] == 1:
        zero_crossings *= 2
    hann_window = scipy.signal.windows.hann(15)[1:-1].astype(np.float32)
    hann_window = hann_window / np.sum(hann_window)
    return np.convolve(zero_crossings, hann_window, mode='same')


@pytest.mark.parametrize('channels', [1, 2])
@pytest.mark.parametrize('length', [
    210,                      # exactly one window
    210 * 3 + 17,             # ragged tail
    da._XINGS_CHUNK_FRAMES,   # exactly one chunk
    da._XINGS_CHUNK_FRAMES + 210 * 5 + 3,   # seam + ragged tail
    da._XINGS_CHUNK_FRAMES * 2 + 1,         # two seams
])
def test_chunked_zero_crossings_bit_identical(channels, length):
    rng = np.random.default_rng(length + channels)
    arr = (rng.standard_normal((channels, length)) * 800).astype(np.float32)
    # Sprinkle exact zeros and sign flips at seam positions to stress the carry.
    if length > da._XINGS_CHUNK_FRAMES:
        arr[:, da._XINGS_CHUNK_FRAMES - 1: da._XINGS_CHUNK_FRAMES + 1] = \
            np.array([[-1.0, 1.0]] * channels, dtype=np.float32)
    assert np.array_equal(da.get_zero_crossings(arr), _original_zero_crossings(arr))


@pytest.mark.parametrize('channels', [1, 2])
def test_sub_window_track_raises_like_original(channels):
    # < 210 samples is degenerate (5 ms of audio). The original whole-array
    # implementation raised ValueError from the empty convolve; the chunked
    # version must not silently return something instead.
    arr = np.ones((channels, 209), dtype=np.float32)
    with pytest.raises(ValueError):
        _original_zero_crossings(arr)
    with pytest.raises(ValueError):
        da.get_zero_crossings(arr)
