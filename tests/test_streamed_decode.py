"""
Streamed-decode equivalence and edge cases.

The 2.2.0 decode path streams ffmpeg output to a disk-backed .npy and
returns a read-only memmap. Sample values must be bit-identical to the
historical one-shot path (communicate -> frombuffer -> astype), which is
reconstructed inline here as ground truth.
"""
import os

import numpy as np
import pytest

import describealaign as da

TEST_MEDIA = os.path.join(os.path.dirname(__file__), '..', 'test_media')
VIDEO = os.path.join(TEST_MEDIA, 'ask_dad_trimmed.mp4')
AUDIO = os.path.join(TEST_MEDIA, 'ask_dad_moviesfortheblind_ep_01_trimmed.mp3')


def _one_shot_decode(media_file, num_channels):
    """The pre-2.2.0 decode, reconstructed as ground truth."""
    command = da.ffmpeg.input(media_file).output(
        '-', format='s16le', acodec='pcm_s16le',
        af='aresample=async=1:first_pts=0', map='0:a:0',
        ac=num_channels, ar=da.AUDIO_SAMPLE_RATE, loglevel='error')
    stream, _ = da.run_ffmpeg_command(command, "decode (test ground truth)")
    return np.frombuffer(stream, np.int16).astype(np.float32) \
        .reshape((-1, num_channels)).T


@pytest.mark.parametrize('num_channels', [1, 2])
@pytest.mark.parametrize('media', [VIDEO, AUDIO], ids=['mp4', 'mp3'])
def test_streamed_decode_matches_one_shot(tmp_path, media, num_channels):
    expected = _one_shot_decode(media, num_channels)
    got = da.parse_audio_from_file(media, num_channels, cache_dir=str(tmp_path))
    assert isinstance(got, np.memmap)
    assert not got.flags.writeable
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


def test_cache_hit_returns_readonly_memmap(tmp_path):
    first = da.parse_audio_from_file(AUDIO, 2, cache_dir=str(tmp_path))
    second = da.parse_audio_from_file(AUDIO, 2, cache_dir=str(tmp_path))
    assert isinstance(second, np.memmap)
    assert not second.flags.writeable
    assert np.array_equal(first, second)
    # Exactly one cache entry, no leftover intermediates.
    names = sorted(os.listdir(tmp_path))
    assert len(names) == 1 and names[0].endswith('.npy')


def test_legacy_cache_entry_still_readable(tmp_path):
    # Entries written by the old np.save path must keep working.
    arr = (np.arange(40, dtype=np.float32) - 20).reshape(2, 20)
    key = da._audio_cache_key(AUDIO, 2)
    np.save(os.path.join(tmp_path, f"{key}.npy"), arr)
    got = da.parse_audio_from_file(AUDIO, 2, cache_dir=str(tmp_path))
    assert np.array_equal(got, arr)


def test_decode_failure_raises_and_leaves_no_intermediates(tmp_path):
    bogus = tmp_path / "not_media.mp4"
    bogus.write_bytes(b"this is not a video")
    with pytest.raises(RuntimeError):
        da.parse_audio_from_file(str(bogus), 2, cache_dir=str(tmp_path / "cache"))
    leftovers = [n for n in os.listdir(tmp_path / "cache")] \
        if (tmp_path / "cache").is_dir() else []
    assert leftovers == []
