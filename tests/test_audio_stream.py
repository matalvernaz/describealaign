"""--audio_stream: align against, and fill gaps from, an audio track other than the first.

A "Dual Audio" release can put a dub first. Friends' HDMAN BluRays (2026-10-04)
open with a Portuguese track flagged default and carry the English second; the
engine decoded only `0:a:0`, so every English description was compared with
Portuguese dialogue and refused, and the dub would have filled every gap.
"""
import hashlib
import json
import os
import subprocess

import argparse
import numpy as np
import pytest

import describealaign as da

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(ROOT, 'test_media', 'ask_dad_trimmed.mp4')
AUDIO = os.path.join(ROOT, 'test_media', 'ask_dad_moviesfortheblind_ep_01_trimmed.mp3')


def _binaries():
    try:
        return da.get_ffmpeg(), da.get_ffprobe()
    except Exception as error:  # no ffmpeg on this machine and no download
        pytest.skip(f'ffmpeg unavailable: {error}')


# --- the decode cache --------------------------------------------------------

def test_the_first_track_keeps_its_historical_cache_key(tmp_path):
    f = tmp_path / 'a.mkv'
    f.write_bytes(b'x' * 100)
    st = os.stat(f)
    old = f"{os.path.abspath(f)}|{st.st_size}|{st.st_mtime_ns}|{da.AUDIO_SAMPLE_RATE}|2"
    expected = hashlib.sha256(old.encode('utf-8')).hexdigest()[:32]
    assert da._audio_cache_key(str(f), 2) == expected
    assert da._audio_cache_key(str(f), 2, audio_stream=0) == expected


def test_another_track_gets_a_key_of_its_own(tmp_path):
    f = tmp_path / 'a.mkv'
    f.write_bytes(b'x' * 100)
    keys = {da._audio_cache_key(str(f), 2, audio_stream=n) for n in (0, 1, 2)}
    assert len(keys) == 3
    assert da._audio_cache_key(str(f), 2, True, 1) != da._audio_cache_key(str(f), 2, False, 1)


def _two_track_file(tmp_path):
    """a:0 silent, a:1 a loud 1 kHz tone."""
    ffmpeg_bin, _ = _binaries()
    path = tmp_path / 'two.mkv'
    subprocess.run([ffmpeg_bin, '-v', 'error', '-y',
                    '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono',
                    '-f', 'lavfi', '-i', 'sine=frequency=1000:sample_rate=44100',
                    '-t', '2', '-map', '0:a', '-map', '1:a', '-c:a', 'pcm_s16le',
                    str(path)], check=True)
    return str(path)


def _rms(arr):
    return float(np.sqrt(np.mean(np.asarray(arr, dtype=np.float64) ** 2)))


def test_the_decode_reads_the_track_asked_for(tmp_path):
    path = _two_track_file(tmp_path)
    cache = str(tmp_path / 'cache')
    first = da.parse_audio_from_file(path, 1, cache_dir=cache)
    second = da.parse_audio_from_file(path, 1, cache_dir=cache, audio_stream=1)
    assert _rms(first) < 1.0
    assert _rms(second) > 1000.0


def test_a_cached_first_track_is_not_handed_back_for_the_second(tmp_path):
    path = _two_track_file(tmp_path)
    cache = str(tmp_path / 'cache')
    da.parse_audio_from_file(path, 1, cache_dir=cache)
    # Second time round both are cache hits; each must still be its own track.
    first = da.parse_audio_from_file(path, 1, cache_dir=cache)
    second = da.parse_audio_from_file(path, 1, cache_dir=cache, audio_stream=1)
    assert _rms(first) < 1.0 < 1000.0 < _rms(second)
    assert len([n for n in os.listdir(cache) if n.endswith('.npy')]) == 2


# --- refusing a track that is not there ----------------------------------------

def test_a_track_the_file_does_not_have_is_refused(monkeypatch):
    layout = {'audio_titles': ['', ''], 'audio_languages': ['por', 'eng'],
              'subtitle_codecs': [], 'n_attachment': 0}
    monkeypatch.setattr(da, '_probe_stream_layout', lambda video_file: layout)
    da._check_audio_stream('in.mkv', 1)
    with pytest.raises(RuntimeError, match='has 2 audio streams'):
        da._check_audio_stream('in.mkv', 2)


@pytest.mark.parametrize('value, index', [('0', 0), ('1', 1), (' 3 ', 3)])
def test_audio_stream_index_accepts_whole_numbers(value, index):
    assert da.audio_stream_index(value) == index


@pytest.mark.parametrize('value', ['-1', 'eng', '1.5', ''])
def test_audio_stream_index_refuses_anything_else(value):
    with pytest.raises(argparse.ArgumentTypeError):
        da.audio_stream_index(value)


# --- the AD's language follows the track it was matched to ---------------------

def test_the_ad_takes_the_language_of_the_track_matched():
    layout = {'audio_languages': ['por', 'eng']}
    assert da._ad_track_language(layout) == 'por'
    assert da._ad_track_language(layout, audio_stream=1) == 'eng'
    assert da._ad_track_language(layout, 'fre', audio_stream=1) == 'fre'
    assert da._ad_track_language(layout, audio_stream=5) is None


def test_the_mux_labels_the_ad_from_the_track_matched(monkeypatch):
    layout = {'audio_titles': ['', ''], 'audio_languages': ['por', 'eng'],
              'subtitle_codecs': [], 'n_attachment': 0}
    captured = {}
    monkeypatch.setattr(da, '_probe_stream_layout', lambda video_file: layout)
    monkeypatch.setattr(da, 'run_ffmpeg_command',
                        lambda command, err_msg: captured.setdefault('args', command.compile(cmd='ffmpeg')))
    da.write_passthrough_media_to_disk('out.mkv', 'in.mkv', 'ad.mp3', 0.0, audio_stream=1)
    args = captured['args']
    assert args[args.index('-metadata:s:0') + 1] == 'language=eng'


# --- end to end: a dub first, the original second -------------------------------

def _dub_first(tmp_path):
    """ask_dad with a noise "dub" first (por) and its own audio second (eng)."""
    ffmpeg_bin, _ = _binaries()
    path = tmp_path / 'dub_first.mkv'
    subprocess.run([ffmpeg_bin, '-v', 'error', '-y', '-i', VIDEO,
                    '-f', 'lavfi', '-i', 'anoisesrc=color=pink:amplitude=0.05:sample_rate=44100',
                    '-map', '0:v', '-map', '1:a', '-map', '0:a', '-shortest',
                    '-c:v', 'copy', '-c:a', 'aac',
                    '-metadata:s:a:0', 'language=por', '-metadata:s:a:1', 'language=eng',
                    str(path)], check=True)
    return str(path)


def _similarity(alignment_dir, video):
    base = os.path.splitext(os.path.basename(video))[0]
    report = os.path.join(alignment_dir, base + '.json')
    return json.load(open(report))['similarity_pct'] if os.path.exists(report) else None


def test_a_dub_first_release_aligns_against_the_original(tmp_path, monkeypatch):
    video = _dub_first(tmp_path)
    monkeypatch.setenv('DESCRIBEALAIGN_AUDIO_CACHE_DIR', str(tmp_path / 'cache'))

    # Against the first track (the "dub"), as before this option existed.
    dub_out, dub_align = tmp_path / 'dub_out', tmp_path / 'dub_align'
    dub_out.mkdir()
    try:
        da.combine(video, AUDIO, stretch_audio=True, yes=True,
                   output_dir=str(dub_out), alignment_dir=str(dub_align))
        dub_similarity = _similarity(str(dub_align), video)
    except da.AlignmentMismatchError:
        dub_similarity = 0.0

    # Against the second (the original). Every read of the video, the
    # alignment's and the gap fill's, must be of that track.
    reads = []
    real_parse = da.parse_audio_from_file

    def recording_parse(media_file, *args, **kwargs):
        if media_file == video:
            reads.append(kwargs.get('audio_stream', 0))
        return real_parse(media_file, *args, **kwargs)

    monkeypatch.setattr(da, 'parse_audio_from_file', recording_parse)
    monkeypatch.setattr(da, 'is_passthrough_alignment', lambda *a, **k: False)
    out, align_dir = tmp_path / 'out', tmp_path / 'align'
    out.mkdir()
    da.combine(video, AUDIO, stretch_audio=True, yes=True,
               output_dir=str(out), alignment_dir=str(align_dir), audio_stream=1)
    similarity = _similarity(str(align_dir), video)
    assert reads and set(reads) == {1}, reads
    print(f"similarity against the dub {dub_similarity}, against the original {similarity}")

    assert similarity is not None and similarity > 30, similarity
    assert dub_similarity is None or dub_similarity < 20, dub_similarity
    outputs = [f for f in os.listdir(out) if f.startswith('ad_')]
    assert len(outputs) == 1
    _, ffprobe_bin = _binaries()
    probe = da.ffmpeg.probe(str(out / outputs[0]), cmd=ffprobe_bin)
    audio = [s for s in probe['streams'] if s['codec_type'] == 'audio']
    # The AD first and labelled from the track it matched; both originals kept.
    assert audio[0]['tags'].get('language') == 'eng'
    assert len(audio) == 3


def test_combine_refuses_a_track_the_file_does_not_have(tmp_path, monkeypatch):
    _binaries()
    monkeypatch.setenv('DESCRIBEALAIGN_AUDIO_CACHE_DIR', str(tmp_path / 'cache'))
    with pytest.raises(RuntimeError, match='audio stream'):
        da.combine(VIDEO, AUDIO, stretch_audio=True, yes=True, output_dir=str(tmp_path),
                   alignment_dir=str(tmp_path / 'align'), audio_stream=1)
