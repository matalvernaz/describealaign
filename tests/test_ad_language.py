"""The AD track carries a language (v2.2.6).

Until v2.2.6 every mux titled the AD track and set its disposition but gave it
no language. Jellyfin's Smart subtitle mode compares the default audio track's
language with the viewer's; an unlabelled track reads as foreign, so English
subtitles came on over English audio, and a client that reads subtitles aloud
for blind viewers talked over the whole programme (2026-10-02, 3,149 described
titles affected).
"""
import argparse
import subprocess

import numpy as np
import pytest

import describealaign as da

LAYOUT = {'audio_titles': [''], 'audio_languages': ['eng'], 'subtitle_codecs': [], 'n_attachment': 0}


def _capture(monkeypatch, layout, write):
    """Compiled ffmpeg arguments of one mux, without running ffmpeg."""
    captured = {}

    def fake_run(command, err_msg):
        captured['args'] = command.compile(cmd='ffmpeg')

    def fake_run_async(command, media_arr, err_msg):
        captured['args'] = command.compile(cmd='ffmpeg')

    monkeypatch.setattr(da, 'run_ffmpeg_command', fake_run)
    monkeypatch.setattr(da, 'run_async_ffmpeg_command', fake_run_async)
    monkeypatch.setattr(da, '_probe_stream_layout', lambda video_file: layout)
    write()
    return captured['args']


def _passthrough(**kwargs):
    return lambda: da.write_passthrough_media_to_disk('out.mkv', 'in.mkv', 'ad.mp3', 0.0, **kwargs)


def _stretch_audio(**kwargs):
    return lambda: da.write_replaced_media_to_disk(
        'out.mkv', np.zeros((2, 4410), dtype=np.float32), video_file='in.mkv', **kwargs)


def _stretch_video(**kwargs):
    return lambda: da.write_replaced_media_to_disk(
        'out.mkv', None, 'in.mkv', 'ad.mp3', 'T', 0.0, 0.0, **kwargs)


MUXES = [_passthrough, _stretch_audio, _stretch_video]


def _option_values(args, flag):
    return [args[i + 1] for i, a in enumerate(args) if a == flag]


@pytest.mark.parametrize('mux', MUXES)
def test_every_mux_labels_the_ad_track(monkeypatch, mux):
    args = _capture(monkeypatch, LAYOUT, mux(ad_language='eng'))
    assert _option_values(args, '-metadata:s:0') == ['language=eng']
    assert _option_values(args, '-metadata:s:a:0') == ['title=AD']


@pytest.mark.parametrize('mux', MUXES)
def test_the_ad_is_output_stream_zero(monkeypatch, mux):
    # `-metadata:s:0` only reaches the AD because the AD is mapped first.
    args = _capture(monkeypatch, LAYOUT, mux(ad_language='eng'))
    assert _option_values(args, '-i')[0] in ('ad.mp3', 'pipe:')  # input 0 is the AD
    maps = _option_values(args, '-map')
    assert maps and maps[0].split(':')[0] == '0', maps  # and it is mapped first


@pytest.mark.parametrize('mux', MUXES)
def test_the_language_defaults_to_the_first_audio_track(monkeypatch, mux):
    layout = dict(LAYOUT, audio_languages=['fre', 'eng'])
    args = _capture(monkeypatch, layout, mux())
    assert _option_values(args, '-metadata:s:0') == ['language=fre']


@pytest.mark.parametrize('mux', MUXES)
def test_an_explicit_language_beats_the_source(monkeypatch, mux):
    layout = dict(LAYOUT, audio_languages=['fre'])
    args = _capture(monkeypatch, layout, mux(ad_language='eng'))
    assert _option_values(args, '-metadata:s:0') == ['language=eng']


@pytest.mark.parametrize('mux', MUXES)
@pytest.mark.parametrize('source', [[''], ['und'], ['UND'], []])
def test_an_unlabelled_source_leaves_the_ad_unlabelled(monkeypatch, mux, source):
    # Better no label than a guessed one: the source's first track says nothing.
    layout = dict(LAYOUT, audio_languages=source, audio_titles=[''] * len(source))
    args = _capture(monkeypatch, layout, mux())
    assert '-metadata:s:0' not in args
    assert _option_values(args, '-metadata:s:a:0') == ['title=AD']


def test_the_audio_only_output_is_unchanged(monkeypatch):
    # An audio file in, an audio file out: no video, no AD track to label.
    args = _capture(monkeypatch, LAYOUT, lambda: da.write_replaced_media_to_disk(
        'out.m4a', np.zeros((2, 4410), dtype=np.float32), video_file=None, ad_language='eng'))
    assert '-metadata:s:0' not in args
    assert '-metadata:s:a:0' not in args


@pytest.mark.parametrize('value, code', [('eng', 'eng'), ('ENG', 'eng'), (' fre ', 'fre')])
def test_ad_language_code_accepts_iso_639_2(value, code):
    assert da.ad_language_code(value) == code


@pytest.mark.parametrize('value', ['en', 'english', 'en-US', '123', ''])
def test_ad_language_code_refuses_anything_else(value):
    with pytest.raises(argparse.ArgumentTypeError):
        da.ad_language_code(value)


# --- the real muxer --------------------------------------------------------

def _binaries():
    try:
        return da.get_ffmpeg(), da.get_ffprobe()
    except Exception as error:  # no ffmpeg on this machine and no download
        pytest.skip(f'ffmpeg unavailable: {error}')


def _make_inputs(tmp_path, extension, source_language):
    ffmpeg_bin, _ = _binaries()
    video = tmp_path / f'in{extension}'
    language = ['-metadata:s:a:0', f'language={source_language}'] if source_language else []
    subprocess.run([ffmpeg_bin, '-v', 'error', '-y',
                    '-f', 'lavfi', '-i', 'testsrc=size=64x48:rate=10:duration=2',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2',
                    '-map', '0:v', '-map', '1:a', '-c:v', 'mpeg4', '-c:a', 'aac',
                    *language, str(video)], check=True)
    ad = tmp_path / 'ad.m4a'
    subprocess.run([ffmpeg_bin, '-v', 'error', '-y',
                    '-f', 'lavfi', '-i', 'sine=frequency=660:duration=2',
                    '-c:a', 'aac', str(ad)], check=True)
    return video, ad


def _first_audio(path):
    _, ffprobe_bin = _binaries()
    probe = da.ffmpeg.probe(str(path), cmd=ffprobe_bin)
    return [s for s in probe['streams'] if s['codec_type'] == 'audio'][0]


@pytest.mark.parametrize('extension', ['.mkv', '.mp4'])
def test_real_mux_writes_the_language(tmp_path, extension):
    video, ad = _make_inputs(tmp_path, extension, source_language=None)
    out = tmp_path / f'out{extension}'
    da.write_passthrough_media_to_disk(str(out), str(video), str(ad), 0.0, ad_language='eng')
    first = _first_audio(out)
    tags = first.get('tags', {})
    assert tags.get('language') == 'eng'
    if extension == '.mkv':
        # A plain .mp4 mux drops track titles in ffmpeg 7.1 and 8.0 with or
        # without a language (measured 2026-10-02), so only MKV can show that
        # the title survived alongside it.
        assert tags.get('title') == 'AD'
    assert first['disposition']['default'] == 1
    assert first['disposition']['visual_impaired'] == 1


@pytest.mark.parametrize('extension', ['.mkv', '.mp4'])
def test_real_mux_inherits_the_source_language(tmp_path, extension):
    video, ad = _make_inputs(tmp_path, extension, source_language='spa')
    out = tmp_path / f'out{extension}'
    da.write_passthrough_media_to_disk(str(out), str(video), str(ad), 0.0)
    assert _first_audio(out).get('tags', {}).get('language') == 'spa'
