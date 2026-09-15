"""The output muxes must use a bounded interleave window (v2.2.4).

max_interleave_delta=0 makes libavformat buffer until it holds a packet for
every output stream; on a Blu-ray remux with a dozen sparse PGS subtitle tracks
ffmpeg buffered the whole source (5.2 GB in 95 s) and was OOM-killed inside
describarr's 7 GiB container on every PAL-rate remux, 2026-09-15.
"""
import numpy as np

import describealaign as da


def _capture_write_command(monkeypatch, layout):
    captured = {}

    def fake_run_async(command, media_arr, err_msg):
        captured['args'] = command.compile(cmd='ffmpeg')

    monkeypatch.setattr(da, 'run_async_ffmpeg_command', fake_run_async)
    monkeypatch.setattr(da, '_probe_stream_layout', lambda video_file: layout)
    da.write_replaced_media_to_disk('out.mkv', np.zeros((2, 4410), dtype=np.float32), video_file='in.mkv')
    return captured['args']


def test_stretch_audio_mux_uses_bounded_interleave_window(monkeypatch):
    args = _capture_write_command(monkeypatch, {'audio_titles': ['English'], 'subtitle_codecs': ['hdmv_pgs_subtitle'] * 11, 'n_attachment': 0})
    assert '-max_interleave_delta' in args
    value = args[args.index('-max_interleave_delta') + 1]
    assert value != '0'
    assert int(value) == da.MUX_MAX_INTERLEAVE_DELTA_US
    # sanity: 30 s window, far below "wait for every stream"
    assert 1_000_000 <= int(value) <= 120_000_000


def test_no_zero_interleave_delta_anywhere_in_source():
    src = open(da.__file__).read()
    assert "'max_interleave_delta': '0'" not in src
