"""
A recording is passed through only if its copy plays the way it was aligned and reaches both ends of the video's own sound.

Passthrough copies the recording's stream untouched at one offset, so the
description track starts and stops where the copied stream does, on its own
stored timing. Captain Underpants (checked 2026-11-11): 37 of 39 episodes'
recordings stopped 53 s before the video, the description tracks were copied in
that short, and a player ends the file when the track it is playing ends, so an
episode never moved on to the next. Other Mommy (2026-11-25) would have been the
same, 75 s short. The alignment, though, reads the recording's samples and
ignores its stored timing, so the copy is measured from its packets: where it
starts in its file, how long it runs, and whether its packets run on without a
jump. Anything else is rendered on the video's own sound, which runs the full
length.
"""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

import describealaign as da

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(ROOT, 'test_media', 'ask_dad_trimmed.mp4')


def _identity(seconds, segments=10, offset=0.0):
    """A straight same-speed map over *seconds* of the recording, placed *offset* into the video."""
    audio_times = np.linspace(0.0, seconds, segments + 1)
    return audio_times, audio_times + offset


def _decide(audio, video, recording, video_sound, copied=None):
    """The decision for a recording of *recording* seconds decoded, whose copy
    (*copied*, (lead, span, jump)) defaults to one stored exactly as it sounds."""
    if copied is None:
        copied = (0.0, recording, 0.0)
    return da._passthrough_decision(audio, video, 1.0, True, False, recording_seconds=recording,
                                    video_audio_seconds=video_sound, copied_span=copied)


# ── reaching both ends ───────────────────────────────────────────────────────

def test_a_recording_that_stops_early_is_not_passed_through():
    audio, video = _identity(2000.0)
    assert not _decide(audio, video, recording=2000.0, video_sound=2075.0)


def test_a_recording_that_starts_late_is_not_passed_through():
    audio, video = _identity(2000.0, offset=30.0)
    assert not _decide(audio, video, recording=2000.0, video_sound=2030.0)


def test_a_recording_that_covers_the_video_is_still_passed_through():
    # Other Mommy's recording starts 0.57 s in; a fraction of a second at
    # either end is ordinary and keeps the untouched copy.
    audio, video = _identity(2000.0, offset=0.57)
    assert _decide(audio, video, recording=2000.0, video_sound=2001.2)


def test_a_recording_that_runs_past_the_video_is_still_passed_through():
    audio, video = _identity(2000.0)
    assert _decide(audio, video, recording=2010.0, video_sound=2000.0)


def test_without_the_lengths_the_decision_is_as_before():
    audio, video = _identity(2000.0)
    assert da._passthrough_decision(audio, video, 1.0, True, False)


@pytest.mark.parametrize('end_gap, passes', [(1.0, True), (1.01, False)])
def test_the_end_limit(end_gap, passes):
    audio, video = _identity(2000.0)
    assert _decide(audio, video, recording=2000.0, video_sound=2000.0 + end_gap) is passes


def test_the_gaps_come_from_the_copy():
    audio, video = _identity(2000.0, offset=0.57)
    start_gap, end_gap = da._passthrough_end_gaps(audio, video, 0.02, 2000.0, 2075.6)
    assert start_gap == pytest.approx(0.59)
    assert end_gap == pytest.approx(75.01)


def test_the_two_allowances_do_not_add_up():
    # Decoded, the recording stops 1.0 s before the video's sound: allowed.
    # Its copy runs 0.05 s shorter than it sounds (within the timing
    # allowance), so the copied track really stops 1.05 s early: refused.
    audio, video = _identity(2000.0)
    assert _decide(audio, video, recording=2000.0, video_sound=2001.0)
    assert not _decide(audio, video, recording=2000.0, video_sound=2001.0, copied=(0.0, 1999.95, 0.0))
    # And a lead within the timing allowance still counts against the start.
    audio, video = _identity(2000.0, offset=0.95)
    assert not _decide(audio, video, recording=2000.0, video_sound=2001.0, copied=(0.08, 2000.0, 0.0))


def test_without_a_measured_copy_nothing_passes_through():
    audio, video = _identity(2000.0)
    assert not da._passthrough_decision(audio, video, 1.0, True, False, recording_seconds=2000.0,
                                        video_audio_seconds=2000.0, copied_span=None)


# ── the copy must play the way the recording was aligned ────────────────────
# The alignment reads the recording's samples from its first one and ignores
# its stored timestamps; passthrough copies the stream with them.

@pytest.mark.parametrize('copied_span, matches', [
    ((0.0, 2000.0, 0.0), True),
    ((-0.025, 2000.027, 0.0), True),   # an AudioVault MP3 as measured: one frame of encoder delay
    ((0.0, 2000.057, 0.0), True),      # a LivingAudio MP3 as measured
    ((0.1, 2000.1, 0.1), True),
    ((5.0, 2000.0, 0.0), False),       # stored 5 s after its file starts: copied in that late
    ((0.15, 2000.0, 0.0), False),      # would put the narration 0.15 s off the picture throughout
    ((0.0, 2000.3, 0.0), False),       # stored timing runs longer than its sound
    ((0.0, 1999.8, 0.0), False),
    ((0.0, 2000.0, 0.5), False),       # stored timing strays 0.5 s mid-file, even if undone later
    (None, False),                     # not read: no passthrough
])
def test_the_copy_must_match_the_decode(copied_span, matches):
    assert da._copy_matches_decode(copied_span, 2000.0) is matches


# ── through combine ──────────────────────────────────────────────────────────

needs_ffmpeg = pytest.mark.skipif(shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None,
                                  reason='needs ffmpeg and ffprobe')


def _recording_from_the_video(tmp_path, seconds=None):
    """The video's own sound as a recording: it lines up exactly, slope 1.0, so
    only how much of the video it reaches decides passthrough."""
    recording = tmp_path / f'recording-{seconds or "whole"}.mp3'
    limit = ['-t', str(seconds)] if seconds else []
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', VIDEO, '-vn', *limit,
                    '-c:a', 'libmp3lame', '-q:a', '4', str(recording)], check=True)
    return recording


def _first_audio_track_end(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                             '-show_entries', 'packet=pts_time,duration_time', '-of', 'csv=p=0', str(path)],
                            capture_output=True, text=True, check=True)
    last = [line for line in result.stdout.splitlines() if line.strip()][-1]
    pts, length = last.split(',')[:2]
    return float(pts) + float(length or 0)


def _combine(tmp_path, monkeypatch, recording, name):
    monkeypatch.setenv('DESCRIBEALAIGN_AUDIO_CACHE_DIR', str(tmp_path / 'cache'))
    out_dir = tmp_path / f'out-{name}'
    align_dir = tmp_path / f'align-{name}'
    da.combine(VIDEO, str(recording), stretch_audio=True, yes=True,
               output_dir=str(out_dir), alignment_dir=str(align_dir))
    outputs = [f for f in os.listdir(out_dir) if f.startswith('ad_')]
    assert len(outputs) == 1
    reports = [f for f in os.listdir(align_dir) if f.endswith('.json') and not f.endswith('.fail.json')]
    report = json.loads((align_dir / reports[0]).read_text())
    return out_dir / outputs[0], report


def _duration(path):
    return float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                 '-of', 'csv=p=0', str(path)], capture_output=True, text=True,
                                check=True).stdout)


@needs_ffmpeg
def test_a_recording_that_reaches_the_end_is_passed_through(tmp_path, monkeypatch):
    # The control: this pairing does take passthrough when nothing is missing,
    # so the renders below are the rules' doing.
    output, report = _combine(tmp_path, monkeypatch, _recording_from_the_video(tmp_path), 'whole')
    assert report['passthrough'] is True


@needs_ffmpeg
def test_a_recording_that_stops_30_s_early_is_rendered_full_length(tmp_path, monkeypatch):
    video_length = _duration(VIDEO)
    output, report = _combine(tmp_path, monkeypatch, _recording_from_the_video(tmp_path, video_length - 30), 'short')
    assert report['passthrough'] is False
    assert _first_audio_track_end(output) > video_length - 1.5


@needs_ffmpeg
def test_a_plain_recording_is_stored_as_it_sounds(tmp_path):
    recording = _recording_from_the_video(tmp_path)
    lead, span, drift = da._copied_stream_span(str(recording))
    assert abs(lead) < 0.1 and drift < 0.01
    assert span == pytest.approx(_duration(recording), abs=0.2)


@needs_ffmpeg
def test_a_recording_whose_first_track_is_stored_late_is_rendered(tmp_path, monkeypatch):
    # An .mka whose first track starts 5 s after the file does (its second
    # starts at 0): decoded, the first track lines up exactly; copied, it
    # would play 5 s late.
    whole = _recording_from_the_video(tmp_path)
    delayed = tmp_path / 'delayed.mka'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-itsoffset', '5', '-i', str(whole), '-i', str(whole),
                    '-map', '0:a:0', '-map', '1:a:0', '-c', 'copy', str(delayed)], check=True)
    lead, _, _ = da._copied_stream_span(str(delayed))
    assert lead == pytest.approx(5.0, abs=0.1)
    _, report = _combine(tmp_path, monkeypatch, delayed, 'delayed')
    assert report['passthrough'] is False


# ── drift: small slips that add up ───────────────────────────────────────────

class _FakeProbe:
    """ffprobe printing *lines*, as subprocess.Popen would hand them over."""

    def __init__(self, lines):
        self.stdout = iter(lines)

    def wait(self):
        return 0

    def poll(self):
        return 0

    def kill(self):
        pass


def _packets(steps, length=0.026):
    """Packet lines whose starts follow each other by *length* plus each step."""
    lines, start = [], 0.0
    for step in steps:
        lines.append(f"{start:.6f},{length:.6f}\n")
        start += length + step
    lines.append(f"{start:.6f},{length:.6f}\n")
    return lines


def _read(monkeypatch, lines):
    monkeypatch.setattr(da.ffmpeg, 'probe', lambda *a, **k: {'format': {'start_time': '0.000000'}})
    monkeypatch.setattr(da.subprocess, 'Popen', lambda *a, **k: _FakeProbe(lines))
    return da._copied_stream_span('recording.mka')


def test_small_slips_that_add_up_count_as_drift(monkeypatch):
    # Sixty 20 ms gaps, then sixty 20 ms overlaps: every step is small and the
    # ends come out right, but in between the copy runs 1.2 s late.
    lead, span, drift = _read(monkeypatch, _packets([0.02] * 60 + [-0.02] * 60))
    assert lead == pytest.approx(0.0)
    assert span == pytest.approx(121 * 0.026, abs=1e-6)
    assert drift == pytest.approx(1.2, abs=1e-6)
    assert not da._copy_matches_decode((lead, span, drift), 121 * 0.026)


def test_a_steady_recording_has_no_drift(monkeypatch):
    lead, span, drift = _read(monkeypatch, _packets([0.0] * 120))
    assert drift == pytest.approx(0.0, abs=1e-9)
    assert da._copy_matches_decode((lead, span, drift), 121 * 0.026)


def test_one_jump_is_drift(monkeypatch):
    _, _, drift = _read(monkeypatch, _packets([0.0] * 30 + [0.5] + [0.0] * 30))
    assert drift == pytest.approx(0.5, abs=1e-6)
