"""
Pass 3: follow a recording whose speed bends inside pass 2's straight lines.

A broadcast whose time compression changes within an act bends between pass 2's
line clusters, and the map is off by up to the gap (Law & Order: SVU S01E07,
2026-10-16: 0.38 s late at 17:22, in sync either side). These pin what pass 3
must do on synthetic features shaped like the engine's (210 frames a second,
three smoothed features scaled to unit spread): follow a smooth slip, leave a
right map bit for bit, never follow noise or an isolated false peak, ask more
evidence of a larger correction, keep holes and junctions, never change which
recording samples the renderer places, and judge a bend at the anchors it was
built from. Then the renderer's side: the shared replace mask, and passthrough,
which muxes at one constant offset and so must not swallow a bend.
"""
import itertools
import json
import os
import sys

import numpy as np
import pytest
import scipy.signal

import describealaign as da

FPS = 210  # pass 2's frames a second (da.REFINE_FRAMES_PER_SECOND)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(ROOT, 'test_media', 'ask_dad_trimmed.mp4')
AUDIO = os.path.join(ROOT, 'test_media', 'ask_dad_moviesfortheblind_ep_01_trimmed.mp3')


def _video_features(seconds, seed=1):
    """Smoothed noise with the features' shape: (frames, 3), unit spread."""
    rng = np.random.default_rng(seed)
    frames = int(seconds * FPS)
    raw = rng.standard_normal((frames + 400, 3))
    window = scipy.signal.windows.hann(17)[1:-1]
    window /= window.sum()
    smooth = np.stack([np.convolve(raw[:, j], window, mode='same') for j in range(3)], axis=1)[200:-200]
    return (smooth / smooth.std(axis=0)).astype(np.float32)


def _recording(video, warp_frames, noise=0.25, seed=2):
    """The recording's features: audio frame t holds the video at warp_frames[t], plus noise."""
    rng = np.random.default_rng(seed)
    positions = np.clip(warp_frames, 0, len(video) - 1.001)
    lower = np.floor(positions).astype(int)
    fraction = (positions - lower)[:, None]
    audio = video[lower] * (1 - fraction) + video[lower + 1] * fraction
    return (audio + noise * rng.standard_normal(audio.shape)).astype(np.float32)


def _bump(seconds, centre, width, size_ms):
    return size_ms / 1000. * np.exp(-((seconds - centre) / width) ** 2)


def _line_case(length=120., offset=3., slip=None, seed=1):
    """A recording on one straight line at native rate, its features, and pass 2's map.
    slip(seconds) -> seconds is how far the truth is off that line."""
    video = _video_features(length + offset + 6, seed)
    audio_seconds = np.arange(int(length * FPS)) / FPS
    truth = audio_seconds + offset + (slip(audio_seconds) if slip else 0.)
    audio = _recording(video, truth * FPS, seed=seed + 1)
    xs = np.array([0., length - 1. / FPS])
    ys = xs + offset
    return xs, ys, audio, video, (audio_seconds, truth)


def _mapped(xs, ys, seconds):
    keep = [0]
    for i in range(1, len(xs)):
        if xs[i] > xs[keep[-1]] and ys[i] > ys[keep[-1]]:
            keep.append(i)
    return np.interp(seconds, np.asarray(xs)[keep], np.asarray(ys)[keep])


# ── following a slip, and leaving a right map alone ─────────────────────────

def test_a_slip_inside_one_line_is_followed():
    xs, ys, audio, video, (seconds, truth) = _line_case(slip=lambda s: _bump(s, 60., 8., 250.))
    new_xs, new_ys, report = da.refine_alignment(xs, ys, audio, video, 1.0)
    assert report['bent'] >= 1
    inside = (seconds > 52) & (seconds < 68)
    error_ms = np.abs(_mapped(new_xs, new_ys, seconds[inside]) - truth[inside]) * 1000
    assert error_ms.max() <= 20, f'{error_ms.max():.1f} ms off after pass 3'
    before_ms = np.abs(_mapped(xs, ys, seconds[inside]) - truth[inside]) * 1000
    assert before_ms.max() > 200
    # pass 2's own nodes keep their values; far from the slip the map stays within
    # the node placement tolerance of pass 2's
    assert new_xs[0] == xs[0] and new_ys[0] == ys[0] and new_xs[-1] == xs[-1] and new_ys[-1] == ys[-1]
    away = (seconds < 25) | (seconds > 95)
    drift_ms = np.abs(_mapped(new_xs, new_ys, seconds[away]) - _mapped(xs, ys, seconds[away])) * 1000
    assert drift_ms.max() <= da.REFINE_TOLERANCE_FRAMES / FPS * 1000 + 1
    # what the renderer places is unchanged, and added nodes keep their distance on both axes
    assert da._replaced_recording_samples(new_xs, new_ys) == da._replaced_recording_samples(xs, ys)
    assert np.min(np.diff(new_xs)) >= da.REFINE_MIN_NODE_GAP_SECONDS
    assert np.min(np.diff(new_ys)) >= da.REFINE_MIN_NODE_GAP_SECONDS
    check = report['sync_check']
    assert check['within_pct'] > report['sync_check_line']['within_pct']


def test_a_right_map_is_left_bit_for_bit():
    xs, ys, audio, video, _ = _line_case()
    new_xs, new_ys, report = da.refine_alignment(xs, ys, audio, video, 1.0)
    assert report['stretches'] == 0 and report['bent'] == 0
    assert np.array_equal(new_xs, xs) and np.array_equal(new_ys, ys)
    assert report['sync_check']['within_pct'] == 100.


def test_noise_gives_no_anchors_and_no_bend():
    xs, ys, _, video, _ = _line_case()
    rng = np.random.default_rng(9)
    noise = _video_features(121., seed=7)[:int(120 * FPS)]
    noise = (noise + 0.25 * rng.standard_normal(noise.shape)).astype(np.float32)
    new_xs, new_ys, report = da.refine_alignment(xs, ys, noise, video, 1.0)
    assert report['bent'] == 0 and np.array_equal(new_ys, ys)
    assert report['sync_check_line']['anchored_pct'] < 2


def test_an_isolated_false_peak_is_not_followed():
    # Half a second of the recording matches the video a second later, strongly.
    xs, ys, audio, video, (seconds, truth) = _line_case()
    false = (seconds >= 60.) & (seconds < 60.5)
    audio[false] = _recording(video, (truth[false] + 1.0) * FPS, noise=0.05, seed=5)
    new_xs, new_ys, report = da.refine_alignment(xs, ys, audio, video, 1.0)
    assert report['bent'] == 0
    assert np.array_equal(new_ys, ys)


def test_a_recording_sharing_only_the_theme_confirms_only_the_theme():
    # Another episode's recording: the same 20 s opening theme, then other
    # content. Pass 2 can map the theme alone (Phineas and Ferb S02E27 against
    # "The Beak"): nearly every measured point anchors, but the theme is all of
    # the video it confirms.
    length = 120.
    video = _video_features(length + 10., seed=21)
    seconds = np.arange(int(length * FPS)) / FPS
    theme = seconds < 20.
    audio = _video_features(length + 10., seed=22)[:len(seconds)].copy()
    audio[theme] = _recording(video, (seconds[theme] + 3.) * FPS, seed=23)
    xs = np.array([0., 20., 20.5])
    ys = np.array([3., 23., 113.])
    _, _, report = da.refine_alignment(xs, ys, audio, video, 1.0)
    check = report['sync_check']
    assert check['anchored_pct'] > 90
    assert check['confirmed_pct'] < 20
    # the right recording, mapped throughout, confirms most of the video
    right_xs, right_ys, right_audio, right_video, _ = _line_case(length=length, seed=21)
    _, _, report = da.refine_alignment(right_xs, right_ys, right_audio, right_video, 1.0)
    assert report['sync_check']['confirmed_pct'] > 80


# ── evidence: more anchors for a larger correction ─────────────────────────

@pytest.mark.parametrize('size_ms, anchors, bent', [
    (50., 3, False), (50., 4, True),            # up to ~100 ms: 4 anchors
    (250., 7, False), (250., 8, True),          # up to ~300 ms: 8
    (400., 15, False), (400., 16, True),        # beyond: 16
])
def test_the_evidence_a_correction_needs(size_ms, anchors, bent):
    points = 60
    lag = np.zeros(points)
    lag[20:40] = size_ms / 1000. * FPS            # a plateau 20 points long
    anchor = np.zeros(points, dtype=bool)
    anchor[:20] = anchor[40:] = True              # the line either side is anchored
    anchor[20:20 + anchors] = True
    _, stretches = da._refine_correction(lag, anchor)
    assert bool(stretches) is bent


def test_the_frame_rate_is_pass_2s():
    assert da.REFINE_FRAMES_PER_SECOND == FPS


def test_small_differences_are_left_alone():
    lag = np.full(60, da.REFINE_ACTIVATE_FRAMES - 0.5)
    _, stretches = da._refine_correction(lag, np.ones(60, dtype=bool))
    assert stretches == []


# ── the map's structure ─────────────────────────────────────────────────────

def _three_runs_case():
    """Runs A (0-40 s) and B (40.3-80 s) joined by a junction (0.3 s of recording
    into 0.05 s of video), a hole (3 s of recording into 8 s of video), run C.
    The truth follows the map, plus a slip centred on the junction."""
    xs = np.array([0., 40., 40.3, 80., 83., 120.])
    ys = np.array([3., 43., 43.05, 82.75, 90.75, 127.75])
    video = _video_features(140., seed=3)
    seconds = np.arange(int(120 * FPS)) / FPS
    truth = _mapped(xs, ys, seconds) + _bump(seconds, 40., 8., 200.)
    audio = _recording(video, truth * FPS, seed=4)
    return xs, ys, audio, video


def test_holes_and_junctions_keep_their_shape():
    xs, ys, audio, video = _three_runs_case()
    mask = da._replace_mask(xs, ys)
    assert list(mask) == [True, False, True, False, True]
    new_xs, new_ys, report = da.refine_alignment(xs, ys, audio, video, 1.0)
    assert report['bent'] >= 1
    assert da._replaced_recording_samples(new_xs, new_ys) == da._replaced_recording_samples(xs, ys)
    # the junction: same recording span, video span within a millisecond of pass 2's
    start = int(np.flatnonzero(new_xs == 40.)[0])
    assert new_xs[start + 1] == 40.3
    assert abs((new_ys[start + 1] - new_ys[start]) - 0.05) < 0.001
    # the hole is untouched: both its ends are outside the slip
    hole = int(np.flatnonzero(new_xs == 80.)[0])
    assert new_xs[hole + 1] == 83. and new_ys[hole + 1] == 90.75


def test_a_backwards_pair_pass_2_left_is_kept_and_does_not_block_a_bend():
    # Pass 2 can leave two nodes a millisecond backwards where lines meet
    # (S01E07 node 128); the slip is elsewhere.
    base_xs, base_ys, audio, video, _ = _line_case(slip=lambda s: _bump(s, 80., 8., 200.))
    xs = np.array([0., 30.0005, 30.0000, base_xs[-1]])
    ys = np.array([3., 33.0005, 33.0000, base_ys[-1]])
    new_xs, new_ys, report = da.refine_alignment(xs, ys, audio, video, 1.0)
    assert report['bent'] >= 1
    assert new_xs[1] == 30.0005 and new_xs[2] == 30.0000
    assert new_ys[1] == 33.0005 and new_ys[2] == 33.0000


def _case_from_map(xs, ys, slip, length, seed):
    video = _video_features(length + 12., seed=seed)
    seconds = np.arange(int(length * FPS)) / FPS
    truth = _mapped(xs, ys, seconds) + slip(seconds)
    audio = _recording(video, truth * FPS, seed=seed + 1)
    return audio, video, seconds, truth


def test_a_junction_at_the_rate_gate_moves_rigidly():
    # Junction J squeezes 1.103 s of recording into 1.0 s of video, just past the
    # renderer's 10 % gate (as SVU S10E03's at 1159.6 s did). The slip changes
    # across it, so bending each node freely would bring J under the gate and the
    # renderer would start placing it: the stretch must be kept by moving J whole.
    xs = np.array([0., 40., 41.103, 80.])
    ys = np.array([3., 43., 44., 82.897])
    assert list(da._replace_mask(xs, ys)) == [True, False, True]
    slip = lambda s: _bump(s, 44., 6., 200.)
    audio, video, seconds, truth = _case_from_map(xs, ys, slip, 80., seed=11)
    new_xs, new_ys, report = da.refine_alignment(xs, ys, audio, video, 1.0)
    assert report['bent'] >= 1
    assert da._replaced_recording_samples(new_xs, new_ys) == da._replaced_recording_samples(xs, ys)
    start = int(np.flatnonzero(new_xs == 40.)[0])
    assert new_xs[start + 1] == 41.103
    assert (new_ys[start + 1] - new_ys[start]) == pytest.approx(1.0, abs=1e-9)
    inside = (seconds > 44) & (seconds < 52)
    error_ms = np.abs(_mapped(new_xs, new_ys, seconds[inside]) - truth[inside]) * 1000
    assert error_ms.max() <= 20


def test_a_slip_at_a_chain_end_is_eased_to_keep_the_hole():
    # The chain ends at a hole that squeezes 2.5 s of recording into 2.25 s of
    # video, just past the gate. The slip runs into the chain's end; moving the
    # end node would let the renderer place the hole, so the correction eases to
    # 0 before it and the rest of the slip is still corrected.
    xs = np.array([0., 60., 62.5, 100.])
    ys = np.array([3., 63., 65.25, 102.75])
    assert list(da._replace_mask(xs, ys)) == [True, False, True]
    slip = lambda s: np.where(s < 60.5, _bump(s, 58., 6., -150.), 0.)
    audio, video, seconds, truth = _case_from_map(xs, ys, slip, 100., seed=13)
    new_xs, new_ys, report = da.refine_alignment(xs, ys, audio, video, 1.0)
    assert report['bent'] >= 1
    assert da._replaced_recording_samples(new_xs, new_ys) == da._replaced_recording_samples(xs, ys)
    end = int(np.flatnonzero(new_xs == 60.)[0])
    assert new_ys[end] == 63. and new_xs[end + 1] == 62.5 and new_ys[end + 1] == 65.25
    inside = (seconds > 48) & (seconds < 55)
    error_ms = np.abs(_mapped(new_xs, new_ys, seconds[inside]) - truth[inside]) * 1000
    assert error_ms.max() <= 20


def test_a_slip_that_runs_up_to_a_junction_is_corrected_up_to_it():
    # SVU S10E03 at 4:13-4:18: the recording drifts to 87 ms off over five
    # seconds and is right again past pass 2's junction. The correction has to
    # run all the way to the junction, not stop a grid point short.
    xs = np.array([0., 50., 50.3, 100.])
    ys = np.array([3., 53., 53.35, 103.05])
    assert list(da._replace_mask(xs, ys)) == [True, False, True]
    slip = lambda s: np.where(s < 50.15, -0.08 * np.clip((s - 44.) / 6., 0., 1.), 0.)
    audio, video, seconds, truth = _case_from_map(xs, ys, slip, 100., seed=17)
    new_xs, new_ys, report = da.refine_alignment(xs, ys, audio, video, 1.0)
    assert report['bent'] >= 1
    inside = (seconds > 46.) & (seconds < 49.4)
    error_ms = np.abs(_mapped(new_xs, new_ys, seconds[inside]) - truth[inside]) * 1000
    assert error_ms.max() <= 20, f'{error_ms.max():.1f} ms off before the junction'


def test_added_nodes_keep_their_distance_in_the_video_too():
    # A recording 6 % longer than the video: 2.1 s of it lands on 1.98 s of
    # picture, under the renderer's 2 s. Spacing only along the recording would
    # make short pieces the renderer drops, and the bend would be refused.
    length, rate = 120., 1.06
    xs = np.array([0., length - 1. / FPS])
    ys = 3. + xs / rate
    slip = lambda s: _bump(s, 60., 4., 150.)
    audio, video, seconds, truth = _case_from_map(xs, ys, slip, length, seed=19)
    new_xs, new_ys, report = da.refine_alignment(xs, ys, audio, video, 1.0)
    assert report['bent'] >= 1
    assert len(new_xs) > 4
    assert np.min(np.diff(new_ys)) >= da.REFINE_MIN_NODE_GAP_SECONDS
    assert np.min(np.diff(new_xs)) >= da.REFINE_MIN_NODE_GAP_SECONDS
    assert da._replaced_recording_samples(new_xs, new_ys) == da._replaced_recording_samples(xs, ys)


def test_spacing_holds_on_both_axes():
    # 2.15 s apart in the recording is 2.03 s of picture: too close on that axis
    audio = np.array([0., 2.15, 4.3])
    video = audio / 1.06
    assert da._refine_spacing(audio, video, [0, 1, 2], 2.1) == [0, 2]
    assert da._refine_spacing(audio, audio, [0, 1, 2], 2.1) == [0, 1, 2]


def test_a_strong_peak_off_the_path_is_not_an_anchor():
    lags = np.arange(-10, 11, 2)
    scores = np.zeros((3, len(lags)))
    scores[:, 5] = 500.                           # every point peaks at lag 0
    scores[1, 5] = 0.
    scores[1, 9] = 500.                           # but the middle one at +8 frames
    path = np.array([5, 5, 5])                    # the path stays at 0
    _, _, anchor = da._refine_peaks(scores, lags, path)
    assert list(anchor) == [True, False, True]
    _, _, anchor = da._refine_peaks(scores, lags)
    assert list(anchor) == [True, True, True]


def test_if_assembly_ever_fails_the_report_describes_the_map_returned(monkeypatch):
    # Unreachable while every stretch is checked on its own; forced here so the
    # fallback is known to return pass 2's map AND report its measurement whole.
    xs, ys, audio, video, _ = _line_case(slip=lambda s: _bump(s, 60., 8., 250.))
    monkeypatch.setattr(da, '_refine_refusal', lambda *args, **kwargs: None)
    calls = iter(range(10 ** 6))
    monkeypatch.setattr(da, '_replaced_recording_samples', lambda *args, **kwargs: [next(calls)])
    new_xs, new_ys, report = da.refine_alignment(xs, ys, audio, video, 1.0)
    assert report['refused'].get('assembly') == 1 and report['bent'] == 0
    assert np.array_equal(new_xs, xs) and np.array_equal(new_ys, ys)
    assert report['sync_check'] == report['sync_check_line']


# ── the renderer's mask, shared ─────────────────────────────────────────────

def _old_mask(audio_desc_times, video_times):
    """replace_aligned_segments' mask before it moved into _replace_mask (v2.2.8)."""
    x = audio_desc_times
    y = video_times
    x_samples = (x * da.AUDIO_SAMPLE_RATE).astype(int)
    y_samples = (y * da.AUDIO_SAMPLE_RATE).astype(int)
    diff_x_samples = np.diff(x_samples)
    diff_y_samples = np.diff(y_samples)
    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = diff_x_samples / diff_y_samples
    num_segments = len(x) - 1
    replace_mask = np.array([
        (diff_y_samples[i] >= da.MIN_DURATION_TO_REPLACE_SECONDS * da.AUDIO_SAMPLE_RATE
         and np.abs(1 - slopes[i]) <= da.MAX_RATE_RATIO_DIFF_ALIGN)
        for i in range(num_segments)
    ])
    for i in range(1, num_segments - 1):
        if replace_mask[i]:
            continue
        if not (replace_mask[i - 1] and replace_mask[i + 1]):
            continue
        if diff_y_samples[i] >= da.MIN_DURATION_TO_REPLACE_SECONDS * da.AUDIO_SAMPLE_RATE:
            continue
        if np.abs(1 - slopes[i]) > da.MAX_RATE_RATIO_DIFF_ALIGN:
            continue
        replace_mask[i] = True
    return replace_mask


@pytest.mark.parametrize('seed', range(40))
def test_the_shared_mask_is_the_renderers_old_rule(seed):
    rng = np.random.default_rng(seed)
    count = int(rng.integers(2, 40))
    video_lengths = rng.choice([0., 0.01, 0.5, 1.99, 2.0, 2.01, 3., 30.], size=count)
    rates = rng.choice([1.0, 0.9, 1.1, 0.8999, 1.1001, 0.95, 1.05, 1.3, 0.5], size=count)
    ys = np.concatenate(([5.], 5. + np.cumsum(video_lengths)))
    xs = np.concatenate(([1.], 1. + np.cumsum(video_lengths * rates)))
    assert list(da._replace_mask(xs, ys)) == list(_old_mask(xs, ys))


def test_exactly_two_seconds_of_video_is_replaced_and_a_sample_less_is_not():
    two = 2 * da.AUDIO_SAMPLE_RATE / da.AUDIO_SAMPLE_RATE
    assert da._replace_mask(np.array([0., two]), np.array([0., two]))[0]
    short = (2 * da.AUDIO_SAMPLE_RATE - 1) / da.AUDIO_SAMPLE_RATE
    assert not da._replace_mask(np.array([0., short]), np.array([0., short]))[0]


# ── a bend may never change which recording samples are placed ─────────────

def _refusal(xs, ys, first, last, chain_ys):
    chain_ys = np.asarray(chain_ys, dtype=float)
    added = np.zeros(last - first + 1, dtype=bool)
    return da._refine_refusal(np.asarray(xs, float), np.asarray(ys, float), first, last,
                              np.asarray(xs[first:last + 1], float), chain_ys, added)


def test_a_piece_pushed_under_two_seconds_is_refused():
    # hole, then a 2.05 s piece, then a long piece; shrinking the piece's video
    # span to 1.95 s would stop the renderer placing it
    xs = [0., 1., 3.05, 30.]
    ys = [0., 4., 6.05, 33.]
    assert list(da._replace_mask(np.array(xs), np.array(ys))) == [False, True, True]
    assert _refusal(xs, ys, 1, 3, [4.1, 6.05, 33.]) == "replaced samples"
    assert _refusal(xs, ys, 1, 3, [4.02, 6.05, 33.]) is None


@pytest.mark.parametrize('rate, bent_rate', [(1.099, 1.101), (0.901, 0.899)])
def test_a_piece_pushed_across_the_rate_gate_is_refused(rate, bent_rate):
    # a 3 s piece at the edge of the renderer's 10 % rate gate, next to a hole
    xs = [0., 1., 1. + 3. * rate, 40.]
    ys = [0., 5., 8., 8. + (40. - 1. - 3. * rate)]
    assert list(da._replace_mask(np.array(xs), np.array(ys))) == [False, True, True]
    bent_end = 5. + (xs[2] - xs[1]) / bent_rate
    assert _refusal(xs, ys, 1, 3, [5., bent_end, ys[3]]) == "replaced samples"


def test_two_short_junctions_in_a_row_keep_their_fill():
    # Two short segments in a row are not filled (the fill goes in order); moving
    # the long pieces either side must leave them that way.
    xs = [0., 20., 20.5, 21., 40.]
    ys = [0., 20., 20.5, 21., 40.]
    mask = da._replace_mask(np.array(xs), np.array(ys))
    assert list(mask) == [True, False, False, True]
    assert _refusal(xs, ys, 0, 4, [0.02, 20.02, 20.52, 21.02, 40.02]) is None


def test_a_loss_and_an_equal_gain_are_refused():
    # Piece P (2.05 s, next to a hole) would drop out while junction J (2 s of
    # recording squeezed past the rate gate) would come in: the summed replaced
    # time is nearly unchanged, but the samples are not the same.
    xs = [0., 1., 3.05, 5.05, 7.10, 40.]
    ys = [0., 4., 6.05, 7.70, 9.80, 42.70]
    assert list(da._replace_mask(np.array(xs), np.array(ys))) == [False, True, False, True, True]
    before = da._replaced_recording_samples(np.array(xs), np.array(ys))
    after_ys = [0., 4.1, 6.05, 8.10, 10.25, 42.70]   # P 1.95 s of video, J 2.05 s
    assert list(da._replace_mask(np.array(xs), np.array(after_ys))) == [False, False, True, True, True]
    after = da._replaced_recording_samples(np.array(xs), np.array(after_ys))
    total = lambda runs: sum(end - start for start, end in runs)
    assert abs(total(after) - total(before)) <= 0.5 * da.AUDIO_SAMPLE_RATE
    assert after != before
    assert _refusal(xs, ys, 1, 5, after_ys[1:]) == "replaced samples"


def test_a_reversal_is_refused():
    xs = [0., 10., 20., 30.]
    ys = [0., 10., 20., 30.]
    assert _refusal(xs, ys, 0, 3, [0., 10., 9.9, 30.]) == "node order"


# ── a bend is judged at the anchors it was built from ──────────────────────

def test_the_verdict_counts_the_anchors_a_bend_loses():
    before = np.full(20, 20.)                     # 20 anchors ~95 ms off
    after = np.zeros(20)
    margins = np.full(20, 200.)
    assert da._refine_verdict(before, after, margins) is None
    # A third of them lose their peak: picking fresh anchors afterwards would
    # hide that; judged at the same anchors it is refused.
    margins[:7] = 10.
    assert da._refine_verdict(before, after, margins) == "lost evidence"


def test_the_verdict_wants_a_real_improvement():
    before = np.full(20, 10.7)                    # ~51 ms
    margins = np.full(20, 200.)
    assert da._refine_verdict(before, np.full(20, 9.0), margins) == "measured no better"
    assert da._refine_verdict(before, np.full(20, 2.0), margins) is None
    far = np.full(20, 2.0)
    far[0] = 30.                                  # a new anchor over 120 ms off
    assert da._refine_verdict(before, far, margins) == "measured no better"


def test_one_anchor_mended_cannot_pay_for_one_spoiled():
    # Before: one anchor 190 ms off, three on time. After: that one on time, but
    # another 124 ms off. The mean improves and the count over 120 ms stays one;
    # anchor by anchor, a good one went bad, so the bend is refused.
    before = np.array([40., 0., 0., 0.])
    after = np.array([0., 26., 0., 0.])
    margins = np.full(4, 200.)
    assert da._refine_verdict(before, after, margins) == "measured no better"


def test_passthrough_needs_the_peaks_to_stay_strong():
    line = np.zeros(20)
    near_zero = np.full(20, 1.0)
    strong = np.full(20, 200.)
    ok, tally = da._refine_passthrough_verdict(line, near_zero, strong)
    assert ok and tally['anchors'] == 20 and tally['within_40ms'] == 20
    # the winning lags stay near zero, but most peaks are too weak to stand on
    weak = np.full(20, 10.)
    weak[:3] = 200.
    ok, tally = da._refine_passthrough_verdict(line, near_zero, weak)
    assert not ok
    assert tally['anchors'] == 3 and tally['within_40ms'] == 3      # only what is supported counts
    # one anchor on time under the map lands 130 ms off at the constant offset,
    # while another comes on time: as many within 40 ms, but a good one went bad
    line_mixed = np.zeros(20)
    line_mixed[0] = 10.                           # 48 ms under the map
    off = near_zero.copy()
    off[0] = 0.
    off[5] = 27.
    assert da._refine_passthrough_verdict(line_mixed, off, strong)[0] is False


# ── the path and the scores ─────────────────────────────────────────────────

@pytest.mark.parametrize('seed', range(5))
def test_the_path_is_the_brute_force_optimum(seed):
    rng = np.random.default_rng(seed)
    lags = np.arange(-6, 7, 2)                    # 7 lags at the engine's 2-frame step
    scores = rng.normal(0, 30, size=(6, len(lags)))
    change = da.REFINE_LAG_CHANGE_COST
    prior = da.REFINE_LAG_COST

    def objective(path):
        total = sum(scores[t, j] - prior * abs(lags[j]) for t, j in enumerate(path))
        return total - change * sum(abs(lags[b] - lags[a]) for a, b in zip(path, path[1:]))

    best = max(objective(p) for p in itertools.product(range(len(lags)), repeat=len(scores)))
    assert objective(list(da._refine_path(scores, lags))) == pytest.approx(best)


def test_the_fast_scores_find_the_exhaustive_peaks():
    xs, ys, audio, video, (seconds, truth) = _line_case(slip=lambda s: _bump(s, 60., 8., 150.))
    grid = np.arange(40 * FPS, 80 * FPS, da.REFINE_GRID_FRAMES).astype(np.int64)
    map_audio, map_video = xs * FPS, ys * FPS
    scores, lags = da._refine_scores(audio, video, map_audio, map_video, grid)
    fast_lag, _, anchor = da._refine_peaks(scores, lags)
    # exhaustive: every frame, every lag
    half = da.REFINE_HALF_WINDOW_FRAMES
    exact = []
    for point in grid:
        frames = np.arange(point - half, point + half + 1)
        mapped = np.interp(frames, map_audio, map_video)
        best_lag, best_score = None, -np.inf
        for lag in range(-40, 41):
            position = mapped + lag
            lower = np.floor(position).astype(int)
            fraction = (position - lower)[:, None]
            v = video[lower] * (1 - fraction) + video[lower + 1] * fraction
            score = np.sum(-.5 - np.log10(1e-4 + np.abs(audio[frames] - v)))
            if score > best_score:
                best_lag, best_score = lag, score
        exact.append(best_lag)
    exact = np.array(exact, dtype=float)
    assert np.max(np.abs(fast_lag[anchor] - exact[anchor])) <= 1.0
    # and the same scale: the margins anchors are judged by are full-window sums
    point = grid[len(grid) // 2]
    frames = np.arange(point - half, point + half + 1)
    mapped = np.interp(frames, map_audio, map_video)
    lag = int(round(exact[len(grid) // 2] / 2.)) * 2
    lower = np.floor(mapped + lag).astype(int)
    fraction = (mapped + lag - lower)[:, None]
    v = video[lower] * (1 - fraction) + video[lower + 1] * fraction
    full_sum = np.sum(-.5 - np.log10(1e-4 + np.abs(audio[frames] - v)))
    fast_sum = scores[len(grid) // 2, int(np.flatnonzero(lags == lag)[0])]
    assert fast_sum == pytest.approx(full_sum, rel=0.15)


# ── passthrough must not swallow a bend ─────────────────────────────────────

def _identity_map(length=100.):
    return np.array([0., length]), np.array([0.25, length + 0.25])


def test_passthrough_decision():
    xs, ys = _identity_map()
    decide = lambda refinement: da._passthrough_decision(xs, ys, 1.0, True, False, refinement)
    assert decide(None) is True                                   # pass 3 off: 2.2.8's answer
    assert decide({"applied": False}) is True
    assert decide({"applied": True, "bent": 1, "chains": 2, "passthrough_ok": True}) is False
    assert decide({"applied": True, "bent": 0, "chains": 2, "passthrough_ok": True}) is True
    assert decide({"applied": True, "bent": 0, "chains": 2, "passthrough_ok": False}) is False
    assert decide({"applied": True, "bent": 0, "chains": 2, "passthrough_ok": None}) is False
    # pass 3 measured nothing (only stretches too short to measure): 2.2.8's answer
    assert decide({"applied": True, "bent": 0, "chains": 0, "passthrough_ok": None}) is True
    assert da._passthrough_decision(xs, ys, 1.0, False, False, None) is False


@pytest.mark.parametrize('drift_ms, ok', [(60., False), (10., True)])
def test_passthrough_needs_the_constant_offset_to_measure_as_well(drift_ms, ok):
    # The map follows the truth, which drifts drift_ms over two minutes: every
    # slope is within passthrough's 0.1 %, so 2.2.8 would mux it at one offset.
    length = 120.
    xs, ys, audio, video, _ = _line_case(length=length, slip=lambda s: drift_ms / 1000. * s / length)
    ys = ys + np.array([0., drift_ms / 1000.])
    slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    assert da.is_passthrough_alignment(xs, ys, slope)
    _, _, report = da.refine_alignment(xs, ys, audio, video, slope)
    assert report['bent'] == 0
    assert report['passthrough_ok'] is ok


def _binaries():
    try:
        return da.get_ffmpeg(), da.get_ffprobe()
    except Exception as error:  # no ffmpeg on this machine and no download
        pytest.skip(f'ffmpeg unavailable: {error}')


@pytest.mark.parametrize('bent, expected', [(1, 'replaced'), (0, 'passthrough')])
def test_combine_takes_the_branch_the_decision_gives(tmp_path, monkeypatch, bent, expected):
    _binaries()
    length = 30.

    def fake_align(video_features, audio_desc_features, video_energy, audio_desc_energy,
                   refine=True, refinement=None):
        xs, ys = np.array([0., length]), np.array([0., length])
        refinement.update({"applied": True, "bent": bent, "stretches": bent, "chains": 1,
                           "refused": {}, "bent_seconds": 0., "max_correction_ms": 0.,
                           "passthrough_ok": True, "line_audio_times": xs, "line_video_times": ys,
                           "sync_check": None, "sync_check_line": None})
        return xs, ys, 70., np.zeros((4, 5)), 1.0

    taken = []

    def spy(name):
        def write(output_filename, *args, **kwargs):
            taken.append(name)
            with open(output_filename, 'wb') as file:
                file.write(b'x' * 200_000)
        return write

    monkeypatch.setattr(da, 'align', fake_align)
    monkeypatch.setattr(da, 'write_passthrough_media_to_disk', spy('passthrough'))
    monkeypatch.setattr(da, 'write_replaced_media_to_disk', spy('replaced'))
    monkeypatch.setattr(da, 'replace_aligned_segments', lambda *a, **k: None)
    da.combine(VIDEO, AUDIO, stretch_audio=True, yes=True, output_dir=str(tmp_path / 'out'),
               alignment_dir=str(tmp_path / 'align'))
    assert taken == [expected]
    report = json.load(open(next((tmp_path / 'align').glob('*.json'))))
    assert report['passthrough'] is (expected == 'passthrough')


# ── reports and switches ────────────────────────────────────────────────────

def test_the_report_keeps_pass_2s_segments_and_adds_the_refined_map(tmp_path):
    line_xs, line_ys = np.array([0., 50., 100.]), np.array([1., 51., 101.])
    bent_xs, bent_ys = np.array([0., 25., 50., 100.]), np.array([1., 26.1, 51., 101.])
    refinement = {"applied": True, "chains": 1, "stretches": 2, "bent": 1, "refused": {"lost evidence": 1},
                  "bent_seconds": 12.5, "max_correction_ms": 100.,
                  "line_audio_times": line_xs, "line_video_times": line_ys,
                  "sync_check": da.sync_check_percentages({"points": 100, "anchors": 80, "within_40ms": 78, "over_120ms": 0}),
                  "sync_check_line": da.sync_check_percentages({"points": 100, "anchors": 80, "within_40ms": 50, "over_120ms": 4})}
    base = str(tmp_path / 'report')
    da.plot_alignment(base, np.zeros((4, 5)), bent_xs, bent_ys, 55.0, 1.0, True, False,
                      refinement=refinement, passthrough=False)
    report = json.load(open(base + '.json'))
    assert len(report['segments']) == 2 and len(report['refined_segments']) == 3
    assert report['segments'][0]['video_end_sec'] == 51.
    assert report['segments'][0]['audio_end_sec'] == 50.
    assert report['refined_segments'][0]['audio_end_sec'] == 25.
    assert report['refinement'] == {"applied": True, "chains": 1, "stretches": 2, "bent": 1,
                                    "refused": {"lost evidence": 1}, "bent_seconds": 12.5,
                                    "max_correction_ms": 100.}
    assert report['sync_check']['within_pct'] == pytest.approx(97.5)
    assert report['sync_check_line']['over_120ms'] == 4
    assert report['passthrough'] is False
    text = open(base + '.txt').read()
    assert 'Refinement: bent 1 of 2 stretches' in text and 'Sync check: 80.0% of 100 points anchored' in text
    assert 'Input file similarity: 55.00%' in text


def test_without_pass_3_the_report_says_so(tmp_path):
    xs, ys = np.array([0., 100.]), np.array([1., 101.])
    base = str(tmp_path / 'report')
    da.plot_alignment(base, np.zeros((4, 5)), xs, ys, 55.0, 1.0, True, False,
                      refinement={"applied": False}, passthrough=True)
    report = json.load(open(base + '.json'))
    assert report['refinement'] == {"applied": False}
    assert report['sync_check'] is None and report['sync_check_line'] is None
    assert report['refined_segments'] == report['segments']
    assert 'Refinement' not in open(base + '.txt').read()


def test_no_refine_reaches_align(monkeypatch, tmp_path):
    _binaries()
    seen = []

    def fake_align(video_features, audio_desc_features, video_energy, audio_desc_energy,
                   refine=True, refinement=None):
        seen.append(refine)
        raise da.AlignmentMismatchError("stop here", diagnostic={"summary": "stop"})

    monkeypatch.setattr(da, 'align', fake_align)
    for no_refine in (True, False):
        with pytest.raises(da.AlignmentMismatchError):
            da.combine(VIDEO, AUDIO, stretch_audio=True, yes=True, output_dir=str(tmp_path / 'out'),
                       alignment_dir=str(tmp_path / 'align'), no_refine=no_refine)
    assert seen == [False, True]


def test_the_command_line_passes_no_refine(monkeypatch):
    calls = []
    monkeypatch.setattr(da, 'combine', lambda *a, **k: calls.append(k))
    monkeypatch.setattr(sys, 'argv', ['describealaign', 'video.mp4', 'audio.mp3', '--no_refine'])
    da.command_line_interface()
    monkeypatch.setattr(sys, 'argv', ['describealaign', 'video.mp4', 'audio.mp3'])
    da.command_line_interface()
    assert [call['no_refine'] for call in calls] == [True, False]


def test_align_reports_pass_2s_map_and_measures_it(tmp_path):
    # On the real test media: the map pass 3 was given is exactly pass 2's (the
    # same as with pass 3 off), and it was measured.
    _binaries()
    video = da.parse_audio_from_file(VIDEO, 2, cache_dir=str(tmp_path))
    video_features = [da.get_energy(video), da.get_zero_crossings(video)] + da.get_freq_bands(video)
    del video
    audio = da.parse_audio_from_file(AUDIO, 2, cache_dir=str(tmp_path), honor_pts=False)
    audio_features = [da.get_energy(audio), da.get_zero_crossings(audio)] + da.get_freq_bands(audio)
    del audio
    refinement = {}
    xs, ys, *_ = da.align(video_features, audio_features, video_features[0], audio_features[0],
                          refine=True, refinement=refinement)
    off = da.align(video_features, audio_features, video_features[0], audio_features[0], refine=False)
    assert np.array_equal(refinement['line_audio_times'], off[0])
    assert np.array_equal(refinement['line_video_times'], off[1])
    assert refinement['applied'] and refinement['sync_check_line']['anchors'] > 0
    if refinement['bent'] == 0:
        assert np.array_equal(xs, off[0]) and np.array_equal(ys, off[1])
