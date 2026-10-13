"""
Pass 2 of align() keeps its candidate points in flat arrays.

The old code held every candidate as a Python tuple in one list per audio
index, plus a set of seen keys and a dict of backpointers: several hundred
bytes a point. A donor with act-level time compression (Law & Order: SVU's
AudioVault packs) fragments into hundreds of line clusters, each padded by
30 s on both sides, so millions of points: 4-7 GB on one 45-minute episode,
enough to take down the 16 GB host the engine runs on.

The arrays must reproduce the old path exactly. _reference_pass2 below is the
old code, copied from describealaign.py at v2.2.7 (main, lines 2033-2034 and
2062-2117, less `del video_interp`), with only the names it read from
align()'s scope bound at the top.
"""
import itertools
import os
import tracemalloc

import numpy as np
import pytest
from sortedcontainers import SortedList

import describealaign as da

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(ROOT, 'test_media', 'ask_dad_trimmed.mp4')
AUDIO = os.path.join(ROOT, 'test_media', 'ask_dad_moviesfortheblind_ep_01_trimmed.mp3')


def _reference_pass2(blocks, num_clusters, num_audio, num_video):
  line_clusters = range(num_clusters)
  audio_desc_features_scaled = range(num_audio)
  video_features_scaled = range(num_video)
  points = [[] for i in range(len(audio_desc_features_scaled))]
  seen_points = set()
  for x, y, cluster_index, quals, bonuses in blocks:
    for i, j, qual, bonus in zip(x.tolist(), y.tolist(), quals.tolist(), bonuses.tolist()):
      point = (i, int(j))
      if point not in seen_points:
        seen_points.add(point)
        points[i].append((j, cluster_index, qual, bonus))
  del seen_points
  points = [sorted(point) for point in points]
  
  best_so_far = SortedList(key=lambda x:x[0])
  best_so_far.add((0, 0, -1, 0, 0))  # video_index, audio_desc_index, cluster_index, qual, cum_qual
  clusters_best_so_far = [(0, 0, 0, -1000) for cluster in line_clusters]
  backpointers = {}
  prev_cache = np.full((len(video_features_scaled), 5), -np.inf)
  prev_cache[0] = (0, 0, -1, 0, 0)  # video_index, audio_desc_index, cluster_index, qual, cum_qual
  reversed_min_points = [min(x)[0] if len(x) > 0 else np.inf for x in points[::-1]]
  forward_min = list(itertools.accumulate(reversed_min_points, min))[::-1]
  del reversed_min_points
  for i in range(len(audio_desc_features_scaled)):
    for j, cluster_index, qual, bonus in points[i]:
      cur_index = best_so_far.bisect_right((j,))
      prev_j, prev_i, prev_cluster_index, prev_qual, best_prev_cum_qual = best_so_far[cur_index-1]
      cluster_last = clusters_best_so_far[cluster_index]
      if cluster_last[3] >= best_prev_cum_qual:
        prev_j, prev_i, prev_qual, best_prev_cum_qual = cluster_last
        prev_cluster_index = cluster_index
      for prev_j_temp in range(max(0, int(j) - 2), int(j) + 1):
        prev_node = prev_cache[prev_j_temp].tolist()
        if cluster_index != prev_node[2]:
          prev_node[4] -= 100 + 100 * ((j - prev_node[0]) - (i - prev_node[1])) ** 2
        if prev_node[1] >= (i - 2) and \
           prev_node[0] <= j and \
           prev_node[4] >= best_prev_cum_qual:
          prev_j, prev_i, prev_cluster_index, prev_qual, best_prev_cum_qual = prev_node
      # The bonus steers the path (cum_qual) only; the stored per-node qual
      # stays the raw match quality so similarity_percent is unchanged.
      cum_qual = best_prev_cum_qual + qual + bonus
      prev_cache[int(j)] = (j, i, cluster_index, qual, cum_qual)
      cum_qual_jump = cum_qual - 1000
      if best_so_far[cur_index-1][4] < cum_qual_jump:
        while (cur_index < len(best_so_far)) and (best_so_far[cur_index][4] <= cum_qual_jump):
          del best_so_far[cur_index]
        best_so_far.add((j, i, cluster_index, qual, cum_qual_jump))
      if forward_min[i] == j and cur_index > 1:
        del best_so_far[:cur_index-1]
      cum_qual_cluster_jump = cum_qual - 50
      if cluster_last[3] < cum_qual_cluster_jump:
        clusters_best_so_far[cluster_index] = (j, i, qual, cum_qual_cluster_jump)
      backpointers[(j, i)] = (prev_j, prev_i, prev_cluster_index, prev_qual, best_prev_cum_qual)
  
  path = [best_so_far[-1]]
  while path[-1][:2] in backpointers:
    path.append(backpointers[path[-1][:2]])
  path.pop()
  path.reverse()
  path = np.array(path)
  return path


def _columns(blocks):
    """The per-cluster lists align() hands to _pass2_candidates (copies: they get consumed)."""
    return ([b[0].copy() for b in blocks], [b[1].copy() for b in blocks],
            [np.full(len(b[0]), b[2], dtype=np.int64) for b in blocks],
            [b[3].copy() for b in blocks], [b[4].copy() for b in blocks])


def _new_pass2(blocks, num_clusters, num_audio, num_video):
    candidates = da._pass2_candidates(*_columns(blocks))
    return da._pass2_path(*candidates, num_clusters=num_clusters,
                          num_audio=num_audio, num_video=num_video)


def _synthetic_blocks(seed, num_audio=6000, num_video=6400, num_lines=60, max_len=2500):
    """Overlapping candidate lines shaped like align()'s: near-unit slopes, some
    lines skipped (their index still spent), near-duplicates of the previous line
    so (i, int(j)) collides across clusters, some whole-number positions, sparse
    confirmed-match bonuses."""
    rng = np.random.default_rng(seed)
    blocks = []
    slope, offset = 1.0, 0.0
    for cluster_index in range(num_lines):
        if rng.random() < 0.1:
            continue
        if blocks and rng.random() < 0.3:
            offset += rng.uniform(-0.45, 0.45)
        else:
            slope = 1.0 + rng.normal(0, 0.02)
            offset = rng.uniform(-150, 450)
        start = int(rng.integers(0, num_audio - 100))
        x = np.arange(start, min(num_audio, start + int(rng.integers(50, max_len))))
        y = slope * x + offset
        x = x[(y >= 4) & (y < num_video - 4)]
        y = slope * x + offset
        if rng.random() < 0.2:
            y = np.floor(y)
        quals = rng.normal(0.5, 2.0, len(x))
        bonuses = np.where(rng.random(len(x)) < 0.05, 10.0, 0.0)
        blocks.append((x, y, cluster_index, quals, bonuses))
    return blocks, dict(num_clusters=num_lines, num_audio=num_audio, num_video=num_video)


@pytest.mark.parametrize('block_points', [None, 7])
@pytest.mark.parametrize('seed', [1, 2, 3, 4])
def test_path_is_identical_to_the_old_code(seed, block_points, monkeypatch):
    if block_points:  # cross the DP's conversion-block seams hundreds of times
        monkeypatch.setattr(da, '_PASS2_DP_BLOCK_POINTS', block_points)
    blocks, sizes = _synthetic_blocks(seed)
    expected = _reference_pass2(blocks, **sizes)
    actual = _new_pass2(blocks, **sizes)
    assert len(expected) > 100  # a real path, not two empty arrays agreeing
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert np.array_equal(actual, expected)


def _later_lines_lower(seed, num_audio=4000, num_video=4400, num_lines=50):
    """Lines that start later sit lower, so an index's smallest position often
    exceeds a later index's: the case where forward_min has to be the minimum
    over every later index, not just this one, or the DP prunes entries a later
    point still needs."""
    rng = np.random.default_rng(seed)
    blocks = []
    for cluster_index in range(num_lines):
        start = int(rng.integers(0, num_audio - 200))
        offset = 380 - 0.09 * start + rng.normal(0, 25)
        slope = 1.0 + rng.normal(0, 0.01)
        x = np.arange(start, min(num_audio, start + int(rng.integers(100, 1500))))
        y = slope * x + offset
        x = x[(y >= 4) & (y < num_video - 4)]
        y = slope * x + offset
        quals = rng.normal(0.5, 2.0, len(x))
        bonuses = np.where(rng.random(len(x)) < 0.05, 10.0, 0.0)
        blocks.append((x, y, cluster_index, quals, bonuses))
    return blocks, dict(num_clusters=num_lines, num_audio=num_audio, num_video=num_video)


@pytest.mark.parametrize('seed', [1, 3])
def test_path_is_identical_when_later_lines_sit_lower(seed):
    blocks, sizes = _later_lines_lower(seed)
    expected = _reference_pass2(blocks, **sizes)
    actual = _new_pass2(blocks, **sizes)
    assert len(expected) > 100
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert np.array_equal(actual, expected)


def test_no_candidates_gives_the_same_empty_path():
    sizes = dict(num_clusters=3, num_audio=500, num_video=500)
    expected = _reference_pass2([], **sizes)
    actual = _new_pass2([], **sizes)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape == (0,)


@pytest.mark.parametrize('x', [
    np.array([10, 50010], dtype=np.int32),
    np.arange(10, 50011, dtype=np.int32),
])
def test_int32_audio_indices_give_the_same_path(x):
    # np.arange makes int32 on Windows under numpy 1.x. Left as int32, the
    # (i, int(j)) key's multiply overflowed and reordered the candidates.
    blocks = [(x, x.astype(np.float64) + 4, 0, np.full(len(x), 2000.0), np.zeros(len(x)))]
    sizes = dict(num_clusters=1, num_audio=50011, num_video=50019)
    expected = _reference_pass2(blocks, **sizes)
    actual = _new_pass2(blocks, **sizes)
    assert len(expected) > 0
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert np.array_equal(actual, expected)


def test_a_repeated_key_keeps_the_earliest_cluster():
    x = np.arange(10, 15)
    blocks = [
        (x, x + 20.25, 0, np.full(5, 1.0), np.zeros(5)),
        (x, x + 20.75, 1, np.full(5, 9.0), np.zeros(5)),   # same int(j): dropped
        (x, x + 21.5, 2, np.full(5, 2.0), np.zeros(5)),    # next int(j): kept
    ]
    i, j, cluster, qual, bonus = da._pass2_candidates(*_columns(blocks))
    assert i.tolist() == [10, 10, 11, 11, 12, 12, 13, 13, 14, 14]
    assert cluster.tolist() == [0, 2] * 5
    assert j.tolist() == [30.25, 31.5, 31.25, 32.5, 32.25, 33.5, 33.25, 34.5, 34.25, 35.5]
    assert qual.tolist() == [1.0, 2.0] * 5


def test_point_lookup_matches_the_old_dict():
    cand_i = np.array([3, 3, 5])
    cand_j = np.array([7.5, 9.0, 2.25])
    assert da._find_pass2_point(cand_i, cand_j, 3, 9.0) == 1
    assert da._find_pass2_point(cand_i, cand_j, 3.0, 7.5) == 0   # prev_cache hands back floats
    assert da._find_pass2_point(cand_i, cand_j, 5, 2.25) == 2
    assert da._find_pass2_point(cand_i, cand_j, 3, 8.0) == -1
    assert da._find_pass2_point(cand_i, cand_j, 3.5, 7.5) == -1
    assert da._find_pass2_point(cand_i, cand_j, 0, 0) == -1      # the origin was never a key
    assert da._find_pass2_point(cand_i, cand_j, 6, 2.25) == -1


def test_lookup_with_a_float_index_does_not_copy_the_candidates():
    # A float search key against the int64 index array made numpy cast the
    # whole array to float64 on every step of the path walk: an 88 MB copy a
    # step on an 11-million-point episode.
    cand_i = np.arange(1_000_000, dtype=np.int64)
    cand_j = cand_i + 0.5
    tracemalloc.start()
    try:
        assert da._find_pass2_point(cand_i, cand_j, 123456.0, 123456.5) == 123456
        assert da._find_pass2_point(cand_i, cand_j, float('inf'), 1.5) == -1
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 64 * 1024, f"{peak} bytes for two lookups"


def test_a_non_finite_position_still_fails():
    x = np.arange(5)
    y = np.array([4.0, 5.0, np.nan, 7.0, 8.0])
    blocks = [(x, y, 0, np.zeros(5), np.zeros(5))]
    with pytest.raises(RuntimeError):
        da._pass2_candidates(*_columns(blocks))


# Under tracemalloc the old structures take ~455 bytes a point on this input
# and the arrays ~130. The bound sits between, with room for numpy versions.
_MAX_BYTES_PER_POINT = 300


def test_memory_per_point_is_bounded():
    blocks, sizes = _synthetic_blocks(7, num_audio=20000, num_video=21000, num_lines=120, max_len=4000)
    num_points = sum(len(b[0]) for b in blocks)
    columns = _columns(blocks)
    tracemalloc.start()
    try:
        candidates = da._pass2_candidates(*columns)
        da._pass2_path(*candidates, **sizes)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak / num_points < _MAX_BYTES_PER_POINT, f"{peak / num_points:.0f} bytes a point"


def _binaries():
    try:
        return da.get_ffmpeg(), da.get_ffprobe()
    except Exception as error:  # no ffmpeg on this machine and no download
        pytest.skip(f'ffmpeg unavailable: {error}')


def test_align_on_real_audio_matches_the_old_code(tmp_path, monkeypatch):
    _binaries()
    seen = {}
    real_candidates, real_path = da._pass2_candidates, da._pass2_path

    def spy_candidates(xs, ys, cluster_ids, quals, bonuses):
        seen['blocks'] = [(x.copy(), y.copy(), int(c[0]) if len(c) else -1, q.copy(), b.copy())
                          for x, y, c, q, b in zip(xs, ys, cluster_ids, quals, bonuses)]
        return real_candidates(xs, ys, cluster_ids, quals, bonuses)

    def spy_path(*args, **sizes):
        seen['sizes'] = sizes
        path = real_path(*args, **sizes)
        seen['path'] = path.copy()  # align() rescales its copy in place
        return path

    monkeypatch.setattr(da, '_pass2_candidates', spy_candidates)
    monkeypatch.setattr(da, '_pass2_path', spy_path)
    video = da.parse_audio_from_file(VIDEO, 2, cache_dir=str(tmp_path))
    video_features = [da.get_energy(video), da.get_zero_crossings(video)] + da.get_freq_bands(video)
    del video
    audio = da.parse_audio_from_file(AUDIO, 2, cache_dir=str(tmp_path), honor_pts=False)
    audio_features = [da.get_energy(audio), da.get_zero_crossings(audio)] + da.get_freq_bands(audio)
    del audio
    da.align(video_features, audio_features, video_features[0], audio_features[0])
    # Each block is one line cluster, labelled with its index in line_clusters.
    labels = [b[2] for b in seen['blocks']]
    assert len(labels) > 1 and labels == sorted(set(labels))
    assert labels[-1] < seen['sizes']['num_clusters']
    expected = _reference_pass2(seen['blocks'], **seen['sizes'])
    assert len(expected) > 0
    assert np.array_equal(seen['path'], expected)
