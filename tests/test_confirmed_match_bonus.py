"""
Pass-2 confirmed-match bonus (v2.2.3).

align() re-scores each candidate line sample-by-sample and pays a flat
penalty to jump lines. When video and description are different cuts of a
film, a line confirmed by pass 1 over tens of seconds could lose to "stay on
the old line, then skip" — heard as narration 19.5 s early followed by a
minute of undescribed picture (Scary Movie 3, 2026-09-14). Samples pass 1
matched now carry CONFIRMED_MATCH_BONUS so a confirmed line cannot be skipped
cheaply. These tests pin the helper that places the bonus.
"""
import numpy as np

import describealaign as da


def test_bonus_lands_only_on_confirmed_samples():
    confirmed = np.array([102, 105, 110])
    bonuses = da.confirmed_match_bonuses(confirmed, start=100, length=20)
    assert bonuses.shape == (20,)
    hits = np.flatnonzero(bonuses)
    assert hits.tolist() == [2, 5, 10]
    assert np.all(bonuses[hits] == da.CONFIRMED_MATCH_BONUS)


def test_confirmed_samples_outside_range_are_ignored_not_wrapped():
    # 95 is before the range, 130 is after; neither may wrap to a valid index.
    confirmed = np.array([95, 100, 119, 130])
    bonuses = da.confirmed_match_bonuses(confirmed, start=100, length=20)
    assert np.flatnonzero(bonuses).tolist() == [0, 19]


def test_bonus_amount_is_overridable_and_default_is_positive():
    bonuses = da.confirmed_match_bonuses(np.array([3]), start=0, length=5, bonus=2.5)
    assert bonuses.tolist() == [0, 0, 0, 2.5, 0]
    assert da.CONFIRMED_MATCH_BONUS > 0


def test_a_confirmed_line_outweighs_one_line_jump():
    """The design point: ~100 confirmed matches (about 35 s of matched content)
    must be worth more than the flat 1000 jump penalty in align()'s pass 2,
    otherwise a confirmed line can still be skipped for free."""
    line_jump_penalty = 1000
    bonuses = da.confirmed_match_bonuses(np.arange(0, 100), start=0, length=100)
    assert bonuses.sum() >= line_jump_penalty
