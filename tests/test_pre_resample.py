"""
Property tests for the PAL/NTSC pre-resample detection.

The detector must:
  - Match known framerate-conversion ratios within ±0.05%.
  - Reject anything else, including commercial-break-induced
    duration-ratio surpluses that look superficially similar
    (e.g. 4 minutes of seam in a 90-min movie ≈ 4.4% surplus,
    which sits between PAL/NTSC and noise).
  - Return the *exact* mathematical target so the resample is
    perfect, not just approximate.
"""
import pytest

import describealaign as da


def test_detects_pal_to_ntsc():
    # Audio dur / video dur for a PAL-source AD aligned to NTSC video.
    target, name = da.detect_framerate_conversion(23.976 / 25.0)
    assert target is not None
    assert "PAL" in name and "NTSC" in name
    assert target == pytest.approx(23.976 / 25.0)


def test_detects_ntsc_to_pal():
    target, name = da.detect_framerate_conversion(25.0 / 23.976)
    assert target is not None
    assert "PAL" in name and "NTSC" in name
    assert target == pytest.approx(25.0 / 23.976)


def test_rejects_unity_ratio():
    """No drift at all — no resample wanted."""
    target, name = da.detect_framerate_conversion(1.000)
    assert target is None
    assert name is None


def test_rejects_commercial_seam_surplus():
    """A broadcast AD with 4 minutes of commercial-break content baked in,
    against a 90-min commercial-free video, has ratio ≈ 1.044. That's
    close to but DISTINCT from the PAL/NTSC target of 1.04271. The
    detector must not misclassify it as a framerate conversion (the
    ratio>1 tolerance is intentionally tight, 0.05%, to avoid this)."""
    target, name = da.detect_framerate_conversion(1.044)
    assert target is None, \
        "commercial-seam surplus must not be misread as a framerate conversion"


def test_accepts_at_ratio_gt_1_tolerance_edge():
    """For ratio > 1 the tolerance is tight (0.05%)."""
    target, name = da.detect_framerate_conversion((25.0 / 23.976) + 0.0004)
    assert target is not None


def test_rejects_ratio_gt_1_just_outside_tight_tolerance():
    target, name = da.detect_framerate_conversion((25.0 / 23.976) + 0.001)
    assert target is None


def test_accepts_at_ratio_lt_1_loose_tolerance():
    """For ratio < 1 the tolerance is loose (0.5%) because commercial-seam
    surplus can't make AD shorter than video — only intro/outro/source-edit
    drift can. Real-world example: Buffy AD source includes a 1-2 second
    intro the streaming video cuts, so ratio drifts a few thousandths of a
    point below the ideal 0.95904."""
    # 0.957 — what a Buffy-class AD might look like; outside the old 0.0005
    # tolerance, inside the new 0.005 tolerance.
    target, name = da.detect_framerate_conversion(0.957)
    assert target is not None
    assert "PAL" in name and "NTSC" in name


def test_rejects_ratio_lt_1_just_outside_loose_tolerance():
    """A ratio of 0.95 — clearly not PAL/NTSC drift — must still reject."""
    target, name = da.detect_framerate_conversion(0.95)
    assert target is None


def test_returns_exact_mathematical_target():
    """Even for a noisy input, the returned ratio is the exact PAL/NTSC
    target (so the resample factor lands on a clean rational)."""
    target, name = da.detect_framerate_conversion(1.0427 + 0.0001)
    assert target == pytest.approx(25.0 / 23.976, abs=1e-12)
