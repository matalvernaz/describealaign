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
    detector must not misclassify it as a framerate conversion."""
    target, name = da.detect_framerate_conversion(1.044)
    assert target is None, \
        "commercial-seam surplus must not be misread as a framerate conversion"


def test_accepts_exactly_at_tolerance_edge():
    """A real-world ratio is rarely exactly 1.04271 — small probe-noise
    is fine within 0.05%."""
    target, name = da.detect_framerate_conversion(1.0 / (23.976 / 25.0) + 0.0004)
    assert target is not None


def test_rejects_just_outside_tolerance():
    """One step beyond tolerance should not match."""
    target, name = da.detect_framerate_conversion((25.0 / 23.976) + 0.001)
    assert target is None


def test_returns_exact_mathematical_target():
    """Even for a noisy input, the returned ratio is the exact PAL/NTSC
    target (so the resample factor lands on a clean rational)."""
    target, name = da.detect_framerate_conversion(1.0427 + 0.0001)
    assert target == pytest.approx(25.0 / 23.976, abs=1e-12)
