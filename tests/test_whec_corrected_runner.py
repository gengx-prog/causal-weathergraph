"""Failure-path checks for the corrected runner's real-calendar contract."""
import numpy as np
import pytest

from revision.make_whec_index_20261009 import placebo_humidity_indices
from revision.run_whec_test_20261009 import validate_placebo_source


def calendar():
    return np.arange(np.datetime64("1979-01-01T00", "h"),
                     np.datetime64("2026-01-01T00", "h"),
                     np.timedelta64(6, "h"))


def test_corrected_calendar_accepts_discovery_only_warmup():
    ts = calendar()
    source = placebo_humidity_indices(len(ts), ts < np.datetime64("2019-01-01"))
    discovery = validate_placebo_source(ts, source)
    assert discovery.sum() == 58440
    assert ts[source[0]] == np.datetime64("2018-01-01T00")


def test_legacy_full_record_wrap_is_rejected():
    ts = calendar()
    with pytest.raises(ValueError, match="Held-out humidity"):
        validate_placebo_source(ts, (np.arange(len(ts)) - 1460) % len(ts))


def test_discovery_only_but_wrong_shift_is_rejected():
    ts = calendar()
    source = placebo_humidity_indices(len(ts), ts < np.datetime64("2019-01-01"))
    source[0] -= 1
    with pytest.raises(ValueError, match="365-day schedule"):
        validate_placebo_source(ts, source)


@pytest.mark.parametrize("kind", ["float", "negative", "outside", "short"])
def test_malformed_source_index_is_rejected(kind):
    ts = calendar()
    source = placebo_humidity_indices(len(ts), ts < np.datetime64("2019-01-01"))
    if kind == "float":
        source = source.astype(float)
    elif kind == "negative":
        source[0] = -1
    elif kind == "outside":
        source[0] = len(ts)
    else:
        source = source[:-1]
    with pytest.raises(ValueError, match="Invalid placebo"):
        validate_placebo_source(ts, source)


def test_missing_calendar_step_is_rejected():
    ts = calendar()
    source = placebo_humidity_indices(len(ts), ts < np.datetime64("2019-01-01"))
    ts[2000] += np.timedelta64(6, "h")
    with pytest.raises(ValueError, match="calendar"):
        validate_placebo_source(ts, source)
