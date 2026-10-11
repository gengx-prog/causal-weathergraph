"""Regression checks for the 2026-10-09 full-record-wrap leakage correction."""
import unittest

import numpy as np

from revision.make_whec_index_20261009 import PLACEBO_SHIFT, placebo_humidity_indices


class PlaceboSourcesTest(unittest.TestCase):
    def test_full_analysis_calendar_has_no_heldout_discovery_source(self):
        times = np.arange(np.datetime64("1979-01-01T00", "h"),
                          np.datetime64("2026-01-01T00", "h"),
                          np.timedelta64(6, "h"))
        training = times < np.datetime64("2019-01-01")
        source = placebo_humidity_indices(len(times), training)
        self.assertEqual(len(times), 68668)
        self.assertEqual(int(training.sum()), 58440)
        self.assertTrue(np.all(training[source[training]]))
        self.assertEqual(times[source[0]], np.datetime64("2018-01-01T00"))
        self.assertEqual(times[source[PLACEBO_SHIFT - 1]], np.datetime64("2018-12-31T18"))
        self.assertTrue(np.all(times[PLACEBO_SHIFT:] - times[source[PLACEBO_SHIFT:]]
                               == np.timedelta64(365, "D")))
        legacy = (np.arange(len(times)) - PLACEBO_SHIFT) % len(times)
        self.assertEqual(int(np.sum(~training[legacy[training]])), 1460)
        self.assertTrue(np.array_equal(source[PLACEBO_SHIFT:], legacy[PLACEBO_SHIFT:]))

    def test_rejects_nonprefix_discovery(self):
        training = np.ones(3000, dtype=bool)
        training[20] = False
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            placebo_humidity_indices(len(training), training)

    def test_rejects_insufficient_warmup(self):
        with self.assertRaisesRegex(ValueError, "Insufficient discovery"):
            placebo_humidity_indices(10, np.ones(10, dtype=bool))


if __name__ == "__main__":
    unittest.main()
