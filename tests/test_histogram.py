#!/usr/bin/env python3

import unittest
import numpy
import pandas
from pandas.testing import assert_frame_equal
from histogram import nice_range, safe_values, histogram, migrate_params, render
from cjwmodule.testing.i18n import i18n_message


class MigrateParamsTest(unittest.TestCase):
    def test_v0(self):
        self.assertEqual(
            migrate_params({"column": "A", "n_buckets": 20}),
            {"column": "A", "n_buckets": 20, "title": ""},
        )

    def test_v1(self):
        self.assertEqual(
            migrate_params({"column": "A", "n_buckets": 20, "title": "Good"}),
            {"column": "A", "n_buckets": 20, "title": "Good"},
        )


class NiceRangeTest(unittest.TestCase):
    def test_big_numbers(self):
        self.assertEqual(
            nice_range(numpy.array([240, 333.3, 12314]), 13), (0, 13000, 13)
        )

    def test_across_zero(self):
        self.assertEqual(nice_range(numpy.array([-8, 22]), 4), (-10, 30, 4))

    def test_small_numbers(self):
        self.assertEqual(nice_range(numpy.array([0.1, 0.41]), 16), (0.1, 0.42, 16))

    def test_small_numbers_across_zero(self):
        self.assertEqual(nice_range(numpy.array([-0.04, 0.8]), 9), (-0.1, 0.8, 9))

    def test_sugests_better_n_ticks(self):
        self.assertEqual(nice_range(numpy.array([-0.04, 0.8]), 10), (-0.1, 0.8, 9))

    def test_integer_ticks(self):
        # Tests adding of additional tick in "die roll" example
        self.assertEqual(nice_range(numpy.array([1, 2, 3, 4, 5, 6]), 5), (1, 7, 6))
        self.assertEqual(nice_range(numpy.array([1, 2, 3, 4, 5, 6]), 6), (1, 7, 6))
        self.assertEqual(nice_range(numpy.array([1, 2, 3, 4, 5, 6]), 7), (1, 7, 6))

    def test_integer_ticks_half_buckets(self):
        # If we end up with ticks 0.5 apart, all buckets should still be evenly
        # spaced
        self.assertEqual(nice_range(numpy.array([1, 2, 3, 4, 5, 6]), 10), (1, 6.5, 11))
        self.assertEqual(nice_range(numpy.array([1, 2, 3, 4, 5, 6]), 11), (1, 6.5, 11))
        self.assertEqual(nice_range(numpy.array([1, 2, 3, 4, 5, 6]), 12), (1, 6.5, 11))
        pass


class SafeValuesTest(unittest.TestCase):
    def test_float(self):
        result = safe_values(pandas.Series([1.0, 2.0, 3.0]))
        expect = numpy.array([1.0, 2.0, 3.0])
        self.assertTrue(numpy.array_equal(result, expect))

    def test_remove_inf(self):
        result = safe_values(pandas.Series([1.0, numpy.inf, -numpy.inf]))
        expect = numpy.array([1.0])
        self.assertTrue(numpy.array_equal(result, expect))

    def test_remove_na(self):
        result = safe_values(pandas.Series([1.0, numpy.nan, 2.0]))
        expect = numpy.array([1.0, 2.0])
        self.assertTrue(numpy.array_equal(result, expect))

    def test_int(self):
        result = safe_values(pandas.Series([1, 2, 3]))
        expect = numpy.array([1, 2, 3])
        self.assertTrue(numpy.array_equal(result, expect))


class HistogramTest(unittest.TestCase):
    def test_basic(self):
        result = histogram(numpy.array([0.0, 1.8, 1.1, 2.0], dtype=numpy.float64), 2)
        self.assertEqual(result, ([1, 3], [0.0, 1.0, 2.0]))

    def test_uses_nice_range(self):
        result = histogram(numpy.array([0.1, 1.8, 1.1, 1.9], dtype=numpy.float64), 2)
        self.assertEqual(result, ([1, 3], [0.0, 1.0, 2.0]))

    def test_uses_nice_n_bins(self):
        # Use 10 bins instead of 12, when it makes the chart nicer
        result = histogram(numpy.array([0.01, 0.95], dtype=numpy.float64), 12)
        self.assertEqual(result[0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(
            # ignore rounding errors: the charting library will round
            [round(x * 10) for x in result[1]],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )


class RenderTest(unittest.TestCase):
    def test_happy_path(self):
        table, error, json_dict = render(
            pandas.DataFrame({"A": [1.1, 2.2, 3.3]}),
            {"column": "A", "n_buckets": 5, "title": "My Title"},
        )
        # Output table is same as input
        assert_frame_equal(table, pandas.DataFrame({"A": [1.1, 2.2, 3.3]}))
        self.assertEqual(error, "")
        self.assertEqual(
            json_dict["data"]["values"],
            [
                {"min": 1.0, "max": 1.5, "n": 1},
                {"min": 1.5, "max": 2.0, "n": 0},
                {"min": 2.0, "max": 2.5, "n": 1},
                {"min": 2.5, "max": 3.0, "n": 0},
                {"min": 3.0, "max": 3.5, "n": 1},
            ],
        )
        self.assertEqual(json_dict["title"]["text"], "My Title")

    def test_zero_sized_domain_is_error(self):
        table, error, json_dict = render(
            pandas.DataFrame({"A": [1.1, 1.1, 1.1]}),
            {"column": "A", "n_buckets": 5, "title": "My Title"},
        )
        # Output table is same as input
        assert_frame_equal(table, pandas.DataFrame({"A": [1.1, 1.1, 1.1]}))
        self.assertEqual(error, i18n_message("errors.notEnoughValues"))
        self.assertTrue(json_dict["title"]["text"])

    def test_zero_values_is_error(self):
        table, error, json_dict = render(
            pandas.DataFrame({"A": [numpy.nan]}),
            {"column": "A", "n_buckets": 5, "title": "My Title"},
        )
        # Output table is same as input
        assert_frame_equal(table, pandas.DataFrame({"A": [numpy.nan]}))
        self.assertEqual(error, i18n_message("errors.notEnoughValues"))
        self.assertTrue(json_dict["title"]["text"])

    def test_no_column_is_error(self):
        table, error, json_dict = render(
            pandas.DataFrame({"A": [5.1, 2.1]}),
            {"column": "", "n_buckets": 5, "title": "My Title"},
        )
        # Output table is same as input
        assert_frame_equal(table, pandas.DataFrame({"A": [5.1, 2.1]}))

        self.assertEqual(error, i18n_message("errors.noColumnSelected"))
        self.assertTrue(json_dict["title"]["text"])
