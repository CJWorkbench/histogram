#!/usr/bin/env python3

import math
import unittest
import numpy
import pandas
from histogram import nice_range, safe_values, histogram


class NiceRangeTest(unittest.TestCase):
    def test_big_numbers(self):
        self.assertEqual(nice_range(240, 12314, 13), (0, 13000, 13))

    def test_across_zero(self):
        self.assertEqual(nice_range(-8, 22, 4), (-10, 30, 4))

    def test_small_numbers(self):
        self.assertEqual(nice_range(0.1, 0.41, 16), (0.1, 0.42, 16))

    def test_small_numbers_across_zero(self):
        self.assertEqual(nice_range(-0.04, 0.8, 9), (-0.1, 0.8, 9))

    def test_sugests_better_n_ticks(self):
        self.assertEqual(nice_range(-0.04, 0.8, 10), (-0.1, 0.8, 9))


class SafeValuesTest(unittest.TestCase):
    def test_convert_float(self):
        result = safe_values(pandas.Series([1.0, 2.0, 3.0]), 1.0)
        expect = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float64)
        self.assertTrue(numpy.array_equal(result, expect))

    def test_convert_inf(self):
        result = safe_values(pandas.Series([1.0, math.inf, -math.inf]), 2.0)
        expect = numpy.array([1.0, 2.0, 2.0], dtype=numpy.float64)
        self.assertTrue(numpy.array_equal(result, expect))

    def test_convert_int(self):
        result = safe_values(pandas.Series([1, 2, 3]), 1.0)
        expect = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float64)
        self.assertTrue(numpy.array_equal(result, expect))

    def test_convert_str(self):
        result = safe_values(pandas.Series(['1', '2', '3']), 1.0)
        expect = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float64)
        self.assertTrue(numpy.array_equal(result, expect))

    def test_convert_obj(self):
        result = safe_values(pandas.Series([1.0, 2.0, None]), 3.0)
        expect = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float64)
        self.assertTrue(numpy.array_equal(result, expect))

    def test_convert_error(self):
        result = safe_values(pandas.Series([1.0, 2.0, 'notanumber']), 3.0)
        expect = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float64)
        self.assertTrue(numpy.array_equal(result, expect))


class HistogramTest(unittest.TestCase):
    def test_basic(self):
        result = histogram(numpy.array([0.0, 1.8, 1.1, 2.0],
                                       dtype=numpy.float64),
                           2)
        self.assertEqual(result, ([1, 3], [0.0, 1.0, 2.0]))

    def test_uses_nice_range(self):
        result = histogram(numpy.array([0.1, 1.8, 1.1, 1.9],
                                       dtype=numpy.float64),
                           2)
        self.assertEqual(result, ([1, 3], [0.0, 1.0, 2.0]))

    def test_uses_nice_n_bins(self):
        # Use 10 bins instead of 12, when it makes the chart nicer
        result = histogram(numpy.array([0.01, 0.95], dtype=numpy.float64), 12)
        self.assertEqual(result[0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(
            # ignore rounding errors: the charting library will round
            [round(x * 10) for x in result[1]],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
