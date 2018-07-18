import itertools
import math
import numpy
import pandas
from typing import List, Tuple


def _pairwise(iterable):
    """
    s -> (s0, s1), (s1, s2), (s2, s3), ....

    From https://docs.python.org/3/library/itertools.html
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def _calc_tick_increment(start: float, stop: float, n_ticks: int) -> float:
    # https://github.com/d3/d3-array/blob/a1ffecb56d668156c8037eee78f6149bbb664061/src/ticks.js
    step = (stop - start) / max(0, n_ticks)
    power = math.floor(math.log10(step))
    error = step / (10 ** power)

    if error >= math.sqrt(50):
        inc = 10
    elif error >= math.sqrt(10):
        inc = 5
    elif error >= math.sqrt(2):
        inc = 2
    else:
        inc = 1

    if power >= 0:
        return inc * (10 ** power)
    else:
        return -(10 ** -power) / inc


def nice_range(start: float, stop: float, n_ticks: int) -> Tuple[float, float]:
    """
    Calculate a (start, stop, n_bins) that will make for round-ish ticks.

    Modeled after D3's
    https://github.com/d3/d3-scale/blob/396d1c95fef85241bc3b4b75d747174251ae8e89/src/linear.js
    """
    step = _calc_tick_increment(start, stop, n_ticks)

    if step > 0:
        start = math.floor(start / step) * step
        stop = math.ceil(stop / step) * step
        step = _calc_tick_increment(start, stop, n_ticks)
    else:
        start = math.ceil(start * step) / step
        stop = math.floor(stop * step) / step
        step = _calc_tick_increment(start, stop, n_ticks)

    if step > 0:
        start = math.floor(start / step) * step
        stop = math.ceil(stop / step) * step
        n_ticks = round((stop - start) / step)
    else:
        start = math.ceil(start * step) / step
        stop = math.floor(stop * step) / step
        n_ticks = round(-step * (stop - start))

    return start, stop, n_ticks


def safe_values(series: pandas.Series, replace: float) -> numpy.ndarray:
    """Cast series to ndarray of float64, replacing errors with `replace`."""
    # to_numeric: errors become NaN. (numpy doesn't do 'coerce')
    number_series = pandas.to_numeric(series, errors='coerce')

    # replace NaN (both from original data and from last step) with given value
    # TODO consider making a separate bin for NaN. Or two separate bins: one
    # for errors and one for NaN.
    #
    # nNAs-from-to_numeric-coerce = nNAs-after-to_numeric - nNAs-before
    number_series.fillna(replace, inplace=True)
    number_series.replace([numpy.inf, -numpy.inf], replace, inplace=True)

    ret = number_series.values
    return ret.astype(numpy.float64)


def histogram(values: numpy.ndarray,
              n_bins: int) -> Tuple[List[float], List[float]]:
    """
    Computes (counts, ticks).

    Always:

    * ``len(counts) == n_bins``
    * ``len(ticks) == n_bins + 1``

    The min and max ticks are chosen to make for "nice" ticks: that is, ticks
    that are round numbers when possible.

    See
    https://docs.scipy.org/doc/numpy-1.14.5/reference/generated/numpy.histogram.html
    for a longer description of return values.
    """
    low = values.min()
    high = values.max()

    low, high, n_bins = nice_range(low, high, n_bins)

    counts, buckets = numpy.histogram(values, bins=n_bins, range=(low, high))
    return (counts.tolist(), buckets.tolist())


def render_message(table, message):
    return (table, '', {
        'title': message,
        'mark': 'point',
        'config': {
            'style': {
                'cell': {
                    'stroke': 'transparent',
                },
            },
        },
    })


def render(table, params):
    error = None
    json_dict = None

    column = params['column']
    if not column:
        return render_message(table, 'Please choose a numeric column')

    raw_series = table[column]
    n_bins = min(2, max(500, int(params['n_buckets'])))
    replace = float(params['replace_missing_number'])

    table_values = safe_values(raw_series, replace)
    if numpy.min(table_values) == numpy.max(table_values):
        return render_message(
            table,
            'Please choose a numeric column with at least two distinct values'
        )

    counts, ticks = histogram(table_values, n_bins)

    bins = [{'min': bounds[0], 'max': bounds[1], 'n': n}
            for n, bounds in zip(counts, _pairwise(ticks))]

    json_dict = {
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.0.json',
        'title': f'Number of {column} records per bin',
        'data': {'values': bins},
        # TODO use Vega-lite "prebinned" feature when it's available.
        # In the meantime, this is modeled after
        # https://github.com/vega/vega-lite/issues/2912#issuecomment-388987973
        'mark': 'rect',
        'encoding': {
            'x': {
                'field': 'min',
                'type': 'quantitative',
                'scale': {
                    'zero': False,
                },
                'axis': {
                    'title': f'{column}',
                    'grid': False,
                    'tickCount': n_bins + 1,
                    'values': ticks,
                },
            },
            'x2': {
                'field': 'max',
                'type': 'quantitative',
            },
            'y2': {
                'field': 'n',
                'type': 'quantitative',
                'axis': {
                    'title': 'Number of records',
                    'domain': False,
                },
            },
            'y': {
                'value': 0,
            },
        },
    }

    return (table, error, json_dict)
