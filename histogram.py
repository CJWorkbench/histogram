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

    # For increment < 1, return -1/increment to prevent floating point accuracy problems
    # ref: https://github.com/d3/d3-array#tickIncrement
    if power >= 0:
        return inc * (10 ** power)
    else:
        return -(10 ** -power) / inc


def nice_range(start: float, stop: float, n_ticks: int) -> Tuple[float, float]:
    """
    Calculate a (start, stop, n_bins) that will make for round-ish ticks.

    Modeled after D3's
    https://github.com/d3/d3-scale/blob/396d1c95fef85241bc3b4b75d747174251ae8e89/src/linear.js

    But we add an extra bin on the end to ensure that the max value has a place to go,
    and that histograms of integers look good. Also we can handle max==min
    """

    # Degenerate case, one value. Bracket with a bin of width 1 (we may get more, after adjustment)
    if start==stop:
        start = math.floor(start)
        stop = math.ceil(stop)
        if (start==stop):
            stop = start+1

    step = _calc_tick_increment(start, stop, n_ticks)

    # negative step signals that it's actually the inverse step (see above)
    if step > 0:
        start = math.floor(start / step) * step
        stop = math.ceil(stop / step) * step
        step = _calc_tick_increment(start, stop, n_ticks)
    else:
        start = math.ceil(start * step) / step
        stop = math.floor(stop * step) / step
        step = _calc_tick_increment(start, stop, n_ticks)

    # Add one more bucket (as compared to d3 algo) by adjusting stop
    if step > 0:
        start = math.floor(start / step) * step
        stop = (math.ceil(stop / step) + 1) * step
        n_ticks = round((stop - start) / step)
    else:
        start = math.ceil(start * step) / step
        stop = (math.floor(stop * step) - 1) / step
        n_ticks = round(-step * (stop - start))

    return start, stop, n_ticks


def safe_values(series: pandas.Series) -> numpy.ndarray:
    """Cast series to ndarray of float64. Remove NaN and info rows"""

    # to_numeric: errors become NaN
    number_series = pandas.to_numeric(series, errors='coerce')
    number_series.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True)
    number_series.dropna(inplace=True)

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
        "title": {
            "text": 'Choose a numeric column',
            'offset': 15,
            'color': '#383838',
            'font': 'Nunito Sans, Helvetica, sans-serif',
            'fontSize': 25,
            'fontWeight': 'normal',
            'anchor': 'middle',
            },
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
    n_bins = max(2, min(500, int(params['n_buckets'])))

    table_values = safe_values(raw_series)
    if len(table_values) == 0:
        return render_message(
            table,
            'Please choose a column with at least one numeric value'
        )

    counts, ticks = histogram(table_values, n_bins)

    bins = [{'min': bounds[0], 'max': bounds[1], 'n': n}
            for n, bounds in zip(counts, _pairwise(ticks))]

    json_dict = {
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.0.json',
        "title": {
            "text": f'Number of {column} records per bin',
            'offset': 15,
            'color': '#383838',
            'font': 'Nunito Sans, Helvetica, sans-serif',
            'fontSize': 20,
            'fontWeight': 'normal',
            },

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
                    'tickSize': 3,
                    'titlePadding': 20,
                    # 'titleFontSize': 15,
                    # 'titleFontWeight': 100, -- do not work?
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
                    'titlePadding': 20,
                },
            },
            'y': {
                'value': 0,
            },
            'color': {'value': '#FBAA6D'},
        },
    }

    return (table, error, json_dict)
