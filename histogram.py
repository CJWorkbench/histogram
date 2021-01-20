import itertools
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from cjwmodule.i18n import trans


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

    # For increment < 1, return -1/increment to prevent floating point accuracy
    # problems.
    # ref: https://github.com/d3/d3-array#tickIncrement
    if power >= 0:
        return inc * (10 ** power)
    else:
        return -(10 ** -power) / inc


def nice_range(values: np.ndarray, n_bins: int) -> Tuple[float, float]:
    """
    Calculate a (start, stop, n_bins) that will make for round-ish ticks.

    Modeled after D3's
    https://github.com/d3/d3-scale/blob/396d1c95fef85241bc3b4b75d747174251ae8e89/src/linear.js
    """
    start = values.min()
    stop = values.max()

    step = _calc_tick_increment(start, stop, n_bins)

    # If every value is on a tick, then this is probably quantized data.
    # Ensure the scale doesn't put the two largest values in the same bucket
    # So [1,2,3,4,5] should give five buckets, not 4
    additional_step = 0
    if step > 0:
        if not np.any(np.mod(values, step)):
            additional_step = 1
    else:
        # equivalent to not np.any(np.mod(values, 1/step))
        # Note, this is the path that values=[1,2,3,4,5,6], n_bins=6 will hit
        # (step=-1 then as (6-1)/6 < 1)
        if not np.any(np.mod(values * step, 1)):
            additional_step = 1

    # negative step signals that it's actually the inverse step (see
    # _calc_tick_increment)
    if step > 0:
        start = math.floor(start / step) * step
        stop = math.ceil(stop / step) * step
        step = _calc_tick_increment(start, stop, n_bins)
    else:
        start = math.ceil(start * step) / step
        stop = math.floor(stop * step) / step
        step = _calc_tick_increment(start, stop, n_bins)

    # d3 algo calls _calc_tick_increment twice, in case first shifts range
    # sufficiently that a better step size is available. We follow their
    # logic and also maybe add an extra step for quantized values
    if step > 0:
        start = math.floor(start / step) * step
        stop = (math.ceil(stop / step) + additional_step) * step
        n_bins = round((stop - start) / step)
    else:
        start = math.ceil(start * step) / step
        stop = (math.floor(stop * step) - additional_step) / step
        n_bins = round(-step * (stop - start))

    return start, stop, n_bins


def safe_values(series: pd.Series) -> np.ndarray:
    """Cast series to ndarray[numeric]; remove NaN, Inf, -Inf."""
    return series[~series.isin(set([np.inf, -np.inf, np.nan]))].to_numpy()


def histogram(values: np.ndarray, n_bins: int) -> Tuple[List[float], List[float]]:
    """
    Computes (counts, ticks).

    Always:

    * ``len(counts) == n_bins``
    * ``len(ticks) == n_bins + 1``

    The min and max ticks are chosen to make for "nice" ticks: that is, ticks
    that are round numbers when possible.

    See
    https://docs.scipy.org/doc/np-1.14.5/reference/generated/np.histogram.html
    for a longer description of return values.
    """

    low, high, n_bins = nice_range(values, n_bins)

    counts, buckets = np.histogram(values, bins=n_bins, range=(low, high))
    return (counts.tolist(), buckets.tolist())


# Displays message in chart output and module error
def render_message(table, message):
    return (
        table,
        message,
        {
            "title": {
                "text": "Please correct the error in this step's data or parameters",  # TODO_i18n
                "offset": 15,
                "color": "#383838",
                "font": "Nunito Sans, Helvetica, sans-serif",
                "fontSize": 15,
                "fontWeight": "normal",
                "anchor": "middle",
            },
            "mark": "point",
            "config": {"style": {"cell": {"stroke": "transparent"}}},
        },
    )


def render(table, params):
    column = params["column"]
    if not column:
        return render_message(
            table, trans("errors.noColumnSelected", "Please choose a number column")
        )

    raw_series = table[column]

    n_bins = max(2, min(500, int(params["n_buckets"])))

    table_values = safe_values(raw_series)
    if not len(table_values) or np.min(table_values) == np.max(table_values):
        return render_message(
            table,
            trans(
                "errors.notEnoughValues",
                "Please choose a number column with at least two distinct values",
            ),
        )

    counts, ticks = histogram(table_values, n_bins)

    bins = [
        {"min": bounds[0], "max": bounds[1], "n": n}
        for n, bounds in zip(counts, _pairwise(ticks))
    ]

    if "title" in params and params["title"] != "":
        title = params["title"]
    else:
        title = f"Histogram of {column}"

    json_dict = {
        "$schema": "https://vega.github.io/schema/vega-lite/v4.0.json",
        "title": {
            "text": title,
            "offset": 15,
            "color": "#383838",
            "font": "Nunito Sans, Helvetica, sans-serif",
            "fontSize": 20,
            "fontWeight": "normal",
        },
        "data": {"values": bins},
        "mark": "bar",
        "encoding": {
            "x": {
                "field": "min",
                "bin": "binned",
                "type": "quantitative",
                "scale": {"zero": False},
                "axis": {
                    "title": column,
                    "grid": False,
                    "tickCount": n_bins + 1,
                    "values": ticks,
                    "tickSize": 3,
                    "titlePadding": 20,
                    # 'titleFontSize': 15,
                    # 'titleFontWeight': 100, -- do not work?
                },
            },
            "x2": {"field": "max", "type": "quantitative"},
            "y": {
                "field": "n",
                "type": "quantitative",
                "axis": {
                    "title": "Frequency",
                    "domain": False,
                    "titlePadding": 20,
                },
            },
            "color": {"value": "#FBAA6D"},
        },
    }

    return (table, "", json_dict)


def _migrate_params_v0_to_v1(params):
    """
    v0: {column, n_buckets}

    v1: {column, n_buckets, title}
    """
    return {**params, "title": ""}


def migrate_params(params):
    if "title" not in params:
        params = _migrate_params_v0_to_v1(params)
    return params
