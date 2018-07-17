import itertools
import numpy
import pandas


def _pairwise(iterable):
    """
    s -> (s0, s1), (s1, s2), (s2, s3), ....

    From https://docs.python.org/3/library/itertools.html
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def render(table, params):
    error = None
    json_dict = None

    raw_series = table[params['column']]
    number_series = pandas.to_numeric(raw_series, errors='coerce')
    number_series.fillna(params['replace_missing_number'])
    n_buckets = params['n_buckets']

    counts, buckets = numpy.histogram(number_series, bins=n_buckets)

    values = [{'bin': f'[{bounds[0]},{bounds[1]})', 'n': n}
              for n, bounds in zip(counts.tolist(), _pairwise(buckets))]

    json_dict = {
        'data': {'values': values},
        'mark': 'bar',
        'encoding': {
            'bin': {
                'field': 'bin',
                'type': 'nominal',
                'axis': {'title': 'Bin'},
            },
            'n': {
                'field': 'count',
                'type': 'quantitative',
                'axis': {'title': 'Count'},
            },
        },
    }

    return (table, error, json_dict)
