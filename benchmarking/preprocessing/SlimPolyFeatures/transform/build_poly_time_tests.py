# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# this is a time benchmark for _build_poly
# _build_poly takes the combo tuples in _active_columns, pulls the columns
# from X, and multiplies them together.
# joblib parallelizes the multiplication of the columns into a single
# column, but is it better than just a regular for loop because of all
# the serialization that happens to send all these column combos around
# to the workers for a fairly simple operation that may be faster locally.


# LINUX TIME TRIALS
# _rows = 100_000
# _columns = 10
# _min_degree = 2
# _max_degree = 4
# number_of_trials = 7
# rest_time = 3

# n_jobs = -1
# joblib_version       time = 25.013 +/- 2.035 sec; mem = 1,189.600 +/- 66.605 MB
# for_loop_version     time = 14.527 +/- 0.855 sec; mem = 1,120.600 +/- 12.516 MB

# n_jobs = 2
# joblib_version       time = 22.086 +/- 1.412 sec; mem = 1,167.000 +/- 39.156 MB
# for_loop_version     time = 14.614 +/- 1.074 sec; mem = 1,124.600 +/- 8.593 MB

import itertools
import numpy as np
import scipy.sparse as ss
from joblib import Parallel, delayed, wrap_non_picklable_objects

from pybear.preprocessing._SlimPolyFeatures._partial_fit._columns_getter \
    import _columns_getter

from pybear.utilities import time_memory_benchmark as tmb



_rows = 100_000
_columns = 10
_min_degree = 2
_max_degree = 4

_active_combos = []
for _degree in range(_min_degree, _max_degree+1):
    __ = list(itertools.combinations_with_replacement(range(_columns), _degree))
    _active_combos += __

X = np.random.uniform(0, 1, (_rows, _columns))


def joblib_version(X):

    @wrap_non_picklable_objects
    def _poly_stacker(_columns):
        return ss.csc_array(_columns.prod(1).reshape((-1, 1)))

    joblib_kwargs = {
        'prefer': 'processes', 'return_as': 'list', 'n_jobs': -1
    }
    POLY = Parallel(**joblib_kwargs)(delayed(_poly_stacker)(
        _columns_getter(X, combo)) for combo in _active_combos)

    POLY = ss.hstack(POLY).astype(np.float64)

    assert POLY.shape == (X.shape[0], len(_active_combos))

    return POLY


def for_loop_version(X):

    POLY = []
    for combo in _active_combos:
        _poly_feature = _columns_getter(X, combo).prod(1).reshape((-1, 1))
        POLY.append(ss.csc_array(_poly_feature))

    POLY = ss.hstack(POLY).astype(np.float64)

    assert POLY.shape == (X.shape[0], len(_active_combos))

    return POLY


out = tmb(
    ('joblib_version', joblib_version, [X], {}),
    ('for_loop_version', for_loop_version, [X], {}),
    number_of_trials=7,
    rest_time=3,
    verbose=True
)


print(out)







