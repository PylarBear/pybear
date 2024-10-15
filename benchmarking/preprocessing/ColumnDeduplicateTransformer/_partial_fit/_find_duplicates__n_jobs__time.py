# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _find_duplicates import _find_duplicates

from pybear.utilities import time_memory_benchmark as tmb

import numpy as np
import joblib


# LINUX 24_10_10
# threads, n_jobs=-1         time = 14.032 +/- 0.137 sec; mem = 17.400 +/- 5.083 MB
# threads, n_jobs=1          time = 17.760 +/- 0.039 sec; mem = -14.600 +/- 1.200 MB
# threads, n_jobs=2          time = 14.189 +/- 0.400 sec; mem = 12.200 +/- 6.112 MB
# threads, n_jobs=3          time = 13.900 +/- 0.266 sec; mem = 0.600 +/- 8.570 MB
# threads, n_jobs=4          time = 14.033 +/- 0.215 sec; mem = -2.400 +/- 1.960 MB
# threads, n_jobs=None       time = 18.173 +/- 0.474 sec; mem = -15.000 +/- 1.095 MB
# processes, n_jobs=-1       time = 14.150 +/- 0.247 sec; mem = 0.000 +/- 0.000 MB
# processes, n_jobs=1        time = 18.135 +/- 0.461 sec; mem = 0.400 +/- 0.800 MB
# processes, n_jobs=2        time = 14.425 +/- 0.064 sec; mem = 0.000 +/- 0.000 MB
# processes, n_jobs=3        time = 14.362 +/- 0.068 sec; mem = 0.000 +/- 0.000 MB
# processes, n_jobs=4        time = 14.236 +/- 0.204 sec; mem = 0.000 +/- 0.000 MB
# processes, n_jobs=None     time = 17.902 +/- 0.230 sec; mem = 0.000 +/- 0.000 MB




dupl = [
    [0,4,11],
    [2,15,18],
    [5,12,14,16]
]


def _find_dupl_processes(
    _X,
    _n_jobs:int=None
) -> list[list[int]]:
    return _find_duplicates(
        _X,
        _rtol=1e-6,
        _atol=1e-6,
        _equal_nan=True,
        _n_jobs=_n_jobs
    )

def _find_dupl_threads(
    _X,
    _n_jobs:int=None
) -> list[list[int]]:
    with joblib.parallel_config(backend='threading', n_jobs=_n_jobs):
        return _find_duplicates(
            _X,
            _rtol=1e-6,
            _atol=1e-6,
            _equal_nan=True,
            _n_jobs=_n_jobs
        )



_rows = 2_000_000
_cols = 20
X = np.random.randint(0, 10, (_rows, _cols))
for _set in dupl:
    for _idx in _set[1:]:
        X[:, _idx] = X[:, _set[0]]


out = tmb(
    ('threads, n_jobs=-1', _find_dupl_threads, [X], {'_n_jobs': -1}),
    ('threads, n_jobs=1', _find_dupl_threads, [X], {'_n_jobs': 1}),
    ('threads, n_jobs=2', _find_dupl_threads, [X], {'_n_jobs': 2}),
    ('threads, n_jobs=3', _find_dupl_threads, [X], {'_n_jobs': 3}),
    ('threads, n_jobs=4', _find_dupl_threads, [X], {'_n_jobs': 4}),
    ('threads, n_jobs=None', _find_dupl_threads, [X], {'_n_jobs': None}),
    ('processes, n_jobs=-1', _find_dupl_processes, [X], {'_n_jobs': -1}),
    ('processes, n_jobs=1', _find_dupl_processes, [X], {'_n_jobs': 1}),
    ('processes, n_jobs=2', _find_dupl_processes, [X], {'_n_jobs': 2}),
    ('processes, n_jobs=3', _find_dupl_processes, [X], {'_n_jobs': 3}),
    ('processes, n_jobs=4', _find_dupl_processes, [X], {'_n_jobs': 4}),
    ('processes, n_jobs=None', _find_dupl_processes, [X], {'_n_jobs': None}),
    rest_time=3,
    number_of_trials=7,
    verbose=True
)





















