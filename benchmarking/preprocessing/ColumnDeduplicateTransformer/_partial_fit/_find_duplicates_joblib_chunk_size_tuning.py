# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# this file varies number of columns pulled from X and passed to joblib

# _find_duplicates signature needs to be temporarily modified to take
# a 'n_cols' argument.


from pybear.preprocessing._ColumnDeduplicateTransformer._partial_fit. \
    _find_duplicates import _find_duplicates

from pybear.utilities import time_memory_benchmark as tmb

import numpy as np
import joblib


# LINUX 25_05_29





dupl = [
    [0,4,11],
    [2,15,18],
    [5,12,14,16]
]


def _find_dupl__processes(
    _X,
    _n_cols:int=None
) -> list[list[int]]:
    return _find_duplicates(
            _X,
            _rtol=1e-6,
            _atol=1e-6,
            _equal_nan=True,
            _n_jobs=-1,
            _n_cols=_n_cols
        )


def _find_dupl__threads(
    _X,
    _n_cols:int=None
) -> list[list[int]]:
    with joblib.parallel_config(backend='threading'):
        return _find_duplicates(
            _X,
            _rtol=1e-6,
            _atol=1e-6,
            _equal_nan=True,
            _n_jobs=-1,
            _n_cols=_n_cols
        )



X = np.random.randint(0, 10, (10_000, 10_000)).astype(np.uint8)
for _set in dupl:
    for _idx in _set[1:]:
        X[:, _idx] = X[:, _set[0]]


out = tmb(
    # ('threads, _n_cols=10', _find_dupl__threads, [X], {'_n_cols': 100}),
    # ('threads, _n_cols=20', _find_dupl__threads, [X], {'_n_cols': 200}),
    # ('threads, _n_cols=30', _find_dupl__threads, [X], {'_n_cols': 300}),
    # ('threads, _n_cols=40', _find_dupl__threads, [X], {'_n_cols': 400}),
    # ('threads, _n_cols=50', _find_dupl__threads, [X], {'_n_cols': 500}),
    ('processes, _n_cols=1000', _find_dupl__processes, [X], {'_n_cols': 1000}),
    ('processes, _n_cols=2000', _find_dupl__processes, [X], {'_n_cols': 2000}),
    ('processes, _n_cols=3000', _find_dupl__processes, [X], {'_n_cols': 3000}),
    # ('processes, _n_cols=40', _find_dupl__processes, [X], {'_n_cols': 400}),
    # ('processes, _n_cols=50', _find_dupl__processes, [X], {'_n_cols': 500}),
    rest_time=1,
    number_of_trials=3,
    verbose=True
)




