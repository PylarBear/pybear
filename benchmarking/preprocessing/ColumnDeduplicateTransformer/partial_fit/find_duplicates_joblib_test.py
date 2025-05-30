# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np

from pybear.preprocessing._ColumnDeduplicateTransformer._partial_fit. \
    _find_duplicates import _find_duplicates



_X = np.random.randint(0, 10, (1_000, 50))



out = _find_duplicates(
    _X,
    _rtol=1e-5,
    _atol=1e-8,
    _equal_nan=True,
    _n_jobs=-1
)










