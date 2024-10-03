# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _find_duplicates import _find_duplicates


import numpy as np


dupl = [
    [0,4],
    [2,5,6]
]

X = np.random.randint(0, 10, (20, 10))
for _set in dupl:
    for _idx in _set[1:]:
        X[:, _idx] = X[:, _set[0]]


out = _find_duplicates(X)

print(out)



















