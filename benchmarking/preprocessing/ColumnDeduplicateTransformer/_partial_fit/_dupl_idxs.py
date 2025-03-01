# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.preprocessing._ColumnDeduplicateTransformer._partial_fit.\
    _dupl_idxs import _dupl_idxs

import numpy as np





og_X = np.random.randint(0,10,(100,20))

duplicates = [[1,8], [3,9,12]]

X = og_X.copy()
for _set in duplicates:
    for _idx in _set[1:]:
        X[:, _idx] = X[:, _set[0]]


duplicates_ = _dupl_idxs(
    X, None, _rtol=1e-6, _atol=1e6, _equal_nan=False, _n_jobs=None
)

# less_duplicates = [[1,8], [3,12]]
# X_less = og_X.copy()
# for _set in less_duplicates:
#     for _idx in _set[1:]:
#         X_less[:, _idx] = X_less[:, _set[0]]
#
# out = _dupl_idxs(
#   X_less, duplicates_, _atol=1e-6, _rtol=1e-6 _equal_nan=False, _n_jobs=None
# )
# print(out)    # should == less_duplicates


# no duplicates
new_X = og_X.copy()
out = _dupl_idxs(
    new_X, duplicates_, _atol=1e-6, _rtol=1e-6, _equal_nan=False, _n_jobs=None
)
print(out)    # should == []

