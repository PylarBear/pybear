# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.ColumnDeduplicateTransformer._type_aliases import DataType

import numpy as np
import pandas as pd

from sklearn.utils.validation import (
    FLOAT_DTYPES,
    _check_sample_weight,
    check_is_fitted,
    check_random_state,
)



def _X_handling(_X: DataType, _n_features_in, _columns) -> None:

    # pizza take this out and put in sklearn validation
    if not hasattr(_X, 'shape'):
        raise TypeError(f"'X' must have shape attribute")
    if not len(_X.shape) == 2:
        raise ValueError(f"'X' must be 2 dimensional")


    # PIZZA! _validate data cannot be done here! this must be under self
    # because it exposes n_features_in_ and feature_names_in_!
    first_call = not hasattr(self, "n_samples_seen_")
    X = self._validate_data(
        X,
        accept_sparse=("csr", "csc"),
        dtype=FLOAT_DTYPES,
        force_all_finite="allow-nan",
        reset=first_call,
    )


    # PIZZA PLACEHOLDER FOR SPARSE CSC, CSR, ....


    # pizza need to put validation of _X against previously seen X's

    if _n_features_in is not None:
        _ = _X.shape[1]
        __ = _n_features_in
        if _ != __:
            raise ValueError(
                f"number of features in passed data ({_}) does not match "
                f"the number of features first seen ({__})."
            )

    # 4 things could happen
    # 1) first passed array, then always passed array
    # 2) first passed array, then sees dataframe
    # 3) first passed dataframe, then sees dataframe
    # 4) first passed dataframe, then sees array

    # if first passed array








