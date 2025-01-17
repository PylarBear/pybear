# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import DataType

import numpy as np
import pandas as pd



def _val_X(
    _X: DataType,
    _interaction_only: bool
) -> None:

    """
    Validate X.
    The container format:
        Must be numpy ndarray, pandas dataframe, or any scipy sparse.
    The dimensions of the container:
        Must have at least 1 sample.
        When interaction_only is True, must have at least 2 columns.
        When interaction_only is False, must have at least 1 column.
    All values within must be numeric.


    Parameters
    ----------
    _X:
        {array-like, scipy sparse matrix} of shape (n_samples,
        n_features) - the data to undergo polynomial expansion.
    _interaction_only:
        bool - If True, only interaction features are produced, that is,
        polynomial features that are products of 'degree' distinct input
        features. Terms with power of 2 or higher for any feature are
        excluded.
        Consider 3 features 'a', 'b', and 'c'. If 'interaction_only' is
        True, 'min_degree' is 1, and 'degree' is 2, then only the first
        degree interaction terms ['a', 'b', 'c'] and the second degree
        interaction terms ['ab', 'ac', 'bc'] are returned in the
        polynomial expansion.


    Return
    ------
    -
        None


    """


    assert isinstance(_interaction_only, bool)


    if not isinstance(_X, (np.ndarray, pd.core.frame.DataFrame)) and not \
        hasattr(_X, 'toarray'):
        raise TypeError(
            f"invalid container for X: {type(_X)}. X must be numpy array, "
            f"pandas dataframe, or any scipy sparce matrix / array."
        )


    _base_msg = (
        f"'X' must be a valid 2 dimensional numpy ndarray, pandas dataframe, "
        f"or scipy sparce matrix or array with at least 1 sample. "
    )

    if _interaction_only:
        if _X.shape[1] < 2:
            _addon_msg = (f"\nWhen only generating interaction terms (:param: "
                f"interaction_only is True), 'X' must have at least 2 features.")
            raise ValueError(_base_msg + _addon_msg)
    elif not _interaction_only:
        if _X.shape[1] < 1:
            _addon_msg = (f"\nWhen generating all polynomial terms (:param: "
                f"interaction_only is False), 'X' must have at least 1 feature.")
            raise ValueError(_base_msg + _addon_msg)





