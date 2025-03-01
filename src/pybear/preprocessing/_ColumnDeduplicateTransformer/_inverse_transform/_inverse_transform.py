# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
import numpy.typing as npt

import numpy as np
import pandas as pd
import scipy.sparse as ss



def _inverse_transform(
    X: Union[npt.NDArray, pd.DataFrame, ss.csc_matrix, ss.csc_array],
    _removed_columns: dict[int, int],
    _feature_names_in: Union[npt.NDArray[str], None]
) -> Union[npt.NDArray, pd.DataFrame, ss.csc_matrix, ss.csc_array]:

    """
    Revert deduplicated data back to its original state. CDT  cannot
    restore any nan-like values that may have been in the original data
    (unless the column was all nans.)


    Parameters
    ----------
    X :
        {array-like, scipy sparse csc} of shape (n_samples,
        n_features - n_features_removed) - A deduplicated data set. The
        container must be numpy ndarray, pandas dataframe or scipy sparse
        csc matrix/array only.
    _removed_columns:
        dict[int, int] - the keys are the indices of duplicate columns
        removed from the original data, indexed by their column location
        in the original data; the values are the column index in the
        original data of the respective duplicate that was kept.
    _feature_names_in:
        Union[npt.NDArray[str], None] - the feature names found during
        fitting, if passed.


    Returns
    -------
    -
        X_inv : {array-like, scipy sparse csc} of shape (n_samples,
            n_features) - Deduplicated data reverted to its original
            state.

    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    assert isinstance(
        X,
        (np.ndarray, pd.core.frame.DataFrame, ss.csc_matrix, ss.csc_array)
    )

    assert isinstance(_removed_columns, dict)
    assert all(map(isinstance, _removed_columns, (int for _ in _removed_columns)))

    if _feature_names_in is not None:
        assert isinstance(_feature_names_in, np.ndarray)
        assert all(
            map(isinstance, _feature_names_in, (str for _ in _feature_names_in))
        )

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # retain what the original format was
    _og_X_format = type(X)

    # if data is a pd df, convert to numpy
    if isinstance(X, pd.core.frame.DataFrame):
        # remove any header that may be on this df, dont want to replicate
        # wrong headers in 'new' columns which would be exposed if
        # feature_names_in_ is not available (meaning that fit() never
        # saw a header via df or _columns, but a df has been passed to
        # inverse_transform.)
        X = X.to_numpy()

    # confirmed via pytest 24_10_10 this need to stay
    # assure _removed_columns keys are accessed ascending
    for k in sorted(_removed_columns.keys()):
        _removed_columns[int(k)] = int(_removed_columns.pop(k))

    # insert blanks into the given data to get dimensions back to original,
    # so indices will match the indices of _parent_dict.
    # must do this from left to right!
    _blank = np.zeros((X.shape[0], 1))

    if isinstance(X, np.ndarray):
        for _rmv_idx in _removed_columns:  # this was sorted above
            X = np.insert(X, _rmv_idx, _blank.ravel(), axis=1)
    elif isinstance(X, (ss.csc_matrix, ss.csc_array)):
        for _rmv_idx in _removed_columns:  # this was sorted above
            X = ss.hstack(
                (X[:, :_rmv_idx], ss.csc_matrix(_blank), X[:, _rmv_idx:]),
                format="csc",
                dtype=X.dtype
            )
    else:
        raise Exception

    del _blank


    # use the _removed_columns dict to put in copies of the parent column

    if isinstance(X, np.ndarray):
        # df was converted to array above!
        for _dupl_idx, _parent_idx in _removed_columns.items():
            X[:, _dupl_idx] = X[:, _parent_idx].copy()
    elif isinstance(X, (ss.csc_matrix, ss.csc_array)):
        for _dupl_idx, _parent_idx in _removed_columns.items():
            X[:, [_dupl_idx]] = X[:, [_parent_idx]].copy()
    else:
        raise Exception


    # if was a dataframe and feature names are available, reattach
    if _og_X_format is np.ndarray:
        pass
    elif _og_X_format is pd.core.frame.DataFrame:
        X = pd.DataFrame(data=X)

        if _feature_names_in is not None:
            X.columns = _feature_names_in
    else:
        X = _og_X_format(X)

    return X














