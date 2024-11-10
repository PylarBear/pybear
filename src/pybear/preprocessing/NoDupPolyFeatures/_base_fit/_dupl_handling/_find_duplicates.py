# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.NoDupPolyFeatures._type_aliases import DataType
from typing_extensions import Union

from ._column_getter import _column_getter
from ._parallel_column_comparer import _parallel_column_comparer

from joblib import Parallel, delayed



def _find_duplicates(
    _X: DataType,
    _rtol: float,
    _atol: float,
    _equal_nan: bool,
    _n_jobs: Union[int, None]
) -> list[list[int]]:

    """
    Find identical columns in X. Create a list of lists, where each list
    indicates the zero-based column indices of columns that are identical.
    For example, if column indices 0 and 23 are identical, and indices 8,
    12, and 19 are identical, the returned object would be
    [[0, 23], [8, 12, 19]]. It is important that the first indices of
    each subset be sorted ascending in the outer container, i.e., in
    this example, 0 is before 8.


    Parameters
    ----------
    _X:
        {array-like, scipy sparse matrix} of shape (n_samples,
        n_features) - The data to be deduplicated.
    _rtol:
        float - the relative difference tolerance for equality
    _atol:
        float - the absolute difference tolerance for equality.
    _equal_nan:
        bool, default = False - When comparing pairs of columns row by
        row:
        If equal_nan is True, exclude from comparison any rows where one
        or both of the values is/are nan. If one value is nan, this
        essentially assumes that the nan value would otherwise be the
        same as its non-nan counterpart. When both are nan, this
        considers the nans as equal (contrary to the default numpy
        handling of nan, where np.nan != np.nan) and will not in and of
        itself cause a pair of columns to be marked as unequal.
        If equal_nan is False and either one or both of the values in
        the compared pair of values is/are nan, consider the pair to be
        not equivalent, thus making the column pair not equal. This is
        in line with the normal numpy handling of nan values.
    n_jobs:
        Union[int, None], default = -1 - The number of joblib Parallel
        jobs to use when comparing columns. The default is to use
        processes, but can be overridden externally using a joblib
        parallel_config context manager. The default number of jobs is
        -1 (all processors).


    Return
    ------
    -
        GROUPS: list[list[int]] - lists indicating the column indices of
            identical columns.


    """


    assert isinstance(_rtol, float)
    assert isinstance(_atol, float)
    assert isinstance(_equal_nan, bool)
    assert isinstance(_n_jobs, (int, type(None)))

    duplicates_: dict[int: list[int]] = {int(i): [] for i in range(_X.shape[1])}

    _all_duplicates = []  # not used later, just a helper to track duplicates

    kwargs = {'return_as':'list', 'prefer':'processes', 'n_jobs':_n_jobs}
    args = (_rtol, _atol, _equal_nan)

    # .shape works for np, pd, and scipy.sparse
    for col_idx1 in range(_X.shape[1] - 1):

        if col_idx1 in _all_duplicates:
            continue

        RANGE = range(col_idx1 + 1, _X.shape[1])
        IDXS = [i for i in RANGE if i not in _all_duplicates]

        hits = Parallel(**kwargs)(
            delayed(_parallel_column_comparer)(
                *_column_getter(_X, col_idx1, col_idx2), *args) for col_idx2 in IDXS
        )

        if any(hits):
            _all_duplicates.append(col_idx1)

        for idx, hit in zip(IDXS, hits):
            if hit:
                duplicates_[col_idx1].append(idx)
                _all_duplicates.append(idx)

        """
        code that worked pre-joblib * * * * * * * * *
        # .shape works for np, pd, and scipy.sparse
        for col_idx2 in range(col_idx1 + 1, _X.shape[1]):

            if col_idx2 in _all_duplicates:
                continue

            if _column_comparer(_X, col_idx1, col_idx2):
                duplicates_[col_idx1].append(col_idx2)
                _all_duplicates.append(col_idx1)
                _all_duplicates.append(col_idx2)
        """

    del _all_duplicates, kwargs, RANGE, IDXS, hits

    # ONLY RETAIN INFO FOR COLUMNS THAT ARE DUPLICATE
    duplicates_ = {int(k): v for k, v in duplicates_.items() if len(v) > 0}

    # UNITE DUPLICATES INTO GROUPS
    GROUPS = []
    for idx1, v1 in duplicates_.items():
        __ = sorted([int(idx1)] + v1)
        GROUPS.append(__)

    # ALL SETS OF DUPLICATES MUST HAVE AT LEAST 2 ENTRIES
    for _set in GROUPS:
        assert len(_set) >= 2

    return GROUPS

















