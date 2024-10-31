# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import DataType
from typing_extensions import Union

import itertools

from ._find_duplicates import _find_duplicates




def _dupl_idxs(
    _X: DataType,
    _duplicates: Union[list[list[int]], None],
    _rtol: float,
    _atol: float,
    _equal_nan: bool,
    _n_jobs: Union[int, None]
) -> list[list[int]]:

    """
    Find groups of identical columns for the current partial fit with
    _find_duplicates. Compare the newest duplicates found in the current
    partial fit with previous duplicates found on earlier partial fits
    and meld together to produce overall duplicates. Any columns
    previously not identified as equal but currently are equal, are
    coincidentally equal and are not added to the final list. Columns
    previously found to be equal but are not currently equal are removed
    from the final lists of duplicates. The only duplicates retained are
    those columns found to be identical for all partial fits.


    Parameters
    ----------
    _X:
        {array-like, scipy sparse matrix} of shape (n_samples,
        n_features) - the data to be deduplicated.
    _duplicates:
        Union[list[list[int]], None] - the duplicate columns carried over
            from the previous partial fits. Is None if on the first
            partial fit.
    _rtol:
        float, default = 1e-5 - The relative difference tolerance for
            equality. See numpy.allclose.
    _atol:
        float, default = 1e-8 - The absolute tolerance parameter for .
            equality. See numpy.allclose.
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
    _n_jobs:
        Union[int, None] - The number of jobs to use with joblib Parallel
        while comparing columns. Default is to use processes, but can be
        overridden externally using a joblib parallel_config context
        manager. The default number of jobs is None, which uses the
        joblib default when n_jobs is None.


    Return
    ------
    -
        duplicates_: list[list[int]] - the groups of identical columns,
            indicated by their zero-based column index positions.


    """



    assert isinstance(_duplicates, (list, type(None)))
    if _duplicates is not None:
        for _set in _duplicates:
            assert isinstance(_set, list)
            assert all(map(isinstance, _set, (int for _ in _set)))
    try:
        float(_rtol)
    except:
        raise Exception
    try:
        float(_atol)
    except:
        raise Exception
    assert isinstance(_equal_nan, bool)
    assert isinstance(_n_jobs, (int, type(None)))


    duplicates_ = _find_duplicates(_X, _rtol, _atol, _equal_nan, _n_jobs)

    # if _duplicates is None, this is the first pass
    if _duplicates is None:
        return duplicates_
    elif _duplicates is not None:
        # duplicates found on subsequent partial fits cannot increase the
        # number of duplicates over what was found on previous partial
        # fits. If later partial fits find new identical columns, it can
        # only be coincidental, as those columns were previously found to
        # be unequal. The number of duplicates can decrease, however, if
        # later partial fits find non-equality in columns that were
        # previously found to be equal.

        # compare the newest duplicates against the previously found
        # duplicates. Only columns that were in the previous list can
        # carry forward, less any that are not in the newest duplicates.

        _serialized_old_duplicates = list(itertools.chain(*_duplicates))
        _serialized_new_duplicates = list(itertools.chain(*duplicates_))

        # only keep the idxs that they have in common - coincidental
        # identicals will fall out, newly found non-identicals fall out
        _intersection = set(
            _serialized_old_duplicates
        ).intersection(_serialized_new_duplicates)

        # now need to know which idxs in the old duplicates were found to
        # no longer be equal
        _diff = set(_serialized_old_duplicates) - set(_intersection)

        # need to find the _diff idxs in the original buckets, and remove
        # them from their respective buckets.
        duplicates_ = [[v for v in _set if v not in _diff] for _set in _duplicates]
        duplicates_ = [_ for _ in duplicates_ if _ != []]

        return duplicates_












