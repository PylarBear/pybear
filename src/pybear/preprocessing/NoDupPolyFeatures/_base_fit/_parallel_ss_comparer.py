# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy.typing as npt

import numpy as np
import joblib
from copy import deepcopy
import numbers

from ....utilities._nan_masking import nan_mask



@joblib.wrap_non_picklable_objects
def _parallel_ss_comparer(
    _column1: npt.NDArray[any],
    _column2: npt.NDArray[any],
    _rtol: numbers.Real,
    _atol: numbers.Real,
    _equal_nan: bool
) -> bool:

    """
    Compare two columns for equality, subject to :param: rtol, :param:
    atol, and :param: equal_nan. The column must have originated from
    a scipy sparse matrix/array. In this case, the 'column' that is
    passed here is constructed as follows. In the _column_getter module,
    the 'indices' and 'data' attributes are extracted from the pertinent
    column in the scipy sparse object, then those vectors are hstacked
    with indices first and data second. Those objects are then passed
    here via the 'column1' and 'column2' parameters.

    For comparison of columns from numpy arrays and pandas dataframes,
    see _parallel_column_comparer.


    Parameters
    ----------
    _column1:
        npt.NDArray[any] - the first 'column' of a pair to compare for
        equality, extracted from a scipy sparse object.
    _column2:
        npt.NDArray[any] - the second 'column' of a pair to compare for
        equality, extracted from a scipy sparse object.
    _rtol:
        numbers.Real - The relative difference tolerance for equality.
        See numpy.allclose.
    _atol:
        numbers.Real - The absolute difference tolerance for equality.
        See numpy.allclose.
    _equal_nan:
        bool - When comparing pairs of columns row by row:
        If equal_nan is True, exclude from comparison any rows where one
        or both of the values is/are nan. If one value is nan, this
        essentially assumes that the nan value would otherwise be the
        same as its non-nan counterpart. When both are nan, this
        considers the nans as equal (contrary to the default numpy
        handling of nan, where numpy.nan != numpy.nan) and will not in
        and of itself cause a pair of columns to be marked as unequal.
        If equal_nan is False and either one or both of the values in
        the compared pair of values is/are nan, consider the pair to be
        not equivalent, thus making the column pair not equal. This is
        in line with the normal numpy handling of nan values.


    Return
    ------
    -
        bool: True, the columns are equal, False, the columns are unequal.

    """


    # columns can be ravel() or reshape((-1,1)), they get ravel() later
    __ = _column1.shape
    assert len(__) == 1 or (len(__)==2 and __[1] == 1)
    __ = _column2.shape
    assert len(__) == 1 or (len(__)==2 and __[1] == 1)
    del __


    # ss can only be numeric data! do not need to worry about
    # characterizing the vectors for different handling if they are
    # non-numeric vs numeric.


    MASK1 = nan_mask(_column1)
    MASK2 = nan_mask(_column2)

    # v^v^v^ ALL OF THIS JUST TO HANDLE SCIPY SPARSE W/O toarray() v^v^v^v^v^v
    # if using scipy sparse, the "column" being compared is an hstack of
    # the indices and values of the dense in that column. It is very
    # possible that the two vectors being compared will have different
    # length, and the numpy vectorization used below will not broadcast.
    # Compare the lengths of the vectors here and if different, make some
    # other assessment that returns early and short-circuits before the
    # code at the bottom, or find a way to make the vectors have same len.



    # if column lengths are equal proceed to tests below, otherwise....
    if len(_column1) != len(_column2):
        # if the 2 columns have unequal len, there is only one chance that
        # they could actually be equal, if they have nans and at least
        # one of the nans is vis-a-vis a zero, in which case the column
        # with nan has an extra slot and the other has no slot because
        # it is a zero
        if np.sum(MASK1) + np.sum(MASK2) == 0:
            # if there are no nans, no chance the columns are equal
            return False

    # if len(_column1) != len(_column2):
    #
    # else: # there are nans in at least one of the columns

    # this is the difficult case --- _equal_nan is True,
    # the vectors are unequal length, and nans are in at
    # least one of them
    # the difficult thing here is that an nan might be
    # vis-a-vis with a non-dense value in the other vector.
    # build dictionaries of the two vectors, key is index and
    # value is the data value
    # the only way we can get here is if columns are from ss
    # column must always have even length
    _dict1_idxs = _column1.ravel()[:len(_column1) // 2]
    _dict1_values = _column1.ravel()[len(_column1) // 2:]
    if 'nan' in map(str, _dict1_values) and not _equal_nan:
        return False

    _dict1 = dict((zip(_dict1_idxs, _dict1_values)))
    del _dict1_idxs, _dict1_values

    _dict2_idxs = _column2.ravel()[:len(_column2) // 2]
    _dict2_values = _column2.ravel()[len(_column2) // 2:]
    if 'nan' in map(str, _dict2_values) and not _equal_nan:
        return False

    _dict2 = dict((zip(_dict2_idxs, _dict2_values)))
    del _dict2_idxs, _dict2_values

    # _dict1 and _dict2 cannot be the same length
    # go thru both dictionaries, look for nans in the values
    # if the key for the nan is not in the other dictionary,
    # it is associated with a non-dense, so pop it.
    for k, v in deepcopy(_dict1).items():
        if str(v) == 'nan' and k not in _dict2:
            _dict1.pop(k)

    for k, v in deepcopy(_dict2).items():
        if str(v) == 'nan' and k not in _dict1:
            _dict2.pop(k)

    # at this point, if the 2 dictionaries have unequal len,
    # short circuit out. if equal, convert back to vectors
    # and send to the code below.
    if len(_dict1) != len(_dict2):
        return False
    else:
        _column1 = np.hstack((list(_dict1.keys()), list(_dict1.values())))
        _column2 = np.hstack((list(_dict2.keys()), list(_dict2.values())))
        del _dict1, _dict2

        MASK1 = nan_mask(_column1)
        MASK2 = nan_mask(_column2)
    # v^v^v^ ALL OF THIS JUST TO HANDLE SCIPY SPARSE W/O toarray() v^v^v^v^v^v


    NOT_NAN_MASK = np.logical_not((MASK1 + MASK2).astype(bool))
    del MASK1, MASK2

    if _equal_nan:
        return np.allclose(
            _column1[NOT_NAN_MASK].astype(np.float64),
            _column2[NOT_NAN_MASK].astype(np.float64),
            rtol=_rtol,
            atol=_atol
        )

    elif not _equal_nan:
        return np.allclose(
            _column1.astype(np.float64),
            _column2.astype(np.float64),
            rtol=_rtol,
            atol=_atol
        )




