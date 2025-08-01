# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import (
    Any,
    Union
)
import numpy.typing as npt

import numbers
import uuid

import numpy as np

from ....utilities import nan_mask



def _parallel_constant_finder(
    _chunk: npt.NDArray[Any],
    _equal_nan: bool,
    _rtol: numbers.Real,
    _atol: numbers.Real
) -> list[Union[uuid.UUID, Any]]:
    """
    Determine if a column holds a constant value, subject to `_equal_nan`,
    `_rtol`, and `_atol`. If there is a constant value, return the value.
    Otherwise, return a uuid4.

    For numerical columns, get the mean of all the values and compare
    against each of the values; if all of the values are within `rtol` /
    `atol` of the mean, then the column is constant.

    For non-numerical columns, count the number of unique values and if
    there is only one, then that column is constant.

    For both data types, if no nan-like values are present then the
    operation is straightforward as above. When nan-like values are
    present and `_equal_nan` is False, then the column is not constant.
    If `_equal_nan` is True, then perform the above operations on the
    non-nan-like values; if the column contains all nan-likes then
    return the nan value.

    Originally this module took in one column from `X` and returned a
    single value. Joblib choked on this one-column-at-a-time approach,
    serializing individual columns just to do this operation was a waste.
    So this module was converted to handle a chunk of columns of `X`,
    scan it, and return a list of results. Handling chunks instead of
    columns still was not enough to make the cost of serializing the data
    worthwhile compared to the low cost of assessing if a column is
    constant. Joblib has been removed completely.

    Parameters
    ----------
    _chunk : NDArray[Any]
        Columns drawn from `X` as a numpy array.
    _equal_nan : bool
        If `equal_nan` is True, exclude nan-likes from computations
        that discover constant columns. This essentially assumes that
        the nan value would otherwise be equal to the mean of the non-nan
        values in the same column. If `equal_nan` is False and any value
        in a column is nan, do not assume that the nan value is equal to
        the mean of the non-nan values in the same column, thus making
        the column non-constant. This is in line with the normal numpy
        handling of nan values.
    _rtol : numbers.Real
        The relative difference tolerance for equality. Must be a
        non-boolean, non-negative, real number. See numpy.allclose.
    _atol : numbers.Real
        The absolute difference tolerance for equality. Must be a
        non-boolean, non-negative, real number. See numpy.allclose.

    Returns
    -------
    _constants : list[Union[uuid.uuid4, Any]]
        A list of the results for each column in `_chunk`. if a column
        is not constant, returns a uuid4 identifier; if it is constant,
        returns the constant value.

    """


    # validation ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    assert isinstance(_chunk, np.ndarray)
    assert isinstance(_equal_nan, bool)
    try:
        float(_rtol)
        assert _rtol >= 0
        float(_atol)
        assert _atol >= 0
    except:
        raise ValueError(
            f"'rtol' and 'atol' must be real, non-negative numbers"
        )
    # END validation ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    if len(_chunk.shape) == 1:
        _chunk = _chunk.reshape((-1, 1))

    _constants = []
    for _c_idx in range(_chunk.shape[1]):

        _column = _chunk[:, _c_idx]

        _nan_mask = nan_mask(_column)
        # if the column is all nans, just short circuit out
        if all(_nan_mask):
            _constants.append(_column[0] if _equal_nan else uuid.uuid4())
            continue


        # determine if is num or str
        _is_num = False
        _is_str = False
        try:
            _column[np.logical_not(_nan_mask)].astype(np.float64)
            _is_num = True
        except Exception as e:
            _is_str = True

        _has_nans = any(_nan_mask)

        # 4 cases for both flt and str:
        # has nan and _equal_nan
        # has nan and not _equal_nan
        # no nan and _equal_nan
        # no nan and not _equal_nan

        if _is_num and _has_nans:
            if not _equal_nan:
                out = uuid.uuid4()
            elif _equal_nan:
                _not_nan_mask = np.logical_not(nan_mask(_column))
                _mean_value = np.mean(_column[_not_nan_mask].astype(np.float64))
                _allclose = np.allclose(
                    _mean_value,
                    _column[_not_nan_mask].astype(np.float64), # float64 is important
                    rtol=_rtol,
                    atol=_atol,
                    equal_nan=True
                )

                out = _mean_value if _allclose else uuid.uuid4()
                del _not_nan_mask, _mean_value, _allclose

        elif _is_num and not _has_nans:
            # no nans, _equal_nan doesnt matter
            _mean_value = np.mean(_column.astype(np.float64)) # float64 is important
            _allclose = np.allclose(
                _column.astype(np.float64),    # float64 is important
                _mean_value,
                rtol=_rtol,
                atol=_atol
            )
            out = _mean_value if _allclose else uuid.uuid4()
            del _mean_value, _allclose

        elif _is_str and _has_nans:
            if not _equal_nan:
                out = uuid.uuid4()
            elif _equal_nan:
                # get uniques of non-nans
                _unq = np.unique(_column[np.logical_not(_nan_mask)])
                out = _unq[0] if len(_unq) == 1 else uuid.uuid4()
                del _unq

        elif _is_str and not _has_nans:
            # no nans, _equal_nan doesnt matter
            _unq = np.unique(_column)
            out = _unq[0] if len(_unq) == 1 else uuid.uuid4()
            del _unq

        else:
            raise Exception(f"algorithm failure")

        _constants.append(out)


        del _column, _is_num, _is_str, _has_nans, _nan_mask


    return _constants





