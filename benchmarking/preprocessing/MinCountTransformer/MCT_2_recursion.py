# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal, Iterable
from typing_extensions import Union
import numpy.typing as npt

from copy import deepcopy
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as ss

from pybear.preprocessing._MinCountTransformer.MinCountTransformer import \
    MinCountTransformer
from pybear.utilities._nan_masking import nan_mask


# this compares whether MCT with max_recursions=2 is identical to MCT
# 2 times with max_recursions=1.



# function to build an X ** * ** * ** * ** * ** * ** * ** * ** * ** *
def _idx_getter(_rows, _zeros):
    return np.random.choice(range(_rows), int(_rows * _zeros), replace=False)


def foo(
    _dupl: list[list[int]] = None,
    _has_nan: Union[int, bool] = False,
    _format: Literal['np', 'pd', 'csc', 'csr', 'coo'] = 'np',
    _dtype: Literal['flt', 'int', 'str', 'obj', 'hybrid'] = 'flt',
    _columns: Union[Iterable[str], None] = None,
    _zeros: Union[float, None] = 0,
    _shape: tuple[int, int] = (20, 5)
) -> npt.NDArray[any]:

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    assert isinstance(_dupl, (list, type(None)))
    if _dupl is not None:
        for idx, _set in enumerate(_dupl):
            assert isinstance(_set, list)
            assert all(map(isinstance, _set, (int for _ in _set)))
            assert len(_set) >= 2, f'_dupl sets must have at least 2 entries'

        # make sure sets are sorted ascending, and first entries are asc
        __ = {_set[0]: sorted(_set) for _set in _dupl}
        _dupl = [__[k] for k in sorted(list(__.keys()))]
        del __

    assert isinstance(_has_nan, (bool, int, float))
    if not isinstance(_has_nan, bool):
        assert int(_has_nan) == _has_nan, f"'_has_nan' must be bool or int >= 0"
    assert _has_nan >= 0, f"'_has_nan' must be bool or int >= 0"
    assert _format in ['np', 'pd', 'csc', 'csr', 'coo']
    assert _dtype in ['flt', 'int', 'str', 'obj', 'hybrid']
    assert isinstance(_columns, (list, np.ndarray, type(None)))
    if _columns is not None:
        assert all(map(isinstance, _columns, (str for _ in _columns)))

    if _zeros is None:
        _zeros = 0
    assert not isinstance(_zeros, bool)
    assert isinstance(_zeros, (float, int))
    assert 0 <= _zeros <= 1, f"zeros must be 0 <= x <= 1"

    if _format in ('csc', 'csr', 'coo') and \
            _dtype in ('str', 'obj', 'hybrid'):
        raise ValueError(
            f"cannot create csc, csr, or coo with str, obj, or hybrid dtypes"
        )

    assert isinstance(_shape, tuple)
    assert all(map(isinstance, _shape, (int for _ in _shape)))
    if _shape[0] < 1:
        raise AssertionError(f"'shape' must have at least one example")
    if _shape[1] < 2:
        raise AssertionError(f"'shape' must have at least 2 columns")

    assert _has_nan <= _shape[0], f"'_has_nan' must be <= n_rows"
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    if _dtype == 'flt':
        X = np.random.uniform(0, 1, _shape)
        if _zeros:
            for _col_idx in range(_shape[1]):
                X[_idx_getter(_shape[0], _zeros), _col_idx] = 0
    elif _dtype == 'int':
        X = np.random.randint(0, 5, _shape)
        if _zeros:
            for _col_idx in range(_shape[1]):
                X[_idx_getter(_shape[0], _zeros), _col_idx] = 0
    elif _dtype == 'str':
        X = np.random.choice(list('abcdefghijk'), _shape, replace=True)
        X = X.astype('<U10')
    elif _dtype == 'obj':
        X = np.random.choice(list('abcdefghijk'), _shape, replace=True)
        X = X.astype(object)
    elif _dtype == 'hybrid':
        _col_shape = (_shape[0], 1)
        X = np.random.uniform(0, 1, _col_shape).astype(object)
        if _zeros:
            X[_idx_getter(_shape[0], _zeros), 0] = 0
        for _cidx in range(1, _shape[1]):
            if _cidx % 3 == 0:
                _ = np.random.uniform(0, 1, _col_shape)
                if _zeros:
                    _[_idx_getter(_shape[0], _zeros), 0] = 0
            elif _cidx % 3 == 1:
                _ = np.random.randint(0, 10, _col_shape)
                if _zeros:
                    _[_idx_getter(_shape[0], _zeros), 0] = 0
            elif _cidx % 3 == 2:
                _ = np.random.choice(list('abcdefghijk'), _col_shape)
            else:
                raise Exception
            X = np.hstack((X, _))
        del _col_shape, _, _cidx
    else:
        raise Exception

    if _dupl is not None:
        for _set in _dupl:
            for _idx in _set[1:]:
                X[:, _idx] = X[:, _set[0]]
        try:
            del _set, _idx
        except:
            pass

    if _format == 'np':
        pass
    elif _format == 'pd':
        X = pd.DataFrame(data=X, columns=_columns)
    # do conversion to sparse after nan sprinkle

    if _has_nan:

        if _format == 'pd':
            if _dtype in ['flt', 'int']:
                _choices = [np.nan, pd.NA, None, 'nan', 'NaN', 'NAN', '<NA>']
            else:
                _choices = [np.nan, pd.NA, None, 'nan', 'NaN', 'NAN', '<NA>']
        else:
            if _dtype == 'flt':
                _choices = [np.nan, None, 'nan', 'NaN', 'NAN']
            elif _dtype == 'int':
                warnings.warn(
                    f"attempting to put nans into an integer dtype, "
                    f"converted to object dtype"
                )
                X = X.astype(object)
                _choices = [np.nan, None, 'nan', 'NaN', 'NAN']
            else:
                _choices = [np.nan, pd.NA, None, 'nan', 'NaN', 'NAN']

        # determine how many nans to sprinkle based on _shape and _has_nan
        if _has_nan is True:
            _sprinkles = max(3, _shape[0] // 10)
        else:
            _sprinkles = _has_nan

        for _c_idx in range(_shape[1]):
            _r_idxs = np.random.choice(range(_shape[0]), _sprinkles, replace=False)
            for _r_idx in _r_idxs:
                if _format == 'pd':
                    X.iloc[_r_idx, _c_idx] = np.random.choice(_choices)
                else:
                    if _dtype in ('str', 'obj'):
                        # it is important to do the str()
                        X[_r_idx, _c_idx] = str(np.random.choice(_choices))
                    else:
                        X[_r_idx, _c_idx] = np.random.choice(_choices)

        del _sprinkles

    # do this after sprinkling the nans
    if _format == 'csc':
        X = ss.csc_array(X)
    elif _format == 'csr':
        X = ss.csr_array(X)
    elif _format == 'coo':
        X = ss.coo_array(X)

    return X
# END function to build an X ** * ** * ** * ** * ** * ** * ** * ** * ** *



_thresh = 3
x_rows = 200
y_rows = 200
x_cols = 6
y_cols = 2

_base_kwargs = {
    'count_threshold': _thresh,
    'ignore_float_columns': True,
    'ignore_non_binary_integer_columns': False,
    'ignore_columns': None,
    'ignore_nan': True,
    'handle_as_bool': lambda X: [_ for _ in range(x_cols) if _ % 3 == 1],
    'delete_axis_0': True,
    'reject_unseen_values': False,
    'max_recursions': 2,
    'n_jobs': -1
}



for count_threshold in [2,3]:
    for ignore_float_columns in [True, False]:
        for ignore_non_binary_integer_columns in [True, False]:
            for ignore_columns in [None, [1,2,3,4]]:
                for ignore_nan in [True, False]:
                    for handle_as_bool in [None, lambda X: [_ for _ in range(x_cols) if _ % 3 == 1]]:
                        for delete_axis_0 in [True, False]:
                            for reject_unseen_values in [True, False]:
                                print(f'- - - - - - - - - - - - - - - - - - - - - - - - - -')
                                print(
                                    f"\ncount_threshold = {count_threshold}" \
                                    f"\nignore_float_columns = {ignore_float_columns}" \
                                    f"\nignore_non_binary_integer_columns = {ignore_non_binary_integer_columns}" \
                                    f"\nignore_columns = {ignore_columns}" \
                                    f"\nignore_nan = {ignore_nan}" \
                                    f"\nhandle_as_bool = {handle_as_bool}" \
                                    f"\ndelete_axis_0 = {delete_axis_0}" \
                                    f"\nreject_unseen_values = {reject_unseen_values}" \
                                )


                                _kwargs = deepcopy(_base_kwargs)
                                _kwargs['count_threshold'] = count_threshold
                                _kwargs['ignore_float_columns'] = ignore_float_columns
                                _kwargs['ignore_non_binary_integer_columns'] = ignore_non_binary_integer_columns
                                _kwargs['ignore_columns'] = ignore_columns
                                _kwargs['ignore_nan'] = ignore_nan
                                _kwargs['handle_as_bool'] = handle_as_bool
                                _kwargs['delete_axis_0'] = delete_axis_0
                                _kwargs['reject_unseen_values'] = reject_unseen_values

                                X = foo(
                                    _dupl=None,
                                    _has_nan=x_rows//10,
                                    _format='np',
                                    _dtype='hybrid',
                                    _zeros=0,
                                    _columns=None,
                                    _shape=(x_rows, x_cols)
                                )

                                y = np.random.randint(0,2, (y_rows, y_cols))

                                print(f'shapes:')
                                print(X.shape)
                                print(y.shape)
                                print()


                                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                # 2 recursion
                                _mct_2rcr = MinCountTransformer(**_kwargs)

                                try:
                                    TRFM_X_2rcr, TRFM_Y_2rcr = _mct_2rcr.fit_transform(X, y)
                                except ValueError:
                                    continue
                                except:
                                    raise

                                _2rcr_support = _mct_2rcr.get_support(indices=False).copy()
                                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

                                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                # 1 recursion X 2
                                _mct_1rcrX2 = MinCountTransformer(**_kwargs)
                                _mct_1rcrX2.set_params(max_recursions=1)

                                TRFM_X_1, TRFM_Y_1 = _mct_1rcrX2.fit_transform(X, y)
                                # because doing 2 separate recursions, need to adjust h_a_b and ignore_columns in between!
                                _1rcr_first_support = np.array(_mct_1rcrX2.get_support(indices=False))
                                if ignore_columns is not None:
                                    _kept_idxs = np.arange(len(_1rcr_first_support))[_1rcr_first_support]
                                    _new_ignore_columns = []
                                    for new_idx, old_idx in enumerate(_kept_idxs):
                                        if old_idx in ignore_columns:
                                            _new_ignore_columns.append(new_idx)
                                    _mct_1rcrX2.ignore_columns = _new_ignore_columns
                                    del _new_ignore_columns
                                if handle_as_bool is not None:
                                    hab = np.fromiter([True if _ % 3 == 1 else False for _ in range(x_cols)], dtype=int)
                                    _mct_1rcrX2.set_params(handle_as_bool=hab)
                                    del hab
                                TRFM_X_2X1rcr, TRFM_Y_2X1rcr = _mct_1rcrX2.fit_transform(TRFM_X_1, TRFM_Y_1)
                                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

                                # y
                                assert np.array_equal(TRFM_Y_2rcr, TRFM_Y_2X1rcr), \
                                    (f"TRFM_Y_2rcr.shape={TRFM_Y_2rcr.shape}, "
                                     f"TRFM_Y_2X1rcr.shape={TRFM_Y_2X1rcr.shape}")

                                # X
                                if np.array_equal(
                                    TRFM_X_2rcr[np.logical_not(nan_mask(TRFM_X_2rcr))],
                                    TRFM_X_2X1rcr[np.logical_not(nan_mask(TRFM_X_2X1rcr))]
                                ):
                                    pass
                                else:
                                    for _row in range(x_rows):

                                        NOT_NAN_MASK1 = np.logical_not(nan_mask(TRFM_X_2rcr[_row]))
                                        NOT_NAN_MASK2 = np.logical_not(nan_mask(TRFM_X_2X1rcr[_row]))

                                        if np.array_equal(
                                            TRFM_X_2rcr[_row][NOT_NAN_MASK1],
                                            TRFM_X_2X1rcr[_row][NOT_NAN_MASK2]
                                        ):
                                            continue
                                        else:
                                            print(f'- - - - - - - - - - - - - - - - - - - - - -')
                                            print(f'TRFM_X_2rcr:')
                                            print(TRFM_X_2rcr[_row])
                                            print()
                                            print(f'TRFM_X_2X1rcr:')
                                            print(TRFM_X_2X1rcr[_row])





                                # get_support
                                _adj_1rcr_support = np.array(_1rcr_first_support.copy())
                                _idxs = np.arange(len(_adj_1rcr_support))[_adj_1rcr_support]
                                _adj_1rcr_support[_idxs] = _mct_1rcrX2.get_support(indices=False)

                                assert np.array_equal(_adj_1rcr_support, _2rcr_support), \
                                    (f"\n_adj_1rcr_support = {_adj_1rcr_support}, "
                                     f"\n_mct_1rcrX2.get_support = {_2rcr_support}")












