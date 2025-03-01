# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from typing_extensions import Union
from typing import Literal
import numpy.typing as npt

import warnings
from pybear.utilities._nan_masking import nan_mask

from pybear.preprocessing._MinCountTransformer._transform. \
    _parallelized_row_masks import _parallelized_row_masks



#  def _parallelized_row_masks(
#     _X_COLUMN: npt.NDArray[DataType],
#     _COLUMN_UNQ_CT_DICT: dict[DataType, int],
#     _instr: list[Union[str, DataType]],
#     _reject_unseen_values: bool,
#     _col_idx: int
# ) -> npt.NDArray[np.uint8]:


class TestParallelizedRowMasks:

    # Fixtures v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    @staticmethod
    @pytest.fixture(scope='module')
    def _rows():
        return 200


    @staticmethod
    @pytest.fixture(scope='module')
    def _pool_size(_rows):
        return _rows // 20    # dont make this ratio > 26 because of alphas


    @staticmethod
    @pytest.fixture(scope='module')
    def _thresh(_rows, _pool_size):
        return _rows // _pool_size


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_factory(_rows, _thresh):

        def _idx_getter(_rows, _zeros):
            return np.random.choice(
                range(_rows), int(_rows * _zeros), replace=False
            )

        def foo(
            _has_nan: Union[int, bool] = False,
            _dtype: Literal['flt', 'int', 'str', 'obj', 'hybrid'] = 'flt',
            _zeros: Union[float, None] = 0,
            _shape: tuple[int, int] = (20, 5)
        ) -> npt.NDArray[any]:

            # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

            assert isinstance(_has_nan, (bool, int, float))
            if not isinstance(_has_nan, bool):
                assert int(_has_nan) == _has_nan, \
                    f"'_has_nan' must be bool or int >= 0"
            assert _has_nan >= 0, f"'_has_nan' must be bool or int >= 0"
            assert _dtype in ['flt', 'int', 'str', 'obj', 'hybrid']

            if _zeros is None:
                _zeros = 0
            assert not isinstance(_zeros, bool)
            assert isinstance(_zeros, (float, int))
            assert 0 <= _zeros <= 1, f"zeros must be 0 <= x <= 1"

            assert isinstance(_shape, tuple)
            assert all(map(isinstance, _shape, (int for _ in _shape)))
            if _shape[0] < 1:
                raise AssertionError(f"'shape' must have at least one example")
            if _shape[1] < 1:
                raise AssertionError(f"'shape' must have at least 1 column")

            assert _has_nan <= _shape[0], f"'_has_nan' must be <= n_rows"
            # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * **

            str_char = 'abcdefghijklmnopqurstuvwxyz'
            target_num_uniques = int(_rows//_thresh)
            letter_pool = str_char[:target_num_uniques]

            if _dtype == 'flt':
                X = np.random.uniform(0, 1, _shape)
                if _zeros:
                    for _col_idx in range(_shape[1]):
                        X[_idx_getter(_shape[0], _zeros), _col_idx] = 0
            elif _dtype == 'int':
                X = np.random.randint(0, target_num_uniques, _shape)
                if _zeros:
                    for _col_idx in range(_shape[1]):
                        X[_idx_getter(_shape[0], _zeros), _col_idx] = 0
            elif _dtype == 'str':
                X = np.random.choice(list(letter_pool), _shape, replace=True)
                X = X.astype('<U10')
            elif _dtype == 'obj':
                X = np.random.choice(list(letter_pool), _shape, replace=True)
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
                        _ = np.random.randint(
                            0, target_num_uniques, _col_shape
                        )
                        if _zeros:
                            _[_idx_getter(_shape[0], _zeros), 0] = 0
                    elif _cidx % 3 == 2:
                        _ = np.random.choice(list(letter_pool), _col_shape)
                    else:
                        raise Exception
                    X = np.hstack((X, _))
                del _col_shape, _, _cidx
            else:
                raise Exception

            if _has_nan:

                if _dtype == 'int':
                    warnings.warn(
                        f"attempting to put nans into an integer dtype, "
                        f"converted to float"
                    )
                    X = X.astype(np.float64)

                # determine how many nans to sprinkle based on _shape and _has_nan
                if _has_nan is True:
                    _sprinkles = max(3, _shape[0] // 10)
                else:
                    _sprinkles = _has_nan

                for _c_idx in range(_shape[1]):
                    _r_idxs = np.random.choice(
                        range(_shape[0]), _sprinkles, replace=False
                    )
                    for _r_idx in _r_idxs:
                        if _dtype in ('str', 'obj'):
                            # it is important to do the str()
                            X[_r_idx, _c_idx] = str(np.nan)
                        else:
                            X[_r_idx, _c_idx] = np.nan

                del _sprinkles

            return X

        return foo


    @staticmethod
    @pytest.fixture
    def good_unq_ct_dict():

        def foo(any_column):
            _cleaned_column = any_column.copy()
            try:  # excepts on np int dtype
                _cleaned_column[nan_mask(_cleaned_column)] = str(np.nan)
            except:
                pass
            __ = dict((zip(*np.unique(_cleaned_column, return_counts=True))))
            # can only have one nan in it
            _nan_ct = 0
            _new_dict = {}
            for _unq, _ct in __.items():
                if str(_unq) == 'nan':
                    _nan_ct += _ct
                else:
                    _new_dict[_unq] = int(_ct)

            if _nan_ct > 0:
                _new_dict[np.nan] = int(_nan_ct)

            return _new_dict

        return foo


    @staticmethod
    @pytest.fixture
    def good_instr():

        def foo(any_unq_ct_dict, _thresh):
            GOOD_INSTR = []
            for unq, ct in any_unq_ct_dict.items():
                if ct < _thresh:
                    GOOD_INSTR.append(unq)

            if len(GOOD_INSTR) >= len(any_unq_ct_dict) - 1:
                GOOD_INSTR.append('DELETE COLUMN')

            return GOOD_INSTR

        return foo

    # END Fixtures v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


    @pytest.mark.parametrize('_dtype', ('str', 'flt', 'int', 'str', 'obj'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    def test_accuracy_making_delete_masks(
        self, _X_factory, good_instr, good_unq_ct_dict, _thresh, _dtype,
        _has_nan, _rows
    ):

        if _dtype == 'int' and _has_nan is True:
            pytest.skip(reason=f"impossible condition, nans in int np")

        _X = _X_factory(
            _has_nan=_has_nan,
            _dtype=_dtype,
            _zeros=None,
            _shape=(_rows, 1)
        ).ravel()

        UNQ_CT_DICT = good_unq_ct_dict(_X)
        assert len(good_instr(UNQ_CT_DICT, _thresh)) > 0, \
            f"if this excepts, it wants to keep all unqs"

        out = _parallelized_row_masks(
            _X,
            UNQ_CT_DICT,
            good_instr(UNQ_CT_DICT, _thresh),
            _reject_unseen_values=False,
            _col_idx=0
        )

        exp = _X.copy().ravel()
        UNQ_CT_DICT = dict((
            zip(map(str, UNQ_CT_DICT.keys()), UNQ_CT_DICT.values())
        ))
        for idx, value in enumerate(exp):
            if UNQ_CT_DICT[str(value)] < _thresh:
                exp[idx] = 1  # overwriting takes dtype of column_type...
            else:
                exp[idx] = 0 # overwriting takes dtype of column_type...

        # ... so convert to uint8
        exp = exp.astype(np.uint8)

        assert np.array_equiv(out, exp)


    @pytest.mark.parametrize('_dtype', ('str', 'flt', 'int', 'str', 'obj'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    def test_accuracy_reject_unseen(
        self, _X_factory, good_instr, good_unq_ct_dict, _thresh, _dtype,
        _has_nan, _rows
    ):

        if _dtype == 'int' and _has_nan is True:
            pytest.skip(reason=f"impossible condition, nans in int np")

        _X = _X_factory(
            _has_nan=_has_nan,
            _dtype=_dtype,
            _zeros=None,
            _shape=(_rows, 1)
        ).ravel()

        UNQ_CT_DICT = good_unq_ct_dict(_X)
        assert len(good_instr(UNQ_CT_DICT, _thresh)) > 0, \
            f"if this excepts, it wants to keep all unqs"

        # put unseen values into data
        if isinstance(_X[0], str):
            _X[np.random.choice(_rows, int(0.9 * _rows))] = 'z'
        else:
            _X[np.random.choice(_rows, int(0.9 * _rows))] = 13

        # False does not except
        out = _parallelized_row_masks(
            _X,
            UNQ_CT_DICT,
            good_instr(UNQ_CT_DICT, _thresh),
            _reject_unseen_values=False,
            _col_idx=0
        )

        # True raises ValueError
        with pytest.raises(ValueError):
            out = _parallelized_row_masks(
                _X,
                UNQ_CT_DICT,
                good_instr(UNQ_CT_DICT, _thresh),
                _reject_unseen_values=True,
                _col_idx=0
            )





































