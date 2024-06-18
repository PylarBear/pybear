# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np

from preprocessing.MinCountTransformer._transform._parallelized_row_masks \
    import _parallelized_row_masks



#  def _parallelized_row_masks(
#     _X_COLUMN: np.ndarray[DataType],
#     _COLUMN_UNQ_CT_DICT: dict[DataType, int],
#     _instr: list[Union[str, DataType]],
#     _reject_unseen_values: bool,
#     _col_idx: int
#     ) -> np.ndarray[np.uint8]:


class TestParallelizedRowMasks:

    @staticmethod
    @pytest.fixture
    def _rows():
        return 100


    @staticmethod
    @pytest.fixture
    def _pool_size(_rows):
        return _rows // 20


    @staticmethod
    @pytest.fixture
    def _thresh(_rows, _pool_size):
        return _rows // _pool_size


    @staticmethod
    @pytest.fixture
    def float_column_no_nan(_pool_size, _rows):
        return np.random.randint(0, _pool_size, (_rows, 1)).astype(np.float64)


    @staticmethod
    @pytest.fixture
    def str_column_no_nan(_pool_size, _rows):

        pool = list('abcdefghijklmnopqrstuvwxyz')[:_pool_size]
        return np.random.choice(pool, _rows, replace=True).reshape((-1,1))


    @staticmethod
    @pytest.fixture
    def float_column_nan(float_column_no_nan, _rows, _thresh):

        NAN_MASK = np.random.choice(np.arange(_rows), int(_thresh-1), replace=False)
        float_column_nan = float_column_no_nan.copy()
        float_column_nan[NAN_MASK, 0] = np.nan

        return float_column_nan


    @staticmethod
    @pytest.fixture
    def str_column_nan(str_column_no_nan, _rows, _thresh):

        NAN_MASK = np.random.choice(np.arange(_rows), int(_thresh-1), replace=False)
        str_column_nan = str_column_no_nan.copy().astype('<U3')
        str_column_nan[NAN_MASK, 0] = 'nan'
        return str_column_nan


    @staticmethod
    @pytest.fixture
    def good_unq_ct_dict():

        def foo(any_column):
            return dict((zip(*np.unique(any_column, return_counts=True))))

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



    def test_accuracy_making_delete_masks(self, float_column_no_nan, good_instr,
        good_unq_ct_dict, float_column_nan, str_column_no_nan, _thresh,
        str_column_nan
        ):

        for column_type in (float_column_no_nan, str_column_no_nan,
            str_column_nan, float_column_nan):

            UNQ_CT_DICT = good_unq_ct_dict(column_type)
            assert len(good_instr(UNQ_CT_DICT, _thresh)) > 0

            out = _parallelized_row_masks(
                column_type,
                UNQ_CT_DICT,
                good_instr(UNQ_CT_DICT, _thresh),
                _reject_unseen_values=False,
                _col_idx=0
            )

            exp = column_type.copy().ravel()
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

            assert np.nan not in column_type[np.logical_not(out)]
            assert 'nan' not in column_type[np.logical_not(out)]

            assert np.nan not in column_type[np.logical_not(exp)]
            assert 'nan' not in column_type[np.logical_not(exp)]


    def test_accuracy_reject_unseen(self, float_column_no_nan, good_instr,
        good_unq_ct_dict, float_column_nan, str_column_no_nan, _thresh,
        str_column_nan
        ):

        for column_type in (float_column_no_nan, str_column_no_nan,
            str_column_nan, float_column_nan):

            UNQ_CT_DICT = good_unq_ct_dict(column_type)
            assert len(good_instr(UNQ_CT_DICT, _thresh)) > 0

            # put unseen values into data
            if isinstance(column_type[0][0], str):
                column_type[-1][0] = 'z'
            else:
                column_type[-1][0] = 13

            # False does not except
            out = _parallelized_row_masks(
                column_type,
                UNQ_CT_DICT,
                good_instr(UNQ_CT_DICT, _thresh),
                _reject_unseen_values=False,
                _col_idx=0
            )

            # True raises ValueError
            with pytest.raises(ValueError):
                out = _parallelized_row_masks(
                    column_type,
                    UNQ_CT_DICT,
                    good_instr(UNQ_CT_DICT, _thresh),
                    _reject_unseen_values=True,
                    _col_idx=0
                )





































