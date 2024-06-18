# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np

from preprocessing.MinCountTransformer._shared._make_instructions._make_instructions \
    import _make_instructions




# def _make_instructions(
    #     _count_threshold: int,
    #     _ignore_float_columns: bool,
    #     _ignore_non_binary_integer_columns: bool,
    #     _ignore_columns: np.ndarray[int],
    #     _ignore_nan: bool,
    #     _handle_as_bool: np.ndarray[int],
    #     _delete_axis_0: bool,
    #     _original_dtypes: OriginalDtypesDtype,
    #     _n_features_in: int,
    #     _total_counts_by_column: TotalCountsByColumnType,
    #     _threshold: int = None
#     ) -> InstructionsType:




@pytest.fixture
def good_tcbc():

    return {
        0: {'a': 500, 'b': 350, 'c': 100, np.nan: 50},
        1: {0: 640, 1: 350, np.nan: 10},
        2: {0: 200, 2.718: 400, 3.141: 300, 6.638: 50, 8.834: 40, np.nan: 10},
        3: {0: 600, 1: 200, 2: 100, 3: 50, 4: 25, np.nan: 25}
    }


@pytest.fixture
def good_og_dtypes():

    return np.array(['obj', 'int', 'float', 'int'], dtype='<U5')




class TestMakeInstructions:


    # all the validation is handled in individual modules, and tested indiv

    """
    # validation
    _count_threshold, _ignore_float_columns, _ignore_non_binary_integer_columns, \
    _ignore_columns, _ignore_nan, _handle_as_bool, _delete_axis_0, \
    _original_dtypes, _n_features_in, _total_counts_by_column, _threshold = \
        _make_instructions_validation(
            _count_threshold,
            _ignore_float_columns,
            _ignore_non_binary_integer_columns,
            _ignore_columns,
            _ignore_nan,
            _handle_as_bool,
            _delete_axis_0,
            _original_dtypes,
            _n_features_in,
            _total_counts_by_column,
            _threshold
        )
    """

    # random spot check validation
    @pytest.mark.parametrize('junk_value', ('junk', np.nan, np.pi, {1: 2}, min))
    def test_random_validation(self, junk_value, good_og_dtypes, good_tcbc):
        with pytest.raises(TypeError):
            _make_instructions(
                _count_threshold=100,
                _ignore_float_columns=True,
                _ignore_non_binary_integer_columns=True,
                _ignore_columns=junk_value,
                _ignore_nan=False,
                _handle_as_bool=[2,3],
                _delete_axis_0=False,
                _original_dtypes=good_og_dtypes,
                _n_features_in=len(good_og_dtypes),
                _total_counts_by_column=good_tcbc,
                _threshold=None
            )

        with pytest.raises(TypeError):
            _make_instructions(
                _count_threshold=junk_value,
                _ignore_float_columns=True,
                _ignore_non_binary_integer_columns=True,
                _ignore_columns=[0,1],
                _ignore_nan=False,
                _handle_as_bool=[2,3],
                _delete_axis_0=False,
                _original_dtypes=good_og_dtypes,
                _n_features_in=len(good_og_dtypes),
                _total_counts_by_column=good_tcbc,
                _threshold=None
            )

        with pytest.raises(TypeError):
            _make_instructions(
                _count_threshold=100,
                _ignore_float_columns=True,
                _ignore_non_binary_integer_columns=junk_value,
                _ignore_columns=[0,1],
                _ignore_nan=False,
                _handle_as_bool=[2,3],
                _delete_axis_0=False,
                _original_dtypes=good_og_dtypes,
                _n_features_in=len(good_og_dtypes),
                _total_counts_by_column=good_tcbc,
                _threshold=None
            )



    def test_it_runs(self, good_tcbc, good_og_dtypes):
        _make_instructions(
            _count_threshold=100,
            _ignore_float_columns=True,
            _ignore_non_binary_integer_columns=True,
            _ignore_columns=[0,1],
            _ignore_nan=False,
            _handle_as_bool=[2,3],
            _delete_axis_0=False,
            _original_dtypes=good_og_dtypes,
            _n_features_in=len(good_og_dtypes),
            _total_counts_by_column=good_tcbc,
            _threshold=None
        )



    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    def test_ignore_all_columns_returns_all_inactive(self, good_tcbc,
                                                     good_og_dtypes):
        out = _make_instructions(
            _count_threshold=100,
            _ignore_float_columns=True,
            _ignore_non_binary_integer_columns=True,
            _ignore_columns=[0,1,2,3],
            _ignore_nan=False,
            _handle_as_bool=[],
            _delete_axis_0=False,
            _original_dtypes=good_og_dtypes,
            _n_features_in=len(good_og_dtypes),
            _total_counts_by_column=good_tcbc,
            _threshold=None
        )

        assert out == {idx: ['INACTIVE'] for idx in range(4)}


    def test_empty_tcbcs_returns_all_inactive(self, good_og_dtypes):
        out = _make_instructions(
            _count_threshold=100,
            _ignore_float_columns=True,
            _ignore_non_binary_integer_columns=True,
            _ignore_columns=[],
            _ignore_nan=False,
            _handle_as_bool=[],
            _delete_axis_0=False,
            _original_dtypes=good_og_dtypes,
            _n_features_in=len(good_og_dtypes),
            _total_counts_by_column={0:{}, 1:{}, 2:{}, 3:{}},
            _threshold=None
        )

        assert out == {idx: ['INACTIVE'] for idx in range(4)}


    def test_ignore_all_floats_returns_all_inactive(self, good_og_dtypes):
        out = _make_instructions(
            _count_threshold=100,
            _ignore_float_columns=True,
            _ignore_non_binary_integer_columns=True,
            _ignore_columns=[],
            _ignore_nan=False,
            _handle_as_bool=[],
            _delete_axis_0=False,
            _original_dtypes=np.array(['float' for _ in good_og_dtypes]),
            _n_features_in=len(good_og_dtypes),
            _total_counts_by_column={idx:{} for idx in range(len(good_og_dtypes))},
            _threshold=None
        )

        assert out == {idx: ['INACTIVE'] for idx in range(4)}

    def test_ignore_all_binint_returns_all_inactive(self, good_og_dtypes):

        _len = range(len(good_og_dtypes))

        out = _make_instructions(
            _count_threshold=100,
            _ignore_float_columns=True,
            _ignore_non_binary_integer_columns=True,
            _ignore_columns=[],
            _ignore_nan=False,
            _handle_as_bool=[],
            _delete_axis_0=False,
            _original_dtypes=np.array(['int' for _ in _len]),
            _n_features_in=len(good_og_dtypes),
            _total_counts_by_column={i: {1: 10, 2: 10, 3: 10} for i in _len},
            _threshold=None
        )

        assert out == {idx: ['INACTIVE'] for idx in range(4)}


    @pytest.mark.parametrize('tcbc',
        (
        {0: {'a': 500}},
        {0: {'a': 475, np.nan: 25}},
        {0: {'a':250, 'b': 200, np.nan: 50}},
        {0: {'a': 250, 'b': 200, 'c': 25, np.nan: 25}}
        )
    )
    def test_rejects_str_into_hab(self, tcbc):

        with pytest.raises(ValueError):
            _make_instructions(
                _count_threshold=100,
                _ignore_float_columns=True,
                _ignore_non_binary_integer_columns=True,
                _ignore_columns=[],
                _ignore_nan=False,
                _handle_as_bool=np.array([0]),
                _delete_axis_0=False,
                _original_dtypes=np.array(['obj']),
                _n_features_in=1,
                _total_counts_by_column=tcbc,
                _threshold=None
            )

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    def test_accuracy(self):


        out = _make_instructions(
            _count_threshold=5,
            _ignore_float_columns=True,
            _ignore_non_binary_integer_columns=True,
            _ignore_columns=[],
            _ignore_nan=False,
            _handle_as_bool=[],
            _delete_axis_0=True,
            _original_dtypes=np.array(['obj']),
            _n_features_in=1,
            _total_counts_by_column={0: {'a':3, 'b': 9, 'c': 3, np.nan: 4}},
            _threshold=None
        )

        assert out == {0: ['a', 'c', np.nan, 'DELETE COLUMN']}
































