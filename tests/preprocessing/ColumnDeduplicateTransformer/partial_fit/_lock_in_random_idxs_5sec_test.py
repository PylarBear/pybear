# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing._ColumnDeduplicateTransformer._partial_fit. \
    _lock_in_random_idxs import _lock_in_random_idxs

import pytest



# def _lock_in_random_idxs(
#     _duplicates: list[list[int]],
#     _do_not_drop: Union[Sequence[int], Sequence[str], None],
#     _columns: Union[Sequence[str], None]
# ) -> tuple[int]:




class Fixtures:

    @staticmethod
    @pytest.fixture()
    def _cols():
        return 5


    @staticmethod
    @pytest.fixture()
    def _duplicates():
        return [[0,1], [2,3]]


    @staticmethod
    @pytest.fixture()
    def _do_not_drop():
        return [0, 1]


    @staticmethod
    @pytest.fixture()
    def _columns(_master_columns, _cols):
        return _master_columns.copy()[:_cols]



class TestLIRIValidation(Fixtures):

    # test validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # ------------------------------
    @pytest.mark.parametrize('junk_duplicates',
        (-1,0,1,3.14,None,True,False,[0,1],(0,1),(0,),{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_duplicates(
        self, junk_duplicates, _do_not_drop, _columns
    ):

        with pytest.raises(AssertionError):
            _lock_in_random_idxs(
                junk_duplicates,
                _do_not_drop,
                _columns
            )


    @pytest.mark.parametrize('bad_duplicates',
        ([['a','b'], ['c','d']], [[2,2],[2,2]])
    )
    def test_rejects_bad_duplicates(
        self, bad_duplicates, _do_not_drop, _columns
    ):
        with pytest.raises(AssertionError):
            _lock_in_random_idxs(
                bad_duplicates,
                _do_not_drop,
                _columns,
            )


    def test_accepts_good_duplicates(self, _do_not_drop, _columns):

        _lock_in_random_idxs(
            [[0,1],[2,3]],
            _do_not_drop,
            _columns
        )
    # ------------------------------


    # ------------------------------
    @pytest.mark.parametrize('junk_do_not_drop',
        (-1,0,1,3.14,True,False,{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_do_not_drop(
        self, _duplicates, junk_do_not_drop, _columns
    ):
        with pytest.raises(AssertionError):
            _lock_in_random_idxs(
                _duplicates,
                junk_do_not_drop,
                _columns
            )


    @pytest.mark.parametrize('bad_do_not_drop',
        ([min, max], [True, False], [[], []])
)
    def test_rejects_bad_do_not_drop(
        self, _duplicates, bad_do_not_drop, _columns
    ):
        with pytest.raises(AssertionError):
            _lock_in_random_idxs(
                _duplicates,
                bad_do_not_drop,
                _columns
            )


    @pytest.mark.parametrize('_do_not_drop',
        ([0,1,2], 'str', None))
    def test_accepts_good_do_not_drop(self, _duplicates, _do_not_drop, _columns):
        if _do_not_drop == 'str':
            _do_not_drop = [_columns[0], _columns[1], _columns[-1]]

        _lock_in_random_idxs(
            _duplicates,
            _do_not_drop,
            _columns
        )
    # ------------------------------

    # ------------------------------
    @pytest.mark.parametrize('junk_columns',
        (-1,0,1,3.14,True,False,[0,1],(0,1),(0,),{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_columns(self, _duplicates, _do_not_drop, junk_columns):

        with pytest.raises(AssertionError):
            _lock_in_random_idxs(
                _duplicates,
                _do_not_drop,
                junk_columns
            )


    @pytest.mark.parametrize('bad_columns', ([0,1,2,3,4], [True, False]))
    def test_rejects_bad_columns(self, _duplicates, _do_not_drop, bad_columns):

        with pytest.raises(AssertionError):
            _lock_in_random_idxs(
                _duplicates,
                _do_not_drop,
                bad_columns
            )


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    def test_accepts_good_columns(
        self, _duplicates, _do_not_drop, _columns, _columns_is_passed
    ):
        _lock_in_random_idxs(
            _duplicates,
            _do_not_drop,
            _columns if _columns_is_passed else None
        )
    # ------------------------------

    # ------------------------------
    def test_rejects_str_do_not_drop_if_no_columns(self, _duplicates, _columns):

        with pytest.raises(AssertionError):
            _lock_in_random_idxs(
                _duplicates=_duplicates,
                _do_not_drop=[_columns[0], _columns[-1]],
                _columns=None
            )
    # ------------------------------

    # END test validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **



class TestLIRIAccuracy(Fixtures):

    # accuracy ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    @pytest.mark.parametrize('_do_not_drop', (None, [0], [0,1], [0,2]))
    def test_no_duplicates(
        self, _columns, _do_not_drop, _columns_is_passed
    ):

        # no duplicates, so _rand_idxs should be empty

        rand_idxs_out = _lock_in_random_idxs(
            _duplicates=[],
            _do_not_drop=_do_not_drop,
            _columns=_columns if _columns_is_passed else None,
        )

        assert isinstance(rand_idxs_out, tuple)
        assert rand_idxs_out == tuple()


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    def test_do_not_drop_is_none(
        self, _duplicates, _columns, _columns_is_passed
    ):

        # without any restrictions from do_not_drop, any idx could be
        # pulled from each set of duplicates

        rand_idxs_out = _lock_in_random_idxs(
            _duplicates,
            _do_not_drop=None,
            _columns=_columns if _columns_is_passed else None,
        )

        assert isinstance(rand_idxs_out, tuple)

        for _idx, _set in enumerate(_duplicates):
            assert list(rand_idxs_out)[_idx] in _set


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    @pytest.mark.parametrize('_do_not_drop', ([0], [0, 2], [0, 1]))
    def test_with_do_not_drop(
        self, _duplicates, _do_not_drop, _columns, _columns_is_passed
    ):

        #     def _duplicates():
        #         return [[0,1], [2,3]]


        rand_idxs_out = _lock_in_random_idxs(
            _duplicates,
            _do_not_drop=_do_not_drop,
            _columns=_columns if _columns_is_passed else None
        )

        assert isinstance(rand_idxs_out, tuple)

        # if there is no conflict with do_not_drop and duplicates, then
        # _liri() should pick the do_not_drop idx as the random idx.
        # but with 2+ do_not_drop idxs from the same set of duplicates,
        # _liri() will pick randomly from (the 2+ do_not_drop idxs from
        # the same set of duplicates). in that case, all we can validate
        # is that that position in rand_idxs_out contains one of the
        # do_not_drop idxs
        for _idx, _set in enumerate(_duplicates):
            dnd_in_set = list(set(_do_not_drop).intersection(_set))
            num_dnd = len(dnd_in_set)
            if num_dnd == 0:
                # kept would be randomly selected from _set
                assert list(rand_idxs_out)[_idx] in _set
            elif num_dnd == 1:
                # kept would be the one that is in dnd
                assert list(rand_idxs_out)[_idx] == dnd_in_set[0]
            elif num_dnd >= 2:
                # kept would be randomly selected from those in dnd
                assert list(rand_idxs_out)[_idx] in dnd_in_set







