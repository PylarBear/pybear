# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _identify_idxs_to_delete import _identify_idxs_to_delete

from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _lock_in_random_idxs import _lock_in_random_idxs

import pytest



# def _identify_idxs_to_delete(
#     _duplicates: list[list[int]],
#     _keep: Literal['first', 'last', 'random'],
#     _do_not_drop: Union[Sequence[int], Sequence[str], None],
#     _columns: Union[Sequence[str], None],
#     _conflict: Literal['raise', 'ignore'],
#     _rand_idxs: tuple[int]
# ) -> dict[int, int]:




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
    def _rand_idxs():
        # len and numbers must match _duplicates
        return (1, 3)


    @staticmethod
    @pytest.fixture()
    def _keep():
        return 'first'


    @staticmethod
    @pytest.fixture()
    def _do_not_drop():
        return [0, 1]


    @staticmethod
    @pytest.fixture()
    def _columns(_master_columns, _cols):
        return _master_columns.copy()[:_cols]


    @staticmethod
    @pytest.fixture()
    def _conflict():
        return 'ignore'


class TestIITDValidation(Fixtures):

    # test validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # ------------------------------
    @pytest.mark.parametrize('junk_duplicates',
        (-1,0,1,3.14,None,True,False,[0,1],(0,1),(0,),{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_duplicates(
        self, junk_duplicates, _keep, _do_not_drop, _columns, _conflict,
        _rand_idxs
    ):

        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                junk_duplicates,
                _keep,
                _do_not_drop,
                _columns,
                _conflict,
                _rand_idxs
            )


    @pytest.mark.parametrize('bad_duplicates',
        ([['a','b'], ['c','d']], [[2,2],[2,2]])
    )
    def test_rejects_bad_duplicates(
        self, bad_duplicates, _keep, _do_not_drop, _columns, _conflict,
        _rand_idxs
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                bad_duplicates,
                _keep,
                _do_not_drop,
                _columns,
                _conflict,
                _rand_idxs
            )


    def test_accepts_good_duplicates(
        self, _keep, _do_not_drop, _columns, _conflict, _rand_idxs
    ):
        _identify_idxs_to_delete(
            [[0,1],[2,3]],
            _keep,
            _do_not_drop,
            _columns,
            _conflict,
            _rand_idxs
        )
    # ------------------------------

    # ------------------------------
    @pytest.mark.parametrize('junk_keep',
        (-1,0,1,3.14,None,True,False,[0,1],(0,),{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_keep(
        self, _duplicates, junk_keep, _do_not_drop, _columns, _conflict,
        _rand_idxs
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                junk_keep,
                _do_not_drop,
                _columns,
                _conflict,
                _rand_idxs
            )


    @pytest.mark.parametrize('bad_keep', ('trash', 'junk', 'garbage'))
    def test_rejects_bad_keep(
        self, _duplicates, bad_keep, _do_not_drop, _columns, _conflict,
        _rand_idxs
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                bad_keep,
                _do_not_drop,
                _columns,
                _conflict,
                _rand_idxs
            )


    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    def test_accepts_good_keep(
        self, _duplicates, _keep, _do_not_drop, _columns, _conflict, _rand_idxs
    ):
        _identify_idxs_to_delete(
            _duplicates,
            _keep,
            _do_not_drop,
            _columns,
            _conflict,
            _rand_idxs
        )
    # ------------------------------

    # ------------------------------
    @pytest.mark.parametrize('junk_do_not_drop',
        (-1,0,1,3.14,True,False,{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_do_not_drop(
        self, _duplicates, _keep, junk_do_not_drop, _columns, _conflict,
        _rand_idxs
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                _keep,
                junk_do_not_drop,
                _columns,
                _conflict,
                _rand_idxs
            )


    @pytest.mark.parametrize('bad_do_not_drop',
        ([min, max], [True, False], [[], []])
)
    def test_rejects_bad_do_not_drop(
        self, _duplicates, _keep, bad_do_not_drop, _columns, _conflict,
        _rand_idxs
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                _keep,
                bad_do_not_drop,
                _columns,
                _conflict,
                _rand_idxs
            )


    @pytest.mark.parametrize('_do_not_drop',
        ([0,1,2], 'str', None))
    def test_accepts_good_do_not_drop(
        self, _duplicates, _keep, _do_not_drop, _columns, _conflict, _rand_idxs
    ):
        if _do_not_drop == 'str':
            _do_not_drop = [_columns[0], _columns[1], _columns[-1]]

        _identify_idxs_to_delete(
            _duplicates,
            _keep,
            _do_not_drop,
            _columns,
            _conflict,
            _rand_idxs
        )
    # ------------------------------

    # ------------------------------
    @pytest.mark.parametrize('junk_columns',
        (-1,0,1,3.14,True,False,[0,1],(0,1),(0,),{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_columns(
        self, _duplicates, _keep, _do_not_drop, junk_columns, _conflict,
        _rand_idxs
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                _keep,
                _do_not_drop,
                junk_columns,
                _conflict,
                _rand_idxs
            )


    @pytest.mark.parametrize('bad_columns', ([0,1,2,3,4], [True, False]))
    def test_rejects_bad_columns(
        self, _duplicates, _keep, _do_not_drop, bad_columns, _conflict,
        _rand_idxs
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                _keep,
                _do_not_drop,
                bad_columns,
                _conflict,
                _rand_idxs
            )


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    def test_accepts_good_columns(
        self, _duplicates, _keep, _do_not_drop, _columns, _conflict,
            _columns_is_passed, _rand_idxs
    ):
        _identify_idxs_to_delete(
            _duplicates,
            _keep,
            _do_not_drop,
            _columns if _columns_is_passed else None,
            _conflict,
            _rand_idxs
        )
    # ------------------------------

    # ------------------------------
    def test_reject_str_do_not_drop_if_no_columns(
        self, _duplicates, _columns, _rand_idxs
    ):

        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates=_duplicates,
                _keep='first',
                _do_not_drop=[_columns[0], _columns[-1]],
                _columns=None,
                _conflict='raise',
                _rand_idxs=_rand_idxs
            )
    # ------------------------------

    # ------------------------------
    @pytest.mark.parametrize('junk_conflict',
        (-1,0,1,3.14,None,True,False,[0,1],(0,),{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_conflict(
        self, _duplicates, _keep, _do_not_drop, _columns, junk_conflict,
        _rand_idxs
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                _keep,
                _do_not_drop,
                _columns,
                junk_conflict,
                _rand_idxs
            )


    @pytest.mark.parametrize('bad_conflict', ('junk', 'trash', 'garbage'))
    def test_rejects_bad_conflict(
        self, _duplicates, _keep, _do_not_drop, _columns, bad_conflict,
        _rand_idxs
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                _keep,
                _do_not_drop,
                _columns,
                bad_conflict,
                _rand_idxs
            )


    @pytest.mark.parametrize('_conflict', ('raise', 'ignore'))
    def test_accepts_good_conflict(
        self, _duplicates, _columns, _conflict, _rand_idxs
    ):

        _identify_idxs_to_delete(
            _duplicates,
            'first',
            [0, 2],
            _columns,
            _conflict,
            _rand_idxs
        )
    # ------------------------------


    # ------------------------------
    @pytest.mark.parametrize('junk_rand_idxs',
        (-1,0,1,3.14,None,True,False,[0,1],(0,),{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_rand_idxs(
        self, _duplicates, _keep, _do_not_drop, _columns, _conflict,
        junk_rand_idxs
    ):

        #     def _duplicates():
        #         return [[0,1], [2,3]]

        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                _keep,
                _do_not_drop,
                _columns,
                _conflict,
                junk_rand_idxs
            )


    @pytest.mark.parametrize('bad_rand_idxs', ((-2, -1), (0,2,4), (0, 1000)))
    def test_rejects_bad_rand_idxs(
        self, _duplicates, _keep, _do_not_drop, _columns, _conflict,
        bad_rand_idxs
    ):

        # length does not match duplicates or idxs are out of range

        #     def _duplicates():
        #         return [[0,1], [2,3]]

        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                _keep,
                _do_not_drop,
                _columns,
                _conflict,
                bad_rand_idxs
            )


    @pytest.mark.parametrize('good_rand_idxs', ((0,2), (1,3)))
    def test_accepts_good_rand_idxs(
        self, _duplicates, _columns, _conflict, good_rand_idxs
    ):

        #     def _duplicates():
        #         return [[0,1], [2,3]]

        _identify_idxs_to_delete(
            _duplicates,
            'first',
            [0, 2],
            _columns,
            _conflict,
            good_rand_idxs
        )


    @pytest.mark.parametrize('duplicates_', ([[0,2],[1,3]], [], [[0,1,2,3]]))
    @pytest.mark.parametrize('rand_idxs_',  ((0,2), (1,3), tuple(), (0,1)))
    def test_rejects_rand_idxs_does_not_match_duplicates(
        self, _columns, _conflict, duplicates_, rand_idxs_
    ):

        #     def _duplicates():
        #         return [[0,1], [2,3]]

        do_not_match = False
        if len(duplicates_) != len(rand_idxs_):
            do_not_match += 1
        else:
            for _idx in range(len(duplicates_)):
                if rand_idxs_[_idx] not in duplicates_[_idx]:
                    do_not_match += 1

        if do_not_match:
            with pytest.raises(AssertionError):
                _identify_idxs_to_delete(
                    duplicates_,
                    'first',
                    None,
                    _columns,
                    _conflict,
                    rand_idxs_
                )
        else:
            _identify_idxs_to_delete(
                duplicates_,
                'first',
                None,
                _columns,
                _conflict,
                rand_idxs_
            )

    # END test validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


class TestIITDConflict(Fixtures):

    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    def test_conflict(
        self, _duplicates, _do_not_drop, _columns, _columns_is_passed,
        _rand_idxs
    ):

        #     def _duplicates():
        #         return [[0, 1], [2, 3]]

        #     def _do_not_drop():
        #         return [0, 1]

        #     def _rand_idxs():
        #         return (1, 3)

        # do not drop idxs are both from the same set of duplicates -----------
        for _dnd in ([0,1], [2,3]):
            for _conflict in ('raise', 'ignore'):
                if _conflict == 'raise':
                    with pytest.raises(ValueError):
                        # two do_not_drop idxs in the same set of duplicates
                        _identify_idxs_to_delete(
                            _duplicates,
                            _keep='first',
                            _do_not_drop=_dnd,   # [0,1], [2,3]
                            _columns=_columns if _columns_is_passed else None,
                            _conflict=_conflict,
                            _rand_idxs=_rand_idxs
                        )
                elif _conflict == 'ignore':
                    # two do_not_drop idxs in the same set of duplicates, but
                    # does not raise because of 'ignore'
                    removed_columns_out = _identify_idxs_to_delete(
                        _duplicates,
                        'first',
                        _do_not_drop=[2, 3],
                        _columns=_columns if _columns_is_passed else None,
                        _conflict=_conflict,
                        _rand_idxs=_rand_idxs
                    )

                    assert isinstance(removed_columns_out, dict)

                    if _dnd == [0, 1]:
                        assert removed_columns_out == {1: 0, 3: 2}
                    elif _dnd == [2, 3]:
                        assert removed_columns_out == {1: 0, 3: 2}
                    else:
                        raise Exception
        # END do not drop idxs are both from the same set of duplicates -------


        # ** ** ** ** **
        for _conflict in ('raise', 'ignore'):
            if _conflict == 'raise':
                with pytest.raises(ValueError):
                    # wants so keep 0 because of 'first' but _do_not_drop [1]
                    _identify_idxs_to_delete(
                        _duplicates,
                        'first',
                        _do_not_drop=[1],
                        _columns=_columns if _columns_is_passed else None,
                        _conflict=_conflict,
                        _rand_idxs=_rand_idxs
                    )

            else:
                # do_not_drop == [1] but 'first' wants to keep [0]
                # this doesnt raise because of 'ignore'
                removed_columns_out = _identify_idxs_to_delete(
                    _duplicates,
                    'first',
                    _do_not_drop=[1],
                    _columns=_columns if _columns_is_passed else None,
                    _conflict=_conflict,
                    _rand_idxs=_rand_idxs
                )

                assert isinstance(removed_columns_out, dict)
                assert removed_columns_out == {0:1, 3:2}

        # ** ** ** ** **


class TestIITDAccuracy(Fixtures):

    # accuracy ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    @pytest.mark.parametrize('_do_not_drop', (None, [0], [0,1], [0,2]))
    @pytest.mark.parametrize('_conflict', ('raise', 'ignore'))
    def test_no_duplicates(
        self, _columns, _keep, _do_not_drop, _conflict, _columns_is_passed
    ):

        # no duplicates, so removed_columns should be empty
        # _rand_idxs should come in empty

        removed_columns_out = _identify_idxs_to_delete(
            _duplicates=[],
            _keep=_keep,
            _do_not_drop=_do_not_drop,
            _columns=_columns if _columns_is_passed else None,
            _conflict=_conflict,
            _rand_idxs=tuple()
        )

        assert isinstance(removed_columns_out, dict)
        assert len(removed_columns_out) == 0


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    def test_do_not_drop_is_none(
        self, _duplicates, _columns, _keep, _columns_is_passed, _rand_idxs
    ):

        # do_not_drop is None, so no conflicts with keep, should always
        # return without exception, and should not muddy the water for
        # keep==random

        removed_columns_out = _identify_idxs_to_delete(
            _duplicates,
            _keep,
            _do_not_drop=None,
            _columns=_columns if _columns_is_passed else None,
            _conflict='raise',
            _rand_idxs=_rand_idxs
        )

        assert isinstance(removed_columns_out, dict)

        for k, v in removed_columns_out.items():
            assert isinstance(k, int)
            assert isinstance(v, int)


        if _keep == 'first':
            assert removed_columns_out == {1:0, 3:2}

        elif _keep == 'last':
            assert removed_columns_out == {0:1, 2:3}

        elif _keep == 'random':
            assert len(removed_columns_out) == 2
            for k, v in removed_columns_out.items():

                assert v != k

                if k == 0:
                    assert v == 1
                elif k == 1:
                    assert v == 0
                elif k == 2:
                    assert v == 3
                elif k == 3:
                    assert v == 2


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    @pytest.mark.parametrize('_do_not_drop', ([1], [1, 3], [0, 1]))
    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    def test_with_do_not_drop(
        self, _duplicates, _keep, _do_not_drop, _columns, _columns_is_passed,
        _rand_idxs
    ):

        #     def _duplicates():
        #         return [[0,1], [2,3]]
        #
        #     def _rand_idxs():
        #         return {1, 3}

        removed_columns_out = _identify_idxs_to_delete(
            _duplicates,
            _keep,
            _do_not_drop=_do_not_drop,
            _columns=_columns if _columns_is_passed else None,
            _conflict='ignore',
            _rand_idxs=_lock_in_random_idxs(
                _duplicates=_duplicates,
                _do_not_drop=_do_not_drop,
                _columns=_columns
            )
        )

        assert isinstance(removed_columns_out, dict)

        for k, v in removed_columns_out.items():
            assert isinstance(k, int)
            assert isinstance(v, int)


        if _keep == 'first':
            if _do_not_drop in ([0], [0, 2]):
                assert removed_columns_out == {1:0, 3:2}
            elif _do_not_drop == [0, 1]:
                assert removed_columns_out == {1:0, 3:2}

        elif _keep == 'last':
            if _do_not_drop == [0]:
                assert removed_columns_out == {1:0, 2:3}
            elif _do_not_drop == [0, 2]:
                assert removed_columns_out == {1: 0, 3: 2}
            elif _do_not_drop == [0, 1]:
                assert removed_columns_out == {0:1, 2:3}


        elif _keep == 'random':

            assert isinstance(removed_columns_out, dict)
            assert len(removed_columns_out) == 2

            if _do_not_drop == [0]:
                for k, v in removed_columns_out.items():
                    if k==1:
                        assert v == 0
                    elif k==2:
                        assert v == 3
                    elif k==3:
                        assert v == 2

            elif _do_not_drop == [0, 2]:
                assert removed_columns_out[1] == 0
                assert removed_columns_out[3] == 2

            elif _do_not_drop == [0, 1]:
                for k, v in removed_columns_out.items():
                    if k == 0:
                        assert v == 1
                    elif k == 1:
                        assert v == 0
                    elif k == 2:
                        assert v == 3
                    elif k == 3:
                        assert v == 2











