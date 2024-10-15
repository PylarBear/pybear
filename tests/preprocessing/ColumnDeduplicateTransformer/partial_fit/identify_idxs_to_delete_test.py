# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _identify_idxs_to_delete import _identify_idxs_to_delete

import pytest




# def _identify_idxs_to_delete(
#     _duplicates: list[list[int]],
#     _keep: KeepType,   # Literal['first', 'last', 'random']
#     _do_not_drop: Union[Iterable[int], Iterable[str], None],
#     _columns: ColumnsType,   # Union[Iterable[str], None]
#     _conflict: ConflictType   # Literal['raise', 'ignore']
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
    @pytest.mark.parametrize('junk_duplicates',
        (-1,0,1,3.14,None,True,False,[0,1],(0,1),(0,),{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_duplicates(
        self, junk_duplicates, _keep, _do_not_drop, _columns, _conflict
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                junk_duplicates,
                _keep,
                _do_not_drop,
                _columns,
                _conflict
            )


    @pytest.mark.parametrize('bad_duplicates',
        ([['a','b'], ['c','d']], [[2,2],[2,2]])
    )
    def test_rejects_bad_duplicates(
        self, bad_duplicates, _keep, _do_not_drop, _columns, _conflict
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                bad_duplicates,
                _keep,
                _do_not_drop,
                _columns,
                _conflict
            )


    @pytest.mark.parametrize('junk_keep',
        (-1,0,1,3.14,None,True,False,[0,1],(0,),{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_keep(
        self, _duplicates, junk_keep, _do_not_drop, _columns, _conflict
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                junk_keep,
                _do_not_drop,
                _columns,
                _conflict
            )


    @pytest.mark.parametrize('bad_keep', ('trash', 'junk', 'garbage'))
    def test_rejects_bad_keep(
        self, _duplicates, bad_keep, _do_not_drop, _columns, _conflict
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                bad_keep,
                _do_not_drop,
                _columns,
                _conflict
            )


    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    def test_accepts_good_keep(
        self, _duplicates, _keep, _do_not_drop, _columns, _conflict
    ):
        _identify_idxs_to_delete(
            _duplicates,
            _keep,
            _do_not_drop,
            _columns,
            _conflict
        )


    @pytest.mark.parametrize('junk_do_not_drop',
        (-1,0,1,3.14,True,False,{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_do_not_drop(
        self, _duplicates, _keep, junk_do_not_drop, _columns, _conflict
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                _keep,
                junk_do_not_drop,
                _columns,
                _conflict
            )


    @pytest.mark.parametrize('bad_do_not_drop',
        ([min, max], [True, False], [[], []])
)
    def test_rejects_bad_do_not_drop(
        self, _duplicates, _keep, bad_do_not_drop, _columns, _conflict
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                _keep,
                bad_do_not_drop,
                _columns,
                _conflict
            )


    @pytest.mark.parametrize('_do_not_drop',
        ([0,1,2], 'str', None))
    def test_accepts_good_do_not_drop(
        self, _duplicates, _keep, _do_not_drop, _columns, _conflict
    ):
        if _do_not_drop == 'str':
            _do_not_drop = [_columns[0], _columns[1], _columns[-1]]

        _identify_idxs_to_delete(
            _duplicates,
            _keep,
            _do_not_drop,
            _columns,
            _conflict
        )


    @pytest.mark.parametrize('junk_columns',
        (-1,0,1,3.14,True,False,[0,1],(0,1),(0,),{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_columns(
        self, _duplicates, _keep, _do_not_drop, junk_columns, _conflict
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                _keep,
                _do_not_drop,
                junk_columns,
                _conflict
            )


    @pytest.mark.parametrize('bad_columns', ([0,1,2,3,4], [True, False]))
    def test_rejects_bad_columns(
        self, _duplicates, _keep, _do_not_drop, bad_columns, _conflict
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                _keep,
                _do_not_drop,
                bad_columns,
                _conflict
            )


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    def test_accepts_good_columns(
        self, _duplicates, _keep, _do_not_drop, _columns, _conflict,
            _columns_is_passed
    ):
        _identify_idxs_to_delete(
            _duplicates,
            _keep,
            _do_not_drop,
            _columns if _columns_is_passed else None,
            _conflict
        )


    @pytest.mark.parametrize('junk_conflict',
        (-1,0,1,3.14,None,True,False,[0,1],(0,),{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_conflict(
        self, _duplicates, _keep, _do_not_drop, _columns, junk_conflict
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                _keep,
                _do_not_drop,
                _columns,
                junk_conflict
            )


    @pytest.mark.parametrize('bad_conflict', ('junk', 'trash', 'garbage'))
    def test_rejects_bad_conflict(
        self, _duplicates, _keep, _do_not_drop, _columns,
        bad_conflict
    ):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates,
                _keep,
                _do_not_drop,
                _columns,
                bad_conflict
            )


    @pytest.mark.parametrize('_conflict', ('raise', 'ignore'))
    def test_accepts_good_conflict(self, _duplicates, _columns, _conflict):
        _identify_idxs_to_delete(
            _duplicates,
            'first',
            [0, 2],
            _columns,
            _conflict
        )


    def test_reject_str_do_not_drop_if_no_columns(self, _duplicates, _columns):
        with pytest.raises(AssertionError):
            _identify_idxs_to_delete(
                _duplicates=_duplicates,
                _keep='first',
                _do_not_drop=[_columns[0], _columns[-1]],
                _columns=None,
                _conflict='raise'
            )


    # END test validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


class TestIITDConflict(Fixtures):

    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    def test_conflict(
        self, _duplicates, _do_not_drop, _columns, _columns_is_passed
    ):

        with pytest.raises(ValueError):
            _identify_idxs_to_delete(
                _duplicates,
                _keep='first',
                _do_not_drop=_do_not_drop,
                _columns=_columns if _columns_is_passed else None,
                _conflict='raise'
            )

        # ** ** ** ** **
        with pytest.raises(ValueError):
            _identify_idxs_to_delete(
                _duplicates,
                'first',
                [1],
                _columns=_columns if _columns_is_passed else None,
                _conflict='raise'   # <=======
            )

        out = _identify_idxs_to_delete(
            _duplicates,
            'first',
            [1],
            _columns=_columns if _columns_is_passed else None,
            _conflict='ignore'   # <=======
        )

        assert isinstance(out, dict)
        assert out == {0:1, 3:2}

        # ** ** ** ** **

        with pytest.raises(ValueError):
            _identify_idxs_to_delete(
                _duplicates,
                'first',
                [0, 1],
                _columns=_columns if _columns_is_passed else None,
                _conflict='raise'   # <=======
            )

        out = _identify_idxs_to_delete(
                _duplicates,
                'first',
                [0, 1],
                _columns=_columns if _columns_is_passed else None,
                _conflict='ignore'   # <=======
            )

        assert isinstance(out, dict)
        assert out == {1:0, 3:2}


class TestIITDAccuracy(Fixtures):

    # accuracy ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    @pytest.mark.parametrize('_do_not_drop', (None, [0], [0,1], [0,2]))
    @pytest.mark.parametrize('_conflict', ('raise', 'ignore'))
    def test_no_duplicates(
        self, _columns, _keep, _do_not_drop, _conflict, _columns_is_passed
    ):

        out = _identify_idxs_to_delete(
            _duplicates=[],
            _keep=_keep,
            _do_not_drop=_do_not_drop,
            _columns=_columns if _columns_is_passed else None,
            _conflict=_conflict
        )

        assert isinstance(out, dict)
        assert len(out) == 0


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    def test_do_not_drop_is_none(
        self, _duplicates, _columns, _keep, _columns_is_passed
    ):

        out = _identify_idxs_to_delete(
            _duplicates,
            _keep,
            _do_not_drop=None,
            _columns=_columns if _columns_is_passed else None,
            _conflict='raise'
        )

        assert isinstance(out, dict)

        for k, v in out.items():
            assert isinstance(k, int)
            assert isinstance(v, int)


        if _keep == 'first':
            assert out == {1:0, 3:2}

        elif _keep == 'last':
            assert out == {0:1, 2:3}

        elif _keep == 'random':
            assert len(out) == 2
            for k, v in out.items():

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
    @pytest.mark.parametrize('_do_not_drop', ([0], [0, 2], [0, 1]))
    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    def test_with_do_not_drop(
        self, _duplicates, _keep, _do_not_drop, _columns, _columns_is_passed
    ):

        out = _identify_idxs_to_delete(
            _duplicates,
            _keep,
            _do_not_drop=_do_not_drop,
            _columns=_columns if _columns_is_passed else None,
            _conflict='ignore'
        )

        assert isinstance(out, dict)

        for k, v in out.items():
            assert isinstance(k, int)
            assert isinstance(v, int)


        if _keep == 'first':
            if _do_not_drop in ([0], [0, 2]):
                assert out == {1:0, 3:2}
            elif _do_not_drop == [0, 1]:
                assert out == {1:0, 3:2}

        elif _keep == 'last':
            if _do_not_drop == [0]:
                assert out == {1:0, 2:3}
            elif _do_not_drop == [0, 2]:
                assert out == {1: 0, 3: 2}
            elif _do_not_drop == [0, 1]:
                assert out == {0:1, 2:3}


        elif _keep == 'random':

            assert isinstance(out, dict)
            assert len(out) == 2

            if _do_not_drop == [0]:
                for k, v in out.items():
                    if k==1:
                        assert v == 0
                    elif k==2:
                        assert v == 3
                    elif k==3:
                        assert v == 2

            elif _do_not_drop == [0, 2]:
                assert out[1] == 0
                assert out[3] == 2

            elif _do_not_drop == [0, 1]:
                for k, v in out.items():
                    if k == 0:
                        assert v == 1
                    elif k == 1:
                        assert v == 0
                    elif k == 2:
                        assert v == 3
                    elif k == 3:
                        assert v == 2











