# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.SlimPolyFeatures._partial_fit. \
    _lock_in_random_combos import _lock_in_random_combos

import itertools

import pytest



# def _lock_in_random_combos(
#     _poly_duplicates: list[list[tuple[int]]],
#     _combinations: list[tuple[int]]
# ) -> tuple[tuple[int]]:







class Fixtures:

    @staticmethod
    @pytest.fixture()
    def _cols():
        return 6


    @staticmethod
    @pytest.fixture()
    def _duplicates():
        return [[(2,3),(4,5)]]


    @staticmethod
    @pytest.fixture()
    def _combos(_cols):
        return list(itertools.combinations(range(_cols), 2))



class TestLIRIValidation(Fixtures):

    # test validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # ------------------------------
    @pytest.mark.parametrize('junk_duplicates',
        (-1,0,1,3.14,None,True,False,'trash',{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_duplicates(self, junk_duplicates, _combos):

        with pytest.raises(AssertionError):
            _lock_in_random_combos(
                junk_duplicates,
                _combos
            )


    @pytest.mark.parametrize('bad_duplicates',
        (
            [[0,1], [2,3]], [[('a',),('b',)], [('c',),('d',)]],
            [[(1,)]], [[(2,2)],[(2,2)]]
         )
    )
    def test_rejects_bad_duplicates(self, bad_duplicates, _combos):

        # not tuples
        # not ints
        # only one int in tuple
        # repeated tuple

        with pytest.raises(AssertionError):
            _lock_in_random_combos(
                bad_duplicates,
                _combos,
            )


    def test_accepts_good_duplicates(self, _duplicates, _combos):

        out = _lock_in_random_combos(
            _duplicates,
            _combos
        )

        assert isinstance(out, tuple)
        assert all(map(isinstance, out, (tuple for _ in out)))

    # ------------------------------


    # ------------------------------
    @pytest.mark.parametrize('junk_combos',
        (-1,0,1,3.14,True,False,'trash',{'a':1}, min, lambda x: x)
    )
    def test_rejects_junk_combos(self, _duplicates, junk_combos):

        with pytest.raises(AssertionError):
            _lock_in_random_combos(
                _duplicates,
                junk_combos
            )


    @pytest.mark.parametrize('bad_combos',
        ([0,1],(0,1),(0,), [0,1,2,3,4], [True, False])
    )
    def test_rejects_bad_combos(self, _duplicates, bad_combos):

        with pytest.raises(AssertionError):
            _lock_in_random_combos(
                _duplicates,
                bad_combos
            )


    def test_accepts_good_combos(self):

        out = _lock_in_random_combos(
            _poly_duplicates=[[(0,1), (0,2)], [(1,2),(1,3)]],
            _combinations=[(0,1), (1,2), (0,2), (0,3), (1,3), (2,3)]
        )

        assert isinstance(out, tuple)
        assert all(map(isinstance, out, (tuple for _ in out)))

    # ------------------------------

    # END test validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **



class TestLIRIAccuracy(Fixtures):

    # accuracy ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    def test_no_duplicates(self, _combos):

        # no duplicates, so _rand_idxs should be empty

        rand_idxs_out = _lock_in_random_combos(
            _poly_duplicates=[],
            _combinations=_combos
        )

        assert isinstance(rand_idxs_out, tuple)
        assert rand_idxs_out == tuple()



    @pytest.mark.parametrize('_poly_duplicates',
        ([[(0,1), (1, 4)]], [[(0, 1), (0,2)], [(2,4), (3,4)]])
    )
    def test_accuracy(self, _poly_duplicates, _combos):

        rand_idxs_out = _lock_in_random_combos(
            _poly_duplicates,
            _combinations=_combos
        )

        assert isinstance(rand_idxs_out, tuple)

        # all we can validate is that that position in rand_idxs_out
        # contains one of the dupl combinations
        for _idx, _set in enumerate(_poly_duplicates):
            # kept would be randomly selected from _set
            assert list(rand_idxs_out)[_idx] in _set







