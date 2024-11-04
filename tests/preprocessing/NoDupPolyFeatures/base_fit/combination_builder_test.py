# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest


import numpy as np

from pybear.preprocessing.NoDupPolyFeatures._base_fit._combination_builder \
    import _combination_builder







class TestCombinationBuilder:


    # def _combination_builder(
    #     _shape: tuple[int, int],
    #     _min_degree: int,
    #     _max_degree: int,
    #     _intx_only: bool
    # ) -> list[tuple[int]]:


    @staticmethod
    @pytest.fixture()
    def _no_constants():
        return np.empty((0,), dtype=np.int32)


    @pytest.mark.parametrize('_shape',
        (-1, 1, 3.14, None, True, 'junk', [1, 2], (2, 3), {'a': 1}, lambda x: x)
    )
    def test_shape_validation(self, _shape, _no_constants):

        if isinstance(_shape, (tuple, list)):

            _combination_builder(
                _shape=_shape,
                _constants=_no_constants,
                _min_degree=1,
                _max_degree=2,
                _intx_only=True
            )

        else:
            with pytest.raises(AssertionError):
                _combination_builder(
                    _shape=_shape,
                    _constants=_no_constants,
                    _min_degree=1,
                    _max_degree=2,
                    _intx_only=True
                )


    @pytest.mark.parametrize('_constants',
        (
                1, None, True, 'junk', [1, 2],
                np.random.randint(0,2,(5,)).astype(np.int32), {'a': 1}
        )
    )
    def test_constants_validation(self, _constants):

        if isinstance(_constants, np.ndarray):

            _combination_builder(
                _shape=(5,3),
                _constants=_constants,
                _min_degree=1,
                _max_degree=2,
                _intx_only=True
            )

        else:

            with pytest.raises(AssertionError):
                _combination_builder(
                    _shape=(5,3),
                    _constants=_constants,
                    _min_degree=1,
                    _max_degree=2,
                    _intx_only=True
                )


    @pytest.mark.parametrize('_min_degree',
        (-1, 0, 1, 3.14, None, True, 'junk', [1, 2], (2, 3), {'a': 1}, lambda x: x)
    )
    def test_min_degree_validation(self, _min_degree, _no_constants):

        if isinstance(_min_degree, int) and not isinstance(_min_degree, bool) and \
            _min_degree >= 0:

            _combination_builder(
                _shape=(5,3),
                _constants=_no_constants,
                _min_degree=_min_degree,
                _max_degree=4,
                _intx_only=False
            )

        else:
            with pytest.raises(AssertionError):
                _combination_builder(
                    _shape=(5, 3),
                    _constants=_no_constants,
                    _min_degree=_min_degree,
                    _max_degree=4,
                    _intx_only=False
                )


    @pytest.mark.parametrize('_max_degree',
        (-1, 1, 3.14, None, True, 'junk', [1, 2], (2, 3), {'a': 1}, lambda x: x)
    )
    def test_max_degree_validation(self, _max_degree, _no_constants):

        if isinstance(_max_degree, int) and not isinstance(_max_degree, bool) and \
            _max_degree > 0:

            _combination_builder(
                _shape=(20,10),
                _constants=_no_constants,
                _min_degree=0,
                _max_degree=_max_degree,
                _intx_only=True
            )

        else:
            with pytest.raises(AssertionError):
                _combination_builder(
                    _shape=(20, 10),
                    _constants=_no_constants,
                    _min_degree=0,
                    _max_degree=_max_degree,
                    _intx_only=True
                )



    @pytest.mark.parametrize('_intx_only',
        (-1, 1, 3.14, None, True, 'junk', [1, 2], (2, 3), {'a': 1}, lambda x: x)
    )
    def test_intx_only_validation(self, _intx_only, _no_constants):

        if isinstance(_intx_only, bool):

            _combination_builder(
                _shape=(5,3),
                _constants=_no_constants,
                _min_degree=0,
                _max_degree=3,
                _intx_only=_intx_only
            )

        else:
            with pytest.raises(AssertionError):
                _combination_builder(
                    _shape=(5, 3),
                    _constants=_no_constants,
                    _min_degree=0,
                    _max_degree=3,
                    _intx_only=_intx_only
                )


    @pytest.mark.parametrize('_intx_only', (True, False))
    def test_blocks_zero_zero(self, _intx_only, _no_constants):
        with pytest.raises(AssertionError):
            _combination_builder(
                _shape=(5, 3),
                _constants=_no_constants,
                _min_degree=0,
                _max_degree=0,
                _intx_only=_intx_only
            )


    @pytest.mark.parametrize('_intx_only', (True, False))
    def test_bumps_min_degree_zero_to_one(self, _intx_only, _no_constants):

        # degree==0 is dealt with separately. if 0, _combination_builder
        # bumps it up to one. if max_degree==0, then shouldnt even be
        # getting into _combination_builder

        out_zero = _combination_builder(
            _shape=(5, 3),
            _constants=_no_constants,
            _min_degree=0,
            _max_degree=2,
            _intx_only=_intx_only
        )

        out_one = _combination_builder(
            _shape=(5, 3),
            _constants=_no_constants,
            _min_degree=0,
            _max_degree=2,
            _intx_only=_intx_only
        )

        assert list(out_zero) == list(out_one)


    @pytest.mark.parametrize('_min_degree', (1, 2))
    @pytest.mark.parametrize('_max_degree', (2, 3))
    @pytest.mark.parametrize('_intx_only', (True, False))
    @pytest.mark.parametrize('_n_features', (2, 3))
    def test_accuracy_no_constants(
        self, _min_degree, _max_degree, _intx_only, _n_features, _no_constants
    ):

        # if min_degree comes in as a zero, it is bumped up to 1

        out = _combination_builder(
            _shape=(20, _n_features),
            _constants=_no_constants,
            _min_degree=_min_degree,
            _max_degree=_max_degree,
            _intx_only=_intx_only
        )

        if _n_features == 2:

            if _min_degree == 1:
                if _max_degree == 2:
                    if _intx_only:
                        assert list(out) == [(0,), (1,), (0, 1)]
                    elif not _intx_only:
                        assert list(out) == \
                               [(0,), (1,), (0, 0), (0, 1), (1, 1)]


                elif _max_degree == 3:
                    if _intx_only:
                        assert list(out) == [(0,), (1,), (0, 1)]

                    elif not _intx_only:
                        assert list(out) == \
                               [
                                    (0,), (1,), (0, 0), (0, 1), (1, 1),
                                    (0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)
                                ]

                else:
                    raise Exception

            elif _min_degree == 2:
                if _max_degree == 2:
                    if _intx_only:
                        assert list(out) == [(0, 1)]
                    elif not _intx_only:
                        assert list(out) == [(0, 0), (0, 1), (1, 1)]
                elif _max_degree == 3:
                    if _intx_only:
                        assert list(out) == [(0, 1)]
                    elif not _intx_only:
                        assert list(out) == \
                               [
                                (0, 0), (0, 1), (1, 1), (0, 0, 0),
                                (0, 0, 1), (0, 1, 1), (1, 1, 1)
                               ]

                else:
                    raise Exception

            else:
                raise Exception

        elif _n_features == 3:

            if _min_degree == 1:
                if _max_degree == 2:
                    if _intx_only:
                        assert list(out) == \
                               [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
                    elif not _intx_only:
                        assert list(out) == \
                               [
                                   (0,), (1,), (2,), (0, 0), (0, 1),
                                   (0, 2), (1, 1), (1, 2), (2, 2)
                               ]
                elif _max_degree == 3:
                    if _intx_only:
                        assert list(out) == \
                               [
                                   (0,), (1,), (2,), (0, 1),
                                   (0, 2), (1, 2), (0, 1, 2)
                               ]
                    elif not _intx_only:
                        assert list(out) == \
                               [
                                   (0,), (1,), (2,), (0, 0), (0, 1), (0, 2),
                                   (1, 1), (1, 2), (2, 2), (0, 0, 0), (0, 0, 1),
                                   (0, 0, 2), (0, 1, 1), (0, 1, 2), (0, 2, 2),
                                   (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2)
                               ]
                else:
                    raise Exception

            elif _min_degree == 2:
                if _max_degree == 2:
                    if _intx_only:
                        assert list(out) == [(0, 1), (0, 2), (1, 2)]
                    elif not _intx_only:
                        assert list(out) == \
                               [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
                elif _max_degree == 3:
                    if _intx_only:
                        assert list(out) == [(0, 1), (0, 2), (1, 2), (0, 1, 2)]
                    elif not _intx_only:
                        assert list(out) == \
                           [
                               (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2),
                               (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1),
                               (0, 1, 2), (0, 2, 2), (1, 1, 1), (1, 1, 2),
                               (1, 2, 2), (2, 2, 2)
                           ]
                else:
                    raise Exception

            else:
                raise Exception

        else:
            raise Exception


    @pytest.mark.parametrize('_intx_only', (True, False))
    def test_accuracy_with_constants(self, _intx_only):

        # if min_degree comes in as a zero, it is bumped up to 1

        out = _combination_builder(
            _shape=(20, 5),
            _constants=np.array([0,2], dtype=np.int32),
            _min_degree=2,
            _max_degree=3,
            _intx_only=_intx_only
        )

        if _intx_only:
            assert list(out) == [(1, 3), (1, 4), (3, 4), (1, 3, 4)]
        elif not _intx_only:
            assert list(out) == \
               [
                    (1, 1), (1, 3), (1, 4), (3, 3), (3, 4), (4, 4),
                    (1, 1, 1), (1, 1, 3), (1, 1, 4), (1, 3, 3), (1, 3, 4),
                    (1, 4, 4), (3, 3, 3), (3, 3, 4), (3, 4, 4), (4, 4, 4)
               ]



