# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._transform. \
    _NDArrayify_integerize_ic_hab import _NDArrayify_integerize_ic_hab

import numpy as np

import pytest



class TestNDArrayify_integerize_ic_hab:

    # use Exception instead of specific errors. exceptions could be raised in
    # _val_ign_cols_hab_callable, feature_name_mapper

    @pytest.mark.parametrize('junk_value',
        (-2.7, -1, 0, 2.7, True, False, 'trash', (True, False), {'a': 1})
    )
    def test_rejects_junk(self, junk_value):

        # X must have copy() method
        # _ignore_columns must be None, callable, Iterable[str], Iterable[int]
        # _handle_as_bool must be None, callable, Iterable[str], Iterable[int]
        # _feature_names_in: Union[npt.NDArray[str], None]

        # not hasattr(X, 'copy') only raises if ic or hab is a callable
        # this wont raise for bad X
        _NDArrayify_integerize_ic_hab(
            junk_value,
            [0, 1],
            [2, 3],
            4,
            None
        )


        with pytest.raises(Exception):
            _NDArrayify_integerize_ic_hab(
                np.random.randint(0, 10, (3, 3)),
                junk_value,
                [1, 2],
                3,
                None
            )

        with pytest.raises(Exception):
            _NDArrayify_integerize_ic_hab(
                np.random.randint(0, 10, (3, 3)),
                [0, 1],
                junk_value,
                3,
                None
            )


    @pytest.mark.parametrize('ic', ([0, 1], lambda x: [0, 1]))
    @pytest.mark.parametrize('hab', ([0, 1], lambda x: [0, 1]))
    def test_rejects_junk_X_if_ic_or_hab_callable(self, ic, hab):

        # not hasattr(X, 'copy') only raises if ic or hab is a callable
        # this wont raise for bad X
        if callable(ic) or callable(hab):
            with pytest.raises(Exception):
                _NDArrayify_integerize_ic_hab(
                    (1, 2, 3, 4, 5),
                    ic,
                    hab,
                    10,
                    None
                )
        else:
            _NDArrayify_integerize_ic_hab(
                (1, 2, 3, 4, 5),
                ic,
                hab,
                10,
                None
            )


    @pytest.mark.parametrize('ic', ('good', 'bad'))
    @pytest.mark.parametrize('hab', ('good', 'bad'))
    def test_catches_callable_exception(self, ic, hab):

        will_raise = False
        if ic == 'bad' or hab == 'bad':
            will_raise = True

        if ic == 'good':
            ic = lambda X: [0, 1]
        elif ic == 'bad':
            ic = lambda x: X.shape[2]
        else:
            raise Exception

        if hab == 'good':
            hab = lambda X: [0, 1]
        elif hab == 'bad':
            hab = lambda X: X.shape[2]
        else:
            raise Exception

        if will_raise:
            with pytest.raises(Exception):
                _NDArrayify_integerize_ic_hab(
                    np.random.uniform(0, 1, (5, 5)),
                    ic,
                    hab,
                    5,
                    None
                )
        else:
            _NDArrayify_integerize_ic_hab(
                np.random.uniform(0, 1, (5, 5)),
                ic,
                hab,
                5,
                None
            )


    @pytest.mark.parametrize('good_ic',
        (
            None, lambda X: [0, 2], lambda X: ['a', 'b'], [], [1, 0], (0, 2),
            np.array([1, 2])
        )
    )
    @pytest.mark.parametrize('good_hab',
        (
            None, lambda X: [0, 2], lambda X: ['a', 'b'], [], [1, 0], (0, 2),
            np.array([1, 2])
        )
    )
    @pytest.mark.parametrize('fni', (None, np.array(list('abc'))))
    def test_accept_good(self, good_ic, good_hab, fni):

        X = np.random.uniform(0, 1, (10, 3))

        # cant have str ic/hab when fni is None
        will_fail = False
        if fni is None:
            # ic callable returns vector of str
            if callable(good_ic):
                if isinstance(good_ic(X)[0], str):
                    will_fail = True
            elif good_ic is None:
                pass
            elif len(good_ic) == 0:
                pass
            # ic passed as vector of str
            elif all(map(isinstance, good_ic, (str for _ in good_ic))):
                will_fail = True

            # hab callable returns vector of str
            if callable(good_hab):
                if isinstance(good_hab(X)[0], str):
                    will_fail = True
            elif good_hab is None:
                pass
            elif len(good_hab) == 0:
                pass
            # hab passed as vector str
            elif all(map(isinstance, good_hab, (str for _ in good_hab))):
                will_fail = True

        if will_fail:
            with pytest.raises(Exception):
                _NDArrayify_integerize_ic_hab(
                    X,
                    good_ic,
                    good_hab,
                    3,
                    fni
                )
            pytest.skip(reason=f"cant do anymore tests after except")
        else:

            out = _NDArrayify_integerize_ic_hab(
                X,
                good_ic,
                good_hab,
                3,
                fni
            )


            if good_ic is None:
                assert np.array_equal(out[0], [])
                assert out[0].dtype == np.int32
            elif callable(good_ic) and isinstance(good_ic(X)[0], str):
                if fni is None:
                    # should have failed & skipped above
                    raise Exception
                else:
                    assert np.array_equal(out[0], [0, 1])
                    assert out[0].dtype == np.int32
            elif np.array_equal(good_ic, [0, 2]):
                assert np.array_equal(out[0], [0, 2])
                assert out[0].dtype == np.int32
            elif callable(good_ic) and isinstance(good_ic(X)[0], int):
                assert np.array_equal(out[0], [0, 2])
                assert out[0].dtype == np.int32
            elif np.array_equal(good_ic, ['a', 'b']):
                if fni is None:
                    # should have failed & skipped above
                    raise Exception
                else:
                    assert np.array_equal(out[0], [0, 1])
                    assert out[0].dtype == np.int32
            elif len(good_ic) == 0:
                assert len(out[0]) == 0
                assert isinstance(out[0], np.ndarray)
                assert out[0].dtype == np.int32
            elif np.array_equal(good_ic, [1, 0]):
                assert np.array_equal(out[0], [1, 0])
                assert out[0].dtype == np.int32
            elif np.array_equal(good_ic, (0, 2)):
                assert np.array_equal(out[0], [0, 2])
                assert out[0].dtype == np.int32
            elif np.array_equal(good_ic, np.array([1, 2])):
                assert np.array_equal(out[0], [1, 2])
                assert out[0].dtype == np.int32
            else:
                raise Exception


            if good_hab is None:
                assert np.array_equal(out[1], [])
                assert out[1].dtype == np.int32
            elif callable(good_hab) and isinstance(good_hab(X)[0], str):
                if fni is None:
                    # should have failed & skipped above
                    raise Exception
                else:
                    assert np.array_equal(out[1], [0, 1])
                    assert out[1].dtype == np.int32
            elif np.array_equal(good_hab, [0, 2]):
                assert np.array_equal(out[1], [0, 2])
                assert out[1].dtype == np.int32
            elif callable(good_hab) and isinstance(good_hab(X)[0], int):
                assert np.array_equal(out[1], [0, 2])
                assert out[1].dtype == np.int32
            elif np.array_equal(good_hab, ['a', 'b']):
                if fni is None:
                    # should have failed & skipped above
                    raise Exception
                else:
                    assert np.array_equal(out[1], [0, 1])
                    assert out[1].dtype == np.int32
            elif len(good_hab) == 0:
                assert len(out[1]) == 0
                assert isinstance(out[1], np.ndarray)
                assert out[1].dtype == np.int32
            elif np.array_equal(good_hab, [1, 0]):
                assert np.array_equal(out[1], [1, 0])
                assert out[1].dtype == np.int32
            elif np.array_equal(good_hab, (0, 2)):
                assert np.array_equal(out[1], [0, 2])
                assert out[1].dtype == np.int32
            elif np.array_equal(good_hab, np.array([1, 2])):
                assert np.array_equal(out[1], [1, 2])
                assert out[1].dtype == np.int32
            else:
                raise Exception





