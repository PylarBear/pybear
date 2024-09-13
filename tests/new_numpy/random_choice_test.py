# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


import pytest

from numpy import (
    arange, array, int32, max as _max, min as _min, pi, unique
)
from pybear.new_numpy._random_ import choice as pb_choice




@pytest.fixture
def good_a():
    return arange(int(1e5), dtype=int32)


@pytest.fixture
def good_str_a():
    return array(list('abcdefghijlmnop'), dtype=object)


@pytest.fixture
def bad_a_empty():
    return array([])

@pytest.fixture
def bad_a_2D():
    return arange(int(1e5), dtype=int32).reshape((-1, 100))


@pytest.fixture
def bad_a_3D():
    return arange(int(1e5), dtype=int32).reshape((-1, 100, 100))




class TestA:

    @pytest.mark.parametrize(
        'a',
        ({'a': 1, 'b': 2}, 'junk string', pi, 3, None, False)
    )
    def test_non_array_like(self, a):
        with pytest.raises(TypeError):
            pb_choice(a, (100,), replace=True)

    def test_bad_shape_empty(self, bad_a_empty):
        with pytest.raises(ValueError):
            pb_choice(bad_a_empty, (0,), replace=True)

    def test_bad_shape_2D(self, bad_a_2D):
        with pytest.raises(ValueError):
            pb_choice(bad_a_2D, (100,), replace=True)

    def test_bad_shape_3D(self, bad_a_3D):
        with pytest.raises(ValueError):
            pb_choice(bad_a_3D, (100,), replace=True)

    def test_good_shape(self, good_a):
        assert pb_choice(good_a, (100,), replace=True).shape == (100,)


class TestShape:

    @pytest.mark.parametrize('shape', (pi, {'a':1, 'b':2}, None, False))
    def reject_not_int_or_tuple(self, shape, good_a):
        with pytest.raises(TypeError):
            pb_choice(good_a, shape, replace=True)

    @pytest.mark.parametrize('shape', ((100,), 100))
    def accepts_int_or_tuple(self, shape, good_a):
        try:
            len(shape)
            assert pb_choice(good_a, shape, replace=True).shape == shape
        except:
            assert pb_choice(good_a, shape, replace=True).shape == (shape,)


class TestReplace:

    @pytest.mark.parametrize('replace', ('q', pi, 3, []))
    def test_rejects_non_bool(self, replace, good_a):
        with pytest.raises(TypeError):
            pb_choice(good_a, (100,), replace=replace)

    @pytest.mark.parametrize('replace', (True, False))
    def test_accepts_bool(self, replace, good_a):
        pb_choice(good_a, (100,), replace=replace)


class TestNJobs:
    def test_accepts_int(self, good_a):
        pb_choice(good_a, (100,), replace=True, n_jobs=1)

    def test_accepts_none(self, good_a):
        pb_choice(good_a, (100,), replace=True, n_jobs=None)

    def test_rejects_float(self, good_a):
        with pytest.raises(ValueError):
            pb_choice(good_a, (100,), replace=True, n_jobs=pi)

    @pytest.mark.parametrize('n_jobs', (0, -2, 9235))
    def test_rejects_bad_ints(self, good_a, n_jobs):
        with pytest.raises(ValueError):
            pb_choice(good_a, (100,), replace=True, n_jobs=n_jobs)

    @pytest.mark.parametrize('n_jobs', ('junk', [], {'a':1}))
    def test_rejects_non_numerics(self, good_a, n_jobs):
        with pytest.raises(TypeError):
            pb_choice(good_a, (100,), replace=True, n_jobs=n_jobs)



def reject_pick_too_big_when_replace_equals_false():
    with pytest.raises(ValueError):
        pb_choice([1,2,3], (10,), replace=False)



@pytest.mark.parametrize('shape', ((100, 100), (2, 5000), (10000,)))
@pytest.mark.parametrize('replace', (True, False))
def test_accuracy_num(good_a, shape, replace):
    PULL = pb_choice(good_a, shape, replace=replace)

    assert PULL.shape == shape

    assert _max(PULL) <= _max(good_a)

    assert _min(PULL) >= _min(good_a)

    if not replace:
        assert _max(unique(PULL, return_counts=True)[1]) == 1



@pytest.mark.parametrize('shape', ((3, 3), (2, 5), (9,)))
@pytest.mark.parametrize('replace', (True, False))
def test_accuracy_str(good_str_a, shape, replace):
    PULL = pb_choice(good_str_a, shape, replace=replace)

    assert PULL.shape == shape

    if not replace:
        assert _max(unique(PULL, return_counts=True)[1]) == 1

    for item in PULL.ravel():
        assert item in good_str_a

















