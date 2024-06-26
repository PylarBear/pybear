# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


import pytest
from pybear.data_validation import arg_kwarg_validater as akv
import numpy as np


@pytest.fixture
def str_allowables():
    return ['a', 'b', 'c']

@pytest.fixture
def int_allowables():
    return [1, 2, 3]

@pytest.fixture
def float_allowables():
    return [1.1, 2.2, 3.3]

@pytest.fixture
def mixed_allowables():
    return ['a', 'b', 'c', 1, 2, 3, False, None]

@pytest.fixture
def bool_allowables():
    return [True, False]

@pytest.fixture
def gn():
    return 'good_name'

@pytest.fixture
def gmn():
    return 'good_module'

@pytest.fixture
def gfn():
    return 'good_function'


# VALIDATION ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
class TestStringInputs:

    def test_accepts_good_names(self, gn, int_allowables, gmn, gfn):
        arg = 1
        out = akv(arg, gn, int_allowables, gmn, gfn, None)
        assert out == arg

    @pytest.mark.parametrize('name', (1, None, [], np.pi, ['a']))
    def test_rejects_bad_name(self, name, int_allowables, gmn, gfn):
        with pytest.raises(TypeError):
            akv(1, name, int_allowables, gmn, gfn, None)


    @pytest.mark.parametrize('module_name', (1, None, [], np.pi, ['a']))
    def test_rejects_bad_module_name(self, gn, int_allowables, module_name, gfn):
        with pytest.raises(TypeError):
            akv(1, gn, int_allowables, module_name, gfn, None)


    @pytest.mark.parametrize('function_name', (1, None, [], np.pi, ['a']))
    def test_rejects_bad_function_name(self, gn, int_allowables, gmn, function_name):
        with pytest.raises(TypeError):
            akv(1, gn, int_allowables, gmn, function_name, None)


class TestArg:

    def test_rejects_dicts(self, gn, int_allowables, gmn, gfn):
        with pytest.raises(TypeError):
            akv({'a': 1}, gn, int_allowables, gmn, gfn)


    @pytest.mark.parametrize('non_iterable',
        (1, 2, 3, None, False, 'a', 'b', 'c')
    )
    def test_accepts_non_iter(self, gn, mixed_allowables, gmn,
        gfn, non_iterable):
        out = akv(non_iterable, gn, mixed_allowables, gmn, gfn)
        assert out == non_iterable


    @pytest.mark.parametrize('iterables',
        (['a','b',1,2], ('a','b',1,2), {'a','b',1,2})
    )
    def test_accepts_iter_of_non_iter(self,
            iterables, gn, mixed_allowables, gmn, gfn
        ):
        out = akv(iterables, gn, mixed_allowables, gmn, gfn)
        assert np.array_equiv(out, np.array(list(iterables), dtype=object))



class TestAllowed:

    @pytest.mark.parametrize('allowed', (1, None, False, np.pi, {'a':1}, 'junk'))
    def test_allowed_rejects_non_array_like(self, gn, allowed, gmn, gfn):
        with pytest.raises(TypeError):
            akv(1, gn, allowed, gmn, gfn, None)


    @pytest.mark.parametrize('allowed',
        ([1,2,3], (1,2,3), {1,2,3}, np.array([1,2,3]))
    )
    def test_allowed_accepts_array_like(self, gn, allowed, gmn, gfn):
        arg = 1
        out = akv(arg, gn, allowed, gmn, gfn, None)
        assert out == arg

    @pytest.mark.parametrize('allowed', ([], (), np.array([])))
    def test_allowed_rejects_empty_array(self, gn, allowed, gmn, gfn):
        with pytest.raises(ValueError):
            akv(1, gn, allowed, gmn, gfn, None)


@pytest.mark.parametrize('return_if_none', (1, None, [], np.pi, ['a'], {'a':1}))
def test_return_if_none_accepts_anything(return_if_none, gn, int_allowables, gmn, gfn):
    arg = 1
    out = akv(arg, gn, int_allowables, gmn, gfn, return_if_none)
    assert out == arg


# END VALIDATION ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

# ACCURACY ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


class TestInts:

    @pytest.mark.parametrize('arg', (1,2,3))
    def test_accepts_allowed_ints(self, arg, gn, int_allowables, gmn, gfn):
        out = akv(arg, gn, int_allowables, gmn, gfn, None)
        assert out == arg

    @pytest.mark.parametrize('arg', (4, None, 'junk', np.pi))
    def test_rejects_not_allowed_ints(self, arg, gn, int_allowables, gmn, gfn):
        with pytest.raises(ValueError):
            akv(arg, gn, int_allowables, gmn, gfn, None)


class TestFloats:

    @pytest.mark.parametrize('arg', (1.1,2.2,3.3))
    def test_accepts_allowed_floats(self, arg, gn, float_allowables, gmn, gfn):
        out = akv(arg, gn, float_allowables, gmn, gfn, None)
        assert out == arg

    @pytest.mark.parametrize('arg', (1,None,'junk', np.pi))
    def test_rejects_not_allowed_floats(self, arg, gn, float_allowables, gmn, gfn):
        with pytest.raises(ValueError):
            akv(arg, gn, float_allowables, gmn, gfn, None)


class TestStrings:

    @pytest.mark.parametrize('arg', ('a','b','c'))
    def test_accepts_allowed_strings_1(self, arg, gn, str_allowables, gmn, gfn):
        out = akv(arg,gn,str_allowables, gmn, gfn, None)
        assert out == arg

    # not case sensitive
    @pytest.mark.parametrize('arg', ('A','B','C'))
    def test_accepts_allowed_strings_2(self, arg, gn, str_allowables, gmn, gfn):
        out = akv(arg,gn,str_allowables, gmn, gfn, None)
        # returns in the case of what is in allowables
        assert out == arg.lower()  # because the allowables are lowercase


    @pytest.mark.parametrize('arg', (1, 'q', 'r', 's'))
    def test_rejects_not_allowed_strings(self, arg, gn, float_allowables, gmn, gfn):
        with pytest.raises(ValueError):
            akv(arg,gn,float_allowables, gmn, gfn, None)


    @pytest.mark.parametrize('arg', (1, None, np.pi))
    def test_rejects_not_allowed_junk(self, arg, gn, float_allowables, gmn, gfn):
        with pytest.raises(ValueError):
            akv(arg,gn,float_allowables, gmn, gfn, None)



class TestMixed:

    @pytest.mark.parametrize('arg', (1,2,3,'a','b','c'))
    def test_accepts_allowed_mixed(self, arg, gn, int_allowables,
                                     str_allowables, gmn, gfn):
        out = akv(arg,gn,int_allowables+str_allowables, gmn, gfn, None)
        assert out == arg

    # not case sensitive
    @pytest.mark.parametrize('arg', (1,2,3,'A','B','C'))
    def test_accepts_allowed_upper_case(self, arg, gn, int_allowables,
                                     str_allowables, gmn, gfn):
        result = akv(arg,gn,int_allowables+str_allowables, gmn, gfn, None)
        # returns in the case of what is in allowables
        try:
            assert result == arg.lower()  # because the allowables are lowercase
        except:
            assert result == arg


    @pytest.mark.parametrize('arg', (8,9,10, 'q', 'r', 's'))
    def test_rejects_not_allowed_mixed(self, arg, gn, int_allowables,
                                       float_allowables, gmn, gfn):
        with pytest.raises(ValueError):
            akv(arg,gn,int_allowables+float_allowables, gmn, gfn, None)


class TestBool:

    @pytest.mark.parametrize('arg', ((0, 1), [0, 1]))
    def test_bool_in_allowed_1(self, arg, gn, bool_allowables, gmn, gfn):
        # should not confuse 0 & 1 with bool
        with pytest.raises(ValueError):
            akv(arg, gn, bool_allowables, gmn, gfn, None)


    @pytest.mark.parametrize('arg', ((True, False), [False, True]))
    def test_bool_in_allowed_2(self, arg, gn, bool_allowables, gmn, gfn):
        # should accept bool & bool
        out = akv(arg, gn, bool_allowables, gmn, gfn, None)
        assert out is arg


    @pytest.mark.parametrize('arg', ((True, False), [False, True]))
    def test_bool_in_allowed_2(self, arg, gn, bool_allowables, gmn, gfn):
        # should not confuse 0 & 1 with bool
        with pytest.raises(ValueError):
            akv(arg, gn, [0, 1], gmn, gfn, None)


class TestSearchesIterables:

    @pytest.mark.parametrize('arg', ([1], [1,2], {1,2,3}))
    def test_accepts_allowed_ints(self, arg, gn, int_allowables, gmn, gfn):
        out = akv(arg,gn,int_allowables, gmn, gfn, None)
        assert np.array_equiv(out, np.array(list(arg), dtype=object))


    @pytest.mark.parametrize('arg', ([1,4], {1,2,4}))
    def test_rejects_not_allowed_ints(self, arg, gn, int_allowables, gmn, gfn):
        with pytest.raises(ValueError):
            akv(arg,gn,int_allowables, gmn, gfn, None)

    @pytest.mark.parametrize('arg', ([1.1], [1.1, 2.2], {1.1, 3.3, 2.2}))
    def test_accepts_allowed_floats(self, arg, gn, float_allowables, gmn, gfn):
        out = akv(arg,gn,float_allowables, gmn, gfn, None)
        assert np.array_equiv(out, np.array(list(arg)))

    @pytest.mark.parametrize('arg', ([1.9, 2.2], {1.1, 2,2, 4.4}))
    def test_rejects_not_allowed_floats(self, arg, gn, float_allowables, gmn, gfn):
        with pytest.raises(ValueError):
            akv(arg,gn,float_allowables, gmn, gfn, None)

    @pytest.mark.parametrize('arg', (['a'],['a', 'b'],['a','b','c']))
    def test_accepts_allowed_strings_1(self, arg, gn, str_allowables, gmn, gfn):
        out = akv(arg,gn,str_allowables, gmn, gfn, None)
        assert np.array_equiv(out, np.array(arg, dtype=object))

    # not case sensitive
    @pytest.mark.parametrize('arg', (['A'],['A', 'B'],['A','B','C']))
    def test_accepts_allowed_strings_2(self, arg, gn, str_allowables, gmn, gfn):
        out = akv(arg,gn,str_allowables, gmn, gfn, None)
        assert np.array_equiv(out, np.char.lower(list(arg)))  # allowables are lowercase

    @pytest.mark.parametrize('arg', (['a','b','q'], ['A', 'B', 'Q']))
    def test_rejects_not_allowed_strings(self, arg, gn, float_allowables, gmn, gfn):
        with pytest.raises(ValueError):
            akv(arg,gn,float_allowables, gmn, gfn, None)


@pytest.mark.parametrize('return_if_none', (4,'junk', np.pi, ['a'], {'a':1}))
def test_return_if_none_works(gn, int_allowables, gmn, gfn, return_if_none):
    output = akv(None, gn, int_allowables+[None], gmn, gfn, return_if_none)
    assert output == return_if_none


























