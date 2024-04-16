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
    return ['a', 'b', 'c', 1, 2, 3]

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
class TestStringArgs:

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


class TestAllowed:
    @pytest.mark.parametrize('allowed', (1, None, np.pi, {'a':1}, {}, 'junk'))
    def test_allowed_rejects_non_array_like(self, gn, allowed, gmn, gfn):
        with pytest.raises(TypeError):
            akv(1, gn, allowed, gmn, gfn, None)


    @pytest.mark.parametrize('allowed', ([1,2,3], (1,2,3), {1,2,3}, {'a', 1}))
    def test_allowed_accepts_array_like(self, gn, allowed, gmn, gfn):
        arg = 1
        out = akv(arg, gn, allowed, gmn, gfn, None)
        assert out == arg

    @pytest.mark.parametrize('allowed', ([], ()))
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


class Ints:

    @pytest.mark.parametrize('arg', (1,2,3))
    def test_accepts_allowed_ints(self, arg, gn, int_allowables, gmn, gfn):
        out = akv(arg,gn,int_allowables, gmn, gfn, None)
        assert out == arg

    @pytest.mark.parametrize('arg', (4,None,'junk', np.pi, []))
    def test_rejects_not_allowed_ints(self, arg, gn, int_allowables, gmn, gfn):
        with pytest.raises(ValueError):
            akv(arg,gn,int_allowables, gmn, gfn, None)


class Floats:

    @pytest.mark.parametrize('arg', (1.1,2.2,3.3))
    def test_accepts_allowed_floats(self, arg, gn, float_allowables, gmn, gfn):
        out = akv(arg,gn,float_allowables, gmn, gfn, None)
        assert out == arg

    @pytest.mark.parametrize('arg', (1,None,'junk', np.pi, []))
    def test_rejects_not_allowed_floats(self, arg, gn, float_allowables, gmn, gfn):
        with pytest.raises(ValueError):
            akv(arg,gn,float_allowables, gmn, gfn, None)


class Strings:

    @pytest.mark.parametrize('arg', ('a','b','c'))
    def test_accepts_allowed_strings(self, arg, gn, str_allowables, gmn, gfn):
        out = akv(arg,gn,str_allowables, gmn, gfn, None)
        assert out == arg

    # not case sensitive
    @pytest.mark.parametrize('arg', ('A','B','C'))
    def test_accepts_allowed_strings(self, arg, gn, str_allowables, gmn, gfn):
        result = akv(arg,gn,str_allowables, gmn, gfn, None)
        # returns in the case of what is in allowables
        assert result == arg.lower()  # because the allowables are lowercase


    @pytest.mark.parametrize('arg', (1, 'q', 'r', 's'))
    def test_rejects_not_allowed_strings(self, arg, gn, float_allowables, gmn, gfn):
        with pytest.raises(ValueError):
            akv(arg,gn,float_allowables, gmn, gfn, None)


    @pytest.mark.parametrize('arg', (1, None, np.pi, []))
    def test_rejects_not_allowed_junk(self, arg, gn, float_allowables, gmn, gfn):
        with pytest.raises(ValueError):
            akv(arg,gn,float_allowables, gmn, gfn, None)



class Mixed:

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


    @pytest.mark.parametrize('arg', (None, np.pi, [], {'a':1}))
    def test_rejects_not_allowed_junk(self, arg, gn, int_allowables,
                                      float_allowables, gmn, gfn):
        with pytest.raises(ValueError):
            akv(arg,gn,int_allowables+float_allowables, gmn, gfn, None)





class SearchesIterables:

    @pytest.mark.parametrize('arg', ([1], [1,2], [1,2,3,1,2,3]))
    def test_accepts_allowed_ints(self, arg, gn, int_allowables, gmn, gfn):
        out = akv(arg,gn,int_allowables, gmn, gfn, None)
        assert out == arg


    @pytest.mark.parametrize('arg', (
                                     [1],
                                     [1,2],
                                     pytest.mark.xfail([1,4], reason="junk"),
                                     pytest.mark.xfail([1,2,4], reason="junk")
                                     )
    )
    def test_rejects_not_allowed_ints(self, arg, gn, int_allowables, gmn, gfn):
        akv(arg,gn,int_allowables, gmn, gfn, None)

    @pytest.mark.parametrize('arg', ([1.1], [1.1, 2.2], [1.1, 3.3, 2.2]))
    def test_accepts_allowed_floats(self, arg, gn, float_allowables, gmn, gfn):
        out = akv(arg,gn,float_allowables, gmn, gfn, None)
        assert out == arg

    @pytest.mark.parametrize('arg', (
                                     [1.1],
                                     [1.1, 2.2],
                                     pytest.mark.xfail([1.9, 2.2], reason="junk"),
                                     pytest.mark.xfail([1.1, 2,2, 4.4], reason="junk")
                                    )
    )
    def test_rejects_not_allowed_floats(self, arg, gn, float_allowables, gmn, gfn):
        akv(arg,gn,float_allowables, gmn, gfn, None)

    @pytest.mark.parametrize('arg', (['a'],['a', 'b'],['a','b','c']))
    def test_accepts_allowed_strings(self, arg, gn, str_allowables, gmn, gfn):
        out = akv(arg,gn,str_allowables, gmn, gfn, None)
        assert out == arg

    # not case sensitive
    @pytest.mark.parametrize('arg', (['A'],['A', 'B'],['A','B','C']))
    def test_accepts_allowed_strings(self, arg, gn, str_allowables, gmn, gfn):
        out = akv(arg,gn,str_allowables, gmn, gfn, None)
        assert out == arg.lower()  # allowables are lowercase

    @pytest.mark.parametrize('arg', (
                                     ['a'],
                                     ['A','B'],
                                     pytest.mark.xfail(['a','b','q'], reason="junk")
                                     )
    )
    def test_rejects_not_allowed_strings(self, arg, gn, float_allowables, gmn, gfn):
        akv(arg,gn,float_allowables, gmn, gfn, None)



class OtherJunk:

    @pytest.mark.parametrize('arg', ({'a':1}, {1,2,3}))
    def test_accepts_allowed(self, arg, gn, gmn, gfn):
        akv(arg,gn, [{'a':1}, {1,2,3}, 'qqq'], gmn, gfn, None)

    @pytest.mark.parametrize('arg', (1, None, 'junk', np.pi, []))
    def test_rejects_not_allowed(self, arg, gn, float_allowables, gmn, gfn):
        with pytest.raises(ValueError):
            akv(arg,gn,[{'a':1}, {1,2,3}, 'qqq'], gmn, gfn, None)



@pytest.mark.parametrize('return_if_none', (4,'junk', np.pi, ['a'], {'a':1}))
def test_return_if_none_works(gn, int_allowables, gmn, gfn, return_if_none):
    output = akv(None, gn, int_allowables+[None], gmn, gfn, return_if_none)
    assert output == return_if_none


























