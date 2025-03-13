# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import numpy as np
import sparse_dict as sd
from pybear.data_validation import validate_user_input as vui
from pybear.sparse_dict._utils import (
                                        sparsity,
                                        outer_len,
                                        inner_len
)
from pybear.sparse_dict._random_ import uniform


class TestUniformValidation:

    @pytest.mark.parametrize('_min',
        ('junk', True, None, {1,2}, [1,2], (1,2), (1,), lambda x: x, {'a':1})
    )
    def test_rejects_junk_min(self, _min):
        with pytest.raises(TypeError):
            uniform(_min, 10_000, (3,3), 50)

    @pytest.mark.parametrize('_min', (float('-inf'), float('inf'), np.nan))
    def test_rejects_bad_min(self, _min):
        with pytest.raises(ValueError):
            uniform(_min, 100, (3,3), 50)

    @pytest.mark.parametrize('_min', (-10, 0, 10))
    def test_accepts_good_min(self, _min):
        uniform(_min, 100, (3,3), 50)


    @pytest.mark.parametrize('_max',
        ('junk', True, None, {1,2}, [1,2], (1,2), (1,), lambda x: x, {'a':1})
    )
    def test_rejects_junk_max(self, _max):
        with pytest.raises(TypeError):
            uniform(-100, _max, (3,3), 50)

    @pytest.mark.parametrize('_max', (float('-inf'), float('inf'), np.nan))
    def test_rejects_bad_max(self, _max):
        with pytest.raises(ValueError):
            uniform(-100, _max, (3,3), 50)

    @pytest.mark.parametrize('_max', (-10, 0, 10))
    def test_accepts_good_max(self, _max):
        uniform(-100, _max, (3,3), 50)


    @pytest.mark.parametrize('_shape',
        ('junk', True, None, lambda x: x, {'a':1}, ('a', 'b'), ('q', ))
    )
    def test_rejects_junk_shape(self, _shape):
        with pytest.raises(TypeError):
            uniform(-5, 5, _shape, 50)


    @pytest.mark.parametrize('_shape',
        ((np.nan, np.nan), (float('inf'), float('inf')),
         np.nan, float('inf'), float('-inf'), (-1,1), (1,-1), (-1,-1))
    )
    def test_rejects_bad_shape(self, _shape):
        with pytest.raises(ValueError):
            uniform(-5, 5, _shape, 50)

    @pytest.mark.parametrize('_shape', ((2,2), (2,)))
    def test_accepts_good_shape(self, _shape):
        uniform(-5, 5, _shape, 50)


    @pytest.mark.parametrize('_shape', (0, (0,), (0,0)))
    def test_accepts_good_zero_shapes(self, _shape):
        uniform(-5, 5, _shape, 50)

    @pytest.mark.parametrize('_sparsity',
        ((1,2), [1,2], {1,2}, {'a':1},'junk', None, True, lambda x: x)
    )
    def test_rejects_junk_sparsity(self, _sparsity):
        with pytest.raises(TypeError):
            uniform(-10, 10, (3,3), _sparsity)

    @pytest.mark.parametrize('_sparsity',
        (np.nan, float('-inf'), float('inf'), -10, 110)
    )
    def test_rejects_bad_sparsity(self, _sparsity):
        with pytest.raises(ValueError):
            uniform(-10, 10, (3,3), _sparsity)

    @pytest.mark.parametrize('_sparsity', (0, 50, 100))
    def test_accept_good_sparsity(self, _sparsity):
        uniform(0, 10, (3,3), _sparsity)




@pytest.mark.parametrize('_min, _max', ((2,10), (0, 10), (-10, 10)))
@pytest.mark.parametrize('_shape', ((8, 12), (12, 8), (5,), (5,0), (0,), (0,0)))
@pytest.mark.parametrize('_sparsity', (0, 25, 50, 75, 100))
class TestUniformAccuracy:


    @staticmethod
    @pytest.fixture
    def min_max_window():
        return 0.01


    @staticmethod
    @pytest.fixture
    def sparsity_window(_shape):
        try:
            return 100 / np.prod(_shape)
        except:
            return 0.5


    @pytest.fixture
    def sparse_dict_output(self, _min, _max, _shape, _sparsity):

        return uniform(_min, _max, _shape, _sparsity)


    @pytest.fixture
    def actual_sparsity(self, sparse_dict_output, _shape):

        if 0 in _shape:
            pytest.skip(reason=f'cannot measure sparsity of an empty object')

        return sparsity(sparse_dict_output)


    def test_min_max(self, sparse_dict_output, _min, _max, _shape, _sparsity):

        if 0 in _shape:
            pytest.skip(reason=f'no values to get min/max of if empty')

        # get exp min / max ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        if _sparsity < 0 or _sparsity > 100:
            raise ValueError(f'sparsity ({_sparsity}) is out of bounds')

        exp_min, exp_max = _min, _max

        if _sparsity == 0:
            # exp_min = exp_min
            # exp_max = exp_max
            pass

        elif _sparsity > 0 and _sparsity < 100:
            exp_min = min(exp_min, 0)
            exp_max = max(exp_max, 0)

        elif _sparsity == 100:
            exp_min, exp_max = 0, 0

        # END get exp min / max ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # get act min / max ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        assert isinstance(sparse_dict_output, dict)
        if len(_shape) == 2:
            assert isinstance(sparse_dict_output[0], dict)

        if len(_shape) == 2:
            pseudo_min = min(map(min, map(dict.values, sparse_dict_output.values())))
            pseudo_max = max(map(max, map(dict.values, sparse_dict_output.values())))
        elif len(_shape) == 1:
            pseudo_min = min(sparse_dict_output.values())
            pseudo_max = max(sparse_dict_output.values())

        if _sparsity == 0:
            act_min = pseudo_min
            act_max = pseudo_max
        else:
            act_min = min(0, pseudo_min)
            act_max = max(0, pseudo_max)

        del pseudo_min, pseudo_max

        # END get act min / max ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        assert act_min >= exp_min, f'act_min ({act_min}) < exp_min ({exp_min})'
        assert act_max <= exp_max, f'act_max ({act_max}) > exp_max ({exp_max})'


    def test_shape(self, sparse_dict_output, _shape):

        if len(_shape) == 2:
            if _shape == (0,0):
                assert len(sparse_dict_output) == 0, \
                    f'act_outer_len != exp_outer_len'
            elif _shape[0] == 0:
                assert len(sparse_dict_output[0]) == 0, \
                    f'act_outer_len != exp_outer_len'
            elif _shape[1] == 0:
                assert len(sparse_dict_output) == _shape[0], \
                    f'act_outer_len != exp_outer_len'
                assert len(sparse_dict_output[0]) == 0, \
                    f'act_inner_len != exp_inner_len'
            else:
                assert outer_len(sparse_dict_output) == _shape[0], \
                    f'act_outer_len != exp_outer_len'
                assert inner_len(sparse_dict_output) == _shape[1], \
                    f'act_inner_len != exp_inner_len'
        elif len(_shape) == 1:

            if _shape == (0,):
                assert len(sparse_dict_output) == 0, \
                    f'act_outer_len != exp_outer_len'
            else:
                assert inner_len(sparse_dict_output) == _shape[0], \
                    f'act_inner_len != exp_inner_len'


    def test_sparsity(self, sparse_dict_output, _sparsity,
                      actual_sparsity, sparsity_window):
        assert actual_sparsity <= _sparsity + sparsity_window, \
            f"actual sparsity above (expected sparsity + sparsity window)"
        assert actual_sparsity >= _sparsity - sparsity_window, \
            f"actual sparsity below (expected sparsity - sparsity window)"


    def test_outer_key_dtype(self, sparse_dict_output, _shape):

        if len(_shape) == 1:
            pytest.skip(reason = f"cannot get outer keys on an inner dict")

        if _shape[0] == 0:
            pytest.skip(reason = f"cannot get outer keys on an empty dict")

        RAW_OUTER_KEY_DTYPES = \
            np.unique(list(map(str, map(type, sparse_dict_output))))

        assert len(RAW_OUTER_KEY_DTYPES) == 1, f'number of outer key dtypes > 1'
        assert RAW_OUTER_KEY_DTYPES[0] == "<class 'int'>", \
                f"outer key dtype is not <class 'int'>"

        del RAW_OUTER_KEY_DTYPES


    def test_inner_key_dtype(self, sparse_dict_output, _shape):

        # GET / VERIFY INNER KEY DTYPE   (MUST ALWAYS BE py_int) ******

        if _shape == (0,0):
            pytest.skip(reason=f"cannot get key types on an empty dict")

        if _shape[0] == 0:
            pytest.skip(reason=f"cannot get key types on an empty dict")

        if len(_shape) == 2 and _shape[1] == 0:
            pytest.skip(reason=f"cannot get inner key types on empty inner dicts")

        if len(_shape)==1:
            INNER_KEYS = [list(sparse_dict_output)]
        elif len(_shape)==2:
            INNER_KEYS = list(map(list, map(dict.keys,
                                            sparse_dict_output.values())))

        DTYPE_HOLDER = []
        for _ in INNER_KEYS:
            DTYPE_HOLDER += _

        INNER_KEY_DTYPES = np.unique(list(map(str, map(type, DTYPE_HOLDER))))
        del DTYPE_HOLDER

        assert len(INNER_KEY_DTYPES) == 1, f'number of inner key dtypes > 1'
        assert INNER_KEY_DTYPES[0] != str(type(int)), \
            f"inner key dtype is not {str(type(int))} ({INNER_KEY_DTYPES[0]})"

        del INNER_KEY_DTYPES


    def test_value_dtype(self, sparse_dict_output, _shape):

        if _shape == (0,0):
            pytest.skip(reason=f"cannot get value types on an empty dict")

        if _shape[0] == 0:
            pytest.skip(reason=f"cannot get value types on an empty dict")

        if len(_shape) == 2 and _shape[1] == 0:
            pytest.skip(reason=f"cannot get value types on empty inner dicts")

        if len(_shape) == 1:
            VALUES = [list(sparse_dict_output.values())]
        elif len(_shape) == 2:
            VALUES = list(map(list, map(dict.values, sparse_dict_output.values())))

        DTYPE_HOLDER = []
        for _ in VALUES:
            DTYPE_HOLDER += _

        VALUE_DTYPES = np.unique(list(map(str, map(type, DTYPE_HOLDER))))

        assert len(VALUE_DTYPES) == 1, f'number of inner value dtypes > 1'
        assert VALUE_DTYPES[0] == str(float), f"act_value_dtype != float"

        del VALUE_DTYPES




































