# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing._InterceptManager._partial_fit. \
    _parallel_constant_finder import _parallel_constant_finder

from pybear.utilities import nan_mask_numerical

import uuid

import numpy as np

import pytest




class TestParallelConstantFinder:


    @pytest.mark.parametrize('dtype', ('flt', 'str'))
    @pytest.mark.parametrize('has_nan', (True, False))
    @pytest.mark.parametrize('equal_nan', (True, False))
    @pytest.mark.parametrize('rtol, atol', ((1e-7, 1e-8), (1e-1, 1e-2)))
    def test_accuracy(self, dtype, has_nan, equal_nan, rtol, atol):

        # Methodology
        # need to test two different data types, numbers and strings
        # with nans and without
        # _parallel_constant_finder will only ever see numpy arrays
        # put a level of noise in the data that in one case is less
        # that rtol/atol, so that the column is considered constant; in
        # the other case the column is not constant because the noise
        # is greater than rtol/atol.

        _noise = 1e-6
        _size = 100

        if dtype=='flt':
            _X = np.random.normal(loc=1, scale=_noise, size=(_size,))

            if has_nan:
                _rand_idxs = \
                    np.random.choice(range(_size), _size//10, replace=False)
                _X[_rand_idxs] = np.nan
        elif dtype=='str':
            # if noise > tol, make a non-constant column
            if _noise > atol:
                _X = np.random.choice(list('abcde'), _size, replace=True)
            else:
                _X = np.full((_size,), 'a')

            _X = _X.astype(object)

            if has_nan:
                _rand_idxs = \
                    np.random.choice(range(_size), _size // 10, replace=False)
                _X[_rand_idxs] = 'nan'
        else:
            raise Exception


        out = _parallel_constant_finder(_X, equal_nan, rtol, atol)[0]



        if dtype == 'flt':
            if has_nan:
                if equal_nan:
                    if _noise <= atol:
                        _not_nan_mask = np.logical_not(nan_mask_numerical(_X))
                        assert out == np.mean(_X[_not_nan_mask])
                    elif _noise > atol:
                        assert isinstance(out, uuid.UUID)
                elif not equal_nan:
                    assert isinstance(out, uuid.UUID)
            elif not has_nan:
                # equal_nan doesnt matter
                if _noise <= atol:
                    assert out == np.mean(_X)
                elif _noise > atol:
                    assert isinstance(out, uuid.UUID)

        elif dtype == 'str':
            if has_nan:
                # _noise and rtol dont matter for str column, but we built the
                # str test vector to be constant or not based on _noise and rtol
                if equal_nan:
                    if _noise <= atol:   # should be constant
                        try:
                            # this will catch if the unique is being returned
                            # as a list (which it should not be). should be
                            # returning a single value.
                            iter(out)
                            if isinstance(out, str):
                                raise Exception
                            raise UnicodeError
                        except UnicodeError:
                            raise Exception
                        except:
                            pass
                        assert out == 'a'
                    elif _noise > atol:
                        assert isinstance(out, uuid.UUID)
                else:
                    assert isinstance(out, uuid.UUID)
            elif not has_nan:
                # equal_nan doesnt matter
                # _noise and rtol dont matter for str column, but we built the
                # str test vector to be constant or not based on _noise and rtol
                if _noise <= atol:    # should be constant
                    try:
                        # this will catch if the unique is being returned
                        # as a list (which it should not be). should be
                        # returning a single value.
                        iter(out)
                        if isinstance(out, str):
                            raise Exception
                        raise UnicodeError
                    except UnicodeError:
                        raise Exception
                    except:
                        pass
                    assert out == 'a'
                elif _noise > atol:
                    assert isinstance(out, uuid.UUID)

        else:
            raise Exception



    @pytest.mark.parametrize('dtype', ('flt', 'str'))
    @pytest.mark.parametrize('equal_nan', (True, False))
    def test_all_nans(self, dtype, equal_nan):

        # Methodology
        # need to test two different data types, numbers and strings
        # that are all nans

        _size = 100

        if dtype=='flt':
            _X = np.full((_size,), np.nan)
        elif dtype=='str':
            _X = np.full((_size,), 'nan').astype(object)
        else:
            raise Exception


        out = _parallel_constant_finder(_X, equal_nan, 1e-5, 1e-8)[0]

        if dtype == 'flt':
            if equal_nan:
                assert str(out) == 'nan'
            elif not equal_nan:
                assert isinstance(out, uuid.UUID)
        elif dtype == 'str':
            if equal_nan:
                try:
                    # this will catch if the unique is being returned
                    # as a list (which it should not be). should be
                    # returning a single value.
                    iter(out)
                    if isinstance(out, str):
                        raise Exception
                    raise UnicodeError
                except UnicodeError:
                    raise Exception
                except:
                    pass
                assert out == 'nan'
            elif not equal_nan:
                assert isinstance(out, uuid.UUID)







