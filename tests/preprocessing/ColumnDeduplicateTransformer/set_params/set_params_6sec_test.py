# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




import pytest

import numpy as np

from pybear.preprocessing._ColumnDeduplicateTransformer. \
    ColumnDeduplicateTransformer import ColumnDeduplicateTransformer as CDT




class TestSetParams:


    @staticmethod
    @pytest.fixture(scope='function')
    def X(_X_factory, _shape):
        return _X_factory(
            _dupl=[[0, 1, _shape[1]-1]],   # <===== important
            _format='np',
            _dtype='int',
            _has_nan=False,
            _columns=None,
            _zeros=None,
            _shape=_shape
        )


    @staticmethod
    @pytest.fixture(scope='function')
    def _kwargs():
        return {
            'keep': 'first',
            'do_not_drop': None,
            'conflict': 'raise',
            'rtol': 1e-5,
            'atol': 1e-8,
            'equal_nan': False,
            'n_jobs': 1  # confliction isnt a problem, 1 or -1 is fine
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _alt_kwargs(_shape):
        return {
            'keep': 'last',
            'do_not_drop': None,
            'conflict': 'ignore',
            'rtol': 1e-6,
            'atol': 1e-9,
            'equal_nan': True,
            'n_jobs': -1  # confliction isnt a problem, 1 or -1 is fine
        }

    # END Fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    def test_equality_set_params_before_and_after_fit(
        self, X, _y_np, _kwargs, _alt_kwargs
    ):

        # test the equality of the data output under:
        # 1) set_params(via init) -> fit -> transform
        # 2) fit -> set_params -> transform

        # set_params(via init) -> fit -> transform
        FirstTestClass = CDT(**_kwargs)
        for param, value in _kwargs.items():
            assert getattr(FirstTestClass, param) == value
        FirstTestClass.fit(X.copy(), _y_np.copy())
        for param, value in _kwargs.items():
            assert getattr(FirstTestClass, param) == value
        FIRST_TRFM_X = FirstTestClass.transform(X.copy())
        for param, value in _kwargs.items():
            assert getattr(FirstTestClass, param) == value
        del FirstTestClass


        # fit -> set_params -> transform
        # all different params to start
        SecondTestClass = CDT(**_alt_kwargs)
        for param, value in _alt_kwargs.items():
            assert getattr(SecondTestClass, param) == value
        SecondTestClass.fit(X.copy(), _y_np.copy())
        for param, value in _alt_kwargs.items():
            assert getattr(SecondTestClass, param) == value
        SECOND_TRFM_X = SecondTestClass.transform(X.copy())
        for param, value in _alt_kwargs.items():
            assert getattr(SecondTestClass, param) == value
        # should not be equal to first transform
        assert not np.array_equal(FIRST_TRFM_X, SECOND_TRFM_X)

        # all params are being changed back to those in FirstTestClass
        SecondTestClass.set_params(**_kwargs)
        for param, value in _kwargs.items():
            assert getattr(SecondTestClass, param) == value
        THIRD_TRFM_X = SecondTestClass.transform(X.copy())
        for param, value in _kwargs.items():
            assert getattr(SecondTestClass, param) == value
        del SecondTestClass

        # CHECK OUTPUT EQUAL REGARDLESS OF WHEN SET_PARAMS
        assert np.array_equiv(FIRST_TRFM_X, THIRD_TRFM_X)


    def test_set_params_between_fit_transforms(
        self, X, _y_np, _kwargs, _alt_kwargs
    ):

        # fit_transform
        FirstTestClass = CDT(**_kwargs)
        for param, value in _kwargs.items():
            assert getattr(FirstTestClass, param) == value
        FIRST_TRFM_X = FirstTestClass.fit_transform(X.copy(), _y_np.copy())
        for param, value in _kwargs.items():
            assert getattr(FirstTestClass, param) == value

        # fit_transform -> set_params -> fit_transform
        SecondTestClass = CDT(**_kwargs)
        for param, value in _kwargs.items():
            assert getattr(SecondTestClass, param) == value
        SecondTestClass.set_params(**_alt_kwargs)
        for param, value in _alt_kwargs.items():
            assert getattr(SecondTestClass, param) == value
        SECOND_TRFM_X = SecondTestClass.fit_transform(X.copy(), _y_np.copy())
        for param, value in _alt_kwargs.items():
            assert getattr(SecondTestClass, param) == value

        # should not be equal to first transform
        assert not np.array_equal(FIRST_TRFM_X, SECOND_TRFM_X)

        # all params are being changed back to the original
        SecondTestClass.set_params(**_kwargs)
        for param, value in _kwargs.items():
            assert getattr(SecondTestClass, param) == value
        THIRD_TRFM_X = SecondTestClass.fit_transform(X.copy(), _y_np.copy())
        for param, value in _kwargs.items():
            assert getattr(SecondTestClass, param) == value


        assert np.array_equiv(FIRST_TRFM_X, THIRD_TRFM_X)


    def test_set_params_output_repeatability(
        self, X, _y_np, _kwargs, _alt_kwargs
    ):

        # changing and changing back on the same class gives same result
        # initialize, fit, transform, keep results
        # set all new params and transform
        # set back to the old params and transform, compare with the first output

        # initialize, fit, transform, and keep result
        TestClass = CDT(**_kwargs)
        for param, value in _kwargs.items():
            assert getattr(TestClass, param) == value
        TestClass.fit(X.copy(), _y_np.copy())
        for param, value in _kwargs.items():
            assert getattr(TestClass, param) == value
        FIRST_TRFM_X = TestClass.transform(X.copy())
        for param, value in _kwargs.items():
            assert getattr(TestClass, param) == value
        # use set_params to change all params.  DO NOT FIT!
        TestClass.set_params(**_alt_kwargs)
        for param, value in _alt_kwargs.items():
            assert getattr(TestClass, param) == value
        SECOND_TRFM_X = TestClass.transform(X.copy())
        for param, value in _alt_kwargs.items():
            assert getattr(TestClass, param) == value
        # should not be equal to first transform
        assert not np.array_equal(FIRST_TRFM_X, SECOND_TRFM_X)

        # use set_params again to change all params back to original values
        TestClass.set_params(**_kwargs)
        for param, value in _kwargs.items():
            assert getattr(TestClass, param) == value
        # transform again, and compare with the first output
        THIRD_TRFM_X = TestClass.transform(X.copy())
        for param, value in _kwargs.items():
            assert getattr(TestClass, param) == value


        assert np.array_equal(FIRST_TRFM_X, THIRD_TRFM_X)











