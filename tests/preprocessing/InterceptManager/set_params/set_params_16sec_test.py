# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np

from pybear.preprocessing.InterceptManager.InterceptManager import \
    InterceptManager as IM

import pytest




class TestSetParams:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (8, 5)


    @staticmethod
    @pytest.fixture(scope='function')
    def X(_X_factory, _shape):
        return _X_factory(
            _constants={0: 1, _shape[1]-1: 2},   # <===== important
            _dupl=None,
            _format='np',
            _dtype='int',
            _has_nan=False,
            _columns=None,
            _zeros=None,
            _shape=_shape
        )


    @staticmethod
    @pytest.fixture(scope='function')
    def y(_shape):
        return np.random.randint(0, 2, (_shape[0], 1), dtype=np.uint8)


    @staticmethod
    @pytest.fixture(scope='module')
    def _kwargs(_shape):
        return {
            'keep': 'last',
            'equal_nan': True,
            'rtol': 1e-5,
            'atol': 1e-8,
            'n_jobs': -1
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _alt_kwargs(_shape):
        return {
            'keep': 'first',
            'equal_nan': False,
            'rtol': 1e-6,
            'atol': 1e-9,
            'n_jobs': 2
        }

    # END Fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    def test_equality_set_params_before_and_after_fit(
        self, X, y, _kwargs, _alt_kwargs
    ):

        # test the equality of the data output under:
        # 1) set_params(via init) -> fit -> transform
        # 2) fit -> set_params -> transform

        # set_params(via init) -> fit -> transform
        FirstTestClass = IM(**_kwargs)
        assert FirstTestClass.keep == 'last'
        assert FirstTestClass.equal_nan is True
        assert FirstTestClass.rtol == 1e-5
        assert FirstTestClass.atol == 1e-8
        assert FirstTestClass.n_jobs == -1
        FirstTestClass.fit(X.copy(), y.copy())
        assert FirstTestClass.keep == 'last'
        assert FirstTestClass.equal_nan is True
        assert FirstTestClass.rtol == 1e-5
        assert FirstTestClass.atol == 1e-8
        assert FirstTestClass.n_jobs == -1
        FIRST_TRFM_X = FirstTestClass.transform(X.copy())
        assert FirstTestClass.keep == 'last'
        assert FirstTestClass.equal_nan is True
        assert FirstTestClass.rtol == 1e-5
        assert FirstTestClass.atol == 1e-8
        assert FirstTestClass.n_jobs == -1
        del FirstTestClass


        # fit -> set_params -> transform
        # all different params to start
        SecondTestClass = IM(**_alt_kwargs)
        assert SecondTestClass.keep == 'first'
        assert SecondTestClass.equal_nan is False
        assert SecondTestClass.rtol == 1e-6
        assert SecondTestClass.atol == 1e-9
        assert SecondTestClass.n_jobs == 2
        SecondTestClass.fit(X.copy(), y.copy())
        assert SecondTestClass.keep == 'first'
        assert SecondTestClass.equal_nan is False
        assert SecondTestClass.rtol == 1e-6
        assert SecondTestClass.atol == 1e-9
        assert SecondTestClass.n_jobs == 2
        SECOND_TRFM_X = SecondTestClass.transform(X.copy())
        assert SecondTestClass.keep == 'first'
        assert SecondTestClass.equal_nan is False
        assert SecondTestClass.rtol == 1e-6
        assert SecondTestClass.atol == 1e-9
        assert SecondTestClass.n_jobs == 2
        # should not be equal to first transform
        assert not np.array_equal(FIRST_TRFM_X, SECOND_TRFM_X)

        # all params are being changed back to those in FirstTestClass
        SecondTestClass.set_params(**_kwargs)
        assert SecondTestClass.keep == 'last'
        assert SecondTestClass.equal_nan is True
        assert SecondTestClass.rtol == 1e-5
        assert SecondTestClass.atol == 1e-8
        assert SecondTestClass.n_jobs == -1
        THIRD_TRFM_X = SecondTestClass.transform(X.copy())
        assert SecondTestClass.keep == 'last'
        assert SecondTestClass.equal_nan is True
        assert SecondTestClass.rtol == 1e-5
        assert SecondTestClass.atol == 1e-8
        assert SecondTestClass.n_jobs == -1
        del SecondTestClass

        # CHECK OUTPUT EQUAL REGARDLESS OF WHEN SET_PARAMS
        assert np.array_equiv(FIRST_TRFM_X, THIRD_TRFM_X)


    def test_set_params_between_fit_transforms(
        self, X, y, _kwargs, _alt_kwargs
    ):

        # fit_transform
        FirstTestClass = IM(**_kwargs)
        assert FirstTestClass.keep == 'last'
        assert FirstTestClass.equal_nan is True
        assert FirstTestClass.rtol == 1e-5
        assert FirstTestClass.atol == 1e-8
        assert FirstTestClass.n_jobs == -1
        FIRST_TRFM_X = FirstTestClass.fit_transform(X.copy(), y.copy())
        assert FirstTestClass.keep == 'last'
        assert FirstTestClass.equal_nan is True
        assert FirstTestClass.rtol == 1e-5
        assert FirstTestClass.atol == 1e-8
        assert FirstTestClass.n_jobs == -1

        # fit_transform -> set_params -> fit_transform
        SecondTestClass = IM(**_kwargs)
        assert SecondTestClass.keep == 'last'
        assert SecondTestClass.equal_nan is True
        assert SecondTestClass.rtol == 1e-5
        assert SecondTestClass.atol == 1e-8
        assert SecondTestClass.n_jobs == -1
        SecondTestClass.set_params(**_alt_kwargs)
        assert SecondTestClass.keep == 'first'
        assert SecondTestClass.equal_nan is False
        assert SecondTestClass.rtol == 1e-6
        assert SecondTestClass.atol == 1e-9
        assert SecondTestClass.n_jobs == 2
        SECOND_TRFM_X = SecondTestClass.fit_transform(X.copy(), y.copy())
        assert SecondTestClass.keep == 'first'
        assert SecondTestClass.equal_nan is False
        assert SecondTestClass.rtol == 1e-6
        assert SecondTestClass.atol == 1e-9
        assert SecondTestClass.n_jobs == 2

        # should not be equal to first transform
        assert not np.array_equal(FIRST_TRFM_X, SECOND_TRFM_X)

        # all params are being changed back to the original
        SecondTestClass.set_params(**_kwargs)
        assert SecondTestClass.keep == 'last'
        assert SecondTestClass.equal_nan is True
        assert SecondTestClass.rtol == 1e-5
        assert SecondTestClass.atol == 1e-8
        assert SecondTestClass.n_jobs == -1
        THIRD_TRFM_X = SecondTestClass.fit_transform(X.copy(), y.copy())
        assert SecondTestClass.keep == 'last'
        assert SecondTestClass.equal_nan is True
        assert SecondTestClass.rtol == 1e-5
        assert SecondTestClass.atol == 1e-8
        assert SecondTestClass.n_jobs == -1

        assert np.array_equiv(FIRST_TRFM_X, THIRD_TRFM_X)


    def test_set_params_output_repeatability(
        self, X, y, _kwargs, _alt_kwargs
    ):

        # changing and changing back on the same class gives same result
        # initialize, fit, transform, keep results
        # set all new params and transform
        # set back to the old params and transform, compare with the first output

        # initialize, fit, transform, and keep result
        TestClass = IM(**_kwargs)
        assert TestClass.keep == 'last'
        assert TestClass.equal_nan is True
        assert TestClass.rtol == 1e-5
        assert TestClass.atol == 1e-8
        assert TestClass.n_jobs == -1
        TestClass.fit(X.copy(), y.copy())
        assert TestClass.keep == 'last'
        assert TestClass.equal_nan is True
        assert TestClass.rtol == 1e-5
        assert TestClass.atol == 1e-8
        assert TestClass.n_jobs == -1
        FIRST_TRFM_X = TestClass.transform(X.copy())
        assert TestClass.keep == 'last'
        assert TestClass.equal_nan is True
        assert TestClass.rtol == 1e-5
        assert TestClass.atol == 1e-8
        assert TestClass.n_jobs == -1

        # use set_params to change all params.  DO NOT FIT!
        TestClass.set_params(**_alt_kwargs)
        assert TestClass.keep == 'first'
        assert TestClass.equal_nan is False
        assert TestClass.rtol == 1e-6
        assert TestClass.atol == 1e-9
        assert TestClass.n_jobs == 2
        SECOND_TRFM_X = TestClass.transform(X.copy())
        assert TestClass.keep == 'first'
        assert TestClass.equal_nan is False
        assert TestClass.rtol == 1e-6
        assert TestClass.atol == 1e-9
        assert TestClass.n_jobs == 2
        # should not be equal to first transform
        assert not np.array_equal(FIRST_TRFM_X, SECOND_TRFM_X)

        # use set_params again to change all params back to original values
        TestClass.set_params(**_kwargs)
        assert TestClass.keep == 'last'
        assert TestClass.equal_nan is True
        assert TestClass.rtol == 1e-5
        assert TestClass.atol == 1e-8
        assert TestClass.n_jobs == -1
        # transform again, and compare with the first output
        THIRD_TRFM_X = TestClass.transform(X.copy())
        assert TestClass.keep == 'last'
        assert TestClass.equal_nan is True
        assert TestClass.rtol == 1e-5
        assert TestClass.atol == 1e-8
        assert TestClass.n_jobs == -1

        assert np.array_equal(FIRST_TRFM_X, THIRD_TRFM_X)















