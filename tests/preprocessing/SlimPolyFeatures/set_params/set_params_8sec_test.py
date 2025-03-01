# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing._SlimPolyFeatures.SlimPolyFeatures import \
    SlimPolyFeatures as SlimPoly

import numpy as np

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
            _dupl=None,
            _format='np',
            _dtype='flt',
            _has_nan=False,
            _columns=None,
            _constants=None,
            _zeros=None,
            _shape=_shape
        )


    @staticmethod
    @pytest.fixture(scope='function')
    def y(_shape):
        return np.random.randint(0, 2, (_shape[0], 1), dtype=np.uint8)


    @staticmethod
    @pytest.fixture(scope='module')
    def _kwargs():
        return {
            'degree': 2,
            'min_degree': 1,
            'keep': 'first',
            'interaction_only': False,
            'scan_X': False,
            'sparse_output': False,
            'feature_name_combiner': lambda _columns, _x: 'abc',
            'rtol': 1e-5,
            'atol': 1e-8,
            'equal_nan': False,
            'n_jobs': 1
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _kwargs_allowed():
        return {
            'keep': 'first',
            'sparse_output': False,
            'feature_name_combiner': lambda _columns, _x: 'abc',
            'n_jobs': 1
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _alt_kwargs_not_blocked():
        return {
            'degree': 3,  # <==== diff
            'min_degree': 2,  # <==== diff
            'keep': 'last',  # <==== diff
            'interaction_only': True,  # <==== diff
            'scan_X': True,  # <==== diff
            'sparse_output': False,
            'feature_name_combiner': lambda _columns, _x: 'xyz', # <==== diff
            'rtol': 1e-4,  # <==== diff
            'atol': 1e-7,  # <==== diff
            'equal_nan': True,  # <==== diff
            'n_jobs': 2  # <==== diff
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _alt_kwargs_allowed():
        return {
            'keep': 'last',  # <==== diff
            'sparse_output': False,
            'feature_name_combiner': lambda _columns, _x: 'xyz', # <==== diff
            'n_jobs': 2  # <==== diff
        }

    # END Fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    def test_blocks_some_params_after_fit(self, X, y, _kwargs):

        # INITIALIZE
        TestCls = SlimPoly(**_kwargs)

        # CAN SET ANYTHING BEFORE FIT
        TestCls.set_params(**_kwargs)

        # 'keep', 'sparse_output', 'feature_name_combiner',
        # and 'n_jobs' not blocked after fit()
        TestCls.fit(X, y)

        allowed_kwargs = {
            'keep': 'last',
            'sparse_output': True,
            'feature_name_combiner': lambda _columns, _x: 'whatever',
            'n_jobs': 2
        }

        TestCls.set_params(**allowed_kwargs)

        for param, value in allowed_kwargs.items():
            assert getattr(TestCls, param) == value

        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        #  ANYTHING OTHER THAN 'keep', 'sparse_output', 'feature_name_combiner',
        #         # and 'n_jobs' are blocked after fit() and warns.
        disallowed_kwargs = {
            'degree': 3,
            'min_degree': 2,
            'interaction_only': True,
            'scan_X': True,
            'rtol': 1e-3,
            'atol': 1e-5,
            'equal_nan': True,
        }

        for param, value in disallowed_kwargs.items():
            _og_value = getattr(TestCls, param)
            with pytest.warns():
                TestCls.set_params(**{param:value})
            # assert did not set the new value,
            # kept old, which was from _kwargs
            assert getattr(TestCls, param) == _og_value == _kwargs[param]

        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        # test a mix of allowed and disallowed applies the allowed and warns
        TestCls.set_params(**_kwargs)

        assert TestCls.keep == 'first'
        assert TestCls.sparse_output is False
        assert TestCls.degree == 2
        assert TestCls.min_degree == 1

        mixed_kwargs = {
            'keep': 'last',
            'sparse_output': True,
            'degree': 3,
            'min_degree': 2
        }

        with pytest.warns():
            TestCls.set_params(**mixed_kwargs)

        # were changed
        assert TestCls.keep == 'last'
        assert TestCls.sparse_output is True

        # were not changed
        assert TestCls.degree == 2
        assert TestCls.min_degree == 1



    def test_equality_set_params_before_and_after_fit(
        self, X, y, _kwargs, _kwargs_allowed, _alt_kwargs_not_blocked,
        _alt_kwargs_allowed
    ):

        # test the equality of the data output under:
        # 1) set_params(via init) -> fit -> transform
        # 2) fit -> set_params -> transform

        # set_params(via init) -> fit -> transform
        FirstTestClass = SlimPoly(**_kwargs)
        for param, value in _kwargs.items():
            assert getattr(FirstTestClass, param) == value
        FirstTestClass.fit(X.copy(), y.copy())
        for param, value in _kwargs.items():
            assert getattr(FirstTestClass, param) == value
        FIRST_TRFM_X = FirstTestClass.transform(X.copy())
        for param, value in _kwargs.items():
            assert getattr(FirstTestClass, param) == value

        del FirstTestClass

        # fit -> set_params -> transform
        # all different params to start
        SecondTestClass = SlimPoly(**_alt_kwargs_not_blocked)
        for param, value in _alt_kwargs_not_blocked.items():
            assert getattr(SecondTestClass, param) == value
        SecondTestClass.fit(X.copy(), y.copy())
        for param, value in _alt_kwargs_not_blocked.items():
            assert getattr(SecondTestClass, param) == value
        SECOND_TRFM_X = SecondTestClass.transform(X.copy())
        for param, value in _alt_kwargs_not_blocked.items():
            assert getattr(SecondTestClass, param) == value

        # should not be equal to first transform
        assert not np.array_equal(FIRST_TRFM_X, SECOND_TRFM_X)

        # params are being changed back to those in FirstTestClass
        # but this class is already fit so most are blocked
        SecondTestClass.set_params(**_kwargs_allowed)
        assert SecondTestClass.degree == 3
        assert SecondTestClass.min_degree == 2
        assert SecondTestClass.keep == 'first'  # <==== allowed
        assert SecondTestClass.interaction_only is True
        assert SecondTestClass.scan_X is True
        assert SecondTestClass.sparse_output is False  # <==== allowed
        assert SecondTestClass.feature_name_combiner([], (1, 2, 3)) == 'abc' # <
        assert SecondTestClass.rtol == 1e-4
        assert SecondTestClass.atol == 1e-7
        assert SecondTestClass.equal_nan is True
        assert SecondTestClass.n_jobs == 1  # <==== allowed
        THIRD_TRFM_X = SecondTestClass.transform(X.copy())
        assert SecondTestClass.degree == 3
        assert SecondTestClass.min_degree == 2
        assert SecondTestClass.keep == 'first'  # <==== allowed
        assert SecondTestClass.interaction_only is True
        assert SecondTestClass.scan_X is True
        assert SecondTestClass.sparse_output is False  # <==== allowed
        assert SecondTestClass.feature_name_combiner([], (1, 2, 3)) == 'abc' # <
        assert SecondTestClass.rtol == 1e-4
        assert SecondTestClass.atol == 1e-7
        assert SecondTestClass.equal_nan is True
        assert SecondTestClass.n_jobs == 1  # <==== allowed
        del SecondTestClass

        # THE PARAMS THAT ARE NOT ALLOWED TO CHANGE AFTER FIT CONTROL THIS
        # SINCE THEY CANT BE CHANGED, THIRD MUST EQUAL SECOND
        assert np.array_equiv(SECOND_TRFM_X, THIRD_TRFM_X)


    def test_set_params_between_fit_transforms(
        self, X, y, _kwargs, _kwargs_allowed, _alt_kwargs_not_blocked,
        _alt_kwargs_allowed
    ):

        # fit_transform
        FirstTestClass = SlimPoly(**_kwargs)
        for param, value in _kwargs.items():
            assert getattr(FirstTestClass, param) == value
        FIRST_TRFM_X = FirstTestClass.fit_transform(X.copy(), y.copy())
        for param, value in _kwargs.items():
            assert getattr(FirstTestClass, param) == value

        # fit_transform -> set_params -> fit_transform
        SecondTestClass = SlimPoly(**_kwargs)
        for param, value in _kwargs.items():
            assert getattr(SecondTestClass, param) == value
        # this is already fit, only can change allowed params
        SecondTestClass.set_params(**_alt_kwargs_allowed)
        assert SecondTestClass.degree == 2
        assert SecondTestClass.min_degree == 1
        assert SecondTestClass.keep == 'last'   # <==== allowed
        assert SecondTestClass.interaction_only is False
        assert SecondTestClass.scan_X is False
        assert SecondTestClass.sparse_output is False   # <==== allowed
        assert SecondTestClass.feature_name_combiner([], (1, 2, 3)) == 'xyz' # <
        assert SecondTestClass.rtol == 1e-5
        assert SecondTestClass.atol == 1e-8
        assert SecondTestClass.equal_nan is False
        assert SecondTestClass.n_jobs == 2   # <==== allowed
        SECOND_TRFM_X = SecondTestClass.fit_transform(X.copy(), y.copy())
        assert SecondTestClass.degree == 2
        assert SecondTestClass.min_degree == 1
        assert SecondTestClass.keep == 'last'   # <==== allowed
        assert SecondTestClass.interaction_only is False
        assert SecondTestClass.scan_X is False
        assert SecondTestClass.sparse_output is False   # <==== allowed
        assert SecondTestClass.feature_name_combiner([], (1, 2, 3)) == 'xyz' # <
        assert SecondTestClass.rtol == 1e-5
        assert SecondTestClass.atol == 1e-8
        assert SecondTestClass.equal_nan is False
        assert SecondTestClass.n_jobs == 2   # <==== allowed

        # THE PARAMS THAT ARE NOT ALLOWED TO CHANGE AFTER FIT CONTROL THIS
        # SINCE THEY CANT BE CHANGED, SECOND MUST EQUAL FIRST
        assert np.array_equal(FIRST_TRFM_X, SECOND_TRFM_X)

        # all params are being changed back to the original
        # but this instance is already fit so most are blocked
        # only set the allowed
        SecondTestClass.set_params(**_kwargs_allowed)
        assert SecondTestClass.degree == 2
        assert SecondTestClass.min_degree == 1
        assert SecondTestClass.keep == 'first'   # <==== allowed
        assert SecondTestClass.interaction_only is False
        assert SecondTestClass.scan_X is False
        assert SecondTestClass.sparse_output is False   # <==== allowed
        assert SecondTestClass.feature_name_combiner([], (1, 2, 3)) == 'abc' # <
        assert SecondTestClass.rtol == 1e-5
        assert SecondTestClass.atol == 1e-8
        assert SecondTestClass.equal_nan is False
        assert SecondTestClass.n_jobs == 1   # <==== allowed
        THIRD_TRFM_X = SecondTestClass.fit_transform(X.copy(), y.copy())
        assert SecondTestClass.degree == 2
        assert SecondTestClass.min_degree == 1
        assert SecondTestClass.keep == 'first'   # <==== allowed
        assert SecondTestClass.interaction_only is False
        assert SecondTestClass.scan_X is False
        assert SecondTestClass.sparse_output is False   # <==== allowed
        assert SecondTestClass.feature_name_combiner([], (1, 2, 3)) == 'abc' # <
        assert SecondTestClass.rtol == 1e-5
        assert SecondTestClass.atol == 1e-8
        assert SecondTestClass.equal_nan is False
        assert SecondTestClass.n_jobs == 1   # <==== allowed

        # THE PARAMS THAT ARE NOT ALLOWED TO CHANGE AFTER FIT CONTROL THIS
        # SINCE THEY CANT BE CHANGED, THIRD MUST EQUAL SECOND
        assert np.array_equiv(SECOND_TRFM_X, THIRD_TRFM_X)


    def test_set_params_output_repeatability(
        self, X, y, _kwargs, _kwargs_allowed, _alt_kwargs_allowed
    ):

        # changing and changing back on the same class gives same result
        # initialize, fit, transform, keep results
        # set new params and transform
        # set back to the old params and transform, compare with the first output
        # most params are blocked after fit

        # initialize, fit, transform, and keep result
        TestClass = SlimPoly(**_kwargs)
        for param, value in _kwargs.items():
            assert getattr(TestClass, param) == value
        TestClass.fit(X.copy(), y.copy())
        for param, value in _kwargs.items():
            assert getattr(TestClass, param) == value
        FIRST_TRFM_X = TestClass.transform(X.copy())
        for param, value in _kwargs.items():
            assert getattr(TestClass, param) == value

        # use set_params to change allowed params.  DO NOT FIT!
        # most params are blocked
        TestClass.set_params(**_alt_kwargs_allowed)
        assert TestClass.degree == 2
        assert TestClass.min_degree == 1
        assert TestClass.keep == 'last'  # <==== allowed
        assert TestClass.interaction_only is False
        assert TestClass.scan_X is False
        assert TestClass.sparse_output is False  # <==== allowed
        assert TestClass.feature_name_combiner([], (1, 2, 3)) == 'xyz' # <===
        assert TestClass.rtol == 1e-5
        assert TestClass.atol == 1e-8
        assert TestClass.equal_nan is False
        assert TestClass.n_jobs == 2  # <==== allowed
        SECOND_TRFM_X = TestClass.transform(X.copy())
        assert TestClass.degree == 2
        assert TestClass.min_degree == 1
        assert TestClass.keep == 'last'  # <==== allowed
        assert TestClass.interaction_only is False
        assert TestClass.scan_X is False
        assert TestClass.sparse_output is False  # <==== allowed
        assert TestClass.feature_name_combiner([], (1, 2, 3)) == 'xyz' # <===
        assert TestClass.rtol == 1e-5
        assert TestClass.atol == 1e-8
        assert TestClass.equal_nan is False
        assert TestClass.n_jobs == 2  # <==== allowed

        # THE PARAMS THAT ARE NOT ALLOWED TO CHANGE AFTER FIT CONTROL THIS
        # SINCE THEY CANT BE CHANGED, SECOND MUST EQUAL FIRST
        assert np.array_equal(FIRST_TRFM_X, SECOND_TRFM_X)

        # use set_params again to change all params back to original values
        TestClass.set_params(**_kwargs_allowed)
        assert TestClass.degree == 2
        assert TestClass.min_degree == 1
        assert TestClass.keep == 'first'  # <==== allowed
        assert TestClass.interaction_only is False
        assert TestClass.scan_X is False
        assert TestClass.sparse_output is False  # <==== allowed
        assert TestClass.feature_name_combiner([], (1, 2, 3)) == 'abc' # <===
        assert TestClass.rtol == 1e-5
        assert TestClass.atol == 1e-8
        assert TestClass.equal_nan is False
        assert TestClass.n_jobs == 1  # <==== allowed
        # transform again, and compare with the first output
        THIRD_TRFM_X = TestClass.transform(X.copy())
        assert TestClass.degree == 2
        assert TestClass.min_degree == 1
        assert TestClass.keep == 'first'  # <==== allowed
        assert TestClass.interaction_only is False
        assert TestClass.scan_X is False
        assert TestClass.sparse_output is False  # <==== allowed
        assert TestClass.feature_name_combiner([], (1, 2, 3)) == 'abc' # <===
        assert TestClass.rtol == 1e-5
        assert TestClass.atol == 1e-8
        assert TestClass.equal_nan is False
        assert TestClass.n_jobs == 1  # <==== allowed

        # THE PARAMS THAT ARE NOT ALLOWED TO CHANGE AFTER FIT CONTROL THIS
        # SINCE THEY CANT BE CHANGED, THIRD MUST EQUAL SECOND (AND FIRST)
        assert np.array_equal(SECOND_TRFM_X, THIRD_TRFM_X)






