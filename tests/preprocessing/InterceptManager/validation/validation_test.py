# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing._InterceptManager._validation._validation import (
    _validation
)

import pandas as pd

import pytest



class TestValidation:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (50, 10)


    @staticmethod
    @pytest.fixture(scope='function')
    def _X(_X_factory, _format, _shape):
        return _X_factory(_format=_format, _shape=_shape)


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]


    @pytest.mark.parametrize('_format', ('np', 'pd'))
    @pytest.mark.parametrize('_keep',
        (
            'first', 'last', 'random', 'none', {'Intercept': 1}, 0, 'string',
            lambda x: 1
         )
    )
    @pytest.mark.parametrize('_equal_nan', (True, False))
    @pytest.mark.parametrize('_rtol', (1e-6, 1e-1))
    @pytest.mark.parametrize('_atol', (1e-6, 1))
    @pytest.mark.parametrize('_n_jobs', (None, -1, 1))
    @pytest.mark.parametrize('columns_is_passed', (True, False))
    def test_accepts_good(
        self, _X, _columns, _format, _keep, _rtol, _atol, _equal_nan, _n_jobs,
        columns_is_passed
    ):

        # test the following degenerate conditions are blocked
        # --- np is passed and _columns is not None
        # --- columns not passed and keep is feature str

        # otherwise, _validation should not block, and all the tests here
        # should be good

        # bad conditions are handled in tests for the individual modules

        if _format == 'pd':
            if columns_is_passed:
                _X = pd.DataFrame(data=_X, columns=_columns)
            elif not columns_is_passed:
                _X = pd.DataFrame(data=_X)

        if columns_is_passed and _format == 'np':
            # At first glance it would seem that this is an impossible condition,
            # that should be disallowed in _validation, and that should either
            # be skipped here or run under pytest.raises. But in practice this
            # is possible and needs to be allowed, because if first fit is on
            # a pd df then feature_names_in_ will exist in IM and will be passed
            # to _validation even if subsequent fits are done with np arrays.
            pass


        if _keep == 'string':

            _keep = _columns[0]

            if not columns_is_passed: # could be np or pd

                if _format == 'np':

                    with pytest.raises(ValueError):
                        _validation(
                            _X,
                            None,
                            _keep,
                            _equal_nan,
                            _rtol,
                            _atol,
                            _n_jobs
                        )
                    pytest.skip(
                        reason=f"cannot have str _keep when columns are not passed"
                    )

                elif _format == 'pd':

                    with pytest.raises(ValueError):
                        _validation(
                            _X,
                            _X.columns,
                            _keep,
                            _equal_nan,
                            _rtol,
                            _atol,
                            _n_jobs
                        )
                    pytest.skip(
                        reason=f"cannot have feature str _keep when columns are "
                        f"not passed"
                    )


        _validation(
            _X,
            _columns if columns_is_passed else None,
            _keep,
            _equal_nan,
            _rtol,
            _atol,
            _n_jobs
        )











