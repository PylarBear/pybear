# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.SlimPolyFeatures.SlimPolyFeatures import \
    SlimPolyFeatures as SlimPoly

from pybear.preprocessing.SlimPolyFeatures._partial_fit. \
    _parallel_constant_finder import _parallel_constant_finder

from pybear.preprocessing.SlimPolyFeatures._partial_fit. \
    _parallel_column_comparer import _parallel_column_comparer



from copy import deepcopy
import numpy as np
import pandas as pd

import pytest







pytest.skip(reason=f"pizza not finished", allow_module_level=True)


# pizza dont forget accuracy when X is 1 column

# pizza need to test edge case of when no poly is produced, both for single
# fit and across partial fits



class TestAccuracy:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (20, 10)


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]


    @staticmethod
    @pytest.fixture(scope='function')
    def _kwargs():
        return {
            'degree': 2,  # will be overwrit during test
            'min_degree': 1,  # will be overwrit during test
            'keep': 'first',
            'interaction_only': False,  # will be overwrit during test
            'scan_X': False,
            'sparse_output': False,
            'feature_name_combiner': 'as_indices',
            'rtol': 1e-5,
            'atol': 1e-8,
            'equal_nan': True,
            'n_jobs': 1
        }


    @pytest.mark.parametrize('X_format', ('np', 'pd'))
    @pytest.mark.parametrize('X_dtype', ('flt', 'int'))
    @pytest.mark.parametrize('degree', (2, 3))
    @pytest.mark.parametrize('min_degree', (1, 2))
    @pytest.mark.parametrize('interaction_only', (True, False))
    def test_accuracy(
        self, _X_factory, _kwargs, X_format, X_dtype, degree, min_degree,
        interaction_only, _columns, _shape
    ):

        X = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format=X_format,
            _dtype=X_dtype,
            _columns=_columns,
            _shape=_shape
        )

        # set _kwargs
        _kwargs['X_format'] = X_format
        _kwargs['X_dtype'] = X_dtype
        _kwargs['degree'] = degree
        _kwargs['min_degree'] = min_degree
        _kwargs['interaction_only'] = interaction_only


        TestCls = SlimPoly(**_kwargs)


        # retain original format
        _og_format = type(X)

        TestCls.fit(X)

        TRFM_X = TestCls.transform(X)


        # ASSERTIONS ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        # poly_combinations_
        # poly_constants_
        # poly_duplicates_
        # dropped_poly_duplicates_
        # kept_poly_duplicates_


        # END ASSERTIONS ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *









