# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import pandas as pd
import scipy.sparse as ss
from uuid import uuid4

from pybear.preprocessing.ColumnDeduplicateTransformer._header_handling. \
    _header_handling_pre import _header_handling_pre



class TestHeaderHandlingPre:

    # _header_handling_pre(
    #     _X: DataType,
    #     _columns: ColumnsType
    # ):

    # if a dataframe is passed:
    #   if _columns is not None, overrides dataframe header
    #   if _columns is None, dataframe header is kept
    # if not a dataframe:
    #   if _columns is not None, _columns is kept
    #   if _columns is None, None is returned


    # fixtures ** * ** * ** * ** * ** *
    @staticmethod
    @pytest.fixture(scope='module')
    def _cols():
        return 5


    @staticmethod
    @pytest.fixture(scope='module')
    def _X(_cols):
        return np.random.randint(0, 3, (20, _cols))


    @staticmethod
    @pytest.fixture(scope='module')
    def df_columns(_cols):
        return [str(uuid4())[:4] for _ in range(_cols)]


    @staticmethod
    @pytest.fixture(scope='module')
    def other_columns(_cols):
        return [str(uuid4())[:4] for _ in range(_cols)]

    # END fixtures ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    def test_returns_correct_columns_array(
        self, _X, _columns_is_passed, other_columns
    ):

        if _columns_is_passed:
            out = _header_handling_pre(_X, other_columns)
            assert np.array_equiv(out, other_columns)

        elif not _columns_is_passed:
            assert _header_handling_pre(_X, None) is None


    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    def test_returns_correct_columns_df(
        self, _X, _columns_is_passed, df_columns, other_columns
    ):

        _X_wip = pd.DataFrame(data=_X.copy(), columns=df_columns)

        if _columns_is_passed:
            out = _header_handling_pre(_X_wip, other_columns)
            assert np.array_equiv(out, other_columns)

        elif not _columns_is_passed:
            out = _header_handling_pre(_X_wip, None)
            assert np.array_equiv(out, df_columns)


    @pytest.mark.parametrize('_sparse_type',
        (
         'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix', 'lil_matrix',
         'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array', 'coo_array',
         'dia_array', 'lil_array', 'dok_array', 'bsr_array',
         )
    )
    @pytest.mark.parametrize('_columns_is_passed', (True, False))
    def test_returns_correct_columns_sparse(
            self, _X, _sparse_type, _columns_is_passed, other_columns
    ):

        if _sparse_type == 'csr_matrix':
            _X_wip = ss.csr_matrix(_X)
        elif _sparse_type == 'csc_matrix':
            _X_wip = ss.csc_matrix(_X)
        elif _sparse_type == 'coo_matrix':
            _X_wip = ss.coo_matrix(_X)
        elif _sparse_type == 'dia_matrix':
            _X_wip = ss.dia_matrix(_X)
        elif _sparse_type == 'lil_matrix':
            _X_wip = ss.lil_matrix(_X)
        elif _sparse_type == 'dok_matrix':
            _X_wip = ss.dok_matrix(_X)
        elif _sparse_type == 'bsr_matrix':
            _X_wip = ss.bsr_matrix(_X)
        elif _sparse_type == 'csr_array':
            _X_wip = ss.csr_array(_X)
        elif _sparse_type == 'csc_array':
            _X_wip = ss.csc_array(_X)
        elif _sparse_type == 'coo_array':
            _X_wip = ss.coo_array(_X)
        elif _sparse_type == 'dia_array':
            _X_wip = ss.dia_array(_X)
        elif _sparse_type == 'lil_array':
            _X_wip = ss.lil_array(_X)
        elif _sparse_type == 'dok_array':
            _X_wip = ss.dok_array(_X)
        elif _sparse_type == 'bsr_array':
            _X_wip = ss.bsr_array(_X)

        if _columns_is_passed:
            out = _header_handling_pre(_X_wip, other_columns)
            assert np.array_equiv(out, other_columns)

        elif not _columns_is_passed:
            assert _header_handling_pre(_X_wip, None) is None

























