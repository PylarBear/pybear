# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager._transform._transform import \
    _transform

from pybear.preprocessing.InterceptManager._shared._set_attributes \
    import _set_attributes

from pybear.preprocessing.InterceptManager._partial_fit._parallel_constant_finder import \
    _parallel_constant_finder

from pybear.utilities import nan_mask

import numpy as np
import pandas as pd
import scipy.sparse as ss

import pytest





@pytest.mark.parametrize('_dtype', ('flt', 'str'), scope='module')
@pytest.mark.parametrize('_has_nan', (True, False), scope='module')
@pytest.mark.parametrize('_equal_nan', (True, False), scope='module')
@pytest.mark.parametrize('_instructions',
    (
        {'keep':[1,3,8], 'delete':[0, 4], 'add':None},
        {'keep':[0,1,4], 'delete':[3, 8], 'add':None},
        {'keep':[0,1,3,4,8], 'delete':None, 'add': {'Intercept': 1}},
        {'keep':[0,1,3,4,8], 'delete':None, 'add': None}
    )
)
class TestTransform:

    # def _transform(
    #     _X: DataType,
    #     _instructions: InstructionType
    # ) -> DataType:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (100, 20)


    @staticmethod
    @pytest.fixture(scope='module')
    def _rtol_atol():
        return (1e-5, 1e-8)


    @staticmethod
    @pytest.fixture(scope='module')
    def _dupl():
        return []


    @staticmethod
    @pytest.fixture(scope='module')
    def _constant_columns_flt():
        # must be in range of shape
        # must match columns in _instructions!
        return {0:np.pi, 1:0, 3:1, 4: np.e, 8: -1}


    @staticmethod
    @pytest.fixture(scope='module')
    def _constant_columns_str():
        # must be in range of shape
        # must match columns in _instructions!
        return {0:'a', 1:'b', 3:'c', 4:'d', 8:'e'}


    @staticmethod
    @pytest.fixture(scope='module')
    def _base_X(_X_factory, _dtype, _has_nan, _shape, _dupl, _constant_columns_flt,
        _constant_columns_str
    ):

        def foo(_constants:dict[int, any], _noise:float):

            return _X_factory(
                _dupl=_dupl,
                _has_nan=_has_nan,
                _format='np',
                _dtype=_dtype,
                _constants=_constant_columns_flt if _dtype=='flt' else _constant_columns_str,
                _noise=_noise,
                _zeros=None,
                _shape=_shape
            )

        return foo


    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    @pytest.mark.parametrize('_format',
        (
            'ndarray', 'df', 'csr_matrix', 'csc_matrix', 'coo_matrix',
            'dia_matrix', 'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array',
            'csc_array', 'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array'
        )
    )
    def test_output(
        self, _dtype, _base_X, _format, _shape, _master_columns, _equal_nan,
        _has_nan, _dupl, _rtol_atol, _instructions, _constant_columns_str, _constant_columns_flt
    ):

        # Methodology: use _set_attributes() to build expected column mask
        # from :fixture: _instructions. (_instructions is conditional based
        # on the test and is modified below.) for np, pd, and ss, iterate over
        # input X and output X simultaneously, using the expected column
        # mask to map columns in output X to their original locations
        # in input X.
        # Columns that are mapped to each other must be array_equal.
        # if they are, that means:
        # 1) that _transform() used '_instructions' from _make_instructions()
        # to mask the same columns as _set_attributes() did here.
        # 2) _transform correctly deleted the masked columns for np, pd, and ss
        # Columns that are not mapped must be constant.

        if _dtype == 'str' and _format not in ['ndarray', 'df']:
            pytest.skip(reason=f"scipy sparse cant take strings")

        if _dtype == 'str':
            _constant_columns = _constant_columns_str
        elif _dtype == 'flt':
            _constant_columns = _constant_columns_flt
        else:
            raise Exception

        # if has_nan and not equal_nan, _base_X puts nans in every column,
        # therefore there can be no constant columns
        if _has_nan and not _equal_nan:
            _instructions['keep'] = list(_constant_columns)
            _instructions['delete'] = None
            _constant_columns = {}


        _X = _base_X(_constants=_constant_columns, _noise=1e-9)

        # data format conversion v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        if _format == 'ndarray':
            _X_wip = _X
        elif _format == 'df':
            _X_wip = pd.DataFrame(
                data=_X,
                columns=_master_columns.copy()[:_shape[1]]
            )
        elif _format == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_X)
        elif _format == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_X)
        elif _format == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_X)
        elif _format == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_X)
        elif _format == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_X)
        elif _format == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_X)
        elif _format == 'bsr_matrix':
            _X_wip = ss._bsr.bsr_matrix(_X)
        elif _format == 'csr_array':
            _X_wip = ss._csr.csr_array(_X)
        elif _format == 'csc_array':
            _X_wip = ss._csc.csc_array(_X)
        elif _format == 'coo_array':
            _X_wip = ss._coo.coo_array(_X)
        elif _format == 'dia_array':
            _X_wip = ss._dia.dia_array(_X)
        elif _format == 'lil_array':
            _X_wip = ss._lil.lil_array(_X)
        elif _format == 'dok_array':
            _X_wip = ss._dok.dok_array(_X)
        elif _format == 'bsr_array':
            _X_wip = ss._bsr.bsr_array(_X)
        else:
            raise Exception
        # END data format conversion v^v^v^v^v^v^v^v^v^v^v^v^v^v^


        # retain the original dtype(s)
        _og_format = type(_X_wip)
        if isinstance(_X_wip, pd.core.frame.DataFrame):
            _og_dtype = _X_wip.dtypes
        else:
            _og_dtype = _X_wip.dtype

        # retain the original num columns
        _og_cols = _X_wip.shape[1]

        # apply the instructions to the original X
        out = _transform(_X_wip, _instructions)

        # get a referee column mask from _set_attributes
        _, _, _ref_column_mask = _set_attributes(
            _constant_columns,
            _instructions,
            _n_features = _shape[1]
        )
        del _

        # ASSERTIONS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        # output format is same as given
        assert isinstance(out, _og_format)

        # output dtypes are same as given
        if isinstance(out, pd.core.frame.DataFrame):
            # pizza, this chops any appended column and doesnt check dtype, think
            # on whether to add that
            assert np.array_equal(
                out.dtypes[:_shape[1]],
                _og_dtype[_ref_column_mask]
            )
        elif '<U' in str(_og_dtype):
            # pizza come back to this, str dtypes are being changed somewhere,
            # find where
            assert '<U' in str(out.dtype)
        else:
            assert out.dtype == _og_dtype

        # out shape & _column_mask
        assert out.shape[1] == sum(_ref_column_mask) + isinstance(_instructions['add'], dict)


        # iterate over the input X and output X simultaneously, use
        # _kept_idxs to map column in output X to their original locations
        # in input X.
        _kept_idxs = np.arange(len(_ref_column_mask))[_ref_column_mask]

        _out_idx = -1
        for _og_idx in range(_shape[1]):

            if _og_idx in _kept_idxs:
                _out_idx += 1

            if isinstance(_X_wip, np.ndarray):
                _og_col = _X_wip[:, _og_idx]
                if _og_idx in _kept_idxs:
                    _out_col = out[:, _out_idx]
            elif isinstance(_X_wip, pd.core.frame.DataFrame):
                _og_col = _X_wip.iloc[:, _og_idx].to_numpy()
                if _og_idx in _kept_idxs:
                    _out_col = out.iloc[:, _out_idx].to_numpy()
            elif hasattr(_X_wip, 'toarray'):
                _og_col = _X_wip.tocsc()[:, [_og_idx]].toarray()
                if _og_idx in _kept_idxs:
                    _out_col = out.tocsc()[:, [_out_idx]].toarray()
            else:
                raise Exception


            if _og_idx in _kept_idxs:
                # then both _og_col and _out_col exist
                # the columns must be array_equal
                assert np.array_equal(
                    _out_col[np.logical_not(nan_mask(_out_col))],
                    _og_col[np.logical_not(nan_mask(_og_col))]
                )
            else:
                # columns that are not in column mask must therefore be constant
                assert _parallel_constant_finder(_og_col, _equal_nan, *_rtol_atol)

        # END ASSERTIONS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **







