# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager._transform._transform import \
    _transform

from pybear.preprocessing.InterceptManager._partial_fit._parallel_constant_finder \
    import _parallel_constant_finder

from pybear.preprocessing.InterceptManager._partial_fit._set_attributes \
    import _set_attributes

from pybear.preprocessing.InterceptManager._partial_fit._find_constants \
    import _find_constants


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
        {'keep':[0,1,3,4,8], 'delete':None, 'add': {'Intercept': 1}}
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
    def _base_constant_columns():
        # must be in range of shape
        # must match columns in _instructions!
        # need to actually build X to fill the values in constant_columns_
        return [0,1,3,4,8]

    @staticmethod
    @pytest.fixture(scope='module')
    def _base_X(_X_factory, _dtype, _has_nan, _shape, _dupl, _base_constant_columns):

        def foo(_constants:list[int], _noise:float):

            return _X_factory(
                _dupl=_dupl,
                _has_nan=_has_nan,
                _format='np',
                _dtype=_dtype,
                _constants=_base_constant_columns,
                _noise=_noise,
                _zeros=None,
                _shape=_shape
            )

        return foo


    # need to get actual constants out of X
    @staticmethod
    @pytest.fixture(scope='function')
    def _constant_columns(_base_X, _equal_nan, _rtol_atol):
        return _find_constants(
            _base_X,
            {},
            _equal_nan,
            _rtol_atol[0],
            _rtol_atol[1],
            _n_jobs=-1
        )



    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    @pytest.mark.parametrize('_format',
        (
            'ndarray', 'df', 'csr_matrix', 'csc_matrix', 'coo_matrix',
            'dia_matrix', 'lil_matrix', 'dok_matrix', 'csr_array',
            'csc_array', 'coo_array', 'dia_array', 'lil_array', 'dok_array'
        )
    )
    def test_output(
        self, _dtype, _base_X, _format, _shape, _master_columns, _equal_nan,
        _has_nan, _dupl, _rtol_atol, _instructions, _constant_columns
    ):

        if _dtype == 'str' and _format not in ['ndarray', 'df']:
            pytest.skip(reason=f"scipy sparse cant take strings")

        # data format conversion v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        if _format == 'ndarray':
            _X_wip = _base_X
        elif _format == 'df':
            _X_wip = pd.DataFrame(
                data=_base_X,
                columns=_master_columns.copy()[:_shape[1]]
            )
        elif _format == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_base_X)
        elif _format == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_base_X)
        elif _format == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_base_X)
        elif _format == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_base_X)
        elif _format == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_base_X)
        elif _format == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_base_X)
        elif _format == 'csr_array':
            _X_wip = ss._csr.csr_array(_base_X)
        elif _format == 'csc_array':
            _X_wip = ss._csc.csc_array(_base_X)
        elif _format == 'coo_array':
            _X_wip = ss._coo.coo_array(_base_X)
        elif _format == 'dia_array':
            _X_wip = ss._dia.dia_array(_base_X)
        elif _format == 'lil_array':
            _X_wip = ss._lil.lil_array(_base_X)
        elif _format == 'dok_array':
            _X_wip = ss._dok.dok_array(_base_X)
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

        # ASSERTIONS ** ** ** ** ** **
        # output format is same as given
        assert isinstance(out, _og_format)

        # output dtypes are same as given
        if isinstance(out, pd.core.frame.DataFrame):
            assert np.array_equal(out.dtypes, _og_dtype[_ref_column_mask])
        else:
            assert out.dtype == _og_dtype

        # out matches _column_mask
        assert out.shape[1] == sum(_ref_column_mask)

        _kept_idxs = np.arange(_shape[1])[_ref_column_mask]

        for _new_idx, _kept_idx in enumerate(_kept_idxs, 0):

            if isinstance(_X_wip, np.ndarray):
                _out_col = out[:, _new_idx]
                _og_col = _X_wip[:, _kept_idx]
            elif isinstance(_X_wip, pd.core.frame.DataFrame):
                _out_col = out.iloc[:, _new_idx]
                _og_col = _X_wip.iloc[:, _kept_idx]
            else:
                _out_col = out.tocsc()[:, [_new_idx]].toarray()
                _og_col = _X_wip.tocsc()[:, [_kept_idx]].toarray()



            if not _has_nan or (_has_nan and _equal_nan):
                assert _parallel_constant_finder(
                    _out_col, _og_col, *_rtol_atol, _equal_nan
                )
            else:
                assert not _parallel_constant_finder(
                    _out_col, _og_col, *_rtol_atol, _equal_nan
                )


            # END ASSERTIONS ** ** ** ** ** **







