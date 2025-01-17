# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager._partial_fit. \
    _column_getter import _column_getter

from pybear.preprocessing.InterceptManager._partial_fit. \
    _parallel_constant_finder import _parallel_constant_finder

from pybear.preprocessing.InterceptManager._shared._set_attributes \
    import _set_attributes

from pybear.preprocessing.InterceptManager._transform._transform import \
    _transform

from pybear.utilities import nan_mask

import numpy as np
import pandas as pd
import scipy.sparse as ss
from copy import deepcopy

import pytest





@pytest.mark.parametrize('_dtype', ('flt', 'str'), scope='module')
@pytest.mark.parametrize('_has_nan',  (True, False), scope='module')
class TestTransform:

    # these tests prove that _transform:
    # - blocks anything that is not numpy ndarray, pandas dataframe, or
    #       ss csc matrix/array
    # - correctly removes columns based on _instructions
    # - format and dtype(s) of the transformed are same as passed
    # - For appended intercept:
    #   - if numpy and ss, constant value is correct and is forced to
    #       dtype of X
    #   - if pd, appended header is correct, and appended constant is
    #       correct dtype (float if num, otherwise object)


    # def _transform(
    #     _X: Union[npt.NDArray, pd.DataFrame, ss.csc_array, ss.csc_matrix],
    #     _instructions: InstructionType
    # ) -> Union[npt.NDArray, pd.DataFrame, ss.csc_array, ss.csc_matrix]:


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
    def _const_col_flt():
        # must be in range of shape
        # must match columns in _instructions!
        return {0:np.pi, 1:0, 3:1, 4: np.e, 8: -1}


    @staticmethod
    @pytest.fixture(scope='module')
    def _const_col_str():
        # must be in range of shape
        # must match columns in _instructions!
        return {0:'a', 1:'b', 3:'c', 4:'d', 8:'e'}


    @staticmethod
    @pytest.fixture(scope='module')
    def _base_X(
        _X_factory, _dtype, _has_nan, _shape, _const_col_flt, _const_col_str
    ):

        def foo(_noise:float):

            return _X_factory(
                _dupl=None,
                _has_nan=_has_nan,
                _format='np',
                _dtype=_dtype,
                _constants=_const_col_flt if _dtype=='flt' else _const_col_str,
                _noise=_noise,
                _zeros=None,
                _shape=_shape
            )

        return foo


    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('_equal_nan', (True, False), scope='function')
    @pytest.mark.parametrize('_instructions',
         (
             {'keep': [1, 3, 8], 'delete': [0, 4], 'add': None},
             {'keep': [0, 1, 4], 'delete': [3, 8], 'add': None},
             {'keep': [0, 1, 3, 4, 8], 'delete': None, 'add': {'Intercept': 1}},
             {'keep': [0, 1, 3, 4, 8], 'delete': None, 'add': None},
             {'keep': [1], 'delete': [0, 3, 4, 8], 'add': None}
         ), scope='function'
     )
    @pytest.mark.parametrize('_format',
        (
            'ndarray', 'df', 'csr_matrix', 'csc_matrix', 'coo_matrix',
            'dia_matrix', 'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array',
            'csc_array', 'coo_array', 'dia_array', 'lil_array', 'dok_array',
            'bsr_array'
        ), scope='function'
    )
    def test_output(
        self, _dtype, _base_X, _format, _shape, _master_columns, _equal_nan,
        _has_nan, _rtol_atol, _instructions, _const_col_str, _const_col_flt
    ):

        # Methodology:
        # even tho it is a fixture, _instructions is conditional based on the
        # test and sometimes is modified below.
        #
        # pass (the possibly modified) :fixture: _instructions to
        # _set_attributes() to build the expected column mask that would be
        # applied during transform.
        # pass X and _instructions to _transform.
        # the transformed X should have kept the columns as in the expected
        # column mask.
        # for np, pd, and ss, iterate over input X and output X simultaneously,
        # using the expected column mask to map columns in output X to their
        # original locations in input X.
        # Columns that are mapped to each other must be array_equal.
        # if they are, that means:
        # 1) that _transform() used _instructions['delete'] from
        # _make_instructions() to mask the same columns as _set_attributes()
        # did here.
        # 2) _transform correctly deleted the masked columns for np, pd, and ss
        # 3) Columns that are not mapped must be constant.

        if _dtype == 'str' and _format not in ['ndarray', 'df']:
            pytest.skip(reason=f"scipy sparse cant take strings")

        # must do this. even though _instructions is function scope when it is
        # modified by 'if _has_nan and not _equal_nan' below it isnt resetting.
        _wip_instr = deepcopy(_instructions)

        if _dtype == 'str':
            _constant_columns = _const_col_str
        elif _dtype == 'flt':
            _constant_columns = _const_col_flt
        else:
            raise Exception


        # if has_nan and not equal_nan, _base_X puts nans in every column,
        # therefore there can be no constant columns
        if _has_nan and not _equal_nan:
            _wip_instr['keep'] = None
            _wip_instr['delete'] = None
            _constant_columns = {}


        _X = _base_X(_noise=1e-9)


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


        # get a referee column mask from _set_attributes
        # _set_attributes builds a column mask based on the idxs that are
        # in _wip_instr['keep'] and _wip_instr['delete']
        _, _, _ref_column_mask = _set_attributes(
            _constant_columns,
            _wip_instr,
            _n_features_in = _shape[1]
        )
        del _
        if _constant_columns == {}:
            assert np.all(_ref_column_mask)

        # apply the instructions to X via _transform
        # everything except ndarray, pd dataframe, and csc are blocked!
        if _format in ['ndarray', 'df', 'csc_matrix', 'csc_array']:
            out = _transform(_X_wip, _wip_instr)
        else:
            with pytest.raises(TypeError):
                _transform(_X_wip, _wip_instr)
            pytest.skip(reason=f"cant do more tests after exception")


        # ASSERTIONS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # output format is same as given
        assert isinstance(out, _og_format)

        # out shape & _column_mask
        assert out.shape[1] == \
               sum(_ref_column_mask) + isinstance(_wip_instr['add'], dict)

        # output dtypes are same as given -----------------------------------
        if isinstance(out, pd.core.frame.DataFrame):

            # check the dtypes for the og columns in X first
            assert np.array_equal(
                out.dtypes[:_shape[1]],
                _og_dtype[_ref_column_mask]
            )

            # check header for og columns
            assert np.array_equal(
                out.columns[:_shape[1]],
                _master_columns.copy()[:_shape[1]][_ref_column_mask]
            )

            # check the dtype & header for the appended column separately
            if _wip_instr['add'] is not None: # should only be dict
                _key = list(_wip_instr['add'].keys())[0]
                _value = _wip_instr['add'][_key]
                try:
                    float(_value)
                    is_num = True
                except:
                    is_num = False

                # header
                assert _key in out

                # dtype
                if is_num:
                    assert out[_key].dtype == np.float64
                else:
                    assert out[_key].dtype == object

                del _key, _value, is_num

        elif _format == 'ndarray' and '<U' in str(_og_dtype):
            # str dtypes are changing in _transform() at
            # _X = np.hstack((
            #     _X,
            #     np.full((_X.shape[0], 1), _value)
            # ))
            # there does not seem to be an obvious connection between what
            # the dtype of _value is and the resultant dtype (for example,
            # _X with dtype '<U10' when appending float(1.0), the output dtype
            # is '<U21' (???, maybe floating point error on the float?) )
            assert '<U' in str(out.dtype)

            # check the values in the appended column (if appended)
            if _wip_instr['add'] is not None: # should only be dict
                _key = list(_wip_instr['add'].keys())[0]
                _value = _wip_instr['add'][_key]
                # the stacked column and the value in it takes the dtype
                # of the original X
                assert out[0, -1] == str(_value)

        else:
            # could be np or ss
            assert out.dtype == _og_dtype

            # check the values in the appended column (if appended)
            if _wip_instr['add'] is not None: # should only be dict
                _key = list(_wip_instr['add'].keys())[0]
                _value = _wip_instr['add'][_key]
                # the stacked column and the value in it takes the dtype
                # of the original X
                assert _column_getter(out, -1).ravel()[0] == _value
        # END output dtypes are same as given -------------------------------


        # iterate over input X and output X simultaneously, use _kept_idxs to
        # map columns in output X to their original locations in input X.
        # This ignores any intercept column that may have been appended,
        # which was already tested above.
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
                    # verify header matches
                    assert _X_wip.columns[_og_idx] == out.columns[_out_idx]

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
                # columns not in column mask must therefore be constant
                assert _parallel_constant_finder(
                    _og_col,
                    _equal_nan,
                    *_rtol_atol
                )

        # END ASSERTIONS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


class TestHStackDtypesOnNP:

    @pytest.mark.parametrize('_dtype', ('flt', 'str', 'obj'))
    @pytest.mark.parametrize('_value', (1, '1', 'a'))
    def test_various_dtypes_hstacked_to_np(self, _X_factory, _dtype, _value):

        # this tests / shows what happens to the X container dtype when
        # various types of values are appended to X from the 'keep' dictionary

        # when hstacking a str constant to a float array, numpy is
        # changing the array to '<U...'
        # otherwise, the stacked value assumes the existing dtype of X

        X = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='np',
            _dtype=_dtype,
            _constants=None,
            _noise=0,
            _zeros=None,
            _shape=(100, 10)
        )

        _instr = {
            'keep': None,
            'delete': None,
            'add': {'Intercept': _value}
        }

        out = _transform(X, _instr)

        if isinstance(_value, str) and _dtype == 'flt':
            assert '<U' in str(out.dtype)
        elif '<U' in str(X.dtype):
            assert '<U' in str(out.dtype)
        else:
            assert X.dtype == out.dtype




