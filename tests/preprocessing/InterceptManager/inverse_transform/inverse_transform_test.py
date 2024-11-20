# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.InterceptManager._inverse_transform. \
    _inverse_transform import _inverse_transform

from pybear.preprocessing.InterceptManager.InterceptManager import \
    InterceptManager

from pybear.preprocessing.InterceptManager._partial_fit. \
    _parallel_constant_finder import _parallel_constant_finder

from pybear.utilities import nan_mask

import numpy as np
import pandas as pd
import scipy.sparse as ss

import pytest







class TestInverseTransform:

    # build an X with duplicates, use IM to manage the constant columns
    # under different parameters (IM transform() should be independently
    # validated), use inverse_transform to reconstruct back to the
    # original X.

    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (20, 10)


    @staticmethod
    @pytest.fixture(scope='module')
    def _rtol_atol():
        return (1e-5, 1e-8)


    @staticmethod
    @pytest.fixture(scope='module')
    def _dupl():
        return [
            [0, 9],
            [2, 4, 7]
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]


    @staticmethod
    @pytest.fixture(scope='function')
    def _const_X(_X_factory, _shape):

        def foo(_has_nan, _format, _dtype, _constants):

            return _X_factory(
                _dupl=None,
                _has_nan=_has_nan,
                _format=_format,
                _dtype=_dtype,
                _constants=_constants,
                _shape=_shape
            )

        return foo

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.skip(reason=f"pizza needs to run github actions!")
    @pytest.mark.parametrize('_dtype', ('flt', 'str'))
    @pytest.mark.parametrize('_keep', ('first', 'first', 'last', 'random', 'none')) # pizza {'Intercept': 1}, 1, 'good_string', 'non_const_string', 'good_callable', 'non_const_callable'), scope='module')
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    @pytest.mark.parametrize('_format',
        (
            'np', 'pd', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
            'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
            'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array'
        )
    )
    def test_accuracy(
        self, _const_X, _format, _keep, _equal_nan, _dtype, _has_nan, _shape,
        _columns, _rtol_atol
    ):

        # pizza revisit this last thing
        # Methodology: use _set_attributes() to build expected column mask
        # from :fixture: _instructions. (_instructions is conditional based
        # on the test and is modified below.) for np, pd, and ss, iterate over
        # input X and output X simultaneously, using the expected column
        # mask to map columns in input X to their locations in output X.
        # Columns that are mapped to each other must be array_equal.
        # Columns that are not mapped must be constant.

        if _dtype == 'str' and _format not in ('np', 'pd'):
            pytest.skip(reason=f"scipy sparse cannot take strings")

        _base_X = _const_X(
            _has_nan=_has_nan,
            _format='np',
            _dtype=_dtype,
            _constants={0:1, _shape[1]-1: 1} if _dtype in ('int', 'flt') else {0: 'a', _shape[1]-1:'b'}
        )

        if _format != 'pd' and isinstance(_keep, str) and \
                _keep not in ('first', 'last', 'random', 'none'):
            with pytest.raises(ValueError):
                InterceptManager(
                    keep=_keep,
                    rtol=1e-5,
                    atol=1e-8,
                    equal_nan=_equal_nan,
                    n_jobs=-1
                ).fit(_base_X)
            pytest.skip(reason=f"cannot pass keep as str if X is not pd df")
        elif _format == 'pd':
            if _keep == 'good_string':
                _keep = _columns[0]
            elif _keep == 'non_const_string':
                _keep = _columns[-1]
        # set callable later
        # elif _keep == 'good_callable':
        #     _keep = lambda x: 0
        # elif _keep == 'non_const_callable':
        #     _keep = lambda x: 1


        if _format == 'np':
            _X_wip = _base_X
        elif _format == 'pd':
            _X_wip = pd.DataFrame(
                data=_base_X,
                columns=_columns
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
        elif _format == 'bsr_matrix':
            _X_wip = ss._bsr.bsr_matrix(_base_X)
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
        elif _format == 'bsr_array':
            _X_wip = ss._bsr.bsr_array(_base_X)
        else:
            raise Exception

        _IM = InterceptManager(
            keep=_keep,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=_equal_nan,
            n_jobs=-1
        )

        # fit v v v v v v v v v v v v v v v v v v v v
        if _keep == 'non_const_string':
            with pytest.raises(ValueError):
                _IM.fit(_X_wip)
            pytest.skip(reason=f"cant do tests without fit")
        elif _keep == 'good_callable':
            _IM.set_params(keep=lambda x: 0)
            _IM.fit(_X_wip)
        elif _keep == 'non_const_callable':
            _IM.set_params(keep=lambda x: 1)
            with pytest.raises(ValueError):
                _IM.fit(_X_wip)
            pytest.skip(reason=f"cant do tests without fit")
        else:
            _IM.fit(_X_wip)
        # fit ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        # transform v v v v v v v v v v v v v v v v v v
        _dedupl_X = _IM.transform(_X_wip, copy=True)
        # transform ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        # inverse transform v v v v v v v v v v v v v v v
        out = _inverse_transform(
            X=_dedupl_X,
            _removed_columns=_IM.removed_columns_,
            _feature_names_in=_columns
        )
        # inverse transform ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^


        assert type(out) is type(_X_wip)

        assert out.shape == _X_wip.shape

        _ref_column_mask = _IM.column_mask_

        print(f'pizza test {_ref_column_mask=}')

        # iterate over the input X and output X simultaneously, use
        # _kept_idxs to map column in output X to their original locations
        # in input X.
        _kept_idxs = np.arange(len(_ref_column_mask))[_ref_column_mask]

        # pizza _out_idx = -1
        for _og_idx in range(_shape[1]):

            # pizza
            # if _og_idx in _kept_idxs:
            #     _out_idx += 1

            if isinstance(_X_wip, np.ndarray):
                _og_col = _X_wip[:, _og_idx]
                # pizza
                # if _og_idx in _kept_idxs:
                #     _out_col = out[:, _out_idx]
                _out_col = out[:, _og_idx]
            elif isinstance(_X_wip, pd.core.frame.DataFrame):
                _og_col = _X_wip.iloc[:, _og_idx].to_numpy()
                # pizza
                # if _og_idx in _kept_idxs:
                #     _out_col = out.iloc[:, _out_idx].to_numpy()
                _out_col = out.iloc[:, _og_idx].to_numpy()
            elif hasattr(_X_wip, 'toarray'):
                _og_col = _X_wip.tocsc()[:, [_og_idx]].toarray()
                # pizza
                # if _og_idx in _kept_idxs:
                #     _out_col = out.tocsc()[:, [_out_idx]].toarray()
                _out_col = out.tocsc()[:, [_og_idx]].toarray()
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






