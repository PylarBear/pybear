# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import pandas as pd
import polars as pl
import scipy.sparse as ss

from pybear.preprocessing._SlimPolyFeatures.SlimPolyFeatures \
    import SlimPolyFeatures as SlimPoly



# TEST FOR EXCEPTS ON BAD X SHAPES ON ARRAY & DF ##########################
class TestExceptsOnBadXShapes:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _columns_dict(_master_columns, _shape):
        return {
            'good':_master_columns.copy()[:_shape[1]],
            'less_col':_master_columns.copy()[:_shape[1]//2],
            'more_col':_master_columns.copy()[:_shape[1]*2],
            None:None
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np_good_col(_X_factory, _shape):
        return _X_factory(_format='np', _dtype='flt', _shape=(_shape[0], _shape[1]))


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np_less_col(_X_factory, _shape):
        return _X_factory(_format='np', _dtype='flt', _shape=(_shape[0], _shape[1]//2))


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np_more_col(_X_factory, _shape):
        return _X_factory(_format='np', _dtype='flt', _shape=(_shape[0], _shape[1]*2))

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # this is intentional, save some time, do only one ss
    X_FORMAT = ['np', 'pd', 'pl', 'csc'] #, 'csr', 'coo', 'dia', 'lil', 'dok', 'bsr']
    SAME_DIFF_COLUMNS = ['good', 'less_col', 'more_col']

    @pytest.mark.parametrize('fst_fit_x_format', X_FORMAT)
    @pytest.mark.parametrize('scd_fit_x_format', X_FORMAT)
    @pytest.mark.parametrize('scd_fit_x_cols', SAME_DIFF_COLUMNS)
    @pytest.mark.parametrize('trfm_x_format', X_FORMAT)
    @pytest.mark.parametrize('trfm_x_cols', SAME_DIFF_COLUMNS)
    def test_excepts_on_bad_x_shapes(self, _kwargs,
        fst_fit_x_format, scd_fit_x_format, scd_fit_x_cols, trfm_x_format,
        trfm_x_cols, _columns_dict, _X_np_good_col, _X_np_more_col,
        _X_np_less_col
    ):

        # pizza u changed this to X_factory in IM

        TestCls = SlimPoly(**_kwargs)

        # object selection ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        def _obj_selector(_format, _cols):

            if _cols == 'good':
                _X_wip = _X_np_good_col
            elif _cols == 'less_col':
                _X_wip = _X_np_less_col
            elif _cols == 'more_col':
                _X_wip = _X_np_more_col
            else:
                raise Exception

            if _format == 'np':
                pass
            elif _format == 'pd':
                _X_wip = pd.DataFrame(data=_X_wip, columns=_columns_dict[_cols])
            elif _format == 'pl':
                _X_wip = pl.from_numpy(data=_X_wip, schema=list(_columns_dict[_cols]))
            elif _format == 'csc':
                _X_wip = ss.csc_array(_X_wip)
            elif _format == 'csr':
                _X_wip = ss.csr_array(_X_wip)
            elif _format == 'coo':
                _X_wip = ss.coo_array(_X_wip)
            elif _format == 'lil':
                _X_wip = ss.lil_array(_X_wip)
            elif _format == 'dok':
                _X_wip = ss.dok_array(_X_wip)
            elif _format == 'dia':
                _X_wip = ss.dia_array(_X_wip)
            elif _format == 'bsr':
                _X_wip = ss.bsr_array(_X_wip)
            else:
                raise Exception

            return _X_wip
        # END object selection ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        fst_fit_X = _obj_selector(
            _format=fst_fit_x_format,
            _cols='good'
        )
        # end first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        scd_fit_X = _obj_selector(
            _format=scd_fit_x_format,
            _cols=scd_fit_x_cols
        )
        # end second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

        # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        trfm_X = _obj_selector(
            _format=trfm_x_format,
            _cols=trfm_x_cols
        )
        # end transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


        # exception WHEN n_features_in_ != FIRST FIT n_features_in_
        # UNDER ALL CIRCUMSTANCES
        n_features_exception = 0
        n_features_exception += any(map(lambda __: \
           __ in ['more_col', 'less_col'], [scd_fit_x_cols, trfm_x_cols]
        ))

        if n_features_exception:
            with pytest.raises(ValueError):
                TestCls.partial_fit(fst_fit_X)
                TestCls.partial_fit(scd_fit_X)
                TestCls.transform(trfm_X)
        else:
            # otherwise, shapes should be good
            TestCls.partial_fit(fst_fit_X)
            TestCls.partial_fit(scd_fit_X)
            TestCls.transform(trfm_X)


    del X_FORMAT, SAME_DIFF_COLUMNS

# END TEST FOR EXCEPTS ON BAD X SHAPES, ON ARRAY & DF ##############





