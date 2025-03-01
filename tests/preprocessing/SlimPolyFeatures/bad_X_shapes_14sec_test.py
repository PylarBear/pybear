# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing._SlimPolyFeatures.SlimPolyFeatures \
    import SlimPolyFeatures as SlimPoly

import pandas as pd
import scipy.sparse as ss

import pytest






# TEST FOR EXCEPTS ON BAD X SHAPES ON ARRAY & DF ##########################
class TestExceptsOnBadXShapes:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
       return (10, 4)


    @staticmethod
    @pytest.fixture(scope='function')
    def _kwargs():
        return {
            'degree': 2,
            'min_degree': 1,
            'scan_X': False,
            'keep': 'first',
            'interaction_only': True,
            'sparse_output': False,
            'feature_name_combiner': "as_indices",
            'equal_nan': True,
            'rtol': 1e-5,
            'atol': 1e-8,
            'n_jobs': 1    # leave this at 1 because of confliction
        }


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
        return _X_factory(
            _format='np', _dtype='flt',
            _shape=(_shape[0], _shape[1])
        )


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np_less_col(_X_factory, _shape):
        return _X_factory(
            _format='np', _dtype='flt',
            _shape=(_shape[0], _shape[1]//2)
        )

    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np_more_col(_X_factory, _shape):
        return _X_factory(
            _format='np', _dtype='flt',
            _shape=(_shape[0], _shape[1]*2)
        )

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # this is intentional, save some time, do only one ss
    X_FORMAT = ['np', 'pd', 'csc'] #, 'csr', 'coo', 'dia', 'lil', 'dok', 'bsr']
    SAME_DIFF_COLUMNS = ['good', 'less_col', 'more_col']

    @pytest.mark.parametrize('fst_fit_x_format', X_FORMAT)
    @pytest.mark.parametrize('scd_fit_x_format', X_FORMAT)
    @pytest.mark.parametrize('scd_fit_x_cols', SAME_DIFF_COLUMNS)
    @pytest.mark.parametrize('trfm_x_format', X_FORMAT)
    @pytest.mark.parametrize('trfm_x_cols', SAME_DIFF_COLUMNS)
    def test_excepts_on_bad_x_shapes(self, _X_factory, _master_columns, _kwargs,
        fst_fit_x_format, scd_fit_x_format, scd_fit_x_cols, trfm_x_format,
        trfm_x_cols, _shape, _columns_dict, _X_np_good_col, _X_np_more_col,
        _X_np_less_col
    ):


        TestCls = SlimPoly(**_kwargs)

        # object selection ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        def _obj_selector(_format, _cols):

            if _cols == 'good':
                _ = _X_np_good_col
            elif _cols == 'less_col':
                _ = _X_np_less_col
            elif _cols == 'more_col':
                _ = _X_np_more_col
            else:
                raise Exception

            if _format == 'np':
                pass
            elif _format == 'pd':
                _ = pd.DataFrame(data=_, columns=_columns_dict[_cols])
            elif _format == 'csc':
                _ = ss.csc_array(_)
            elif _format == 'csr':
                _ = ss.csr_array(_)
            elif _format == 'coo':
                _ = ss.coo_array(_)
            elif _format == 'lil':
                _ = ss.lil_array(_)
            elif _format == 'dok':
                _ = ss.dok_array(_)
            elif _format == 'dia':
                _ = ss.dia_array(_)
            elif _format == 'bsr':
                _ = ss.bsr_array(_)
            else:
                raise Exception
            return _
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





