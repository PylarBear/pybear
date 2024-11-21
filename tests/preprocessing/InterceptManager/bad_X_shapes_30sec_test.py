# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.InterceptManager.InterceptManager \
    import InterceptManager as IM

import pandas as pd
import scipy.sparse as ss

import pytest







# TEST FOR EXCEPTS ON BAD X SHAPES ON ARRAY & DF ##########################
class TestExceptsOnBadXShapes:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
       return (20, 10)


    @staticmethod
    @pytest.fixture(scope='function')
    def _kwargs():
        return {
            'keep': 'first',
            'equal_nan': False,
            'rtol': 1e-5,
            'atol': 1e-8,
            'n_jobs': 1  # leave this a 1 because of confliction... pizza verify!
        }



    # all of these to try to get some more speed... didnt help that much

    @staticmethod
    @pytest.fixture(scope='module')
    def _row_dict(_shape):
       return {
            'good': _shape[0],
            'less_row': _shape[0] // 2,
            'more_row': _shape[0] * 2
        }

    @staticmethod
    @pytest.fixture(scope='module')
    def _col_dict(_shape):
        return {
            'good':_shape[1],
            'less_col':_shape[1]//2,
            'more_col':_shape[1]*2
        }

    @staticmethod
    @pytest.fixture(scope='module')
    def _columns_dict(_master_columns, _col_dict):
        return {
            'good':_master_columns.copy()[:_col_dict['good']],
            'less_col':_master_columns.copy()[:_col_dict['less_col']],
            'more_col':_master_columns.copy()[:_col_dict['more_col']],
            None:None
        }

    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np_good_row_good_col(_X_factory, _shape, _row_dict, _col_dict):
        return _X_factory(
            _format='np', _dtype='flt',
            _shape=(_row_dict['good'], _col_dict['good'])
        )


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np_good_row_less_col(_X_factory, _shape, _row_dict, _col_dict):
        return _X_factory(
            _format='np', _dtype='flt',
            _shape=(_row_dict['good'], _col_dict['less_col'])
        )

    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np_good_row_more_col(_X_factory, _shape, _row_dict, _col_dict):
        return _X_factory(
            _format='np', _dtype='flt',
            _shape=(_row_dict['good'], _col_dict['more_col'])
        )

    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np_less_row_good_col(_X_factory, _shape, _row_dict, _col_dict):
        return _X_factory(
            _format='np', _dtype='flt',
            _shape=(_row_dict['less_row'], _col_dict['good'])
        )

    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np_less_row_less_col(_X_factory, _shape, _row_dict, _col_dict):
        return _X_factory(
            _format='np', _dtype='flt',
            _shape=(_row_dict['less_row'], _col_dict['less_col'])
        )

    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np_less_row_more_col(_X_factory, _shape, _row_dict, _col_dict):
        return _X_factory(
            _format='np', _dtype='flt',
            _shape=(_row_dict['less_row'], _col_dict['more_col'])
        )

    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np_more_row_good_col(_X_factory, _shape, _row_dict, _col_dict):
        return _X_factory(
            _format='np', _dtype='flt',
            _shape=(_row_dict['more_row'], _col_dict['good'])
        )

    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np_more_row_less_col(_X_factory, _shape, _row_dict, _col_dict):
        return _X_factory(
            _format='np', _dtype='flt',
            _shape=(_row_dict['more_row'], _col_dict['less_col'])
        )

    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np_more_row_more_col(_X_factory, _shape, _row_dict, _col_dict):
        return _X_factory(
            _format='np', _dtype='flt',
            _shape=(_row_dict['more_row'], _col_dict['more_col'])
        )
    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # this is intentional, save some time, do only one ss
    X_FORMAT = ['np', 'pd', 'csc'] #, 'csr', 'coo', 'dia', 'lil', 'dok', 'bsr']
    SAME_DIFF_COLUMNS = ['good', 'less_col', 'more_col']
    SAME_DIFF_ROWS = ['good', 'less_row', 'more_row']

    @pytest.mark.parametrize('fst_fit_x_format', X_FORMAT)
    @pytest.mark.parametrize('scd_fit_x_format', X_FORMAT)
    @pytest.mark.parametrize('scd_fit_x_cols', SAME_DIFF_COLUMNS)
    @pytest.mark.parametrize('scd_fit_x_rows', SAME_DIFF_ROWS)
    @pytest.mark.parametrize('trfm_x_format', X_FORMAT)
    @pytest.mark.parametrize('trfm_x_cols', SAME_DIFF_COLUMNS)
    @pytest.mark.parametrize('trfm_x_rows', SAME_DIFF_ROWS)
    def test_excepts_on_bad_x_shapes(self, _X_factory, _master_columns, _kwargs,
        fst_fit_x_format, scd_fit_x_format, scd_fit_x_cols, scd_fit_x_rows,
        trfm_x_format, trfm_x_cols, trfm_x_rows, _shape, _columns_dict,
        _X_np_good_row_good_col, _X_np_good_row_less_col, _X_np_good_row_more_col,
        _X_np_less_row_good_col, _X_np_less_row_less_col, _X_np_less_row_more_col,
        _X_np_more_row_good_col, _X_np_more_row_less_col, _X_np_more_row_more_col
    ):


        TestCls = IM(**_kwargs)

        # object selection ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        def _obj_selector(_format, _rows, _cols):
            if _rows == 'good':
                if _cols == 'good':
                    _ = _X_np_good_row_good_col
                elif _cols == 'less_col':
                    _ = _X_np_good_row_less_col
                elif _cols == 'more_col':
                    _ = _X_np_good_row_more_col
            elif _rows == 'less_row':
                if _cols == 'good':
                    _ = _X_np_less_row_good_col
                elif _cols == 'less_col':
                    _ = _X_np_less_row_less_col
                elif _cols == 'more_col':
                    _ = _X_np_less_row_more_col
            elif _rows == 'more_row':
                if _cols == 'good':
                    _ = _X_np_more_row_good_col
                elif _cols == 'less_col':
                    _ = _X_np_more_row_less_col
                elif _cols == 'more_col':
                    _ = _X_np_more_row_more_col
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
            _rows='good',
            _cols='good'
        )
        # end first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        scd_fit_X = _obj_selector(
            _format=scd_fit_x_format,
            _rows=scd_fit_x_rows,
            _cols=scd_fit_x_cols
        )
        # end second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

        # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        trfm_X = _obj_selector(
            _format=trfm_x_format,
            _rows=trfm_x_rows,
            _cols=trfm_x_cols
        )
        # end transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


        # sklearn_exception WHEN n_features_in_ != FIRST FIT n_features_in_
        # UNDER ALL CIRCUMSTANCES
        n_features_exception = 0
        n_features_exception += any(map(lambda __: \
           __ in ['more_col', 'less_col'], [scd_fit_x_cols, trfm_x_cols]
        ))

        if n_features_exception:
            # sklearn.base.BaseEstimator._validate_data handles this,
            # let it raise whatever
            with pytest.raises(Exception):
                TestCls.partial_fit(fst_fit_X)
                TestCls.partial_fit(scd_fit_X)
                TestCls.transform(trfm_X)
        else:
            # otherwise, shapes should be good
            TestCls.partial_fit(fst_fit_X)
            TestCls.partial_fit(scd_fit_X)
            TestCls.transform(trfm_X)


    del X_FORMAT, SAME_DIFF_COLUMNS, SAME_DIFF_ROWS

# END TEST FOR EXCEPTS ON BAD X SHAPES, ON ARRAY & DF ##############





