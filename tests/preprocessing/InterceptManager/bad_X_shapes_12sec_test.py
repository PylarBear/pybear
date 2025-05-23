# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.preprocessing._InterceptManager.InterceptManager \
    import InterceptManager as IM



# TEST FOR EXCEPTS ON BAD X SHAPES ON ARRAY & DF ##########################
class TestExceptsOnBadXShapes:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

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

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # this is intentional, save some time, they all have shape attr and are
    # handled by pybear.base.validate_data
    X_FORMAT = ['np', 'pl'] # , 'pd', 'csr', 'coo', 'dia', 'lil', 'dok', 'bsr']
    SAME_DIFF_COLUMNS = ['good', 'less_col', 'more_col']
    SAME_DIFF_ROWS = ['good', 'less_row', 'more_row']

    @pytest.mark.parametrize('fst_fit_x_format', X_FORMAT)
    @pytest.mark.parametrize('scd_fit_x_format', X_FORMAT)
    @pytest.mark.parametrize('scd_fit_x_cols', SAME_DIFF_COLUMNS)
    @pytest.mark.parametrize('scd_fit_x_rows', SAME_DIFF_ROWS)
    @pytest.mark.parametrize('trfm_x_format', X_FORMAT)
    @pytest.mark.parametrize('trfm_x_cols', SAME_DIFF_COLUMNS)
    @pytest.mark.parametrize('trfm_x_rows', SAME_DIFF_ROWS)
    def test_excepts_on_bad_x_shapes(self, _X_factory, _kwargs, _columns_dict,
        fst_fit_x_format, scd_fit_x_format, scd_fit_x_cols, scd_fit_x_rows,
        trfm_x_format, trfm_x_cols, trfm_x_rows, _shape, _row_dict, _col_dict
    ):


        TestCls = IM(**_kwargs)

        # first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        fst_fit_X = _X_factory(
            _format=fst_fit_x_format,
            _columns=_columns_dict['good'],
            _shape=(_row_dict['good'], _col_dict['good'])
        )
        # end first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        scd_fit_X = _X_factory(
            _format=scd_fit_x_format,
            _columns=_columns_dict[scd_fit_x_cols],
            _shape=(_row_dict[scd_fit_x_rows], _col_dict[scd_fit_x_cols])
        )
        # end second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

        # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        trfm_X = _X_factory(
            _format=trfm_x_format,
            _columns=_columns_dict[trfm_x_cols],
            _shape=(_row_dict[trfm_x_rows], _col_dict[trfm_x_cols])
        )
        # end transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


        # exception WHEN n_features_in_ != FIRST FIT n_features_in_
        # UNDER ALL CIRCUMSTANCES
        n_features_exception = 0
        n_features_exception += any(map(lambda __: \
           __ in ['more_col', 'less_col'], [scd_fit_x_cols, trfm_x_cols]
        ))

        if n_features_exception:
            # pybear validate_data() handles this
            with pytest.raises(ValueError):
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





