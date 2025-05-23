# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from typing_extensions import Any

import numpy as np

from pybear.preprocessing._InterceptManager._partial_fit._find_constants \
    import _find_constants



class TestFindConstants_Num:


    # def _find_constants(
    #     _X: InternalDataContainer,
    #     _old_constant_columns: dict[int, any],
    #     _equal_nan: bool,
    #     _rtol: Real,
    #     _atol: Real,
    #     _n_jobs: Union[Integral, None]
    # ) -> dict[int, Any]:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture(scope='module')
    def _fc_args():
        return {
            '_rtol': 1e-6,
            '_atol': 1e-6,
            '_n_jobs': 1   # leave this at 1 because of contention
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _init_constants():
        return {1:2, 2:2, 4:2, 6:2, 7:1}


    @staticmethod
    @pytest.fixture(scope='module')
    def _more_constants():
        return {1:2, 2:2, 3:0, 4:2, 6:2, 7:1}


    @staticmethod
    @pytest.fixture(scope='module')
    def _less_constants():
        return {1:2, 2:2, 6:2, 7:1}

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    @pytest.mark.parametrize('_format', 
        ('coo_array', 'dia_array', 'bsr_array',
         'dok_array', 'lil_array', 'csr_array')
    )
    def test_blocks_not_csc(
        self, _X_factory, _format, _fc_args, _columns, _shape
    ):

        _X_wip = _X_factory(
            _format=_format, _dtype='flt', _columns=_columns,
            _has_nan=False, _constants=None, _shape=_shape
        )


        # this is raised by scipy let it raise whatever
        with pytest.raises(Exception):
            _find_constants(
                _X_wip,
                _old_constant_columns=None,
                _equal_nan=True,
                **_fc_args
            )


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'csc_array'))
    @pytest.mark.parametrize('_dtype', ('flt', 'int'))
    @pytest.mark.parametrize('_constants_set', ('init', 'no', 'more', 'less'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_first_pass(
        self, _X_factory, _format, _dtype, _constants_set, _has_nan, _equal_nan,
        _init_constants, _more_constants, _less_constants, _fc_args, _columns,
        _shape
    ):

        # verifies accuracy of _find_constants on a single pass

        # as of 24_12_16 _columns_getter only allows ss that are
        # indexable, dont test with coo, dia, bsr

        # using these just to run more tests, the fact that they are
        # 'more' or less compared to each other is not important, theyre
        # the fixtures available
        if _constants_set == 'init':
            _constants = _init_constants
        elif _constants_set == 'no':
            _constants = {}
        elif _constants_set == 'more':
            _constants = _more_constants
        elif _constants_set == 'less':
            _constants = _less_constants
        else:
            raise Exception

        # build X
        _X_wip = _X_factory(
            _format=_format, _dtype=_dtype, _columns=_columns,
            _has_nan=_has_nan, _constants=_constants, _shape=_shape
        )

        # get constant idxs and their values
        out: dict[int, Any] = _find_constants(
            _X_wip,
            _old_constant_columns=None,   # first pass! occ must be None!
            _equal_nan=_equal_nan,
            **_fc_args
        )

        # on first pass, the output of _find_constants is returned directly.
        # assert found constants indices and values vs expected are the same
        if (not _equal_nan and _has_nan) or _constants_set == 'no':
            # with no constants, or not _equal_nan, there can be no constants
            assert out == {}
        elif _constants_set == 'init':
            # num out constant columns == num given constant columns
            assert len(out) == len(_init_constants)
            # out constant column idxs == given constant column idxs
            assert np.array_equal(
                sorted(list(out)),
                sorted(list(_init_constants))
            )
            # out constant column values == given constant column values
            for _idx, _value in out.items():
                if str(_value) == 'nan':
                    assert str(_init_constants[_idx]) == 'nan'
                else:
                    assert np.isclose(
                        _value, _init_constants[_idx], rtol=1e-6, atol=1e-6
                    )
        elif _constants_set == 'more':
            # num out constant columns == num given constant columns
            assert len(out) == len(_more_constants)
            # out constant column idxs == given constant column idxs
            assert np.array_equal(
                sorted(list(out)),
                sorted(list(_more_constants))
            )
            # out constant column values == given constant column values
            for _idx, _value in out.items():
                if str(_value) == 'nan':
                    assert str(_more_constants[_idx]) == 'nan'
                else:
                    assert np.isclose(
                        _value, _more_constants[_idx], rtol=1e-6, atol=1e-6
                    )
        elif _constants_set == 'less':
            # num out constant columns == num given constant columns
            assert len(out) == len(_less_constants)
            # out constant column idxs == given constant column idxs
            assert np.array_equal(
                sorted(list(out)),
                sorted(list(_less_constants))
            )
            # out constant column values == given constant column values
            for _idx, _value in out.items():
                if str(_value) == 'nan':
                    assert str(_less_constants[_idx]) == 'nan'
                else:
                    assert np.isclose(
                        _value, _less_constants[_idx], rtol=1e-6, atol=1e-6
                    )
        else:
            raise Exception


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'csc_array'))
    @pytest.mark.parametrize('_dtype', ('flt', 'int'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_less_constants_found(
        self, _X_factory, _format, _dtype, _has_nan, _equal_nan, _init_constants,
        _less_constants, _fc_args, _columns, _shape
    ):

        # verifies accuracy of _find_constants when second partial fit
        # has less constants than the first

        # as of 24_12_16 _columns_getter only allows ss that are
        # indexable, dont test with coo, dia, bsr

        # build first X
        _first_X_wip = _X_factory(
            _format=_format, _dtype=_dtype, _columns=_columns,
            _has_nan=_has_nan, _constants=_init_constants, _shape=_shape
        )

        # get first partial fit constants
        _first_fit_constants: dict[int, Any] = _find_constants(
            _first_X_wip,
            _old_constant_columns=None,   # first pass! occ must be None!
            _equal_nan=_equal_nan,
            **_fc_args
        )

        # build second X - less constant columns
        _scd_X_wip = _X_factory(
            _format=_format, _dtype=_dtype, _columns=_columns,
            _has_nan=_has_nan, _constants=_less_constants, _shape=_shape
        )

        # get second partial fit constants
        _scd_fit_constants: dict[int, Any] = _find_constants(
            _scd_X_wip,
            _old_constant_columns=_first_fit_constants, # <=========
            _equal_nan=_equal_nan,
            **_fc_args
        )

        # on a partial fit where less duplicates are found, outputted melded
        # duplicates should reflect the lesser columns
        if _has_nan and not _equal_nan:
            assert _scd_fit_constants == {}
        else:
            assert np.array_equal(
                sorted(list(_scd_fit_constants)),
                sorted(list(_less_constants))
            )
            for _col_idx, _value in _scd_fit_constants.items():
                if str(_value) == 'nan':
                    assert str(_less_constants[_col_idx]) == 'nan'
                else:
                    assert np.isclose(
                        _value, _less_constants[_col_idx], rtol=1e-6, atol=1e-6
                    )


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'csc_array'))
    @pytest.mark.parametrize('_dtype', ('flt', 'int'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_more_constants_found(
        self, _X_factory, _format, _dtype, _has_nan, _equal_nan, _init_constants,
        _more_constants, _fc_args, _columns, _shape
    ):

        # verifies accuracy of _find_constants when second partial fit
        # has more constants than the first

        # as of 24_12_16 _columns_getter only allows ss that are
        # indexable, dont test with coo, dia, bsr

        # build first X
        _first_X_wip = _X_factory(
            _format=_format, _dtype=_dtype, _columns=_columns,
            _has_nan=_has_nan, _constants=_init_constants, _shape=_shape
        )

        # get first partial fit constants
        _first_fit_constants: dict[int, Any] = _find_constants(
            _first_X_wip,
            _old_constant_columns=None,   # first pass! occ must be None
            _equal_nan=_equal_nan,
            **_fc_args
        )

        # build second X - more constant columns
        _scd_X_wip = _X_factory(
            _format=_format, _dtype=_dtype, _columns=_columns,
            _has_nan=_has_nan, _constants=_more_constants, _shape=_shape
        )

        # get second partial fit constants
        _scd_fit_constants: dict[int, Any] = _find_constants(
            _scd_X_wip,
            _old_constant_columns=_first_fit_constants, # <=========
            _equal_nan=_equal_nan,
            **_fc_args
        )

        # on a partial fit where more duplicates are found, outputted melded
        # duplicates should not add the newly found columns
        if _has_nan and not _equal_nan:
            assert _scd_fit_constants == {}
        else:
            assert np.array_equal(
                sorted(list(_scd_fit_constants)),
                sorted(list(_init_constants))
            )
            for _col_idx, _value in _scd_fit_constants.items():
                if str(_value) == 'nan':
                    assert str(_init_constants[_col_idx]) == 'nan'
                else:
                    assert np.isclose(
                        _value, _init_constants[_col_idx], rtol=1e-6, atol=1e-6
                    )


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'csc_array'))
    @pytest.mark.parametrize('_dtype', ('flt', 'int'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_more_and_less_constants_found(
        self, _X_factory, _format, _dtype, _has_nan, _equal_nan, _init_constants,
        _less_constants, _more_constants, _fc_args, _columns, _shape
    ):

        # verifies accuracy of _find_constants when partial fits after the
        # first have both more and less constants

        # as of 24_12_16 _columns_getter only allows ss that are
        # indexable, dont test with coo, dia, bsr

        # build first X
        _first_X_wip = _X_factory(
            _format=_format, _dtype=_dtype, _columns=_columns,
            _has_nan=_has_nan, _constants=_init_constants, _shape=_shape
        )

        # get first partial fit constants
        _first_fit_constants: dict[int, Any] = _find_constants(
            _first_X_wip,
            _old_constant_columns=None,  # first pass!  occ must be None!
            _equal_nan=_equal_nan,
            **_fc_args
        )

        # build second X - more constant columns
        _scd_X_wip = _X_factory(
            _format=_format, _dtype=_dtype, _columns=_columns,
            _has_nan=_has_nan, _constants=_more_constants, _shape=_shape
        )

        # get second partial fit constants
        _scd_fit_constants: dict[int, Any] = _find_constants(
            _scd_X_wip,
            _old_constant_columns=_first_fit_constants,  # <=========
            _equal_nan=_equal_nan,
            **_fc_args
        )

        # build third X - less constant columns
        _third_X_wip = _X_factory(
            _format=_format, _dtype=_dtype, _columns=_columns,
            _has_nan=_has_nan, _constants=_less_constants, _shape=_shape
        )

        # get third partial fit constants
        _third_fit_constants: dict[int, Any] = _find_constants(
            _third_X_wip,
            _old_constant_columns=_scd_fit_constants,  # <=========
            _equal_nan=_equal_nan,
            **_fc_args
        )

        # on a partial fit where more duplicates are found, outputted melded
        # duplicates should not add the newly found columns
        # on a partial fit where less duplicates are found, outputted melded
        # duplicates should reflect the lesser columns
        # the net effect should be that final output is the lesser columns
        if _has_nan and not _equal_nan:
            assert _third_fit_constants == {}
        else:
            assert np.array_equal(
                sorted(list(_third_fit_constants)),
                sorted(list(_less_constants))
            )
            for _col_idx, _value in _third_fit_constants.items():
                if str(_value) == 'nan':
                    assert str(_less_constants[_col_idx]) == 'nan'
                else:
                    assert np.isclose(
                        _value, _less_constants[_col_idx], rtol=1e-6, atol=1e-6
                    )


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'csc_array'))
    @pytest.mark.parametrize('_dtype', ('flt', 'int'))
    def test_ss_all_zeros(self, _format, _dtype, _shape, _fc_args):

        # build X
        _X_wip = np.zeros(_shape).astype(np.uint8)

        # get constant idxs and their values
        out: dict[int, Any] = _find_constants(
            _X_wip,
            _old_constant_columns=None,   # first pass! occ must be None!
            _equal_nan=True,
            **_fc_args
        )

        assert np.array_equal(list(out.keys()), list(range(_shape[1])))








