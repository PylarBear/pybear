# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager._partial_fit._find_constants \
    import _find_constants

import numpy as np

import pytest






class TestFindConstants_Num:

    # def _find_constants(
    #     _X: DataFormatType,
    #     _old_constant_columns: dict[int, any],
    #     _equal_nan: bool,
    #     _rtol: Real,
    #     _atol: Real,
    #     _n_jobs: Union[Integral, None]
    # ) -> dict[int, any]:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @staticmethod
    @pytest.fixture(scope='module')
    def _rtol():
        return 1e-6


    @staticmethod
    @pytest.fixture(scope='module')
    def _atol():
        return 1e-6


    @staticmethod
    @pytest.fixture(scope='module')
    def _n_jobs():
        return 1   # leave this at 1 because of contention


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (100, 20)


    @staticmethod
    @pytest.fixture(scope='module')
    def _noise():
        return 1e-9


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns[:_shape[1]]


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_base(_X_factory, _columns, _shape, _noise):

        def foo(_format, _dtype, _has_nan, _constants):

            return _X_factory(
                _dupl=None,
                _has_nan=_has_nan,
                _format=_format,
                _dtype=_dtype,
                _columns=_columns,
                _constants=_constants,
                _noise=_noise,
                _zeros=None,
                _shape=_shape
            )

        return foo


    @staticmethod
    @pytest.fixture(scope='module')
    def _init_constants():
        return {1:2, 3:2, 8:2, 12:2, 15:1}


    @staticmethod
    @pytest.fixture(scope='module')
    def _more_constants():
        return {1:2, 3:2, 4:0, 8:2, 12:2, 15:1}


    @staticmethod
    @pytest.fixture(scope='module')
    def _no_constants():
        return {}


    @staticmethod
    @pytest.fixture(scope='module')
    def _less_constants():
        return {1:2, 3:2, 12:2, 15:1}

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **



    @pytest.mark.parametrize('_format', ('np', 'pd', 'csr', 'csc', 'lil'))
    @pytest.mark.parametrize('_dtype', ('flt', 'int'))
    @pytest.mark.parametrize('_constants_set', ('init', 'no', 'more', 'less'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_first_pass(
        self, _X_base, _format, _dtype, _constants_set, _has_nan, _equal_nan,
        _init_constants, _no_constants, _more_constants, _less_constants,
        _rtol, _atol, _shape, _n_jobs
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
            _constants = _no_constants
        elif _constants_set == 'more':
            _constants = _more_constants
        elif _constants_set == 'less':
            _constants = _less_constants
        else:
            raise Exception

        # build X
        _X_wip = _X_base(
            _format=_format,
            _dtype=_dtype,
            _has_nan=_has_nan,
            _constants=_constants
        )

        # get constant idxs and their values
        out: dict[int, any] = _find_constants(
            _X_wip,
            _old_constant_columns=None,   # first pass! occ must be None!
            _equal_nan=_equal_nan,
            _rtol=_rtol,
            _atol=_atol,
            _n_jobs=_n_jobs
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
                        _value,
                        _init_constants[_idx],
                        rtol=_rtol,
                        atol=_atol
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
                        _value,
                        _more_constants[_idx],
                        rtol=_rtol,
                        atol=_atol
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
                        _value,
                        _less_constants[_idx],
                        rtol=_rtol,
                        atol=_atol
                    )
        else:
            raise Exception


    @pytest.mark.parametrize('_format', ('np', 'pd', 'csr', 'csc', 'lil'))
    @pytest.mark.parametrize('_dtype', ('flt', 'int'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_less_constants_found(
        self, _X_base, _format, _dtype, _has_nan, _equal_nan, _init_constants,
        _less_constants, _rtol, _atol, _shape, _n_jobs
    ):

        # verifies accuracy of _find_constants when second partial fit
        # has less constants than the first

        # as of 24_12_16 _columns_getter only allows ss that are
        # indexable, dont test with coo, dia, bsr

        # build first X
        _first_X_wip = _X_base(
            _format=_format,
            _dtype=_dtype,
            _has_nan=_has_nan,
            _constants=_init_constants
        )

        # get first partial fit constants
        _first_fit_constants: dict[int, any] = _find_constants(
            _first_X_wip,
            _old_constant_columns=None,   # first pass! occ must be None!
            _equal_nan=_equal_nan,
            _rtol=_rtol,
            _atol=_atol,
            _n_jobs=_n_jobs
        )

        # build second X - less constant columns
        _scd_X_wip = _X_base(
            _format=_format,
            _dtype=_dtype,
            _has_nan=_has_nan,
            _constants=_less_constants
        )

        # get second partial fit constants
        _scd_fit_constants: dict[int, any] = _find_constants(
            _scd_X_wip,
            _old_constant_columns=_first_fit_constants, # <=========
            _equal_nan=_equal_nan,
            _rtol=_rtol,
            _atol=_atol,
            _n_jobs=_n_jobs
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
                        _value,
                        _less_constants[_col_idx],
                        rtol=_rtol,
                        atol=_atol
                    )


    @pytest.mark.parametrize('_format', ('np', 'pd', 'csr', 'csc', 'lil'))
    @pytest.mark.parametrize('_dtype', ('flt', 'int'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_more_constants_found(
        self, _X_base, _format, _dtype, _has_nan, _equal_nan, _init_constants,
        _more_constants, _rtol, _atol, _shape, _n_jobs
    ):

        # verifies accuracy of _find_constants when second partial fit
        # has more constants than the first

        # as of 24_12_16 _columns_getter only allows ss that are
        # indexable, dont test with coo, dia, bsr

        # build first X
        _first_X_wip = _X_base(
            _format=_format,
            _dtype=_dtype,
            _has_nan=_has_nan,
            _constants=_init_constants
        )

        # get first partial fit constants
        _first_fit_constants: dict[int, any] = _find_constants(
            _first_X_wip,
            _old_constant_columns=None,   # first pass! occ must be None
            _equal_nan=_equal_nan,
            _rtol=_rtol,
            _atol=_atol,
            _n_jobs=_n_jobs
        )

        # build second X - more constant columns
        _scd_X_wip = _X_base(
            _format=_format,
            _dtype=_dtype,
            _has_nan=_has_nan,
            _constants=_more_constants
        )

        # get second partial fit constants
        _scd_fit_constants: dict[int, any] = _find_constants(
            _scd_X_wip,
            _old_constant_columns=_first_fit_constants, # <=========
            _equal_nan=_equal_nan,
            _rtol=_rtol,
            _atol=_atol,
            _n_jobs=_n_jobs
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
                        _value,
                        _init_constants[_col_idx],
                        rtol=_rtol,
                        atol=_atol
                    )


    @pytest.mark.parametrize('_format', ('np', 'pd', 'csr', 'csc', 'lil'))
    @pytest.mark.parametrize('_dtype', ('flt', 'int'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_more_and_less_constants_found(
        self, _X_base, _format, _dtype, _has_nan, _equal_nan, _init_constants,
        _less_constants, _more_constants, _rtol, _atol, _shape, _n_jobs
    ):

        # verifies accuracy of _find_constants when partial fits after the
        # first have both more and less constants

        # as of 24_12_16 _columns_getter only allows ss that are
        # indexable, dont test with coo, dia, bsr

        # build first X
        _first_X_wip = _X_base(
            _format=_format,
            _dtype=_dtype,
            _has_nan=_has_nan,
            _constants=_init_constants
        )

        # get first partial fit constants
        _first_fit_constants: dict[int, any] = _find_constants(
            _first_X_wip,
            _old_constant_columns=None,  # first pass!  occ must be None!
            _equal_nan=_equal_nan,
            _rtol=_rtol,
            _atol=_atol,
            _n_jobs=_n_jobs
        )

        # build second X - more constant columns
        _scd_X_wip = _X_base(
            _format=_format,
            _dtype=_dtype,
            _has_nan=_has_nan,
            _constants=_more_constants
        )

        # get second partial fit constants
        _scd_fit_constants: dict[int, any] = _find_constants(
            _scd_X_wip,
            _old_constant_columns=_first_fit_constants,  # <=========
            _equal_nan=_equal_nan,
            _rtol=_rtol,
            _atol=_atol,
            _n_jobs=_n_jobs
        )

        # build third X - less constant columns
        _third_X_wip = _X_base(
            _format=_format,
            _dtype=_dtype,
            _has_nan=_has_nan,
            _constants=_less_constants
        )

        # get third partial fit constants
        _third_fit_constants: dict[int, any] = _find_constants(
            _third_X_wip,
            _old_constant_columns=_scd_fit_constants,  # <=========
            _equal_nan=_equal_nan,
            _rtol=_rtol,
            _atol=_atol,
            _n_jobs=_n_jobs
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
                        _value,
                        _less_constants[_col_idx],
                        rtol=_rtol,
                        atol=_atol
                    )


    @pytest.mark.parametrize('_format', ('np', 'pd', 'csr', 'csc', 'coo'))
    @pytest.mark.parametrize('_dtype', ('flt', 'int'))
    def test_ss_all_zeros(self, _format, _dtype, _shape, _n_jobs):

        # build X
        _X_wip = np.zeros(_shape).astype(np.uint8)

        # get constant idxs and their values
        out: dict[int, any] = _find_constants(
            _X_wip,
            _old_constant_columns=None,   # first pass! occ must be None!
            _equal_nan=True,
            _rtol=1e-5,
            _atol=1e-8,
            _n_jobs=_n_jobs
        )

        assert np.array_equal(list(out.keys()), list(range(_shape[1])))








