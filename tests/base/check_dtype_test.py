# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.base._check_dtype import check_dtype

import numpy as np

import pytest




class TestCheckDtype:


    # def check_dtype(
    #     X,
    #     allowed: Literal['numeric', 'any']='any'
    # ) -> None:

    # validation * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    @pytest.mark.parametrize('junk_allowed',
        (-2.7, -1, 0, 1, 2.7, None, [0,1], (0,1), {'A':1}, lambda x: x)
    )
    def test_blocks_junk_allowed(self, junk_allowed):

        with pytest.raises(TypeError):
            check_dtype(
                np.random.randint(0, 10, (8,5)),
                allowed=junk_allowed
            )


    @pytest.mark.parametrize('bad_allowed',
        ('junk', 'trash', 'garbage', 'waste', 'rubbish')
    )
    def test_blocks_junk_allowed(self, bad_allowed):

        with pytest.raises(ValueError):
            check_dtype(
                np.random.randint(0, 10, (8,5)),
                allowed=bad_allowed
            )


    @pytest.mark.parametrize('good_allowed', ('numeric', 'any'))
    def test_blocks_junk_allowed(self, good_allowed):

        out = check_dtype(
            np.random.randint(0, 10, (8,5)),
            allowed=good_allowed
        )

        assert out is None

    # END validation * * * * * * * * * * * * * * * * * * * * * * * * * *



    @pytest.mark.parametrize('_allowed', ('numeric', 'any'))
    @pytest.mark.parametrize('_format',
        ('np', 'pd_df', 'pd_series', 'csr', 'csc', 'coo')
    )
    @pytest.mark.parametrize('_dtype', ('flt', 'int', 'str', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    def test_accuracy_numeric(
        self, _X_factory, _allowed, _format, _dtype, _has_nan
    ):

        # skip impossible conditions -- -- -- -- -- -- -- -- -- -- -- --

        if _format in ['csr', 'csc', 'coo'] and \
                _dtype in ['str', 'obj', 'hybrid']:
            pytest.skip(reason=f"impossible condition")

        # cant be hybrid (by the conftest meaning of hybrid) for pd_series
        if _format == 'pd_series' and _dtype == 'hybrid':
            pytest.skip(reason=f"impossible condition")

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # build X -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _X_wip = _X_factory(
            _dupl=None,
            _constants=None,
            _format=_format if 'pd' not in _format else 'pd',
            _zeros=None,
            _columns=None,
            _has_nan=_has_nan,
            _shape=(20,10),
            _dtype=_dtype
        )

        if _format == 'pd_series':
            _X_wip = _X_wip.iloc[:, 0].squeeze()
        # END build X -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        if _allowed == 'any':

            out = check_dtype(
                _X_wip,
                allowed=_allowed
            )

            assert out is None

        elif _allowed == 'numeric':

            if _dtype in ['int', 'flt']:
                out = check_dtype(
                    _X_wip,
                    allowed=_allowed
                )

                assert out is None

            elif _dtype in ['str', 'obj', 'hybrid']:

                with pytest.raises(ValueError):
                    check_dtype(
                        _X_wip,
                        allowed=_allowed
                    )

        else:
            raise Exception


















