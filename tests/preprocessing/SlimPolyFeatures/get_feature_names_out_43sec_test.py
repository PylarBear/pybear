# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing._SlimPolyFeatures.SlimPolyFeatures import \
    SlimPolyFeatures as SlimPoly

import itertools
import numpy as np
from pybear.base.exceptions import NotFittedError

import pytest



class TestAlwaysExceptsBeforeFit:


    @pytest.mark.parametrize('_keep', ('first', 'last', 'random', 'none'))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_always_except_before_fit(self, _keep, _equal_nan):

        with pytest.raises(NotFittedError):
            SlimPoly().get_feature_names_out()


@pytest.mark.parametrize('_format', ('np', 'pd'), scope='module')
@pytest.mark.parametrize('_instance_state',
    ('after_fit', 'after_transform'), scope='module'
)
class TestInputFeaturesRejects:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (8, 4)


    @staticmethod
    @pytest.fixture(scope='module')
    def _X(_X_factory, _format, _master_columns, _shape):

        return _X_factory(
            _dupl=None,
            _format=_format,
            _dtype='flt',
            _columns=_master_columns.copy()[:_shape[1]] if _format == 'pd' else None,
            _constants=None,
            _shape=_shape
        )


    @staticmethod
    @pytest.fixture(scope='function')
    def _TestCls(_instance_state, _X):

        _TestCls = SlimPoly()

        if _instance_state == 'after_fit':
            _TestCls.fit(_X)
            return _TestCls
        elif _instance_state == 'after_transform':
            _TestCls.fit_transform(_X)
            return _TestCls
        else:
            raise Exception

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('junk_input_features',
        (float('inf'), np.pi, 'garbage', {'junk': 3}, list(range(10)))
    )
    def test_input_features_rejects_junk(
        self, _TestCls, junk_input_features
    ):

        # **** CAN ONLY TAKE LIST-TYPE OF STRS OR None

        with pytest.raises(ValueError):
            _TestCls.get_feature_names_out(junk_input_features)


    def test_input_features_rejects_bad(
        self, _format, _TestCls, _shape
    ):

        # -------------
        # SHOULD RAISE ValueError IF
        # len(input_features) != n_features_in_
        with pytest.raises(ValueError):
            # columns too long
            _TestCls.get_feature_names_out(
                [f"x{i}" for i in range(2 * _shape[1])]
            )

        with pytest.raises(ValueError):
            # columns too short
            _TestCls.get_feature_names_out(
                [f"x{i}" for i in range(_shape[1]//2)]
            )

        if _format == 'pd':
            # WITH HEADER PASSED, SHOULD RAISE ValueError IF
            # column names not same as originally passed during fit
            with pytest.raises(ValueError):
                _TestCls.get_feature_names_out([f"x{i}" for i in range(_shape[1])])

        # -------------


@pytest.mark.parametrize('_format, _pd_columns_is_passed',
    (
        ('np', False),
        ('pd', True),
        ('pd', False),
    ),
    scope='module'
)
@pytest.mark.parametrize('_dtype', ('flt', 'int'), scope='module')
@pytest.mark.parametrize('_instance_state',
    ('after_fit', 'after_transform'), scope='module'
)
@pytest.mark.parametrize('_min_degree', (1, 2), scope='module')
@pytest.mark.parametrize('_interaction_only', (True, False), scope='module')
class TestGetFeatureNamesOut:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (8, 4)


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]


    @staticmethod
    @pytest.fixture(scope='function')
    def _wip_kwargs(_min_degree, _interaction_only):
        return {
            'degree': 3,
            'min_degree': _min_degree,
            'interaction_only': _interaction_only,
            'scan_X': False,
            'keep': 'first',
            'sparse_output': False,
            'feature_name_combiner': 'as_indices',
            'equal_nan': False,
            'rtol': 1e-5,
            'atol': 1e-8,
            'n_jobs': 1  # leave this at 1 because of confliction
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _X(
        _X_factory, _format, _dtype, _pd_columns_is_passed,
        _columns, _shape
    ):

        __ = _columns if (_format == 'pd' and _pd_columns_is_passed) else None

        return _X_factory(
            _dupl=None,
            _format=_format,
            _dtype=_dtype,
            _columns=__,
            _constants=None,
            _shape=_shape
        )


    @staticmethod
    @pytest.fixture(scope='function')
    def _TestCls(_instance_state, _wip_kwargs, _X):

        _TestCls = SlimPoly(**_wip_kwargs)

        if _instance_state == 'after_fit':
            _TestCls.fit(_X)
            return _TestCls
        elif _instance_state == 'after_transform':
            _TestCls.fit_transform(_X)
            return _TestCls
        else:
            raise Exception

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('_input_features_is_passed', (True, False))
    def test_accuracy(
        self, _X, _wip_kwargs, _format, _columns, _TestCls, _min_degree, _shape,
        _interaction_only, _input_features_is_passed, _pd_columns_is_passed
    ):

        # get actual ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        if _input_features_is_passed:
            out = _TestCls.get_feature_names_out(_columns)
        else:
            out = _TestCls.get_feature_names_out(None)
        # END get actual ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # determine expected ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # build polynomial header for 'as_indices'
        # dont bother testing 'as_features' or callable, the accuracy of building
        # the POLY portion of the header is handled in gfno_poly_3sec_test
        if _interaction_only:
            _fxn = itertools.combinations
        elif not _interaction_only:
            _fxn = itertools.combinations_with_replacement

        _POLY_HEADER = []
        for _poly_degree in range(2, _TestCls.degree+1):
            _POLY_HEADER += list(map(str, _fxn(range(_shape[1]), _poly_degree)))
        # END build polynomial header for 'as_indices'


        if _min_degree != 1:
            _EXP_HEADER = np.array(_POLY_HEADER).astype(object)
        elif _min_degree == 1:
            # build X header (if :param: min_degree == 1)

            # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN
            # ['x0', ..., 'x(n-1)][column_mask_]
            _GENERIC_HEADER = np.array(
                [f"x{i}" for i in range(_shape[1])],
                dtype=object
            )

            # WITH HEADER PASSED TO input_features SHOULD RETURN
            # self.feature_names_in_[column_mask_]
            # self.feature_names_in_ is being passed here to input_features

            if _format == 'np':
                _X_HEADER = _columns if _input_features_is_passed else _GENERIC_HEADER
            elif _format == 'pd':
                if _pd_columns_is_passed:
                    _X_HEADER = _columns
                elif not _pd_columns_is_passed:
                    if _input_features_is_passed:
                        _X_HEADER = _columns
                    else:
                        _X_HEADER = _GENERIC_HEADER
            else:
                raise Exception

            _EXP_HEADER = np.hstack((_X_HEADER, _POLY_HEADER)).astype(object)


        # END determine expected ** * ** * ** * ** * ** * ** * ** * ** * ** * s

        assert isinstance(out, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(out)}")

        assert out.dtype == object, \
            (f"get_feature_names_out dtype should be object, but "
             f"returned {out.dtype}")

        if _format == 'np':
            # WHEN NO HEADER PASSED TO (partial_)fit()
            if _input_features_is_passed:
                # SHOULD RETURN SLICED PASSED input_features
                assert np.array_equiv(out, _EXP_HEADER), \
                    (f"get_feature_names_out(_columns) after fit() != "
                     f"sliced array of valid input features")
            if not _input_features_is_passed:
                assert np.array_equiv(out, _EXP_HEADER), \
                    (f"get_feature_names_out(None) after fit() != sliced "
                     f"array of generic headers")
        elif _format == 'pd':
            if _pd_columns_is_passed:
                assert np.array_equiv(out, _EXP_HEADER), \
                    (f"get_feature_names_out(_columns) after fit() != "
                     f"sliced array of feature_names_in_")
            elif not _pd_columns_is_passed:
                if _input_features_is_passed:
                    assert np.array_equiv(out, _EXP_HEADER), \
                        (f"get_feature_names_out(_columns) after fit() != "
                         f"sliced array of valid input features")
                elif not _input_features_is_passed:
                    assert np.array_equiv(out, _EXP_HEADER), \
                        (f"get_feature_names_out(None) after fit() != sliced "
                        f"array of generic headers")
        else:
            raise Exception








