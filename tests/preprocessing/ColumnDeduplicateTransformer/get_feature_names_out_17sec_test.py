# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from copy import deepcopy

import numpy as np

from pybear.preprocessing import ColumnDeduplicateTransformer as CDT
from pybear.base.exceptions import NotFittedError



class TestAlwaysExceptsBeforeFit:


    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_always_except_before_fit(self, _keep, _equal_nan):

        with pytest.raises(NotFittedError):
            CDT().get_feature_names_out()


@pytest.mark.parametrize('_format', ('np', 'pd'), scope='module')
@pytest.mark.parametrize('_instance_state',
    ('after_fit', 'after_transform'), scope='module'
)
class TestInputFeaturesRejects:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='function')
    def _TestCls(_instance_state, _X_factory, _format, _columns, _shape):

        _TestCls = CDT()

        _X = _X_factory(
                _dupl=None,
                _format=_format,
                _dtype='flt',
                _columns=_columns if _format == 'pd' else None,
                _shape=_shape
            )

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
    def test_input_features_rejects_junk(self, _TestCls, junk_input_features):

        # **** CAN ONLY TAKE LIST-TYPE OF STRS OR None

        with pytest.raises(ValueError):
            _TestCls.get_feature_names_out(junk_input_features)


    def test_input_features_rejects_bad(self, _format, _TestCls, _shape):

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
    (('np', False), ('pd', True), ('pd', False)), scope='module'
)
@pytest.mark.parametrize('_dtype',
    ('flt', 'int', 'str', 'obj', 'hybrid'), scope='module'
)
@pytest.mark.parametrize('_instance_state',
    ('after_fit', 'after_transform'), scope='module'
)
@pytest.mark.parametrize('_keep', ('first', 'last'), scope='module')
@pytest.mark.parametrize('_dupls', ('none', 'dupls1', 'dupls2'), scope='module')
class TestGetFeatureNamesOut:


    @pytest.mark.parametrize('_input_features_is_passed', (True, False))
    def test_accuracy(
        self, _X_factory, _instance_state, _kwargs, _dupls, _format,
        _dtype, _columns, _keep, _shape, _input_features_is_passed,
        _pd_columns_is_passed
    ):

        # build X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        if _dupls == 'none':
            _wip_dupls =  []
        elif _dupls == 'dupls1':
            _wip_dupls =  [[1, _shape[1] - 2]]
        elif _dupls == 'dupls2':
            _wip_dupls =  [
                [0, _shape[1] - 1],
                [1, _shape[1] - 2]
            ]
        else:
            raise Exception

        _X = _X_factory(
            _dupl=_wip_dupls,
            _format=_format,
            _dtype=_dtype,
            _columns=_columns if (_format == 'pd' and _pd_columns_is_passed) else None,
            _shape=_shape
        )

        # END build X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # prepare the CDT instance ** * ** * ** * ** * ** * ** * ** * **
        _new_kwargs = deepcopy(_kwargs)
        _new_kwargs['keep'] = _keep
        _new_kwargs['equal_nan'] = True

        _TestCls = CDT(**_new_kwargs)

        if _instance_state == 'after_fit':
            _TestCls.fit(_X)
        elif _instance_state == 'after_transform':
            _TestCls.fit_transform(_X)
        else:
            raise Exception
        # END prepare the CDT instance ** * ** * ** * ** * ** * ** * **

        # get actual ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        if _input_features_is_passed:
            out = _TestCls.get_feature_names_out(_columns)
        else:
            out = _TestCls.get_feature_names_out(None)
        # END get actual ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # determine expected ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
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
            _EXP_HEADER = _columns if _input_features_is_passed else _GENERIC_HEADER
        elif _format=='pd':
            if _pd_columns_is_passed:
                _EXP_HEADER = _columns
            elif not _pd_columns_is_passed:
                if _input_features_is_passed:
                    _EXP_HEADER = _columns
                else:
                    _EXP_HEADER = _GENERIC_HEADER
        else:
            raise Exception

        # build column_mask_ - - - - - - - - - - - - - - - - - - - - - - - -
        MASK = np.ones((_shape[1], ), dtype=bool)

        _dupls_to_drop = []

        # with no dupls no columns are dropped, this :for: is skipped
        # and MASK is unchanged
        for _dupl_set in _wip_dupls:
            _sorted_dupl_set = sorted(_dupl_set)
            if _keep == 'first':
                _dupls_to_drop.extend(_sorted_dupl_set[1:])
            elif _keep == 'last':
                _dupls_to_drop.extend(_sorted_dupl_set[:-1])


        # END build column_mask_ - - - - - - - - - - - - - - - - - - - - - -

        MASK[_dupls_to_drop] = False
        _EXP_HEADER = _EXP_HEADER[MASK]
        del MASK, _dupls_to_drop
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





