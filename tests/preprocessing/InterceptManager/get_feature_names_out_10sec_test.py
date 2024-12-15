# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager.InterceptManager import \
    InterceptManager as IM

import numpy as np
from pybear.exceptions import NotFittedError

import pytest





class TestAlwaysExceptsBeforeFit:


    @pytest.mark.parametrize('_keep', ('first', 'last', 'random', 'none'))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_always_except_before_fit(self, _keep, _equal_nan):

        with pytest.raises(NotFittedError):
            IM().get_feature_names_out()




@pytest.mark.parametrize('_format', ('np', 'pd'), scope='module')
@pytest.mark.parametrize('_instance_state',
    ('after_fit', 'after_transform'), scope='module'
)
class TestGetFeatureNamesOutRejects:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (50, 10)


    @staticmethod
    @pytest.fixture(scope='module')
    def _X(_X_factory, _format, _master_columns, _shape):

        return _X_factory(
            _dupl=None,
            _format=_format,
            _dtype='flt',
            _columns=_master_columns.copy()[:_shape[1]] if _format == 'pd' else None,
            _constants=None,
            _noise=1e-9,
            _shape=_shape
        )


    @staticmethod
    @pytest.fixture(scope='function')
    def _TestCls(_instance_state, _X):

        _TestCls = IM()

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



@pytest.mark.parametrize('_format', ('np', 'pd'), scope='module')
@pytest.mark.parametrize('_dtype',
    ('flt', 'int', 'str', 'obj', 'hybrid'), scope='module'
)
@pytest.mark.parametrize('_instance_state',
    ('after_fit', 'after_transform'), scope='module'
)
@pytest.mark.parametrize('_keep',
    ('first', 'last', 'none', {'Intercept': 1}), scope='module'
)
@pytest.mark.parametrize('_constants',
    ('none', 'constants1', 'constants2'), scope='module'
)
class TestGetFeatureNamesOut:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (50, 10)


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]


    @staticmethod
    @pytest.fixture(scope='function')
    def _wip_kwargs(_keep):
        return {
            'keep': _keep,
            'equal_nan': True,
            'rtol': 1e-5,
            'atol': 1e-8,
            'n_jobs': 1     # leave this at 1 because of confliction
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _wip_constants(_constants, _dtype, _shape):

        if _constants == 'none':
            return {}
        elif _constants == 'constants1':
            if _dtype in ('flt', 'int'):
                return {1: 1, _shape[1]-2: 1}
            else:
                return {1: 'a', _shape[1]-2: 'b'}
        elif _constants == 'constants2':
            if _dtype in ('flt', 'int'):
                return {0: 0, _shape[1]-1: np.nan}
            else:
                return {1: 'a', _shape[1]-1: 'nan'}


    @staticmethod
    @pytest.fixture(scope='module')
    def _X(_X_factory, _format, _dtype, _columns, _wip_constants, _shape):

        return _X_factory(
            _dupl=None,
            _format=_format,
            _dtype=_dtype,
            _columns=_columns if _format == 'pd' else None,
            _constants=_wip_constants,
            _noise=1e-9,
            _shape=_shape
        )


    @staticmethod
    @pytest.fixture(scope='function')
    def _TestCls(_instance_state, _wip_kwargs, _X):

        _TestCls = IM(**_wip_kwargs)

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
    def test_valid_input_features(
        self, _X, _wip_kwargs, _wip_constants, _format, _columns, _TestCls,
        _keep, _shape, _input_features_is_passed
    ):

        # get actual ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        if _input_features_is_passed:
            out = _TestCls.get_feature_names_out(_columns)
        else:
            out = _TestCls.get_feature_names_out(None)
        # END get actual ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # determine expected ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        if _format == 'np' and not _input_features_is_passed:
            # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN
            # ['x0', ..., 'x(n-1)][column_mask_]
            _EXP_COLUMNS = np.array(
                [f"x{i}" for i in range(_shape[1])],
                dtype=object
            )
        else:
            # WITH HEADER PASSED SHOULD RETURN
            # self.feature_names_in_[column_mask_]
            # self.feature_names_in_ is being passed here for np input_features
            # and is always returned for pd
            _EXP_COLUMNS = _columns

        # build column_mask_ - - - - - - - - - - - - - - - - - - - - - - - -
        MASK = np.ones((_shape[1], ), dtype=bool)

        _sorted_constants = sorted(list(_wip_constants.keys()))

        if len(_sorted_constants):
            if _keep == 'none':
                for c_idx in _sorted_constants:
                    MASK[c_idx] = False
            elif _keep == 'first':
                for c_idx in _sorted_constants[1:]:
                    MASK[c_idx] = False
            elif _keep == 'last':
                for c_idx in _sorted_constants[:-1]:
                    MASK[c_idx] = False
            elif isinstance(_keep, dict):
                for c_idx in _sorted_constants:
                    MASK[c_idx] = False
        # elif not len(_sorted_constants):
        #     with no constants no columns are dropped, MASK is unchanged

        # END build column_mask_ - - - - - - - - - - - - - - - - - - - - - -

        _EXP_COLUMNS = _EXP_COLUMNS[MASK]
        del MASK

        if isinstance(_keep, dict):
            _EXP_COLUMNS = np.hstack((
                _EXP_COLUMNS,
                list(_keep.keys())[0]
            )).astype(object)
        # END determine expected ** * ** * ** * ** * ** * ** * ** * ** * ** * s

        assert isinstance(out, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(out)}")

        assert out.dtype == object, \
            (f"get_feature_names_out dtype should be object, but "
             f"returned {out.dtype}")

        if _format == 'np' and not _input_features_is_passed:
            assert np.array_equiv(out, _EXP_COLUMNS), \
                (f"get_feature_names_out(None) after fit() != sliced array "
                 f"of generic headers")
        else:
            # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features,
            # SHOULD RETURN SLICED PASSED input_features
            # PD SHOULD ALWAYS RETURN SLICED feature_names_in_
            assert np.array_equiv(out, _EXP_COLUMNS), \
                (f"get_feature_names_out(_columns) after fit() != sliced array "
                 f"of valid input features")








