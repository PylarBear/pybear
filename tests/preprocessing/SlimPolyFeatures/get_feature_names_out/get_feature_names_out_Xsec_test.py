# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.SlimPolyFeatures.SlimPolyFeatures import \
    SlimPolyFeatures as SlimPoly

from typing import Literal, Iterable
from typing_extensions import Union

import numpy as np
from sklearn.exceptions import NotFittedError

import pytest


pytest.skip(reason=f"pizza not finished", allow_module_level=True)



class Fixtures:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (8, 4)


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]


    @staticmethod
    @pytest.fixture(scope='module')
    def _X(_X_factory, _shape):

        def foo(
            _format: Literal['np', 'pd', 'csr'],
            _dtype: Literal['flt', 'int'],
            _columns: Union[Iterable[str], None],
        ):

            return _X_factory(
                _dupl=None,
                _format=_format,
                _dtype=_dtype,
                _columns=_columns,
                _shape=_shape
            )

        return foo


    @staticmethod
    @pytest.fixture(scope='module')
    def _kwargs():
        return {
            'degree': 2,
            'min_degree': 1,
            'scan_X': False,
            'keep': 'first',
            'interaction_only': False,
            'sparse_output': False,
            'feature_name_combiner': "as_indices",
            'equal_nan': True,
            'rtol': 1e-5,
            'atol': 1e-8,
            'n_jobs': 1     # leave this at 1 because of confliction
        }





@pytest.mark.parametrize('_format', ('np',), scope='module')
@pytest.mark.parametrize('_dtype', ('flt', 'int'), scope='module')
@pytest.mark.parametrize('_keep', ('first', 'last', 'random'), scope='module')
@pytest.mark.parametrize('_min_degree', (1, 2), scope='module')
@pytest.mark.parametrize('_intx_only', (True, False), scope='module')
@pytest.mark.parametrize('_feature_name_combiner', ('as_feature_names', 'as_indices'), scope='module')
class TestGetFeatureNamesOutNonPd(Fixtures):


    @staticmethod
    @pytest.fixture(scope='module')
    def _wip_kwargs(_kwargs, _keep, _min_degree, _intx_only, _feature_name_combiner):

        _kwargs['keep'] = _keep
        _kwargs['min_degree'] = _min_degree
        _kwargs['interaction_only'] = _intx_only
        _kwargs['feature_name_combiner'] = _feature_name_combiner

        return _kwargs


    @staticmethod
    @pytest.fixture(scope='module')
    def _wip_X(_X, _format, _dtype, _shape):

        return _X(
            _format=_format,
            _dtype=_dtype,
            _columns=None
        )


    # v^v^v^v^v^v^ always NotFittedError before fit v^v^v^v^v^v^v^v^v^v^v^
    def test_always_except_before_fit(self, _wip_X, _wip_kwargs):

        TestCls = SlimPoly(**_wip_kwargs)

        with pytest.raises(NotFittedError):
            TestCls.get_feature_names_out()

    # v^v^v^v^v^v^ end before fit v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


    # v^v^v^v^v^v^ after_fit v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    @pytest.mark.parametrize('junk_input_features',
        (float('inf'), np.pi, 'garbage', {'junk': 3}, list(range(10)))
    )
    def test_fitted__rejects_junk(
        self, _wip_X, _wip_kwargs, junk_input_features
    ):

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit(_wip_X)

        # vvv NO COLUMN NAMES PASSED (NP) vvv
        # **** CAN ONLY TAKE LIST-TYPE OF STRS OR None

        with pytest.raises(ValueError):
            TestCls.get_feature_names_out(junk_input_features)


    def test_fitted__rejects_bad(self, _wip_X, _wip_kwargs, _shape):

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit(_wip_X)

        # -------------
        # WITH NO HEADER PASSED, SHOULD RAISE ValueError IF
        # len(input_features) != n_features_in_
        with pytest.raises(ValueError):
            # columns too long
            TestCls.get_feature_names_out(
                [f"x{i}" for i in range(2 * _shape[1])]
            )

        with pytest.raises(ValueError):
            # columns too short
            TestCls.get_feature_names_out(
                [f"x{i}" for i in range(_shape[1]//2)]
            )
        # -------------


    def test_fitted__input_features_is_None(
        self, _wip_X, _wip_kwargs, _keep, _shape
    ):

        # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN
        # ['x0', ..., 'x(n-1)] + POLY_FEATURES

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit(_wip_X)

        # if _keep == 'last':
        #     for c_idx in _sorted_constants:
        #         MASK[c_idx] = False
        # elif _keep == 'first':
        #     for c_idx in _sorted_constants[1:]:
        #         MASK[c_idx] = False
        # elif _keep == 'random':
        #     for c_idx in _sorted_constants[1:]:
        #         MASK[c_idx] = False

        _KEPT_COLUMNS = np.hstack((
            [f"x{i}" for i in range(_shape[1])],
            list(map(str, TestCls.expansion_combinations_))
        ))

        out = TestCls.get_feature_names_out(None)

        assert isinstance(out, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(out)}")

        assert out.dtype == object, \
            (f"get_feature_names_out dtype should be object, but "
             f"returned {out.dtype}")

        assert np.array_equiv(out, _KEPT_COLUMNS), \
            (f"get_feature_names_out(None) after fit() != sliced array "
             f"of generic headers")


    def test_fitted__valid_input_features(
        self, _wip_X, _wip_kwargs, _columns, _keep, _shape
    ):

        # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features,
        # SHOULD RETURN SLICED PASSED COLUMNS

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit(_wip_X)

        MASK = np.ones((_shape[1], ), dtype=bool)


        # if _keep == 'last':
        #     for c_idx in _sorted_constants:
        #         MASK[c_idx] = False
        # elif _keep == 'first':
        #     for c_idx in _sorted_constants[1:]:
        #         MASK[c_idx] = False
        # elif _keep == 'random':
        #     for c_idx in _sorted_constants[1:]:
        #         MASK[c_idx] = False

        _KEPT_COLUMNS = np.hstack((
            _columns,
            list(map(str, TestCls.expansion_combinations_))
        ))

        out = TestCls.get_feature_names_out(None)

        assert isinstance(out, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(out)}")

        assert out.dtype == object, \
            (f"get_feature_names_out dtype should be object, but "
             f"returned {out.dtype}")

        assert np.array_equiv(out, _KEPT_COLUMNS), \
            (f"get_feature_names_out(None) after fit() != sliced array "
             f"of generic headers")
        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^



    # v^v^v^v^v^v^ END test_access_methods_after_fit v^v^v^v^v^v^v^v^v^v^v^v^v^



    # v^v^v^v^v^v^ test_access_methods_after_transform v^v^v^v^v^v^v^v^v^v^v^v^
    @pytest.mark.parametrize('junk_input_features',
        (float('inf'), np.pi, 'garbage', {'junk': 3}, list(range(10)))
    )
    def test_transformed__rejects_junk(
        self, _wip_X, _wip_kwargs, junk_input_features
    ):

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit_transform(_wip_X)

        # vvv NO COLUMN NAMES PASSED (NP) vvv
        # **** CAN ONLY TAKE LIST-TYPE OF STRS OR None

        with pytest.raises(ValueError):
            TestCls.get_feature_names_out(junk_input_features)


    def test_transformed__rejects_bad(self, _wip_X, _wip_kwargs, _shape):

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit_transform(_wip_X)

        # -------------
        # WITH NO HEADER PASSED, SHOULD RAISE ValueError IF
        # len(input_features) != n_features_in_
        with pytest.raises(ValueError):
            TestCls.get_feature_names_out(
                [f"x{i}" for i in range(2 * _shape[1])]
            )

        with pytest.raises(ValueError):
            TestCls.get_feature_names_out(
                [f"x{i}" for i in range(_shape[1] // 2)]
            )
        # -------------


    def test_transformed__input_features_is_None(
            self, _wip_X, _wip_kwargs, _wip_constants, _keep, _shape
    ):

        # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN
        # ['x0', ..., 'x(n-1)][column_mask_]

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit_transform(_wip_X)

        # if len(_sorted_constants):
        #     if _keep == 'none':
        #         for c_idx in _sorted_constants:
        #             MASK[c_idx] = False
        #     elif _keep == 'first':
        #         for c_idx in _sorted_constants[1:]:
        #             MASK[c_idx] = False
        #     elif isinstance(_keep, dict):
        #         for c_idx in _sorted_constants:
        #             MASK[c_idx] = False

        _KEPT_COLUMNS = np.hstack((
            _columns,
            list(map(str, TestCls.expansion_combinations_))
        ))

        out = TestCls.get_feature_names_out(None)

        assert isinstance(out, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(out)}")

        assert out.dtype == object, \
            (f"get_feature_names_out dtype should be object, but "
             f"returned {out.dtype}")

        assert np.array_equiv(out, _KEPT_COLUMNS), \
            (f"get_feature_names_out(None) after fit() != sliced array "
             f"of generic headers")


    def test_transformed__valid_input_features(
            self, _wip_X, _columns, _wip_kwargs, _wip_constants, _keep, _shape
    ):

        # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features,
        # SHOULD RETURN SLICED PASSED COLUMNS

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit_transform(_wip_X)

        # if len(_sorted_constants):
        #     if _keep == 'none':
        #         for c_idx in _sorted_constants:
        #             MASK[c_idx] = False
        #     elif _keep == 'first':
        #         for c_idx in _sorted_constants[1:]:
        #             MASK[c_idx] = False
        #     elif isinstance(_keep, dict):
        #         for c_idx in _sorted_constants:
        #             MASK[c_idx] = False

        _KEPT_COLUMNS = np.hstack((
            _columns,
            list(map(str, TestCls.expansion_combinations_))
        ))

        out = TestCls.get_feature_names_out(None)

        assert isinstance(out, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(out)}")

        assert out.dtype == object, \
            (f"get_feature_names_out dtype should be object, but "
             f"returned {out.dtype}")

        assert np.array_equiv(out, _KEPT_COLUMNS), \
            (f"get_feature_names_out(None) after fit() != sliced array "
             f"of generic headers")

        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

    # v^v^v^v^v^v^ END test_access_methods_after_transform v^v^v^v^v^v^v^v^v^v^









@pytest.mark.parametrize('_format', ('pd',), scope='module')
@pytest.mark.parametrize('_dtype', ('flt', 'int'), scope='module')
@pytest.mark.parametrize('_keep', ('first', 'last', 'random'), scope='module')
@pytest.mark.parametrize('_min_degree', (1, 2), scope='module')
@pytest.mark.parametrize('_intx_only', (True, False), scope='module')
@pytest.mark.parametrize('_feature_name_combiner', ('as_feature_names', 'as_indices'), scope='module')
class TestGetFeatureNamesOutPd(Fixtures):


    @staticmethod
    @pytest.fixture(scope='module')
    def _wip_kwargs(_kwargs, _keep):

        _kwargs['keep'] = _keep

        return _kwargs


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
    def _wip_X(_X, _columns, _format, _dtype, _wip_constants, _shape):

        return _X(
            _format=_format,
            _dtype=_dtype,
            _columns=_columns,
            _constants=_wip_constants
        )


    # v^v^v^v^v^v^ always NotFittedError before fit v^v^v^v^v^v^v^v^v^v^v^
    def test_always_except_before_fit(self, _wip_X, _wip_kwargs):

        TestCls = SlimPoly(**_wip_kwargs)

        with pytest.raises(NotFittedError):
            TestCls.get_feature_names_out()

    # v^v^v^v^v^v^ end before fit v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


    # v^v^v^v^v^v^ after_fit v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    @pytest.mark.parametrize('junk_input_features',
        (float('inf'), np.pi, 'garbage', {'junk': 3}, list(range(10)))
    )
    def test_fitted__rejects_junk(
        self, _wip_X, _wip_kwargs, junk_input_features
    ):

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit(_wip_X)

        # **** CAN ONLY TAKE LIST-TYPE OF STRS OR None

        with pytest.raises(ValueError):
            TestCls.get_feature_names_out(junk_input_features)


    def test_fitted__rejects_bad(self, _wip_X, _columns, _wip_kwargs, _shape):

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit(_wip_X)

        # -------------
        # WITH HEADER PASSED, SHOULD RAISE ValueError IF
        # len(input_features) != n_features_in_

        with pytest.raises(ValueError):
            # columns too long
            TestCls.get_feature_names_out(
                [f"x{i}" for i in range(2 * _shape[1])]
            )

        with pytest.raises(ValueError):
            # columns too short
            TestCls.get_feature_names_out(
                [f"x{i}" for i in range(_shape[1]//2)]
            )

        # WITH HEADER PASSED, SHOULD RAISE ValueError IF
        # column names not same as originally passed during fit
        with pytest.raises(ValueError):
            TestCls.get_feature_names_out(np.flip(_columns))

        # -------------


    def test_fitted__input_features_is_None(
        self, _wip_X, _columns, _wip_kwargs, _wip_constants, _keep, _shape
    ):

        # WITH HEADER PASSED AND input_features=None, SHOULD RETURN
        # self.feature_names_in_[column_mask_], where
        # self.feature_names_in_ == _columns

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit(_wip_X)

        MASK = np.ones((_shape[1], ), dtype=bool)

        if _keep not in ('none', 'first') and not isinstance(_keep, dict):
            raise Exception

        _sorted_constants = sorted(list(_wip_constants.keys()))

        if len(_sorted_constants):
            if _keep == 'none':
                for c_idx in _sorted_constants:
                    MASK[c_idx] = False
            elif _keep == 'first':
                for c_idx in _sorted_constants[1:]:
                    MASK[c_idx] = False
            elif isinstance(_keep, dict):
                for c_idx in _sorted_constants:
                    MASK[c_idx] = False

        _CHOPPED_COLUMNS = _columns.copy()[MASK]

        if isinstance(_keep, dict):
            _CHOPPED_COLUMNS = np.hstack((
                _CHOPPED_COLUMNS,
                list(_keep.keys())[0]
            )).astype(object)


        out = TestCls.get_feature_names_out(None)

        assert isinstance(out, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(out)}")

        assert out.dtype == object, \
            (f"get_feature_names_out dtype should be object, but "
             f"returned {out.dtype}")

        assert np.array_equiv(out, _CHOPPED_COLUMNS), \
            (f"get_feature_names_out(None) after fit() != sliced feature_names_in_")


    def test_fitted__valid_input_features(
        self, _wip_X, _columns, _wip_kwargs, _wip_constants, _keep, _shape
    ):

        # WHEN HEADER PASSED TO (partial_)fit() AND VALID input_features,
        # SHOULD RETURN SLICED feature_names_in_

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit(_wip_X)

        MASK = np.ones((_shape[1], ), dtype=bool)

        if _keep not in ('none', 'first') and not isinstance(_keep, dict):
            raise Exception

        _sorted_constants = sorted(list(_wip_constants.keys()))

        if len(_sorted_constants):
            if _keep == 'none':
                for c_idx in _sorted_constants:
                    MASK[c_idx] = False
            elif _keep == 'first':
                for c_idx in _sorted_constants[1:]:
                    MASK[c_idx] = False
            elif isinstance(_keep, dict):
                for c_idx in _sorted_constants:
                    MASK[c_idx] = False

        _CHOPPED_COLUMNS = _columns.copy()[MASK]

        if isinstance(_keep, dict):
            _CHOPPED_COLUMNS = np.hstack((
                _CHOPPED_COLUMNS,
                list(_keep.keys())[0]
            )).astype(object)


        out = TestCls.get_feature_names_out(_columns)

        assert isinstance(out, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(out)}")

        assert out.dtype == object, \
            (f"get_feature_names_out dtype should be object, but "
             f"returned {out.dtype}")

        assert np.array_equiv(out, _CHOPPED_COLUMNS), \
            (f"get_feature_names_out(_columns) after fit() != sliced array "
             f"of valid input features")


    # v^v^v^v^v^v^ END test_access_methods_after_fit v^v^v^v^v^v^v^v^v^v^v^v^v^



    # v^v^v^v^v^v^ test_access_methods_after_transform v^v^v^v^v^v^v^v^v^v^v^v^
    @pytest.mark.parametrize('junk_input_features',
        (float('inf'), np.pi, 'garbage', {'junk': 3}, list(range(10)))
    )
    def test_transformed__rejects_junk(
        self, _wip_X, _wip_kwargs, junk_input_features
    ):

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit_transform(_wip_X)

        # **** CAN ONLY TAKE LIST-TYPE OF STRS OR None

        with pytest.raises(ValueError):
            TestCls.get_feature_names_out(junk_input_features)


    def test_transformed__rejects_bad(
        self, _wip_X, _columns, _wip_kwargs, _shape
    ):

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit_transform(_wip_X)

        # -------------
        # WITH HEADER PASSED, SHOULD RAISE ValueError IF
        # len(input_features) != n_features_in_

        with pytest.raises(ValueError):
            TestCls.get_feature_names_out(
                [f"x{i}" for i in range(2 * _shape[1])]
            )

        with pytest.raises(ValueError):
            TestCls.get_feature_names_out(
                [f"x{i}" for i in range(_shape[1] // 2)]
            )

        # WITH HEADER PASSED, SHOULD RAISE ValueError IF
        # column names not same as originally passed during fit
        with pytest.raises(ValueError):
            TestCls.get_feature_names_out(np.flip(_columns))
        # -------------


    def test_transformed__input_features_is_None(
            self, _wip_X, _columns, _wip_kwargs, _wip_constants, _keep, _shape
    ):

        # WITH HEADER PASSED AND input_features=None, SHOULD RETURN
        # self.feature_names_in_[column_mask_], where
        # feature_names_in_ == _columns

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit_transform(_wip_X)

        _COLUMNS = np.array([f"x{i}" for i in range(_shape[1])], dtype=object)

        MASK = np.ones((_shape[1],), dtype=bool)

        if _keep not in ('none', 'first') and not isinstance(_keep, dict):
            raise Exception

        _sorted_constants = sorted(list(_wip_constants.keys()))

        if len(_sorted_constants):
            if _keep == 'none':
                for c_idx in _sorted_constants:
                    MASK[c_idx] = False
            elif _keep == 'first':
                for c_idx in _sorted_constants[1:]:
                    MASK[c_idx] = False
            elif isinstance(_keep, dict):
                for c_idx in _sorted_constants:
                    MASK[c_idx] = False

        _CHOPPED_COLUMNS = _columns.copy()[MASK]
        del _COLUMNS

        if isinstance(_keep, dict):
            _CHOPPED_COLUMNS = np.hstack((
                _CHOPPED_COLUMNS,
                list(_keep.keys())[0]
            )).astype(object)

        out = TestCls.get_feature_names_out(None)

        assert isinstance(out, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(out)}")

        assert out.dtype == object, \
            (f"get_feature_names_out dtype should be object, but "
             f"returned {out.dtype}")

        assert np.array_equiv(out, _CHOPPED_COLUMNS), \
            (f"get_feature_names_out(None) after fit() != sliced feature_names_in_")


    def test_transformed__valid_input_features(
            self, _wip_X, _columns, _wip_kwargs, _wip_constants, _keep, _shape
    ):

        # WHEN HEADER PASSED TO (partial_)fit() AND VALID input_features,
        # SHOULD RETURN SLICED PASSED COLUMNS

        TestCls = SlimPoly(**_wip_kwargs)
        TestCls.fit_transform(_wip_X)

        MASK = np.ones((_shape[1],), dtype=bool)

        if _keep not in ('none', 'first') and not isinstance(_keep, dict):
            raise Exception

        _sorted_constants = sorted(list(_wip_constants.keys()))

        if len(_sorted_constants):
            if _keep == 'none':
                for c_idx in _sorted_constants:
                    MASK[c_idx] = False
            elif _keep == 'first':
                for c_idx in _sorted_constants[1:]:
                    MASK[c_idx] = False
            elif isinstance(_keep, dict):
                for c_idx in _sorted_constants:
                    MASK[c_idx] = False

        _CHOPPED_COLUMNS = _columns.copy()[MASK]

        if isinstance(_keep, dict):
            _CHOPPED_COLUMNS = np.hstack((
                _CHOPPED_COLUMNS,
                list(_keep.keys())[0]
            )).astype(object)

        out = TestCls.get_feature_names_out(_columns)

        assert isinstance(out, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(out)}")

        assert out.dtype == object, \
            (f"get_feature_names_out dtype should be object, but "
             f"returned {out.dtype}")

        assert np.array_equiv(out, _CHOPPED_COLUMNS), \
            (f"get_feature_names_out(_columns) after fit() != sliced array "
             f"of valid input features")

    # v^v^v^v^v^v^ END test_access_methods_after_transform v^v^v^v^v^v^v^v^v^v^










