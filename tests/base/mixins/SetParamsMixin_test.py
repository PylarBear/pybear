# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.base.mixins._SetParamsMixin import SetParamsMixin

import numpy as np

import pytest


pytest.skip(reason='pizza not started, not finished', allow_module_level=True)





class Fixtures:


    @staticmethod
    @pytest.fixture(scope='function')
    def _shape():
        return (10, 5)


    @staticmethod
    @pytest.fixture(scope='function')
    def _X_np(_shape):
        return np.random.randint(0, 10, _shape)


    @staticmethod
    @pytest.fixture(scope='function')
    def DummyEstimator():

        class Foo(SetParamsMixin):

            def __init__(self):
                self._is_fitted = False   # <===== leading under
                self.bananas = 7
                self.fries = 9
                self.ethanol = 5
                self.apples = 4


            def reset(self):

                self._is_fitted = False


            def fit(self, X, y=None):
                self.reset()
                self._is_fitted = True
                return self


            def score(self, X, y=None):
                return np.random.uniform(0, 1)


            def predict(self, X):

                assert self._is_fitted

                return np.random.randint(0, 2, X.shape[0])


        return Foo  # <====== not initialized


    @staticmethod
    @pytest.fixture(scope='function')
    def _est_kwargs():
        return {
            'bananas': True,
            'fries': 'yes',
            'ethanol': 1,
            'apples': [0, 1]
        }


    @staticmethod
    @pytest.fixture(scope='function')
    def DummyTransformer():

        class Bar(SetParamsMixin):

            def __init__(self):
                self._is_fitted = False   #  <==== leading under
                self.tbone = 3
                self.wings = 7
                self.bacon = 9
                self.sausage = 5
                self.hambone = 4


            def reset(self):
                try:
                    delattr(self, '_fill_value')
                except:
                    pass
                self._is_fitted = False


            def fit(self, X):
                self.reset()
                self._fill_value = np.random.uniform(0, 1)
                self._is_fitted = True
                return self


            def transform(self, X):

                assert self._is_fitted

                return np.full(X.shape, self._fill_value)


            def fit_transform(self, X):
                return self.fit(X).transform(X)


        return Bar  # <====== not initialized


    @staticmethod
    @pytest.fixture(scope='function')
    def _trfm_kwargs():
        return {
            'tbone': False,
            'wings': 'yes',
            'bacon': 0,
            'sausage': [4, 4],
            'hambone': False
        }


    @staticmethod
    @pytest.fixture(scope='function')
    def DummyGridSearch():

        class Baz(SetParamsMixin):

            def __init__(
                self,
                estimator,
                param_grid,
                *,
                scoring='balanced_accuracy',
                refit=False
            ):
                self.estimator = estimator
                self.param_grid = param_grid
                self.scoring = scoring
                self.refit = refit


            def fit(self, X, y=None):

                self.best_params_ = {}

                for _param in self.param_grid:
                    self.best_params_[_param] = \
                        np.random.choice(self.param_grid[_param])


        return Baz  # <====== not initialized


    @staticmethod
    @pytest.fixture(scope='function')
    def _gscv_kwargs(DummyEstimator):
        return {
            'estimator': DummyEstimator(),
            'param_grid': {
                'fries': [7, 8, 9],
                'ethanol': [5, 6, 7],
                'apples': [3, 4, 5]
            },
            'refit': True,
            'scoring': 'balanced_accuracy'
        }




@pytest.mark.parametrize('top_level_object', ('single_est', 'single_trfm'))
@pytest.mark.parametrize('state', ('pre-fit', 'post-fit'))
class TestSetParams__NonEmbedded(Fixtures):

    # simple est/trfms should be straightforward to test. verify that
    # bad params bounce off. verify that good params are set correctly.


    @staticmethod
    @pytest.fixture(scope='function')
    def _kwargs(top_level_object, _est_kwargs, _trfm_kwargs):

        if top_level_object == 'single_est':
            return _est_kwargs
        elif top_level_object == 'single_trfm':
            return _trfm_kwargs
        else:
            raise Exception


    @staticmethod
    @pytest.fixture(scope='function')
    def TopLevelObject(
        top_level_object, state, DummyEstimator, DummyTransformer, _X_np, _kwargs
    ):

        if top_level_object == 'single_est':
            foo = DummyEstimator
        elif top_level_object == 'single_trfm':
            foo = DummyTransformer
        else:
            raise Exception


        if state == 'pre-fit':
            foo(**_kwargs)
        elif state == 'post-fit':
            foo(**_kwargs).fit(_X_np)

        return foo

    # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    # ^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v


    def test_init_was_correctly_applied(self, TopLevelObject, _kwargs):

        # assert TestCls initiated correctly
        for _param, _value in _kwargs.items():
            assert getattr(TopLevelObject, _param) == _value


    def test_rejects_unknown_param(
        self, TopLevelObject, junk_param
    ):

        with pytest.raises(ValueError):
            TopLevelObject.set_params(garbage=True)

        with pytest.raises(ValueError):
            TopLevelObject.set_params(estimator__trash=True)


    def test_set_params_correctly_applies(
        self, top_level_object, TopLevelObject, _kwargs
    ):

        if top_level_object == 'single_est':
            _kwargs['bananas'] = False
            _kwargs['fries'] = 'no'
            _kwargs['ethanol'] = 0
            _kwargs['apples'] = 'yikes'
        elif top_level_object == 'single_trfm':
            _kwargs['tbone'] = True
            _kwargs['wings'] = 'np'
            _kwargs['bacon'] = 1
            _kwargs['sausage'] = False
            _kwargs['hambome'] = [1, 1]
        else:
            raise Exception


        TopLevelObject.set_params(**_kwargs)

        # assert new values set correctly
        for _param, _value in _kwargs.items():
            assert getattr(TopLevelObject, _param) == _value


    @pytest.mark.skip(reason='pizza')
    def test_equality_set_params_before_and_after_fit(
        self, X, y, _args, _kwargs, _rows, mmct
    ):

        alt_args = [_rows // 25]
        alt_kwargs = {
            'ignore_float_columns': True,
            'ignore_non_binary_integer_columns': True,
            'ignore_columns': [0, 2],
            'ignore_nan': False,
            'delete_axis_0': False,
            'handle_as_bool': None,
            'reject_unseen_values': False,
            'max_recursions': 1,
            'n_jobs': -1
        }

        # INITIALIZE->FIT_TRFM
        IFTCls = MCT(*alt_args, **alt_kwargs)
        SPFT_TRFM_X, SPFT_TRFM_Y = IFTCls.fit_transform(X.copy(), y.copy())

        # FIT->SET_PARAMS->TRFM
        FSPTCls = MCT(*_args, **_kwargs)
        FSPTCls.fit(X.copy(), y.copy())
        FSPTCls.set_params(count_threshold=alt_args[0], **alt_kwargs)
        FSPT_TRFM_X, FSPT_TRFM_Y = FSPTCls.transform(X.copy(), y.copy())
        assert FSPTCls._count_threshold == alt_args[0]

        # CHECK X AND Y EQUAL REGARDLESS OF WHEN SET_PARAMS
        assert np.array_equiv(SPFT_TRFM_X.astype(str), FSPT_TRFM_X.astype(str)), \
            f"SPFT_TRFM_X != FSPT_TRFM_X"

        assert np.array_equiv(SPFT_TRFM_Y, FSPT_TRFM_Y), \
            f"SPFT_TRFM_Y != FSPT_TRFM_Y"


        # VERIFY transform AGAINST REFEREE OUTPUT WITH SAME INPUTS
        MOCK_X = mmct().trfm(
            X.copy(),
            None,
            alt_kwargs['ignore_columns'],
            alt_kwargs['ignore_nan'],
            alt_kwargs['ignore_non_binary_integer_columns'],
            alt_kwargs['ignore_float_columns'],
            alt_kwargs['handle_as_bool'],
            alt_kwargs['delete_axis_0'],
            alt_args[0]
        )

        assert np.array_equiv(FSPT_TRFM_X.astype(str), MOCK_X.astype(str)), \
            f"FSPT_TRFM_X != MOCK_X"

        del MOCK_X


    @pytest.mark.skip(reason='pizza')
    def test_set_params_between_fit_transforms(
        self, X, y, _args, _kwargs, _rows, mmct
    ):

        alt_args = [_rows // 25]
        alt_kwargs = {
            'ignore_float_columns': True,
            'ignore_non_binary_integer_columns': True,
            'ignore_columns': [0, 2],
            'ignore_nan': False,
            'delete_axis_0': False,
            'handle_as_bool': None,
            'reject_unseen_values': False,
            'max_recursions': 1,
            'n_jobs': -1
        }


        # INITIALIZE->FIT_TRFM
        IFTCls = MCT(*alt_args, **alt_kwargs)
        SPFT_TRFM_X, SPFT_TRFM_Y = IFTCls.fit_transform(X.copy(), y.copy())

        # FIT_TRFM->SET_PARAMS->FIT_TRFM
        FTSPFTCls = MCT(*alt_args, **alt_kwargs)
        FTSPFTCls.set_params(max_recursions=2)
        FTSPFTCls.fit_transform(X.copy(), y.copy())
        assert FTSPFTCls._max_recursions == 2
        FTSPFTCls.set_params(**alt_kwargs)
        FTSPFT_TRFM_X, FTSPFT_TRFM_Y = \
            FTSPFTCls.fit_transform(X.copy(), y.copy())
        assert FTSPFTCls._max_recursions == 1

        assert np.array_equiv(
            SPFT_TRFM_X.astype(str), FTSPFT_TRFM_X.astype(str)
        ), \
            f"SPFT_TRFM_X != FTSPFT_TRFM_X"

        assert np.array_equiv(SPFT_TRFM_Y, FTSPFT_TRFM_Y), \
            f"SPFT_TRFM_Y != FTSPFT_TRFM_Y"



@pytest.mark.skip(reason=f'pizza')
@pytest.mark.parametrize('top_level_object',
    ('single_est', 'single_trfm')
)
@pytest.mark.parametrize('state', ('pre-init', 'pre-fit', 'post-fit'))
class TestSetParams(Fixtures):


    @staticmethod
    @pytest.fixture(scope='function')
    def _kwargs(top_level_object, _est_kwargs, _trfm_kwargs, _gscv_kwargs):

        if top_level_object == 'single_est':
            return _est_kwargs
        elif top_level_object == 'single_trfm':
            return _trfm_kwargs
        elif top_level_object == 'GSCV_est':
            return _gscv_kwargs
        else:
            raise Exception


    @staticmethod
    @pytest.fixture(scope='function')
    def TopLevelObject(
        top_level_object, state, DummyEstimator, DummyTransformer,
        DummyGridSearch, _X_np, _kwargs
    ):

        if top_level_object == 'single_est':
            foo = DummyEstimator
        elif top_level_object == 'single_trfm':
            foo = DummyTransformer
        elif top_level_object == 'GSCV_est':
            foo = DummyGridSearch
        else:
            raise Exception

        if state == 'pre-init':
            pass
        elif state == 'pre-fit':
            foo(**_kwargs)
        elif state == 'post-fit':
            foo(**_kwargs).fit(_X_np)

        return foo




    # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    # ^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v


    def test_init_was_correctly_applied(self, TopLevelObject, _kwargs, state):

        if state == 'pre-init':
            pytest.skip(reason='not init-ed')

        # assert TestCls initiated correctly
        for _param, _value in _kwargs.items():
            assert getattr(TopLevelObject, _param) == _value


    @pytest.mark.parametrize(f'junk_param',
        ({'garbage': True}, {'estimator__trash': True})
    )
    def test_rejects_unknown_param(
        self, TopLevelObject, state, junk_param
    ):

        with pytest.raises(ValueError):
            TopLevelObject.set_params(**junk_param)


    def test_set_params_correctly_applies(self, TopLevelObject, _kwargs, state):


        # set new params before fit
        _scd_params = {'count_threshold': _args[0]} | _kwargs
        _scd_params['count_threshold'] = 39
        _scd_params['ignore_float_columns'] = False
        _scd_params['ignore_non_binary_integer_columns'] = False
        _scd_params['ignore_columns'] = [0, 1]
        _scd_params['ignore_nan'] = False
        _scd_params['delete_axis_0'] = False
        _scd_params['handle_as_bool'] = [2, 3]
        _scd_params['reject_unseen_values'] = True
        _scd_params['max_recursions'] = 1   # <==== no change
        _scd_params['n_jobs'] = 2

        TestCls.set_params(**_scd_params)

        # assert new values set correctly
        for _param, _value in _scd_params.items():
            assert getattr(TestCls, _param) == _value



