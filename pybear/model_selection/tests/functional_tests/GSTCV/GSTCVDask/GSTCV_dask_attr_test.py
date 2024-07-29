# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

import numpy as np
import dask.array as da
from distributed import Client
from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask




class TestGSTCVDaskAttr:


    @pytest.fixture
    @staticmethod
    def param_grid():
        # use pytest mock classifier params!
        return [{'ctr': [1,2,3], 'sleep': np.linspace(0.1,0.3,3)},
                  {'ctr': [4,5,6], 'sleep': np.linspace(0.3,0.5,3)}]


    @staticmethod
    def refit_1(_cv_results):
        _len = len(_cv_results['rank_test_balanced_accuracy'])
        return np.arange(_len)[_cv_results['rank_test_balanced_accuracy']==1][0]


    @staticmethod
    def refit_2(_cv_results):
        return 0


    @pytest.fixture
    @staticmethod
    def _client():
        client = Client(n_workers=1, threads_per_worker=2)
        yield client
        client.close()


    @pytest.mark.parametrize('_refit', (refit_1, refit_2, 'balanced_accuracy', None))
    def test_dask_GSTCV(self, _refit, _mock_classifier, param_grid, _client):

        TestCls = GSTCVDask(
            _mock_classifier,
            param_grid,
            scoring=['accuracy', 'balanced_accuracy'],
            thresholds=np.linspace(0.1,0.9,3),
            n_jobs=-1,
            cv=3,
            refit=_refit,
            error_score=np.nan,
            verbose=0,
            return_train_score=True,
            iid=True,
            scheduler=None,
            cache_cv=False
        )

        ATTR_NAMES = [
            'cv_results_',
            'best_estimator_',
            'best_score_',
            'best_params_',
            'best_index_',
            'scorer_',
            'n_splits_',
            'refit_time_',
            # pizza what about these?
            # 'multimetric_',
            # 'classes_',
            # 'n_features_in_',
            # 'feature_names_in_'
        ]

        METHOD_NAMES = [
            'decision_function',
            'fit',
            'get_params',
            'inverse_transform',
            'predict',
            'predict_log_proba',
            'predict_proba',
            'score',
            'set_params',
            'transform',
            'visualize'
        ]


        # PRE-FIT ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

        # has no attributes
        for attr in ATTR_NAMES:
            assert not hasattr(TestCls, attr)

        # has all methods
        for meth in METHOD_NAMES:
            assert hasattr(TestCls, meth)

        # END PRE-FIT ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


        _X = da.random.randint(0,10,(100,10))
        _y = da.random.randint(0,2,(100,1))

        with _client:
            TestCls.fit(_X, _y)


        # AFTER FIT ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        if _refit is None:

            for attr in ATTR_NAMES:
                if attr in ['cv_results_', 'scorer_', 'n_splits_']:
                    assert hasattr(TestCls, attr)
                else:
                    assert not hasattr(TestCls, attr)

        else:
            if callable(_refit):
                ATTR_NAMES.remove('best_score_')

            # conditionally has attributes
            for attr in ATTR_NAMES:
                assert hasattr(TestCls, attr)

        # has all methods
        for meth in METHOD_NAMES:
            assert hasattr(TestCls, meth)

        # END AFTER FIT ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
















