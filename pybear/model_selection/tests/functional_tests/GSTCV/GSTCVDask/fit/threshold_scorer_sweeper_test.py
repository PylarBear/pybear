# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import pandas as pd
import dask.array as da

from model_selection.GSTCV._master_scorer_dict import master_scorer_dict

from model_selection.GSTCV._GSTCVMixin._fit._threshold_scorer_sweeper import \
    _threshold_scorer_sweeper



class TestThresholdScorerSweeper:

    # @joblib.wrap_non_picklable_objects
    # def _threshold_scorer_sweeper(
    #         threshold: Union[float, int],
    #         y_test: YWIPType,
    #         _predict_proba: YWIPType,
    #         SCORER_DICT: ScorerWIPType,
    #         **scorer_params
    #     ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:


    y_np_test_data = np.random.randint(0, 2, (100, 1))

    y_da_test_data = da.random.randint(0, 2, (100, 1))

    y_np_predict_proba = np.random.uniform(0, 1, (100, 1))

    y_da_predict_proba = da.random.uniform(0, 1, (100, 1))






    @pytest.mark.parametrize('junk_threshold',
        (-1, 3.14, 'trash', min, True, None, [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    @pytest.mark.parametrize('y_test', (y_np_test_data, y_da_test_data))
    @pytest.mark.parametrize('predict_proba', (y_np_predict_proba, y_da_predict_proba))
    def test_rejects_junk_threshold(self, junk_threshold, y_test, predict_proba):
        with pytest.raises(ValueError):
            _threshold_scorer_sweeper(
                junk_threshold,
                y_test,
                predict_proba,
                master_scorer_dict
            )


    @pytest.mark.parametrize('junk_scorer_dict',
        (-1, 3.14, 'trash', min, True, None, [0,1], (0,1), {0,1}, lambda x: x)
    )
    @pytest.mark.parametrize('y_test', (y_np_test_data, y_da_test_data))
    @pytest.mark.parametrize('predict_proba', (y_np_predict_proba, y_da_predict_proba))
    def test_rejects_junk_scorer_dict(self, junk_scorer_dict, y_test, predict_proba):
        with pytest.raises(AssertionError):
            _threshold_scorer_sweeper(
                0.5,
                y_test,
                predict_proba,
                junk_scorer_dict
            )


    @pytest.mark.parametrize('junk_y_test',
        (-1, 3.14, 'trash', min, True, None, [0,1], (0,1), {0,1}, lambda x: x,
         pd.DataFrame(data=np.random.randint(0,10,(100,30))))
    )
    @pytest.mark.parametrize('predict_proba', (y_np_predict_proba, y_da_predict_proba))
    def test_rejects_non_dask_np_ytest(self, junk_y_test, predict_proba):
        with pytest.raises(AssertionError):
            _threshold_scorer_sweeper(
                0.5,
                junk_y_test,
                predict_proba,
                master_scorer_dict
            )


    @pytest.mark.parametrize('junk_predict_proba',
        (-1, 3.14, 'trash', min, True, None, [0,1], (0,1), {0,1}, lambda x: x,
         pd.DataFrame(data=np.random.randint(0,10,(100,30))))
    )
    @pytest.mark.parametrize('y_test', (y_np_test_data, y_da_test_data))
    def test_rejects_non_dask_np_predict_proba(self, y_test, junk_predict_proba):
        with pytest.raises(AssertionError):
            _threshold_scorer_sweeper(
                .5,
                y_test,
                junk_predict_proba,
                master_scorer_dict
            )



    @pytest.mark.parametrize('y_test', (y_np_test_data, y_da_test_data))
    @pytest.mark.parametrize('predict_proba', (y_np_predict_proba, y_da_predict_proba))
    def test_good_ytest_predict_proba(self, y_test, predict_proba):
        with pytest.raises(ValueError):
            _threshold_scorer_sweeper(
                0.5,
                y_test,
                predict_proba,
                master_scorer_dict
            )


    @pytest.mark.parametrize('threshold', (0, 0.5, 1))
    @pytest.mark.parametrize('y_test', (y_np_test_data, y_da_test_data))
    @pytest.mark.parametrize('predict_proba', (y_np_predict_proba, y_da_predict_proba))
    def test_good_ytest_predict_proba(self, threshold, y_test, predict_proba):

        out_scores, out_times = _threshold_scorer_sweeper(
            threshold,
            y_test,
            predict_proba,
            master_scorer_dict
        )

        assert isinstance(out_scores, np.ndarray)
        assert np.min(out_scores) >= 0
        assert np.max(out_scores) <= 1

        assert isinstance(out_times, np.ndarray)
        assert np.min(out_times) > 0








