# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV

from pybear.model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask as \
    dask_GSTCV


pytest.skip(
    reason=f"only test fixtures if they are changed",
    allow_module_level=True
)


class TestDaskGridSearchFixtures:


    # 24_08_24 dask GSCV doesnt allow refit fxn w 2+ scorers (but does w 1)
    # hash them out here, but keep the fixtures in the conftest file in
    # case someday


    def test_single_estimators(self,
        dask_est_log,
        dask_GSCV_est_log_one_scorer_prefit,
        dask_GSCV_est_log_one_scorer_postfit_refit_false,
        dask_GSCV_est_log_one_scorer_postfit_refit_str,
        dask_GSCV_est_log_one_scorer_postfit_refit_fxn,
        dask_GSTCV_est_log_one_scorer_prefit,
        dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_da,
        dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da,
        dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_da,
        dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_ddf,
        dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_ddf,
        dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_ddf,
        dask_GSCV_est_log_two_scorers_prefit,
        dask_GSCV_est_log_two_scorers_postfit_refit_false,
        dask_GSCV_est_log_two_scorers_postfit_refit_str,
        # dask_GSCV_est_log_two_scorers_postfit_refit_fxn,  # THIS FAILS
        dask_GSTCV_est_log_two_scorers_prefit,
        dask_GSTCV_est_log_two_scorers_postfit_refit_false_fit_on_da,
        dask_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_da,
        dask_GSTCV_est_log_two_scorers_postfit_refit_fxn_fit_on_da,
        dask_GSTCV_est_log_two_scorers_postfit_refit_false_fit_on_ddf,
        dask_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_ddf,
        dask_GSTCV_est_log_two_scorers_postfit_refit_fxn_fit_on_ddf,
        # _client
    ):

        # 25_04_13 time is same w/ & w/o client

        name_gscv_tuples = [
            (f'dask_GSCV_est_log_one_scorer_prefit',
             dask_GSCV_est_log_one_scorer_prefit),
            (f'dask_GSCV_est_log_one_scorer_postfit_refit_false',
             dask_GSCV_est_log_one_scorer_postfit_refit_false),
            (f'dask_GSCV_est_log_one_scorer_postfit_refit_str',
             dask_GSCV_est_log_one_scorer_postfit_refit_str),
            (f'dask_GSCV_est_log_one_scorer_postfit_refit_fxn',
             dask_GSCV_est_log_one_scorer_postfit_refit_fxn),
            (f'dask_GSTCV_est_log_one_scorer_prefit',
             dask_GSTCV_est_log_one_scorer_prefit),
            (f'dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_da',
             dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_da),
            (f'dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da',
             dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_da),
            (f'dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_da',
             dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_da),
            (f'dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_ddf',
             dask_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_ddf),
            (f'dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_ddf',
             dask_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_ddf),
            (f'dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_ddf',
             dask_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_ddf),
            (f'dask_GSCV_est_log_two_scorers_prefit',
             dask_GSCV_est_log_two_scorers_prefit),
            (f'dask_GSCV_est_log_two_scorers_postfit_refit_false',
             dask_GSCV_est_log_two_scorers_postfit_refit_false),
            (f'dask_GSCV_est_log_two_scorers_postfit_refit_str',
             dask_GSCV_est_log_two_scorers_postfit_refit_str),
            # THIS FAILS
            # (f'dask_GSCV_est_log_two_scorers_postfit_refit_fxn',
            #  dask_GSCV_est_log_two_scorers_postfit_refit_fxn),
            (f'dask_GSTCV_est_log_two_scorers_prefit',
             dask_GSTCV_est_log_two_scorers_prefit),
            (f'dask_GSTCV_est_log_two_scorers_postfit_refit_false_fit_on_da',
             dask_GSTCV_est_log_two_scorers_postfit_refit_false_fit_on_da),
            (f'dask_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_da',
             dask_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_da),
            (f'dask_GSTCV_est_log_two_scorers_postfit_refit_fxn_fit_on_da',
             dask_GSTCV_est_log_two_scorers_postfit_refit_fxn_fit_on_da),
            (f'dask_GSTCV_est_log_two_scorers_postfit_refit_false_fit_on_ddf',
             dask_GSTCV_est_log_two_scorers_postfit_refit_false_fit_on_ddf),
            (f'dask_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_ddf',
             dask_GSTCV_est_log_two_scorers_postfit_refit_str_fit_on_ddf),
            (f'dask_GSTCV_est_log_two_scorers_postfit_refit_fxn_fit_on_ddf',
             dask_GSTCV_est_log_two_scorers_postfit_refit_fxn_fit_on_ddf)
        ]

        for idx, (_name, _gscv_or_gstcv) in enumerate(name_gscv_tuples):

            __ = _gscv_or_gstcv

            if 'GSCV' in _name:
                assert isinstance(__, dask_GridSearchCV)
            elif 'GSTCV' in _name:
                assert isinstance(__, dask_GSTCV)

            _est = getattr(__, 'estimator')

            assert isinstance(_est, type(dask_est_log))

            _scoring = getattr(__, 'scoring')
            if 'one_scorer' in _name:
                assert isinstance(_scoring, str) or len(_scoring) == 1
            elif 'two_scorers' in _name:
                assert len(_scoring) == 2

            _refit = getattr(__, 'refit')
            if 'prefit' in _name:
                assert _refit is False
            elif 'postfit_refit_false' in _name:
                assert _refit is False
            elif 'postfit_refit_str' in _name:
                assert isinstance(_refit, str)
            elif 'postfit_refit_fxn' in _name:
                assert callable(_refit)
            else:
                raise Exception(f"invalid fixture name '{_name}'")


            if 'prefit' in _name:
                assert not hasattr(__, 'scorer_')

            elif 'postfit' in _name:
                assert hasattr(__, 'scorer_')

                if 'refit_false' in _name:
                    assert not hasattr(__, 'best_estimator_')
                elif 'refit_str' in _name or 'refit_fxn' in _name:
                    assert hasattr(__, 'best_estimator_')








