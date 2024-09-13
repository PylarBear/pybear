# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as ddf
import dask_expr._collection as ddf2
import sys
from pathlib import Path

from distributed import Client

from helper_functions import (
    init_gscv,
    method_output_try_handler,
    method_output_except_handler
)
from make_data import make_data
from est__param_grid import make_est__param_grid




# THIS MODULE CAPTURES THE OUTPUT (VALUES OR EXC INFO) OF METHOD CALLS
# TO SK GSCV, DASK GSCV, GSTCV, GSTCVDask AFTER fit() WITH ARRAYS AND
# DATAFRAMES IN VARIOUS STATES OF GOOD OR BAD.

# THIS DOES NOT TEST METHODS THAT DONT TAKE X, LIKE get_metadata_routing,
# get_params, set_params.


# X
# decision_function(X)
# inverse_transform(Xt)
# predict(X)
# predict_log_proba(X)
# predict_proba(X)
# score_samples(X)
# transform(X)



if __name__ == '__main__':

    with Client(n_workers=None, threads_per_worker=1):

        _rows, _cols = 100, 3

        DD = make_data(_rows, _cols)

        GOOD_OR_BAD_X = ['good', 'bad_features', 'bad_data']
        DTYPES = ['array', 'dataframe']
        TYPES = ['sklearn', 'dask', 'gstcv_sklearn', 'gstcv_dask']

        sk_clf, dask_clf, _param_grid = make_est__param_grid()

        METHOD_NAMES = [
            'decision_function',
            'inverse_transform',
            'predict',
            'predict_log_proba',
            'predict_proba',
            'score_samples',
            'transform'
        ]

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        ALL_OTHER_COMBINATIONS = []
        for e in DTYPES:  # for fit
            for d in GOOD_OR_BAD_X:
                for b in DTYPES:  # for method
                    for a in TYPES:
                        ALL_OTHER_COMBINATIONS.append(f'fit_{e}__{d}_{b}_X_{a}')

        ALL_OTHER_METHOD_ARRAY_DICT = {
            k: pd.DataFrame(
                index=METHOD_NAMES,
                columns=['OUTPUT'],
                dtype=object
            ) for k in ALL_OTHER_COMBINATIONS
        }



        ctr = 0
        for _fit_dtype in DTYPES:

            for _gscv_type in TYPES:

                ctr += 1

                print(f'Fitting {ctr} of {len(TYPES)*len(DTYPES)}...')
                print(f'trial = {_gscv_type}, fit type = {_fit_dtype}')

                # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

                if 'dask' in _gscv_type:
                    if _fit_dtype == 'array':
                        base_X = DD['good_da_X']
                        base_y = DD['good_da_y']
                    elif _fit_dtype == 'dataframe':
                        base_X = DD['good_ddf_X']
                        base_y = DD['good_da_y']
                elif 'sklearn' in _gscv_type:
                    if _fit_dtype == 'array':
                        base_X = DD['good_np_X']
                        base_y = DD['good_np_y']
                    elif _fit_dtype == 'dataframe':
                        base_X = DD['good_pd_X']
                        base_y = DD['good_pd_y']

                test_cls = init_gscv(
                    sk_clf,
                    dask_clf,
                    _gscv_type,
                    _param_grid,
                    'balanced_accuracy',
                    'balanced_accuracy'
                )

                test_cls.fit(base_X, base_y)

                for _dtype in DTYPES:

                    if 'dask' in _gscv_type:
                        if 'array' in _dtype:
                            _format = 'da'
                        elif 'dataframe' in _dtype:
                            _format = 'ddf'
                    elif 'sklearn' in _gscv_type:
                        if 'array' in _dtype:
                            _format = 'np'
                        elif 'dataframe' in _dtype:
                            _format = 'pd'

                    for good_or_bad_x in GOOD_OR_BAD_X:

                        _X = DD[f'{good_or_bad_x}_{_format}_X']

                        trial = f'fit_{_fit_dtype}__{good_or_bad_x}_{_dtype}_X_{_gscv_type}'

                        if 'dask' in _gscv_type:
                            if _fit_dtype == 'array':
                                assert isinstance(base_X, da.core.Array)
                                assert isinstance(base_y, da.core.Array)
                            elif _fit_dtype == 'dataframe':
                                assert isinstance(base_X, ddf2.DataFrame)
                                assert isinstance(base_y, da.core.Array)
                        elif 'sklearn' in _gscv_type:
                            if _fit_dtype == 'array':
                                assert isinstance(base_X, np.ndarray)
                                assert isinstance(base_y, np.ndarray)
                            elif _fit_dtype == 'dataframe':
                                assert isinstance(base_X, pd.core.frame.DataFrame)
                                assert isinstance(base_y, pd.core.frame.DataFrame)

                        for method in METHOD_NAMES:

                            try:
                                try:
                                    __ = getattr(test_cls, method)(_X).compute()
                                except:
                                    __ = getattr(test_cls, method)(_X)

                                ALL_OTHER_METHOD_ARRAY_DICT = \
                                    method_output_try_handler(
                                        trial,
                                        method,
                                        __,
                                        ALL_OTHER_METHOD_ARRAY_DICT
                                    )
                            except:
                                ALL_OTHER_METHOD_ARRAY_DICT = \
                                    method_output_except_handler(
                                        trial,
                                        method,
                                        sys.exc_info()[1],
                                        ALL_OTHER_METHOD_ARRAY_DICT
                                    )


            ALL_OTHER_SINGLE_DF = pd.DataFrame(
                index=METHOD_NAMES,
                columns=list(ALL_OTHER_METHOD_ARRAY_DICT.keys()),
                dtype='<U100'
            ).fillna('-')

            for _key, DATA_DF in ALL_OTHER_METHOD_ARRAY_DICT.items():
                ALL_OTHER_SINGLE_DF.loc[:, _key] = DATA_DF.to_numpy().ravel()

            ALL_OTHER_SINGLE_DF = ALL_OTHER_SINGLE_DF.T

            desktop_path = Path.home() / "Desktop"
            filename = 'gscv_bad_X_comparison_dump__all_methods.csv'
            file_path = desktop_path / filename

            ALL_OTHER_SINGLE_DF.to_csv(file_path, index=True)

            del ALL_OTHER_SINGLE_DF











