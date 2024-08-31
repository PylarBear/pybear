# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pandas as pd
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




# THIS MODULE CAPTURES THE OUTPUT (VALUES OR EXC INFO) OF score() CALLS
# TO SK GSCV, DASK GSCV, GSTCV, GSTCVDask AFTER fit() WITH ARRAYS AND
# DATAFRAMES IN VARIOUS STATES OF GOOD OR BAD.





# X, y
# score(X, y)









if __name__ == '__main__':

    with Client(n_workers=None, threads_per_worker=1):

        _rows, _cols = 100, 3

        DD = make_data(_rows, _cols)

        GOOD_OR_BAD_X = ['good', 'bad_features', 'bad_data', 'bad_rows']
        GOOD_OR_BAD_y = [
            'good', 'bad_features', 'bad_classes', 'bad_data', 'bad_rows'
        ]
        DTYPES = ['array', 'dataframe']
        TYPES = ['sklearn', 'dask', 'gstcv_sklearn', 'gstcv_dask']

        sk_clf, dask_clf, _param_grid = make_est__param_grid()

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        SCORE_COMBINATIONS = []
        for e in DTYPES:    # type used for fit
            for d in GOOD_OR_BAD_X:
                for b in DTYPES:    # type passed to method
                    for c in GOOD_OR_BAD_y:
                        for a in TYPES:
                            if 'bad_rows' in d and 'bad_rows' in c:
                                continue
                            SCORE_COMBINATIONS.append(
                                f'fit_{e}__{d}_{b}_X_{c}_{b}_y_{a}'
                            )

        SCORE_METHOD_ARRAY_DICT = {
            k: pd.DataFrame(
                index=['score - 1 scorer', 'score - 2 scorers'],
                columns=['OUTPUT'],
                dtype=object
            ) for k in SCORE_COMBINATIONS
        }


        ctr = 0
        for _fit_dtype in DTYPES:
            for _scoring in (['balanced_accuracy'], ['accuracy', 'balanced_accuracy']):
                for _gscv_type in TYPES:

                    ctr += 1

                    print(f'Fitting {ctr} of {2 * 2 * len(TYPES)}...')
                    print(f'fit type = {_fit_dtype}, gscv = {_gscv_type}, '
                          f'scoring = {_scoring}'
                    )

                    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

                    if _fit_dtype == 'array':
                        if 'dask' in _gscv_type:
                            base_X = DD['good_da_X']
                            base_y = DD['good_da_y']
                        elif 'sklearn' in _gscv_type:
                            base_X = DD['good_np_X']
                            base_y = DD['good_np_y']
                    elif _fit_dtype == 'dataframe':
                        if 'dask' in _gscv_type:
                            base_X = DD['good_ddf_X']
                            base_y = DD['good_da_y']
                        elif 'sklearn' in _gscv_type:
                            base_X = DD['good_pd_X']
                            base_y = DD['good_pd_y']

                    test_cls = init_gscv(
                        sk_clf,
                        dask_clf,
                        _gscv_type,
                        _param_grid,
                        _scoring[0] if len(_scoring) == 1 else _scoring,
                        'balanced_accuracy'
                    )

                    test_cls.fit(base_X, base_y)

                    del base_X, base_y

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

                            for good_or_bad_y in GOOD_OR_BAD_y:

                                if 'bad_rows' in good_or_bad_x and \
                                    'bad_rows' in good_or_bad_y:
                                    continue

                                _y = DD[f'{good_or_bad_y}_{_format}_y']

                                trial = f'fit_{_fit_dtype}__'
                                trial += f'{good_or_bad_x}_{_dtype}_X_'
                                trial += f'{good_or_bad_y}_{_dtype}_y_{_gscv_type}'

                                method = f'score - {len(_scoring)} scorer'
                                method += 's' if len(_scoring) == 2 else ''

                                try:
                                    __ = getattr(test_cls, 'score')(_X, _y)
                                    try:
                                        __.compute()
                                    except:
                                        pass

                                    SCORE_METHOD_ARRAY_DICT = \
                                        method_output_try_handler(
                                            trial,
                                            method,
                                            __,
                                            SCORE_METHOD_ARRAY_DICT
                                        )
                                except:
                                    SCORE_METHOD_ARRAY_DICT = \
                                        method_output_except_handler(
                                            trial,
                                            method,
                                            sys.exc_info()[1],
                                            SCORE_METHOD_ARRAY_DICT
                                        )


        SCORE_SINGLE_DF = pd.DataFrame(
            index=['score - 1 scorer', 'score - 2 scorers'],
            columns=list(SCORE_METHOD_ARRAY_DICT.keys()),
            dtype='<U100'
        ).fillna('-')

        for _key, DATA_DF in SCORE_METHOD_ARRAY_DICT.items():
            SCORE_SINGLE_DF.loc[:, _key] = DATA_DF.to_numpy().ravel()

        SCORE_SINGLE_DF = SCORE_SINGLE_DF.T

        desktop_path = Path.home() / "Desktop"
        filename = 'gscv_bad_X_bad_y_comparison_dump__score.csv'
        file_path = desktop_path / filename

        SCORE_SINGLE_DF.to_csv(file_path, index=True)

        del SCORE_SINGLE_DF





