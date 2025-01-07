# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import sys
from pathlib import Path

import pandas as pd

from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder
from sklearn.linear_model import (
    LogisticRegression as sk_Logistic,
    LinearRegression as sk_LinearRegression
)
from sklearn.model_selection import GridSearchCV as sk_GridSearchCV


from dask_ml.preprocessing import OneHotEncoder as dask_OneHotEncoder
from dask_ml.linear_model import (
    LogisticRegression as dask_Logistic,
    LinearRegression as dask_LinearRegression
)
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV


from model_selection.GSTCV._GSTCV.GSTCV import GSTCV
from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask


from sklearn.pipeline import Pipeline






PIPE_OR_EST = ['pipe', 'est']
SK_OR_DASK = ['sk', 'dask']
SHALLOW_OR_DEEP = ['shallow', 'deep']
GOOD_OR_BAD_PARAMS = ['bad_gscv_params'] #['good_params', 'bad_gscv_params', 'bad_estimator_params']

sk_steps = [
    ('sk_onehot', sk_OneHotEncoder()),
    ('sk_logistic', sk_Logistic())
]
dask_steps = [
    ('dask_onehot', dask_OneHotEncoder()),
    ('dask_logistic', dask_Logistic())
]

ctr = 0
GSCV_OUTPUT = []
GSTCV_OUTPUT = []
DF_INDEX = []
for pipe_or_est in PIPE_OR_EST:
    for sk_or_dask in SK_OR_DASK:

        if pipe_or_est == 'pipe':
            _estimator = Pipeline(
                steps = sk_steps if sk_or_dask == 'sk' else dask_steps
            )
        elif pipe_or_est == 'est':
            _estimator = sk_Logistic() if sk_or_dask == 'sk' else dask_Logistic()

        if sk_or_dask == 'sk':
            GSCV = sk_GridSearchCV(estimator=_estimator, param_grid={})
            WIP_GSTCV = GSTCV(estimator=_estimator, param_grid={})

        elif sk_or_dask == 'dask':
            GSCV = dask_GridSearchCV(estimator=_estimator, param_grid={})
            WIP_GSTCV = GSTCVDask(estimator=_estimator, param_grid={})

        for shallow_or_deep in SHALLOW_OR_DEEP:
            _deep = True if shallow_or_deep == 'deep' else False
            for good_or_bad_params in GOOD_OR_BAD_PARAMS:

                ctr += 1
                itr = f'{pipe_or_est}_{sk_or_dask}_{shallow_or_deep}_{good_or_bad_params}'
                print(f'running iter {ctr}: {itr}')
                DF_INDEX.append(itr)

                if pipe_or_est == 'pipe':
                    assert hasattr(GSCV.estimator, 'steps')
                    assert hasattr(WIP_GSTCV.estimator, 'steps')
                else:
                    assert not hasattr(GSCV.estimator, 'steps')
                    assert not hasattr(WIP_GSTCV.estimator, 'steps')

                if good_or_bad_params == 'good_params':
                    params = GSCV.get_params(deep=_deep)
                elif good_or_bad_params == 'bad_estimator_params':
                    params = sk_LinearRegression().get_params(deep=True)
                    params = {f'estimator__{k}':v for k,v in params.items()}
                elif good_or_bad_params == 'bad_gscv_params':
                    params = sk_LinearRegression().get_params(deep=True)

                # sk / dask output ** * ** * ** * ** * ** * ** * ** * ** *
                try:
                    out = GSCV.set_params(**params)
                    GSCV_OUTPUT.append(out)
                except Exception as e:
                    GSCV_OUTPUT.append(f'{sys.exc_info()[1]!r}')
                # END sk / dask output ** * ** * ** * ** * ** * ** * ** *

                # GSTCV output ** * ** * ** * ** * ** * ** * ** * ** *
                try:
                    out = WIP_GSTCV.set_params(**params)
                    GSTCV_OUTPUT.append(out)
                except Exception as e:
                    GSTCV_OUTPUT.append(f'{sys.exc_info()[1]!r}')
                # END sk / dask output ** * ** * ** * ** * ** * ** * ** *





DF = pd.DataFrame(index=DF_INDEX, columns=['GSCV_OUTPUT', 'GSTCV_OUTPUT'])
DF.loc[:, 'GSCV_OUTPUT'] = GSCV_OUTPUT
DF.loc[:, 'GSTCV_OUTPUT'] = GSTCV_OUTPUT

desktop_path = Path.home() / "Desktop"
filename = r'set_params_output_comparison.csv'
path = desktop_path / filename
DF.to_csv(path)

































