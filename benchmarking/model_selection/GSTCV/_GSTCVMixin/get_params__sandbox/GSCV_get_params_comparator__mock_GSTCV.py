# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV
from sklearn.linear_model import \
    LogisticRegression as sklearn_LogisticRegression
from sklearn.linear_model import LinearRegression as sklearn_LinearRegression
from sklearn.pipeline import Pipeline

from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression
from dask_ml.linear_model import LinearRegression as dask_LinearRegression



# MAKE _GSTCV get_params deep & not deep LOOK THE SAME AS sk/da-gscv get_params **

# FOR A STAND-ALONE ESTIMATOR, deep & not deep ARE THE SAME
# test_estimator = LogisticRegression()
# test_estimator.get_params(deep=True)
# test_estimator.get_params(deep=False)

# FOR GSCV WITH 1 ESTIMATOR, deep & not deep ARE DIFFERENT



# MOCK THE _GSTCV CLASS AND ITS get_params METHOD ** ** ** ** ** ** ** **
class mock_GSTCV:


    def __init__(self, estimator, param_grid):
        self.estimator = estimator
        self.param_grid = param_grid

        if 'pipe' in str(type(estimator)).lower():
            self.dask_estimator = False
            for (_name, _est) in self.estimator.steps:
                if 'dask' in str(type(_est)).lower():
                    self.dask_estimator = True
        else:
            self.dask_estimator = 'dask' in str(type(estimator)).lower()

        self.thresholds = np.linspace(0, 1, 11)
        self.scoring = None
        self.n_jobs = -1 if self.dask_estimator else None
        if not self.dask_estimator: self.pre_dispatch = '2*n_jobs'
        self.cv = None
        self.refit = True
        if not self.dask_estimator: self.verbose = 0
        self.error_score = 'raise' if self.dask_estimator else np.nan
        self.return_train_score = False
        self.random_state = None
        if self.dask_estimator: self.iid = True
        if self.dask_estimator: self.scheduler = None
        if self.dask_estimator: self.cache_cv = True


    def get_params(self, deep=True):

        paramsdict = {}

        paramsdict['estimator'] = self.estimator
        if self.dask_estimator: paramsdict['cache_cv'] = self.cache_cv
        paramsdict['cv'] = self.cv
        paramsdict['error_score'] = self.error_score
        if self.dask_estimator: paramsdict['iid'] = self.iid
        paramsdict['n_jobs'] = self.n_jobs
        paramsdict['param_grid'] = self.param_grid
        if not self.dask_estimator: paramsdict[
            'pre_dispatch'] = self.pre_dispatch
        paramsdict['refit'] = self.refit
        paramsdict['return_train_score'] = self.return_train_score
        if self.dask_estimator: paramsdict['scheduler'] = self.scheduler
        paramsdict['scoring'] = self.scoring
        paramsdict['thresholds'] = self.thresholds
        if not self.dask_estimator: paramsdict['verbose'] = self.verbose

        # CORRECT FOR BOTH SIMPLE ESTIMATOR OR PIPELINE
        if deep:
            for k, v in self.estimator.get_params(deep=True).items():
                paramsdict[f'estimator__{k}'] = v

        # ALPHABETIZE paramsdict
        paramsdict = {k: paramsdict.pop(k) for k in sorted(paramsdict)}

        return paramsdict
# END MOCK _GSTCV CLASS ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


# SET COMPARISON PARAMETERS ** * ** * ** * ** * ** * ** * ** * ** * ** *
show_sklearn = True
show_dask = True
if show_sklearn + show_dask == 0:
    raise KeyboardInterrupt(f"show_sklearn and show_dask cannot both be False")
# END SET COMPARISON PARAMETERS ** * ** * ** * ** * ** * ** * ** * ** *


# build single estimators ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
sklearn_est = sklearn_LogisticRegression()
sklearn_one_est_param_grid = {'C': [1e-3, 1e-2, 1e-1], 'tol': [1e-9, 1e-8, 1e-7]}

sklearn_gscv_single_est = sklearn_GridSearchCV(
    estimator=sklearn_est,
    param_grid=sklearn_one_est_param_grid
)
sklearn_mock_gstcv_single_est = mock_GSTCV(
    estimator=sklearn_est,
    param_grid=sklearn_one_est_param_grid
)
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
dask_est = dask_LogisticRegression()
dask_one_est_param_grid = {'C': [1e-3, 1e-2, 1e-1], 'tol': [1e-9, 1e-8, 1e-7]}

dask_gscv_single_est = dask_GridSearchCV(
    estimator=dask_est,
    param_grid=dask_one_est_param_grid
)
dask_mock_gstcv_single_est = mock_GSTCV(
    estimator=dask_est,
    param_grid=dask_one_est_param_grid
)
# END build single estimators ** * ** * ** * ** * ** * ** * ** * ** * **

# build pipelines ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
sklearn_pipe = Pipeline(
    steps=[
        ('pipe_class_1', sklearn_LogisticRegression()),
        ('pipe_class_2', sklearn_LinearRegression())
    ],
    verbose=True
)

sklearn_pipe_param_grid = {
    'pipe_class_1__C': [1e-3, 1e-2, 1e-1],
    'pipe_class_1__tol': [1e-9, 1e-8, 1e-7],
    'pipe_class_2__C': [1e-3, 1e-2, 1e-1],
    'pipe_class_2__tol': [1e-9, 1e-8, 1e-7],
}

sklearn_gscv_pipe_est = sklearn_GridSearchCV(
    estimator=sklearn_pipe,
    param_grid=sklearn_pipe_param_grid
)
sklearn_mock_gstcv_pipe_est = mock_GSTCV(
    estimator=sklearn_pipe,
    param_grid=sklearn_pipe_param_grid
)
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
dask_pipe = Pipeline(
    steps=[
        ('pipe_class_1', dask_LogisticRegression()),
        ('pipe_class_2', dask_LinearRegression())
    ],
    verbose=True
)

dask_pipe_param_grid = {
    'pipe_class_1__C': [1e-3, 1e-2, 1e-1],
    'pipe_class_1__tol': [1e-9, 1e-8, 1e-7],
    'pipe_class_2__C': [1e-3, 1e-2, 1e-1],
    'pipe_class_2__tol': [1e-9, 1e-8, 1e-7],
}

dask_gscv_pipe_est = dask_GridSearchCV(
    estimator=dask_pipe,
    param_grid=dask_pipe_param_grid
)
dask_mock_gstcv_pipe_est = mock_GSTCV(
    estimator=dask_pipe,
    param_grid=dask_pipe_param_grid
)
# END build pipelines ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


while True:
    __ = input(f"\n(d)ump to csv or (p)rint to screen or (s)skip > ").lower()
    if __ not in 'dps':
        print(f'\ntry again. must be d p or s.')
    else:
        break

DF_DICT = dict()
DF_SHEET_NAME = dict()
ctr = 0
for estimator_type in ['single', 'pipe']:
    for _deep in [True, False]:
        ctr += 1
        print(f'\n\033[92mPermutation {ctr} of 4')
        print(f'estimator type = {estimator_type}, deep={_deep}\033[0m')

        # set the estimators based on user settings ** * ** * ** * ** *
        if estimator_type == 'single':
            if show_sklearn:
                sk_test_est = sklearn_gscv_single_est
                sk_test_mock_gstcv = sklearn_mock_gstcv_single_est
            if show_dask:
                d_test_est = dask_gscv_single_est
                d_test_mock_gstcv = dask_mock_gstcv_single_est
        elif estimator_type == 'pipe':
            if show_sklearn:
                sk_test_est = sklearn_gscv_pipe_est
                sk_test_mock_gstcv = sklearn_mock_gstcv_pipe_est
            if show_dask:
                d_test_est = dask_gscv_pipe_est
                d_test_mock_gstcv = dask_mock_gstcv_pipe_est
        # END set the estimators based on user settings ** * ** * ** *

        # get params for sk/dask & mock GSTCV ** * ** * ** * ** * ** *
        if show_sklearn:
            sk_params = sk_test_est.get_params(deep=_deep)
            sk_gstcv_params = sk_test_mock_gstcv.get_params(deep=_deep)
        if show_dask:
            d_params = d_test_est.get_params(deep=_deep)
            d_gstcv_params = d_test_mock_gstcv.get_params(deep=_deep)
        # END get params for sk/dask & mock GSTCV ** * ** * ** * ** * **

        # mash params together to get uniques for DF row labels ** * **
        if show_sklearn:
            ALL_PARAMS = sk_params | sk_gstcv_params
        if show_dask:
            ALL_PARAMS = d_params | d_gstcv_params
        ALL_PARAMS = sorted(list(ALL_PARAMS.keys()))
        # END mash params together to get uniques for DF row labels ** *


        DF = pd.DataFrame(index=ALL_PARAMS, dtype=object).fillna('-')
        pd.options.display.width = 0

        # fill non-shared param values ** * ** * ** * ** * ** * ** * **
        for _param in ALL_PARAMS:
            if show_sklearn:
                try:
                    DF.loc[_param, 'SK_REF'] = sk_params.get(_param, 'NOT IN')
                except:
                    DF.loc[_param, 'SK_REF'] = f'\033[91mEXCEPTED\033[0m'

                try:
                    DF.loc[_param, 'SK_MOCK'] = sk_gstcv_params.get(_param, 'NOT IN')
                except:
                    DF.loc[_param, 'SK_MOCK'] = f'\033[91mEXCEPTED\033[0m'

            if show_dask:
                try:
                    DF.loc[_param, 'DA_REF'] = d_params.get(_param, 'NOT IN')
                except:
                    DF.loc[_param, 'DA_REF'] = f'\033[91mEXCEPTED\033[0m'

                try:
                    DF.loc[_param, 'DA_MOCK'] = d_gstcv_params.get(_param, 'NOT IN')
                except:
                    DF.loc[_param, 'DA_MOCK'] = f'\033[91mEXCEPTED\033[0m'
        # END fill non-shared param values ** * ** * ** * ** * ** * ** *

        DF_DICT[ctr] = DF
        DF_SHEET_NAME[ctr] = f'{estimator_type}_deep_{_deep}'

        if __ == 'p':
            print(DF)
        elif __ == 's':
            pass
        elif __ == 'd':
            pass

if __ == 'd':

    desktop_path = Path.home() / "Desktop"
    filename = r'get_params_comparison.xlsx'
    path = desktop_path / filename
    with pd.ExcelWriter(path, engine=None, mode="w") as writer:
        for idx, key in enumerate(DF_DICT, 1):
            DF_DICT[key].to_excel(
                writer,
                sheet_name=DF_SHEET_NAME[idx],
                engine=None,
                na_rep=None
            )

elif __ == 's':
    pass


# END MAKE _GSTCV get_params deep & not deep LOOK THE SAME AS x_gscv get_params





# PROVE OUT THAT _GSTCV().get_params() MOCK OUTPUT CAN BE PASSED TO SKLEARN / DASK

sklearn_mock_gstcv_single_est = mock_GSTCV(
    estimator=sklearn_est,
    param_grid=sklearn_one_est_param_grid
)
sk_mock_est_params = sklearn_mock_gstcv_single_est.get_params(deep=True)

dask_mock_gstcv_single_est = mock_GSTCV(
    estimator=dask_est,
    param_grid=dask_one_est_param_grid
)
da_mock_est_params = dask_mock_gstcv_single_est.get_params(deep=True)

sklearn_mock_gstcv_pipe_est = mock_GSTCV(
    estimator=sklearn_pipe,
    param_grid=sklearn_pipe_param_grid
)
sk_mock_pipe_params = sklearn_mock_gstcv_pipe_est.get_params(deep=True)

dask_mock_gstcv_pipe_est = mock_GSTCV(
    estimator=dask_pipe,
    param_grid=dask_pipe_param_grid
)
da_mock_pipe_params = dask_mock_gstcv_pipe_est.get_params(deep=True)

sklearn_gscv_single_est.set_params(**sk_mock_est_params)
dask_gscv_single_est.set_params(**da_mock_est_params)

sklearn_gscv_pipe_est.set_params(**sk_mock_pipe_params)
dask_gscv_pipe_est.set_params(**da_mock_pipe_params)











