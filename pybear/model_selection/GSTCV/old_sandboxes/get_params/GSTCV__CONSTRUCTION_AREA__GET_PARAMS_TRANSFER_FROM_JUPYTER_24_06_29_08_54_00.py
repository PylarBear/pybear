# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
import pandas as pd
import dask.array as da
import sys, os, base64, io, time
import scipy.sparse as ss
import functools
import odf
from copy import deepcopy

from sklearn.datasets import make_classification

import dask.array as da
import dask.dataframe as ddf

from sklearn.datasets import make_classification as sklearn_make_classification
from sklearn.model_selection import \
    train_test_split as sklearn_train_test_split
from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV
from sklearn.linear_model import \
    LogisticRegression as sklearn_LogisticRegression
from sklearn.linear_model import LinearRegression as sklearn_LinearRegression
from sklearn.pipeline import Pipeline

from dask_ml.datasets import make_classification as dask_make_classification
from dask_ml.model_selection import train_test_split as dask_train_test_split
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression
from dask_ml.linear_model import LinearRegression as dask_LinearRegression

from sklearn.base import is_classifier

# BEAR 24_02_26_14_32_00
# UNDERSTAND THE SIMILARITIES / DIFFERENCES OF SK/DASK SHALLOW/DEEP GSCV.get_params()
# IS JUST SLAPPED IN FROM ANOTHER CONSTRUCTION FILE AND HAS NOT BEEN VERIFIED IN ANY WAY IN THIS FILE

########################################################################################################################
########################################################################################################################
########################################################################################################################
# UNDERSTAND THE SIMILARITIES / DIFFERENCES OF SK/DASK SHALLOW/DEEP GSCV.get_params()
# BUILDING THE OUTPUT FOR get_params() IS THE SAME FOR BOTH
# For shallow SIMPLY GET ALL THE args/kwargs FOR SK/DASK GSCV & ADD AN ENTRY FOR ESTIMATOR.
# FOR deep, ADD ONTO shallow OUTPUT THE OUTPUT FROM get_params(deep) ON THE estimator, WHETHER IT IS A SINGLE ESTIMATOR OR A PIPELINE.
# (shallow/deep IS IRRELEVANT FOR est, BUT MATTERS FOR pipe). THE DIFFERENCE IN OUTPUT LIES IN THE DIFFERENT args/kwargs OF
# SKLEARN/DASK GSCV AND THE DIFFERENT args/kwargs OF THE SK/DASK VERSIONS OF THE ESTIMATOR(S).


_deep = True
use_shallow_param_deleter = False
show_est = False
show_pipe = True

if show_est + show_pipe == 0:
    raise KeyboardInterrupt(f"cant both be False")


def shallow_param_deleter(shallow_params, full_params):
    full_params = {k: v for k, v in full_params.items() if
                   k not in shallow_params}

    return full_params


sk_no_pipe_searcher = sklearn_GridSearchCV(
    estimator=sklearn_LinearRegression(),
    param_grid={'C': np.logspace(-3, 3, 7)}
)

shallow_params = sk_no_pipe_searcher.get_params(deep=False)
sk_no_pipe_params = sk_no_pipe_searcher.get_params(deep=_deep)
if use_shallow_param_deleter:
    sk_no_pipe_params = shallow_param_deleter(shallow_params,
                                              sk_no_pipe_params)
sk_no_pipe_params

sklearn_LinearRegression().get_params(deep=True)

dask_no_pipe_searcher = dask_GridSearchCV(
    estimator=dask_LinearRegression(),
    param_grid={'C': np.logspace(-3, 3, 7)}
)

shallow_params = dask_no_pipe_searcher.get_params(deep=False)
da_no_pipe_params = dask_no_pipe_searcher.get_params(deep=_deep)
if use_shallow_param_deleter:
    da_no_pipe_params = shallow_param_deleter(shallow_params,
                                              da_no_pipe_params)
da_no_pipe_params

dask_LinearRegression().get_params(deep=True)

sk_pipe_searcher = sklearn_GridSearchCV(
    estimator=Pipeline(
        steps=[('pipe1', sklearn_LinearRegression()),
               ('pipe2', sklearn_LinearRegression())
               ]
    ),
    param_grid={'pipe1__C': np.logspace(-3, 3, 7)}
)

shallow_params = sk_pipe_searcher.get_params(deep=False)
sk_pipe_params = sk_pipe_searcher.get_params(deep=_deep)
if use_shallow_param_deleter:
    sk_pipe_params = shallow_param_deleter(shallow_params, sk_pipe_params)
sk_pipe_params

Pipeline(
    steps=[('pipe1', sklearn_LinearRegression()),
           ('pipe2', sklearn_LinearRegression())
           ]
).get_params(deep=True)

dask_pipe_searcher = dask_GridSearchCV(
    estimator=Pipeline(
        steps=[('pipe1', dask_LinearRegression()),
               ('pipe2', dask_LinearRegression()),
               ]
    ),
    param_grid={'pipe1__C': np.logspace(-3, 3, 7)}
)

shallow_params = dask_pipe_searcher.get_params(deep=False)
da_pipe_params = dask_pipe_searcher.get_params(deep=_deep)
if use_shallow_param_deleter:
    da_pipe_params = shallow_param_deleter(shallow_params, da_pipe_params)
da_pipe_params

Pipeline(
    steps=[('pipe1', dask_LinearRegression()),
           ('pipe2', dask_LinearRegression()),
           ]
).get_params(deep=True)

MERGED_GET_PARAMS = {}
if show_est:
    MERGED_GET_PARAMS = MERGED_GET_PARAMS | sk_no_pipe_params | da_no_pipe_params
elif show_pipe:
    MERGED_GET_PARAMS = MERGED_GET_PARAMS | sk_pipe_params | da_pipe_params

ALL_FIELDS = sorted(list({k for k, v in MERGED_GET_PARAMS.items()}))

COLUMNS = []
if show_est:
    COLUMNS += ['sk_no_pipe_params', 'da_no_pipe_params']
elif show_pipe:
    COLUMNS += ['sk_pipe_params', 'da_pipe_params']

DF = pd.DataFrame(index=ALL_FIELDS, columns=COLUMNS).fillna('-')

for k in ALL_FIELDS:
    if show_est:
        try:
            DF.loc[k, 'da_no_pipe_params'] = da_no_pipe_params.get(k, 'NOT IN')
        except:
            DF.loc[k, 'da_no_pipe_params'] = 'EXCEPTED'
        try:
            DF.loc[k, 'sk_no_pipe_params'] = sk_no_pipe_params.get(k, 'NOT IN')
        except:
            DF.loc[k, 'sk_no_pipe_params'] = 'EXCEPTED'
    if show_pipe:
        try:
            DF.loc[k, 'da_pipe_params'] = da_pipe_params.get(k, 'NOT IN')
        except:
            DF.loc[k, 'da_pipe_params'] = 'EXCEPTED'
        try:
            DF.loc[k, 'sk_pipe_params'] = sk_pipe_params.get(k, 'NOT IN')
        except:
            DF.loc[k, 'sk_pipe_params'] = 'EXCEPTED'

if show_est:
    for k, v in sk_no_pipe_params.items():
        if k not in da_no_pipe_params:
            DF.loc[k, 'da_no_pipe_params'] = 'SKLEARN ONLY'

    for k, v in da_no_pipe_params.items():
        if k not in sk_no_pipe_params:
            DF.loc[k, 'sk_no_pipe_params'] = 'DASK ONLY'

if show_pipe:
    for k, v in sk_pipe_params.items():
        if k not in da_pipe_params:
            DF.loc[k, 'da_pipe_params'] = 'SKLEARN ONLY'

    for k, v in da_pipe_params.items():
        if k not in sk_pipe_params:
            DF.loc[k, 'sk_pipe_params'] = 'DASK ONLY'


# END UNDERSTAND THE SIMILARITIES / DIFFERENCES OF SK/DASK SHALLOW/DEEP GSCV.get_params()
########################################################################################################################
########################################################################################################################
########################################################################################################################


# MAKE _GSTCV get_params deep & not deep LOOK THE SAME AS x_gscv get_params ** ** ** ** ** ** ** ** ** ** ** **

# FOR A STAND-ALONE ESTIMATOR, deep & not deep ARE THE SAME
# test_estimator = LogisticRegression()
# test_estimator.get_params(deep=True)
# test_estimator.get_params(deep=False)

# FOR GSCV WITH 1 ESTIMATOR, deep & not deep ARE DIFFERENT


# MOCK THE _GSTCV CLASS AND ITS get_params METHOD

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
        # paramsdict['thresholds'] = self.thresholds
        if not self.dask_estimator: paramsdict['verbose'] = self.verbose

        # CORRECT FOR BOTH SIMPLE ESTIMATOR OR PIPELINE
        if deep:
            paramsdict = paramsdict | {f'estimator__{k}': v for k, v in
                                       self.estimator.get_params(
                                           deep=True).items()}

        # ALPHABETIZE paramsdict
        paramsdict = {k: paramsdict.pop(k) for k in sorted(paramsdict)}

        return paramsdict


show_sklearn = False
show_dask = True
if show_sklearn + show_dask == 0:
    raise KeyboardInterrupt(f"show_sklearn and show_dask cannot both be False")

sklearn_est = sklearn_LogisticRegression()
dask_est = dask_LogisticRegression()

sklearn_one_est_param_grid = {'C': [1e-3, 1e-2, 1e-1],
                              'tol': [1e-9, 1e-8, 1e-7]}
dask_one_est_param_grid = {'C': [1e-3, 1e-2, 1e-1], 'tol': [1e-9, 1e-8, 1e-7]}

sklearn_gscv_single_est = sklearn_GridSearchCV(estimator=sklearn_est,
                                               param_grid=sklearn_one_est_param_grid)
sklearn_mock_gstcv_single_est = mock_GSTCV(estimator=sklearn_est,
                                           param_grid=sklearn_one_est_param_grid)
dask_gscv_single_est = dask_GridSearchCV(estimator=dask_est,
                                         param_grid=dask_one_est_param_grid)
dask_mock_gstcv_single_est = mock_GSTCV(estimator=dask_est,
                                        param_grid=dask_one_est_param_grid)

sklearn_pipe = Pipeline(
    steps=[
        ('pipe_class_1', sklearn_LogisticRegression()),
        ('pipe_class_2', sklearn_LinearRegression())
    ],
    verbose=True
)

dask_pipe = Pipeline(
    steps=[
        ('pipe_class_1', dask_LogisticRegression()),
        ('pipe_class_2', dask_LinearRegression())
    ],
    verbose=True
)

sklearn_pipe_param_grid = {
    'pipe_class_1__C': [1e-3, 1e-2, 1e-1],
    'pipe_class_1__tol': [1e-9, 1e-8, 1e-7],
    'pipe_class_2__C': [1e-3, 1e-2, 1e-1],
    'pipe_class_3__tol': [1e-9, 1e-8, 1e-7],
}

dask_pipe_param_grid = {
    'pipe_class_1__C': [1e-3, 1e-2, 1e-1],
    'pipe_class_1__tol': [1e-9, 1e-8, 1e-7],
    'pipe_class_2__C': [1e-3, 1e-2, 1e-1],
    'pipe_class_2__tol': [1e-9, 1e-8, 1e-7],
}

sklearn_gscv_pipe_est = sklearn_GridSearchCV(estimator=sklearn_pipe,
                                             param_grid=sklearn_pipe_param_grid)
sklearn_mock_gstcv_pipe_est = mock_GSTCV(estimator=sklearn_pipe,
                                         param_grid=sklearn_pipe_param_grid)
dask_gscv_pipe_est = dask_GridSearchCV(estimator=dask_pipe,
                                       param_grid=dask_pipe_param_grid)
dask_mock_gstcv_pipe_est = mock_GSTCV(estimator=dask_pipe,
                                      param_grid=dask_pipe_param_grid)

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

        sk_params = sk_test_est.get_params(deep=_deep)
        d_params = d_test_est.get_params(deep=_deep)
        sk_gstcv_params = sk_test_mock_gstcv.get_params(deep=_deep)
        d_gstcv_params = d_test_mock_gstcv.get_params(deep=_deep)

        ALL_PARAMS = {}
        if show_sklearn:
            ALL_PARAMS = ALL_PARAMS | sk_params | sk_gstcv_params
        if show_dask:
            ALL_PARAMS = ALL_PARAMS | d_params | d_gstcv_params
        ALL_PARAMS = sorted(list(ALL_PARAMS.keys()))

        COLUMNS = []
        if show_sklearn:
            COLUMNS += ['SK_REF', 'SK_MOCK']
        if show_dask:
            COLUMNS += ['DA_REF', 'DA_MOCK']

        DF = pd.DataFrame(index=ALL_PARAMS, columns=COLUMNS,
                          dtype=object).fillna('-')
        pd.options.display.width = 0

        for _param in ALL_PARAMS:
            if show_sklearn:
                try:
                    DF.loc[_param, 'SK_REF'] = sk_params.get(_param, 'NOT IN')
                except:
                    DF.loc[_param, 'SK_REF'] = f'\033[91mEXCEPTED\033[0m'

                try:
                    DF.loc[_param, 'SK_MOCK'] = sk_gstcv_params.get(_param,
                                                                    'NOT IN')
                except:
                    DF.loc[_param, 'SK_MOCK'] = f'\033[91mEXCEPTED\033[0m'

            if show_dask:
                try:
                    DF.loc[_param, 'DA_REF'] = d_params.get(_param, 'NOT IN')
                except:
                    DF.loc[_param, 'DA_REF'] = f'\033[91mEXCEPTED\033[0m'

                try:
                    DF.loc[_param, 'DA_MOCK'] = d_gstcv_params.get(_param,
                                                                   'NOT IN')
                except:
                    DF.loc[_param, 'DA_MOCK'] = f'\033[91mEXCEPTED\033[0m'

        DF_DICT[ctr] = DF
        DF_SHEET_NAME[ctr] = f'{estimator_type}_deep_{_deep}'

        if __ == 'p':
            print(DF)
        elif __ == 's':
            pass
        elif __ == 'd':
            pass

if __ == 'd':

    with pd.ExcelWriter(
            rf'/home/bear/Desktop/get_params_comparison_{input("give a 1 digit number for the filename")}.ods',
            engine=None,
            mode="w") as writer:
        for idx, key in enumerate(DF_DICT, 1):
            DF_DICT[key].to_excel(writer, sheet_name=DF_SHEET_NAME[idx],
                                  engine=None, na_rep=None)

elif __ == 's':
    pass

# for referee_param in referee_params:
#     if referee_param not in gstcv_params:
#         raise Exception(f'{referee_param} is in {package}.get_params but not in gstcv.get_params')

# for gstcv_param in gstcv_params:
#     if gstcv_param not in referee_params:
#         raise Exception(f'{gstcv_param} is in gstcv.get_params but not in {package}.get_params')


# END BEAR TRIES TO MAKE _GSTCV get_params deep & not deep LOOK THE SAME AS x_gscv get_params ** ** ** ** ** ** ** ** ** ** ** **


# PROVE OUT THAT _GSTCV().get_params() MOCK OUTPUT CAN BE PASSED TO SKLEARN / DASK

sklearn_mock_gstcv_single_est = mock_GSTCV(estimator=sklearn_est,
                                           param_grid=sklearn_one_est_param_grid)
sk_mock_est_params = sklearn_mock_gstcv_single_est.get_params(deep=True)

dask_mock_gstcv_single_est = mock_GSTCV(estimator=dask_est,
                                        param_grid=dask_one_est_param_grid)
da_mock_est_params = dask_mock_gstcv_single_est.get_params(deep=True)

sklearn_mock_gstcv_pipe_est = mock_GSTCV(estimator=sklearn_pipe,
                                         param_grid=sklearn_pipe_param_grid)
sk_mock_pipe_params = sklearn_mock_gstcv_pipe_est.get_params(deep=True)

dask_mock_gstcv_pipe_est = mock_GSTCV(estimator=dask_pipe,
                                      param_grid=dask_pipe_param_grid)
da_mock_pipe_params = dask_mock_gstcv_pipe_est.get_params(deep=True)

sklearn_gscv_single_est.set_params(**sk_mock_est_params)
dask_gscv_single_est.set_params(**da_mock_est_params)

sklearn_gscv_pipe_est.set_params(**sk_mock_pipe_params)
dask_gscv_pipe_est.set_params(**da_mock_pipe_params)











