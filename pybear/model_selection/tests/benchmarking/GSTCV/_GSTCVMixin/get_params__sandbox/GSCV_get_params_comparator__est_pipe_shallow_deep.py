# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV
from sklearn.linear_model import LinearRegression as sklearn_LinearRegression
from sklearn.pipeline import Pipeline

from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from dask_ml.linear_model import LinearRegression as dask_LinearRegression



# UNDERSTAND THE SIMILARITIES / DIFFERENCES OF SK/DASK SHALLOW/DEEP GSCV.get_params()
# BUILDING THE OUTPUT FOR get_params() IS THE SAME FOR BOTH
# For shallow:
#   GET ALL THE args/kwargs FOR SK/DASK GSCV & ADD AN ENTRY FOR ESTIMATOR.
# For deep:
#   ADD ONTO shallow OUTPUT THE OUTPUT FROM get_params(deep) ON THE estimator,
#   WHETHER IT IS A SINGLE ESTIMATOR OR A PIPELINE.
#   (shallow/deep IS IRRELEVANT FOR est, BUT MATTERS FOR pipe).
#   THE DIFFERENCE IN OUTPUT BETWEEN SK & DASK LIES IN THE DIFFERENT args/kwargs OF
#   GSCV AND THE DIFFERENT args/kwargs OF THE SK/DASK VERSIONS OF THE ESTIMATOR(S).



# SET COMPARISON PARAMETERS ** * ** * ** * ** * ** * ** * **
_deep = True
use_shallow_param_deleter = False
show_est = True
show_pipe = False

if show_est + show_pipe == 0:
    raise KeyboardInterrupt(f"cant both be False")
# END SET COMPARISON PARAMETERS ** * ** * ** * ** * ** * ** *


def shallow_param_deleter(shallow_params, full_params):
    full_params = {k: v for k, v in full_params.items() if k not in shallow_params}

    return full_params


# BUILD EMPTY DF TO HOLD THE VARIOUS GSCV get_params() OUTPUT

########################################################################
# mash together all the params from each GSCV into one list of uniques
# and use that as the row labels in the DF

# sk no pipe ** * ** *
sk_no_pipe_searcher = sklearn_GridSearchCV(
    estimator=sklearn_LinearRegression(),
    param_grid={'C': np.logspace(-3, 3, 7)}
)

sk_no_pipe_params = sk_no_pipe_searcher.get_params(deep=_deep)
if use_shallow_param_deleter:
    sk_no_pipe_params = shallow_param_deleter(
        sk_no_pipe_searcher.get_params(deep=False),
        sk_no_pipe_params
    )

del sk_no_pipe_searcher
# END sk no pipe ** * ** *

# dask no pipe ** * ** *
dask_no_pipe_searcher = dask_GridSearchCV(
    estimator=dask_LinearRegression(),
    param_grid={'C': np.logspace(-3, 3, 7)}
)

da_no_pipe_params = dask_no_pipe_searcher.get_params(deep=_deep)
if use_shallow_param_deleter:
    da_no_pipe_params = shallow_param_deleter(
        dask_no_pipe_searcher.get_params(deep=False),
        da_no_pipe_params
    )

del dask_no_pipe_searcher
# dask no pipe ** * ** *


# sk pipe ** * ** * ** *
sk_pipe_searcher = sklearn_GridSearchCV(
    estimator=Pipeline(
        steps=[
                ('pipe1', sklearn_LinearRegression()),
                ('pipe2', sklearn_LinearRegression())
        ]
    ),
    param_grid={'pipe1__C': np.logspace(-3, 3, 7)}
)

sk_pipe_params = sk_pipe_searcher.get_params(deep=_deep)
if use_shallow_param_deleter:
    sk_pipe_params = shallow_param_deleter(
        sk_pipe_searcher.get_params(deep=False),
        sk_pipe_params
    )

del sk_pipe_searcher
# END sk pipe ** * ** * ** *

# da pipe ** * ** * **
dask_pipe_searcher = dask_GridSearchCV(
    estimator=Pipeline(
        steps=[
                ('pipe1', dask_LinearRegression()),
                ('pipe2', dask_LinearRegression()),
        ]
    ),
    param_grid={'pipe1__C': np.logspace(-3, 3, 7)}
)

shallow_params = dask_pipe_searcher.get_params(deep=False)
da_pipe_params = dask_pipe_searcher.get_params(deep=_deep)
if use_shallow_param_deleter:
    da_pipe_params = shallow_param_deleter(shallow_params, da_pipe_params)
# END da pipe ** * ** * **


if show_est:
    MERGED_GET_PARAMS = sk_no_pipe_params | da_no_pipe_params
elif show_pipe:
    MERGED_GET_PARAMS = sk_pipe_params | da_pipe_params


# this will be the rows of the DF
ALL_FIELDS = sorted(list(MERGED_GET_PARAMS.keys()))
########################################################################

if show_est:
    COLUMNS = ['sk_no_pipe_params', 'da_no_pipe_params']
elif show_pipe:
    COLUMNS = ['sk_pipe_params', 'da_pipe_params']

DF = pd.DataFrame(index=ALL_FIELDS, columns=COLUMNS).fillna('-')

# END BUILD EMPTY DF TO HOLD THE VARIOUS GSCV get_params() OUTPUT ######


# FILL THE EMPTY DF
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




while True:
    __ = input(f"\n(d)ump to csv or (p)rint to screen or (s)skip > ").lower()
    if __ not in 'dps':
        print(f'\ntry again. must be d p or s.')
    else:
        break


if __ == 'p':
    print(DF)
elif __ == 'd':
    path = rf'/home/bear/Desktop/get_params_comparison.ods'
    with pd.ExcelWriter(path, engine=None, mode="w") as writer:
        sheet_name = f"{'deep' if _deep else 'shallow'}_{'est' if show_est else 'pipe'}"
        DF.to_excel(writer, sheet_name=sheet_name, engine=None, na_rep=None)
elif __ == 's':
    pass

# END UNDERSTAND THE SIMILARITIES / DIFFERENCES OF SK/DASK SHALLOW/DEEP GSCV.get_params()
###############################################################################
###############################################################################
###############################################################################




























