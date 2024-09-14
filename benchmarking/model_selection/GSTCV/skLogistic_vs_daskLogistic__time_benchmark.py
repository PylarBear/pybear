# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
import pandas as pd
import string

from sklearn.linear_model import LogisticRegression as sklearn_Logistic

import dask.array as da
import dask.dataframe as ddf
import dask_expr._collection as ddf2
from dask_ml.linear_model import LogisticRegression as dask_Logistic

from pybear.utilities._benchmarking import time_memory_benchmark as tmb



_rows = 2_000_000
_columns = 10
COLUMNS = list(string.ascii_lowercase[:_columns])

sk_np_X = np.random.randint(0,10,(_rows,_columns))
sk_np_y = np.random.randint(0,2,(_rows,))
assert isinstance(sk_np_X, np.ndarray)
assert isinstance(sk_np_y, np.ndarray)

sk_df_X = pd.DataFrame(data=sk_np_X, columns=COLUMNS)
sk_df_y = pd.DataFrame(data=sk_np_y, columns=['Y'])
assert isinstance(sk_df_X, pd.core.frame.DataFrame)
assert isinstance(sk_df_y, pd.core.frame.DataFrame)

da_da_X = da.array(sk_np_X).rechunk((_rows//10, _columns))
da_da_y = da.array(sk_np_y).rechunk((_rows//10,))
assert isinstance(da_da_X, da.core.Array)
assert isinstance(da_da_y, da.core.Array)

da_df_X = ddf.from_array(da_da_X, columns=COLUMNS, chunksize=(_rows//10,))
da_df_y = ddf.from_array(da_da_y, columns=['Y'], chunksize=(_rows//10,))
assert isinstance(da_df_X, (ddf.core.DataFrame, ddf2.DataFrame))
assert isinstance(da_df_y, (ddf.core.DataFrame, ddf2.DataFrame))




def logistic_np(X, y):
    sk_logistic_np = sklearn_Logistic(max_iter=10_000, tol=1e-6)
    sk_logistic_np.fit(X, y)
    return sk_logistic_np


def logistic_da(X, y):
    da_logistic_da = dask_Logistic(max_iter=10_000, tol=1e-6)
    da_logistic_da.fit(X, y)
    return da_logistic_da


def logistic_pdf(X, y):
    sk_logistic_df = sklearn_Logistic(max_iter=10_000, tol=1e-6)
    sk_logistic_df.fit(sk_df_X, sk_df_y)
    return sk_logistic_df


# dask logistic cant take ddfs, dask array only
def logistic_ddf(X, y):
    da_logistic_ddf = sklearn_Logistic(max_iter=10_000, tol=1e-6)
    da_logistic_ddf.fit(X, y)
    return da_logistic_ddf



tmb(
    ('logistic_np', logistic_np, [sk_np_X, sk_np_y], {}),
    ('logistic_pdf', logistic_pdf, [sk_df_X, sk_df_y], {}),
    ('logistic_da', logistic_da, [da_da_X, da_da_y], {}),
    # ('logistic_ddf', logistic_ddf, [da_df_X, da_df_X], {}),
    rest_time=1,
    number_of_trials=3
)















