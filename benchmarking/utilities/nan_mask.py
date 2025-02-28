# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
import pandas as pd
import polars as pl

from pybear.utilities._nan_masking import nan_mask



# this model is to inspect the mask output from nan_mask for polars dataframes



if __name__ == '__main__':

    # numeric -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    X = np.random.uniform(0, 1, (5,3))
    X[[0],[1]] = np.nan
    X[[2], [2]] = np.nan
    X[[1], [0]] = np.nan

    print(X)

    X = pl.from_numpy(X)

    print(X)

    NAN_MASK = nan_mask(X)
    print(NAN_MASK)

    # END numeric -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    # numeric 1D -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    X = np.random.uniform(0, 1, (5,))
    X[1] = np.nan
    X[2] = np.nan

    print(X)

    X = pl.Series(X)

    print(X)

    NAN_MASK = nan_mask(X)
    print(NAN_MASK)

    # END numeric 1D -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # string -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    print(f'-'*40)

    X = np.random.choice(list('abcde'), (5,3)).astype('<U3')
    X[[0], [1]] = 'nan'
    X[[2], [2]] = 'nan'
    X[[1], [0]] = 'nan'

    print(X)

    X = pl.from_numpy(X)

    print(X)

    NAN_MASK = nan_mask(X)
    print(NAN_MASK)



    # from_pandas string -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    print(f'-'*40)

    X = pd.DataFrame(np.random.choice(list('abcde'), (5,3))).astype('<U3')
    X.iloc[[0], [1]] = 'nan'
    X.iloc[[2], [2]] = pd.NA
    X.iloc[[1], [0]] = np.nan

    print(X)

    X = pl.from_pandas(X)

    print(X)

    NAN_MASK = nan_mask(X)
    print(NAN_MASK)



    # 1D string -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    print(f'-'*40)

    X = pd.Series(np.random.choice(list('abcde'), (10,))).astype('<U3')
    X.iloc[0] = np.nan
    X.iloc[1] = 'nan'
    X.iloc[2] = pd.NA
    X.iloc[3] = None

    print(X)

    X = pl.from_pandas(X)

    print(X)

    NAN_MASK = nan_mask(X)
    print(NAN_MASK)





