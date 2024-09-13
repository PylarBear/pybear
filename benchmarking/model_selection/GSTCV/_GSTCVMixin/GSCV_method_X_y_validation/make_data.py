# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pandas as pd
import string
import dask.array as da
import dask.dataframe as ddf
from dask_ml.datasets import make_classification as dask_make_classification




def make_ddf(DD, data_name, COLUMNS):
    return ddf.from_dask_array(DD[data_name], columns=COLUMNS)


def make_data(_rows, _cols):
    # make good np, pd, da, ddf objects ** * ** * ** * ** * ** * **
    GOOD_COLUMNS = list(string.ascii_lowercase[:_cols])

    DD = {}  # DATA_DICT

    DD['good_da_X'], DD['good_da_y'] = dask_make_classification(
        n_samples=_rows,
        n_features=_cols,
        n_informative=_cols,
        n_redundant=0,
        chunks=(_rows, _cols)
    )
    DD['good_np_X'] = DD['good_da_X'].compute()
    DD['good_np_y'] = DD['good_da_y'].compute()

    DD['good_pd_X'] = pd.DataFrame(DD['good_np_X'], columns=GOOD_COLUMNS)
    DD['good_pd_y'] = pd.DataFrame(DD['good_np_y'], columns=['y'])
    DD['good_ddf_X'] = make_ddf(DD, 'good_da_X', GOOD_COLUMNS)
    DD['good_ddf_y'] = make_ddf(DD, 'good_da_y', ['y'])
    # END make good np, pd, da, ddf objects ** * ** * ** * ** * **

    # make bad features np, pd, da, ddf objects ** * ** * ** * ** *
    _bad_cols = 2 * _cols
    BAD_COLUMNS = list(string.ascii_lowercase[:_bad_cols])

    DD['bad_features_da_X'], bad_features_da_y = \
        dask_make_classification(
            n_samples=_rows,
            n_features=_bad_cols,
            n_informative=_bad_cols,
            n_redundant=0,
            chunks=(_rows, _bad_cols)
        )

    DD['bad_features_da_y'] = da.vstack(
        (bad_features_da_y, bad_features_da_y)).transpose()
    del bad_features_da_y

    DD['bad_features_np_X'] = DD['bad_features_da_X'].compute()
    DD['bad_features_np_y'] = DD['bad_features_da_y'].compute()

    DD['bad_features_ddf_X'] = make_ddf(DD, 'bad_features_da_X', BAD_COLUMNS)
    DD['bad_features_ddf_y'] = make_ddf(DD, 'bad_features_da_y', ['y1', 'y2'])
    DD['bad_features_pd_X'] = DD['bad_features_ddf_X'].compute()
    DD['bad_features_pd_y'] = DD['bad_features_ddf_y'].compute()
    # END make bad features np, pd, da, ddf objects ** * ** * ** *

    # make bad rows X & y ** * ** * ** * ** * ** * ** * ** * ** * **
    _bad_rows = 2 * _rows

    DD['bad_rows_da_X'], DD['bad_rows_da_y'] = dask_make_classification(
        n_samples=_bad_rows,
        n_features=_cols,
        n_informative=_cols,
        n_redundant=0,
        chunks=(_bad_rows, _cols)
    )
    DD['bad_rows_np_X'] = DD['bad_rows_da_X'].compute()
    DD['bad_rows_np_y'] = DD['bad_rows_da_y'].compute()

    DD['bad_rows_ddf_X'] = make_ddf(DD, 'bad_rows_da_X', GOOD_COLUMNS)
    DD['bad_rows_ddf_y'] = make_ddf(DD, 'bad_rows_da_y', ['y'])
    DD['bad_rows_pd_X'] = DD['bad_rows_ddf_X'].compute()
    DD['bad_rows_pd_y'] = DD['bad_rows_ddf_y'].compute()
    # END make bad rows X & y ** * ** * ** * ** * ** * ** * ** * **

    # make bad data X & y ** * ** * ** * ** * ** * ** * ** * ** * **
    DD['bad_data_da_X'] = da.random.choice(
        list('abcdefghijklmnop'),
        (_rows, _cols),
        replace=True
    )
    DD['bad_data_ddf_X'] = make_ddf(DD, 'bad_data_da_X', GOOD_COLUMNS)
    DD['bad_data_np_X'] = DD['bad_data_da_X'].compute()
    DD['bad_data_pd_X'] = DD['bad_data_ddf_X'].compute()

    DD['bad_data_da_y'] = da.random.choice(
        list('abcdefghijklmnop'),
        (_rows,),
        replace=True
    )
    DD['bad_data_ddf_y'] = make_ddf(DD, 'bad_data_da_y', ['y'])
    DD['bad_data_np_y'] = DD['bad_data_da_y'].compute()
    DD['bad_data_pd_y'] = DD['bad_data_ddf_y'].compute()
    # END make bad data X & Y ** * ** * ** * ** * ** * ** * ** * ** *

    # make bad classes y ** * ** * ** * ** * ** * ** * ** * ** * **
    DD['bad_classes_da_y'] = da.random.randint(0, 5, _rows)
    DD['bad_classes_np_y'] = DD['bad_classes_da_y'].compute()
    DD['bad_classes_ddf_y'] = make_ddf(DD, 'bad_classes_da_y', ['y'])
    DD['bad_classes_pd_y'] = DD['bad_classes_ddf_y'].compute()
    # END make bad classes y ** * ** * ** * ** * ** * ** * ** * ** *


    return DD



