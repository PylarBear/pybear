# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# pizza this is a fluid situation
# need to finalize containers, and in validate_data, min columns & min rows


import numpy as np
import pandas as pd
import scipy.sparse as ss  # pizza is aspirational
import dask.array as da   # pizza
import dask.dataframe as ddf  #pizza
import dask_expr._collection as ddf2



def _val_X(X: any) -> None:


    if not isinstance(
        X,
        (np.ndarray, pd.core.frame.DataFrame, pd.core.series.Series,
        da.core.Array, ddf.core.frame.DataFrame, ddf.core.series.Series,
        ddf2.core.frame.DataFrame, ddf2.core.series.Series,
        ss._csr.csr_matrix, ss._csc.csc_matrix,
        ss._coo.coo_matrix, ss._dia.dia_matrix, ss._lil.lil_matrix, ss._dok.dok_matrix,
        ss._bsr.bsr_matrix, ss._csr.csr_array, ss._csc.csc_array, ss._coo.coo_array,
        ss._dia.dia_array, ss._lil.lil_array, ss._dok.dok_array, ss._bsr.bsr_array
         )
    ):
        raise TypeError('pizza placeholder, finalize allowed containers')


    # _handle_X_y reshapes X to 2D always

    # min_features ostensibly is 1, but is not explicitly checked (1D are allowed)
    # min_rows, make a decision, minimum count threshold is 2, so maybe 2 or 3 min_rows?













