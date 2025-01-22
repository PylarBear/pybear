# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# pizza this is a fluid situation
# need to finalize containers, and in validate_data, min columns & min rows



from .._type_aliases import XContainer

import numpy as np
import pandas as pd
import scipy.sparse as ss  # pizza is aspirational



def _val_X(
    X: XContainer
) -> None:


    if not isinstance(
        X,
        (
            np.ndarray,
            pd.core.frame.DataFrame,
            pd.core.series.Series,
            ss._csr.csr_matrix, ss._csc.csc_matrix, ss._coo.coo_matrix,
            ss._dia.dia_matrix, ss._lil.lil_matrix, ss._dok.dok_matrix,
            ss._bsr.bsr_matrix, ss._csr.csr_array, ss._csc.csc_array,
            ss._coo.coo_array, ss._dia.dia_array, ss._lil.lil_array,
            ss._dok.dok_array, ss._bsr.bsr_array
        )
    ):
        raise TypeError(
            f'invalid data container for X, {type(X)}. pizza placeholder, '
            f'finalize allowed containers'
        )


    # pizza
    # validate_data() reshapes X to 2D always (for now)













