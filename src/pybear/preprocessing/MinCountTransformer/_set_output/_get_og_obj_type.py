# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing_extensions import Union
from .._type_aliases import XContainer, YContainer

import numpy as np
import pandas as pd
import scipy.sparse as ss





def _get_og_obj_type(
    OBJECT: Union[XContainer, YContainer],
    _original_obj_type: Union[str, None]  # use self._x_original_obj_type
) -> str:

    """
    pizza
    get dtypes of first-seen data object.


    Parameters
    ----------
        OBJECT:
            [pizza ndarray, pandas.DataFrame, pandas.Series] - data object
        _original_obj_type:
            Union[str, None], use self._x_original_obj_type


    Return
    ------
    -
        _original_obj_type: Union[str, None] - validated object type


    """


    # pizza, keep this note v v v , does it mean anything for set_output?
    # _x_original_obj_type ONLY MATTERS WHEN _handle_X_y IS CALLED
    # THROUGH transform() (OR fit_transform())
    # self._y_original_obj_type ONLY MATTERS WHEN _handle_X_y IS
    # CALLED THROUGH transform() (OR fit_transform())

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # validate OBJECT -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if not isinstance(OBJECT,
        (
            type(None), np.ndarray, pd.core.frame.DataFrame, pd.core.series.Series,
            ss.csr_matrix, ss.csr_array, ss.csc_matrix, ss.csc_array,
            ss.coo_matrix, ss.coo_array, ss.dia_matrix, ss.dia_array, ss.lil_matrix,
            ss.lil_array, ss.dok_matrix, ss.dok_array, ss.bsr_matrix, ss.bsr_array
         )
    ):
        raise TypeError(f"unrecognized container {type(OBJECT)}")
    # END validate OBJECT -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # validate og_obj_dtypes -- -- -- -- -- -- -- -- -- -- -- -- -- --
    _allowed = [
        None,
        'numpy_array',
        'pandas_dataframe',
        'pandas_series',
        'scipy_sparse_csr_matrix',
        'scipy_sparse_csr_array',
        'scipy_sparse_csc_matrix',
        'scipy_sparse_csc_array',
        'scipy_sparse_coo_matrix',
        'scipy_sparse_coo_array',
        'scipy_sparse_dia_matrix',
        'scipy_sparse_dia_array',
        'scipy_sparse_lil_matrix',
        'scipy_sparse_lil_array',
        'scipy_sparse_dok_matrix',
        'scipy_sparse_dok_array',
        'scipy_sparse_bsr_matrix',
        'scipy_sparse_bsr_array'
    ]

    _err_msg_x = f"_original_obj_type must be in type(None), {', '.join(_allowed[1:])}"

    try:
        if _original_obj_type is not None:
            _original_obj_type = _original_obj_type.lower()
    except:
        raise TypeError(_err_msg_x)

    if _original_obj_type not in _allowed:
        raise ValueError(_err_msg_x)

    del _allowed, _err_msg_x
    # END validate og_obj_dtypes -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # pizza come back to this. this is setting the obj_type on every pass,
    # it isnt checking against a 'first-seen'. do u want to do some kind of
    # 'reset' here like _check_feature_names_in?

    if isinstance(OBJECT, type(None)):
        return None
    elif isinstance(OBJECT, np.ndarray):
        return 'numpy_array'
    elif isinstance(OBJECT, pd.core.frame.DataFrame):
        return 'pandas_dataframe'
    elif isinstance(OBJECT, pd.core.series.Series):
        return 'pandas_series'
    elif isinstance(OBJECT, ss.csr_matrix):
        return 'scipy_sparse_csr_matrix'
    elif isinstance(OBJECT, ss.csr_array):
        return 'scipy_sparse_csr_array'
    elif isinstance(OBJECT, ss.csc_matrix):
        return 'scipy_sparse_csc_matrix'
    elif isinstance(OBJECT, ss.csc_array):
        return 'scipy_sparse_csc_array'
    elif isinstance(OBJECT, ss.coo_matrix):
        return 'scipy_sparse_coo_matrix'
    elif isinstance(OBJECT, ss.coo_array):
        return 'scipy_sparse_coo_array'
    elif isinstance(OBJECT, ss.dia_matrix):
        return 'scipy_sparse_dia_matrix'
    elif isinstance(OBJECT, ss.dia_array):
        return 'scipy_sparse_dia_array'
    elif isinstance(OBJECT, ss.lil_matrix):
        return 'scipy_sparse_lil_matrix'
    elif isinstance(OBJECT, ss.lil_array):
        return 'scipy_sparse_lil_array'
    elif isinstance(OBJECT, ss.dok_matrix):
        return 'scipy_sparse_dok_matrix'
    elif isinstance(OBJECT, ss.dok_array):
        return 'scipy_sparse_dok_array'
    elif isinstance(OBJECT, ss.bsr_matrix):
        return 'scipy_sparse_bsr_matrix'
    elif isinstance(OBJECT, ss.bsr_array):
        return 'scipy_sparse_bsr_array'
    else:
        raise Exception










