# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
import numpy.typing as npt
from .._type_aliases import SparseTypes

import pandas as pd



def _val_X(
    _X: Union[npt.NDArray[any], pd.DataFrame, SparseTypes]
) -> None:

    """
    Pizza put words

    """

    _err_msg = (
        f"'X' must be a valid 2 dimensional numpy ndarray, pandas dataframe, "
        f"or scipy sparce matrix or array, with at least 2 columns and 1 "
        f"example."
    )

    # pizza, sklearn _validate_data is not catching this!
    if _X is None:
        raise TypeError(_err_msg)


    # pizza, sklearn _validate_data is not catching this!
    if len(_X.shape) != 2:
        raise ValueError(_err_msg)






