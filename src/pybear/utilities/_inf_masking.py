# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




import numpy as np
import pandas as pd
import scipy.sparse as ss

from typing_extensions import Union, TypeAlias
import numpy.typing as npt



SparseTypes: TypeAlias = Union[
    ss._csr.csr_matrix,
    ss._csc.csc_matrix,
    ss._coo.coo_matrix,
    ss._dia.dia_matrix,
    ss._bsr.bsr_matrix,
    ss._csr.csr_array,
    ss._csc.csc_array,
    ss._coo.coo_array,
    ss._dia.dia_array,
    ss._bsr.bsr_array
]




def inf_masking(
    obj: Union[npt.NDArray, pd.DataFrame, SparseTypes]
) -> npt.NDArray[bool]:

    pass



















