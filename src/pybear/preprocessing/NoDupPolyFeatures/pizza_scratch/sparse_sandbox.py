# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# pizza this is scratch


import numpy as np
import scipy.sparse as ss
from pybear.utilities._nan_masking import nan_mask


X = np.random.randint(0,3,(5,3))#.astype(np.float64)
SS_X = ss.csc_array(X)
# SS_X.data[[0, 2, 4]] = np.nan


print(SS_X.data)
print(SS_X.dtype)
print(nan_mask(SS_X.data))



