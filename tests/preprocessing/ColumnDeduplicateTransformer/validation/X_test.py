# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.ColumnDeduplicateTransformer._validation._X \
    import _val_X

import numpy as np
import pandas as pd
import scipy.sparse as ss

import pytest




def test_X_cannot_be_none():

    with pytest.raises(TypeError):
        _val_X(None)



@pytest.mark.parametrize('X_format',
     ('np', 'pd', 'csr', 'csc', 'dok', 'bsr_matrix', 'bsr_array')
)
def test_accepts_np_pd_ss(X_format):

    # accepts all scipy sparse except bsr

    _base_X = np.random.uniform(0, 1, (20,13))

    if X_format == 'np':
        _X = _base_X
    elif X_format == 'pd':
        _X = pd.DataFrame(_base_X)
    elif X_format == 'csr':
        _X = ss.csr_array(_base_X)
    elif X_format == 'csc':
        _X = ss.csc_array(_base_X)
    elif X_format == 'dok':
        _X = ss.dok_array(_base_X)
    elif X_format == 'bsr_array':
        _X = ss.bsr_array(_base_X)
    elif X_format == 'bsr_matrix':
        _X = ss.bsr_matrix(_base_X)
    else:
        raise Exception


    if X_format in ('bsr_matrix', 'bsr_array'):
        with pytest.raises(TypeError):
            _val_X(_X)
    else:
        _val_X(_X)






