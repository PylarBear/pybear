# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.base._num_samples import num_samples

import uuid
import numpy as np
import pandas as pd
import scipy.sparse as ss

import pytest



class TestNumSamples:

    # disallows objects that does not have 'shape' attr
    # requires all scipy sparse be 2D
    # requires all other data-bearing objects be 1D or 2D


    def test_rejects_things_w_o_shape_attr(self):

        X = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ]

        with pytest.raises(ValueError):
            num_samples(X)


    @pytest.mark.parametrize('X_format', ('np', 'pd', 'csr'))
    @pytest.mark.parametrize('X_shape',
        ((20, 3), (50, 5), (24, 13), (12, ), (77, ), (2, 4, 5), (8, 8, 8, 8))
    )
    def test_num_samples(self, X_format: str, X_shape: tuple[int, ...]):

        # skip disallowed conditions -- -- -- -- -- -- -- -- -- -- --
        if X_format == 'pd' and len(X_shape) > 2:
            pytest.skip(reason=f"pd cannot be more than 2D")

        if X_format == 'csr' and len(X_shape) != 2:
            pytest.skip(reason=f"pybear blocks scipy shape != 2")
        # end skip disallowed conditions -- -- -- -- -- -- -- -- -- --

        _base_X = np.random.randint(0, 10, X_shape)

        if len(X_shape) == 1:
            _columns = ['y']
        else:
            _columns = [str(uuid.uuid4())[:4] for _ in range(X_shape[1])]

        if X_format == 'np':
            _X = _base_X.copy()
        elif X_format == 'pd':
            _X = pd.DataFrame(data=_base_X, columns=_columns)
        elif X_format == 'csr':
            _X = ss.csr_array(_base_X)
        else:
            raise Exception

        if len(X_shape) in [1, 2]:
            assert num_samples(_X) == X_shape[0]
        else:
            with pytest.raises(ValueError):
                num_samples(_X)














