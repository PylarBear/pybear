# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing._MinCountTransformer._validation._validation \
    import _validation

import numpy as np

import pytest



class TestMCTValidation:

    # all the internals of it are validated in other modules, just make
    # sure that it works

    def test_it_works(self):


        _validation(
            _X=np.random.randint(0, 10, (10, 10)),
            _count_threshold=10,
            _ignore_float_columns=True,
            _ignore_non_binary_integer_columns=True,
            _ignore_nan=True,
            _delete_axis_0=False,
            _ignore_columns=None,
            _handle_as_bool=None,
            _reject_unseen_values=True,
            _max_recursions=2,
            _n_features_in=3,
            _feature_names_in=None,
        )









