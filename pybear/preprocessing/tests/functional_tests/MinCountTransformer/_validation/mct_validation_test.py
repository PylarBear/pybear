# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np

from preprocessing.MinCountTransformer._validation._mct_validation import \
    _mct_validation


class TestMCTValidation:

    # all the internals of it are validated in other modules, just make
    # sure that it works

    def test_it_works(self):


        _mct_validation(
            _count_threshold=10,
            _ignore_float_columns=True,
            _ignore_non_binary_integer_columns=True,
            _ignore_nan=True,
            _delete_axis_0=False,
            _ignore_columns=np.array([], dtype=int),
            _handle_as_bool=np.array([], dtype=int),
            _reject_unseen_values=True,
            _max_recursions=2,
            _n_jobs=-1,
            _mct_has_been_fit=True,
            _n_features_in=3,
            _feature_names_in=None,
            _original_dtypes=np.array(['int', 'float', 'obj'], dtype='<U5'),
        )







