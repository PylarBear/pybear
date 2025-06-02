# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



#     n_features_in_
#     feature_names_in_
#     original_dtypes_
#     total_counts_by_column_
#     instructions_

# original_dtypes_, total_counts_by_column_, instructions_ should have
# the setter blocked, and be not accessible before fit. this is tested
# in attr_method_access_test with 1 & 2 rcr.

# n_features_in_ 1 & 2 rcr accuracy against columns and each other
# incidentally tested in attr_method_access

# feature_names_in_ 1 & 2 rcr accuracy against columns and each other
# incidentally tested in attr_method_access

# original_dtypes_ is a list-like of MCT-assigned dtypes for each column.
# the accuracy of building one of them for real data is tested in
# parallel_dtype_unqs_cts_test.

# total_counts_by_column_ is a dict of uniques and counts in each column.
# the accuracy of building one of them for real data is tested in
# parallel_dtype_unqs_cts_test. the accuracy of merging 2 of these is
# tested with fabricated tcbcs is in tcbc_merger_test.

# accuracy of original_dtypes_ and total_counts_by_column_ for an instance
# that has been / has not been reset is tested in MinCountTransformer_test.

# instructions_ is built directly off of total_counts_by_column_. the
# accuracy of instructions_ vis-a-vis fabricated total_counts_by_column
# is tested in make_instructions_test.

# so we dont need atomic level testing. just validate that the correct
# types are being returned.



import numpy as np

import pytest

from pybear.preprocessing import MinCountTransformer as MCT



class TestAccuracy:


    @pytest.mark.parametrize('_recursions', (1,2))
    def test_accuracy(self, _recursions):

        _shape = (100, 4)

        _count_threshold = _shape[0] // 10

        _kwargs = {
            'count_threshold': _count_threshold,
            'ignore_float_columns': True,
            'ignore_non_binary_integer_columns': False,
            'delete_axis_0': False,
            'max_recursions': _recursions
        }

        _pool_size = (_shape[0] // (_count_threshold + 1))

        _X_wip = np.random.randint(0, 2, _shape[0]).astype(np.uint8)
        _X_wip = np.vstack((
            _X_wip,
            np.random.randint(0, _pool_size, _shape[0])
        )).astype(np.int32)
        _X_wip = np.vstack((
            _X_wip,
            np.random.uniform(0, 1, _shape[0])
        )).astype(np.float64)
        _X_wip = np.vstack((
            _X_wip,
            np.random.choice(
                list(list('abcdefghijklmnop')[:_pool_size]),
                _shape[0]
            )
        ))

        _X_wip = _X_wip.transpose()
        _MCT = MCT(**_kwargs)

        _MCT.fit_transform(_X_wip)
        n_features_in_ = _MCT.n_features_in_

        # original_dtypes -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        original_dtypes = _MCT.original_dtypes_
        assert isinstance(original_dtypes, np.ndarray)
        assert len(original_dtypes) == n_features_in_
        assert all(map(
            isinstance,
            list(original_dtypes),
            (str for _ in original_dtypes)
        ))
        assert np.array_equal(
            ['bin_int', 'int', 'float', 'obj'],
            original_dtypes
        )
        # END original_dtypes -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # total_counts_by_column_ -- -- -- -- -- -- -- -- -- -- -- -- -- --
        tcbc_ = _MCT.total_counts_by_column_
        assert isinstance(tcbc_, dict)
        assert len(tcbc_) == n_features_in_
        assert np.array_equal(sorted(list(tcbc_.keys())), range(n_features_in_))
        for k, v in tcbc_.items():
            assert isinstance(k, int)
            assert isinstance(v, dict)
            assert all(map(isinstance, v.values(), (int for _ in v)))
        # END total_counts_by_column_ -- -- -- -- -- -- -- -- -- -- -- --

        # instructions_ -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        tcbc_ = _MCT.instructions_
        assert isinstance(tcbc_, dict)
        assert len(tcbc_) == n_features_in_
        assert np.array_equal(sorted(list(tcbc_.keys())), range(n_features_in_))
        for k, v in tcbc_.items():
            assert isinstance(k, int)
            assert isinstance(v, list)
        # END instructions_ -- -- -- -- -- -- -- -- -- -- -- -- -- -- --




