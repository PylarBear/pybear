# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.preprocessing import ColumnDeduplicateTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.estimator_checks import (
    check_transformer_general,
    check_transformers_unfitted,
    check_transformers_unfitted_stateless,
    check_transformer_data_not_an_array
)

# pizza 24_10_28_10_23_00 when u work on this, go back to MCT and see
# how the tests were set up and the descriptions!


check_transformer_general(
    'MinCountTransformer',
    MinCountTransformer(count_threshold=5)
)

check_transformers_unfitted(
    'MinCountTransformer',
    MinCountTransformer(count_threshold=5)
)








