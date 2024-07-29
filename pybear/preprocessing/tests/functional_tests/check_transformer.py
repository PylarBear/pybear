# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from preprocessing import MinCountTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.estimator_checks import (
    check_transformer_general,
    check_transformers_unfitted,
    check_transformers_unfitted_stateless,
    check_transformer_data_not_an_array
)


check_transformer_general(
    'MinCountTransformer',
    MinCountTransformer(count_threshold=5)
)

check_transformers_unfitted(
    'MinCountTransformer',
    MinCountTransformer(count_threshold=5)
)








