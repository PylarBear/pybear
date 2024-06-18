# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from preprocessing import MinCountTransformer

from sklearn.datasets import load_breast_cancer

import pandas as pd


data = load_breast_cancer()

RAW_X = data.data.astype(int)
X = pd.DataFrame(data=RAW_X, columns=data.feature_names, dtype=int)
X = X.drop(columns=['worst area', 'mean area'], inplace=False)
y = data.target


print(f'one recursion:')
test_cls = MinCountTransformer(
    count_threshold=5,
    ignore_non_binary_integer_columns=False
)

test_cls.partial_fit(X, y)

test_cls.test_threshold(threshold=5, clean_printout=True)

print(f'** * ' * 20)
print()
print(f'** * ' * 20)
print(f'three recursions:')

test_cls = MinCountTransformer(
    count_threshold=5,
    ignore_non_binary_integer_columns=False,
    max_recursions=3
)

test_cls.fit_transform(X, y)

test_cls.test_threshold(threshold=5, clean_printout=True)








