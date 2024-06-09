# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from model_selection.autogridsearch._autogridsearch_wrapper._validation. \
    _numerical_params import _numerical_param_value



total_passes = 3
key = 'a'
value = [[1 ,2, 3], 4, 'fixed_integer']
print(f'len(value) = {len(value)}')
print(f'value = {value}')

value = _numerical_param_value(
    key,
    value,
    total_passes
    )

print(value)






