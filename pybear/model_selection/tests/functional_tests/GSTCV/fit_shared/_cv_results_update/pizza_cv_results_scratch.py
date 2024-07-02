# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.utils import permuter
from sklearn.model_selection import ParameterGrid  # -> Generator

# 24_07_02_08_25_00
# see if permuter and ParameterGrid lay out permutations in the same order
# .... and it appears that they do

param_grid = {'a':[1,2,3], 'b':[True, False]}

idxs = permuter(list(map(list, param_grid.values())))
for (a_idx, b_idx) in idxs:
    print(f'a: {param_grid["a"][a_idx]},', f'b: {param_grid["b"][b_idx]}')

print(f'** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ')
[print(_) for _ in list(ParameterGrid(param_grid))]


"""

permuter
a: 1, b: True
a: 1, b: False
a: 2, b: True
a: 2, b: False
a: 3, b: True
a: 3, b: False

** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * 

ParameterGrid
{'a': 1, 'b': True}
{'a': 1, 'b': False}
{'a': 2, 'b': True}
{'a': 2, 'b': False}
{'a': 3, 'b': True}
{'a': 3, 'b': False}

"""







