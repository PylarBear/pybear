# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text.variable_finder import variable_finder


out = variable_finder(
    text=None,
    filepath=r'./variable_finder_code_sample'
)


[print(i) for i in out]

