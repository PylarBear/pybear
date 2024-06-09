# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# a scratch file trying to get wrapper class docs to show when a
# wrapped gscv is made via autogridsearch_wrapper
# no dice as of 24_05_30_09_21_00

# PIZZA

from docs_test_file_1__PIZZA import wrapper_function

from sklearn.model_selection import GridSearchCV


NewClass = wrapper_function(GridSearchCV)


new_class_instance = NewClass('help')














