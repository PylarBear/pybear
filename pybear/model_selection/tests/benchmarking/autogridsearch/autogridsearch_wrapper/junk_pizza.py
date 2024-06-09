# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from pybear.model_selection import autogridsearch_wrapper
from sklearn.model_selection import GridSearchCV


def junk_function(x):
    return x + 2


# def make_testcase(data, myfunc, docstring):
#
#     def test_something():
#         result = myfunc(data)
#         self.assertTrue(result > 0)
#
#     clsdict = {'test_something': test_something,
#                '__doc__': docstring}
#
#     return type('ATest', (InitDocStringDonorClass,), clsdict)
#
#
# MyTest = make_testcase(3, junk_function, 'This is a docstring')
#
# tron = MyTest()
#
# print(MyTest.__doc__)




def make_testcase(data, myfunc, docstring):


    AutoGridSearch = autogridsearch_wrapper(GridSearchCV)


    AutoGridSearch.__init__.__doc__ = 'rumpelstiltskin'

    clsdict = {
        '__doc__': docstring
    }

    return type('AutoGridSearch', (AutoGridSearch,), clsdict)


MyTest = make_testcase(3, junk_function, 'This is a docstring')


print(MyTest.__init__.__doc__)





