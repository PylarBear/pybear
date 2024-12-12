# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# PIZZA, U NEED TO TEST FOR WHEN THERE ARE DUPLS AND CONSTANTS IN
# X

# ALL @properties SHOULD RETURN NONE
# EVERYTHING SHOULD BE A NO-OP EXCEPT FOR PARTIAL FIT.
# VERIFY THE STATE OF transform(), IS IT JUST A NO-OP OR DOES IT STILL TERMINATE PYTHON.


#         _dupl = [[0,2]]
#
#         TEST_X = _X_factory(
#             _dupl=_dupl,
#             _format='np',
#             _dtype='flt',
#             _has_nan=False,
#             _columns=None,
#             _zeros=None,
#             _shape=(20,3)
#         )


#         # set constants v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
#         if constants == 'constants1':
#             constants = None
#         elif constants == 'constants2':
#             if X_dtype in ('flt', 'int'):
#                 constants = {0: 1, 2: 1, 9: 1}
#             elif X_dtype in ('str', 'obj', 'hybrid'):
#                 constants = {0: 1, 2: 'a', 9: 'b'}
#             else:
#                 raise Exception
#         elif constants == 'constants3':
#             if X_dtype in ('flt', 'int'):
#                 constants = {0: 1, 1: 1, 6: np.nan, 8: 1}
#             elif X_dtype in ('str', 'obj', 'hybrid'):
#                 constants = {0: 'a', 1: 'b', 6: 'nan', 8: '1'}
#             else:
#                 raise Exception
#         else:
#             raise Exception
#         # END set constants v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v



