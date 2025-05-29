# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# pizza
# this was the joblib test from IM.
# use this as a template for SPF joblib test.


# import pytest
#
# import numpy as np
#
# from pybear.preprocessing._InterceptManager._partial_fit._find_constants import  \
#     _find_constants
#
#
#
# class TestJoblib:
#
#     # as of 25_05_21 all the tests were passing but things in the joblib
#     # part should have been failing. apparently all the other tests are
#     # using few enough columns that the joblib part isnt engaged. this
#     # test is explicitly designed to engage the joblib code.
#
#     # need to have >= _n_cols number of columns to engage joblib
#     # just do minor checks for accuracy
#
#
#     def test_it_works(self, _X_factory):
#
#         # as of 25_05_28 _n_cols is 1000 so use something 2X+ bigger
#         _shape = (623, 2538)
#
#         _rand_idxs = sorted(np.random.randint(0, _shape[1], 10).tolist())
#         _rand_constants = np.random.randint(0, 10, 10).tolist()
#         _constants = dict((zip(_rand_idxs, _rand_constants)))
#
#         _X = _X_factory(
#             _dupl=None,
#             _has_nan=False,
#             _format='np',
#             _dtype='flt',
#             _columns=None,
#             _constants=_constants,
#             _noise=0,
#             _zeros=None,
#             _shape=_shape
#         )
#
#
#         out = _find_constants(
#             _X,
#              _equal_nan=True,
#              _rtol=1e-5,
#              _atol=1e-8
#         )
#
#         # returns a dict
#         for idx, (k, v) in enumerate(out.items()):
#             assert k == _rand_idxs[idx]
#             assert v == _rand_constants[idx]
#
#
#
#
