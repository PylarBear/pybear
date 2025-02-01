# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# the accuracy of get_feature_names_out() output relies solely on
# base._get_feature_names_out() and MCT.get_support(). the accuracy of
# both are thoroughly tested elsewhere. test this module minimally.
# test that raises before fit and column_mask_ slice is applied to
# _get_feature_names_out().



from typing import Literal

import uuid

import numpy as np
import pandas as pd

import pytest

from pybear.preprocessing import MinCountTransformer as MCT
from pybear.base.exceptions import NotFittedError
from pybear.base._get_feature_names_out import get_feature_names_out



class Fixtures:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (10, 5)


    @staticmethod
    @pytest.fixture(scope='module')
    def _kwargs(_shape):
        return {
            'count_threshold': _shape[0] // 5,
            'ignore_float_columns': True,
            'ignore_non_binary_integer_columns': False,
            'n_jobs': 1
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _dropped_col_idx(_shape):
        return np.random.choice(_shape[1])


    @staticmethod
    @pytest.fixture(scope='module')
    def _X(_shape, _dropped_col_idx):
        # a set of data that will have one column removed

        def foo(
            _format: Literal['np', 'pd'],
            _columns_are_passed: bool
        ):

            # float columns are ignored
            _base_X = np.random.uniform(0, 1, _shape)

            _base_X[:, _dropped_col_idx] = \
                np.full(_shape[0], 1)

            if _columns_are_passed:
                _columns = [str(uuid.uuid4())[:5] for _ in range(_shape[1])]
            else:
                _columns = None

            if _format == 'np':
                _X_wip = _base_X
            elif _format == 'pd':
                _X_wip = pd.DataFrame(
                    data=_base_X,
                    columns=_columns
                )
            else:
                raise Exception

            return _X_wip

        return foo


    # END fixures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


# base._get_feature_names_out handles validation, tested elsewhere.


class TestAlwaysExceptsBeforeFit(Fixtures):


    @pytest.mark.parametrize('_format', ('np', 'pd'))
    @pytest.mark.parametrize('_columns_are_passed', (True, False))
    def test_always_except_before_fit(self,
        _X, _format, _columns_are_passed
    ):

        _X_wip = _X(_format, _columns_are_passed)

        _MCT = MCT()

        with pytest.raises(NotFittedError):
            _MCT.get_feature_names_out()

        _MCT.fit(_X_wip)
        _MCT.get_feature_names_out()
        _MCT.reset()

        with pytest.raises(NotFittedError):
            _MCT.get_feature_names_out()



class TestMinimalGFNOAccuracy(Fixtures):


    @pytest.mark.parametrize('_format', ('np', 'pd'))
    @pytest.mark.parametrize('_columns_are_passed', (True, False))
    def test_accuracy(
        self, _X, _shape, _kwargs, _format, _columns_are_passed,
        _dropped_col_idx
    ):

        _X_wip = _X(_format, _columns_are_passed)

        if _format == 'pd' and _columns_are_passed:
            _columns = np.array(_X_wip.columns)
        else:
            _columns = None

        # make the ref vector from base._get_feature_names_out()
        # and the rigged input data (we know what index will be dropped)
        _ref_out = get_feature_names_out(
            # using list(columns) just to see if it takes non-ndarray
            # whereas 'feature_names_in_' must be ndarray
            _input_features= list(_columns) if _columns is not None else _columns,
            feature_names_in_=_columns,
            n_features_in_=_shape[1]
        )
        _ref_mask = np.ones(_shape[1]).astype(bool)
        _ref_mask[_dropped_col_idx] = False

        _MCT = MCT(**_kwargs)
        _MCT.fit(_X_wip)

        assert _MCT.get_support().shape[0] == _shape[1]
        # _X should always lose one column
        assert np.sum(_MCT.get_support()) == (_shape[1] - 1)

        out = _MCT.get_feature_names_out()

        assert np.array_equal(out, _ref_out[_ref_mask]), \
            f"{_columns=}, \n{out=}"











