# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numbers
import re

import numpy as np

from pybear.feature_extraction.text._TextRemover.TextRemover import TextRemover
from pybear.base import is_fitted







# TextRemover is always "fit"
class TestAttrAccess:


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_list():
        return np.random.choice(
            list('abcdefghijklmnop'),
            (10,),
            replace=True
        ).tolist()


    # @staticmethod
    # @pytest.fixture
    # def _attrs():
    #     return [
    #         'str_remove',
    #         'regexp_remove',
    #         'regexp_flags',
    #         'row_support_'
    #     ]


    @pytest.mark.parametrize('has_seen_data', (True, False))
    def test_attr_access(self, has_seen_data, _X_list):

        TestCls = TextRemover(str_remove={' ', ',', '.', ';'})

        assert is_fitted(TestCls) is True

        if has_seen_data:
            TestCls.fit(_X_list)

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # all attrs should be accessible always
        assert getattr(TestCls, 'str_remove') == {' ', ',', '.', ';'}
        assert getattr(TestCls, 'regexp_remove') is None
        assert getattr(TestCls, 'regexp_flags') is None

        # 'row_support_' needs a transform to have been done
        with pytest.raises(AttributeError):
            getattr(TestCls, 'row_support_')

        # 'row_support_' cannot be set
        with pytest.raises(AttributeError):
            setattr(TestCls, 'row_support_', any)

        # 'n_rows_' needs a transform to have been done
        with pytest.raises(AttributeError):
            getattr(TestCls, 'n_rows_')

        # 'n_rows_' cannot be set
        with pytest.raises(AttributeError):
            setattr(TestCls, 'n_rows_', any)



# TextRemover is always "fit"
class TestMethodAccess:


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_list():

        return np.random.choice(list('abcdefghijklmnop'), (10,), replace=True).tolist()


    # @staticmethod
    # @pytest.fixture(scope='function')
    # def _methods():
    #     return [
    #         'partial_fit',
    #         'fit',
    #         'fit_transform',
    #         'get_params',
    #         'set_params',
    #         'transform',
    #         'score'
    #     ]


    @pytest.mark.parametrize('has_seen_data', (True, False))
    def test_access_methods(self, _X_list, has_seen_data):


        TestCls = TextRemover(regexp_remove='[a-m]')

        assert is_fitted(TestCls) is True

        if has_seen_data:
            TestCls.fit(_X_list)

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        with pytest.raises(NotImplementedError):
            getattr(TestCls, 'get_metadata_routing')()

        out = getattr(TestCls, 'get_params')()
        assert isinstance(out, dict)
        assert all(map(isinstance, out.keys(), (str for _ in out.keys())))
        for param in ['str_remove', 'regexp_remove', 'regexp_flags']:
            assert param in out


        out = getattr(TestCls, 'set_params')(**{'regexp_flags': re.I | re.X})
        assert isinstance(out, TextRemover)
        assert TestCls.regexp_flags == re.IGNORECASE|re.VERBOSE

         # v v v v v must see X every time, put these last v v v v v v v

        out = getattr(TestCls, 'transform')(_X_list)
        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))

        # 'row_support_' should be accessible now
        out = TestCls.row_support_
        assert isinstance(out, np.ndarray)
        assert len(out) == len(_X_list)
        assert all(map(isinstance, out, (np.bool_ for _ in out)))

        # 'n_rows_' should be accessible now
        out = TestCls.n_rows_
        assert isinstance(out, numbers.Integral)
        assert out == len(_X_list)

        # create a new instance to remove n_rows_ & row_support_
        TestCls = TextRemover(regexp_remove='[a-m]')

        out = getattr(TestCls, 'score')(_X_list)
        assert out is None

        out = getattr(TestCls, 'fit')(_X_list)
        assert isinstance(out, TextRemover)

        out = getattr(TestCls, 'partial_fit')(_X_list)
        assert isinstance(out, TextRemover)

        out = getattr(TestCls, 'fit_transform')(_X_list)
        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))

        # 'n_rows_' should be accessible now
        out = TestCls.n_rows_
        assert isinstance(out, numbers.Integral)
        assert out == len(_X_list)

        # 'row_support_' should be accessible now
        out = TestCls.row_support_
        assert isinstance(out, np.ndarray)
        assert len(out) == len(_X_list)
        assert all(map(isinstance, out, (np.bool_ for _ in out)))














