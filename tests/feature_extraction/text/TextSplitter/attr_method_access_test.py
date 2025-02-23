# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextSplitter.TextSplitter import TextSplitter
from pybear.base import is_fitted







# TextSplitter is always "fit"
class TestAttrAccess:


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_list():
        return np.random.choice(list('abcdefghijklmnop'), (10,), replace=True).tolist()


    # @staticmethod
    # @pytest.fixture
    # def _attrs():
    #     return [
    #         'regexp',
    #         'sep',
    #         'maxsplit',
    #         'flags'
    #     ]


    @pytest.mark.parametrize('has_seen_data', (True, False))
    def test_attr_access(self, has_seen_data, _X_list):

        TestCls = TextSplitter(str_maxsplit=-1)

        assert is_fitted(TestCls) is True

        if has_seen_data:
            TestCls.fit(_X_list)

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # all attrs should be accessible always
        assert getattr(TestCls, 'str_sep') is None
        assert getattr(TestCls, 'str_maxsplit') == -1
        assert getattr(TestCls, 'regexp_sep') is None
        assert getattr(TestCls, 'regexp_maxsplit') is None
        assert getattr(TestCls, 'regexp_flags') is None



# TextSplitter is always "fit"
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


        TestCls = TextSplitter(str_maxsplit=-1)

        assert is_fitted(TestCls) is True

        if has_seen_data:
            TestCls.fit(_X_list)

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        out = getattr(TestCls, 'get_params')()
        assert isinstance(out, dict)
        assert all(map(isinstance, out.keys(), (str for _ in out.keys())))
        for param in ['str_sep', 'str_maxsplit', 'regexp_sep',
                'regexp_maxsplit', 'regexp_flags']:
            assert param in out


        out = getattr(TestCls, 'set_params')(**{})
        assert isinstance(out, TextSplitter)


         # v v v v v must see X every time, put these last v v v v v v v

        out = getattr(TestCls, 'transform')(_X_list)
        assert isinstance(out, list)
        assert all(map(isinstance, out, (list for _ in out)))


        out = getattr(TestCls, 'score')(_X_list)
        assert out is None

        out = getattr(TestCls, 'fit')(_X_list)
        assert isinstance(out, TextSplitter)

        out = getattr(TestCls, 'partial_fit')(_X_list)
        assert isinstance(out, TextSplitter)

        out = getattr(TestCls, 'fit_transform')(_X_list)
        assert isinstance(out, list)
        assert all(map(isinstance, out, (list for _ in out)))














