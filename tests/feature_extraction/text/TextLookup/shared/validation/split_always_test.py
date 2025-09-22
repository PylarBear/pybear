# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from typing import (
    Any,
)

import re

import numpy as np
import pandas as pd

from pybear.feature_extraction.text._TextLookup._shared._validation._split_always \
    import _val_split_always



class TestSplitAlways:


    @staticmethod
    @pytest.fixture(scope='module')
    def container_maker():

        def foo(obj: list[str], container: Any):
            if container is np.array:
                out = np.array(obj)
                assert isinstance(out, np.ndarray)
            elif container is pd.Series:
                out = pd.Series(obj)
                assert isinstance(out, pd.Series)
            else:
                out = container(obj)
                assert isinstance(out, container)

            return out

        return foo

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    @pytest.mark.parametrize('junk_split_always',
        (-2.7, -1, 0, 1, 2.7, True, False, 'trash', [0,1], (1,),
         {1,2}, {'a':1}, {1: 2}, {True: [1]}, lambda x: x)
    )
    def test_rejects_junk(self, junk_split_always):

        with pytest.raises(TypeError):
            _val_split_always(junk_split_always)


    def test_rejects_empty(self):

        # empty dict
        with pytest.raises(ValueError):
            _val_split_always({})

        # empty value
        with pytest.raises(ValueError):
            _val_split_always({'egg': ['tuna', 'on', 'toast'], 'cheese': []})


    @pytest.mark.parametrize('container', (list, set, tuple, np.array, pd.Series))
    def test_accepts_good(self, container_maker, container):

        # must be None | dict[str | re.compile, Sequence[str]]

        key_pool = ['patty', 'sauce', re.compile('^lettuce$'),
                re.compile('tomato.+', re.I)]

        value_pool = ['four', 'score', 'and', 'seven', 'years', 'ago']

        keys = np.random.choice(key_pool, 3, replace=False).tolist()

        values = []
        values.append(
            container_maker(
                np.random.choice(value_pool, 3, replace=False).tolist(),
                container
            )
        )

        _dict_1 = dict((zip(keys, values)))

        assert _val_split_always(None) is None

        assert _val_split_always(_dict_1) is None








