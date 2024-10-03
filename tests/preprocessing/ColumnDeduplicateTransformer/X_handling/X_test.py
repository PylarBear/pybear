# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.ColumnDeduplicateTransformer._X_handling._X import _X_handling



import numpy as np
import pandas as pd
from uuid import uuid4

import pytest


pytest.skip(reason='pizza isnt done', allow_module_level=True)

class TestX:

    # fixtures ** * ** *
    @staticmethod
    @pytest.fixture(scope='module')
    def X_np():
        return np.random.randint(0,10,(10,5))

    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(X_np):
        return [str(uuid4())[:4] for _ in range(X_np.shape[1])]
    # END fixtures ** * ** *


    def test_accepts_ndarray(self, X_np):
        _val_X(X_np)


    def test_accepts_df(self, X_np, _columns):
        _val_X(pd.DataFrame(data=X_np, columns=_columns))


    # pizza
    # def test_accepts_other_stuff(self):
    #     pass


    @pytest.mark.parametrize('junk_X',
        (-1, 0, 1, 3.14, True, None, 'trash', [0,1], {0,1}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk(self, junk_X):
        with pytest.raises(TypeError):
            _val_X(junk_X)


    # pizza
    # def test_rejects_other_stuff(self):
    #     pass








