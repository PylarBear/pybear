# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import pandas as pd
import polars as pl

from pybear.feature_extraction.text._TextStatistics.TextStatistics import \
    TextStatistics as TS




class TestTextStatistics:

    def test_1D_inputs(self):

        _X = ['Say!', 'I', 'like', 'green', 'eggs', 'and', 'ham!']

        TS().fit(list(_X))

        TS().fit(set(_X))

        TS().fit(tuple(_X))

        TS().fit(np.array(_X))

        TS().fit(pd.Series(_X))

        TS().fit(pl.Series(_X))

        _2D_X = [['I', 'like', 'green'], ['eggs', 'and', 'ham!']]

        with pytest.raises(TypeError):
            TS().fit(list(_2D_X))

        with pytest.raises(TypeError):
            TS().fit(np.array(_2D_X))

        with pytest.raises(TypeError):
            TS().fit(pd.DataFrame(_2D_X))

        with pytest.raises(TypeError):
            TS().fit(pl.from_numpy(np.array(_2D_X)))


    def test_rejects_numeric(self):

        with pytest.raises(TypeError):
            TS().fit(np.random.randint(0,10,(5,5)))


        _X = [['I', 'like', 'green'], ['eggs', 'and', 13]]
        with pytest.raises(TypeError):
            TS().fit(_X)





