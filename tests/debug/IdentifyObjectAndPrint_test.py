# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import numpy as np
import pandas as pd
from string import ascii_lowercase



from pybear.debug import IdentifyObjectAndPrint as ioap


@pytest.fixture
def good_array():
    _rows = 5
    _cols = 3
    return np.random.randint(0,10,(_rows,_cols))


@pytest.fixture
def good_dataframe():
    _rows = 5
    _cols = 3

    return pd.DataFrame(
                        data=np.random.randint(0,10,(_rows,_cols)),
                        columns=list(ascii_lowercase[:_cols])
    )


@pytest.mark.skip
class TestIOAPAcceptsAnyObject:

    @pytest.mark.parametrize(
        'x',
        (0, np.pi, True, None, [1,2,3], 'some string')
    )
    def test_1(self, x):
        ioap(x)

    def test_2(self, good_dataframe):
        ioap(good_dataframe)

    def test_3(self, good_array):
        ioap(good_array)

















