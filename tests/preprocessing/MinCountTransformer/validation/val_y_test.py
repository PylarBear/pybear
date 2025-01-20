# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._validation._val_y \
    import _val_y

import numpy as np
import pandas as pd
import scipy.sparse as ss  # pizza is aspirational
import dask.array as da
import dask.dataframe as ddf

import pytest



class TestValY:


    # def _val_y(
    #     y: YContainer
    # ) -> None:


    @staticmethod
    @pytest.fixture(scope='module')
    def _y_np():
        return np.random.randint(0, 10, (10, 10))



    @pytest.mark.parametrize('junk_y',
        (0, 1, True, 'junk', [1, 2], (1, 2), {1, 2}, {'a': 1}, lambda y: y)
    )
    def test_rejects_junk(self, junk_y):

        with pytest.raises(TypeError):
            _val_y(junk_y)


    def test_rejects_bad_container(self, _y_np):

        with pytest.raises(TypeError):
            _val_y(ss.csr_matrix(_y_np))

        with pytest.raises(TypeError):
            _val_y(ss.coo_array(_y_np))

        with pytest.raises(TypeError):
            _val_y(da.from_array(_y_np))

        with pytest.raises(TypeError):
            _val_y(ddf.from_array(_y_np))

        with pytest.raises(TypeError):
            _val_y(ddf.from_array(ddf.from_array(_y_np[:, 0]).squeeze()))


    def test_accepts_good_y(self, _y_np):

        assert _val_y(None) is None
        assert _val_y(_y_np) is None
        assert _val_y(pd.DataFrame(_y_np)) is None
        assert _val_y(pd.Series(_y_np[:, 0])) is None



















