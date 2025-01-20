# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._validation._val_X \
    import _val_X

import numpy as np
import pandas as pd
import scipy.sparse as ss  # pizza is aspirational
import dask.array as da
import dask.dataframe as ddf

import pytest



class TestValX:


    # def _val_X(
    #     X: XContainer
    # ) -> None:


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np():
        return np.random.randint(0, 10, (10, 10))



    @pytest.mark.parametrize('junk_X',
        (0, 1, True, None, 'junk', [1, 2], (1, 2), {1, 2}, {'a': 1}, lambda x: x)
    )
    def test_rejects_junk(self, junk_X):

        with pytest.raises(TypeError):
            _val_X(junk_X)


    def test_rejects_bad_container(self, _X_np):

        with pytest.raises(TypeError):
            _val_X(da.from_array(_X_np))

        with pytest.raises(TypeError):
            _val_X(ddf.from_array(_X_np))

        with pytest.raises(TypeError):
            _val_X(ddf.from_array(ddf.from_array(_X_np[:, 0]).squeeze()))


    def test_accepts_good_X(self, _X_np):

        assert _val_X(_X_np) is None
        assert _val_X(pd.DataFrame(_X_np)) is None
        assert _val_X(pd.Series(_X_np[:, 0])) is None
        assert _val_X(ss._csr.csr_matrix(_X_np)) is None
        assert _val_X(ss._csc.csc_matrix(_X_np)) is None
        assert _val_X(ss._coo.coo_matrix(_X_np)) is None
        assert _val_X(ss._dia.dia_matrix(_X_np)) is None
        assert _val_X(ss._lil.lil_matrix(_X_np)) is None
        assert _val_X(ss._dok.dok_matrix(_X_np)) is None
        assert _val_X(ss._bsr.bsr_matrix(_X_np)) is None
        assert _val_X(ss._csr.csr_array(_X_np)) is None
        assert _val_X(ss._csc.csc_array(_X_np)) is None
        assert _val_X(ss._coo.coo_array(_X_np)) is None
        assert _val_X(ss._dia.dia_array(_X_np)) is None
        assert _val_X(ss._lil.lil_array(_X_np)) is None
        assert _val_X(ss._dok.dok_array(_X_np)) is None
        assert _val_X(ss._bsr.bsr_array(_X_np)) is None










