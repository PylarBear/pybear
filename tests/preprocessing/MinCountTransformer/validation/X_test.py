# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing._MinCountTransformer._validation._X \
    import _val_X

import uuid
import numpy as np
import pandas as pd
import scipy.sparse as ss
import dask.array as da
import dask.dataframe as ddf

import pytest



class TestValX:


    # def _val_X(
    #     X: XContainer
    # ) -> None:

    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (10, 10)


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np(_shape):
        return np.random.randint(0, 10, _shape)


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_shape):
        return [str(uuid.uuid4())[:5] for _ in range(_shape[1])]

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('junk_X',
        (0, 1, True, None, 'junk', [1, 2], (1, 2), {1, 2}, {'a': 1}, lambda x: x)
    )
    def test_rejects_junk(self, junk_X):

        with pytest.raises(TypeError):
            _val_X(junk_X)


    def test_rejects_bad_container(self, _X_np, _columns, _shape):

        with pytest.raises(TypeError):
            _val_X(da.from_array(_X_np))

        with pytest.raises(TypeError):
            _val_X(ddf.from_array(_X_np))

        with pytest.raises(TypeError):
            _val_X(ddf.from_array(ddf.from_array(_X_np[:, 0]).squeeze()))

        with pytest.raises(TypeError):
            assert _val_X(pd.Series(_X_np[:, 0]))


        # numpy_recarray ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        _dtypes1 = [np.uint8 for _ in range(_shape[1]//2)]
        _dtypes2 = ['<U1' for _ in range(_shape[1]//2)]
        _formats = [list(zip(_columns, _dtypes1 + _dtypes2))]
        X_NEW = np.recarray(
            (_shape[0],), names=_columns, formats=_formats, buf=_X_np
        )
        del _dtypes1, _dtypes2, _formats

        with pytest.raises(TypeError):
            _val_X(X_NEW)
        del X_NEW
        # END numpy_recarray ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # numpy_masked_array ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        with pytest.warns():
            _val_X(np.ma.array(_X_np))
        # END numpy_masked_array ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    def test_accepts_good_X(self, _X_np):

        assert _val_X(_X_np) is None
        assert _val_X(np.ma.array(_X_np, mask=False)) is None
        assert _val_X(pd.DataFrame(_X_np)) is None
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










