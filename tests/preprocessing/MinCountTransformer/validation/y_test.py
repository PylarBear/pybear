# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing._MinCountTransformer._validation._y \
    import _val_y

import uuid
import numpy as np
import pandas as pd
import scipy.sparse as ss
import dask.array as da
import dask.dataframe as ddf

import pytest



class TestValY:


    # def _val_y(
    #     y: YContainer
    # ) -> None:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (10, 10)


    @staticmethod
    @pytest.fixture(scope='module')
    def _y_np(_shape):
        return np.random.randint(0, 10, _shape)


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_shape):
        return [str(uuid.uuid4())[:5] for _ in range(_shape[1])]

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('junk_y',
        (0, 1, True, 'junk', [1, 2], (1, 2), {1, 2}, {'a': 1}, lambda y: y)
    )
    def test_rejects_junk(self, junk_y):

        with pytest.raises(TypeError):
            _val_y(junk_y)


    def test_rejects_bad_container(self, _y_np, _columns, _shape):

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

        # numpy_recarray ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        _dtypes1 = [np.uint8 for _ in range(_shape[1]//2)]
        _dtypes2 = ['<U1' for _ in range(_shape[1]//2)]
        _formats = [list(zip(_columns, _dtypes1 + _dtypes2))]
        Y_NEW = np.recarray(
            (_shape[0],), names=_columns, formats=_formats, buf=_y_np
        )
        del _dtypes1, _dtypes2, _formats

        with pytest.raises(TypeError):
            _val_y(Y_NEW)
        del Y_NEW
        # END numpy_recarray ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    def test_masked_array_warns(self, _y_np):

        # numpy_masked_array ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        with pytest.warns():
            _val_y(np.ma.array(_y_np))
        # END numpy_masked_array ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    def test_accepts_good_y(self, _y_np, _columns):

        assert _val_y(None) is None
        assert _val_y(_y_np) is None
        assert _val_y(_y_np[:, 0].ravel()) is None
        assert _val_y(pd.DataFrame(_y_np, columns=_columns)) is None
        assert _val_y(
            pd.DataFrame(_y_np, columns=_columns).iloc[0, :].to_frame()
        ) is None
        assert _val_y(pd.Series(_y_np[:, 0])) is None



















