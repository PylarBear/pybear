# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import uuid
import numpy as np
import dask.array as da
import dask.dataframe as ddf
import pandas as pd

from pybear.preprocessing.MinCountTransformer import MinCountTransformer
from pybear.preprocessing.MinCountTransformer._handle_X_y import _handle_X_y








class TestHandleXY:

    @staticmethod
    @pytest.fixture
    def _name():
        return type(MinCountTransformer).__name__


    @staticmethod
    @pytest.fixture
    def X_np():
        return np.random.randint(0, 10, (50, 10)).astype(np.uint8)


    @staticmethod
    @pytest.fixture
    def X_pd(X_np):
        columns = [str(uuid.uuid4())[:4] for _ in range(X_np.shape[1])]
        return pd.DataFrame(data=X_np, columns=columns)


    @staticmethod
    @pytest.fixture
    def y_np():
        return np.random.randint(0, 2, (50, 1)).astype(np.uint8)


    @staticmethod
    @pytest.fixture
    def y_pd(y_np):
        columns = [str(uuid.uuid4())[:4] for _ in range(y_np.shape[1])]
        return pd.DataFrame(data=y_np, columns=columns)


    @staticmethod
    @pytest.fixture
    def _allowed_og_types():
        return ['pandas_series', 'pandas_dataframe', 'numpy_array', None]



    # def _handle_X_y(
    #         X: XType,
    #         y: YType,
    #         _name: str,
    #         __x_original_obj_type: Union[str, None],
    #         __y_original_obj_type: Union[str, None],
    #     ) -> tuple[XType, YType, str, str, Union[np.ndarray[int], None]]:


    @pytest.mark.parametrize('_x_og_type',
        (0, 1, True, None, np.nan, min, [0,1], (1,2), {2,3}, lambda x: x,
         'pandas_series', 'pandas_dataframe', 'numpy_array')
    )
    @pytest.mark.parametrize('_y_og_type',
        (0, 1, True, None, np.nan, min, [0,1], (1,2), {2,3}, lambda x: x,
         'pandas_series', 'pandas_dataframe', 'numpy_array')
    )
    def test_accept_good_og_types_type_error_bad(self, _name, X_np, y_np,
        _x_og_type, _y_og_type, _allowed_og_types):

        if _x_og_type in _allowed_og_types and _y_og_type in _allowed_og_types:

                # lower case
                out = _handle_X_y(
                    X_np,
                    y_np,
                    _name,
                    _x_og_type,
                    _y_og_type
                )

                assert isinstance(out[0], np.ndarray)
                assert isinstance(out[1], np.ndarray)
                if _x_og_type is not None:
                    assert out[2] == _x_og_type
                if _y_og_type is not None:
                    assert out[3] == _y_og_type
                assert out[4] is None

                # case insensitive - upper case
                out = _handle_X_y(
                    X_np,
                    y_np,
                    _name,
                    _x_og_type.upper() if isinstance(_x_og_type, str) else _x_og_type,
                    _y_og_type.upper() if isinstance(_y_og_type, str) else _y_og_type
                )

                assert isinstance(out[0], np.ndarray)
                assert isinstance(out[1], np.ndarray)
                if _x_og_type is not None:
                    assert out[2] == _x_og_type
                if _y_og_type is not None:
                    assert out[3] == _y_og_type
                assert out[4] is None

        else:
            with pytest.raises(TypeError):
                _handle_X_y(
                    X_np,
                    y_np,
                    _name,
                    _x_og_type,
                    _y_og_type
                )


    @pytest.mark.parametrize('_x_og_type',
        ('junk', 'garbage', 'trash', 'rubbish', 'refuse', 'waste')
    )
    @pytest.mark.parametrize('_y_og_type',
        ('junk', 'garbage', 'trash', 'rubbish', 'refuse', 'waste')
    )
    def test_accept_good_og_types_value_error_bad(self, _name, X_np, y_np,
        _x_og_type, _y_og_type, _allowed_og_types):

        with pytest.raises(ValueError):
            _handle_X_y(
                X_np,
                y_np,
                _name,
                _x_og_type,
                _y_og_type
            )



    ####################################################################
    # TEST X ###########################################################
    @pytest.mark.parametrize('junk_X',
        (0, True, None, np.pi, min, lambda x: x, {'a': 1}, {1,2}, 'junk')
    )
    def test_reject_junk_X(self, _name, junk_X):

        with pytest.raises(TypeError):
            _handle_X_y(junk_X, None, _name, None, None)


    def test_accept_X_dask_np_pd(self, _name, X_np, X_pd):
        # **
        out = _handle_X_y(X_np[:,0].ravel(), None, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(X_np[:,0].reshape((-1, 1)), out[0])
        assert out[1] is None
        assert out[2] == 'numpy_array'
        assert out[3] is None
        assert out[4] is None
        # **
        # **
        out = _handle_X_y(X_np, None, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(X_np, out[0])
        assert out[1] is None
        assert out[2] == 'numpy_array'
        assert out[3] is None
        assert out[4] is None
        # **
        # **
        out = _handle_X_y(X_pd, None, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(X_pd.to_numpy(), out[0])
        assert out[1] is None
        assert out[2] == 'pandas_dataframe'
        assert out[3] is None
        assert np.array_equiv(out[4], X_pd.columns)
        # **
        # **
        X_series = X_pd.iloc[:, 0].squeeze()
        out = _handle_X_y(X_series, None, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(X_series.to_numpy().reshape((-1, 1)), out[0])
        assert out[1] is None
        assert out[2] == 'pandas_series'
        assert out[3] is None
        assert np.array_equiv(out[4], X_series.to_frame().columns)
        # **
        # **
        # ** dask, as single [] (would be one row) **
        X_da = da.from_array(X_np, chunks=(2,2))
        out = _handle_X_y(X_da[:,0], None, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(X_da[:,0].compute().reshape((-1,1)), out[0])
        assert out[1] is None
        assert out[2] == 'numpy_array'
        assert out[3] is None
        assert out[4] is None
        # **
        # **
        # ** dask, as [[], ...] **
        X_da = da.from_array(X_np, chunks=(2,2))
        out = _handle_X_y(X_da, None, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(X_da.compute(), out[0])
        assert out[1] is None
        assert out[2] == 'numpy_array'
        assert out[3] is None
        assert out[4] is None
        # **
        # **
        X_da = da.from_array(X_np, chunks=(2, 2))
        out = _handle_X_y(X_da, None, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(X_da.compute(), out[0])
        assert out[1] is None
        assert out[2] == 'numpy_array'
        assert out[3] is None
        assert out[4] is None
        # **
        # **
        X_ddf = ddf.from_pandas(X_pd, npartitions=2)
        out = _handle_X_y(X_ddf, None, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(X_ddf.to_dask_array().compute(), out[0])
        assert out[1] is None
        assert out[2] == 'pandas_dataframe'
        assert out[3] is None
        assert np.array_equiv(out[4], X_ddf.columns)
        # **
        # **
        X_ddf_series = X_ddf.iloc[:, 0].squeeze()
        out = _handle_X_y(X_ddf_series, None, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(
            X_ddf_series.to_dask_array().compute().reshape((-1,1)),
            out[0]
        )
        assert out[1] is None
        assert out[2] == 'pandas_series'
        assert out[3] is None
        assert np.array_equiv(out[4], X_ddf_series.to_frame().columns)
        # END ** dask **


    def test_gets_correct_X_column_name_off_series(self, X_np, y_np):

        col_name = 'xyz'
        x_as_pd_series = pd.Series(X_np[:,0].ravel(), name=col_name)

        out = _handle_X_y(
            x_as_pd_series,
            y_np,
            'MinCountTransformer',
            None,
            None
        )

        out_col_name = out[-1]

        assert isinstance(out_col_name, np.ndarray)
        assert out_col_name == col_name


        out = _handle_X_y(
            ddf.from_pandas(x_as_pd_series, npartitions=1),
            y_np,
            'MinCountTransformer',
            None,
            None
        )

        out_col_name = out[-1]

        assert isinstance(out_col_name, np.ndarray)
        assert out_col_name == col_name

    # END TEST X #######################################################
    ####################################################################



    ####################################################################
    # TEST Y ###########################################################
    def test_accept_Y_dask_np_pd(self, _name, X_np, X_pd, y_np, y_pd):
        # **
        out = _handle_X_y(X_np, y_np[:,0].ravel(), _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(out[0], X_np)
        assert isinstance(out[1], np.ndarray)
        assert np.array_equiv(out[1], y_np[:,0].ravel())
        assert out[2] == 'numpy_array'
        assert out[3] == 'numpy_array'
        assert out[4] is None
        # **
        # **
        out = _handle_X_y(X_np, y_np, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(out[0], X_np)
        assert isinstance(out[1], np.ndarray)
        assert np.array_equiv(out[1], y_np)
        assert out[2] == 'numpy_array'
        assert out[3] == 'numpy_array'
        assert out[4] is None
        # **
        # **
        out = _handle_X_y(X_pd, y_pd, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(X_pd.to_numpy(), out[0])
        assert isinstance(out[1], np.ndarray)
        assert np.array_equiv(y_pd, out[1])
        assert out[2] == 'pandas_dataframe'
        assert out[3] == 'pandas_dataframe'
        assert np.array_equiv(out[4], X_pd.columns)
        # **
        # **
        y_series = y_pd.iloc[:, 0].squeeze()
        out = _handle_X_y(X_pd, y_series, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(X_pd.to_numpy(), out[0])
        assert isinstance(out[1], np.ndarray)
        assert np.array_equiv(y_series.to_numpy().reshape((-1,1)), out[1])
        assert out[2] == 'pandas_dataframe'
        assert out[3] == 'pandas_series'
        assert np.array_equiv(out[4], X_pd.columns)
        # **
        # **
        # ** dask, as [] **
        X_da = da.from_array(X_np, chunks=(2,2))[:,0].ravel()
        y_da = da.from_array(y_np, chunks=(2,1))[:,0].ravel()
        out = _handle_X_y(X_da, y_da, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(X_da.compute().reshape((-1, 1)), out[0])
        assert isinstance(out[1], np.ndarray)
        assert np.array_equiv(y_da.compute(), out[1])
        assert out[2] == 'numpy_array'
        assert out[3] == 'numpy_array'
        assert out[4] is None
        # **
        # **
        # ** dask, as [[], ...] **
        X_da = da.from_array(X_np, chunks=(2, 2))
        y_da = da.from_array(y_np, chunks=(2, 1))
        out = _handle_X_y(X_da, y_da, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(X_da.compute(), out[0])
        assert isinstance(out[1], np.ndarray)
        assert np.array_equiv(y_da.compute(), out[1])
        assert out[2] == 'numpy_array'
        assert out[3] == 'numpy_array'
        assert out[4] is None
        # **
        # **
        X_ddf = ddf.from_pandas(X_pd, npartitions=2)
        y_ddf = ddf.from_pandas(y_pd, npartitions=2)
        out = _handle_X_y(X_ddf, y_ddf, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(X_ddf.to_dask_array().compute(), out[0])
        assert isinstance(out[1], np.ndarray)
        assert np.array_equiv(y_ddf.to_dask_array().compute(), out[1])
        assert out[2] == 'pandas_dataframe'
        assert out[3] == 'pandas_dataframe'
        assert np.array_equiv(out[4], X_ddf.columns)
        # **
        # **
        X_ddf_series = X_ddf.iloc[:, 0].squeeze()
        y_ddf_series = y_ddf.iloc[:, 0].squeeze()
        out = _handle_X_y(X_ddf_series, y_ddf_series, _name, None, None)

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(
            X_ddf_series.to_frame().to_dask_array().compute().reshape((-1,1)),
            out[0]
        )
        assert isinstance(out[1], np.ndarray)
        assert np.array_equiv(
            y_ddf_series.to_frame().to_dask_array().compute(),
            out[1]
        )
        assert out[2] == 'pandas_series'
        assert out[3] == 'pandas_series'
        assert np.array_equiv(out[4], X_ddf_series.to_frame().columns)
        # END ** dask **


    # END TEST Y #######################################################
    ####################################################################






    def test_ValueError_row_mismatch(self, _name, X_np, y_np):

        bad_y = y_np.ravel()[:-1].reshape((-1,1))

        with pytest.raises(ValueError):
            _handle_X_y(
                X_np,
                bad_y,
                _name,
                'numpy_array',
                'numpy_array'
            )

















