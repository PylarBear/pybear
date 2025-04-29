# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import dask.array as da
import dask.dataframe as ddf

from pybear.model_selection.GSTCV._GSTCVDask._fit._fold_splitter import \
    _fold_splitter



class TestFoldSplitter:

    # def _fold_splitter(
    #     train_idxs: Union[GenericSlicerType, DaskSlicerType],
    #     test_idxs: Union[GenericSlicerType, DaskSlicerType],
    #     *data_objects: Union[XDaskWIPType, YDaskWIPType]
    # ) -> tuple[tuple[XDaskWIPType, YDaskWIPType, ...]]:


    # fixtures -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    @staticmethod
    @pytest.fixture(scope='function')
    def _base_X(_rows, _cols):
        return da.random.randint(0, 10, (_rows, _cols))


    @staticmethod
    @pytest.fixture(scope='function')
    def _format_helper(_rows, _cols):

        # pizza would this be better off in conftest? (is something similar
        # used in other test modules?)

        def foo(_base, _format: str, _dim: int):

            """Cast dummy numpy array to desired container."""

            # _X can be X or y in the tests

            if _dim == 1 and len(_base.shape)==1:
                _X = _base.copy()
            elif _dim == 2 and len(_base.shape)==2:
                _X = _base.copy()
            elif _dim == 1 and len(_base.shape)==2:
                _X = _base[:, 0].copy().ravel()
            elif _dim == 2 and len(_base.shape)==1:
                _X = _base.copy().reshape((-1, 1))
            else:
                raise Exception


            if _format == 'da':
                pass
            elif _format == 'ddf':
                if _dim == 1:
                    _X = ddf.from_dask_array(_X).squeeze()
                elif _dim == 2:
                    _X = ddf.from_dask_array(_X)
            else:
                raise Exception

            return _X

        return foo

    # END fixtures -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    @pytest.mark.parametrize('bad_data_object',
        (1, 3.14, True, False, None, 'junk', min, [0,1], (0,1), {0,1},
        {'a':1}, lambda x: x, np.random.randint(0,10,(5,3)))
    )
    def test_rejects_everything_not_dask(self, bad_data_object):

        with pytest.raises(TypeError):
            _fold_splitter(
                [0,2,4],
                [1,3],
                bad_data_object
            )


    @pytest.mark.parametrize('bad_data_object',
        (
            da.random.randint(0, 10, (3, 3, 3)),
            da.random.randint(0, 10, (3, 3, 3, 3)),
        )
    )
    def test_rejects_bad_shape(self, bad_data_object):

        with pytest.raises(AssertionError):
            _fold_splitter(
                [0,2,4],
                [1,3],
                bad_data_object
            )


    @pytest.mark.parametrize('_X1_format', ('da', 'ddf'))
    @pytest.mark.parametrize('_X1_dim', (1, 2))
    @pytest.mark.parametrize('_X2_format', ('da', 'ddf'))
    @pytest.mark.parametrize('_X2_dim', (1, 2))
    def test_accuracy(
        self, _rows, _base_X, _format_helper, _X1_format, _X1_dim, _X2_format,
        _X2_dim
    ):

        _X1 = _format_helper(_base_X, _X1_format, _X1_dim)
        _X2 = _format_helper(_base_X, _X2_format, _X2_dim)

        _helper_mask = da.random.randint(0, 2, (_rows,)).astype(bool)
        mask_train = da.arange(_rows)[_helper_mask]
        mask_test = da.arange(_rows)[da.logical_not(_helper_mask)]
        del _helper_mask

        if _X1_dim == 1:
            _X1_ref_train = _base_X[:, 0][mask_train]
            _X1_ref_test = _base_X[:, 0][mask_test]
        elif _X1_dim == 2:
            _X1_ref_train = _base_X[mask_train]
            _X1_ref_test = _base_X[mask_test]
        else:
            raise Exception

        if _X2_dim == 1:
            _X2_ref_train = _base_X[:, 0][mask_train]
            _X2_ref_test = _base_X[:, 0][mask_test]
        elif _X2_dim == 2:
            _X2_ref_train = _base_X[mask_train]
            _X2_ref_test = _base_X[mask_test]
        else:
            raise Exception

        out = _fold_splitter(mask_train, mask_test, _X1, _X2)

        assert isinstance(out, tuple)
        assert all(map(isinstance, out, (tuple for i in out)))

        assert type(out[0][0]) == type(_X1)
        if _X1_format == 'da':
            assert np.array_equal(out[0][0].compute(), _X1_ref_train)
        elif _X1_format == 'ddf':
            assert np.array_equal(out[0][0].compute().to_numpy(), _X1_ref_train)
        else:
            raise Exception

        assert type(out[0][1]) == type(_X1)
        if _X1_format == 'da':
            assert np.array_equal(out[0][1].compute(), _X1_ref_test)
        elif _X1_format == 'ddf':
            assert np.array_equal(out[0][1].compute().to_numpy(), _X1_ref_test)
        else:
            raise Exception

        assert type(out[1][0]) == type(_X2)
        if _X2_format == 'da':
            assert np.array_equal(out[1][0].compute(), _X2_ref_train)
        elif _X2_format == 'ddf':
            assert np.array_equal(out[1][0].compute().to_numpy(), _X2_ref_train)
        else:
            raise Exception

        assert type(out[1][1]) == type(_X2)
        if _X2_format == 'da':
            assert np.array_equal(out[1][1].compute(), _X2_ref_test)
        elif _X2_format == 'ddf':
            assert np.array_equal(out[1][1].compute().to_numpy(), _X2_ref_test)
        else:
            raise Exception





