# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import pandas as pd
import scipy.sparse as ss
import polars as pl


from pybear.model_selection.GSTCV._GSTCV._fit._fold_splitter import \
    _fold_splitter



class TestSKFoldSplitter:

    # def _fold_splitter(
    #     train_idxs: Union[GenericSlicerType, SKSlicerType],
    #     test_idxs: Union[GenericSlicerType, SKSlicerType],
    #     *data_objects: Union[XSKWIPType, YSKWIPType],
    # ) -> tuple[tuple[XSKWIPType, YSKWIPType], ...]:


    # fixtures -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    @staticmethod
    @pytest.fixture(scope='function')
    def _base_X(_rows, _cols):
        return np.random.randint(0, 10, (_rows, _cols))


    @staticmethod
    @pytest.fixture(scope='function')
    def _format_helper():

        # pizza would this be better off in conftest? (is something similar
        # used in other test modules?)   look in get_kfold_test

        def foo(_base, _format: str, _dim: int):

            """Cast dummy numpy array to desired container."""

            # _X can be X or y in the tests

            if _format == 'ss' and _dim == 1:
                raise ValueError(f"cant have 1D scipy sparse")

            if _format == 'py_set' and _dim == 2:
                raise ValueError(f"cant have 2D set")

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


            if _format == 'py_list':
                if _dim == 1:
                    _X = list(_X)
                elif _dim == 2:
                    _X = list(map(list, _X))
            elif _format == 'py_tup':
                if _dim == 1:
                    _X = tuple(_X)
                elif _dim == 2:
                    _X = tuple(map(tuple, _X))
            elif _format == 'py_set':
                if _dim == 1:
                    _X = set(_X)
                elif _dim == 2:
                    # should have raised above
                    raise Exception
            elif _format == 'np':
                pass
            elif _format == 'pd':
                if _dim == 1:
                    _X = pd.Series(_X)
                elif _dim == 2:
                    _X = pd.DataFrame(_X)
            elif _format == 'ss':
                if _dim == 1:
                    # should have raised above
                    raise Exception
                elif _dim == 2:
                    _X = ss.csr_array(_X)
            elif _format == 'pl':
                if _dim == 1:
                    _X = pl.Series(_X)
                elif _dim == 2:
                    _X = pl.from_numpy(_X)
            else:
                raise Exception

            return _X

        return foo

    # END fixtures -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    @pytest.mark.parametrize('bad_data_object',
        (1, 3.14, True, False, None, 'junk', min, [0,1], (0,1), {0,1},
        {'a':1}, lambda x: x)
    )
    def test_rejects_everything_without_shape_attr(self, bad_data_object):

        with pytest.raises(AttributeError):
            _fold_splitter(
                [0,2,4],
                [1,3],
                bad_data_object
            )


    @pytest.mark.parametrize('bad_data_object',
        (
            np.random.randint(0, 10, (3, 3, 3)),
            np.random.randint(0, 10, (3, 3, 3, 3)),
        )
    )
    def test_rejects_bad_shape(self, bad_data_object):

        with pytest.raises(AssertionError):
            _fold_splitter(
                [0,2,4],
                [1,3],
                bad_data_object
            )


    @pytest.mark.parametrize('_X1_format',
        ('py_list', 'py_tup', 'py_set', 'np', 'pd', 'ss', 'pl')
    )
    @pytest.mark.parametrize('_X1_dim', (1, 2))
    @pytest.mark.parametrize('_X2_format',
        ('py_list', 'py_tup', 'py_set', 'np', 'pd', 'ss', 'pl')
    )
    @pytest.mark.parametrize('_X2_dim', (1, 2))
    def test_accuracy(
        self, _rows, _base_X, _format_helper, _X1_format, _X1_dim, _X2_format,
        _X2_dim
    ):

        if (_X1_dim == 1 and _X1_format == 'ss') \
                or (_X2_dim == 1 and _X2_format == 'ss'):
            pytest.skip(reason=f"cant have 1D scipy sparse")
        if (_X1_dim == 2 and _X1_format == 'py_set') \
                or (_X2_dim == 2 and _X2_format == 'py_set'):
            pytest.skip(reason=f"cant have 2D set")

        # END skip impossible -- -- -- -- -- -- -- -- -- -- -- -- -- --

        _X1 = _format_helper(_base_X, _X1_format, _X1_dim)
        _X2 = _format_helper(_base_X, _X2_format, _X2_dim)


        _helper_mask = np.random.randint(0, 2, (_rows,)).astype(bool)
        mask_train = np.arange(_rows)[_helper_mask]
        mask_test = np.arange(_rows)[np.logical_not(_helper_mask)]
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

        # 25_04_28 only takes objects that have 'shape' attr
        if _X1_format in ('py_list', 'py_tup', 'py_set') \
                or _X2_format in ('py_list', 'py_tup', 'py_set'):
            with pytest.raises(AttributeError):
                _fold_splitter(mask_train, mask_test, _X1, _X2)
            pytest.skip(reason=f"cant do more tests")
        else:
            out = _fold_splitter(mask_train, mask_test, _X1, _X2)

        assert isinstance(out, tuple)
        assert all(map(isinstance, out, (tuple for i in out)))


        # shouldnt get to here for py_list, py_tup, py_set

        assert type(out[0][0]) == type(_X1)
        if _X1_format == 'np':
            assert np.array_equiv(out[0][0], _X1_ref_train)
        elif _X1_format == 'pd':
            assert np.array_equiv(out[0][0].to_numpy(), _X1_ref_train)
        elif _X1_format == 'ss':
            assert np.array_equiv(out[0][0].toarray(), _X1_ref_train)
        elif _X1_format == 'pl':
            assert np.array_equiv(out[0][0].to_numpy(), _X1_ref_train)
        else:
            raise Exception

        assert type(out[0][1]) == type(_X1)
        if _X1_format == 'np':
            assert np.array_equiv(out[0][1], _X1_ref_test)
        elif _X1_format == 'pd':
            assert np.array_equiv(out[0][1].to_numpy(), _X1_ref_test)
        elif _X1_format == 'ss':
            assert np.array_equiv(out[0][1].toarray(), _X1_ref_test)
        elif _X1_format == 'pl':
            assert np.array_equiv(out[0][1].to_numpy(), _X1_ref_test)
        else:
            raise Exception


        assert type(out[1][0]) == type(_X2)
        if _X2_format == 'np':
            assert np.array_equiv(out[1][0], _X2_ref_train)
        elif _X2_format == 'pd':
            assert np.array_equiv(out[1][0].to_numpy(), _X2_ref_train)
        elif _X2_format == 'ss':
            assert np.array_equiv(out[1][0].toarray(), _X2_ref_train)
        elif _X2_format == 'pl':
            assert np.array_equiv(out[1][0].to_numpy(), _X2_ref_train)
        else:
            raise Exception

        assert type(out[1][1]) == type(_X2)
        if _X2_format == 'np':
            assert np.array_equiv(out[1][1], _X2_ref_test)
        elif _X2_format == 'pd':
            assert np.array_equiv(out[1][1].to_numpy(), _X2_ref_test)
        elif _X2_format == 'ss':
            assert np.array_equiv(out[1][1].toarray(), _X2_ref_test)
        elif _X2_format == 'pl':
            assert np.array_equiv(out[1][1].to_numpy(), _X2_ref_test)
        else:
            raise Exception





