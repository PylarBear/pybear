# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




import pytest

from uuid import uuid4
import numpy as np
import pandas as pd
import scipy.sparse as ss

from pybear.utilities._nan_masking import nan_mask



# this tests the X_factory fixture in conftest that makes X for the other tests


class TestXFactory:

    # def _X_factory():
    #
    #     def foo(
    #         _dupl:list[list[int]]=None,
    #         _has_nan:bool=False,
    #         _format:Literal['np', 'pd', 'csc', 'csr', 'coo']='np',
    #         _dtype:Literal['flt','int','str','obj','hybrid']='flt',
    #         _columns:Union[Iterable[str], None]=None,
    #         _shape:tuple[int,int]=(20,5)
    #     ) -> npt.NDArray[any]:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (200, 10)

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *




    # test validation ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('junk_dupl',
        (-1,0,1,np.pi,True,'junk',{'a':1},min,lambda x: x)
    )
    def test_rejects_junk_dupl(self, _X_factory, junk_dupl):
        with pytest.raises(AssertionError):
            _X_factory(_dupl=junk_dupl)

    @pytest.mark.parametrize('bad_dupl',
        ([0,1], ['a','b'], (1,2,3))
    )
    def test_rejects_bad_dupl(self, _X_factory, bad_dupl):
        with pytest.raises(AssertionError):
            _X_factory(_dupl=bad_dupl)

    @pytest.mark.parametrize('good_dupl', ([[1,2,3],[0,4]], [[0,2]]))
    def test_accepts_good_dupl(self, _X_factory, good_dupl):
        _X_factory(_dupl=good_dupl)

    # - - - - - -

    @pytest.mark.parametrize('junk_has_nan',
        ('junk',[0,1],(1,),{'a':1},min,lambda x: x)
    )
    def test_rejects_junk_has_nan(self, _X_factory, junk_has_nan):
        with pytest.raises(AssertionError):
            _X_factory(_has_nan=junk_has_nan)

    @pytest.mark.parametrize('bad_has_nan', (-1,0,np.pi))
    def test_accepts_bool_has_nan(self, _X_factory, bad_has_nan):
        _X_factory(_has_nan=bad_has_nan)

    @pytest.mark.parametrize('bool_has_nan', (True,False))
    def test_accepts_bool_has_nan(self, _X_factory, bool_has_nan):
        _X_factory(_has_nan=bool_has_nan)

    @pytest.mark.parametrize('int_has_nan', (3,4,5))
    def test_accepts_int_has_nan(self, _X_factory, int_has_nan):
        _X_factory(_has_nan=int_has_nan)

    # - - - - - -

    @pytest.mark.parametrize('junk_format',
        (-1,0,1,np.pi,True,[0,1],(1,),{'a':1},min,lambda x: x)
    )
    def test_rejects_junk_format(self, _X_factory, junk_format):
        with pytest.raises(AssertionError):
            _X_factory(_format=junk_format)

    @pytest.mark.parametrize('bad_format', ('trash', 'junk', 'garbage'))
    def test_rejects_bad_format(self, _X_factory, bad_format):
        with pytest.raises(AssertionError):
            _X_factory(_format=bad_format)

    @pytest.mark.parametrize('good_format', ('np', 'pd', 'csc', 'csr', 'coo'))
    def test_accepts_good_format(self, _X_factory, good_format):
        _X_factory(_format=good_format)

    # - - - - - -

    @pytest.mark.parametrize('junk_dtype',
        (-1,0,1,np.pi,True,[0,1],(1,),{'a':1},min,lambda x: x)
    )
    def test_rejects_junk_dtype(self, _X_factory, junk_dtype):
        with pytest.raises(AssertionError):
            _X_factory(_dtype=junk_dtype)

    @pytest.mark.parametrize('bad_dtype', ('trash', 'junk', 'garbage'))
    def test_rejects_bad_dtype(self, _X_factory, bad_dtype):
        with pytest.raises(AssertionError):
            _X_factory(_dtype=bad_dtype)

    @pytest.mark.parametrize('good_dtype', ('flt', 'int', 'str', 'obj', 'hybrid'))
    def test_accepts_good_dtype(self, _X_factory, good_dtype):
        _X_factory(_dtype=good_dtype)

    # - - - - - -

    @pytest.mark.parametrize('junk_columns',
        (-1,0,1,np.pi,True,'trash',{'a':1},min,lambda x: x)
    )
    def test_rejects_junk_columns(self, _X_factory, junk_columns):
        with pytest.raises(AssertionError):
            _X_factory(_columns=junk_columns)

    @pytest.mark.parametrize('bad_columns', ([0,1], [[0,1]], {1,2}))
    def test_rejects_bad_columns(self, _X_factory, bad_columns):
        with pytest.raises(AssertionError):
            _X_factory(_columns=bad_columns)

    @pytest.mark.parametrize('good_columns', (list('abcde'),))
    def test_accepts_good_columns(self, _X_factory, good_columns):
        _X_factory(_columns=good_columns)

    # - - - - - -

    @pytest.mark.parametrize('junk_constants',
        (-1,0,1,np.pi,True,'trash',{'a':1},min,lambda x: x)
    )
    def test_rejects_junk_constants(self, _X_factory, junk_constants):
        with pytest.raises(AssertionError):
            _X_factory(_constants=junk_constants)

    @pytest.mark.parametrize('bad_constants', ([[0,1]], ['a','b']))
    def test_rejects_bad_constants(self, _X_factory, bad_constants):
        with pytest.raises(AssertionError):
            _X_factory(_constants=bad_constants)

    @pytest.mark.parametrize('good_constants', ([0,2],))
    def test_accepts_good_constants(self, _X_factory, good_constants):
        _X_factory(_constants=good_constants)

    # - - - - - -

    @pytest.mark.parametrize('junk_zeros',
        ('trash',[0,1],(1,),{0,1},{'a':1},min,lambda x: x)
    )
    def test_rejects_junk_zeros(self, _X_factory, junk_zeros):
        with pytest.raises(AssertionError):
            _X_factory(_zeros=junk_zeros)

    @pytest.mark.parametrize('bad_zeros', (-1,True,False,np.pi,))
    def test_rejects_bad_zeros(self, _X_factory, bad_zeros):
        with pytest.raises(AssertionError):
            _X_factory(_zeros=bad_zeros)

    @pytest.mark.parametrize('good_zeros', (0,0.5,1))
    def test_accepts_good_zeros(self, _X_factory, good_zeros):
        _X_factory(_zeros=good_zeros)

    # - - - - - -

    @pytest.mark.parametrize('junk_shape',
        (-1,0,1,np.pi,True,'trash',{'a':1},min,lambda x: x)
    )
    def test_rejects_junk_shape(self, _X_factory, junk_shape):
        with pytest.raises(AssertionError):
            _X_factory(_shape=junk_shape)

    @pytest.mark.parametrize('bad_shape', (('a',1), (2,1), [[0,1]], {1,2}))
    def test_rejects_bad_shape(self, _X_factory, bad_shape):
        with pytest.raises(AssertionError):
            _X_factory(_shape=bad_shape)

    @pytest.mark.parametrize('good_shape', ((20,5), (100,3), (13,13)))
    def test_accepts_good_shape(self, _X_factory, good_shape):
        _X_factory(_shape=good_shape)


    # END test validation ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # test accuracy

    @pytest.mark.parametrize('_format', ('np', 'pd', 'csc', 'csr', 'coo'))
    @pytest.mark.parametrize('_dtype', ('flt', 'int', 'str', 'obj', 'hybrid'))
    def test_format_dtype_shape_accuracy(self, _X_factory, _format, _dtype, _shape):

        if _format in ('csc', 'csr', 'coo') and _dtype in ('str', 'obj', 'hybrid'):
            with pytest.raises(Exception):
                _X_factory(_format=_format, _dtype=_dtype, _shape=_shape)
            pytest.skip(reason=f"scipy sparse cant take string types")
        else:
            out = _X_factory(_format=_format, _dtype=_dtype, _shape=_shape)

        assert out.shape == _shape

        if _format == 'np':
            assert isinstance(out, np.ndarray)
        elif _format == 'pd':
            assert isinstance(out, pd.core.frame.DataFrame)
        elif _format == 'csc':
            assert isinstance(out, (ss._csc.csc_matrix, ss._csc.csc_array))
        elif _format == 'csr':
            assert isinstance(out, (ss._csr.csr_matrix, ss._csr.csr_array))
        elif _format == 'coo':
            assert isinstance(out, (ss._coo.coo_matrix, ss._coo.coo_array))


        if _format in ('np', 'csc', 'csr', 'coo'):
            if _dtype == 'flt':
                assert out.dtype == np.float64
            elif _dtype == 'int':
                assert out.dtype in [np.int32, np.int64]
            elif _dtype == 'str':
                assert '<U' in str(out.dtype)
            elif _dtype == 'obj':
                assert out.dtype == object
            elif _dtype == 'hybrid':
                assert out.dtype == object
        elif _format == 'pd':
            assert isinstance(out, pd.core.frame.DataFrame)
            if _dtype == 'flt':
                assert all([__ == np.float64 for __ in out.dtypes])
            elif _dtype == 'int':
                assert all([__ in [np.int32, np.int64] for __ in out.dtypes])
            elif _dtype == 'str':
                assert all([__ == object for __ in out.dtypes])
            elif _dtype == 'obj':
                assert all([__ == object for __ in out.dtypes])
            elif _dtype == 'hybrid':
                assert all([__ == object for __ in out.dtypes])


    @pytest.mark.parametrize('_columns', (True, None))
    def test_columns_accuracy(self, _X_factory, _columns, _shape):

        _ref_columns = [str(uuid4())[:4] for _ in range(_shape[1])]

        out = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='pd',
            _dtype='flt',
            _columns=_ref_columns if _columns else None,
            _shape=_shape
        )

        if _columns:
            assert np.array_equiv(out.columns, _ref_columns)
        else:
            assert np.array_equiv(out.columns, range(_shape[1]))


    @pytest.mark.parametrize('_constants', ([0,2], None))
    def test_constants_accuracy(self, _X_factory, _constants, _shape):

        out = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _constants=_constants,
            _shape=_shape
        )

        for c_idx in range(out.shape[1]):
            if c_idx in (_constants or []):
                assert len(np.unique(out[:, c_idx])) == 1
            else:
                assert len(np.unique(out[:, c_idx])) > 1


    @pytest.mark.parametrize('_dupl',
            #  vvvvvvvvvvvv should except
        (None, [[1], [0,2]], [[0,2,3],[1,4]], [[0,4],[1,3]])
    )
    def test_dupl_accuracy(self, _X_factory, _dupl, _shape):

        if _dupl is not None and min(map(len, _dupl)) < 2:
            with pytest.raises(Exception):
                _X_factory(
                    _format='np',
                    _dtype='flt',
                    _dupl=_dupl,
                    _shape=_shape
                )
        else:
            out = _X_factory(
                _format='np',
                _dtype='flt',
                _dupl=_dupl,
                _shape=_shape
            )

            if _dupl is None:
                for c_idx_1 in range(_shape[1]-1):
                    for c_idx_2 in range(c_idx_1+1, _shape[1]):
                        assert not np.array_equal(out[:, c_idx_1], out[:, c_idx_2])
            else:
                for _set in _dupl:
                    for _idx in _set[1:]:
                        assert np.array_equal(out[:, _idx], out[:, _set[0]])



    @pytest.mark.parametrize('_format', ('np', 'pd', 'csc', 'csr', 'coo'))
    @pytest.mark.parametrize('_dtype', ('flt', 'int', 'str', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_has_nan', (True, False, 4, 8))
    def test_nans_accuracy(self, _X_factory, _format, _dtype, _has_nan, _shape):

        if _format in ('csc', 'csr', 'coo') and _dtype in ('str', 'obj', 'hybrid'):
            with pytest.raises(ValueError):
                _X_factory(
                    _dupl=None,
                    _has_nan=True,
                    _format=_format,
                    _dtype=_dtype,
                    _columns=None,
                    _shape=_shape
                )
        else:
            # out would be in the format given
            # to make this easier, convert to np
            out = _X_factory(
                _dupl=None,
                _has_nan=_has_nan,
                _format=_format,
                _dtype=_dtype,
                _columns=None,
                _shape=_shape
            )


            if isinstance(out, np.ndarray):
                pass
            elif isinstance(out, pd.core.frame.DataFrame):
                out = out.to_numpy()
            else:
                out = out.toarray()

            # if

            for c_idx in range(_shape[1]):
                _num_nans = np.sum(nan_mask(out[:, c_idx]))
                if _has_nan is False:
                    assert _num_nans == 0
                elif _has_nan is True:
                    assert 3 <= _num_nans <= max(3, _shape[0]//10)
                elif isinstance(_has_nan, int):
                    assert _num_nans == _has_nan
                else:
                    raise Exception


    @pytest.mark.parametrize('_format', ('np', 'pd'))
    @pytest.mark.parametrize('_dtype', ('flt', 'int', 'hybrid'))
    @pytest.mark.parametrize('_zeros', (0, 0.25, 0.75, 1))
    def test_zeros_accuracy(self, _X_factory, _shape, _format, _dtype, _zeros):

        # nans are applied after the zeros. get accuracy w/o nans.

        _tol = 0.2

        out = _X_factory(
            _dupl=None,
            _format=_format,
            _dtype=_dtype,
            _has_nan=False,
            _columns=None,
            _zeros=_zeros,
            _shape=_shape
        )

        if _format == 'pd':
            out = out.to_numpy()


        if _dtype in ['flt', 'int']:
            MASK = (out == 0)
            _frac = np.sum(MASK) / out.size
            if _zeros == 0:
                assert _frac <= _tol
            elif _zeros == 1:
                assert _frac >= 1 - _tol
            else:
                assert (_frac - _tol) <= _zeros <= (_frac + _tol)

        elif _dtype == 'hybrid':
            for _c_idx in range(_shape[1]):
                try:
                    out[:, _c_idx].astype(np.float64)
                except:
                    continue

                # to get here, column must be numeric
                _ = out[:, _c_idx]
                MASK = (_ == 0)
                _frac = np.sum(MASK) / _.size

                if _zeros == 0:
                    assert _frac <= _tol
                elif _zeros == 1:
                    assert _frac >= 1 - _tol
                else:
                    assert (_frac - _tol) <= _zeros <= (_frac + _tol)




