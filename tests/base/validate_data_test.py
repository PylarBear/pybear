# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.base._validate_data import validate_data

import numpy as np

import pytest





class Fixtures:

    @staticmethod
    @pytest.fixture()
    def _X_np():
        return np.random.randint(0, 10, (20, 10))


    @staticmethod
    @pytest.fixture()
    def _good_accept_sparse():
        return ('csr', 'csc', 'coo')




class TestValidateData_ParamValidation(Fixtures):

    # def validate_data(
    #     X,
    #     *,
    #     copy_X:bool=True,
    #     cast_to_ndarray:bool=False,
    #     accept_sparse:Iterable[Literal[
    #         "csr", "csc", "coo", "dia", "lil", "dok", "bsr"
    #     ]]=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
    #     dtype:Literal['numeric','any']='any',
    #     require_all_finite:bool=True,
    #     cast_inf_to_nan:bool=True,
    #     standardize_nan:bool=True,
    #     ensure_2d:bool=True,
    #     order:Literal['C', 'F']='C',
    #     ensure_min_features:numbers.Integral=1,
    #     ensure_min_samples:numbers.Integral=1
    # ):

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # no validation for X, the entire module is for validating X!

    @pytest.mark.parametrize('_param',
        ('copy_X', 'cast_to_ndarray', 'require_all_finite',
        'cast_inf_to_nan', 'standardize_nan', 'ensure_2d')
    )
    @pytest.mark.parametrize('junk',
        (-2.7, -1, 0, 1, 2.7, None, 'junk', [0,1], (0,1), {'A':1}, lambda x: x)
    )
    def test_junk_bool_params(self, _X_np, _good_accept_sparse, _param, junk):

        with pytest.raises(TypeError):
            validate_data(
                _X_np,
                copy_X=junk if _param=='copy_X' else False,
                cast_to_ndarray=junk if _param=='cast_to_ndarray' else False,
                accept_sparse=_good_accept_sparse,
                dtype='any',
                require_all_finite=junk if _param=='require_all_finite' else False,
                cast_inf_to_nan=junk if _param=='cast_inf_to_nan' else False,
                standardize_nan=junk if _param=='standardize_nan' else False,
                ensure_2d=junk if _param=='ensure_2d' else False,
                order='C',
                ensure_min_features=1,
                ensure_min_samples=1
            )


    @pytest.mark.parametrize('_param',
        ('copy_X', 'cast_to_ndarray', 'require_all_finite',
        'cast_inf_to_nan', 'standardize_nan', 'ensure_2d')
    )
    def test_good_bool_params(self, _X_np, _good_accept_sparse, _param):

        out = validate_data(
            _X_np,
            copy_X=(_param=='copy_X'),
            cast_to_ndarray=(_param=='cast_to_ndarray'),
            accept_sparse=_good_accept_sparse,
            dtype='any',
            require_all_finite=(_param=='require_all_finite'),
            cast_inf_to_nan=(_param=='cast_inf_to_nan'),
            standardize_nan=(_param=='standardize_nan'),
            ensure_2d=(_param=='ensure_2d'),
            order='C',
            ensure_min_features=1,
            ensure_min_samples=1
        )

        assert isinstance(out, np.ndarray)


    def test_require_all_finite_conditionals(
        self, _X_np, _good_accept_sparse
    ):

        # f"if :param: require_all_finite is True, then :param: "
        # f"standardize_nan must be False."

        with pytest.raises(ValueError):
            validate_data(
                _X_np,
                copy_X=False,
                cast_to_ndarray=False,
                accept_sparse=_good_accept_sparse,
                dtype='any',
                require_all_finite=True,
                cast_inf_to_nan=True,
                standardize_nan=False,
                ensure_2d=False,
                order='C',
                ensure_min_features=1,
                ensure_min_samples=1
            )

        # f"if :param: require_all_finite is True, then :param: "
        # f"cast_inf_to_nan must be False."

        with pytest.raises(ValueError):
            validate_data(
                _X_np,
                copy_X=False,
                cast_to_ndarray=False,
                accept_sparse=_good_accept_sparse,
                dtype='any',
                require_all_finite=True,
                cast_inf_to_nan=False,
                standardize_nan=True,
                ensure_2d=False,
                order='C',
                ensure_min_features=1,
                ensure_min_samples=1
            )


    # accept_sparse -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk',
        (-2.7, -1, 0, 1, 2.7, 'junk', [['csc','csr']], {'A':1}, lambda x: x)
    )
    def test_rejects_junk_accept_sparse(self, _X_np, junk):

        with pytest.raises(TypeError):
            validate_data(
                _X_np,
                copy_X=False,
                cast_to_ndarray=False,
                accept_sparse=junk,
                dtype='any',
                require_all_finite=False,
                cast_inf_to_nan=False,
                standardize_nan=False,
                ensure_2d=False,
                order='C',
                ensure_min_features=1,
                ensure_min_samples=1
            )


    @pytest.mark.parametrize('bad', (['love', 'happiness'], {'junk', 'trash'}))
    def test_rejects_bad_accept_sparse(self, _X_np, bad):

        with pytest.raises(ValueError):
            validate_data(
                _X_np,
                copy_X=False,
                cast_to_ndarray=False,
                accept_sparse=bad,
                dtype='any',
                require_all_finite=False,
                cast_inf_to_nan=False,
                standardize_nan=False,
                ensure_2d=False,
                order='C',
                ensure_min_features=1,
                ensure_min_samples=1
            )


    @pytest.mark.parametrize('good',
        (None, False, ('csc', 'csr', 'coo'), {'bsr', 'dia'}, ('lil', 'dok'))
    )
    def test_good_accept_sparse(self, _X_np, good):

        out = validate_data(
            _X_np,
            copy_X=False,
            cast_to_ndarray=True,
            accept_sparse=good,
            dtype='any',
            require_all_finite=False,
            cast_inf_to_nan=False,
            standardize_nan=False,
            ensure_2d=False,
            order='C',
            ensure_min_features=1,
            ensure_min_samples=1
        )

        assert isinstance(out, np.ndarray)

    # END accept_sparse -- -- -- -- -- -- -- -- --


    # dtype -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_dtype',
        (-2.7, -1, 0, 1, 2.7, [0,1], (0,1), {'A':1}, lambda x: x)
    )
    def test_rejects_junk_dtype(self, _X_np, _good_accept_sparse, junk_dtype):

        with pytest.raises(TypeError):
            validate_data(
                _X_np,
                copy_X=False,
                cast_to_ndarray=False,
                accept_sparse=_good_accept_sparse,
                dtype=junk_dtype,
                require_all_finite=False,
                cast_inf_to_nan=False,
                standardize_nan=False,
                ensure_2d=False,
                order='C',
                ensure_min_features=1,
                ensure_min_samples=1
            )


    @pytest.mark.parametrize('bad_dtype', ('love', 'happiness', 'junk', 'trash'))
    def test_rejects_bad_dtype(self, _X_np, _good_accept_sparse, bad_dtype):

        with pytest.raises(ValueError):
            validate_data(
                _X_np,
                copy_X=False,
                cast_to_ndarray=False,
                accept_sparse=_good_accept_sparse,
                dtype=bad_dtype,
                require_all_finite=False,
                cast_inf_to_nan=False,
                standardize_nan=False,
                ensure_2d=False,
                order='C',
                ensure_min_features=1,
                ensure_min_samples=1
            )


    @pytest.mark.parametrize('good_dtype', ('numeric', 'any'))
    def test_good_dtype(self, _X_np, _good_accept_sparse, good_dtype):

        out = validate_data(
            _X_np,
            copy_X=False,
            cast_to_ndarray=True,
            accept_sparse=_good_accept_sparse,
            dtype=good_dtype,
            require_all_finite=False,
            cast_inf_to_nan=False,
            standardize_nan=False,
            ensure_2d=False,
            order='C',
            ensure_min_features=1,
            ensure_min_samples=1
        )

        assert isinstance(out, np.ndarray)

    # END dtype -- -- -- -- -- -- -- -- -- -- --


    # order -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_order',
        (-2.7, -1, 0, 1, 2.7, True, None, [0,1], (0,1), {'A':1}, lambda x: x)
    )
    def test_rejects_junk_order(self, _X_np, _good_accept_sparse, junk_order):

        with pytest.raises(TypeError):
            validate_data(
                _X_np,
                copy_X=False,
                cast_to_ndarray=False,
                accept_sparse=_good_accept_sparse,
                dtype='any',
                require_all_finite=False,
                cast_inf_to_nan=False,
                standardize_nan=False,
                ensure_2d=False,
                order=junk_order,
                ensure_min_features=1,
                ensure_min_samples=1
            )


    @pytest.mark.parametrize('bad_order', ('q', 'r', 'S', 'T'))
    def test_rejects_bad_order(self, _X_np, _good_accept_sparse, bad_order):

        with pytest.raises(ValueError):
            validate_data(
                _X_np,
                copy_X=False,
                cast_to_ndarray=False,
                accept_sparse=_good_accept_sparse,
                dtype='any',
                require_all_finite=False,
                cast_inf_to_nan=False,
                standardize_nan=False,
                ensure_2d=False,
                order=bad_order,
                ensure_min_features=1,
                ensure_min_samples=1
            )


    @pytest.mark.parametrize('good_order', ('c', 'C', 'f', 'F'))
    def test_good_order(self, _X_np, _good_accept_sparse, good_order):

        out = validate_data(
            _X_np,
            copy_X=False,
            cast_to_ndarray=True,
            accept_sparse=_good_accept_sparse,
            dtype='any',
            require_all_finite=False,
            cast_inf_to_nan=False,
            standardize_nan=False,
            ensure_2d=False,
            order=good_order,
            ensure_min_features=1,
            ensure_min_samples=1
        )

        assert isinstance(out, np.ndarray)
    # order -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # ensure_min_features -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_min_features',
        (-2.7, 2.7, True, False, None, [0,1], (0,1), {'A':1}, lambda x: x, min)
    )
    def test_rejects_junk_min_features(
        self, _X_np, _good_accept_sparse, junk_min_features
    ):

        with pytest.raises(TypeError):
            validate_data(
                _X_np,
                copy_X=False,
                cast_to_ndarray=False,
                accept_sparse=_good_accept_sparse,
                dtype='any',
                require_all_finite=False,
                cast_inf_to_nan=False,
                standardize_nan=False,
                ensure_2d=False,
                order='C',
                ensure_min_features=junk_min_features,
                ensure_min_samples=1
            )


    def test_rejects_bad_min_features(self, _X_np, _good_accept_sparse):

        with pytest.raises(ValueError):
            validate_data(
                _X_np,
                copy_X=False,
                cast_to_ndarray=False,
                accept_sparse=_good_accept_sparse,
                dtype='any',
                require_all_finite=False,
                cast_inf_to_nan=False,
                standardize_nan=False,
                ensure_2d=False,
                order='C',
                ensure_min_features=-1,   # <============
                ensure_min_samples=1
            )


    @pytest.mark.parametrize('good_min_features', (0, 1, 2))
    def test_good_min_features(self, _X_np, _good_accept_sparse, good_min_features):

        out = validate_data(
            _X_np,
            copy_X=False,
            cast_to_ndarray=True,
            accept_sparse=_good_accept_sparse,
            dtype='any',
            require_all_finite=False,
            cast_inf_to_nan=False,
            standardize_nan=False,
            ensure_2d=False,
            order='C',
            ensure_min_features=good_min_features,
            ensure_min_samples=1
        )

        assert isinstance(out, np.ndarray)
    # END ensure_min_features -- -- -- -- -- -- -- -- -- -- -- --

    # ensure_min_samples -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_min_samples',
        (-2.7, 2.7, True, False, None, [0,1], (0,1), {'A':1}, lambda x: x, min)
    )
    def test_rejects_junk_min_samples(
        self, _X_np, _good_accept_sparse, junk_min_samples
    ):

        with pytest.raises(TypeError):
            validate_data(
                _X_np,
                copy_X=False,
                cast_to_ndarray=False,
                accept_sparse=_good_accept_sparse,
                dtype='any',
                require_all_finite=False,
                cast_inf_to_nan=False,
                standardize_nan=False,
                ensure_2d=False,
                order='C',
                ensure_min_features=1,
                ensure_min_samples=junk_min_samples
            )


    def test_rejects_bad_min_samples(self, _X_np, _good_accept_sparse):

        with pytest.raises(ValueError):
            validate_data(
                _X_np,
                copy_X=False,
                cast_to_ndarray=False,
                accept_sparse=_good_accept_sparse,
                dtype='any',
                require_all_finite=False,
                cast_inf_to_nan=False,
                standardize_nan=False,
                ensure_2d=False,
                order='C',
                ensure_min_features=1,
                ensure_min_samples=-1  # <===========
            )


    @pytest.mark.parametrize('good_min_samples', (0, 1, 2))
    def test_good_min_samples(self, _X_np, _good_accept_sparse, good_min_samples):

        out = validate_data(
            _X_np,
            copy_X=False,
            cast_to_ndarray=True,
            accept_sparse=_good_accept_sparse,
            dtype='any',
            require_all_finite=False,
            cast_inf_to_nan=False,
            standardize_nan=False,
            ensure_2d=False,
            order='C',
            ensure_min_features=1,
            ensure_min_samples=good_min_samples
        )

        assert isinstance(out, np.ndarray)
    # END ensure_min_samples -- -- -- -- -- -- -- -- -- -- -- -- --

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *





class TestValidateDataAccuracy(Fixtures):

    """
    # avoid multiple copies of X. do not set 'copy_X' for each of the
    # functions to True! create only one copy of X, set copy_X to False
    # for all the functions.
    if copy_X:
        _X = X.copy()
    else:
        _X = X
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # accept_sparse
    check_scipy_sparse(
        _X,
        allowed=accept_sparse
    )
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if cast_to_ndarray:
        _X = _cast_to_ndarray(
            _X,
            copy_X=False
        )
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if ensure_2d:
        _X = _ensure_2D(
            _X,
            copy_X=False
        )
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if isinstance(_X, np.ndarray):
        _X = set_order(
            _X,
            order=order,
            copy_X=False
        )
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    check_dtype(
        _X,
        allowed=dtype
    )
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if require_all_finite:

        _X = check_is_finite(
            _X,
            allow_nan=not require_all_finite,
            allow_inf=not require_all_finite,
            cast_inf_to_nan=cast_inf_to_nan,
            standardize_nan=standardize_nan,
            copy_X=False
        )
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    check_shape(
        _X,
        min_features=ensure_min_features,
        min_samples=ensure_min_samples,
        allowed_dimensionality=(1,2) if ensure_min_features==1 else (2,)
        # if n_features_in_ is 1, then dimensionality could be 1 or 2,
        # for any number of features greater than 1 dimensionality must
        # be 2.
    )
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    """































