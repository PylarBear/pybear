# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._transform._conflict_warner import \
    _conflict_warner

import numpy as np

import pytest



class TestConflictWarner:


    # def _conflict_warner(
    #     _original_dtypes: OriginalDtypesType,
    #     _handle_as_bool: Union[InternalHandleAsBoolType, None],
    #     _ignore_columns: Union[InternalIgnoreColumnsType, None],
    #     _ignore_float_columns: bool,
    #     _ignore_non_binary_integer_columns: bool,
    #     _threshold: Union[int, Iterable[int]],
    #     _n_features_in: int
    # ) -> None:



    @staticmethod
    @pytest.fixture(scope='module')
    def n_features_in():
        return 10


    @staticmethod
    @pytest.fixture(scope='module')
    def allowed():
        return ['bin_int', 'int', 'float']  # leave 'obj' out of this, too complicated


    @staticmethod
    @pytest.fixture(scope='module')
    def good_og_dtypes(n_features_in, allowed):
        return np.random.choice(allowed, n_features_in, replace=True)


    # basic validation ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # ignore_columns -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('bad_ignore_columns',
        (-2.7, -1, 0, 1, 2.7, True, 'junk', {'a':1}, lambda x: x)
    )
    def test_rejects_bad_ignore_columns(
        self, good_og_dtypes, bad_ignore_columns, n_features_in
    ):

        # only None or list-like-int

        with pytest.raises(Exception):
            _conflict_warner(
                _original_dtypes=good_og_dtypes,
                _handle_as_bool=None,
                _ignore_columns=bad_ignore_columns,
                _ignore_float_columns=False,
                _ignore_non_binary_integer_columns=False,
                _threshold=2,
                _n_features_in=n_features_in
            )


    @pytest.mark.parametrize('good_ignore_columns', (None, 'ndarray'))
    def test_accepts_good_ignore_columns(
        self, good_og_dtypes, good_ignore_columns, n_features_in
    ):

        # only None or list-like-int
        if good_ignore_columns == 'ndarray':
            good_ignore_columns = np.array([0, n_features_in-2]).astype(np.int32)

        out = _conflict_warner(
            _original_dtypes=good_og_dtypes,
            _handle_as_bool=None,
            _ignore_columns=good_ignore_columns,
            _ignore_float_columns=False,
            _ignore_non_binary_integer_columns=False,
            _threshold=2,
            _n_features_in=n_features_in
        )

        assert out is None
    # END ignore_columns -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    # handle_as_bool -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('bad_handle_as_bool',
        (-2.7, -1, 0, 1, 2.7, True, 'junk', {'a':1}, lambda x: x)
    )
    def test_rejects_bad_handle_as_bool(
        self, good_og_dtypes, bad_handle_as_bool, n_features_in
    ):

        # only None or list-like-int

        with pytest.raises(Exception):
            _conflict_warner(
                _original_dtypes=good_og_dtypes,
                _handle_as_bool=bad_handle_as_bool,
                _ignore_columns=None,
                _ignore_float_columns=False,
                _ignore_non_binary_integer_columns=False,
                _threshold=2,
                _n_features_in=n_features_in
            )


    @pytest.mark.parametrize('good_handle_as_bool', (None, 'ndarray'))
    def test_accepts_good_handle_as_bool(
        self, good_og_dtypes, good_handle_as_bool, n_features_in
    ):

        # only None or list-like-int
        if good_handle_as_bool == 'ndarray':
            good_handle_as_bool = np.array([0, n_features_in-2]).astype(np.int32)

        out = _conflict_warner(
            _original_dtypes=good_og_dtypes,
            _handle_as_bool=good_handle_as_bool,
            _ignore_columns=None,
            _ignore_float_columns=False,
            _ignore_non_binary_integer_columns=False,
            _threshold=2,
            _n_features_in=n_features_in
        )

        assert out is None
    # END handle_as_bool -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    # threshold -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('bad_threshold',
        (-2.7, -1, 0, 1, 2.7, True, 'junk', {'a':1}, lambda x: x)
    )
    def test_rejects_bad_threshold(
        self, good_og_dtypes, bad_threshold, n_features_in
    ):

        # only int or list-like-int

        with pytest.raises(Exception):
            _conflict_warner(
                _original_dtypes=good_og_dtypes,
                _handle_as_bool=None,
                _ignore_columns=None,
                _ignore_float_columns=False,
                _ignore_non_binary_integer_columns=False,
                _threshold=bad_threshold,
                _n_features_in=n_features_in
            )


    @pytest.mark.parametrize('good_threshold', ('int', 'Iterable[int]'))
    def test_accepts_good_threshold(
        self, good_og_dtypes, good_threshold, n_features_in
    ):

        if good_threshold == 'int':
            _threshold = 2
        elif good_threshold == 'Iterable[int]':
            _threshold = [2 for _ in range(n_features_in)]

        out = _conflict_warner(
            _original_dtypes=good_og_dtypes,
            _handle_as_bool=None,
            _ignore_columns=None,
            _ignore_float_columns=False,
            _ignore_non_binary_integer_columns=False,
            _threshold=_threshold,
            _n_features_in=n_features_in
        )

        assert out is None
    # END threshold -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # END basic validation ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    @pytest.mark.parametrize('og_dtypes', ['short', 'good', 'long'])
    @pytest.mark.parametrize('ignore_columns', ['low', 'good', 'high'])
    @pytest.mark.parametrize('handle_as_bool', ['low', 'good', 'high'])
    @pytest.mark.parametrize('threshold', ['short', 'good', 'long'])
    def test_rejects_bad_param_lens_wrt_n_features_in(
        self, allowed, og_dtypes, ignore_columns, handle_as_bool, threshold,
        n_features_in
    ):

        len_dict = {
            'short': n_features_in // 2,
            'good': n_features_in,
            'long': 2 * n_features_in
        }

        idx_dict = {
            'low': [-n_features_in-1, 0, 1],
            'good': [0, 1, n_features_in-1],
            'high': [0, 1, n_features_in]
        }

        _og_dtypes = np.random.choice(allowed, len_dict[og_dtypes], replace=True)
        _ignore_columns = np.array(idx_dict[ignore_columns]).astype(np.int32)
        _handle_as_bool = np.array(idx_dict[handle_as_bool]).astype(np.int32)
        _threshold = [2 for _ in range(len_dict[threshold])]

        if og_dtypes=='good' and ignore_columns=='good' \
                and handle_as_bool=='good' and threshold=='good':

            _conflict_warner(
                _original_dtypes=_og_dtypes,
                _handle_as_bool=_handle_as_bool,
                _ignore_columns=_ignore_columns,
                _ignore_float_columns=False,
                _ignore_non_binary_integer_columns=False,
                _threshold=_threshold,
                _n_features_in=n_features_in
            )

        else:
            with pytest.raises(Exception):
                _conflict_warner(
                    _original_dtypes=_og_dtypes,
                    _handle_as_bool=_handle_as_bool,
                    _ignore_columns=_ignore_columns,
                    _ignore_float_columns=False,
                    _ignore_non_binary_integer_columns=False,
                    _threshold=_threshold,
                    _n_features_in=n_features_in
                )

    # END basic validation ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    @pytest.mark.parametrize('handle_as_bool', (None, 'empty'))
    def test_hab_none_and_empty(
        self, good_og_dtypes, handle_as_bool, n_features_in
    ):

        # None or empty short-circuits out, returns None

        # only None or ndarray, dtype = np.int32
        if handle_as_bool == 'empty':
            handle_as_bool = np.array([]).astype(np.int32)

        out = _conflict_warner(
            _original_dtypes=good_og_dtypes,
            _handle_as_bool=handle_as_bool,
            _ignore_columns=None,
            _ignore_float_columns=False,
            _ignore_non_binary_integer_columns=False,
            _threshold=2,
            _n_features_in=n_features_in
        )

        assert out is None




    options = ('none', 'empty', 'vector1', 'vector2', 'vector3')
    @pytest.mark.parametrize('_handle_as_bool', options)
    @pytest.mark.parametrize('_ignore_columns', options)
    @pytest.mark.parametrize('_ignore_float_columns', (True, False))
    @pytest.mark.parametrize('_ignore_non_binary_integer_columns', (True, False))
    @pytest.mark.parametrize('_threshold', ('int', 'Iterable[int]'))
    def test_accuracy(
        self, good_og_dtypes, n_features_in, _handle_as_bool, _ignore_columns,
        _ignore_float_columns, _ignore_non_binary_integer_columns, _threshold
    ):

        if n_features_in < 6:
            raise Exception(f"'n_features_in' must be >= 6 for this test")

        # if intersection between ignore_float_columns and ignore_columns, doesnt
        # matter, ignored either way

        # if intersection between ignore_non_binary_integer_columns and
        # ignore_columns, doesnt matter, ignored either way

        # so we need to address when
        # warn on handle_as_bool intersects ignore_columns
        # warn on handle_as_bool intersects ignore_float_columns
        # warn on handle_as_bool intersects ignore_non_binary_integer_columns
        # warn on handle_as_bool intersects threshold==1


        vector1 = [0, 1, 2],
        vector2 = [n_features_in-3, n_features_in-2, n_features_in-1]
        vector3 = [2, 3, 4]

        param_input_dict = {
            'none': None,
            'empty': np.array([], dtype=np.int32).ravel(),
            'vector1': np.array(vector1, dtype=np.int32).ravel(),
            'vector2': np.array(vector2, dtype=np.int32).ravel(),
            'vector3': np.array(vector3, dtype=np.int32).ravel()
        }

        hab = param_input_dict[_handle_as_bool]
        ic = param_input_dict[_ignore_columns]

        if _ignore_float_columns:
            _float_columns = np.arange(n_features_in)[(good_og_dtypes=='float')]
        else:
            _float_columns = np.array([])

        if _ignore_non_binary_integer_columns:
            _non_bin_int_columns = np.arange(n_features_in)[(good_og_dtypes=='int')]
        else:
            _non_bin_int_columns = np.array([])

        if _threshold == 'int':
            _threshold = 2
        else:
            while True:
                _threshold = \
                    [int(np.random.randint(1, 3)) for _ in range(n_features_in)]
                if any(np.array(_threshold)==1):
                    break

        if hab is not None:
            if ic is not None:
                hab_vs_ic = \
                    bool(len(set(list(hab)).intersection(list(ic))))
            else:
                hab_vs_ic = False
            hab_vs_ign_flt = \
                bool(len(set(list(hab)).intersection(list(_float_columns))))
            hab_vs_ign_non_bin_int = \
                bool(len(set(list(hab)).intersection(list(_non_bin_int_columns))))
            if isinstance(_threshold, int):
                hab_vs_threshold = False
            else:
                __ = np.array(_threshold)
                __ = np.arange(len(__))[(__==1)]
                hab_vs_threshold = bool(len(set(list(hab)).intersection(__)))
        else:
            hab_vs_ic = False
            hab_vs_ign_flt = False
            hab_vs_ign_non_bin_int = False
            hab_vs_threshold = False


        del _float_columns, _non_bin_int_columns

        # warn on handle_as_bool intersects ignore_columns
        # warn on handle_as_bool intersects ignore_float_columns
        # warn on handle_as_bool intersects ignore_non_binary_integer_columns
        # warn on handle_as_bool intersects threshold==1

        # handle_as_bool is None or empty, never warn.
        if hab_vs_ic or hab_vs_ign_flt or hab_vs_ign_non_bin_int \
                or hab_vs_threshold:
            with pytest.warns():
                out = _conflict_warner(
                    _original_dtypes=good_og_dtypes,
                    _handle_as_bool=hab,
                    _ignore_columns=ic,
                    _ignore_float_columns=_ignore_float_columns,
                    _ignore_non_binary_integer_columns= \
                        _ignore_non_binary_integer_columns,
                    _threshold=_threshold,
                    _n_features_in=n_features_in
                )
                assert out is None

        else:
            out = _conflict_warner(
                _original_dtypes=good_og_dtypes,
                _handle_as_bool=hab,
                _ignore_columns=ic,
                _ignore_float_columns=_ignore_float_columns,
                _ignore_non_binary_integer_columns= \
                    _ignore_non_binary_integer_columns,
                _threshold=_threshold,
                _n_features_in=n_features_in
            )
            assert out is None








