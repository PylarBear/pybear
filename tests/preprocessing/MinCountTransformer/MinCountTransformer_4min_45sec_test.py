# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer.MinCountTransformer import \
    MinCountTransformer

import uuid
from copy import deepcopy
import numpy as np
np.random.seed(0)
import pandas as pd
import scipy.sparse as ss
import polars as pl
from sklearn.preprocessing import OneHotEncoder

from pybear.utilities._nan_masking import nan_mask

import pytest




bypass = False


# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
# SET X, y DIMENSIONS AND DEFAULT THRESHOLD (_args) FOR TESTING MCT

@pytest.fixture(scope='session')
def _mct_rows():
    # _mct_rows must be between 50 and 750
    # this is fixed, all MCT test (not mmct test) objects have this many rows
    # (mmct rows is set by the construction parameters when a suitable set
    # of vectors for building mmct is found, remember)
    return 100


@pytest.fixture(scope='session')
def _mct_cols():
    # _mct_cols must be > 0
    # this sets the number of columns for each data type! not the total
    # number of columns in X! See the logic inside build_test_objects_for_MCT
    # to get how many columns are actually returned. That number is held
    # in fixture 'x_cols'.
    return 2


@pytest.fixture(scope='session')
def y_rows(x_rows):
    return x_rows


@pytest.fixture(scope='session')
def y_cols():
    return 2


@pytest.fixture(scope='session')
def _args(_mct_rows):
    return [_mct_rows // 20]


@pytest.fixture(scope='function')
def _kwargs():
    return {
        'ignore_float_columns': True,
        'ignore_non_binary_integer_columns': True,
        'ignore_columns': None,
        'ignore_nan': True,
        'delete_axis_0': True,
        'handle_as_bool': None,
        'reject_unseen_values': False,
        'max_recursions': 1,
        'n_jobs': 1   # leave this set a 1 because of confliction
    }

# END SET X, y DIMENSIONS AND DEFAULT THRESHOLD (_args) FOR TESTING MCT
# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^




# build X, NO_NAN_X, DTYPE_KEY, x_rows, x_cols for MCT test (not mmct test!)

@pytest.fixture(scope='session')
def build_test_objects_for_MCT(mmct, _mct_rows, _mct_cols, _args):

    # This constructs a test array "X" of randomly filled vectors that
    # have certain criteria like a certain number of certain types of
    # columns, certain amounts of uniques, certain proportions of uniques,
    # to make X manipulable with certain outcomes across all tests. The
    # vectors are filled randomly and may not always be generated with
    # the expected characteristics in one shot, so this iterates over and
    # over until vectors are created that pass certain tests done on them
    # by mmct.

    ctr = 0
    while True:  # LOOP UNTIL DATA IS SUFFICIENT TO BE USED FOR ALL THE TESTS

        ctr += 1
        _tries = 10
        if ctr >= _tries:
            raise Exception(
                f"\033[91mMinCountThreshold failed at {_tries} attempts "
                f"to generate an appropriate X for test\033[0m")

        # vvv CORE TEST DATA vvv *************************
        # CREATE _mct_cols COLUMNS OF BINARY INTEGERS
        _X = np.random.randint(
            0, 2, (_mct_rows, _mct_cols)
        ).astype(object)
        # CREATE _mct_cols COLUMNS OF NON-BINARY INTEGERS
        _X = np.hstack((
            _X, np.random.randint(
                0, _mct_rows // 15, (_mct_rows, _mct_cols)
            ).astype(object)
        ))
        # CREATE _mct_cols COLUMNS OF FLOATS
        _X = np.hstack((
            _X, np.random.uniform(
                0, 1, (_mct_rows, _mct_cols)
            ).astype(object)
        ))
        # CREATE _mct_cols COLUMNS OF STRS
        _alpha = 'abcdefghijklmnopqrstuvwxyz'
        _alpha = _alpha + _alpha.upper()
        for _ in range(_mct_cols):
            _X = np.hstack((_X,
                np.random.choice(
                    list(_alpha[:_mct_rows // 10]),
                    (_mct_rows,),
                    replace=True
                ).astype(object).reshape((-1, 1))
            ))
        # END ^^^ CORE TEST DATA ^^^ *************************

        # CREATE A COLUMN OF STRS THAT WILL ALWAYS BE DELETED BY FIRST RECURSION
        DUM_STR_COL = np.fromiter(('dum' for _ in range(_mct_rows)), dtype='<U3')
        DUM_STR_COL[0] = 'one'
        DUM_STR_COL[1] = 'two'
        DUM_STR_COL[2] = 'six'
        DUM_STR_COL[3] = 'ten'

        _X = np.hstack((_X, DUM_STR_COL.reshape((-1, 1)).astype(object)))
        del DUM_STR_COL

        # _X SHAPE SHOULD BE (x_rows, 4 * x_cols + 1)
        x_rows = _mct_rows
        x_cols = 4 * _mct_cols + 1

        _DTYPE_KEY = [
            k for k in ['bin_int', 'int', 'float', 'obj'] for j in range(_mct_cols)
        ]
        _DTYPE_KEY += ['obj']

        # KEEP THIS FOR TESTING IF DTYPES RETRIEVED CORRECTLY WITH np.nan MIXED IN
        _NO_NAN_X = _X.copy()

        # FLOAT/STR ONLY --- NO_NAN_X MUST BE REDUCED WHEN STR COLUMNS ARE
        # TRANSFORMED
        FLOAT_STR_X = _NO_NAN_X[:, 2 * _mct_cols:4 * _mct_cols].copy()
        # mmct() args = MOCK_X, MOCK_Y, ign_col, ign_nan, ignore_flt_col,
        # ignore_non_binary_int_col, handle_as_bool, delete_axis_0, ct_thresh
        _X1 = mmct().trfm(_X, None, None, True, True, True, None, True, *_args)
        if np.array_equiv(_X1, FLOAT_STR_X):
            del _X1
            continue
        del _X1

        del FLOAT_STR_X

        # PEPPER 10% OF CORE DATA WITH np.nan
        for _ in range(x_rows * x_cols // 10):
            row_coor = np.random.randint(0, x_rows)
            col_coor = np.random.randint(0, x_cols - 1)
            if col_coor < 3 * _mct_cols:
                _X[row_coor, col_coor] = np.nan
            elif col_coor >= 3 * _mct_cols:
                _X[row_coor, col_coor] = 'nan'
        del row_coor, col_coor

        # MAKE EVERY CORE COLUMN HAVE 2 VALUES THAT CT FAR EXCEEDS
        # count_threshold SO DOESNT ALLOW FULL DELETE
        _repl = x_rows // 3
        _get_idxs = lambda: np.random.choice(range(x_rows), _repl, replace=False)
        # 24_06_05_13_16_00 the assignments here cannot be considated using
        # lambda functions - X is being passed to mmct and it is saying cannot
        # pickle
        for idx in range(_mct_cols):
            _X[_get_idxs(), _mct_cols + idx] = \
                int(np.random.randint(0, x_rows // 20) + idx)
            _X[_get_idxs(), _mct_cols + idx] = \
                int(np.random.randint(0, x_rows // 20) + idx)
            _X[_get_idxs(), 2 * _mct_cols + idx] = \
                np.random.uniform(0, 1) + idx
            _X[_get_idxs(), 2 * _mct_cols + idx] = \
                np.random.uniform(0, 1) + idx
            _X[_get_idxs(), 3 * _mct_cols + idx] = _alpha[:x_rows // 15][idx]
            _X[_get_idxs(), 3 * _mct_cols + idx] = _alpha[:x_rows // 15][idx + 1]

        del idx, _repl, _alpha

        # VERIFY ONE RECURSION OF mmct DELETED THE SACRIFICIAL LAST COLUMN
        # (CORE COLUMNS ARE RIGGED TO NOT BE DELETED)
        # MOCK_X, MOCK_Y, ign_col, ign_nan, ignore_non_binary_int_col,
        # ignore_flt_col, handle_as_bool, delete_axis_0, ct_thresh
        _X1 = mmct().trfm(_X, None, None, False, False, False, None, True, *_args)
        assert not np.array_equiv(_X1[:, -1], _X[:, -1]), \
            "Mock MinCountTransformer did not delete last column"

        if len(_X1.shape) != 2 or 0 in _X1.shape:
            # IF ONE RECURSION DELETES EVERYTHING, BUILD NEW X
            continue
        elif np.array_equiv(_X1[:, :-1], _X[:, :-1]):
            # IF ONE RECURSION DOESNT DELETE ANY ROWS OF THE CORE COLUMNS,
            # BUILD NEW X
            continue

        # IF NUM OF RIGGED IDENTICAL NUMBERS IN ANY FLT COLUMN < THRESHOLD,
        # BUILD NEW X
        for flt_col_idx in range(2 * _mct_cols, 3 * _mct_cols, 1):
            _max_ct = np.unique(
                _X[:, flt_col_idx], return_counts=True
            )[1].max(axis=0)
            if _max_ct < _args[0]:
                continue
        del _max_ct

        # TRFM OF NON-BINARY INTEGER COLUMNS MUST NOT DELETE EVERYTHING,
        # BUT MUST DELETE SOMETHING
        try:
            _X1 = mmct().trfm(
                _X[:, _mct_cols:(2 * _mct_cols)].copy(),
                None,
                None,
                True,
                False,
                True,
                None,
                False,
                *_args
            )
            # MOCK_X, MOCK_Y, ign_col, ign_nan, ignore_non_binary_int_col,
            # ignore_flt_col, handle_as_bool, delete_axis_0, ct_thresh
            if np.array_equiv(_X1, _X[:, _mct_cols:(2 * _mct_cols)].copy()):
                continue
        except:
            continue

        try_again = False
        # IF ALL CTS OF EVERY STR UNIQUE IS >= THRESHOLD, BUILD NEW X
        for str_col_idx in range(x_cols - 1, x_cols - _mct_cols - 1, -1):
            _min_ct = min(np.unique(_X[:, str_col_idx], return_counts=True)[1])
            if _min_ct >= _args[0]:
                try_again = True
                break
        if try_again:
            continue
        del _min_ct

        # IF X CANNOT TAKE 2 RECURSIONS WITH THRESHOLD==3, BUILD NEW X
        try_again = False
        _X1 = mmct().trfm(_X, None, None, False, False, False, None, True, 3)
        # MOCK_X, MOCK_Y, ign_col, ign_nan, ignore_non_binary_int_col,
        # ignore_flt_col, handle_as_bool, delete_axis_0, ct_thresh
        try:
            # THIS SHOULD EXCEPT IF ALL ROWS/COLUMNS WOULD BE DELETED
            _X2 = mmct().trfm(_X1, None, None, False, False, False, None, True, 3)
            # SECOND RECURSION SHOULD ALSO DELETE SOMETHING, BUT NOT EVERYTHING
            if np.array_equiv(_X1, _X2):
                try_again = True
        except:
            try_again = True

        if try_again:
            continue

        del try_again, _X1, _X2

        # IF X PASSED ALL THESE PRE-CONDITION TESTS, IT IS GOOD TO USE FOR TEST
        break

    # IF X PASSED ALL THESE PRE-CONDITION TESTS, IT IS GOOD TO USE FOR TEST

    return _X, _NO_NAN_X, _DTYPE_KEY, x_rows, x_cols


@pytest.fixture(scope='session')
def X(build_test_objects_for_MCT):
    return build_test_objects_for_MCT[0]


@pytest.fixture(scope='session')
def NO_NAN_X(build_test_objects_for_MCT):
    return build_test_objects_for_MCT[1]


@pytest.fixture(scope='session')
def DTYPE_KEY(build_test_objects_for_MCT):
    return build_test_objects_for_MCT[2]


@pytest.fixture(scope='session')
def x_rows(build_test_objects_for_MCT):
    return build_test_objects_for_MCT[3]


@pytest.fixture(scope='session')
def x_cols(build_test_objects_for_MCT):
    return build_test_objects_for_MCT[4]


@pytest.fixture(scope='session')
def COLUMNS(x_cols):
    return [str(uuid.uuid4())[:4] for _ in range(x_cols)]

# END build X, NO_NAN_X, DTYPE_KEY, x_rows, x_cols for MCT test (not mmct test!)


# Build y for MCT tests (not mmct test!) ** * ** * ** * ** * ** * ** * **

@pytest.fixture(scope='session')
def y(y_rows, y_cols):
    return np.random.randint(0, 2, (y_rows, y_cols), dtype=np.uint8)

# Build y for MCT tests (not mmct test!) ** * ** * ** * ** * ** * ** * **

# END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# TEST FOR EXCEPTS ON NON-BOOL _ignore_float_columns,
# _ignore_non_binary_integer_columns, _ignore_nan, _delete_axis_0,
# reject_unseen_values ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestBoolKwargsAcceptBoolRejectNonBool:

    @pytest.mark.parametrize('ign_float', [True, False])
    @pytest.mark.parametrize('ign_bin_int', [True, False])
    @pytest.mark.parametrize('del_ax_0', [True, False])
    @pytest.mark.parametrize('ign_nan', [True, False])
    @pytest.mark.parametrize('rej_unseen', [True, False])
    def test_bool_kwargs_accept_bool(self, X, y, _args, _kwargs, ign_float,
        ign_bin_int, del_ax_0, ign_nan, rej_unseen
    ):

        _kwargs['ignore_float_columns'] = ign_float
        _kwargs['ignore_non_binary_integer_columns'] = ign_bin_int
        _kwargs['delete_axis_0'] = del_ax_0
        _kwargs['ignore_nan'] = ign_nan
        _kwargs['reject_unseen_values'] = rej_unseen

        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit_transform(X, y)


    JUNK = [None, np.pi, 0, 1, min, (1,2), [1,2], {1,2}, {'a':1}, lambda x: x]

    @pytest.mark.parametrize('ign_float', JUNK)
    def test_ign_float_reject_non_bool(self, X, y, _args, _kwargs, ign_float):

        _kwargs['ignore_float_columns'] = ign_float

        TestCls = MinCountTransformer(*_args, **_kwargs)

        # if any are non-bool, should fail
        with pytest.raises(TypeError):
            TestCls.fit_transform(X, y)


    @pytest.mark.parametrize('ign_bin_int', JUNK)
    def test_ign_bin_int_reject_non_bool(self, X, y, _args, _kwargs, ign_bin_int):

        _kwargs['ignore_non_binary_integer_columns'] = ign_bin_int

        TestCls = MinCountTransformer(*_args, **_kwargs)

        # if any are non-bool, should fail
        with pytest.raises(TypeError):
            TestCls.fit_transform(X, y)


    @pytest.mark.parametrize('del_ax_0', JUNK)
    def test_del_ax_0_reject_non_bool(self, X, y, _args, _kwargs, del_ax_0):

        _kwargs['delete_axis_0'] = del_ax_0

        TestCls = MinCountTransformer(*_args, **_kwargs)

        # if any are non-bool, should fail
        with pytest.raises(TypeError):
            TestCls.fit_transform(X, y)


    @pytest.mark.parametrize('ign_nan', JUNK)
    def test_ign_nan_reject_non_bool(self, X, y, _args, _kwargs, ign_nan):

        _kwargs['ignore_nan'] = ign_nan

        TestCls = MinCountTransformer(*_args, **_kwargs)

        # if any are non-bool, should fail
        with pytest.raises(TypeError):
            TestCls.fit_transform(X, y)


    @pytest.mark.parametrize('rej_unseen', JUNK)
    def test_rej_unseen_reject_non_bool(self, X, y, _args, _kwargs, rej_unseen):

        _kwargs['reject_unseen_values'] = rej_unseen

        TestCls = MinCountTransformer(*_args, **_kwargs)

        # if any are non-bool, should fail
        with pytest.raises(TypeError):
            TestCls.fit_transform(X, y)

# END TEST FOR EXCEPTS ON NON-BOOL _ignore_float_columns,
# _ignore_non_binary_integer_columns, _ignore_nan, _delete_axis_0,
# reject_unseen_values ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


# TEST FOR TypeError ON JUNK count_threshold, max_recursions, set_output, n_jobs
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestJunkCountThresholdMaxRecursionsSetOutputNjobs:

    BASELINE_JUNK = [True, False, [1,2], (1,2), {1,2}, {'a': 1},
                    lambda x: x, min, None, np.pi, np.nan, float('inf')
    ]

    # JUNK_CT_THRESH = BASELINE_JUNK.copy() + ['junk']
    JUNK_RECURSIONS = BASELINE_JUNK.copy() + ['junk']
    JUNK_OUTPUT_TYPE = BASELINE_JUNK.copy()
    JUNK_OUTPUT_TYPE.remove(None)
    JUNK_N_JOBS = BASELINE_JUNK.copy()
    JUNK_N_JOBS.remove(None)
    JUNK_N_JOBS += ['junk']


    @pytest.mark.parametrize('junk_ct_thresh', (None, lambda x: x))
    def test_junk_ct_thresh(self, X, y, _kwargs, junk_ct_thresh):

        TestCls = MinCountTransformer(junk_ct_thresh, **_kwargs)

        with pytest.raises(TypeError):
            TestCls.fit_transform(X, y)


    @pytest.mark.parametrize('junk_recursions', JUNK_RECURSIONS)
    def test_junk_recursions(self, X, y, _args, _kwargs, junk_recursions):

        _kwargs['max_recursions'] = junk_recursions

        TestCls = MinCountTransformer(*_args, **_kwargs)

        with pytest.raises(TypeError):
            TestCls.fit_transform(X, y)


    @pytest.mark.parametrize('junk_n_jobs', JUNK_N_JOBS)
    def test_junk_n_jobs(self, X, y, _args, _kwargs, junk_n_jobs):

        _kwargs['n_jobs'] = junk_n_jobs

        TestCls = MinCountTransformer(*_args, **_kwargs)

        with pytest.raises(TypeError):
            TestCls.fit_transform(X, y)


    @pytest.mark.parametrize('junk_output_type', JUNK_OUTPUT_TYPE)
    def test_junk_output_type(self, X, y, _args, _kwargs, junk_output_type):

        _kwargs['n_jobs'] = junk_output_type

        TestCls = MinCountTransformer(*_args, **_kwargs)

        with pytest.raises(TypeError):
            TestCls.set_output(transform=junk_output_type)
            TestCls.fit_transform(X, y)

# END TEST FOR TypeError ON JUNK count_threshold, max_recursions, set_output, n_jobs


# TEST FOR ValueError ON BAD count_threshold, max_recursions, set_output, n_jobs
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestBadCountThresholdMaxRecursionsSetOutputNjobs:

    BAD_CT_THRESH = [-2, 1, 100_000_000, 'bad_list_1', 'bad_list_2']
    BAD_RECURSIONS = [-1, 0]
    BAD_OUTPUT_TYPE = ['dask_array', 'wrong_junk']
    BAD_N_JOBS = [-2, 0]


    @pytest.mark.parametrize('bad_ct_thresh', BAD_CT_THRESH)
    def test_bad_ct_thresh(self, X, y, _kwargs, bad_ct_thresh):

        if bad_ct_thresh == 'bad_list_1':
            bad_ct_thresh = [1 for _ in range(X.shape[1])]
        elif bad_ct_thresh == 'bad_list_2':
            bad_ct_thresh = range(X.shape[1])  # has a zero in it

        TestCls = MinCountTransformer(bad_ct_thresh, **_kwargs)

        with pytest.raises(ValueError):
            TestCls.fit_transform(X, y)


    @pytest.mark.parametrize('bad_recursions', BAD_RECURSIONS)
    def test_bad_recursions(self, X, y, _args, _kwargs, bad_recursions):

        _kwargs['max_recursions'] = bad_recursions

        TestCls = MinCountTransformer(*_args, **_kwargs)

        with pytest.raises(ValueError):
            TestCls.fit_transform(X, y)


    @pytest.mark.parametrize('bad_n_jobs', BAD_N_JOBS)
    def test_bad_n_jobs(self, X, y, _args, _kwargs, bad_n_jobs):

        _kwargs['n_jobs'] = bad_n_jobs

        TestCls = MinCountTransformer(*_args, **_kwargs)

        with pytest.raises(ValueError):
            TestCls.fit_transform(X, y)


    @pytest.mark.parametrize('bad_output_type', BAD_OUTPUT_TYPE)
    def test_bad_output_type(self, X, y, _args, _kwargs, bad_output_type):

        TestCls = MinCountTransformer(*_args, **_kwargs)

        with pytest.raises(ValueError):
            TestCls.set_output(transform=bad_output_type)
            TestCls.fit_transform(X, y)


# END TEST FOR ValueError ON BAD count_threshold, max_recursions, set_output, n_jobs


# TEST FOR ACCEPTS GOOD count_threshold, max_recursions, set_output, n_jobs
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestGoodCountThresholdMaxRecursionsSetOutputNjobs:


    @pytest.mark.parametrize('good_ct_thresh', [3, 5])
    def test_good_ct_thresh(self, X, y, _kwargs, good_ct_thresh):

        MinCountTransformer(good_ct_thresh, **_kwargs).fit_transform(X, y)


    @pytest.mark.parametrize('good_recursions', [1, 10])
    def test_good_recursions(self, X, y, _args, _kwargs, good_recursions):

        _kwargs['max_recursions'] = good_recursions

        MinCountTransformer(*_args, **_kwargs).fit_transform(X, y)


    @pytest.mark.parametrize('good_n_jobs', [-1, 1, 10, None])
    def test_good_n_jobs(self, X, y, _args, _kwargs, good_n_jobs):

        _kwargs['n_jobs'] = good_n_jobs

        MinCountTransformer(*_args, **_kwargs).fit_transform(X, y)


    @pytest.mark.parametrize('good_output_type',
        [None, 'default', 'pandas', 'polars']
    )
    def test_good_output_type(self, X, y, _args, _kwargs, good_output_type):

        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.set_output(transform=good_output_type)
        TestCls.fit_transform(X, y[:, 0])


# END TEST ACCEPTS GOOD count_threshold, max_recursions, set_output, n_jobs


# TEST FOR GOOD / BAD ignore_columns / handle_as_bool ##################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestIgnoreColumnsHandleAsBool:

    # both can take list of int, list of str, callable, or None

    @staticmethod
    @pytest.fixture
    def _y(y):
        return y[:, 0].copy()


    @pytest.mark.parametrize('input_format', ('numpy', 'pd_df'))
    @pytest.mark.parametrize('kwarg_input', (
        0, 1, 3.14, True, False, 'junk', np.nan, {0: 'a', 1: 'b'},
        [True, False, None, 'junk'], 'get_from_COLUMNS', [np.nan],
        'bad_callable'
        )
    )
    @pytest.mark.parametrize('_kwarg',
        ('ignore_columns', 'handle_as_bool', 'both')
    )
    def test_junk_ign_cols_handle_as_bool(self, X, COLUMNS, _y, _args, _kwargs,
         input_format, kwarg_input, _kwarg, _mct_cols):

        if _kwarg == 'ignore_columns':
            if kwarg_input == 'get_from_COLUMNS':
                _kwargs['ignore_columns'] = [1, 3, COLUMNS[6]]
            elif kwarg_input == 'bad_callable':
                _kwargs['ignore_columns'] = lambda X: 'unrecognizable junk'
            else:
                _kwargs['ignore_columns'] = kwarg_input
            _kwargs['handle_as_bool'] = None
        elif _kwarg == 'handle_as_bool':
            if kwarg_input == 'bad_callable':
                _kwargs['handle_as_bool'] = lambda X: 'unrecognizable junk'
            elif kwarg_input == 'get_from_COLUMNS':
                _kwargs['handle_as_bool'] = [1, 3, COLUMNS[6]]
            else:
                _kwargs['handle_as_bool'] = None
            _kwargs['ignore_columns'] = kwarg_input
        elif _kwarg == 'both':
            if kwarg_input == 'get_from_COLUMNS':
                _kwargs['ignore_columns'] = [1, 3, COLUMNS[6]]
                _kwargs['handle_as_bool'] = [1, 3, COLUMNS[6]]
            else:
                _kwargs['ignore_columns'] = kwarg_input
                _kwargs['handle_as_bool'] = kwarg_input

        TestCls = MinCountTransformer(*_args, **_kwargs)

        if input_format == 'numpy':
            X_NEW, y_NEW = X.copy(), _y
        elif input_format == 'pd_df':
            X_NEW = pd.DataFrame(data=X, columns=COLUMNS, dtype=object)
            y_NEW = pd.DataFrame(data=_y, columns=['y'], dtype=object)
        else:
            raise Exception

        with pytest.raises(TypeError):
            TestCls.fit_transform(X_NEW, y_NEW)


    @pytest.mark.parametrize('input_format', ('numpy', 'pd_df', 'pd_series'))
    @pytest.mark.parametrize('kwarg_input', ([1000, 1001, 1002], ))
    @pytest.mark.parametrize('_kwarg', ('ignore_columns', 'handle_as_bool', 'both'))
    def test_bad_ign_cols_handle_as_bool(self, X, COLUMNS, _y, _args, _kwargs,
         input_format, kwarg_input, _kwarg, _mct_cols):

        if _kwarg == 'ignore_columns':
            _kwargs['ignore_columns'] = kwarg_input
            _kwargs['handle_as_bool'] = None
        elif _kwarg == 'handle_as_bool':
            _kwargs['handle_as_bool'] = kwarg_input
            _kwargs['ignore_columns'] = None
        elif _kwarg == 'both':
            _kwargs['ignore_columns'] = kwarg_input
            _kwargs['handle_as_bool'] = kwarg_input

        TestCls = MinCountTransformer(*_args, **_kwargs)

        if input_format == 'numpy':
            X_NEW, y_NEW = X.copy(), _y
        elif input_format == 'pd_df':
            X_NEW = pd.DataFrame(data=X, columns=COLUMNS, dtype=object)
            y_NEW = pd.DataFrame(data=_y, columns=['y'], dtype=object)
        elif input_format == 'pd_series':
            X_NEW = pd.Series(
                data=X[:, _mct_cols], name=COLUMNS[0], dtype=np.float64
            )
            y_NEW = pd.Series(data=_y, name='y', dtype=object)

        with pytest.raises(ValueError):
            TestCls.fit_transform(X_NEW, y_NEW)


    @pytest.mark.parametrize('input_format', ('numpy', 'pd_df'))
    @pytest.mark.parametrize('kwarg_input', (
        [],
        [0],
        'make_from_cols',
        'get_from_COLUMNS',
        'good_callable',
        )
    )
    @pytest.mark.parametrize('_kwarg',
        ('ignore_columns', 'handle_as_bool', 'both')
)
    def test_accepts_good_ign_cols_handle_as_bool(self, X, COLUMNS, _y, _args,
        _kwargs, input_format, kwarg_input, _kwarg, _mct_cols, x_cols):

        if kwarg_input == 'get_from_COLUMNS' and input_format == 'numpy':
            pytest.skip(
                reason=f"cannot use column names when header is not given"
            )

        if kwarg_input == 'make_from_cols':
            kwarg_input = list(range(_mct_cols, 2*_mct_cols))

        if input_format == 'numpy':
            X_NEW, y_NEW = X.copy(), _y
        elif input_format == 'pd_df':
            X_NEW = pd.DataFrame(data=X, columns=COLUMNS, dtype=object)
            y_NEW = pd.DataFrame(data=_y, columns=['y'], dtype=object)
        else:
            raise Exception

        _is_series = len(X_NEW.shape) == 1

        if _kwarg == 'ignore_columns':
            if kwarg_input == 'get_from_COLUMNS':
                _kwargs['ignore_columns'] = \
                    COLUMNS[:1] if _is_series else [COLUMNS[_] for _ in [2,4,6]]
            elif kwarg_input == 'good_callable':
                _kwargs['ignore_columns'] = \
                    lambda X: [0] if _is_series else list(range(3*_mct_cols, x_cols))
            else:
                _kwargs['ignore_columns'] = kwarg_input
            _kwargs['handle_as_bool'] = None
        elif _kwarg == 'handle_as_bool':
            if kwarg_input == 'get_from_COLUMNS':
                NON_BIN_INT_COLS = \
                    [COLUMNS[_] for _ in range(_mct_cols, 2*_mct_cols)]
                _kwargs['handle_as_bool'] = \
                    COLUMNS[:1] if _is_series else NON_BIN_INT_COLS
                del NON_BIN_INT_COLS
            elif kwarg_input == 'good_callable':
                _kwargs['handle_as_bool'] = \
                    lambda X: [0] if _is_series else list(range(_mct_cols, 2*_mct_cols))
            else:
                _kwargs['handle_as_bool'] = kwarg_input
            _kwargs['ignore_columns'] = None
        elif _kwarg == 'both':
            if kwarg_input == 'get_from_COLUMNS':
                _kwargs['ignore_columns'] = \
                    COLUMNS[:1] if _is_series else [COLUMNS[_] for _ in [2,4,6]]
                NON_BIN_INT_COLS = \
                    [COLUMNS[_] for _ in range(_mct_cols, 2 * _mct_cols)]
                _kwargs['handle_as_bool'] = \
                    COLUMNS[:1] if _is_series else NON_BIN_INT_COLS
                del NON_BIN_INT_COLS
            elif kwarg_input == 'good_callable':
                _kwargs['ignore_columns'] = \
                    lambda X: [0] if _is_series else list(range(3 * _mct_cols, x_cols))
                _kwargs['handle_as_bool'] = \
                    lambda X: [0] if _is_series else list(range(_mct_cols, 2*_mct_cols))
            else:
                _kwargs['ignore_columns'] = kwarg_input
                _kwargs['handle_as_bool'] = kwarg_input


        TestCls = MinCountTransformer(*_args, **_kwargs)

        TestCls.fit_transform(X_NEW, y_NEW)

        del X_NEW, y_NEW, _is_series

# END TEST GOOD / BAD ignore_columns / handle_as_bool ##################


# TEST CORRECT DTYPES ARE RETRIEVED W/ OR W/O np.nan MIXED IN ##########
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAssignedDtypesWithAndWithoutNansMixedIn:

    def test_no_nan(self, _args, _kwargs, DTYPE_KEY, NO_NAN_X, y):
        # PASS NON-np.nan DATA AND COMPARE TO DTYPE_KEY
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit(NO_NAN_X, y)
        assert np.array_equiv(TestCls.original_dtypes_, DTYPE_KEY)

        del TestCls

    def test_with_nan(self, _args, _kwargs, DTYPE_KEY, X, y):
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit(X, y)
        assert np.array_equiv(TestCls.original_dtypes_, DTYPE_KEY)

        del TestCls

# END TEST CORRECT DTYPES ARE RETRIEVED W/ OR W/O np.nan MIXED IN ######


# ALWAYS ACCEPTS y==None IS PASSED TO fit(), partial_fit(), AND transform() ###
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestFitPartialFitTransformAcceptYEqualsNone:

    def test_fit(self, _args, _kwargs, X):
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit(X, None)

    def test_partial_fit_after_partial_fit(self, _args, _kwargs, X, y):
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.partial_fit(X, y)
        TestCls.partial_fit(X, None)

    def test_partial_fit_after_fit(self, _args, _kwargs, X, y):
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit(X, y)
        TestCls.partial_fit(X, None)

    def test_transform_after_fit(self, _args, _kwargs, X, y):
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit(X, y)
        TestCls.transform(X, None)

    def test_fit_transform(self, _args, _kwargs, X):
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit_transform(X, None)

# END ALWAYS ACCEPTS y==None IS PASSED TO fit(), partial_fit(), AND transform()


# TEST FOR EXCEPTS ON BAD y ROWS FOR DF, SERIES, ARRAY #################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestExceptsOnBadYRows:

    @staticmethod
    @pytest.fixture
    def row_checker():
        def foo(y_obj, X_rows):
            if y_obj is None:
                return False
            elif y_obj.shape[0] != X_rows:
                return True
            else:
                return False

        return foo


    @staticmethod
    @pytest.fixture
    def y_builder(y_rows:int, y_cols:int):

        def foo(
            y: np.ndarray,
            new_dtype: str,
            diff_rows: str
        ) -> np.ndarray:

            if new_dtype is None:
                NEW_Y = None
            else:
                NEW_Y = y.copy()

                if diff_rows == 'good':
                    pass
                elif diff_rows == 'less_row':
                    if len(NEW_Y.shape) == 1:
                        NEW_Y = NEW_Y[:y_rows // 2]
                    elif len(NEW_Y.shape) == 2:
                        NEW_Y = NEW_Y[:y_rows // 2, :]
                    else:
                        raise Exception(f"y_builder() NEW_Y.shape logic failed")
                    assert NEW_Y.shape[0] == y_rows // 2
                elif diff_rows == 'more_row':
                    if len(NEW_Y.shape) == 1:
                        NEW_Y = np.hstack((NEW_Y, NEW_Y))
                    elif len(NEW_Y.shape) == 2:
                        NEW_Y = np.vstack((NEW_Y, NEW_Y))
                    else:
                        raise Exception(f"y_builder() NEW_Y.shape logic failed")
                    assert NEW_Y.shape[0] == 2 * y_rows

                if 'pandas' in new_dtype:
                    NEW_Y = pd.DataFrame(data=NEW_Y, columns=None, dtype=object)
                if new_dtype == 'pandas_series':
                    NEW_Y = NEW_Y.squeeze()

            return NEW_Y

        return foo


    FIRST_FIT_y_DTYPE = ['numpy', 'pandas_dataframe', 'pandas_series']
    SECOND_FIT_y_DTYPE = ['numpy', 'pandas_dataframe', 'pandas_series']
    SECOND_FIT_y_DIFF_ROWS = ['good', 'less_row', 'more_row']
    TRANSFORM_y_DTYPE = ['numpy', 'pandas_dataframe', 'pandas_series']
    TRANSFORM_y_DIFF_ROWS = ['good', 'less_row', 'more_row']


    @pytest.mark.parametrize('fst_fit_y_dtype', FIRST_FIT_y_DTYPE)
    @pytest.mark.parametrize('scd_fit_y_dtype', SECOND_FIT_y_DTYPE)
    @pytest.mark.parametrize('scd_fit_y_rows', SECOND_FIT_y_DIFF_ROWS)
    @pytest.mark.parametrize('trfm_y_dtype', TRANSFORM_y_DTYPE)
    @pytest.mark.parametrize('trfm_y_rows', TRANSFORM_y_DIFF_ROWS)
    def test_except_on_bad_y_rows(self,
        X, y, _args, _kwargs, row_checker, y_builder, fst_fit_y_dtype,
        scd_fit_y_dtype, scd_fit_y_rows, trfm_y_dtype, trfm_y_rows, x_rows
    ):
        # only in transform. y is ignored in all of the 'fit's

        TestCls = MinCountTransformer(*_args, **_kwargs)

        # first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        fst_fit_Y = y_builder(
            y.copy(),
            new_dtype=fst_fit_y_dtype,
            diff_rows='good'
        )
        # end first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        scd_fit_Y = y_builder(
            y.copy(),
            new_dtype=scd_fit_y_dtype,
            diff_rows=scd_fit_y_rows
        )
        # end second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        trfm_Y = y_builder(
            y.copy(),
            new_dtype=trfm_y_dtype,
            diff_rows=trfm_y_rows
        )
        # end transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        value_error = 0
        # True only if y rows != X rows ** ** ** ** ** ** ** ** ** ** **
        value_error += row_checker(trfm_Y, x_rows)
        # END True only if y rows != X rows ** ** ** ** ** ** ** ** ** *

        TestCls.partial_fit(X, fst_fit_Y)
        TestCls.partial_fit(X, scd_fit_Y)

        if value_error:
            with pytest.raises(ValueError):
                TestCls.transform(X, trfm_Y)

        elif not value_error:
            TestCls.transform(X, trfm_Y)

    del FIRST_FIT_y_DTYPE, SECOND_FIT_y_DTYPE, SECOND_FIT_y_DIFF_ROWS
    del TRANSFORM_y_DTYPE, TRANSFORM_y_DIFF_ROWS

# END TEST FOR EXCEPTS ON BAD y ROWS FOR DF, SERIES, ARRAY #############


# TEST EXCEPTS ANYTIME X==None IS PASSED TO fit(), partial_fit(), AND transform()
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestExceptsAnytimeXisNone:

    def test_excepts_anytime_x_is_none(self, X, _args, _kwargs):

        with pytest.raises(ValueError):
            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.fit(None)


        with pytest.raises(ValueError):
            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.partial_fit(X)
            TestCls.partial_fit(None)

        with pytest.raises(ValueError):
            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.fit(X)
            TestCls.partial_fit(None)

        with pytest.raises(ValueError):
            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.fit(X)
            TestCls.transform(None)

        with pytest.raises(ValueError):
            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.fit_transform(None)

        del TestCls

# END TEST EXCEPTS ANYTIME X==None IS PASSED TO fit(), partial_fit(), OR transform()


# VERIFY ACCEPTS X AS SINGLE COLUMN ##################################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAcceptsXAsSingleColumn:


    def test_X_as_single_column(self, X, y, _args, _kwargs, COLUMNS):

        TestCls = MinCountTransformer(*_args, **_kwargs)

        _y = y[:, 0].copy()

        NEW_X = X[:, 0].copy().reshape((-1, 1))

        # numpy
        TestCls.fit_transform(NEW_X.copy(), _y)

        # pandas w header
        TestCls.fit_transform(
            pd.DataFrame(data=NEW_X.copy(), columns=COLUMNS[:1]),
            _y
        )

        # pandas w/o header
        TestCls.fit_transform(pd.DataFrame(data=NEW_X.copy()), _y)

# END VERIFY ACCEPTS X AS SINGLE COLUMN ###################################


# VERIFY REJECTS X AS 1D COLUMN / SERIES ##################################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestRejectsXAs1DColumnOrSeries:


    def test_X_as_1D_column(self, X, y, _args, _kwargs, COLUMNS):

        TestCls = MinCountTransformer(*_args, **_kwargs)

        _y = y[:, 0].copy()

        NEW_X = X[:, 0].copy()   # not reshaped to 2D

        # numpy
        with pytest.raises(ValueError):
            TestCls.fit_transform(NEW_X.copy(), _y)

        # pandas series
        with pytest.raises(ValueError):
            TestCls.fit_transform(pd.Series(data=NEW_X.copy().ravel()), _y)

# END VERIFY REJECTS X AS 1D COLUMN / SERIES ##############################


# TEST FOR EXCEPTS ON BAD X SHAPES ON SERIES & ARRAY ##########################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestExceptsOnBadXShapes:

    FIRST_FIT_X_DTYPE = list(reversed(['numpy', 'pandas']))
    SECOND_FIT_X_DTYPE = list(reversed(['numpy', 'pandas']))
    SECOND_FIT_X_SAME_DIFF_COLUMNS = \
        list(reversed(['good', 'less_col', 'more_col']))
    SECOND_FIT_X_SAME_DIFF_ROWS = list(reversed(['good', 'less_row', 'more_row']))
    TRANSFORM_X_DTYPE = list(reversed(['numpy', 'pandas']))
    TRANSFORM_X_SAME_DIFF_COLUMNS = \
        list(reversed(['good', 'less_col', 'more_col']))
    TRANSFORM_X_SAME_DIFF_ROWS = list(reversed(['good', 'less_row', 'more_row']))

    @staticmethod
    @pytest.fixture
    def X_builder(x_rows: int, x_cols):

        def foo(
            X: np.ndarray,
            new_dtype: str,
            diff_cols: str,
            diff_rows: str
        ):

            if new_dtype is None:
                NEW_X = None
            else:
                if diff_cols == 'good':
                    NEW_X = X.copy()
                elif diff_cols == 'less_col':
                    NEW_X = X.copy()[:, :x_cols // 2]
                elif diff_cols == 'more_col':
                    NEW_X = np.hstack((X.copy(), X.copy()))

                if diff_rows == 'good':
                    pass
                elif diff_rows == 'less_row':
                    if len(NEW_X.shape) == 1:
                        NEW_X = NEW_X[:x_rows // 2]
                    elif len(NEW_X.shape) == 2:
                        NEW_X = NEW_X[:x_rows // 2, :]
                elif diff_rows == 'more_row':
                    if len(NEW_X.shape) == 1:
                        NEW_X = np.hstack((NEW_X, NEW_X))
                    elif len(NEW_X.shape) == 2:
                        NEW_X = np.vstack((NEW_X, NEW_X))

                if 'pandas' in new_dtype:
                    NEW_X = pd.DataFrame(data=NEW_X, columns=None, dtype=object)

            return NEW_X

        return foo


    # end X_builder ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # pizza, maybe this can be merged with DF test below, and add ss
    @pytest.mark.parametrize('fst_fit_x_dtype', FIRST_FIT_X_DTYPE)
    @pytest.mark.parametrize('scd_fit_x_dtype', SECOND_FIT_X_DTYPE)
    @pytest.mark.parametrize('scd_fit_x_cols', SECOND_FIT_X_SAME_DIFF_COLUMNS)
    @pytest.mark.parametrize('scd_fit_x_rows', SECOND_FIT_X_SAME_DIFF_ROWS)
    @pytest.mark.parametrize('trfm_x_dtype', TRANSFORM_X_DTYPE)
    @pytest.mark.parametrize('trfm_x_cols', TRANSFORM_X_SAME_DIFF_COLUMNS)
    @pytest.mark.parametrize('trfm_x_rows', TRANSFORM_X_SAME_DIFF_ROWS)
    def test_excepts_on_bad_x_shapes(self, X, y, _args, _kwargs, X_builder,
        fst_fit_x_dtype, scd_fit_x_dtype, scd_fit_x_cols, scd_fit_x_rows,
        trfm_x_dtype, trfm_x_cols, trfm_x_rows, x_rows, x_cols
    ):

        TestCls = MinCountTransformer(*_args, **_kwargs)

        # first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        fst_fit_X = X_builder(
            X.copy(),
            new_dtype=fst_fit_x_dtype,
            diff_cols='good',
            diff_rows='good'
        )
        # end first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        scd_fit_X = X_builder(
            X.copy(),
            new_dtype=scd_fit_x_dtype,
            diff_cols=scd_fit_x_cols,
            diff_rows=scd_fit_x_rows
        )
        # end second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

        # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        trfm_X = X_builder(
            X.copy(),
            new_dtype=trfm_x_dtype,
            diff_cols=trfm_x_cols,
            diff_rows=trfm_x_rows
        )
        # end transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        fst_fit_y = y.copy()[:, 0].ravel()
        scd_fit_y = fst_fit_y.copy()
        trfm_y = fst_fit_y.copy()

        value_error = 0

        # ValueError WHEN ROWS OF y != X ROWS ONLY UNDER transform
        if trfm_X is not None and trfm_y.shape[0] != trfm_X.shape[0]:
            value_error += 1

        try:
            fst_fit_X.shape[1]
            fst_fit_X_cols = fst_fit_X.shape[1]
        except:
            fst_fit_X_cols = 1

        try:
            scd_fit_X.shape[1]
            scd_fit_X_cols = scd_fit_X.shape[1]
        except:
            scd_fit_X_cols = 1

        if scd_fit_X_cols != fst_fit_X_cols:
            value_error += 1

        try:
            trfm_X.shape[1]
            trfm_X_cols = trfm_X.shape[1]
        except:
            trfm_X_cols = 1

        if trfm_X_cols != fst_fit_X_cols:
            value_error += 1


        if value_error:
            with pytest.raises(ValueError):
                TestCls.partial_fit(fst_fit_X, fst_fit_y)
                TestCls.partial_fit(scd_fit_X, scd_fit_y)
                TestCls.transform(trfm_X, trfm_y)
        elif not value_error:
            TestCls.partial_fit(fst_fit_X, fst_fit_y)
            TestCls.partial_fit(scd_fit_X, scd_fit_y)
            TestCls.transform(trfm_X, trfm_y)


    del FIRST_FIT_X_DTYPE, SECOND_FIT_X_DTYPE, SECOND_FIT_X_SAME_DIFF_COLUMNS,
    del SECOND_FIT_X_SAME_DIFF_ROWS, TRANSFORM_X_DTYPE
    del TRANSFORM_X_SAME_DIFF_ROWS, TRANSFORM_X_SAME_DIFF_COLUMNS

# END TEST FOR EXCEPTS ON BAD X SHAPES, ON SERIES & ARRAY ##############


# TEST ValueError WHEN SEES A DF HEADER  DIFFERENT FROM FIRST-SEEN HEADER

@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestValueErrorDifferentHeader:

    NAMES = ['GOOD_DF', 'BAD_DF', 'NO_HDR_DF']

    @pytest.mark.parametrize('fst_fit_name', NAMES)
    @pytest.mark.parametrize('scd_fit_name', NAMES)
    @pytest.mark.parametrize('trfm_name', NAMES)
    def test_value_error_different_header(self, X, _args, _kwargs, COLUMNS,
        fst_fit_name, scd_fit_name, trfm_name):

        GOOD_DF = pd.DataFrame(data=X, columns=np.char.lower(COLUMNS))
        BAD_DF = pd.DataFrame(data=X, columns=np.char.upper(COLUMNS))
        NO_HDR_DF = pd.DataFrame(data=X, columns=None)

        if fst_fit_name == 'GOOD_DF':
            fst_fit_X = GOOD_DF.copy()
        elif fst_fit_name == 'BAD_DF':
            fst_fit_X = BAD_DF.copy()
        elif fst_fit_name == 'NO_HDR_DF':
            fst_fit_X = NO_HDR_DF.copy()

        if scd_fit_name == 'GOOD_DF':
            scd_fit_X = GOOD_DF.copy()
        elif scd_fit_name == 'BAD_DF':
            scd_fit_X = BAD_DF.copy()
        elif scd_fit_name == 'NO_HDR_DF':
            scd_fit_X = NO_HDR_DF.copy()

        if trfm_name == 'GOOD_DF':
            trfm_X = GOOD_DF.copy()
        elif trfm_name == 'BAD_DF':
            trfm_X = BAD_DF.copy()
        elif trfm_name == 'NO_HDR_DF':
            trfm_X = NO_HDR_DF.copy()

        scd_value_error = 0
        trfm_value_error = 0
        scd_warn = 0
        trfm_warn = 0

        if scd_fit_name != fst_fit_name:
            scd_value_error += 1
            if 'NO_HDR_DF' in [fst_fit_name, scd_fit_name]:
                scd_value_error -= 1
                scd_warn += 1

        if trfm_name != fst_fit_name:
            trfm_value_error += 1
            if 'NO_HDR_DF' in [fst_fit_name, trfm_name]:
                trfm_value_error -= 1
                trfm_warn += 1

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.partial_fit(fst_fit_X)
        if scd_value_error:
            with pytest.raises(ValueError):
                TestCls.partial_fit(scd_fit_X)
        elif scd_warn:
            with pytest.warns():
                TestCls.partial_fit(scd_fit_X)
        else:
            TestCls.partial_fit(scd_fit_X)

        if trfm_value_error:
            with pytest.raises(ValueError):
                TestCls.transform(trfm_X)
        elif trfm_warn:
            with pytest.warns():
                TestCls.transform(trfm_X)
        else:
            TestCls.transform(trfm_X)
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        TestCls = MinCountTransformer(*_args, **_kwargs)

        TestCls.fit(fst_fit_X)

        if trfm_value_error:
            with pytest.raises(ValueError):
                TestCls.transform(trfm_X)
        elif trfm_warn:
            with pytest.warns():
                TestCls.transform(trfm_X)
        else:
            TestCls.transform(trfm_X)
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # SHOULD NEVER EXCEPT FOR HEADER ISSUE
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit_transform(fst_fit_X)


# END TEST ValueError WHEN SEES A DF HEADER  DIFFERENT FROM FIRST-SEEN HEADER


# TEST FOR ValueError ON BAD X DF COLUMNS, ROWS ########################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestValueErrorBadXDFColumnsRows:


    @staticmethod
    @pytest.fixture
    def X_builder(x_rows:int, x_cols:int):
        def foo(
                X: np.ndarray,
                COLUMNS: list,
                diff_cols: str,
                diff_rows: str
            ):

            if diff_cols == 'good':
                NEW_X = X.copy()
                NEW_X_HDR = COLUMNS.copy()
                assert NEW_X.shape[0] == x_rows
                assert NEW_X.shape[1] == x_cols
            elif diff_cols == 'less_col':
                NEW_X = X.copy()[:, :x_cols // 2]
                NEW_X_HDR = COLUMNS.copy()[:x_cols // 2]
                assert NEW_X.shape[0] == x_rows
                assert NEW_X.shape[1] == x_cols // 2
            elif diff_cols == 'more_col':
                NEW_X = np.hstack((X.copy(), X.copy()))
                __ = np.array([str(uuid.uuid4())[:4] for _ in range(x_cols)])
                NEW_X_HDR = np.hstack((COLUMNS.copy(), __))
                assert NEW_X.shape[0] == x_rows
                assert NEW_X.shape[1] == 2 * x_cols

            if diff_rows == 'good':
                # KEEP X & HDR FROM COLUMN SECTION
                pass
            elif diff_rows == 'less_row':
                if len(NEW_X.shape) == 1:
                    NEW_X = NEW_X[:x_rows // 2]
                elif len(NEW_X.shape) == 2:
                    NEW_X = NEW_X[:x_rows // 2, :]
                else:
                    raise Exception(f"X_builder() NEW_X.shape logic failed")
                # KEEP HDR FROM COLUMN SECTION
                assert NEW_X.shape[0] == x_rows // 2
            elif diff_rows == 'more_row':
                if len(NEW_X.shape) == 1:
                    NEW_X = np.hstack((NEW_X, NEW_X))
                elif len(NEW_X.shape) == 2:
                    NEW_X = np.vstack((NEW_X, NEW_X))
                else:
                    raise Exception(f"X_builder() NEW_X.shape logic failed")
                # KEEP HDR FROM COLUMN SECTION
                assert NEW_X.shape[0] == 2 * x_rows

            NEW_X = pd.DataFrame(data=NEW_X, columns=NEW_X_HDR, dtype=object)

            return NEW_X

        return foo


    # end X_builder ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    SECOND_FIT_X_SAME_DIFF_COLUMNS = ['good', 'less_col', 'more_col']
    SECOND_FIT_X_SAME_DIFF_ROWS = ['good', 'less_row', 'more_row']
    TRANSFORM_X_SAME_DIFF_COLUMNS = ['good', 'less_col', 'more_col']
    TRANSFORM_X_SAME_DIFF_ROWS = ['good', 'less_row', 'more_row']

    @pytest.mark.parametrize('scd_fit_x_cols', SECOND_FIT_X_SAME_DIFF_COLUMNS)
    @pytest.mark.parametrize('scd_fit_x_rows', SECOND_FIT_X_SAME_DIFF_ROWS)
    @pytest.mark.parametrize('trfm_x_cols', TRANSFORM_X_SAME_DIFF_COLUMNS)
    @pytest.mark.parametrize('trfm_x_rows', TRANSFORM_X_SAME_DIFF_ROWS)

    def test_value_error_bad_X_DF_columns_rows(self, X, y, COLUMNS, _args,
        _kwargs, scd_fit_x_cols, scd_fit_x_rows, trfm_x_cols, trfm_x_rows,
        X_builder, x_rows, x_cols):

        TestCls = MinCountTransformer(*_args, **_kwargs)

        # first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        fst_fit_X = X_builder(
            X.copy(),
            COLUMNS,
            diff_cols='good',
            diff_rows='good'
        )
        # end first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        scd_fit_X = X_builder(
            X.copy(),
            COLUMNS,
            diff_cols=scd_fit_x_cols,
            diff_rows=scd_fit_x_rows
        )
        # end second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        trfm_X = X_builder(
            X.copy(),
            COLUMNS,
            diff_cols=trfm_x_cols,
            diff_rows=trfm_x_rows
        )
        # end transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        fst_fit_y = y.copy()[:, 0].ravel()
        scd_fit_y = fst_fit_y.copy()
        trfm_y = fst_fit_y.copy()


        value_error = 0

        # ValueError WHEN ROWS OF y (IF PASSED) != X ROWS IN transform ONLY
        if trfm_X is not None and trfm_y.shape[0] != trfm_X.shape[0]:
            value_error += 1

        try:
            fst_fit_X.shape[1]
            fst_fit_X_cols = fst_fit_X.shape[1]
        except:
            fst_fit_X_cols = 1

        try:
            scd_fit_X.shape[1]
            scd_fit_X_cols = scd_fit_X.shape[1]
        except:
            scd_fit_X_cols = 1

        if scd_fit_X_cols != fst_fit_X_cols:
            value_error += 1

        try:
            trfm_X.shape[1]
            trfm_X_cols = trfm_X.shape[1]
        except:
            trfm_X_cols = 1

        if trfm_X_cols != fst_fit_X_cols:
            value_error += 1

        if value_error:
            with pytest.raises(ValueError):
                TestCls.partial_fit(fst_fit_X, fst_fit_y)
                TestCls.partial_fit(scd_fit_X, scd_fit_y)
                TestCls.transform(trfm_X, trfm_y)
        elif not value_error:
            TestCls.partial_fit(fst_fit_X, fst_fit_y)
            TestCls.partial_fit(scd_fit_X, scd_fit_y)
            TestCls.transform(trfm_X, trfm_y)


    del SECOND_FIT_X_SAME_DIFF_COLUMNS, SECOND_FIT_X_SAME_DIFF_ROWS
    del TRANSFORM_X_SAME_DIFF_COLUMNS, TRANSFORM_X_SAME_DIFF_ROWS

# END TEST FOR ValueError ON BAD X DF COLUMNS, ROWS ####################


# TEST ignore_float_columns WORKS ######################################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestIgnoreFloatColumnsWorks:

    def test_ignore_float_columns_works(
        self, X, NO_NAN_X, y, _args, _kwargs, _mct_cols
    ):

        # FLOAT ONLY COLUMNS SHOULD BE 3rd GROUP OF COLUMNS
        FLOAT_ONLY_X = NO_NAN_X[:, (2 * _mct_cols):(3 * _mct_cols)]

        # ignore_float_columns = False SHOULD delete all columns and rows
        _kwargs['ignore_float_columns'] = False
        TestCls = MinCountTransformer(*_args, **_kwargs)

        # this isnt excepting when using regular X (with nans). But
        # is working with NO_NAN_X.... y?
        with pytest.raises(ValueError):
            TestCls.fit_transform(FLOAT_ONLY_X, y)

        # ignore_float_columns = True SHOULD not delete anything
        _kwargs['ignore_float_columns'] = True
        TestCls = MinCountTransformer(*_args, **_kwargs)

        OUTPUT_FLOAT_ONLY_X, OUTPUT_FLOAT_ONLY_y = \
                                TestCls.fit_transform(FLOAT_ONLY_X, y)

        X_MASK = np.logical_not(np.isnan(OUTPUT_FLOAT_ONLY_X.astype(np.float64)))
        Y_MASK = np.logical_not(np.isnan(FLOAT_ONLY_X.astype(np.float64)))
        assert np.array_equiv(OUTPUT_FLOAT_ONLY_X[X_MASK], FLOAT_ONLY_X[Y_MASK])
        del X_MASK, Y_MASK

        X_MASK = np.logical_not(np.isnan(OUTPUT_FLOAT_ONLY_y.astype(np.float64)))
        Y_MASK = np.logical_not(np.isnan(y.astype(np.float64)))
        assert np.array_equiv(OUTPUT_FLOAT_ONLY_y[X_MASK], y[Y_MASK])

        del TestCls, FLOAT_ONLY_X, OUTPUT_FLOAT_ONLY_X, OUTPUT_FLOAT_ONLY_y

# END TEST ignore_float_columns WORKS ##################################


# TEST ignore_non_binary_integer_columns WORKS #########################

@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestIgnoreNonBinaryIntegerColumnsWorks:

    def test_ignore_non_binary_integer_columns_works(
        self, X, y, _args, _kwargs, _mct_cols
    ):

        # NON-BINARY INTEGER COLUMNS SHOULD BE 2nd GROUP OF COLUMNS
        NON_BIN_INT_ONLY_X = X[:, _mct_cols:(2 * _mct_cols)]

        # ignore_non_binary_integer_columns = False deletes some rows
        _kwargs['ignore_non_binary_integer_columns'] = False
        TestCls = MinCountTransformer(_args[0]+1, **_kwargs)

        OUTPUT_NON_BIN_INT_ONLY_X = TestCls.fit_transform(NON_BIN_INT_ONLY_X, y)[0]
        X_MASK = np.logical_not(
            np.isnan(OUTPUT_NON_BIN_INT_ONLY_X.astype(np.float64))
        )
        Y_MASK = np.logical_not(np.isnan(NON_BIN_INT_ONLY_X.astype(np.float64)))
        assert not np.array_equiv(
            OUTPUT_NON_BIN_INT_ONLY_X[X_MASK],
            NON_BIN_INT_ONLY_X[Y_MASK]
        )
        del OUTPUT_NON_BIN_INT_ONLY_X, X_MASK, Y_MASK


        # ignore_non_binary_integer_columns = True, ignore_nan = True
        # SHOULD not delete anything
        _kwargs['ignore_non_binary_integer_columns'] = True
        TestCls = MinCountTransformer(*_args, **_kwargs)

        OUTPUT_NON_BIN_INT_ONLY_X = TestCls.fit_transform(NON_BIN_INT_ONLY_X, y)[0]
        X_MASK = np.logical_not(
            np.isnan(OUTPUT_NON_BIN_INT_ONLY_X.astype(np.float64))
        )
        Y_MASK = np.logical_not(np.isnan(NON_BIN_INT_ONLY_X.astype(np.float64)))
        assert np.array_equiv(
            OUTPUT_NON_BIN_INT_ONLY_X[X_MASK],
            NON_BIN_INT_ONLY_X[Y_MASK]
        )
        del OUTPUT_NON_BIN_INT_ONLY_X, X_MASK, Y_MASK

        del TestCls, NON_BIN_INT_ONLY_X

# END TEST ignore_non_binary_integer_columns WORKS #####################




# TEST ignore_nan WORKS ################################################

@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestIgnoreNanWorks:

    @pytest.mark.parametrize('has_nan', [False, True])
    @pytest.mark.parametrize('nan_type', ['np_nan', 'str_nan'])
    @pytest.mark.parametrize('_ignore_nan', [False, True])
    def test_ignore_nan_works(
        self, _args, _kwargs, has_nan, nan_type, _ignore_nan, x_rows
    ):

        _kwargs['ignore_float_columns'] = False

        # RIG A VECTOR SO THAT ONE CAT WOULD BE KEPT, ANOTHER CAT WOULD
        # BE DELETED, AND nan WOULD BE DELETED
        NOT_DEL_VECTOR = np.random.choice(
            [2, 3],
            x_rows - _args[0] + 1,
            replace=True
        ).astype(np.float64)

        # SPRINKLE nan INTO THIS VECTOR
        if has_nan:
            MASK = np.random.choice(
                range(x_rows - _args[0] + 1),
                _args[0] - 1,
                replace=False
            )
            NOT_DEL_VECTOR[MASK] = np.nan
        if nan_type == 'np_nan':
            pass
        elif nan_type == 'str_nan':
            NOT_DEL_VECTOR = NOT_DEL_VECTOR.astype(object)


        # STACK ON A VECTOR OF VALUES THAT WILL BE ALWAYS BE DELETED
        TEST_X = np.hstack((
            NOT_DEL_VECTOR, [2.5 for _ in range(_args[0] - 1)]
        )).ravel()

        del NOT_DEL_VECTOR

        # RESHAPE, MCT REQUIRES X TO BE 2D
        TEST_X = TEST_X.reshape((-1, 1))

        TEST_Y = np.random.randint(0, 2, len(TEST_X))


        _kwargs['ignore_nan'] = _ignore_nan

        TestCls = MinCountTransformer(*_args, **_kwargs)

        TestCls.fit(TEST_X, TEST_Y)
        TRFM_X, TRFM_Y = TestCls.transform(TEST_X, TEST_Y)

        _a = _args[0] - 1
        _b = has_nan * np.logical_not(_ignore_nan) * (_args[0] - 1)
        correct_x_and_y_len = x_rows - _a - _b
        del _a, _b

        assert len(TRFM_X) == correct_x_and_y_len, \
            f"TRFM_X is not the correct length after transform"

        assert len(TRFM_Y) == correct_x_and_y_len, \
            f"TRFM_X is not the correct length after transform"

        if TestCls.ignore_nan == True:
            # 2.5's SHOULD BE DELETED, BUT NOT nan
            if nan_type == 'str_nan':
                MASK = (TEST_X != 2.5).astype(bool)
            elif nan_type == 'np_nan':
                MASK = (TEST_X != 2.5).astype(bool)
        elif TestCls.ignore_nan == False:
            # 2.5's AND nan SHOULD BE DELETED
            if nan_type == 'str_nan':
                MASK = ((TEST_X != 2.5) * (TEST_X.astype(str) != f'{np.nan}'))
            elif nan_type == 'np_nan':
                MASK = ((TEST_X != 2.5) * np.logical_not(np.isnan(TEST_X)))

        REF_X = TEST_X[MASK]
        REF_Y = TEST_Y.reshape((-1, 1,))[MASK].ravel()

        assert len(REF_X) == correct_x_and_y_len, \
            f"REF_X is not the correct length"

        assert len(REF_Y) == correct_x_and_y_len, \
            f"REF_X is not the correct length"

        del correct_x_and_y_len

        def _formatter(ARRAY_1, MASK=None):
            if MASK is None:
                MASK = [True for _ in ARRAY_1.ravel()]
            return ARRAY_1[MASK].ravel().astype(np.float64)


        if has_nan and TestCls.ignore_nan == True:
            NAN_MASK = np.logical_not(np.isnan(REF_X.astype(np.float64)))

            assert np.array_equiv(
                _formatter(TRFM_X, MASK=NAN_MASK),
                _formatter(REF_X, MASK=NAN_MASK)
            ), f"TRFM_X != EXPECTED X"

            assert np.array_equiv(
                _formatter(TRFM_Y, MASK=NAN_MASK),
                _formatter(REF_Y, MASK=NAN_MASK)
            ), f"TRFM_Y != EXPECTED Y"

            del NAN_MASK
        else:

            assert np.array_equiv(_formatter(TRFM_X), _formatter(REF_X)), \
                f"TRFM_X != EXPECTED X"

            assert np.array_equiv(_formatter(TRFM_Y), _formatter(REF_Y)), \
                f"TRFM_Y != EXPECTED Y"

# END TEST ignore_nan WORKS ############################################



# TEST ignore_columns WORKS ############################################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestIgnoreColumnsWorks:

    def test_ignore_columns_works(
        self, NO_NAN_X, y, _args, _kwargs, _mct_cols
    ):

        # USE FLOAT AND STR COLUMNS
        NEW_X = NO_NAN_X[:, 2 * _mct_cols:].copy()

        _args = [2 * _args[0]]
        _kwargs['ignore_float_columns'] = True

        # DEMONSTRATE THAT THIS THRESHOLD WILL ALTER X (AND y)
        # MANY OR ALL STR ROWS SHOULD BE DELETED
        TestCls = MinCountTransformer(*_args, **_kwargs)
        _everything_was_deleted = False
        try:
            OUTPUT_X, OUTPUT_y = TestCls.fit_transform(NEW_X, y)
        except:
            # this means that everything was deleted, which is good
            _everything_was_deleted = True

        # if not everything was deleted, look to see if anything was deleted
        if not _everything_was_deleted:
            err_msg = lambda i: (f"ignore_columns {i} was not altered "
                f"when high threshold on str columns")
            assert not np.array_equiv(OUTPUT_X, NEW_X), err_msg('X')
            assert not np.array_equiv(OUTPUT_y, y), err_msg('y')
            del OUTPUT_X, OUTPUT_y, err_msg

        del _everything_was_deleted

        _kwargs['ignore_columns'] = np.arange(_mct_cols, NEW_X.shape[1], 1)

        # SHOW THAT WHEN THE COLUMNS ARE IGNORED THAT X (AND y) ARE NOT ALTERED
        TestCls = MinCountTransformer(*_args, **_kwargs)
        OUTPUT_X, OUTPUT_y = TestCls.fit_transform(NEW_X, y)
        err_msg = lambda i: (f"ignore_columns {i} was altered when the only "
                             f"columns that could change were ignored")
        assert np.array_equiv(OUTPUT_X, NEW_X), err_msg('X')
        assert np.array_equiv(OUTPUT_y, y), err_msg('y')

        del OUTPUT_X, OUTPUT_y, err_msg

        del NEW_X, TestCls

# END TEST ignore_columns WORKS ########################################


# TEST handle_as_bool WORKS ############################################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestHandleAsBoolWorks:

    def test_handle_as_bool_works(
        self, X, NO_NAN_X, y, _args, _kwargs, _mct_cols, x_rows, x_cols
    ):

        _kwargs['ignore_non_binary_integer_columns'] = False

        # USE NON_BINARY_INT COLUMNS
        NEW_X = NO_NAN_X[:, _mct_cols:2 * _mct_cols].copy()

        # RIG ONE OF THE COLUMNS WITH ENOUGH ZEROS THAT IT WOULD BE DELETED
        # WHEN HANDLED AS AN INT --- BECAUSE EACH INT WOULD BE < count_threshold,
        # DELETING THEM, LEAVING A COLUMN OF ALL ZEROS, WHICH WOULD THEN
        # BE DELETED
        RIGGED_INTEGERS = np.zeros(x_rows, dtype=np.uint32)
        for row_idx in range(1, _args[0] + 2):
            RIGGED_INTEGERS[row_idx] = row_idx
        NEW_X[:, -1] = RIGGED_INTEGERS
        del RIGGED_INTEGERS

        # DEMONSTRATE THAT ONE CHOP WHEN NOT HANDLED AS BOOL WILL SHRINK
        # ROWS AND ALSO DELETE 1 COLUMN FROM X
        _kwargs['handle_as_bool'] = None
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TRFM_X = TestCls.fit_transform(NEW_X)
        assert TRFM_X.shape[1] == NEW_X.shape[1] - 1
        assert TRFM_X.shape[0] < NEW_X.shape[0]
        del TRFM_X, TestCls

        # DEMONSTRATE THAT WHEN ZERO-PEPPERED COLUMN IS HANDLED AS A
        # BOOL, THE COLUMN IS RETAINED
        _kwargs['handle_as_bool'] = [NEW_X.shape[1] - 1]
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TRFM_X = TestCls.fit_transform(NEW_X)
        assert TRFM_X.shape[1] == NEW_X.shape[1]

        # TEST handle_as_bool CANNOT BE USED ON STR ('obj') COLUMNS
        # STR COLUMNS SHOULD BE [:, 3*_mct_cols:] ON ORIGINAL X
        # PICK ONE COLUMN IS STR; ONE EACH FROM BIN-INT, INT, AND FLOAT
        for col_idx in [
            _mct_cols - 1,
            2 * _mct_cols - 1,
            3 * _mct_cols - 1,
            3 * _mct_cols
        ]:
            _kwargs['handle_as_bool'] = [col_idx]
            TestCls = MinCountTransformer(*_args, **_kwargs)
            if col_idx in range(3 * _mct_cols, x_cols): # IF IS STR SHOULD RAISE
                with pytest.raises(ValueError):
                    TestCls.fit(X, y)
            else:
                TestCls.fit(X, y)  # OTHERWISE SHOULD PASS

            del TestCls

        # DEMONSTRATE THAT AFTER fit() WITH VALID handle_as_bool, IF
        # handle_as_bool IS CHANGED TO INVALID, RAISES ValueError
        _kwargs['handle_as_bool'] = [_mct_cols + 1]  # A NON-BINARY INT COLUMN
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.partial_fit(X)

        # pizza, need to reset() 'handle_as_bool' is blocked otherwise
        TestCls.reset()

        TestCls.set_params(handle_as_bool=[x_cols - 1])  # STR COLUMN
        with pytest.raises(ValueError):
            TestCls.partial_fit(X)

        TestCls._handle_as_bool = [x_cols - 1]  # STR COLUMN
        with pytest.raises(ValueError):
            TestCls.partial_fit(X)

        TestCls._handle_as_bool = [x_cols - 1]  # STR COLUMN
        with pytest.raises(ValueError):
            TestCls.transform(X)

        del NEW_X, TestCls, TRFM_X, col_idx

# END TEST handle_as_bool WORKS ########################################


# TEST delete_axis_0 WORKS #############################################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestDeleteAxis0Works:

    def test_delete_axis_0_works(
        self, NO_NAN_X, y, COLUMNS, _args, _kwargs, _mct_cols
    ):

        args = [2 * _args[0]]

        # 1) USE FLOAT AND STR COLUMNS ONLY, ignore_float_columns=True
        # 2A) RUN MCT WHEN STRS ARE NOT DUMMIED, THE FLOAT COLUMN OUTPUT
        #   IS SUPPOSED TO BE THE "TRUTH" FOR FLOAT COLUMNS WHEN DUMMIED AND
        #   delete_axis_0 = True
        # 2B) Ensure some rows were actually deleted (BUT NOT ALL) by comparing
        #   NO_DUMMY_BASELINE_FLOAT_DF against ORIGINAL UN-TRANSFORMED FLOAT_DF
        # 3A) BUILD THE DUMMIED EQUIVALENT OF THE FLOAT AND STR COLUMN DF
        # 3B) PROVE NO ROWS ARE DELETED FROM DUMMIED WHEN delete_axis_0 is False
        # 3C) RUN delete_axis_0 True ON DUMMIED, COMPARE THE FLT COLUMNS AGAINST
        #   "TRUTH" TO SEE IF THE RESULTS ARE EQUAL

        # 1) USE FLOAT AND STR COLUMNS ONLY * * * * * * * * * * * * * * * *
        FLOAT_STR_DF = pd.DataFrame(
            data=NO_NAN_X[:, 2 * _mct_cols:4 * _mct_cols].copy(),
            columns=COLUMNS[2 * _mct_cols:4 * _mct_cols].copy()
        )

        FLOAT_DF = pd.DataFrame(
            data=NO_NAN_X[:, 2 * _mct_cols:3 * _mct_cols].copy(),
            columns=COLUMNS[2 * _mct_cols:3 * _mct_cols].copy()
        )  # "TRUTH" for when delete_axis_0 = False

        # KEEP STR_DF TO DO OneHot
        STR_DF = pd.DataFrame(
                data=NO_NAN_X[:, 3 * _mct_cols:4 * _mct_cols].copy(),
                columns=COLUMNS[3 * _mct_cols:4 * _mct_cols].copy()
        )
        # END 1) USE FLOAT AND STR COLUMNS ONLY * * * * * * * * * * * * * * *

        # 2A) RUN MCT WHEN STRS ARE NOT DUMMIED, THE FLOAT COLUMN OUTPUT
        #   IS SUPPOSED TO BE THE "TRUTH" FOR FLOAT COLUMNS WHEN DUMMIED AND
        #   delete_axis_0 = True
        ChopStrTestCls = MinCountTransformer(*args, **_kwargs)
        # DOESNT MATTER WHAT delete_axis_0 IS SET TO, THERE ARE NO BOOL COLUMNS
        ChopStrTestCls.set_params(ignore_float_columns=True)
        STR_FLT_NO_DUMMY_BASELINE = ChopStrTestCls.fit_transform(FLOAT_STR_DF)

        STR_FLT_NO_DUMMY_BASELINE_DF = pd.DataFrame(
            data=STR_FLT_NO_DUMMY_BASELINE,
            columns=ChopStrTestCls.get_feature_names_out(None)
        )
        del ChopStrTestCls, STR_FLT_NO_DUMMY_BASELINE, FLOAT_STR_DF

        NO_DUMMY_BASELINE_FLOAT_DF = \
            STR_FLT_NO_DUMMY_BASELINE_DF.iloc[:, :_mct_cols]
        del STR_FLT_NO_DUMMY_BASELINE_DF
        # END 2A * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        # 2B) Ensure some rows were actually deleted (but not all) by comparing
        #       NO_DUMMY_BASELINE_FLOAT_DF against ORIGINAL UN-TRANSFORMED FLOAT_DF
        assert not NO_DUMMY_BASELINE_FLOAT_DF.equals(FLOAT_DF), \
            f"MinCountTransform of FLOAT_STR_DF did not delete any rows"
        assert NO_DUMMY_BASELINE_FLOAT_DF.shape[0] > 0, \
            f"ALL ROWS WERE DELETED WHEN DURING BASELINE NO DUMMY TRANSFORM"
        # END 2B * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        # 3A) BUILD THE DUMMIED EQUIVALENT OF THE FLOAT AND STR COLUMN DF
        onehot = OneHotEncoder(
            categories='auto',
            drop=None,
            sparse_output=False,
            dtype=np.uint8,
            handle_unknown='error',
            min_frequency=None,
            max_categories=None,
            feature_name_combiner='concat'
        )

        onehot.fit(STR_DF)

        DUMMIED_STR_DF = pd.DataFrame(
            data=onehot.transform(STR_DF),
            columns=onehot.get_feature_names_out()
        )
        FULL_DUMMIED_STR_FLOAT_DF = pd.concat((FLOAT_DF, DUMMIED_STR_DF), axis=1)
        del onehot, STR_DF, DUMMIED_STR_DF
        # END 3A * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        # 3B) PROVE NO ROWS ARE DELETED FROM DUMMIED WHEN delete_axis_0 is False
        DummyDontDeleteAxis0TestCls = MinCountTransformer(*args, **_kwargs)
        DummyDontDeleteAxis0TestCls.set_params(
            delete_axis_0=False, ignore_float_columns=True
        )
        DUMMIED_FLT_STR_DONT_DELETE_AXIS_0 = \
            DummyDontDeleteAxis0TestCls.fit_transform(FULL_DUMMIED_STR_FLOAT_DF)

        DUMMIED_FLT_STR_DONT_DELETE_AXIS_0_DF = pd.DataFrame(
            data=DUMMIED_FLT_STR_DONT_DELETE_AXIS_0,
            columns=DummyDontDeleteAxis0TestCls.get_feature_names_out(None)
        )
        del DummyDontDeleteAxis0TestCls, DUMMIED_FLT_STR_DONT_DELETE_AXIS_0

        DUMMIED_FLT_DONT_DELETE_AXIS_0_DF = \
            DUMMIED_FLT_STR_DONT_DELETE_AXIS_0_DF.iloc[:, :_mct_cols]
        del DUMMIED_FLT_STR_DONT_DELETE_AXIS_0_DF

        # Compare DUMMIED_FLT_DONT_DELETE_AXIS_0_DF against FLOAT_DF
        assert DUMMIED_FLT_DONT_DELETE_AXIS_0_DF.equals(FLOAT_DF), \
            (f"floats with dummies and delete_axis_0=False do not "
             f"equal original untransformed floats (rows were deleted)")

        del DUMMIED_FLT_DONT_DELETE_AXIS_0_DF
        # END 3B * * * * * * * * * * * * * * * * * * * * * * * * * * * *


        # 3C) RUN delete_axis_0 True ON DUMMIED STRINGS, COMPARE THE FLT COLUMNS
        # AGAINST NON-DUMMIED STRINGS TO SEE IF THE RESULTS ARE EQUAL
        DummyDeleteAxis0TestCls = MinCountTransformer(*args, **_kwargs)
        DummyDeleteAxis0TestCls.set_params(
            delete_axis_0=True, ignore_float_columns=True
        )
        DUMMIED_FLT_STR_DELETE_0_X = \
            DummyDeleteAxis0TestCls.fit_transform(FULL_DUMMIED_STR_FLOAT_DF)

        DUMMIED_FLT_STR_DELETE_0_X_DF = pd.DataFrame(
            data=DUMMIED_FLT_STR_DELETE_0_X,
            columns=DummyDeleteAxis0TestCls.get_feature_names_out(None)
        )
        del DummyDeleteAxis0TestCls

        # Compare DUM_MIN_COUNTED_DELETE_0_FLOAT_DF against NO_DUMMY_BASELINE_FLOAT_DF
        DUMMIED_FLT_DELETE_0_X_DF = \
            DUMMIED_FLT_STR_DELETE_0_X_DF.iloc[:, :_mct_cols]
        del DUMMIED_FLT_STR_DELETE_0_X_DF

        assert DUMMIED_FLT_DELETE_0_X_DF.equals(NO_DUMMY_BASELINE_FLOAT_DF), \
            (f"floats with dummies and delete_axis_0=True do not "
             f"equal no-dummy baseline floats")

        del DUMMIED_FLT_DELETE_0_X_DF
        # END 3C * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        del FLOAT_DF, FULL_DUMMIED_STR_FLOAT_DF, NO_DUMMY_BASELINE_FLOAT_DF

# END TEST delete_axis_0 WORKS #########################################



# TEST OUTPUT TYPES ####################################################

@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestOutputTypes:


    @pytest.mark.parametrize('x_input_type',
        ['numpy_array', 'pandas_dataframe', 'scipy_csr']
    )
    @pytest.mark.parametrize('y_input_type',
        ['numpy_array', 'pandas_dataframe', 'pandas_series']
    )
    @pytest.mark.parametrize('output_type',
        [None, 'default','pandas', 'polars']
    )
    def test_output_types(
        self, X, y, _args, _kwargs, x_input_type, y_input_type, output_type
    ):

        NEW_X = X[:, :1].copy()
        NEW_COLUMNS = X[:1].copy()
        NEW_Y = y[:, 0].copy()


        if x_input_type == 'numpy_array':
            TEST_X = NEW_X.copy()
        elif x_input_type == 'pandas_dataframe':
            TEST_X = pd.DataFrame(data=NEW_X, columns=NEW_COLUMNS)
        elif x_input_type == 'scipy_csr':
            TEST_X = ss.csr_array(NEW_X.astype(np.float64))
        else:
            raise Exception

        if y_input_type == 'numpy_array':
            TEST_Y = NEW_Y.copy()
        elif 'pandas' in y_input_type:
            TEST_Y = pd.DataFrame(data=NEW_Y, columns=['y'])
            if y_input_type == 'pandas_series':
                TEST_Y = TEST_Y.squeeze()

        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.set_output(transform=output_type)

        TRFM_X, TRFM_Y = TestCls.fit_transform(TEST_X, TEST_Y)

        # y output container is never changed
        if output_type is None:
            assert type(TRFM_X) == type(TEST_X), \
                (f"output_type is None, X output type ({type(TRFM_X)}) != "
                 f"X input type ({type(TEST_X)})")
            assert type(TRFM_Y) == type(TEST_Y), \
                (f"output_type is None, Y output type ({type(TRFM_Y)}) != "
                 f"Y input type ({type(TEST_Y)})")
        elif output_type == 'default':
            assert isinstance(TRFM_X, np.ndarray), \
                f"output_type is default, TRFM_X is {type(TRFM_X)}"
            assert type(TRFM_Y) == type(TEST_Y), \
                (f"output_type is default, Y output type ({type(TRFM_Y)}) != "
                 f"Y input type ({type(TEST_Y)})")
        elif output_type == 'pandas':
            assert isinstance(TRFM_X, pd.core.frame.DataFrame), \
                f"output_type is pandas dataframe, TRFM_X is {type(TRFM_X)}"
            assert type(TRFM_Y) == type(TEST_Y), \
                (f"output_type is pandas, Y output type ({type(TRFM_Y)}) != "
                 f"Y input type ({type(TEST_Y)})")
        elif output_type == 'polars':
            assert isinstance(TRFM_X, pl.dataframe.frame.DataFrame), \
                f"output_type is polars, TRFM_X is {type(TRFM_X)}"
            assert type(TRFM_Y) == type(TEST_Y), \
                (f"output_type is polars, Y output type ({type(TRFM_Y)}) != "
                 f"Y input type ({type(TEST_Y)})")
        else:
            raise Exception

# TEST OUTPUT TYPES ####################################################



# TEST CONDITIONAL ACCESS TO partial_fit() AND fit() ###################
# 1) partial_fit() should allow unlimited number of subsequent partial_fits()
# 2) one call to fit() should allow subsequent attempts to partial_fit()
# 3) one call to fit() should allow later attempts to fit() (2nd fit will reset)
# 4) calls to partial_fit() should allow later attempt to fit() (fit will reset)
# 5) fit_transform() should allow calls ad libido
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestConditionalAccessToPartialFitAndFit:

    def test_conditional_access_to_partial_fit_and_fit(
        self, X, y, _args, _kwargs
    ):

        TestCls = MinCountTransformer(*_args, **_kwargs)
        TEST_X = X.copy()
        TEST_Y = y.copy()

        # 1)
        for _ in range(5):
            TestCls.partial_fit(TEST_X, TEST_Y)

        del TestCls

        # 2)
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit(TEST_X, TEST_Y)
        TestCls.partial_fit(TEST_X, TEST_Y)

        del TestCls

        # 3)
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit(TEST_X, TEST_Y)
        TestCls.fit(TEST_X, TEST_Y)

        del TestCls

        # 4) a call to fit() after a previous partial_fit() should be allowed
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.partial_fit(TEST_X, TEST_Y)
        TestCls.fit(TEST_X, TEST_Y)

        # 5) fit transform should allow calls ad libido
        for _ in range(5):
            TestCls.fit_transform(TEST_X, TEST_Y)

        del TEST_X, TEST_Y, TestCls

# END TEST CONDITIONAL ACCESS TO partial_fit() AND fit() ###############



# TEST CONDITIONAL ACCESS TO RECURSION #################################
# 1) access to partial_fit, fit or transform when max_recursions > 1 is blocked
# 2) access fit & transform when max_recursions > 1 can only be through fit_transform
# 3) access to partial_fit, fit or transform when max_recursions == 1 is not blocked
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestConditionalAccessToRecursion:

    def test_conditional_access_to_recursion(self, X, y, _args, _kwargs):

        _kwargs['max_recursions'] = 3

        TEST_X = X.copy()
        TEST_Y = y.copy()

        TestCls = MinCountTransformer(*_args, **_kwargs)

        # 1)
        with pytest.raises(ValueError):
            MinCountTransformer(*_args, **_kwargs).partial_fit(TEST_X, TEST_Y)

        with pytest.raises(ValueError):
            MinCountTransformer(*_args, **_kwargs).fit(TEST_X, TEST_Y)

        with pytest.raises(ValueError):
            MinCountTransformer(*_args, **_kwargs).transform(TEST_X, TEST_Y)

        # 2)
        for _ in range(5):
            TestCls.fit_transform(TEST_X, TEST_Y)

        # 3)
        _kwargs['max_recursions'] = 1
        TestCls = MinCountTransformer(*_args, **_kwargs)
        for _name, cls_method in zip(
            ['fit', 'partial_fit', 'transform'],
            [TestCls.fit, TestCls.partial_fit, TestCls.transform]
        ):

            cls_method(TEST_X, TEST_Y)

        del TEST_X, TEST_Y, TestCls, _name, cls_method, _

# END TEST CONDITIONAL ACCESS TO RECURSION #############################


# TEST ALL COLUMNS WILL BE DELETED #####################################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAllColumnsWillBeDeleted:

    def test_all_columns_will_be_deleted(
        self, _args, _kwargs, _mct_rows, x_cols, x_rows
    ):

        # CREATE VERY SPARSE DATA
        TEST_X = np.zeros((_mct_rows, x_cols), dtype=np.uint8)
        TEST_Y = np.random.randint(0, 2, _mct_rows)

        for col_idx in range(x_cols):
            MASK = np.random.choice(range(x_rows), 2, replace=False), col_idx
            TEST_X[MASK] = 1
        del MASK

        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit(TEST_X, TEST_Y)

        # pizza probably separate this out, make a new folder since this is a method directly on MCT
        # pizza replace with print_instructions
        # TestCls.test_threshold()
        # print(f'^^^ mask building instructions should be displayed above ^^^')

        with pytest.raises(ValueError):
            TestCls.transform(TEST_X, TEST_Y)

        del TEST_X, TEST_Y, col_idx, TestCls

# TEST ALL COLUMNS WILL BE DELETED #####################################


# TEST ALL ROWS WILL BE DELETED ########################################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAllRowsWillBeDeleted:

    def test_all_rows_will_be_deleted(
        self, _args, _kwargs, _mct_rows, x_cols
    ):

        # ALL FLOATS
        TEST_X = np.random.uniform(0, 1, (_mct_rows, x_cols))
        TEST_Y = np.random.randint(0, 2, _mct_rows)

        _kwargs['ignore_float_columns'] = False
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit(TEST_X, TEST_Y)

        # pizza probably separate this out, make a new folder since this is a method directly on MCT
        # pizza replace with print_instructions
        # TestCls.test_threshold()
        # print(f'^^^ mask building instructions should be displayed above ^^^')

        with pytest.raises(ValueError):
            TestCls.transform(TEST_X, TEST_Y)

        del TEST_X, TEST_Y, TestCls

# TEST ALL ROWS WILL BE DELETED ########################################


# TEST BIN INT COLUMN WITH ALL ABOVE THRESHOLD NOT DELETED #############
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestBinIntAboveThreshNotDeleted:

    def test_bin_int_above_thresh_not_deleted(self, _kwargs):

        TestCls = MinCountTransformer(2, **_kwargs)

        NEW_X = np.array(
            [['a',0], ['b',0], ['a',1], ['b',1], ['c',0]], dtype=object
        )
        NEW_Y = np.array([0, 1, 0, 1, 1], dtype=np.uint8)

        TestCls.fit(NEW_X, NEW_Y)

        TRFM_X, TRFM_Y = TestCls.transform(NEW_X, NEW_Y)

        assert TRFM_X.shape[1]==2, \
            f"bin int column with all values above threshold was deleted"

        assert TRFM_X.shape[0]==4, \
            f"TRFM_X should have 4 rows but has {TRFM_X.shape[0]}"

# END TEST BIN INT COLUMN WITH ALL ABOVE THRESHOLD NOT DELETED #########


# TEST ACCURACY ********************************************************

class TestAccuracy:


    @pytest.mark.parametrize('_dtype', ('flt', 'int', 'str', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('ignore_float_columns', [True, False])
    @pytest.mark.parametrize('ignore_non_binary_integer_columns', [True, False])
    @pytest.mark.parametrize('ignore_columns', [None, [0, 1, 2, 3]])
    @pytest.mark.parametrize('ignore_nan', [True, False])
    @pytest.mark.parametrize('handle_as_bool', ('hab_1', 'hab_2', 'hab_3'))
    @pytest.mark.parametrize('delete_axis_0', [False, True])
    @pytest.mark.parametrize('reject_unseen_values', [False, True])
    def test_2rcr_get_support_accuracy(
        self, _dtype, _has_nan, ignore_non_binary_integer_columns, ignore_nan,
        ignore_float_columns, ignore_columns, handle_as_bool, delete_axis_0,
        reject_unseen_values, _args, _kwargs, X, y, x_rows, x_cols,
    ):

        # this module tests that 2 recursion MCT get_support() matches
        # the columns in the X output from transform

        if _dtype in ('str', 'obj'):
            HANDLE_AS_BOOL = None
        elif handle_as_bool == 'hab_1':
            HANDLE_AS_BOOL = None
        elif handle_as_bool == 'hab_2':
            HANDLE_AS_BOOL = [1]
        elif handle_as_bool == 'hab_3':
            HANDLE_AS_BOOL = lambda X: [1]
        else:
            raise Exception

        kwargs = deepcopy(_kwargs)

        kwargs['ignore_columns'] = ignore_columns
        kwargs['ignore_float_columns'] = ignore_float_columns
        kwargs['ignore_non_binary_integer_columns'] = ignore_non_binary_integer_columns
        kwargs['ignore_nan'] = ignore_nan
        kwargs['handle_as_bool'] = HANDLE_AS_BOOL
        kwargs['delete_axis_0'] = delete_axis_0
        kwargs['reject_unseen_values'] = reject_unseen_values
        kwargs['max_recursions'] = 1

        TestCls = MinCountTransformer(*_args, **kwargs)

        TRFM_X, TRFM_Y = TestCls.fit_transform(X, y)

        _get_support = TestCls.get_support(indices=False)
        assert len(_get_support) == x_cols == X.shape[1]
        assert sum(_get_support) == TRFM_X.shape[1]

        _row_support = TestCls.get_row_support(indices=False)

        # y
        assert np.array_equal(y[_row_support], TRFM_Y)


        # verify X was changed
        assert TRFM_X.shape != X.shape

        # individual X columns that were kept
        _kept_idx = 0
        for c_idx, _kept in enumerate(_get_support):

            if _kept is True:

                assert np.array_equal(
                    X[_row_support, c_idx],
                    TRFM_X[:, _kept_idx]
                )

                _kept_idx += 1



    @pytest.mark.parametrize('count_threshold', [2, 3])
    @pytest.mark.parametrize('ignore_float_columns', [True, False])
    @pytest.mark.parametrize('ignore_non_binary_integer_columns', [True, False])
    @pytest.mark.parametrize('ignore_columns', [None, [0, 1, 2, 3]])
    @pytest.mark.parametrize('ignore_nan', [True, False])
    @pytest.mark.parametrize('handle_as_bool', ('hab_1', 'hab_2', 'hab_3'))
    @pytest.mark.parametrize('delete_axis_0', [False, True])
    @pytest.mark.parametrize('reject_unseen_values', [False, True])
    def test_accuracy_one_rcr(self, _kwargs, X, y, count_threshold, ignore_columns,
        ignore_float_columns, ignore_non_binary_integer_columns, ignore_nan, mmct,
        handle_as_bool, delete_axis_0, reject_unseen_values, _mct_cols, x_cols
    ):

        # this compares 1rcrX1 outputs of MCT and mmct

        if handle_as_bool == 'hab_1':
            HANDLE_AS_BOOL = None
        elif handle_as_bool == 'hab_2':
            HANDLE_AS_BOOL = list(range(_mct_cols, 2 * _mct_cols))
        elif handle_as_bool == 'hab_3':
            HANDLE_AS_BOOL = lambda X: list(range(_mct_cols, 2 * _mct_cols))
        else:
            raise Exception

        args = [count_threshold]
        _kwargs['ignore_float_columns'] = ignore_float_columns
        _kwargs['ignore_non_binary_integer_columns'] = \
            ignore_non_binary_integer_columns
        _kwargs['ignore_columns'] = ignore_columns
        _kwargs['ignore_nan'] = ignore_nan
        _kwargs['handle_as_bool'] = HANDLE_AS_BOOL
        _kwargs['delete_axis_0'] = delete_axis_0
        _kwargs['reject_unseen_values'] = reject_unseen_values
        _kwargs['max_recursions'] = 1

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        TestCls = MinCountTransformer(*args, **_kwargs)
        TRFM_X1, TRFM_Y1 = TestCls.fit_transform(X.copy(), y.copy())

        # validate MCT 1rcrX1 get_support and object dimensions make sense
        _support = TestCls.get_support(indices=False)
        assert len(_support) == X.shape[1]
        assert TRFM_X1.shape[1] == sum(_support)
        del _support

        _row_support = TestCls.get_row_support(indices=False)
        assert len(_row_support) == X.shape[0]
        assert TRFM_X1.shape[0] == sum(_row_support)

        # assert columns in get_support and those actually in the output match
        _get_support_idxs = TestCls.get_support(indices=True)
        _actual_idxs = []
        for col_idx in range(TRFM_X1.shape[1]):
            for col_idx2 in range(X.shape[1]):
                NOT_NAN_MASK = np.logical_not(nan_mask(TRFM_X1[:, col_idx]))
                if np.array_equal(
                    TRFM_X1[:, col_idx][NOT_NAN_MASK],
                    X[_row_support, col_idx2][NOT_NAN_MASK]
                ):
                    _actual_idxs.append(col_idx2)

        assert np.array_equal(_get_support_idxs, _actual_idxs), \
            f"get_support: {_get_support_idxs}, actual_idxs: {_actual_idxs}"
        # END assert columns in get_support and those actually in the data match

        # END validate MCT 1rcrX1 get_support and object dimensions make sense

        del _get_support_idxs, _actual_idxs, _row_support

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # BUILD MOCK_X & MOCK_Y

        try:
            _ignore_columns = ignore_columns(X)
        except:
            _ignore_columns = ignore_columns
        try:
            _handle_as_bool = HANDLE_AS_BOOL(X)
        except:
            _handle_as_bool = HANDLE_AS_BOOL

        MmctCls = mmct()
        MOCK_X1, MOCK_Y1 = MmctCls.trfm(
            X.copy(), y.copy(), _ignore_columns, ignore_nan,
            ignore_non_binary_integer_columns, ignore_float_columns,
            _handle_as_bool, delete_axis_0, count_threshold
        )

        assert len(MmctCls.get_support_) == X.shape[1]
        assert MOCK_X1.shape[1] == sum(MmctCls.get_support_)

        # END BUILD MOCK_X & MOCK_Y #
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        assert np.array_equal(
            TestCls.get_support(indices=False),
            MmctCls.get_support_
        )

        assert TRFM_X1.shape == MOCK_X1.shape
        assert np.array_equiv(
            TRFM_X1[np.logical_not(nan_mask(TRFM_X1))],
            MOCK_X1[np.logical_not(nan_mask(MOCK_X1))]
        )

        assert TRFM_Y1.shape == MOCK_Y1.shape
        assert np.array_equiv(TRFM_Y1.astype(str), MOCK_Y1.astype(str))


    @pytest.mark.parametrize('count_threshold', [2, 3])
    @pytest.mark.parametrize('ignore_float_columns', [True, False])
    @pytest.mark.parametrize('ignore_non_binary_integer_columns', [True, False])
    @pytest.mark.parametrize('ignore_columns', [None, [0, 1, 2, 3]])
    @pytest.mark.parametrize('ignore_nan', [True, False])
    @pytest.mark.parametrize('handle_as_bool', ('hab_1', 'hab_2', 'hab_3'))
    @pytest.mark.parametrize('delete_axis_0', [True, False])
    @pytest.mark.parametrize('reject_unseen_values', (True, False))
    def test_accuracy_two_rcr_one_shot(self, _kwargs, X, y, count_threshold,
        ignore_columns, ignore_float_columns, ignore_non_binary_integer_columns,
        ignore_nan, handle_as_bool, delete_axis_0, reject_unseen_values,
        _mct_cols, x_cols, mmct
    ):

        # this compares 2rcr output of MCT against 2x1rcr output of mmct

        if handle_as_bool == 'hab_1':
            HANDLE_AS_BOOL = None
        elif handle_as_bool == 'hab_2':
            HANDLE_AS_BOOL = list(range(_mct_cols, 2 * _mct_cols))
        elif handle_as_bool == 'hab_3':
            HANDLE_AS_BOOL = lambda X: list(range(_mct_cols, 2 * _mct_cols))
        else:
            raise Exception

        args = [count_threshold]
        _kwargs['ignore_float_columns'] = ignore_float_columns
        _kwargs['ignore_non_binary_integer_columns'] = \
            ignore_non_binary_integer_columns
        _kwargs['ignore_columns'] = ignore_columns
        _kwargs['ignore_nan'] = ignore_nan
        _kwargs['handle_as_bool'] = HANDLE_AS_BOOL
        _kwargs['delete_axis_0'] = delete_axis_0
        _kwargs['reject_unseen_values'] = reject_unseen_values
        _kwargs['max_recursions'] = 2

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # MCT one-shot 2 rcr
        TestCls = MinCountTransformer(*args, **_kwargs)
        TRFM_X, TRFM_Y = TestCls.fit_transform(X.copy(), y.copy())
        # validate MCT 2rcrX1 get_support and object dimensions make sense
        _support = TestCls.get_support(indices=False)
        assert len(_support) == X.shape[1]
        assert TRFM_X.shape[1] == sum(_support)

        _row_support = TestCls.get_row_support(indices=False)
        assert len(_row_support) == X.shape[0]
        assert TRFM_X.shape[0] == sum(_row_support)

        # assert columns in get_support and those actually in the data match
        _get_support_idxs = TestCls.get_support(indices=True)
        _actual_idxs = []
        for col_idx in range(TRFM_X.shape[1]):
            for col_idx2 in range(X.shape[1]):
                NOT_NAN_MASK = np.logical_not(nan_mask(TRFM_X[:, col_idx]))
                if np.array_equal(
                    TRFM_X[:, col_idx][NOT_NAN_MASK],
                    X[_row_support, col_idx2][NOT_NAN_MASK]
                ):
                    _actual_idxs.append(col_idx2)


        assert np.array_equal(_get_support_idxs, _actual_idxs), \
            f"get_support: {_get_support_idxs}, actual_idxs: {_actual_idxs}"

        del _support, _actual_idxs, _get_support_idxs

        # END assert columns in get_support and those actually in the data match

        # END validate MCT 2rcrX1 get_support and object dimensions make sense

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # BUILD MOCK_X & MOCK_Y

        try:
            _ignore_columns = ignore_columns(X)
        except:
            _ignore_columns = ignore_columns
        try:
            _handle_as_bool = HANDLE_AS_BOOL(X)
        except:
            _handle_as_bool = HANDLE_AS_BOOL

        # ** * ** * ** * ** * **
        # mmct first recursion
        mmct_first_rcr = mmct()   # give class a name to access attr later
        MOCK_X1, MOCK_Y1 = mmct_first_rcr.trfm(
            X.copy(), y.copy(), _ignore_columns, ignore_nan,
            ignore_non_binary_integer_columns, ignore_float_columns,
            _handle_as_bool, delete_axis_0, count_threshold
        )

        assert len(mmct_first_rcr.get_support_) == X.shape[1]
        assert MOCK_X1.shape[1] == sum(mmct_first_rcr.get_support_)

        # ** * ** * ** * ** * ** *

        # ADJUST ign_columns & handle_as_bool FOR 2ND RECURSION ** ** ** **
        # USING mmct get_support

        if MOCK_X1.shape[1] == X.shape[1]:
            scd_ignore_columns = _ignore_columns
            mmct_scd_handle_as_bool = _handle_as_bool
        else:
            NEW_COLUMN_MASK = mmct_first_rcr.get_support_

            OG_IGN_COL_MASK = np.zeros(X.shape[1]).astype(bool)
            OG_IGN_COL_MASK[_ignore_columns] = True
            scd_ignore_columns = \
                np.arange(sum(NEW_COLUMN_MASK))[OG_IGN_COL_MASK[NEW_COLUMN_MASK]]
            if _ignore_columns is None or len(scd_ignore_columns) == 0:
                scd_ignore_columns = None

            OG_H_A_B_MASK = np.zeros(X.shape[1]).astype(bool)
            OG_H_A_B_MASK[_handle_as_bool] = True
            mmct_scd_handle_as_bool = \
                np.arange(sum(NEW_COLUMN_MASK))[OG_H_A_B_MASK[NEW_COLUMN_MASK]]
            if _handle_as_bool is None or len(mmct_scd_handle_as_bool) == 0:
                mmct_scd_handle_as_bool = None

            del NEW_COLUMN_MASK, OG_IGN_COL_MASK, OG_H_A_B_MASK
        # END ADJUST ign_columns & handle_as_bool FOR 2ND RECURSION ** ** **


        # ** * ** * ** * ** * ** * **
        # mmct 2nd recursion
        mmct_scd_rcr = mmct()
        MOCK_X2, MOCK_Y2 = mmct_scd_rcr.trfm(
            MOCK_X1, MOCK_Y1, scd_ignore_columns, ignore_nan,
            ignore_non_binary_integer_columns, ignore_float_columns,
            mmct_scd_handle_as_bool, delete_axis_0, count_threshold
        )
        # ** * ** * ** * ** * ** * **

        assert len(mmct_scd_rcr.get_support_) == MOCK_X1.shape[1]
        assert MOCK_X2.shape[1] == sum(mmct_scd_rcr.get_support_)

        del MOCK_X1, MOCK_Y1, scd_ignore_columns, mmct_scd_handle_as_bool

        # END BUILD MOCK_X & MOCK_Y #
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        _adj_get_support = np.array(mmct_first_rcr.get_support_.copy())
        _idxs = np.arange(len(_adj_get_support))[_adj_get_support]
        _adj_get_support[_idxs] = mmct_scd_rcr.get_support_
        del _idxs

        assert np.array_equal(
            TestCls.get_support(indices=False),
            _adj_get_support
        )

        del _adj_get_support


        assert TRFM_X.shape == MOCK_X2.shape
        assert np.array_equiv(
            TRFM_X[np.logical_not(nan_mask(TRFM_X))],
            MOCK_X2[np.logical_not(nan_mask(MOCK_X2))]
        ), (f'TestCls y output WITH max_recursions=2 FAILED')

        assert TRFM_Y.shape == MOCK_Y2.shape
        assert np.array_equiv(TRFM_Y, MOCK_Y2), \
            (f'TestCls y output WITH max_recursions=2 FAILED')


    @pytest.mark.parametrize('count_threshold', [2, 3])
    @pytest.mark.parametrize('ignore_float_columns',[True, False])
    @pytest.mark.parametrize('ignore_non_binary_integer_columns', [True, False])
    @pytest.mark.parametrize('ignore_columns', [None, [0, 1, 2, 3]])
    @pytest.mark.parametrize('ignore_nan', [True, False])
    @pytest.mark.parametrize('handle_as_bool', ('hab_1',  'hab_2', 'hab_3'))
    @pytest.mark.parametrize('delete_axis_0', [False, True])
    @pytest.mark.parametrize('reject_unseen_values', [False, True])
    def test_accuracy_two_rcr_two_shot(self, _kwargs, X, y, count_threshold,
        ignore_columns, ignore_float_columns, ignore_non_binary_integer_columns,
        ignore_nan, handle_as_bool, delete_axis_0, reject_unseen_values,
        _mct_cols, x_cols, mmct
    ):

        # compare MCT 2rcrX1 outputs are equal to MCT 1rcrX2 outputs
        # compare MCT 1rcrX2 output against mmct 1rcrX2 output

        if handle_as_bool == 'hab_1':
            HANDLE_AS_BOOL = None
        elif handle_as_bool == 'hab_2':
            HANDLE_AS_BOOL = list(range(_mct_cols, 2 * _mct_cols))
        elif handle_as_bool == 'hab_3':
            HANDLE_AS_BOOL = lambda X: list(range(_mct_cols, 2 * _mct_cols))
        else:
            raise Exception


        args = [count_threshold]

        _kwargs['ignore_columns'] = ignore_columns
        _kwargs['ignore_float_columns'] = ignore_float_columns
        _kwargs['ignore_non_binary_integer_columns'] = ignore_non_binary_integer_columns
        _kwargs['ignore_nan'] = ignore_nan
        _kwargs['handle_as_bool'] = HANDLE_AS_BOOL
        _kwargs['delete_axis_0'] = delete_axis_0
        _kwargs['reject_unseen_values'] = reject_unseen_values


        # need to make accommodations for mmct kwargs, cant take callables
        try:
            _ignore_columns = ignore_columns(X)
        except:
            _ignore_columns = ignore_columns

        try:
            _handle_as_bool = HANDLE_AS_BOOL(X)
        except:
            _handle_as_bool = HANDLE_AS_BOOL

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # MCT first one-shot rcr
        TestCls1 = MinCountTransformer(*args,**_kwargs)
        TRFM_X_1, TRFM_Y_1 = TestCls1.fit_transform(X.copy(), y.copy())

        _first_support = TestCls1.get_support(indices=False).copy()

        # validate MCT 1rcrX1 get_support and object dimensions make sense
        assert len(_first_support) == X.shape[1]
        assert TRFM_X_1.shape[1] == sum(_first_support)

        # assert columns in get_support and those actually in the data match
        _get_support_idxs = TestCls1.get_support(indices=True)
        _row_support = TestCls1.get_row_support(indices=False)
        _actual_idxs = []
        for col_idx in range(TRFM_X_1.shape[1]):
            for col_idx2 in range(X.shape[1]):
                NOT_NAN_MASK = np.logical_not(nan_mask(TRFM_X_1[:, col_idx]))
                if np.array_equal(
                    TRFM_X_1[:, col_idx][NOT_NAN_MASK],
                    X[_row_support, col_idx2][NOT_NAN_MASK]
                ):
                    _actual_idxs.append(col_idx2)

        assert np.array_equal(_get_support_idxs, _actual_idxs), \
            f"get_support: {_get_support_idxs}, actual_idxs: {_actual_idxs}"

        del _get_support_idxs, _row_support, _actual_idxs

        # END assert columns in get_support and those actually in the data match

        # END validate MCT 1rcrX1 get_support and object dimensions make sense

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



        # ADJUST ign_columns & handle_as_bool FOR 2ND RECURSION
        # USING MCT get_support
        NEW_COLUMN_MASK = TestCls1.get_support(indices=False)

        OG_IGN_COL_MASK = np.zeros(X.shape[1]).astype(bool)
        OG_IGN_COL_MASK[_ignore_columns] = True
        mct_scd_ignore_columns = \
            np.arange(sum(NEW_COLUMN_MASK))[OG_IGN_COL_MASK[NEW_COLUMN_MASK]]
        if _ignore_columns is None or len(mct_scd_ignore_columns) == 0:
            mct_scd_ignore_columns = None

        OG_H_A_B_MASK = np.zeros(X.shape[1]).astype(bool)
        OG_H_A_B_MASK[_handle_as_bool] = True
        mct_scd_handle_as_bool = \
            np.arange(sum(NEW_COLUMN_MASK))[OG_H_A_B_MASK[NEW_COLUMN_MASK]]
        if _handle_as_bool is None or len(mct_scd_handle_as_bool) == 0:
            mct_scd_handle_as_bool = None

        del NEW_COLUMN_MASK, OG_IGN_COL_MASK, OG_H_A_B_MASK
        # END ADJUST ign_columns & handle_as_bool FOR 2ND RECURSION

        _kwargs['ignore_columns'] = mct_scd_ignore_columns
        _kwargs['handle_as_bool'] = mct_scd_handle_as_bool

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # MCT second one-shot rcr
        TestCls2 = MinCountTransformer(*args,**_kwargs)
        TRFM_X_2, TRFM_Y_2 = TestCls2.fit_transform(TRFM_X_1, TRFM_Y_1)

        # validate MCT 1rcrX2 get_support and object dimensions make sense
        _second_support = TestCls2.get_support(indices=False)
        assert len(_second_support) == TRFM_X_1.shape[1]
        assert TRFM_X_2.shape[1] == sum(_second_support)

        # assert columns in get_support and those actually in the data match
        _get_support_idxs = TestCls2.get_support(indices=True)
        _row_support = TestCls2.get_row_support(indices=False)
        _actual_idxs = []
        for col_idx in range(TRFM_X_2.shape[1]):
            for col_idx2 in range(TRFM_X_1.shape[1]):
                NOT_NAN_MASK = np.logical_not(nan_mask(TRFM_X_2[:, col_idx]))
                if np.array_equal(
                    TRFM_X_2[:, col_idx][NOT_NAN_MASK],
                    TRFM_X_1[_row_support, col_idx2][NOT_NAN_MASK]
                ):
                    _actual_idxs.append(col_idx2)


        assert np.array_equal(_get_support_idxs, _actual_idxs), \
            f"get_support: {_get_support_idxs}, actual_idxs: {_actual_idxs}"

        del _second_support, _actual_idxs, _get_support_idxs, _row_support

        # END assert columns in get_support and those actually in the data match

        # END validate MCT 1rcrX2 get_support and object dimensions make sense

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # reset _kwargs
        _kwargs['ignore_columns'] = ignore_columns
        _kwargs['handle_as_bool'] = HANDLE_AS_BOOL

        # compare 2 one-shot MCTs against 1 two-shot MCT - -- - -- - -- - -- - --
        TestCls_2SHOT = MinCountTransformer(*args, **_kwargs)  # use original kwargs!
        TestCls_2SHOT.set_params(max_recursions=2)
        TRFM_X_2SHOT, TRFM_Y_2SHOT = TestCls_2SHOT.fit_transform(X.copy(), y.copy())

        _adj_get_support_mct = np.array(_first_support.copy())
        _idxs = np.arange(len(_adj_get_support_mct))[_adj_get_support_mct]
        _adj_get_support_mct[_idxs] = TestCls2.get_support(indices=False)
        del _idxs

        # compare MCT 2rcrX1 outputs are equal to MCT 1rcrX2 outputs
        assert len(TestCls_2SHOT.get_support(indices=False)) == X.shape[1]
        assert sum(TestCls_2SHOT.get_support(indices=False)) == TRFM_X_2SHOT.shape[1]

        assert np.array_equal(
            _adj_get_support_mct,
            TestCls_2SHOT.get_support(indices=False)
        )

        del _adj_get_support_mct

        assert TRFM_X_2.shape == TRFM_X_2SHOT.shape
        assert np.array_equiv(
            TRFM_X_2[np.logical_not(nan_mask(TRFM_X_2))],
            TRFM_X_2SHOT[np.logical_not(nan_mask(TRFM_X_2SHOT))]
        ), f'1X2rcr X output != 2X1rcr X output'

        assert TRFM_Y_2.shape == TRFM_Y_2SHOT.shape
        assert np.array_equiv(TRFM_Y_2, TRFM_Y_2SHOT), \
            f'1X2rcr Y output != 2X1rcr Y output'

        del TRFM_X_2SHOT, TRFM_Y_2SHOT
        # END compare 2 one-shot MCTs against 1 two-shot MCT - -- - -- - -- - --

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # BUILD MOCK_X & MOCK_Y #
        # mmct first recursion
        mmct_first_rcr = mmct()   # give class a name to access attr later
        MOCK_X1, MOCK_Y1 = mmct_first_rcr.trfm(
            X.copy(), y.copy(), _ignore_columns, ignore_nan,
            ignore_non_binary_integer_columns, ignore_float_columns,
            _handle_as_bool, delete_axis_0, count_threshold
        )

        # validate mmct 1rcrX1 get_support and object dimensions make sense
        assert len(mmct_first_rcr.get_support_) == X.shape[1]
        assert MOCK_X1.shape[1] == sum(mmct_first_rcr.get_support_)


        # ** * ** * ** * ** * ** *
        # compare MCT 1rcrX1 output against mmct 1rcrX1 output

        assert np.array_equal(
            TestCls1.get_support(indices=False),
            mmct_first_rcr.get_support_
        )

        assert TRFM_X_1.shape == MOCK_X1.shape
        assert np.array_equiv(
            TRFM_X_1[np.logical_not(nan_mask(TRFM_X_1))],
            MOCK_X1[np.logical_not(nan_mask(MOCK_X1))]
        ), f'X output for 1X PASSES THRU TestCls WITH max_recursions=1 FAILED'

        assert TRFM_Y_1.shape == MOCK_Y1.shape
        assert np.array_equiv(TRFM_Y_1, MOCK_Y1), \
            (f'y output for 1X PASSES THRU TestCls WITH max_recursions=1 FAILED')

        del TRFM_X_1, TRFM_Y_1

        # ** * ** * ** * ** * ** *

        # ADJUST ign_columns & handle_as_bool FOR 2ND RECURSION ** ** ** **
        # USING mmct get_support
        NEW_COLUMN_MASK = mmct_first_rcr.get_support_

        OG_IGN_COL_MASK = np.zeros(X.shape[1]).astype(bool)
        OG_IGN_COL_MASK[_ignore_columns] = True
        mmct_scd_ignore_columns = \
            np.arange(sum(NEW_COLUMN_MASK))[OG_IGN_COL_MASK[NEW_COLUMN_MASK]]
        if _ignore_columns is None or len(mmct_scd_ignore_columns) == 0:
            mmct_scd_ignore_columns = None

        OG_H_A_B_MASK = np.zeros(x_cols).astype(bool)
        OG_H_A_B_MASK[_handle_as_bool] = True
        mmct_scd_handle_as_bool = \
            np.arange(sum(NEW_COLUMN_MASK))[OG_H_A_B_MASK[NEW_COLUMN_MASK]]
        if _handle_as_bool is None or len(mmct_scd_handle_as_bool) == 0:
            mmct_scd_handle_as_bool = None

        del NEW_COLUMN_MASK, OG_IGN_COL_MASK, OG_H_A_B_MASK
        # # END ADJUST ign_columns & handle_as_bool FOR 2ND RECURSION ** ** **

        # compare mmct_scd_ignore_columns to mct_scd_ignore_columns,
        # mmct_scd_handle_as_bool to mmct_scd_handle_as_bool
        assert np.array_equal(mmct_scd_ignore_columns, mct_scd_ignore_columns)
        assert np.array_equal(mmct_scd_handle_as_bool, mct_scd_handle_as_bool)

        # ** * ** * ** * ** * ** * **
        # mmct 2nd recursion
        mmct_scd_rcr = mmct()   # give class a name to access attr later
        MOCK_X2, MOCK_Y2 = mmct_scd_rcr.trfm(
            MOCK_X1.copy(), MOCK_Y1.copy(), mmct_scd_ignore_columns, ignore_nan,
            ignore_non_binary_integer_columns, ignore_float_columns,
            mmct_scd_handle_as_bool, delete_axis_0, count_threshold
        )

        # validate mmct 1rcrX2 get_support and object dimensions make sense
        assert len(mmct_scd_rcr.get_support_) == MOCK_X1.shape[1]
        assert MOCK_X2.shape[1] == sum(mmct_scd_rcr.get_support_)

        del MOCK_X1, MOCK_Y1, mmct_scd_ignore_columns, mmct_scd_handle_as_bool

        # END BUILD MOCK_X & MOCK_Y #
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # vvvv *************************************************
        # where <function array_equiv> 2133: AssertionError

        # compare MCT 1rcrX2 output against mmct 1rcrX2 output
        _adj_get_support_mct = np.array(TestCls1.get_support(indices=False).copy())
        _idxs = np.arange(len(_adj_get_support_mct))[_adj_get_support_mct]
        _adj_get_support_mct[_idxs] = TestCls2.get_support(indices=False)
        del _idxs

        _adj_get_support_mmct = np.array(mmct_first_rcr.get_support_.copy())
        _idxs = np.arange(len(_adj_get_support_mmct))[_adj_get_support_mmct]
        _adj_get_support_mmct[_idxs] = mmct_scd_rcr.get_support_
        del _idxs

        assert np.array_equal(_adj_get_support_mct, _adj_get_support_mmct)

        del _adj_get_support_mct, _adj_get_support_mmct


        assert TRFM_X_2.shape == MOCK_X2.shape
        assert np.array_equiv(
            TRFM_X_2[np.logical_not(nan_mask(TRFM_X_2))],
            MOCK_X2[np.logical_not(nan_mask(MOCK_X2))]
        ), f'X output for 2X PASSES THRU TestCls WITH max_recursions=1 FAILED'

        assert TRFM_Y_2.shape == MOCK_Y2.shape
        assert np.array_equiv(TRFM_Y_2, MOCK_Y2), \
            (f'y output for 2X PASSES THRU TestCls WITH max_recursions=1 FAILED')

        del TRFM_X_2, TRFM_Y_2
        del _ignore_columns, _handle_as_bool

# END TEST ACCURACY ****************************************************



# TEST MANY PARTIAL FITS == ONE BIG FIT ********************************
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestManyPartialFitsEqualOneBigFit:

    def test_many_partial_fits_equal_one_big_fit(
        self, X, y, _args, _kwargs, x_rows
    ):

        # TEST THAT ONE-SHOT partial_fit() / transform() ==
        # ONE-SHOT fit() / transform() ** ** ** ** ** ** ** ** ** ** **
        OneShotPartialFitTestCls = MinCountTransformer(*_args, **_kwargs)
        OneShotPartialFitTestCls.partial_fit(X, y)
        ONE_SHOT_PARTIAL_FIT_TRFM_X, ONE_SHOT_PARTIAL_FIT_TRFM_Y = \
            OneShotPartialFitTestCls.transform(X, y)

        OneShotFullFitTestCls = MinCountTransformer(*_args, **_kwargs)
        OneShotFullFitTestCls.partial_fit(X, y)
        ONE_SHOT_FULL_FIT_TRFM_X, ONE_SHOT_FULL_FIT_TRFM_Y = \
            OneShotFullFitTestCls.transform(X, y)

        assert np.array_equiv(
                ONE_SHOT_PARTIAL_FIT_TRFM_X.astype(str),
                ONE_SHOT_FULL_FIT_TRFM_X.astype(str)
            ), f"one shot partial fit trfm X != one shot full fit trfm X"

        assert np.array_equiv(
                ONE_SHOT_PARTIAL_FIT_TRFM_Y,
                ONE_SHOT_FULL_FIT_TRFM_Y
            ), f"one shot partial fit trfm Y != one shot full fit trfm Y"
        # END TEST THAT ONE-SHOT partial_fit() / transform() ==
        # ONE-SHOT fit() / transform() ** ** ** ** ** ** ** ** ** ** **

        # TEST PARTIAL FIT COUNTS ARE DOUBLED WHEN FULL DATA IS partial_fit() 2X
        SingleFitTestClass = MinCountTransformer(*_args, **_kwargs)
        SingleFitTestClass.fit(X, y)
        CT1 = SingleFitTestClass._total_counts_by_column

        DoublePartialFitTestClass = MinCountTransformer(*_args, **_kwargs)
        DoublePartialFitTestClass.partial_fit(X, y)
        DoublePartialFitTestClass.partial_fit(X, y)
        CT2 = DoublePartialFitTestClass._total_counts_by_column

        # convert keys to strs to deal with nans
        CT1 = {i:{str(k):v for k,v in _.items()} for i,_ in CT1.items()}
        CT2 = {i:{str(k):v for k,v in _.items()} for i,_ in CT2.items()}
        for _c_idx in CT1:
            for _unq in CT1[_c_idx]:
                assert CT2[_c_idx][_unq] == 2 * CT1[_c_idx][_unq]
        del CT1, CT2
        # END TEST PARTIAL FIT COUNTS ARE DOUBLED WHEN FULL DATA IS partial_fit() 2X

        # STORE CHUNKS TO ENSURE THEY STACK BACK TO THE ORIGINAL X/y
        _chunks = 5
        X_CHUNK_HOLDER = []
        Y_CHUNK_HOLDER = []
        for row_chunk in range(_chunks):
            MASK1 = row_chunk * x_rows // _chunks
            MASK2 = (row_chunk + 1) * x_rows // _chunks
            X_CHUNK_HOLDER.append(X[MASK1:MASK2, :])
            Y_CHUNK_HOLDER.append(y[MASK1:MASK2, :])
        del MASK1, MASK2

        assert np.array_equiv(
            np.vstack(X_CHUNK_HOLDER).astype(str), X.astype(str)
            ), \
            f"agglomerated X chunks != original X"
        assert np.array_equiv(np.vstack(Y_CHUNK_HOLDER), y), \
            f"agglomerated Y chunks != original Y"

        PartialFitPartialTrfmTestCls = MinCountTransformer(*_args, **_kwargs)
        PartialFitOneShotTrfmTestCls = MinCountTransformer(*_args, **_kwargs)
        OneShotFitTransformTestCls = MinCountTransformer(*_args, **_kwargs)

        # PIECEMEAL PARTIAL FIT
        for X_CHUNK, Y_CHUNK in zip(X_CHUNK_HOLDER, Y_CHUNK_HOLDER):
            PartialFitPartialTrfmTestCls.partial_fit(X_CHUNK, Y_CHUNK)
            PartialFitOneShotTrfmTestCls.partial_fit(X_CHUNK, Y_CHUNK)

        # PIECEMEAL TRANSFORM ******************************************
        # THIS MUST BE IN ITS OWN LOOP, ALL FITS MUST BE DONE BEFORE
        # DOING ANY TRFMS
        PARTIAL_TRFM_X_HOLDER = []
        PARTIAL_TRFM_Y_HOLDER = []
        for X_CHUNK, Y_CHUNK in zip(X_CHUNK_HOLDER, Y_CHUNK_HOLDER):
            PARTIAL_TRFM_X, PARTIAL_TRFM_Y = \
                PartialFitPartialTrfmTestCls.transform(X_CHUNK, Y_CHUNK)
            PARTIAL_TRFM_X_HOLDER.append(PARTIAL_TRFM_X)
            PARTIAL_TRFM_Y_HOLDER.append(PARTIAL_TRFM_Y)

        del PartialFitPartialTrfmTestCls, PARTIAL_TRFM_X, PARTIAL_TRFM_Y

        # AGGLOMERATE PARTIAL TRFMS FROM PARTIAL FIT
        FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM = \
            np.vstack(PARTIAL_TRFM_X_HOLDER)
        FULL_TRFM_Y_FROM_PARTIAL_FIT_PARTIAL_TRFM = \
            np.vstack(PARTIAL_TRFM_Y_HOLDER)
        # END PIECEMEAL TRANSFORM **************************************

        # DO ONE-SHOT TRANSFORM OF X,y ON THE PARTIALLY FIT INSTANCE
        _, __ = PartialFitOneShotTrfmTestCls.transform(X, y)
        FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM = _
        FULL_TRFM_Y_FROM_PARTIAL_FIT_ONESHOT_TRFM = __
        del _, __

        # ONE-SHOT FIT TRANSFORM
        FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM, FULL_TRFM_Y_ONE_SHOT_FIT_TRANSFORM = \
            OneShotFitTransformTestCls.fit_transform(X, y)

        # ASSERT ALL AGGLOMERATED X AND Y TRFMS ARE EQUAL
        assert np.array_equiv(
                FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM.astype(str),
                FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM.astype(str)
            ), \
            f"compiled trfm X from partial fit / partial trfm != one-shot fit/trfm X"

        assert np.array_equiv(
                FULL_TRFM_Y_ONE_SHOT_FIT_TRANSFORM,
                FULL_TRFM_Y_FROM_PARTIAL_FIT_PARTIAL_TRFM
            ), \
            f"compiled trfm y from partial fit / partial trfm != one-shot fit/trfm y"

        assert np.array_equiv(
            FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM.astype(str),
            FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM.astype(str)
            ), f"trfm X from partial fits / one-shot trfm != one-shot fit/trfm X"

        assert np.array_equiv(
            FULL_TRFM_Y_ONE_SHOT_FIT_TRANSFORM,
            FULL_TRFM_Y_FROM_PARTIAL_FIT_ONESHOT_TRFM
            ), f"trfm y from partial fits / one-shot trfm != one-shot fit/trfm y"

# END TEST MANY PARTIAL FITS == ONE BIG FIT ****************************


# TEST LATER PARTIAL FITS ACCEPT NEW UNIQUES ***************************
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestLaterPartialFitsAcceptNewUniques:

    def test_later_partial_fits_accept_new_uniques(
        self, NO_NAN_X, y, _args, _kwargs, _mct_cols
    ):

        X1 = NO_NAN_X[:, _mct_cols:(2 * _mct_cols)].copy()  # non-bin-int columns
        X1 = X1.astype(np.float64).astype(np.int32)
        y1 = y.copy()
        # 10X THE VALUES IN THE COPY OF DATA TO INTRODUCE NEW UNIQUE VALUES
        X2 = (10 * X1.astype(np.float64)).astype(np.int32)
        y2 = y.copy()

        STACKED_X = np.vstack((X1, X2)).astype(np.float64).astype(np.int32)
        STACKED_Y = np.vstack((y1, y2)).astype(np.uint8)

        args = [3 * _args[0]]
        _kwargs['ignore_non_binary_integer_columns'] = False

        # 2 PARTIAL FITS - - -  - - - - - - - - - - - - - - - - - - - - -
        PartialFitTestCls = MinCountTransformer(*args, **_kwargs)

        PartialFitTestCls.partial_fit(X1, y1)
        PartialFitTestCls.partial_fit(X2, y2)
        PARTIAL_FIT_X, PARTIAL_FIT_Y = \
            PartialFitTestCls.transform(STACKED_X, STACKED_Y)

        assert not PARTIAL_FIT_X.shape[0] == 0, \
            f'transform for 2 partial fits deleted all rows'

        # VERIFY SOME ROWS WERE ACTUALLY DELETED
        assert not np.array_equiv(PARTIAL_FIT_X, STACKED_X), \
            (f'later partial fits accept new uniques --- '
             f'transform did not delete any rows')
        # END 2 PARTIAL FITS - - -  - - - - - - - - - - - - - - - - - - -

        # 1 BIG FIT - - -  - - - - - - - - - - - - - - - - - - - - - - -
        SingleFitTestCls = MinCountTransformer(*args, **_kwargs)
        SingleFitTestCls.fit(STACKED_X, STACKED_Y)
        SINGLE_FIT_X, SINGLE_FIT_Y = \
            SingleFitTestCls.transform(STACKED_X, STACKED_Y)

        assert not SINGLE_FIT_X.shape[0] == 0, \
            f'transform for one big fit deleted all rows'
        # END 1 BIG FIT - - -  - - - - - - - - - - - - - - - - - - - - -

        # compare 2 partial fits to 1 big fit, should be equal
        assert np.array_equiv(PARTIAL_FIT_X, SINGLE_FIT_X), \
            (f"new uniques in partial fits -- partial fitted X does not "
             f"equal single fitted X")
        assert np.array_equiv(PARTIAL_FIT_Y, SINGLE_FIT_Y), \
            (f"new uniques in partial fits -- partial fitted y does not "
             f"equal single fitted y")


# END TEST LATER PARTIAL FITS ACCEPT NEW UNIQUES ***********************


# TEST TRANSFORM CONDITIONALLY ACCEPTS NEW UNIQUES *********************
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestTransformConditionallyAcceptsNewUniques:

    def test_transform_conditionally_accepts_new_uniques(
        self, NO_NAN_X, y, _args, _kwargs, _mct_cols, x_rows
    ):

        # USE STR COLUMNS
        X1 = NO_NAN_X[:, (3 * _mct_cols):(4 * _mct_cols)].copy()
        y1 = y.copy()

        # fit() & transform() ON X1 TO PROVE X1 PASSES transform()
        TestCls = MinCountTransformer(2*_args[0], **_kwargs)
        TestCls.fit(X1, y1)
        OUT_X, OUT_Y = TestCls.transform(X1, y1)
        assert OUT_X.shape[0] > 0
        del TestCls, OUT_X, OUT_Y

        # PEPPER ONE OF THE STR COLUMNS WITH A UNIQUE THAT WAS NOT SEEN DURING fit()
        X2 = X1.copy()
        new_unqs = list('1234567890')
        MASK = np.random.choice(range(x_rows), len(new_unqs), replace=False)
        # put str(number) into str column of alphas
        X2[MASK, 0] = new_unqs
        del new_unqs, MASK
        TestCls = MinCountTransformer(2*_args[0], **_kwargs)
        TestCls.fit(X1, y1)

        # DEMONSTRATE NEW VALUES ARE ACCEPTED WHEN reject_unseen_values = False
        TestCls.set_params(reject_unseen_values=False)
        assert TestCls.reject_unseen_values is False
        TestCls.transform(X2, y1)

        # DEMONSTRATE NEW VALUES ARE REJECTED WHEN reject_unseen_values = True
        TestCls.set_params(reject_unseen_values=True)
        assert TestCls.reject_unseen_values is True
        with pytest.raises(ValueError):
            TestCls.transform(X2, y1)

        del X1, y1, X2, TestCls


# END TEST TRANSFORM CONDITIONALLY ACCEPT NEW UNIQUES ******************













