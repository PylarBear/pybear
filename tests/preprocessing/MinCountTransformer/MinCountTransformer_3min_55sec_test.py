# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
import uuid

import numpy as np
np.random.seed(1)
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError
import dask.array as da
import dask.dataframe as ddf
import dask_expr._collection as ddf2
from dask_ml.wrappers import Incremental, ParallelPostFit

from pybear.preprocessing.MinCountTransformer.MinCountTransformer import \
    MinCountTransformer




bypass = False


# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
# SET X, y DIMENSIONS AND DEFAULT THRESHOLD (_args) FOR TESTING MCT

@pytest.fixture(scope='session')
def _mct_rows():
    # _mct_rows must be between 50 and 750
    # this is fixed, all MCT test (not mmct test) objects have this many rows
    # (mmct rows is set by the construction parameters when a suitable set
    # of vectors for building mmct is found, remember)
    return 200


@pytest.fixture(scope='session')
def _mct_cols():
    # _mct_cols must be > 0
    # this sets the number of columns for each data type! not the total
    # number of columns in X! See the logic inside build_test_objects_for_MCT
    # to get how many columns are actually returned. That number is held
    # in fixture 'x_cols'.
    return 2


@pytest.fixture(scope='function')
def y_rows(x_rows):
    return x_rows


@pytest.fixture(scope='function')
def y_cols():
    return 2


@pytest.fixture(scope='function')
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
        'n_jobs': -1
    }

# END SET X, y DIMENSIONS AND DEFAULT THRESHOLD (_args) FOR TESTING MCT
# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^




# build X, NO_NAN_X, DTYPE_KEY, x_rows, x_cols for MCT test (not mmct test!)

# build_test_objects_for_MCT builds this objects in one shot in conftest

@pytest.fixture(scope='function')
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
        _X = np.random.randint(0, 2, (_mct_rows, _mct_cols)).astype(object)
        # CREATE _mct_cols COLUMNS OF NON-BINARY INTEGERS
        _X = np.hstack((
            _X, np.random.randint(
                0, _mct_rows // 15, (_mct_rows, _mct_cols)
            ).astype(object)
        ))
        # CREATE _mct_cols COLUMNS OF FLOATS
        _X = np.hstack((
            _X, np.random.uniform(0, 1, (_mct_rows, _mct_cols)).astype(object)
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

        _DTYPE_KEY = [k for k in ['int', 'int', 'float', 'obj'] for j in
                      range(_mct_cols)]
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
            _X[_get_idxs(), 2 * _mct_cols + idx] = np.random.uniform(0, 1) + idx
            _X[_get_idxs(), 2 * _mct_cols + idx] = np.random.uniform(0, 1) + idx
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
                False,
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
        _X1 = mmct().trfm(_X, None, None, False, False, False, False, True, 3)
        # MOCK_X, MOCK_Y, ign_col, ign_nan, ignore_non_binary_int_col,
        # ignore_flt_col, handle_as_bool, delete_axis_0, ct_thresh
        try:
            # THIS SHOULD EXCEPT IF ALL ROWS/COLUMNS WOULD BE DELETED
            _X2 = mmct().trfm(_X1, None, None, False, False, False, False, True, 3)
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


@pytest.fixture(scope='function')
def X(build_test_objects_for_MCT):
    return build_test_objects_for_MCT[0]


@pytest.fixture(scope='function')
def NO_NAN_X(build_test_objects_for_MCT):
    return build_test_objects_for_MCT[1]


@pytest.fixture(scope='function')
def DTYPE_KEY(build_test_objects_for_MCT):
    return build_test_objects_for_MCT[2]


@pytest.fixture(scope='function')
def x_rows(build_test_objects_for_MCT):
    return build_test_objects_for_MCT[3]


@pytest.fixture(scope='function')
def x_cols(build_test_objects_for_MCT):
    return build_test_objects_for_MCT[4]


@pytest.fixture(scope='function')
def COLUMNS(x_cols):
    return [str(uuid.uuid4())[:4] for _ in range(x_cols)]

# END build X, NO_NAN_X, DTYPE_KEY, x_rows, x_cols for MCT test (not mmct test!)


# Build y for MCT tests (not mmct test!) ** * ** * ** * ** * ** * ** * **

@pytest.fixture(scope='function')
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

        # should not except
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

    JUNK_CT_THRESH = BASELINE_JUNK.copy() + ['junk']
    JUNK_RECURSIONS = BASELINE_JUNK.copy() + ['junk']
    JUNK_OUTPUT_TYPE = BASELINE_JUNK.copy()
    JUNK_OUTPUT_TYPE.remove(None)
    JUNK_N_JOBS = BASELINE_JUNK.copy()
    JUNK_N_JOBS.remove(None)
    JUNK_N_JOBS += ['junk']


    @pytest.mark.parametrize('junk_ct_thresh', JUNK_CT_THRESH)
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

    BAD_CT_THRESH = [-2, 1, 100_000_000]
    BAD_RECURSIONS = [-1, 0]
    BAD_OUTPUT_TYPE = ['dask_array', 'wrong_junk']
    BAD_N_JOBS = [-2, 0]


    @pytest.mark.parametrize('bad_ct_thresh', BAD_CT_THRESH)
    def test_bad_ct_thresh(self, X, y, _kwargs, bad_ct_thresh):

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


    GOOD_CT_THRESH = [3, 5]
    GOOD_RECURSIONS = [1, 10]
    GOOD_OUTPUT_TYPE = [
        None, 'default', 'NUMPY_ARRAY', 'Pandas_Dataframe', 'pandas_series'
    ]
    GOOD_N_JOBS = [-1, 1, 10, None]


    @pytest.mark.parametrize('good_ct_thresh', GOOD_CT_THRESH)
    def test_good_ct_thresh(self, X, y, _kwargs, good_ct_thresh):

        MinCountTransformer(good_ct_thresh, **_kwargs).fit_transform(X, y)


    @pytest.mark.parametrize('good_recursions', GOOD_RECURSIONS)
    def test_good_recursions(self, X, y, _args, _kwargs, good_recursions):

        _kwargs['max_recursions'] = good_recursions

        MinCountTransformer(*_args, **_kwargs).fit_transform(X, y)


    @pytest.mark.parametrize('good_n_jobs', GOOD_N_JOBS)
    def test_good_n_jobs(self, X, y, _args, _kwargs, good_n_jobs):

        _kwargs['n_jobs'] = good_n_jobs

        MinCountTransformer(*_args, **_kwargs).fit_transform(X, y)


    @pytest.mark.parametrize('good_output_type', GOOD_OUTPUT_TYPE)
    def test_good_output_type(self, X, y, _args, _kwargs, good_output_type):

        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.set_output(transform=good_output_type)
        TestCls.fit_transform(X[:, 0], y[:, 0])


# END TEST ACCEPTS GOOD count_threshold, max_recursions, set_output, n_jobs


# TEST FOR GOOD / BAD ignore_columns / handle_as_bool ##################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestIgnoreColumnsHandleAsBool:

    # both can take list of int, list of str, callable, or None

    @staticmethod
    @pytest.fixture
    def _y(y):
        return y[:, 0].copy()


    @pytest.mark.parametrize('input_format',
        ('numpy_recarray', 'numpy_masked_array')
    )
    @pytest.mark.parametrize('kwarg_input',
        ([1, 3, 5], [True, False, None, 'junk'], True, False, None, 'junk',
         np.nan, [np.nan], {0: 'a', 1: 'b'}, [1000, 1001, 1002]
         )
    )
    @pytest.mark.parametrize('_kwarg',
        ('ignore_columns', 'handle_as_bool', 'both')
    )
    def test_rec_arrays_and_masked_arrays(self, input_format, kwarg_input,
        _kwarg, x_rows, x_cols, X, COLUMNS, _y, _args, _kwargs
    ):

        # this pulled out of the old big tests
        # this actually tests X & y inputs (coincidentally with various
        # ignore_columns & handle as bool inputs.)

        if _kwarg == 'ignore_columns':
            _kwargs['ignore_columns'] = kwarg_input
            _kwargs['handle_as_bool'] = None
        elif _kwarg == 'handle_as_bool':
            _kwargs['handle_as_bool'] = kwarg_input
            _kwargs['ignore_columns'] = None
        elif _kwarg == 'both':
            _kwargs['ignore_columns'] = kwarg_input
            _kwargs['handle_as_bool'] = kwarg_input

        if input_format == 'numpy_recarray':
            _dtypes1 = [np.uint8 for _ in range(x_cols // 2)]
            _dtypes2 = ['<U1' for _ in range(x_cols // 2)]
            _formats = [list(zip(COLUMNS, _dtypes1 + _dtypes2))]
            del _dtypes1, _dtypes2
            X_NEW = np.recarray((x_rows,), names=COLUMNS, formats=_formats, buf=X)
            del _formats
            y_NEW = _y
        elif input_format == 'numpy_masked_array':
            X_NEW = np.ma.array(X, mask=False)
            y_NEW = np.ma.array(_y, mask=False)

        with pytest.raises(TypeError):
            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.fit_transform(X_NEW, y_NEW)



    @pytest.mark.parametrize('input_format', ('numpy', 'pd_df', 'pd_series'))
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
        elif input_format == 'pd_series':
            if kwarg_input == 'bad_callable':
                _kwargs['ignore_columns'] = lambda X: 'unrecognizable junk'
                _kwargs['handle_as_bool'] = lambda X: 'unrecognizable junk'
            X_NEW = pd.Series(data=X[:, _mct_cols], name=COLUMNS[0], dtype=np.float64)
            y_NEW = pd.Series(data=_y, name='y', dtype=object)

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


    @pytest.mark.parametrize('input_format', ('numpy', 'pd_df', 'pd_series'))
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
            if input_format == 'pd_series':
                pytest.skip(reason=f"columns are out of range for a series")
            else:
                kwarg_input = list(range(_mct_cols, 2*_mct_cols))

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
                NON_BIN_INT_COLS = [COLUMNS[_] for _ in range(_mct_cols, 2*_mct_cols)]
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
                NON_BIN_INT_COLS = [COLUMNS[_] for _ in range(_mct_cols, 2 * _mct_cols)]
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


# ALWAYS ACCEPTS y==None IS PASSED TO fit(), partial_fit(), AND transform() #######
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
        value_error += row_checker(fst_fit_Y, x_rows)
        value_error += row_checker(scd_fit_Y, x_rows)
        value_error += row_checker(trfm_Y, x_rows)
        # END True only if y rows != X rows ** ** ** ** ** ** ** ** ** *

        if value_error:
            with pytest.raises(ValueError):
                TestCls.partial_fit(X, fst_fit_Y)
                TestCls.partial_fit(X, scd_fit_Y)
                TestCls.transform(X, trfm_Y)

        elif not value_error:
            TestCls.partial_fit(X, fst_fit_Y)
            TestCls.partial_fit(X, scd_fit_Y)
            TestCls.transform(X, trfm_Y)

    del FIRST_FIT_y_DTYPE, SECOND_FIT_y_DTYPE, SECOND_FIT_y_DIFF_ROWS
    del TRANSFORM_y_DTYPE, TRANSFORM_y_DIFF_ROWS

# END TEST FOR EXCEPTS ON BAD y ROWS FOR DF, SERIES, ARRAY #############


# TEST EXCEPTS ANYTIME X==None IS PASSED TO fit(), partial_fit(), AND transform()
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestExceptsAnytimeXisNone:

    def test_excepts_anytime_x_is_none(self, X, _args, _kwargs):

        with pytest.raises(TypeError):
            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.fit(None)


        with pytest.raises(TypeError):
            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.partial_fit(X)
            TestCls.partial_fit(None)

        with pytest.raises(TypeError):
            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.fit(X)
            TestCls.partial_fit(None)

        with pytest.raises(TypeError):
            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.fit(X)
            TestCls.transform(None)

        with pytest.raises(TypeError):
            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.fit_transform(None)

        del TestCls

# END TEST EXCEPTS ANYTIME X==None IS PASSED TO fit(), partial_fit(), OR transform()


# VERIFY ACCEPTS X AS SINGLE COLUMN / SERIES ##################################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAcceptsXAsSingleColumnOrSeries:

    @staticmethod
    @pytest.fixture
    def _fst_fit_y(y):
        return y[:, 0].copy()

    @staticmethod
    @pytest.fixture
    def NEW_X(X):
        return X[:, 0].copy()

    @pytest.mark.parametrize('_fst_fit_x_dtype',
        ['numpy', 'pandas_dataframe', 'pandas_series']
    )
    @pytest.mark.parametrize('_fst_fit_x_hdr', [True, None])
    def test_X_as_single_column(self, _args, _kwargs, COLUMNS, _fst_fit_y, NEW_X,
        _fst_fit_x_dtype, _fst_fit_x_hdr):

        if _fst_fit_x_dtype == 'numpy':
            if _fst_fit_x_hdr:
                pytest.skip(reason=f"numpy cannot have header")
            else:
                _fst_fit_X = NEW_X.copy()

        if 'pandas' in _fst_fit_x_dtype:
            if _fst_fit_x_hdr:
                _fst_fit_X = pd.DataFrame(data=NEW_X, columns=COLUMNS[:1])
            else:
                _fst_fit_X = pd.DataFrame(data=NEW_X)

        # not elif!
        if _fst_fit_x_dtype == 'pandas_series':
            _fst_fit_X = _fst_fit_X.squeeze()

        TestCls = MinCountTransformer(*_args, **_kwargs)

        TestCls.fit_transform(_fst_fit_X, _fst_fit_y)

# END VERIFY ACCEPTS X AS SINGLE COLUMN / SERIES ##############################


# TEST FOR EXCEPTS ON BAD X SHAPES ON SERIES & ARRAY ##########################
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestExceptsOnBadXShapes:

    FIRST_FIT_X_DTYPE = list(reversed(['numpy', 'pandas_series']))
    SECOND_FIT_X_DTYPE = list(reversed(['numpy', 'pandas_series']))
    SECOND_FIT_X_SAME_DIFF_COLUMNS = \
        list(reversed(['good', 'less_col', 'more_col']))
    SECOND_FIT_X_SAME_DIFF_ROWS = list(reversed(['good', 'less_row', 'more_row']))
    TRANSFORM_X_DTYPE = list(reversed(['numpy', 'pandas_series']))
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
                if new_dtype == 'pandas_series':
                    NEW_X = NEW_X.iloc[:, 0].squeeze()

            return NEW_X

        return foo


    # end X_builder ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

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

            # CANT HAVE 'more_col' or 'less_col' WHEN X IS A SERIES
            _reason = f"CANT HAVE 'more_col' or 'less_col' WHEN X IS A SERIES"
            if scd_fit_x_dtype == 'pandas_series' and \
                    scd_fit_x_cols in ['less_col', 'more_col']:
                pytest.skip(reason=_reason)
            if trfm_x_dtype == 'pandas_series' and \
                    trfm_x_cols in ['less_col', 'more_col']:
                pytest.skip(reason=_reason)

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

            # ValueError WHEN ROWS OF y != X ROWS UNDER ALL CIRCUMSTANCES
            if fst_fit_X is not None and fst_fit_y.shape[0] != fst_fit_X.shape[0]:
                value_error += 1
            if scd_fit_X is not None and scd_fit_y.shape[0] != scd_fit_X.shape[0]:
                value_error += 1
            if trfm_X is not None and trfm_y.shape[0] != trfm_X.shape[0]:
                value_error += 1

            # ValueError WHEN n_features_in_ != FIRST FIT n_features_in_
            # UNDER ALL OTHER CIRCUMSTANCES
            value_error += True in [__ in ['more_col', 'less_col'] for __ in
                                    [scd_fit_x_cols, trfm_x_cols]]

            # ValueError WHEN COLUMNS PASSED TO
            # i) A LATER {partial_}fit() OR
            # 2) TRANSFORM DO NOT MATCH COLUMNS SEEN ON ANY PREVIOUS
            # {partial_}fit() COLUMNS CANNOT BE "BAD" WHEN SEEN FOR THE
            # FIRST TIME

            if scd_fit_X is None:
                pass
            elif scd_fit_X.shape != fst_fit_X.shape:
                value_error += 1
            if trfm_X is None:
                pass
            elif trfm_X.shape != fst_fit_X.shape:
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

        value_error = 0

        value_error += (scd_fit_name != fst_fit_name)
        value_error += (trfm_name != fst_fit_name)
        value_error += (trfm_name != scd_fit_name)

        if value_error == 0:
            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.partial_fit(fst_fit_X)
            TestCls.partial_fit(scd_fit_X)
            TestCls.transform(trfm_X)

            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.fit(fst_fit_X)
            TestCls.transform(trfm_X)

            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.fit_transform(fst_fit_X)  # SHOULD NOT EXCEPT

        elif value_error != 0:
            with pytest.raises(ValueError):
                TestCls = MinCountTransformer(*_args, **_kwargs)
                TestCls.partial_fit(fst_fit_X)
                TestCls.partial_fit(scd_fit_X)
                TestCls.transform(trfm_X)

                TestCls = MinCountTransformer(*_args, **_kwargs)
                TestCls.fit(fst_fit_X)
                TestCls.transform(trfm_X)

                TestCls = MinCountTransformer(*_args, **_kwargs)
                TestCls.fit_transform(fst_fit_X)  # SHOULD NOT EXCEPT

    del NAMES

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

        # ValueError WHEN ROWS OF y (IF PASSED) != X ROWS UNDER ALL CIRCUMSTANCES
        if fst_fit_X is not None and fst_fit_y.shape[0] != fst_fit_X.shape[0]:
            value_error += 1
        if scd_fit_X is not None and scd_fit_y.shape[0] != scd_fit_X.shape[0]:
            value_error += 1
        if trfm_X is not None and trfm_y.shape[0] != trfm_X.shape[0]:
            value_error += 1

        # THESE TRIGGER ERROR FROM feature_names_in_
        value_error += (scd_fit_x_cols != 'good')
        value_error += (trfm_x_cols != 'good')
        value_error += (trfm_x_cols != scd_fit_x_cols)
        # END THESE TRIGGER ERROR FROM feature_names_in_

        # ValueError WHEN n_features_in_ != FIRST FIT n_features_in_
        # UNDER ALL OTHER CIRCUMSTANCES
        _ = ['more_col', 'less_col']
        __ = [scd_fit_x_cols, trfm_x_cols]
        value_error += any([i in _ for i in __])
        del _, __

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
        TestCls = MinCountTransformer(*_args, **_kwargs)

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

        if TestCls._ignore_nan == True:
            # 2.5's SHOULD BE DELETED, BUT NOT nan
            if nan_type == 'str_nan':
                MASK = (TEST_X != 2.5).astype(bool)
            elif nan_type == 'np_nan':
                MASK = (TEST_X != 2.5).astype(bool)
        elif TestCls._ignore_nan == False:
            # 2.5's AND nan SHOULD BE DELETED
            if nan_type == 'str_nan':
                MASK = ((TEST_X != 2.5) * (TEST_X.astype(str) != f'{np.nan}'))
            elif nan_type == 'np_nan':
                MASK = ((TEST_X != 2.5) * np.logical_not(np.isnan(TEST_X)))

        REF_X = TEST_X[MASK]
        REF_Y = TEST_Y[MASK]

        assert len(REF_X) == correct_x_and_y_len, \
            f"REF_X is not the correct length"

        assert len(REF_Y) == correct_x_and_y_len, \
            f"REF_X is not the correct length"

        del correct_x_and_y_len

        def _formatter(ARRAY_1, MASK=None):
            if MASK is None:
                MASK = [True for _ in ARRAY_1.ravel()]
            return ARRAY_1[MASK].ravel().astype(np.float64)


        if has_nan and TestCls._ignore_nan == True:
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

        with pytest.raises(ValueError):
            TestCls.set_params(handle_as_bool=[x_cols - 1])  # STR COLUMN

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

        # USE FLOAT AND STR COLUMNS, DUMMY THE STRS AND RUN delete_axis_0
        # True, DO THE SAME WHEN STRS ARE NOT DUMMIED, BUT ONLY USE FLOAT
        # COLUMNS TO SEE IF THE RESULTS ARE EQUAL; PROVE NO ROWS ARE
        # DELETED FROM DUMMIED WHEN delete_axis_0 is False
        FLOAT_STR_X = NO_NAN_X[:, 2 * _mct_cols:4 * _mct_cols].copy()
        FLOAT_STR_COLUMNS = COLUMNS[2 * _mct_cols:4 * _mct_cols].copy()
        FLOAT_STR_DF = pd.DataFrame(
            data=FLOAT_STR_X,
            columns=FLOAT_STR_COLUMNS
        )
        FLOAT_DF = pd.DataFrame(
            data=FLOAT_STR_X[:, :_mct_cols],
            columns=FLOAT_STR_COLUMNS[:_mct_cols]
        )  # "TRUTH" for when delete_axis_0 = False
        STR_DF = pd.DataFrame(
            data=FLOAT_STR_X[:, _mct_cols:],
            columns=FLOAT_STR_COLUMNS[_mct_cols:]
        )
        del FLOAT_STR_X, FLOAT_STR_COLUMNS

        # get remaining float rows after strs are chopped with MinCountTransformer
        # THIS IS SUPPOSED TO BE THE "TRUTH" FOR WHEN delete_axis_0 = True
        ChopStrTestCls = MinCountTransformer(*_args, **_kwargs)
        STR_MIN_COUNTED_X = ChopStrTestCls.fit_transform(FLOAT_STR_DF)

        STR_MIN_COUNTED_X_DF = pd.DataFrame(
            data=STR_MIN_COUNTED_X,
            columns=ChopStrTestCls.get_feature_names_out(None)
        )
        del ChopStrTestCls, STR_MIN_COUNTED_X, FLOAT_STR_DF

        # "TRUTH" for when delete_axis_0 = True
        STR_MIN_COUNTED_FLOAT_DF = STR_MIN_COUNTED_X_DF.iloc[:, :_mct_cols]

        del STR_MIN_COUNTED_X_DF
        # END get remaining float rows after strs are chopped with MinCountTransformer

        # GET DUMMIES FOR STR COLUMNS & CHOP USING delete_axis_0 = True / False
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
        FULL_DUMMIED_STR_DF = pd.concat((FLOAT_DF, DUMMIED_STR_DF), axis=1)
        del onehot, STR_DF, DUMMIED_STR_DF

        _kwargs['delete_axis_0'] = True
        ChopDummyDeleteAxis0TestCls = MinCountTransformer(*_args, **_kwargs)
        FULL_DUM_MIN_COUNTED_DELETE_0_X = \
            ChopDummyDeleteAxis0TestCls.fit_transform(FULL_DUMMIED_STR_DF)
        FULL_DUMMIED_DELETE_0_DF = pd.DataFrame(
            data=FULL_DUM_MIN_COUNTED_DELETE_0_X,
            columns=ChopDummyDeleteAxis0TestCls.get_feature_names_out(None)
        )
        del ChopDummyDeleteAxis0TestCls, FULL_DUM_MIN_COUNTED_DELETE_0_X

        # COMPARE AGAINST STR_MIN_COUNTED_FLOAT_DF
        DUM_MIN_COUNTED_DELETE_0_FLOAT_DF = \
            FULL_DUMMIED_DELETE_0_DF.iloc[:, :_mct_cols]
        del FULL_DUMMIED_DELETE_0_DF

        _kwargs['delete_axis_0'] = False
        ChopDummyDontDeleteAxis0TestCls = MinCountTransformer(*_args, **_kwargs)
        FULL_DUM_MIN_COUNTED_DONT_DELETE_0_X = \
            ChopDummyDontDeleteAxis0TestCls.fit_transform(FULL_DUMMIED_STR_DF)
        FULL_DUMMIED_DONT_DELETE_0_DF = pd.DataFrame(
            data=FULL_DUM_MIN_COUNTED_DONT_DELETE_0_X,
            columns=ChopDummyDontDeleteAxis0TestCls.get_feature_names_out(None)
        )
        del ChopDummyDontDeleteAxis0TestCls, FULL_DUM_MIN_COUNTED_DONT_DELETE_0_X

        # COMPARE AGAINST FLOAT_DF
        DUM_MIN_COUNTED_DONT_DELETE_0_FLOAT_DF = \
            FULL_DUMMIED_DONT_DELETE_0_DF.iloc[:, :_mct_cols]
        del FULL_DUMMIED_DONT_DELETE_0_DF

        # COMPARE:
        # 1) Ensure some rows were actually deleted by comparing
        #       STR_MIN_COUNTED_FLOAT_DF against FLOAT_DF
        # 2) Compare DUM_MIN_COUNTED_DELETE_0_FLOAT_DF against
        #       STR_MIN_COUNTED_FLOAT_DF
        # 3) Compare DUM_MIN_COUNTED_DONT_DELETE_0_FLOAT_DF against FLOAT_DF

        assert not STR_MIN_COUNTED_FLOAT_DF.equals(FLOAT_DF), \
            f"MinCountTransform of FLOAT_STR_DF did not delete any rows"
        assert DUM_MIN_COUNTED_DELETE_0_FLOAT_DF.equals(STR_MIN_COUNTED_FLOAT_DF), \
            (f"rows after MinCount on dummies with delete_axis_0=True do not "
             f"equal rows from MinCount on strings")
        assert DUM_MIN_COUNTED_DONT_DELETE_0_FLOAT_DF.equals(FLOAT_DF), \
            (f"rows after MinCount on dummies with delete_axis_0=False do not "
             f"equal original rows")

        del FLOAT_DF, FULL_DUMMIED_STR_DF, DUM_MIN_COUNTED_DONT_DELETE_0_FLOAT_DF
        del DUM_MIN_COUNTED_DELETE_0_FLOAT_DF, STR_MIN_COUNTED_FLOAT_DF

# END TEST delete_axis_0 WORKS #########################################



# TEST OUTPUT TYPES ####################################################

@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestOutputTypes:


    @pytest.mark.parametrize('x_input_type',
        ['numpy_array', 'pandas_dataframe', 'pandas_series']
    )
    @pytest.mark.parametrize('y_input_type',
        ['numpy_array', 'pandas_dataframe', 'pandas_series']
    )
    @pytest.mark.parametrize('output_type',
        [None, 'default', 'numpy_array','pandas_dataframe', 'pandas_series']
    )
    def test_output_types(
        self, X, y, _args, _kwargs, x_input_type, y_input_type, output_type
    ):

        NEW_X = X[:, 0].copy()
        NEW_COLUMNS = X[:1].copy()
        NEW_Y = y[:, 0].copy()


        if x_input_type == 'numpy_array':
            TEST_X = NEW_X.copy()
        elif 'pandas' in x_input_type:
            TEST_X = pd.DataFrame(data=NEW_X, columns=NEW_COLUMNS)
            if x_input_type == 'pandas_series':
                TEST_X = TEST_X.squeeze()

        if y_input_type == 'numpy_array':
            TEST_Y = NEW_Y.copy()
        elif 'pandas' in y_input_type:
            TEST_Y = pd.DataFrame(data=NEW_Y, columns=['y'])
            if y_input_type == 'pandas_series':
                TEST_Y = TEST_Y.squeeze()

        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.set_output(transform=output_type)

        TRFM_X, TRFM_Y = TestCls.fit_transform(TEST_X, TEST_Y)

        if output_type is None:
            assert type(TRFM_X) == type(TEST_X), \
                (f"output_type is None, X output type ({type(TRFM_X)}) != "
                 f"X input type ({type(TEST_X)})")
            assert type(TRFM_Y) == type(TEST_Y), \
                (f"output_type is None, Y output type ({type(TRFM_Y)}) != "
                 f"Y input type ({type(TEST_Y)})")
        elif output_type in ['default', 'numpy_array']:
            assert isinstance(TRFM_X, np.ndarray), \
                f"output_type is default or numpy_array, TRFM_X is {type(TRFM_X)}"
            assert isinstance(TRFM_Y, np.ndarray), \
                f"output_type is default or numpy_array, TRFM_Y is {type(TRFM_Y)}"
        elif output_type == 'pandas_dataframe':
            # pandas.core.frame.DataFrame
            assert isinstance(TRFM_X, pd.core.frame.DataFrame), \
                f"output_type is pandas dataframe, TRFM_X is {type(TRFM_X)}"
            assert isinstance(TRFM_Y, pd.core.frame.DataFrame), \
                f"output_type is pandas dataframe, TRFM_Y is {type(TRFM_Y)}"
        elif output_type == 'pandas_series':
            # pandas.core.series.Series
            assert isinstance(TRFM_X, pd.core.series.Series), \
                f"output_type is pandas series, TRFM_X is {type(TRFM_X)}"
            assert isinstance(TRFM_Y, pd.core.series.Series), \
                f"output_type is pandas sereis, TRFM_Y is {type(TRFM_Y)}"


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

        TestCls.test_threshold()
        print(f'^^^ mask building instructions should be displayed above ^^^')

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

        TestCls.test_threshold()
        print(f'^^^ mask building instructions should be displayed above ^^^')

        with pytest.raises(ValueError):
            TestCls.transform(TEST_X, TEST_Y)

        del TEST_X, TEST_Y, TestCls

# TEST ALL ROWS WILL BE DELETED ########################################


# TEST BIN INT COLUMN WITH ALL ABOVE THRESHOLD NOT DELETED #############
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestBinIntAboveThreshNotDeleted:

    def test_bin_int_above_thresh_not_deleted(self, _kwargs):

        TestCls = MinCountTransformer(2, **_kwargs)

        NEW_X = np.array([['a',0], ['b',0], ['a',1], ['b',1], ['c',0]], dtype=object)
        NEW_Y = np.array([0, 1, 0, 1, 1], dtype=np.uint8)

        TestCls.fit(NEW_X, NEW_Y)

        TRFM_X, TRFM_Y = TestCls.transform(NEW_X, NEW_Y)

        assert TRFM_X.shape[1]==2, \
            f"bin int column with all values above threshold was deleted"

        assert TRFM_X.shape[0]==4, \
            f"TRFM_X should have 4 rows but has {TRFM_X.shape[0]}"

# END TEST BIN INT COLUMN WITH ALL ABOVE THRESHOLD NOT DELETED #########


# TEST ACCURACY ********************************************************
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAccuracy:

    @pytest.mark.parametrize('count_threshold', [2, 3])
    @pytest.mark.parametrize('ignore_float_columns', [True, False])
    @pytest.mark.parametrize('ignore_non_binary_integer_columns', [True, False])
    @pytest.mark.parametrize('ignore_columns', [None, [0, 1, 2, 3]])
    @pytest.mark.parametrize('ignore_nan', [True, False])
    @pytest.mark.parametrize('handle_as_bool', ('hab_1', 'hab_2', 'hab_3'))
    @pytest.mark.parametrize('delete_axis_0', [False, True])
    @pytest.mark.parametrize('max_recursions', [1, 2])
    def test_accuracy(self, _kwargs, X, y, count_threshold, ignore_columns,
        ignore_float_columns, ignore_non_binary_integer_columns, ignore_nan,
        handle_as_bool, delete_axis_0, max_recursions, _mct_cols, x_cols, mmct
    ):

        if handle_as_bool == 'hab_1':
            HANDLE_AS_BOOL = None
        elif handle_as_bool == 'hab_2':
            HANDLE_AS_BOOL = list(range(_mct_cols, 2 * _mct_cols))
        elif handle_as_bool == 'hab_3':
            HANDLE_AS_BOOL = lambda X: list(range(_mct_cols, 2 * _mct_cols))

        args = [count_threshold]
        _kwargs['ignore_float_columns'] = ignore_float_columns
        _kwargs['ignore_non_binary_integer_columns'] = \
            ignore_non_binary_integer_columns
        _kwargs['ignore_columns'] = ignore_columns
        _kwargs['ignore_nan'] = ignore_nan
        _kwargs['handle_as_bool'] = HANDLE_AS_BOOL
        _kwargs['delete_axis_0'] = delete_axis_0
        _kwargs['max_recursions'] = max_recursions

        TEST_X = X.copy()
        TEST_Y = y.copy()

        TestCls = MinCountTransformer(*args, **_kwargs)
        TRFM_X, TRFM_Y = TestCls.fit_transform(TEST_X, TEST_Y)

        ###########################################
        ###########################################
        # MANUALLY OPERATE ON MOCK_X & MOCK_Y #####
        MOCK_X = X.copy()
        MOCK_Y = y.copy()

        try:
            _ignore_columns = ignore_columns(TEST_X)
        except:
            _ignore_columns = ignore_columns
        try:
            _handle_as_bool = HANDLE_AS_BOOL(TEST_X)
        except:
            _handle_as_bool = HANDLE_AS_BOOL

        if max_recursions == 1:
            MOCK_X, MOCK_Y = mmct().trfm(
                MOCK_X, MOCK_Y, _ignore_columns, ignore_nan,
                ignore_non_binary_integer_columns,
                ignore_float_columns, _handle_as_bool,
                delete_axis_0, count_threshold
            )

        elif max_recursions == 2:

            mmct_first_rcr = mmct()   # give class a name to access attr later
            MOCK_X1, MOCK_Y1 = mmct_first_rcr.trfm(
                MOCK_X, MOCK_Y, _ignore_columns, ignore_nan,
                ignore_non_binary_integer_columns,
                ignore_float_columns, _handle_as_bool,
                delete_axis_0, count_threshold
            )

            if MOCK_X.shape[1] == TEST_X.shape[1]:
                new_ignore_columns = _ignore_columns
                new_handle_as_bool = _handle_as_bool
            else:
                # ADJUST ign_columns & handle_as_bool FOR 2ND RECURSION
                NEW_COLUMN_MASK = np.arange(x_cols)[mmct_first_rcr.get_support_]

                OG_IGN_COL_MASK = np.zeros(x_cols).astype(bool)
                OG_IGN_COL_MASK[_ignore_columns] = True
                new_ignore_columns = \
                    np.arange(len(NEW_COLUMN_MASK))[OG_IGN_COL_MASK[NEW_COLUMN_MASK]]

                OG_H_A_B_MASK = np.zeros(x_cols).astype(bool)
                OG_H_A_B_MASK[_handle_as_bool] = True
                new_handle_as_bool = \
                    np.arange(len(NEW_COLUMN_MASK))[OG_H_A_B_MASK[NEW_COLUMN_MASK]]

                del NEW_COLUMN_MASK, OG_IGN_COL_MASK, OG_H_A_B_MASK

            MOCK_X, MOCK_Y = mmct().trfm(
                MOCK_X1, MOCK_Y1, new_ignore_columns, ignore_nan,
                ignore_non_binary_integer_columns,
                ignore_float_columns, new_handle_as_bool,
                delete_axis_0, count_threshold
            )

            del MOCK_X1, MOCK_Y1, new_ignore_columns, new_handle_as_bool

        else:
            raise Exception(
                f"Test is not designed to handle more than 2 recursions"
            )

        # END MANUALLY OPERATE ON MOCK_X & MOCK_Y #
        ###########################################
        ###########################################

        mask_maker = \
            lambda XX: np.logical_not(np.char.lower(XX.astype(str)) == 'nan')

        try:
            # pizza vvvv failing when run all tests
            assert np.array_equiv(TRFM_X.astype(str), MOCK_X.astype(str))
            # ^^^^^ ^^^^^^ ^^^^^ ^^^^^
            assert np.array_equiv(TRFM_Y.astype(str), MOCK_Y.astype(str))

        except Exception as e:

            if max_recursions == 1:
                raise AssertionError(e)

            elif max_recursions > 1:
                _kwargs['max_recursions'] = 1
                NewTestCls_1 = MinCountTransformer(*args,**_kwargs)
                NEW_TRFM_X, NEW_TRFM_Y = NewTestCls_1.fit_transform(TEST_X, TEST_Y)

                # ADJUST ign_columns & handle_as_bool FOR 2ND RECURSION
                NEW_COLUMN_MASK = NewTestCls_1.get_support(True)

                OG_IGN_COL_MASK = np.zeros(x_cols).astype(bool)
                OG_IGN_COL_MASK[_ignore_columns] = True
                new_ignore_columns = \
                    np.arange(len(NEW_COLUMN_MASK))[OG_IGN_COL_MASK[NEW_COLUMN_MASK]]

                OG_H_A_B_MASK = np.zeros(x_cols).astype(bool)
                OG_H_A_B_MASK[_handle_as_bool] = True
                new_handle_as_bool = \
                    np.arange(len(NEW_COLUMN_MASK))[OG_H_A_B_MASK[NEW_COLUMN_MASK]]

                del NEW_COLUMN_MASK, OG_IGN_COL_MASK, OG_H_A_B_MASK

                _kwargs['ignore_columns'] = new_ignore_columns
                _kwargs['handle_as_bool'] = new_handle_as_bool
                NewTestCls_2 = MinCountTransformer(*args,**_kwargs)
                NEW_TRFM_X, NEW_TRFM_Y = \
                    NewTestCls_2.fit_transform(NEW_TRFM_X, NEW_TRFM_Y)

                # vvvv pizza ****************************************
                # where <function array_equiv> 2133: AssertionError
                assert np.array_equiv(NEW_TRFM_X[mask_maker(NEW_TRFM_X)],
                    MOCK_X[mask_maker(MOCK_X)]), (f'{max_recursions}X PASSES '
                    f'THRU TestCls WITH max_recursions=1 FAILED')
                # ^^^^^ ************************************************

                assert np.array_equiv(NEW_TRFM_Y[mask_maker(NEW_TRFM_Y)],
                    MOCK_Y[mask_maker(MOCK_Y)]), (f'{max_recursions}X PASSES '
                    f'THRU TestCls WITH max_recursions=1 FAILED')

                del NEW_TRFM_X, NEW_TRFM_Y

        del _ignore_columns, _handle_as_bool



# END TEST ACCURACY ****************************************************


# TEST FIT->SET_PARAMS->TRFM == SET_PARAMS->FIT_TRFM *******************
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestFitSetParamsTransform_SetParamsFitTransform:

        # DEFAULT ARGS/KWARGS
        # _args = [_mct_rows // 20]
        # _kwargs = {
        #                 'ignore_float_columns': True,
        #                 'ignore_non_binary_integer_columns': True,
        #                 'ignore_columns': None,
        #                 'ignore_nan': True,
        #                 'delete_axis_0': True,
        #                 'handle_as_bool': None,
        #                 'reject_unseen_values': False,
        #                 'max_recursions': 1,
        #                 'n_jobs': -1
        # }

    def test_interchangable(self, X, y, _args, _kwargs, _mct_rows, mmct):

        alt_args = [_mct_rows // 25]
        alt_kwargs = {
            'ignore_float_columns': True,
            'ignore_non_binary_integer_columns': True,
            'ignore_columns': [0, 2],
            'ignore_nan': False,
            'delete_axis_0': False,
            'handle_as_bool': None,
            'reject_unseen_values': False,
            'max_recursions': 1,
            'n_jobs': -1
        }

        TEST_X = X.copy()
        TEST_Y = y.copy()


        # SET_PARAMS->FIT_TRFM
        SPFTCls = MinCountTransformer(*alt_args, **alt_kwargs)
        SPFT_TRFM_X, SPFT_TRFM_Y = SPFTCls.fit_transform(TEST_X, TEST_Y)

        # FIT->SET_PARAMS->TRFM
        FSPTCls = MinCountTransformer(*_args, **_kwargs)
        FSPTCls.fit(TEST_X, TEST_Y)
        FSPTCls.set_params(count_threshold=alt_args[0], **alt_kwargs)
        FSPT_TRFM_X, FSPT_TRFM_Y = FSPTCls.transform(TEST_X, TEST_Y)

        assert np.array_equiv(SPFT_TRFM_X.astype(str), FSPT_TRFM_X.astype(str)), \
            f"SPFT_TRFM_X != FSPT_TRFM_X"

        assert np.array_equiv(SPFT_TRFM_Y, FSPT_TRFM_Y), \
            f"SPFT_TRFM_Y != FSPT_TRFM_Y"

        MOCK_X = mmct().trfm(
            TEST_X,
            None,
            alt_kwargs['ignore_columns'],
            alt_kwargs['ignore_nan'],
            alt_kwargs['ignore_non_binary_integer_columns'],
            alt_kwargs['ignore_float_columns'],
            alt_kwargs['handle_as_bool'],
            alt_kwargs['delete_axis_0'],
            alt_args[0]
        )

        assert np.array_equiv(FSPT_TRFM_X.astype(str), MOCK_X.astype(str)), \
            f"FSPT_TRFM_X != MOCK_X"

        del TEST_X, TEST_Y, MOCK_X

# END TEST FIT->SET_PARAMS->TRFM == SET_PARAMS->FIT_TRFM ***************


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
        DoublePartialFitTestClass = MinCountTransformer(*_args, **_kwargs)

        SingleFitTestClass.fit(X, y)
        DoublePartialFitTestClass.partial_fit(X, y)

        DoublePartialFitTestClass.partial_fit(X, y)

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

        X1 = NO_NAN_X[:, _mct_cols:(2 * _mct_cols)].copy().astype(np.float64).astype(np.int32)
        y1 = y.copy()
        # 10X THE VALUES IN THE COPY OF DATA TO INTRODUCE NEW UNIQUE VALUES
        X2 = (10 * X1.astype(np.float64)).astype(np.int32)
        y2 = y.copy()

        STACKED_X = np.vstack((X1, X2)).astype(np.float64).astype(np.int32)
        STACKED_Y = np.vstack((y1, y2)).astype(np.uint8)

        args = [2 * _args[0]]
        _kwargs['ignore_non_binary_integer_columns'] = False

        PartialFitTestCls = MinCountTransformer(*args, **_kwargs)

        PartialFitTestCls.partial_fit(X1, y1)
        PartialFitTestCls.partial_fit(X2, y2)
        PARTIAL_FIT_X, PARTIAL_FIT_Y = \
            PartialFitTestCls.transform(STACKED_X, STACKED_Y)

        # VERIFY SOME ROWS WERE ACTUALLY DELETED
        assert not np.array_equiv(PARTIAL_FIT_X, np.vstack((X1, X2))), \
            (f'later partial fits accept new uniques --- '
             f'transform did not delete any rows')

        SingleFitTestCls = MinCountTransformer(*args, **_kwargs)
        SingleFitTestCls.fit(STACKED_X, STACKED_Y)
        SINGLE_FIT_X, SINGLE_FIT_Y = \
            SingleFitTestCls.transform(STACKED_X, STACKED_Y)

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
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit(X1, y1)
        TestCls.transform(X1, y1)
        del TestCls

        # PEPPER ONE OF THE STR COLUMNS WITH A UNIQUE THAT WAS NOT SEEN DURING fit()
        X2 = X1.copy()
        MASK = np.random.choice(range(x_rows), 10, replace=False)
        X2[MASK, 0] = list('1234567890')
        del MASK
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit(X1, y1)

        # DEMONSTRATE NEW VALUES ARE ACCEPTED WHEN reject_unseen_values = False
        TestCls.set_params(reject_unseen_values=False)
        TestCls.transform(X2, y1)

        # DEMONSTRATE NEW VALUES ARE REJECTED WHEN reject_unseen_values = True
        with pytest.raises(ValueError):
            TestCls.set_params(reject_unseen_values=True)
            TestCls.transform(X2, y1)

        del X1, y1, X2, TestCls


# END TEST TRANSFORM CONDITIONALLY ACCEPT NEW UNIQUES ******************


# TEST DASK Incremental + ParallelPostFit == ONE BIG sklearn fit_transform()
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestDaskIncrementalParallelPostFit:


    @staticmethod
    @pytest.fixture
    def MCT_not_wrapped(_args, _kwargs):
        return MinCountTransformer(*_args, **_kwargs)

    @staticmethod
    @pytest.fixture
    def MCT_wrapped_parallel(_args, _kwargs):
        return ParallelPostFit(MinCountTransformer(*_args, **_kwargs))

    @staticmethod
    @pytest.fixture
    def MCT_wrapped_incremental(_args, _kwargs):
        return Incremental(MinCountTransformer(*_args, **_kwargs))

    @staticmethod
    @pytest.fixture
    def MCT_wrapped_both(_args, _kwargs):
        return ParallelPostFit(Incremental(MinCountTransformer(*_args, **_kwargs)))



    FORMATS = ['da', 'ddf_df', 'ddf_series', 'np', 'pddf', 'pdseries']
    @pytest.mark.parametrize('x_format', FORMATS)
    @pytest.mark.parametrize('y_format', FORMATS + [None])
    @pytest.mark.parametrize('wrappings', ('incr', 'ppf', 'both', 'none'))
    def test_always_fits_X_y_always_excepts_transform_with_y(self, wrappings,
        MCT_wrapped_parallel, MCT_wrapped_incremental, MCT_not_wrapped,
        MCT_wrapped_both, NO_NAN_X, COLUMNS, y, x_format, y_format, _args, 
        _kwargs, _mct_cols, x_rows, x_cols, y_rows
    ):

        # no difference with or without Client

        # USE NUMERICAL COLUMNS ONLY 24_03_27_11_45_00
        # NotImplementedError: Cannot use auto rechunking with object dtype.
        # We are unable to estimate the size in bytes of object data

        X = NO_NAN_X.copy()[:, :3*_mct_cols].astype(np.float64)
        COLUMNS = COLUMNS.copy()[:3 * _mct_cols]

        if wrappings == 'incr':
            _test_cls = MCT_wrapped_parallel
        elif wrappings == 'ppf':
            _test_cls = MCT_wrapped_incremental
        elif wrappings == 'both':
            _test_cls = MCT_wrapped_both
        elif wrappings == 'none':
            _test_cls = MCT_not_wrapped

        _X = X.copy()
        _np_X = _X.copy()
        _chunks = (x_rows//5, x_cols)
        if x_format in ['pddf', 'pdseries']:
            _X = pd.DataFrame(data=_X, columns=COLUMNS)
        if x_format == 'pdseries':
            _X = _X.iloc[:, 0].squeeze()
            assert isinstance(_X, pd.core.series.Series)
            _np_X = _X.to_frame().to_numpy()
        if x_format in ['da', 'ddf_df', 'ddf_series']:
            _X = da.from_array(_X, chunks=_chunks)
        if x_format in ['ddf_df', 'ddf_series']:
            _X = ddf.from_array(_X, chunksize=_chunks)
        if x_format == 'ddf_series':
            _X = _X.iloc[:, 0].squeeze()
            assert isinstance(_X, (ddf.core.Series, ddf2.Series))
            _np_X = _X.compute().to_frame().to_numpy()

        # confirm there is an X
        _X.shape

        _y = y.copy()
        _np_y = _y.copy()
        _chunks = (y_rows//5, 2)
        if y_format in ['pddf', 'pdseries']:
            _y = pd.DataFrame(data=_y, columns=['y1', 'y2'])
        if y_format == 'pdseries':
            _y = _y.iloc[:, 0].squeeze()
            assert isinstance(_y, pd.core.series.Series)
            _np_y = _y.to_frame().to_numpy()
        if y_format in ['da', 'ddf_df', 'ddf_series']:
            _y = da.from_array(_y, chunks=_chunks)
        if y_format in ['ddf_df', 'ddf_series']:
            _y = ddf.from_array(_y, chunksize = _chunks)
        if y_format == 'ddf_series':
            _y = _y.iloc[:, 0].squeeze()
            assert isinstance(_y, (ddf.core.Series, ddf2.Series))
            _np_y = _y.compute().to_frame().to_numpy()
        if y_format is None:
            _y = None

        # confirm there is a y
        if _y is not None:
            _y.shape

        _was_fitted = False
        # incr covers fit() so should accept all objects for fits
        _dask = ['da', 'ddf_df', 'ddf_series']
        _non_dask = ['np', 'pddf', 'pdseries']

        a = x_format in _dask and y_format in _non_dask
        b = x_format in _non_dask and y_format in _dask
        if wrappings in ['ppf', 'both'] and (a + b) == 1:

            with pytest.raises(UnboundLocalError):
                _test_cls.partial_fit(_X, _y)

            with pytest.raises(UnboundLocalError):
                _test_cls.fit(_X, _y)

        else:
            _test_cls.partial_fit(_X, _y)
            _test_cls.fit(_X, _y)
            _was_fitted = True

        del _dask, _non_dask

        # ^^^ END fit ^^^

        # vvv transform vvv
        if _was_fitted:

            if x_format not in ['pdseries', 'ddf_series']:
                assert _X.shape[1] == X.shape[1]

            # always TypeError when try to pass y with ParallelPostFit
            _x_was_transformed = False
            _y_was_transformed = False
            if wrappings in ['ppf', 'both', 'incr']:

                with pytest.raises(TypeError):
                    _test_cls.transform(_X, _y)

                # always transforms with just X
                TRFM_X = _test_cls.transform(_X)
                _x_was_transformed = True

            elif wrappings in ['none']:

                if _y is not None:
                    _test_cls.transform(_X, _y)
                    TRFM_X, TRFM_Y = _test_cls.fit_transform(_X, _y)
                    _x_was_transformed = True
                    _y_was_transformed = True
                else:
                    _test_cls.transform(_X)
                    TRFM_X = _test_cls.fit_transform(_X, _y)
                    _x_was_transformed = True

            if _x_was_transformed:
                if x_format == 'np':
                    assert isinstance(TRFM_X, np.ndarray)
                if x_format == 'pddf':
                    assert isinstance(TRFM_X, pd.core.frame.DataFrame)
                if x_format == 'pdseries':
                    assert isinstance(TRFM_X, pd.core.series.Series)
                if x_format == 'da' and wrappings == 'none':
                    assert isinstance(TRFM_X, np.ndarray)
                elif x_format == 'da':
                    assert isinstance(TRFM_X, da.core.Array)
                if x_format == 'ddf_df' and wrappings == 'none':
                    assert isinstance(TRFM_X, pd.core.frame.DataFrame)
                elif x_format == 'ddf_df':
                    assert isinstance(TRFM_X, (ddf.core.DataFrame, ddf2.DataFrame))
                if x_format == 'ddf_series' and wrappings == 'none':
                    assert isinstance(TRFM_X, pd.core.series.Series)
                elif x_format == 'ddf_series':
                    assert isinstance(TRFM_X, (ddf.core.Series, ddf2.Series))

                if _y_was_transformed:
                    if y_format == 'np':
                        assert isinstance(TRFM_Y, np.ndarray)
                    if y_format == 'pddf':
                        assert isinstance(TRFM_Y, pd.core.frame.DataFrame)
                    if y_format == 'pdseries':
                        assert isinstance(TRFM_Y, pd.core.series.Series)
                    if y_format == 'da' and wrappings == 'none':
                        assert isinstance(TRFM_Y, np.ndarray)
                    elif y_format == 'da':
                        assert isinstance(TRFM_Y, da.core.Array)
                    if y_format == 'ddf_df' and wrappings == 'none':
                        assert isinstance(TRFM_Y, pd.core.frame.DataFrame)
                    elif y_format == 'ddf_df':
                        assert isinstance(TRFM_Y,
                            (ddf.core.DataFrame, ddf2.DataFrame)
                        )
                    if y_format == 'ddf_series' and wrappings == 'none':
                        assert isinstance(TRFM_Y, pd.core.series.Series)
                    elif y_format == 'ddf_series':
                        assert isinstance(TRFM_Y, (ddf.core.Series, ddf2.Series))

                # CONVERT TO NP ARRAY FOR COMPARISON AGAINST ONE-SHOT fit_trfm()
                try:
                    TRFM_X = TRFM_X.to_frame()
                except:
                    pass

                try:
                    TRFM_X = TRFM_X.compute()
                except:
                    pass

                try:
                    TRFM_X = TRFM_X.to_numpy()
                except:
                    pass

                if _y_was_transformed:

                    try:
                        TRFM_Y = TRFM_Y.to_frame()
                    except:
                        pass

                    try:
                        TRFM_Y = TRFM_Y.compute()
                    except:
                        pass

                    try:
                        TRFM_Y = TRFM_Y.to_numpy()
                    except:
                        pass

                # END CONVERT TO NP ARRAY FOR COMPARISON AGAINST ONE-SHOT fit_trfm()

                FitTransformTestCls = MinCountTransformer(*_args, **_kwargs)
                if _y_was_transformed:
                    FT_TRFM_X, FT_TRFM_Y = \
                        FitTransformTestCls.fit_transform(_np_X, _np_y)
                else:
                    FT_TRFM_X = FitTransformTestCls.fit_transform(_np_X)

                assert isinstance(TRFM_X, np.ndarray)
                assert isinstance(FT_TRFM_X, np.ndarray)
                assert np.array_equiv(
                        TRFM_X.astype(str), FT_TRFM_X.astype(str)), \
                    (f"transformed X  != transformed np X on single fit/transform")

                if _y_was_transformed:
                    assert isinstance(TRFM_Y, np.ndarray)
                    assert isinstance(FT_TRFM_Y, np.ndarray)
                    assert np.array_equiv(
                        TRFM_Y.astype(str),
                        FT_TRFM_Y.astype(str)
                    ), f"transformed Y != transformed np Y on single fit/transform"

# END TEST DASK Incremental + ParallelPostFit == ONE BIG sklearn fit_transform()


# ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM, ATTR ACCURACY; FOR 1 RECURSION
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestAttrAccuracyBeforeAndAfterFitAndTransform:

    def test_attr_accuracy(
        self, X, COLUMNS, y, _args, _kwargs, DTYPE_KEY, x_cols
    ):

        NEW_X = X.copy()
        NEW_Y = y.copy()
        NEW_X_DF = pd.DataFrame(data=X, columns=COLUMNS)
        NEW_Y_DF = pd.DataFrame(data=y, columns=['y1', 'y2'])

        # BEFORE FIT ***************************************************

        TestCls = MinCountTransformer(*_args, **_kwargs)

        # ALL OF THESE SHOULD GIVE AttributeError
        with pytest.raises(AttributeError):
            TestCls.feature_names_in_

        with pytest.raises(AttributeError):
            TestCls.n_features_in_

        with pytest.raises(AttributeError):
            TestCls.original_dtypes_

        with pytest.raises(AttributeError):
            TestCls.original_dtypes_ = list('abcde')

        del TestCls
        # END BEFORE FIT ***********************************************

        # AFTER FIT ****************************************************
        for data_dtype in ['np', 'pd']:
            if data_dtype == 'np':
                TEST_X, TEST_Y = NEW_X.copy(), NEW_Y.copy()
            elif data_dtype == 'pd':
                TEST_X, TEST_Y = NEW_X_DF.copy(), NEW_Y_DF.copy()

            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.fit(TEST_X, TEST_Y)

            # ONLY EXCEPTION SHOULD BE feature_names_in_ IF NUMPY
            if data_dtype == 'pd':
                assert np.array_equiv(TestCls.feature_names_in_, COLUMNS), \
                    f"feature_names_in_ after fit() != originally passed columns"
            elif data_dtype == 'np':
                with pytest.raises(AttributeError):
                    TestCls.feature_names_in_

            assert TestCls.n_features_in_ == x_cols, \
                f"n_features_in_ after fit() != number of originally passed columns"

            assert np.array_equiv(TestCls._original_dtypes, DTYPE_KEY), \
                f"_original_dtypes after fit() != originally passed dtypes"

        del data_dtype, TEST_X, TEST_Y, TestCls

        # END AFTER FIT ************************************************

        # AFTER TRANSFORM **********************************************

        for data_dtype in ['np', 'pd']:

            if data_dtype == 'np':
                TEST_X, TEST_Y = NEW_X.copy(), NEW_Y.copy()
            elif data_dtype == 'pd':
                TEST_X, TEST_Y = NEW_X_DF.copy(), NEW_Y_DF.copy()

            TestCls = MinCountTransformer(*_args, **_kwargs)
            TestCls.fit_transform(TEST_X, TEST_Y)

            # ONLY EXCEPTION SHOULD BE feature_names_in_ WHEN NUMPY
            if data_dtype == 'pd':
                assert np.array_equiv(TestCls.feature_names_in_, COLUMNS), \
                    f"feature_names_in_ after fit() != originally passed columns"
            elif data_dtype == 'np':
                with pytest.raises(AttributeError):
                    TestCls.feature_names_in_

            assert TestCls.n_features_in_ == x_cols, \
                f"n_features_in_ after fit() != number of originally passed columns"

            assert np.array_equiv(TestCls._original_dtypes, DTYPE_KEY), \
                f"_original_dtypes after fit() != originally passed dtypes"

        del data_dtype, TEST_X, TEST_Y, TestCls
        # END AFTER TRANSFORM ******************************************

        del NEW_X, NEW_Y, NEW_X_DF, NEW_Y_DF

# END ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM, ATTR ACCURACY; FOR 1 RECURSION


# ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM; FOR 1 RECURSION ***
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class Test1RecursionAccessMethodsBeforeAndAfterFitAndTransform:

    def test_access_methods_before_fit(self, X, y, _args, _kwargs):

        TestCls = MinCountTransformer(*_args, **_kwargs)

        # **************************************************************
        # vvv BEFORE FIT vvv *******************************************

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        with pytest.raises(NotFittedError):
            TestCls.test_threshold()

        # fit()
        # fit_transform()

        # get_feature_names_out()
        with pytest.raises(NotFittedError):
            TestCls.get_feature_names_out(None)

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TestCls.get_metadata_routing()

        # get_params()
        TestCls.get_params(True)

        # get_row_support()
        with pytest.raises(NotFittedError):
            TestCls.get_row_support(True)

        # get_support()
        with pytest.raises(NotFittedError):
            TestCls.get_support(True)

        # ** _handle_X_y()

        # inverse_transform()
        with pytest.raises(NotFittedError):
            TestCls.inverse_transform(X)

        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TestCls.set_output(transform='pandas_dataframe')

        # set_params()
        KEYS = [
            'count_threshold', 'ignore_float_columns',
            'ignore_non_binary_integer_columns', 'ignore_columns', 'ignore_nan',
            'handle_as_bool', 'delete_axis_0', 'reject_unseen_values',
            'max_recursions', 'n_jobs'
        ]
        VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
        test_kwargs = dict((zip(KEYS, VALUES)))
        TestCls.set_params(**test_kwargs)
        ATTRS = [
            TestCls._count_threshold, TestCls._ignore_float_columns,
            TestCls._ignore_non_binary_integer_columns, TestCls._ignore_columns,
            TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
            TestCls._reject_unseen_values, TestCls._max_recursions, TestCls._n_jobs
        ]
        for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
            assert _attr == _value, f'set_params() did not set {_key}'

        # DEMONSTRATE EXCEPTS FOR UNKNOWN PARAM
        with pytest.raises(ValueError):
            TestCls.set_params(garbage=1)

        del TestCls, KEYS, VALUES, ATTRS

        TestCls = MinCountTransformer(*_args, **_kwargs)
        # transform()
        with pytest.raises(NotFittedError):
            TestCls.transform(X, y)

        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        # END ^^^ BEFORE FIT ^^^ ***************************************
        # **************************************************************

    def test_access_methods_after_fit(self, X, COLUMNS, y, _args, _kwargs):

        # **************************************************************
        # vvv AFTER FIT vvv ********************************************

        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit(X, y)

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        TestCls.test_threshold()
        print(f'^^^ mask building instructions should be displayed above ^^^')

        # fit()
        # fit_transform()

        # get_feature_names_out() **************************************
        # vvv NO COLUMN NAMES PASSED (NP) vvv
        # **** CAN ONLY TAKE LIST-TYPE OF STRS OR None
        JUNK_ARGS = [float('inf'), np.pi, 'garbage', {'junk': 3},
                     [*range(len(COLUMNS))]
        ]

        for junk_arg in JUNK_ARGS:
            with pytest.raises(TypeError):
                TestCls.get_feature_names_out(junk_arg)

        del JUNK_ARGS

        # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN
        # ['x0', ..., 'x(n-1)][COLUMN MASK]
        _COLUMNS = np.array([f"x{i}" for i in range(len(COLUMNS))])
        ACTIVE_COLUMNS = _COLUMNS[TestCls.get_support(False)]
        del _COLUMNS
        assert np.array_equiv( TestCls.get_feature_names_out(None), ACTIVE_COLUMNS), \
            (f"get_feature_names_out(None) after fit() != sliced array of "
            f"generic headers")
        del ACTIVE_COLUMNS

        # WITH NO HEADER PASSED, SHOULD RAISE ValueError IF
        # len(input_features) != n_features_in_
        with pytest.raises(ValueError):
            TestCls.get_feature_names_out([f"x{i}" for i in range(2 * len(COLUMNS))])

        # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features,
        # SHOULD RETURN SLICED PASSED COLUMNS
        RETURNED_FROM_GFNO = TestCls.get_feature_names_out(COLUMNS)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but "
             f"returned {type(RETURNED_FROM_GFNO)}")
        _ACTIVE_COLUMNS = np.array(COLUMNS)[TestCls.get_support(False)]
        assert np.array_equiv(RETURNED_FROM_GFNO, _ACTIVE_COLUMNS), \
            f"get_feature_names_out() did not return original columns"

        del junk_arg, RETURNED_FROM_GFNO, TestCls, _ACTIVE_COLUMNS

        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv

        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit(pd.DataFrame(data=X, columns=COLUMNS), y)

        # WITH HEADER PASSED AND input_features=None, SHOULD RETURN
        # SLICED ORIGINAL COLUMNS
        _ACTIVE_COLUMNS = np.array(COLUMNS)[TestCls.get_support(False)]
        assert np.array_equiv(TestCls.get_feature_names_out(None), _ACTIVE_COLUMNS), \
            f"get_feature_names_out(None) after fit() != originally passed columns"
        del _ACTIVE_COLUMNS

        # WITH HEADER PASSED, SHOULD RAISE TypeError IF input_features
        # FOR DISALLOWED TYPES

        JUNK_COL_NAMES = [
            [*range(len(COLUMNS))], [*range(2 * len(COLUMNS))], {'a': 1, 'b': 2}
        ]
        for junk_col_names in JUNK_COL_NAMES:
            with pytest.raises(TypeError):
                TestCls.get_feature_names_out(junk_col_names)

        del JUNK_COL_NAMES

        # WITH HEADER PASSED, SHOULD RAISE ValueError IF input_features DOES
        # NOT EXACTLY MATCH ORIGINALLY FIT COLUMNS
        JUNK_COL_NAMES = [np.char.upper(COLUMNS), np.hstack((COLUMNS, COLUMNS)), []]
        for junk_col_names in JUNK_COL_NAMES:
            with pytest.raises(ValueError):
                TestCls.get_feature_names_out(junk_col_names)

        # WHEN HEADER PASSED TO (partial_)fit() AND input_features IS THAT HEADER,
        # SHOULD RETURN SLICED VERSION OF THAT HEADER

        RETURNED_FROM_GFNO = TestCls.get_feature_names_out(COLUMNS)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, "
             f"but returned {type(RETURNED_FROM_GFNO)}")

        assert np.array_equiv(RETURNED_FROM_GFNO,
                      np.array(COLUMNS)[TestCls.get_support(False)]), \
            f"get_feature_names_out() did not return original columns"

        del junk_col_names, RETURNED_FROM_GFNO
        # END ^^^ COLUMN NAMES PASSED (PD) ^^^

        # END get_feature_names_out() **********************************

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TestCls.get_metadata_routing()

        # get_params()
        TestCls.get_params(True)

        # get_row_support()
        with pytest.raises(AttributeError):
            TestCls.get_row_support(False)

        # get_support()
        for _indices in [True, False]:
            __ = TestCls.get_support(_indices)
            assert isinstance(__, np.ndarray), (f"get_support() did not return "
                                                f"numpy.ndarray")

            if not _indices:
                assert __.dtype == 'bool', \
                    f"get_support with indices=False did not return a boolean array"
                assert len(__) == TestCls.n_features_in_, \
                    f"len(get_support(False)) != n_features_in_"
                assert sum(__) == len(TestCls.get_feature_names_out(None))
            elif _indices:
                assert 'int' in str(__.dtype).lower(), \
                    (f"get_support with indices=True did not return an array of "
                     f"integers")
                assert len(__) == len(TestCls.get_feature_names_out(None))

        del TestCls, _indices, __,

        # ** _handle_X_y()

        # inverse_transform() ********************
        TestCls = MinCountTransformer(*_args, **_kwargs)
        TestCls.fit(X, y)  # X IS NP ARRAY

        # SHOULD RAISE ValueError IF X IS NOT A 2D ARRAY
        for junk_x in [[], [[]]]:
            with pytest.raises(ValueError):
                TestCls.inverse_transform(junk_x)

        # SHOULD RAISE TypeError IF X IS NOT A LIST-TYPE
        for junk_x in [None, 'junk_string', 3, np.pi]:
            with pytest.raises(TypeError):
                TestCls.inverse_transform(junk_x)

        # SHOULD RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF
        # RETAINED COLUMNS
        TRFM_X = TestCls.transform(X)
        TRFM_MASK = TestCls.get_support(False)
        __ = np.array(COLUMNS)
        for obj_type in ['np', 'pd']:
            for diff_cols in ['more', 'less', 'same']:
                if diff_cols == 'same':
                    TEST_X = TRFM_X.copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[TRFM_MASK])
                elif diff_cols == 'less':
                    TEST_X = TRFM_X[:, :2].copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[TRFM_MASK][:2])
                elif diff_cols == 'more':
                    TEST_X = np.hstack((TRFM_X.copy(), TRFM_X.copy()))
                    if obj_type == 'pd':
                        _COLUMNS = np.hstack((__[TRFM_MASK],
                                              np.char.upper(__[TRFM_MASK])
                        ))
                        TEST_X = pd.DataFrame(data=TEST_X, columns=_COLUMNS)

                if diff_cols == 'same':
                    TestCls.inverse_transform(TEST_X)
                else:
                    with pytest.raises(ValueError):
                        TestCls.inverse_transform(TEST_X)

        INV_TRFM_X = TestCls.inverse_transform(TRFM_X)

        assert isinstance(INV_TRFM_X, np.ndarray), \
            f"output of inverse_transform() is not a numpy array"
        assert INV_TRFM_X.shape[0] == TRFM_X.shape[0], \
            f"rows in output of inverse_transform() do not match input rows"
        assert INV_TRFM_X.shape[1] == TestCls.n_features_in_, \
            (f"columns in output of inverse_transform() do not match "
             f"originally fitted columns")

        __ = np.logical_not(TestCls.get_support(False))
        assert np.array_equiv(INV_TRFM_X[:, __],
                              np.zeros((TRFM_X.shape[0], sum(__)))
            ), \
            (f"back-filled parts of inverse_transform() output do not slice "
             f"to a zero array")
        del __

        assert np.array_equiv(
            TRFM_X.astype(str),
            INV_TRFM_X[:, TestCls.get_support(False)].astype(str)
            ), (f"output of inverse_transform() does not reduce back to "
                f"the output of transform()")

        del junk_x, TRFM_X, TRFM_MASK, obj_type, diff_cols
        del TEST_X, INV_TRFM_X, TestCls

        # END inverse_transform() **********

        TestCls = MinCountTransformer(*_args, **_kwargs)

        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TestCls.set_output(transform='pandas_dataframe')

        # set_params()
        KEYS = [
            'count_threshold', 'ignore_float_columns',
            'ignore_non_binary_integer_columns', 'ignore_columns', 'ignore_nan',
            'handle_as_bool', 'delete_axis_0', 'reject_unseen_values',
            'max_recursions', 'n_jobs'
        ]
        VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
        test_kwargs = dict((zip(KEYS, VALUES)))

        TestCls.set_params(**test_kwargs)
        ATTRS = [
            TestCls._count_threshold, TestCls._ignore_float_columns,
            TestCls._ignore_non_binary_integer_columns, TestCls._ignore_columns,
            TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
            TestCls._reject_unseen_values, TestCls._max_recursions,
            TestCls._n_jobs
        ]
        for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
            assert _attr == _value, f'set_params() did not set {_key}'

        del TestCls, KEYS, VALUES, ATTRS

        # transform()
        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        # END ^^^ AFTER FIT ^^^ ****************************************
        # **************************************************************

    def test_access_methods_after_transform(self, X, COLUMNS, y, _args, _kwargs):

        # **************************************************************
        # vvv AFTER TRANSFORM vvv **************************************
        FittedTestCls = MinCountTransformer(*_args, **_kwargs).fit(X, y)
        TransformedTestCls = MinCountTransformer(*_args, **_kwargs).fit(X, y)
        TRFM_X, TRFM_Y = TransformedTestCls.transform(X, y)

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        # SHOULD BE THE SAME AS AFTER FIT
        TransformedTestCls.test_threshold()
        print(f'^^^ mask building instructions should be displayed above ^^^')

        # fit()
        # fit_transform()

        # get_feature_names_out() **************************************
        # vvv NO COLUMN NAMES PASSED (NP) vvv

        # # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features,
        # SHOULD RETURN ORIGINAL (UNSLICED) COLUMNS
        RETURNED_FROM_GFNO = TransformedTestCls.get_feature_names_out(COLUMNS)


        _ACTIVE_COLUMNS = np.array(COLUMNS)[TransformedTestCls.get_support(False)]
        assert np.array_equiv(RETURNED_FROM_GFNO, _ACTIVE_COLUMNS), \
            (f"get_feature_names_out() after transform did not return "
             f"sliced original columns")

        del RETURNED_FROM_GFNO
        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv
        PDTransformedTestCls = MinCountTransformer(*_args, **_kwargs)
        PDTransformedTestCls.fit_transform(pd.DataFrame(data=X, columns=COLUMNS), y)

        # WITH HEADER PASSED AND input_features=None,
        # SHOULD RETURN SLICED ORIGINAL COLUMNS
        assert np.array_equiv(PDTransformedTestCls.get_feature_names_out(None),
                      np.array(COLUMNS)[PDTransformedTestCls.get_support(False)]), \
            (f"get_feature_names_out(None) after transform() != "
             f"originally passed columns")

        del PDTransformedTestCls
        # END ^^^ COLUMN NAMES PASSED (PD) ^^^

        # END get_feature_names_out() **********************************

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TransformedTestCls.get_metadata_routing()

        # get_params()
        assert TransformedTestCls.get_params(True) == \
                FittedTestCls.get_params(True), \
            f"get_params() after transform() != before transform()"

        # get_row_support()
        for _indices in [True, False]:
            __ = TransformedTestCls.get_row_support(_indices)
            assert isinstance(__, np.ndarray), \
                f"get_row_support() did not return numpy.ndarray"

            if not _indices:
                assert __.dtype == 'bool', \
                    (f"get_row_support with indices=False did not return a "
                     f"boolean array")
            elif _indices:
                assert 'int' in str(__.dtype).lower(), \
                    (f"get_row_support with indices=True did not return an "
                     f"array of integers")

        del __

        # get_support()
        assert np.array_equiv(FittedTestCls.get_support(False),
            TransformedTestCls.get_support(False)), \
            f"get_support(False) after transform() != get_support(False) before"

        # ** _handle_X_y()

        # inverse_transform() ************

        assert np.array_equiv(
            FittedTestCls.inverse_transform(TRFM_X).astype(str),
            TransformedTestCls.inverse_transform(TRFM_X).astype(str)), \
            (f"inverse_transform(TRFM_X) after transform() != "
             f"inverse_transform(TRFM_X) before transform()")

        # END inverse_transform() **********

        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TransformedTestCls.set_output(transform='pandas_dataframe')
        TransformedTestCls.transform(X, y)

        del TransformedTestCls

        # set_params()
        TestCls = MinCountTransformer(*_args, **_kwargs)
        KEYS = [
            'count_threshold', 'ignore_float_columns',
            'ignore_non_binary_integer_columns', 'ignore_columns', 'ignore_nan',
            'handle_as_bool', 'delete_axis_0', 'reject_unseen_values',
            'max_recursions', 'n_jobs'
        ]
        VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
        test_kwargs = dict((zip(KEYS, VALUES)))

        TestCls.set_params(**test_kwargs)
        TestCls.fit_transform(X, y)
        ATTRS = [
            TestCls._count_threshold, TestCls._ignore_float_columns,
            TestCls._ignore_non_binary_integer_columns, TestCls._ignore_columns,
            TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
            TestCls._reject_unseen_values, TestCls._max_recursions,
            TestCls._n_jobs
        ]
        for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
            assert _attr == _value, f'set_params() did not set {_key}'

        del KEYS, VALUES, ATTRS

        # transform()
        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        del FittedTestCls, TestCls, TRFM_X, TRFM_Y

        # END ^^^ AFTER TRANSFORM ^^^ **********************************
        # **************************************************************

# END ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM; FOR 1 RECURSION


# ACCESS ATTR BEFORE fit() AND AFTER fit_transform(), ATTR ACCURACY
# FOR 2 RECURSIONS ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class Test2RecursionAccessAttrsBeforeAndAfterFitAndTransform:

    def test_2_recursion_access_attrs(
        self, X, y, _kwargs, COLUMNS, DTYPE_KEY, x_cols
    ):

        NEW_X = X.copy()
        NEW_Y = y.copy()
        NEW_X_DF = pd.DataFrame(data=X, columns=COLUMNS)
        NEW_Y_DF = pd.DataFrame(data=y, columns=['y1', 'y2'])

        args = [3]
        OneRecurTestCls = MinCountTransformer(*args, **_kwargs)

        _kwargs['max_recursions'] = 2

        # BEFORE FIT ***************************************************

        TwoRecurTestCls = MinCountTransformer(*args, **_kwargs)

        # ALL OF THESE SHOULD GIVE AttributeError
        with pytest.raises(AttributeError):
            TwoRecurTestCls.feature_names_in_

        with pytest.raises(AttributeError):
            TwoRecurTestCls.n_features_in_

        with pytest.raises(AttributeError):
            TwoRecurTestCls.original_dtypes_

        with pytest.raises(AttributeError):
            TwoRecurTestCls.original_dtypes_ = list('abcde')

        # END BEFORE FIT ***********************************************

        # AFTER fit_transform() ****************************************
        for data_dtype in ['np', 'pd']:
            if data_dtype == 'np':
                TEST_X, TEST_Y = NEW_X.copy(), NEW_Y.copy()
            elif data_dtype == 'pd':
                TEST_X, TEST_Y = NEW_X_DF.copy(), NEW_Y_DF.copy()

            OneRecurTestCls.fit_transform(TEST_X, TEST_Y)
            TwoRecurTestCls.fit_transform(TEST_X, TEST_Y)

            assert OneRecurTestCls.n_features_in_ == x_cols, \
                (f"OneRecur.n_features_in_ after fit_transform() != "
                 f"number of originally passed columns")
            assert TwoRecurTestCls.n_features_in_ == x_cols, \
                (f"TwoRecur.n_features_in_ after fit_transform() != "
                 f"number of originally passed columns")

            # ONLY EXCEPTION SHOULD BE feature_names_in_ WHEN NUMPY
            if data_dtype == 'pd':
                assert np.array_equiv(TwoRecurTestCls.feature_names_in_, COLUMNS), \
                    (f"2 recurrence feature_names_in_ after fit_transform() != "
                     f"originally passed columns")

                assert np.array_equiv(TwoRecurTestCls.feature_names_in_,
                    OneRecurTestCls.feature_names_in_), \
                    (f"2 recurrence feature_names_in_ after fit_transform() != 1 "
                     f"recurrence feature_names_in_ after fit_transform()")
            elif data_dtype == 'np':
                with pytest.raises(AttributeError):
                    TwoRecurTestCls.feature_names_in_

            # n_features_in_ SHOULD BE EQUAL FOR OneRecurTestCls AND TwoRecurTestCls
            _, __ = OneRecurTestCls.n_features_in_, TwoRecurTestCls.n_features_in_
            assert _ == __, (f"OneRecurTestCls.n_features_in_ ({_}) != "
                             f"TwoRecurTestcls.n_features_in_ ({__})")
            del _, __

            assert np.array_equiv(TwoRecurTestCls._original_dtypes, DTYPE_KEY), \
                f"_original_dtypes after fit_transform() != originally passed dtypes"

        # END AFTER fit_transform() ************************************

        del NEW_X, NEW_Y, NEW_X_DF, NEW_Y_DF, data_dtype, TEST_X, TEST_Y
        del OneRecurTestCls, TwoRecurTestCls

# END ACCESS ATTR BEFORE fit() AND AFTER fit_transform()
# ATTR ACCURACY; FOR 2 RECURSIONS **************************************


# ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM; FOR 2 RECURSIONS **
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class Test2RecursionAccessMethodsBeforeAndAfterFitAndTransform:

    # CREATE AN INSTANCE WITH ONLY 1 RECURSION TO COMPARE 1X-TRFMED OBJECTS
    # AGAINST 2X-TRFMED OBJECTS
    @staticmethod
    @pytest.fixture
    def OneRecurTestCls(_kwargs):
        args = [3]
        _kwargs['ignore_columns'] = None
        _kwargs['ignore_nan'] = False
        _kwargs['ignore_non_binary_integer_columns'] = False
        _kwargs['ignore_float_columns'] = False
        _kwargs['delete_axis_0'] = True
        _kwargs['max_recursions'] = 1

        return MinCountTransformer(*args, **_kwargs)


    @staticmethod
    @pytest.fixture
    def TwoRecurTestCls(_kwargs):
        args = [3]
        _kwargs['ignore_columns'] = None
        _kwargs['ignore_nan'] = False
        _kwargs['ignore_non_binary_integer_columns'] = False
        _kwargs['ignore_float_columns'] = False
        _kwargs['delete_axis_0'] = True
        _kwargs['max_recursions'] = 2

        return MinCountTransformer(*args, **_kwargs)

    def test_before_fit_transform(self, OneRecurTestCls, TwoRecurTestCls,
        X, y, _args, _kwargs):

        # **************************************************************
        # vvv BEFORE fit_transform() vvv *******************************

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.test_threshold()

        # fit()
        # fit_transform()

        # get_feature_names_out()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.get_feature_names_out(None)

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TwoRecurTestCls.get_metadata_routing()

        # get_params()
        # ALL PARAMS SHOULD BE THE SAME EXCEPT FOR max_recursions
        _ = OneRecurTestCls.get_params(True)
        del _['max_recursions']
        __ = TwoRecurTestCls.get_params(True)
        del __['max_recursions']
        assert _ == __, (f"pre-fit 1 recursion instance get_params() != "
                         f"get_params() from 2 recursion instance")
        del _, __

        # get_row_support()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.get_row_support(True)

        # get_support()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.get_support(True)

        # ** _handle_X_y()

        # inverse_transform()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.inverse_transform(X)

        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TwoRecurTestCls.set_output(transform='pandas_dataframe')

        # set_params()
        TestCls = MinCountTransformer(*_args, **_kwargs)
        KEYS = [
            'count_threshold', 'ignore_float_columns',
            'ignore_non_binary_integer_columns', 'ignore_columns', 'ignore_nan',
            'handle_as_bool', 'delete_axis_0', 'reject_unseen_values',
            'max_recursions', 'n_jobs'
        ]
        VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
        test_kwargs = dict((zip(KEYS, VALUES)))

        TestCls.set_params(**test_kwargs)
        ATTRS = [
            TestCls._count_threshold, TestCls._ignore_float_columns,
            TestCls._ignore_non_binary_integer_columns, TestCls._ignore_columns,
            TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
            TestCls._reject_unseen_values, TestCls._max_recursions, TestCls._n_jobs
        ]

        for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
            assert _attr == _value, f'set_params() did not set {_key}'

        del TestCls, KEYS, VALUES, ATTRS

        TwoRecurTestCls = MinCountTransformer(*_args, **_kwargs)
        # transform()
        with pytest.raises(AttributeError):
            TwoRecurTestCls.transform(X, y)

        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        # END ^^^ BEFORE fit_transform() ^^^ ***************************
        # **************************************************************

        del TwoRecurTestCls


    def test_after_fit_transform(self, OneRecurTestCls, TwoRecurTestCls, X,
        NO_NAN_X, COLUMNS, y, _args, _kwargs, x_rows, x_cols):

        ONE_RCR_TRFM_X, ONE_RCR_TRFM_Y = OneRecurTestCls.fit_transform(X, y)
        TWO_RCR_TRFM_X, TWO_RCR_TRFM_Y = TwoRecurTestCls.fit_transform(X, y)

        # **************************************************************
        # vvv AFTER fit_transform() vvv ********************************

        # ** _base_fit()
        # ** _check_is_fitted()

        # ** test_threshold()
        assert not np.array_equiv(ONE_RCR_TRFM_X, TWO_RCR_TRFM_X), \
            f"ONE_RCR_TRFM_X == TWO_RCR_TRFM_X when it shouldnt"

        assert (OneRecurTestCls._total_counts_by_column !=
                TwoRecurTestCls._total_counts_by_column), \
            (f"OneRecurTestCls._total_counts_by_column == "
             f"TwoRecurTestCls._total_counts_by_column when it shouldnt")

        _ONE_delete_instr = OneRecurTestCls._make_instructions(_args[0])
        _TWO_delete_instr = TwoRecurTestCls._make_instructions(_args[0])
        # THE FOLLOWING MUST BE TRUE BECAUSE TEST DATA BUILD VALIDATION
        # REQUIRES 2 RECURSIONS W CERTAIN KWARGS DOES DELETE SOMETHING
        assert _TWO_delete_instr != _ONE_delete_instr, \
            (f"fit-trfmed 2 recursion delete instr == fit-trfmed 1 recursion "
             f"delete instr and should not")

        # THE NUMBER OF COLUMNS IN BOTH delete_instr DICTS ARE EQUAL
        assert len(_TWO_delete_instr) == len(_ONE_delete_instr), \
            (f"number of columns in TwoRecurTestCls delete instr != number of "
             f"columns in OneRecurTestCls delete instr")

        # LEN OF INSTRUCTIONS IN EACH COLUMN FOR TWO RECUR MUST BE >=
        # INSTRUCTIONS FOR ONE RECUR BECAUSE THEYVE BEEN MELDED
        for col_idx in _ONE_delete_instr:
            _, __ = len(_TWO_delete_instr[col_idx]), len(_ONE_delete_instr[col_idx])
            assert _ >= __, (f"number of instruction in TwoRecurTestCls count "
                         f"is not >= number of instruction in OneRecurTestCls"
            )

        # ALL THE ENTRIES FROM 1 RECURSION ARE IN THE MELDED INSTRUCTION DICT
        # OUTPUT OF MULTIPLE RECURSIONS
        for col_idx in _ONE_delete_instr:
            for unq in list(map(str, _ONE_delete_instr[col_idx])):
                if unq in ['INACTIVE', 'DELETE COLUMN']:
                    continue
                assert unq in list(map(str, _TWO_delete_instr[col_idx])), \
                    f"{unq} is in 1 recur delete instructions but not 2 recur"

        del _ONE_delete_instr, _TWO_delete_instr, _, __, col_idx, unq

        TwoRecurTestCls.test_threshold(clean_printout=True)
        print(f'^^^ mask building instructions should be displayed above ^^^')

        with pytest.raises(ValueError):
            TwoRecurTestCls.test_threshold(2 * _args[0])

        # fit()
        # fit_transform()

        # get_feature_names_out() **************************************
        # vvv NO COLUMN NAMES PASSED (NP) vvv

        # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN
        # SLICED ['x0', ..., 'x(n-1)]
        _COLUMNS = np.array([f"x{i}" for i in range(len(COLUMNS))])
        _ACTIVE_COLUMNS = _COLUMNS[TwoRecurTestCls.get_support(False)]
        del _COLUMNS
        assert np.array_equiv(
            TwoRecurTestCls.get_feature_names_out(None),
            _ACTIVE_COLUMNS
        ), (f"get_feature_names_out(None) after fit_transform() != sliced "
            f"array of generic headers"
        )
        del _ACTIVE_COLUMNS

        # WITH NO HEADER PASSED, SHOULD RAISE ValueError IF len(input_features) !=
        # n_features_in_
        with pytest.raises(ValueError):
            _COLUMNS = [f"x{i}" for i in range(2 * len(COLUMNS))]
            TwoRecurTestCls.get_feature_names_out(_COLUMNS)
            del _COLUMNS

        # WHEN NO HEADER PASSED TO fit_transform() AND VALID input_features,
        # SHOULD RETURN SLICED PASSED COLUMNS
        RETURNED_FROM_GFNO = TwoRecurTestCls.get_feature_names_out(COLUMNS)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"TwoRecur.get_feature_names_out should return numpy.ndarray, "
             f"but returned {type(RETURNED_FROM_GFNO)}")

        assert np.array_equiv(RETURNED_FROM_GFNO,
            np.array(COLUMNS)[TwoRecurTestCls.get_support(False)]), \
            f"TwoRecur.get_feature_names_out() did not return original columns"

        del RETURNED_FROM_GFNO

        # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

        # vvv COLUMN NAMES PASSED (PD) vvv
        ONE_RCR_TRFM_X, ONE_RCR_TRFM_Y = \
            OneRecurTestCls.fit_transform(pd.DataFrame(data=X, columns=COLUMNS), y)

        TWO_RCR_TRFM_X, TWO_RCR_TRFM_Y = \
            TwoRecurTestCls.fit_transform(pd.DataFrame(data=X, columns=COLUMNS), y)

        # WITH HEADER PASSED AND input_features=None:
        # SHOULD RETURN SLICED ORIGINAL COLUMNS
        assert np.array_equiv(
            TwoRecurTestCls.get_feature_names_out(None),
            np.array(COLUMNS)[TwoRecurTestCls.get_support(False)]
            ), (f"TwoRecur.get_feature_names_out(None) after fit_transform() != "
            f"sliced originally passed columns"
        )

        # WHEN HEADER PASSED TO fit_transform() AND input_features IS THAT HEADER,
        # SHOULD RETURN SLICED VERSION OF THAT HEADER
        RETURNED_FROM_GFNO = TwoRecurTestCls.get_feature_names_out(COLUMNS)
        assert isinstance(RETURNED_FROM_GFNO, np.ndarray), \
            (f"get_feature_names_out should return numpy.ndarray, but returned "
             f"{type(RETURNED_FROM_GFNO)}")
        assert np.array_equiv(
            RETURNED_FROM_GFNO,
            np.array(COLUMNS)[TwoRecurTestCls.get_support(False)]
            ), f"get_feature_names_out() did not return original columns"

        del RETURNED_FROM_GFNO
        # END ^^^ COLUMN NAMES PASSED (PD) ^^^
        # END get_feature_names_out() **********************************

        # get_metadata_routing()
        with pytest.raises(NotImplementedError):
            TwoRecurTestCls.get_metadata_routing()

        # get_params()
        # ALL PARAMS SHOULD BE THE SAME EXCEPT FOR max_recursions
        _ = OneRecurTestCls.get_params(True)
        del _['max_recursions']
        __ = TwoRecurTestCls.get_params(True)
        del __['max_recursions']
        assert _ == __, (f"pre-fit 1 recursion instance get_params() != "
                         f"get_params() from 2 recursion instance")
        del _, __

        # get_row_support()
        for _indices in [True, False]:
            _ONE = OneRecurTestCls.get_row_support(_indices)
            _TWO = TwoRecurTestCls.get_row_support(_indices)

            assert isinstance(_ONE, np.ndarray), \
                f"get_row_support() for 1 recursion did not return numpy.ndarray"
            assert isinstance(_TWO, np.ndarray), \
                f"get_row_support() for 2 recursions did not return numpy.ndarray"

            if not _indices:
                assert _ONE.dtype == 'bool', (f"get_row_support with indices=False "
                          f"for 1 recursion did not return a boolean array")
                assert _TWO.dtype == 'bool', (f"get_row_support with indices=False "
                              f"for 2 recursions did not return a boolean array")
                # len(ROW SUPPORT TWO RECUR) AND len(ROW SUPPORT ONE RECUR)
                # MUST EQUAL NUMBER OF ROWS IN X
                assert len(_ONE) == x_rows, \
                    (f"row_support vector length for 1 recursion != rows in "
                     f"passed data"
                )
                assert len(_TWO) == x_rows, \
                    (f"row_support vector length for 2 recursions != rows in "
                     f"passed data"
                )
                # NUMBER OF Trues in ONE RECUR MUST == NUMBER OF ROWS IN
                # ONE RCR TRFM X; SAME FOR TWO RCR
                assert sum(_ONE) == ONE_RCR_TRFM_X.shape[0], \
                    f"one rcr Trues IN row_support != TRFM X rows"
                assert sum(_TWO) == TWO_RCR_TRFM_X.shape[0], \
                    f"two rcr Trues IN row_support != TRFM X rows"
                # NUMBER OF Trues IN ONE RECUR MUST BE >= NUMBER OF Trues
                # IN TWO RECUR
                assert sum(_ONE) >= sum(_TWO), \
                    f"two recursion has more rows kept in it that one recursion"
                # ANY Trues IN TWO RECUR MUST ALSO BE True IN ONE RECUR
                assert np.unique(_ONE[_TWO])[0] == True, \
                    (f"Rows that are to be kept in 2nd recur (True) were False "
                     f"in 1st recur")
            elif _indices:
                assert 'int' in str(_ONE.dtype).lower(), \
                    (f"get_row_support with indices=True for 1 recursion did not "
                     f"return an array of integers")
                assert 'int' in str(_TWO.dtype).lower(), \
                    (f"get_row_support with indices=True for 2 recursions did not "
                     f"return an array of integers")
                # len(row_support) ONE RECUR MUST == NUMBER OF ROWS IN ONE RCR
                # TRFM X; SAME FOR TWO RCR
                assert len(_ONE) == ONE_RCR_TRFM_X.shape[0], \
                    f"one rcr len(row_support) as idxs does not equal TRFM X rows"
                assert len(_TWO) == TWO_RCR_TRFM_X.shape[0], \
                    f"two rcr len(row_support) as idxs does not equal TRFM X rows "
                # NUMBER OF ROW IDXS IN ONE RECUR MUST BE >= NUM ROW IDXS IN TWO RECUR
                assert len(_ONE) >= len(_TWO), \
                    f"two recursion has more rows kept in it that one recursion"
                # INDICES IN TWO RECUR MUST ALSO BE IN ONE RECUR
                for row_idx in _TWO:
                    assert row_idx in _ONE, (f"Rows that are to be kept by 2nd "
                                             f"recur were not kept by 1st recur")

        del _ONE, _TWO, row_idx, _indices
        del ONE_RCR_TRFM_X, ONE_RCR_TRFM_Y, TWO_RCR_TRFM_X, TWO_RCR_TRFM_Y

        # get_support()
        for _indices in [True, False]:
            _ = OneRecurTestCls.get_support(_indices)
            __ = TwoRecurTestCls.get_support(_indices)
            assert isinstance(_, np.ndarray), \
                f"2 recursion get_support() did not return numpy.ndarray"
            assert isinstance(__, np.ndarray), \
                f"2 recursion get_support() did not return numpy.ndarray"

            if not _indices:
                assert _.dtype == 'bool', \
                    (f"1 recursion get_support with indices=False did not "
                     f"return a boolean array")
                assert __.dtype == 'bool', \
                    (f"2 recursion get_support with indices=False did not "
                     f"return a boolean array")

                # len(ROW SUPPORT TWO RECUR) AND len(ROW SUPPORT ONE RECUR)
                # MUST EQUAL NUMBER OF COLUMNS IN X
                assert len(_) == x_cols, \
                    f"1 recursion len(get_support({_indices})) != X columns"
                assert len(__) == x_cols, \
                    f"2 recursion len(get_support({_indices})) != X columns"
                # NUM COLUMNS IN 1 RECURSION MUST BE <= NUM COLUMNS IN X
                assert sum(_) <= x_cols, \
                    (f"impossibly, number of columns kept by 1 recursion > number "
                     f"of columns in X")
                # NUM COLUMNS IN 2 RECURSION MUST BE <= NUM COLUMNS IN 1 RECURSION
                assert sum(__) <= sum(_),\
                    (f"impossibly, number of columns kept by 2 recursion > number "
                     f"of columns kept by 1 recursion")
                # INDICES IN TWO RECUR MUST ALSO BE IN ONE RECUR
                assert np.unique(_[__])[0] == True, (f"Columns that are to be "
                         f"kept in 2nd recur (True) were False in 1st recur")
            elif _indices:
                assert 'int' in str(_.dtype).lower(), (f"1 recursion get_support "
                    f"with indices=True did not return an array of integers")
                assert 'int' in str(__.dtype).lower(), (f"2 recursion get_support "
                    f"with indices=True did not return an array of integers")
                # ONE RECURSION COLUMNS MUST BE <= n_features_in_
                assert len(_) <= x_cols, \
                    f"impossibly, 1 recursion len(get_support({_indices})) > X columns"
                # TWO RECURSION COLUMNS MUST BE <= ONE RECURSION COLUMNS
                assert len(__) <= len(_), \
                    (f"2 recursion len(get_support({_indices})) > 1 "
                     f"recursion len(get_support({_indices}))")
                # INDICES IN TWO RECUR MUST ALSO BE IN ONE RECUR
                for col_idx in __:
                    assert col_idx in _, (f"Columns that are to be kept by "
                              f"2nd recur were not kept by 1st recur")

        del TwoRecurTestCls, _, __, _indices, col_idx

        # ** _handle_X_y()

        # inverse_transform() ********************
        TwoRecurTestCls = MinCountTransformer(count_threshold=3, **_kwargs)
        # X IS NP ARRAY
        TRFM_X, TRFM_Y = TwoRecurTestCls.fit_transform(X, y)

        # SHOULD RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF RETAINED COLUMNS
        __ = np.array(COLUMNS)
        TRFM_MASK = TwoRecurTestCls.get_support(False)
        for obj_type in ['np', 'pd']:
            for diff_cols in ['more', 'less', 'same']:
                if diff_cols == 'same':
                    TEST_X = TRFM_X.copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[TRFM_MASK])
                elif diff_cols == 'less':
                    TEST_X = TRFM_X[:, :2].copy()
                    if obj_type == 'pd':
                        TEST_X = pd.DataFrame(data=TEST_X, columns=__[TRFM_MASK][:2])
                elif diff_cols == 'more':
                    TEST_X = np.hstack((TRFM_X.copy(), TRFM_X.copy()))
                    if obj_type == 'pd':
                        _COLUMNS = np.hstack((__[TRFM_MASK],
                                              np.char.upper(__[TRFM_MASK])
                        ))
                        TEST_X = pd.DataFrame(data=TEST_X, columns=_COLUMNS)
                        del _COLUMNS

                if diff_cols == 'same':
                    TwoRecurTestCls.inverse_transform(TEST_X)
                else:
                    with pytest.raises(ValueError):
                        TwoRecurTestCls.inverse_transform(TEST_X)

        INV_TRFM_X = TwoRecurTestCls.inverse_transform(TRFM_X)

        assert isinstance(INV_TRFM_X, np.ndarray), \
            f"output of inverse_transform() is not a numpy array"
        assert INV_TRFM_X.shape[0] == TRFM_X.shape[0], \
            f"rows in output of inverse_transform() do not match input rows"
        assert INV_TRFM_X.shape[1] == TwoRecurTestCls.n_features_in_, \
            (f"columns in output of inverse_transform() do not match originally "
             f"fitted columns")

        __ = np.logical_not(TwoRecurTestCls.get_support(False))
        _ZERO_ARRAY = np.zeros((TRFM_X.shape[0], sum(__)))
        assert np.array_equiv(INV_TRFM_X[:, __], _ZERO_ARRAY), (f"back-filled "
            f"parts of inverse_transform() output do not slice to a zero array")
        del __, _ZERO_ARRAY

        assert np.array_equiv(TRFM_X.astype(str),
            INV_TRFM_X[:, TwoRecurTestCls.get_support(False)].astype(str)), \
            (f"output of inverse_transform() does not reduce back to the output "
             f"of transform()")

        del TRFM_X, TRFM_Y, TRFM_MASK, obj_type, diff_cols, TEST_X, INV_TRFM_X

        # END inverse_transform() **********

        # ** _make_instructions()
        # ** _must_be_fitted()
        # partial_fit()
        # ** _reset()

        # set_output()
        TwoRecurTestCls.set_output(transform='pandas_dataframe')
        assert TwoRecurTestCls._output_transform == 'pandas_dataframe'
        TwoRecurTestCls.fit_transform(X, y)
        assert TwoRecurTestCls._output_transform == 'pandas_dataframe'

        del TwoRecurTestCls

        # set_params()
        TestCls = MinCountTransformer(*_args, **_kwargs)
        KEYS = [
            'count_threshold', 'ignore_float_columns',
            'ignore_non_binary_integer_columns', 'ignore_columns', 'ignore_nan',
            'handle_as_bool', 'delete_axis_0', 'reject_unseen_values',
            'max_recursions', 'n_jobs'
        ]
        VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
        test_kwargs = dict((zip(KEYS, VALUES)))

        TestCls.set_params(**test_kwargs)
        ATTRS = [
            TestCls._count_threshold, TestCls._ignore_float_columns,
            TestCls._ignore_non_binary_integer_columns, TestCls._ignore_columns,
            TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
            TestCls._reject_unseen_values, TestCls._max_recursions, TestCls._n_jobs
        ]
        for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
            assert _attr == _value, f'set_params() did not set {_key}'

        del TestCls, KEYS, VALUES, ATTRS

        # transform()
        # ** _validate_delete_instr()
        # ** _validate_feature_names()
        # ** _validate()

        del OneRecurTestCls,

        # END ^^^ AFTER fit_transform() ^^^ ****************************
        # **************************************************************


# END ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM; FOR 2 RECURSIONS

















