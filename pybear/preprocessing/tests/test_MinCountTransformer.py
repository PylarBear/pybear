from copy import deepcopy
import time
import uuid

import numpy as np
import pandas as pd
import dask.array as da
from dask_ml.wrappers import Incremental, ParallelPostFit
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder

from preprocessing._data import MinCountTransformer
from mock_min_count_trfm import mmct



bypass = False

test_start_time = time.perf_counter()

_rows = 200
_cols = 2

orig_args = [_rows // 20]
orig_kwargs = {
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

assert _rows in range(50, 751), f"_rows must be between 50 and 750"
assert _cols > 0, f"_cols must be > 0"

ctr = 0
while True:  # LOOP UNTIL DATA IS SUFFICIENT TO BE USED FOR ALL THE TESTS
    ctr += 1
    # print(f'X GENERATION COUNTER = {ctr}')
    _tries = 10
    if ctr >= _tries:
        raise Exception(f"\033[91mMinCountThreshold failed at {_tries} attempts "
                        f"to generate an appropriate X for test\033[0m")

    # vvv CORE TEST DATA vvv *************************
    # CREATE x_cols COLUMNS OF BINARY INTEGERS
    X = np.random.randint(0, 2, (_rows, _cols)).astype(object)
    # CREATE x_cols COLUMNS OF NON-BINARY INTEGERS
    X = np.hstack((X, np.random.randint(0, _rows // 15, (_rows, _cols)).astype(object)))
    # CREATE x_cols COLUMNS OF FLOATS
    X = np.hstack((X, np.random.uniform(0, 1, (_rows, _cols)).astype(object)))
    # CREATE x_cols COLUMNS OF STRS
    _alpha = 'abcdefghijklmnopqrstuvwxyz'
    _alpha = _alpha + _alpha.upper()
    for _ in range(_cols):
        X = np.hstack((X,
                       np.random.choice(
                                        list(_alpha[:_rows // 10]),
                                        (_rows,),
                                        replace=True
                       ).astype(object).reshape((-1, 1))
        ))
    # END ^^^ CORE TEST DATA ^^^ *************************

    # CREATE A COLUMN OF STRS THAT WILL ALWAYS BE DELETED BY FIRST RECURSION
    DUM_STR_COL = np.fromiter(('dum' for _ in range(_rows)), dtype='<U3')
    DUM_STR_COL[0] = 'one'
    DUM_STR_COL[1] = 'two'
    DUM_STR_COL[2] = 'six'
    DUM_STR_COL[3] = 'ten'

    X = np.hstack((X, DUM_STR_COL.reshape((-1, 1)).astype(object)))
    del DUM_STR_COL

    # X SHAPE SHOULD BE (x_rows, 4 * x_cols + 1)
    x_rows = _rows
    x_cols = 4 * _cols + 1

    DTYPE_KEY = [k for k in ['int', 'int', 'float', 'obj'] for j in range(_cols)] + ['obj']

    # KEEP THIS FOR TESTING IF DTYPES RETRIEVED CORRECTLY WITH np.nan MIXED IN
    NO_NAN_X = X.copy()

    # FLOAT/STR ONLY --- NO_NAN_X MUST BE REDUCED WHEN STR COLUMNS ARE TRANSFORMED
    FLOAT_STR_X = NO_NAN_X[:, 2 * _cols:4 * _cols].copy()
    # mmct().trfm args = MOCK_X, MOCK_Y, ign_col, ign_nan, ignore_non_binary_int_col,
    # ignore_flt_col, handle_as_bool, delete_axis_0, ct_thresh
    X1 = mmct().trfm(X, None, None, True, True, True, None, True, *orig_args)
    if np.array_equiv(X1, FLOAT_STR_X):
        del X1
        continue
    del X1

    del FLOAT_STR_X

    # PEPPER 10% OF CORE DATA WITH np.nan
    for _ in range(x_rows * x_cols // 10):
        row_coor = np.random.randint(0, x_rows)
        col_coor = np.random.randint(0, x_cols - 1)
        if col_coor < 3 * _cols:
            X[row_coor, col_coor] = np.nan
        elif col_coor >= 3 * _cols:
            X[row_coor, col_coor] = 'nan'
    del row_coor, col_coor

    # MAKE EVERY CORE COLUMN HAVE 2 VALUES THAT CT FAR EXCEEDS count_threshold SO DOESNT ALLOW FULL DELETE
    _repl = x_rows // 3
    for idx in range(_cols):
        X[np.random.choice(range(x_rows), _repl, replace=False), _cols + idx] = \
            int(np.random.randint(0, x_rows // 20) + idx)
        X[np.random.choice(range(x_rows), _repl, replace=False), _cols + idx] = \
            int(np.random.randint(0, x_rows // 20) + idx)
        X[np.random.choice(range(x_rows), _repl, replace=False), 2 * _cols + idx] = \
            np.random.uniform(0, 1) + idx
        X[np.random.choice(range(x_rows), _repl, replace=False), 2 * _cols + idx] = \
            np.random.uniform(0, 1) + idx
        X[np.random.choice(range(x_rows), _repl, replace=False), 3 * _cols + idx] = \
            _alpha[:x_rows // 15][idx]
        X[np.random.choice(range(x_rows), _repl, replace=False), 3 * _cols + idx] = \
            _alpha[:x_rows // 15][idx + 1]

    del idx, _repl, _alpha

    # VERIFY ONE RECURSION OF mmct DELETED THE SACRIFICIAL LAST COLUMN (CORE COLUMNS ARE RIGGED TO NOT BE DELETED)
    # MOCK_X, MOCK_Y, ign_col, ign_nan, ignore_non_binary_int_col, ignore_flt_col, handle_as_bool, delete_axis_0, ct_thresh
    X1 = mmct().trfm(X, None, None, False, False, False, None, True, *orig_args)
    assert not np.array_equiv(X1[:,-1], X[:,-1]), \
        "Mock MinCountTransformer did not delete last column"

    if len(X1.shape) != 2 or 0 in X1.shape:  # IF ONE RECURSION DELETES EVERYTHING, BUILD NEW X
        continue
    elif np.array_equiv(X1[:, :-1],
                        X[:, :-1]):  # IF ONE RECURSION DOESNT DELETE ANY ROWS OF THE CORE COLUMNS, BUILD NEW X
        continue

    # IF NUM OF RIGGED IDENTICAL NUMBERS IN ANY FLT COLUMN < THRESHOLD, BUILD NEW X
    for flt_col_idx in range(2 * _cols, 3 * _cols, 1):
        if np.unique(X[:, flt_col_idx], return_counts=True)[1].max(axis=0) < orig_args[0]:
            continue

    # TRFM OF NON-BINARY INTEGER COLUMNS MUST NOT DELETE EVERYTHING, BUT MUST DELETE SOMETHING
    try:
        X1 = \
        mmct().trfm(X[:, _cols:(2 * _cols)].copy(), None, None, True, False, True, False, False, *orig_args)
        # MOCK_X, MOCK_Y, ign_col, ign_nan, ignore_non_binary_int_col, ignore_flt_col, handle_as_bool, delete_axis_0, ct_thresh
        if np.array_equiv(X1, X[:, _cols:(2 * _cols)].copy()):
            continue
    except:
        continue

    try_again = False
    # IF ALL CTS OF EVERY STR UNIQUE IS >= THRESHOLD, BUILD NEW X
    for str_col_idx in range(x_cols - 1, x_cols - _cols - 1, -1):
        if min(np.unique(X[:, str_col_idx], return_counts=True)[1]) >= orig_args[0]:
            try_again = True
            break
    if try_again:
        continue

    # IF X CANNOT TAKE 2 RECURSIONS WITH THRESHOLD==3, BUILD NEW X
    try_again = False
    X1 = mmct().trfm(X, None, None, False, False, False, False, True, 3)
    # MOCK_X, MOCK_Y, ign_col, ign_nan, ignore_non_binary_int_col, ignore_flt_col, handle_as_bool, delete_axis_0, ct_thresh
    try:
        # THIS SHOULD EXCEPT IF ALL ROWS/COLUMNS WOULD BE DELETED
        X2 = mmct().trfm(X1, None, None, False, False, False, False, True, 3)
        # SECOND RECURSION SHOULD ALSO DELETE SOMETHING, BUT NOT EVERYTHING
        if np.array_equiv(X1, X2):
            try_again = True
    except:
        try_again = True

    if try_again:
        continue

    del try_again, X1, X2

    # IF X PASSED ALL THESE PRE-CONDITION TESTS, IT IS GOOD TO USE FOR TEST
    break

COLUMNS = [str(uuid.uuid4())[:4] for _ in range(x_cols)]

y_rows = x_rows
y_cols = 2
y = np.random.randint(0, 2, (y_rows, y_cols), dtype=np.uint8)




args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST FOR EXCEPTS ON NON-BOOL _ignore_float_columns, _ignore_non_binary_integer_columns, _ignore_nan, _delete_axis_0, reject_unseen_values ############
if not (bypass):
    print(
        f'\n\033[92mTesting non-bool _ignore_float_columns, _ignore_non_binary_integer_columns, _ignore_nan, _delete_axis_0, reject_unseen_values\033[0m')


    def print_error(args, kwargs):
        print(f'args:{args}')
        print(f'kwargs:')
        [print(f'   {k}: {v}') for k, v in kwargs.items()]


    IGN_FLOAT_COLS = [True, False, None, 'junk', np.nan]
    IGN_BIN_INT_COLS = [True, False, None, 'junk', np.nan]
    DEL_AX_0 = [True, False, None, 'junk', np.nan]
    IGN_NAN = [True, False, None, 'junk', np.nan]
    REJ_UNSEEN = [True, False, None, 'junk', np.nan]

    total_bool_trials = 1
    for trial_set in [IGN_FLOAT_COLS, IGN_BIN_INT_COLS, DEL_AX_0, IGN_NAN, REJ_UNSEEN]:
        total_bool_trials *= len(trial_set)

    ctr = 0
    for ign_float in IGN_FLOAT_COLS:
        kwargs['ignore_float_columns'] = ign_float
        for ign_bin_int in IGN_BIN_INT_COLS:
            kwargs['ignore_non_binary_integer_columns'] = ign_bin_int
            for del_ax_0 in DEL_AX_0:
                kwargs['delete_axis_0'] = del_ax_0
                for ign_nan in IGN_NAN:
                    kwargs['ignore_nan'] = ign_nan
                    for rej_unseen in REJ_UNSEEN:
                        kwargs['reject_unseen_values'] = rej_unseen

                        # ctr += 1
                        # print(f"Running bool trial {ctr} of {total_bool_trials}")

                        TestCls = MinCountTransformer(*args, **kwargs)

                        type_error = 0
                        for _kwarg in [ign_float, ign_bin_int, ign_nan, del_ax_0, rej_unseen]:
                            if not isinstance(_kwarg, bool): type_error += 1

                        try:
                            TestCls.fit_transform(X, y)
                            assert type_error == 0, f"\033[91mMinCountTransformer did not except with bad bool inputs\033[0m"
                        except TypeError:
                            if not type_error > 0:
                                print_error(args, kwargs)
                                raise AssertionError(
                                    f"\033[91mMinCountTransformer raised TypeError with no bad bool inputs\033[0m")
                        except ValueError:
                            if not type_error > 0:
                                print_error(args, kwargs)
                                raise AssertionError(
                                    f"\033[91mMinCountTransformer raised ValueError with no bad bool inputs\033[0m")
                        except AssertionError as e1:
                            print_error(args, kwargs)
                            raise AssertionError(e1)
                        except Exception as e2:
                            print_error(args, kwargs)
                            raise Exception(
                                f"Non-bool test excepted for reason other than TypeError or AssertionError --- {e2}")

    del IGN_FLOAT_COLS, IGN_BIN_INT_COLS, DEL_AX_0, IGN_NAN, REJ_UNSEEN, total_bool_trials, trial_set, ctr, ign_float,
    del ign_bin_int, del_ax_0, ign_nan, rej_unseen, TestCls, type_error, _kwarg, print_error

    print(f'\033[92mAll bool tests passed.\033[0m')
# END TEST FOR EXCEPTS ON NON-BOOL _ignore_float_columns, _ignore_non_binary_integer_columns, _ignore_nan, _delete_axis_0, reject_unseen_values ########


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST CORRECT DTYPES ARE RETRIEVED W/ OR W/O np.nan MIXED IN #######################################################
if not (bypass):
    print(f"\n\033[92mTesting accuracy of dtypes retrieved with or without np.nan mixed into data\033[0m")
    # PASS NON-np.nan DATA AND COMPARE TO DTYPE_KEY
    TestCls = MinCountTransformer(*args, **kwargs)
    TestCls.fit(NO_NAN_X, y)
    if not np.array_equiv(TestCls.original_dtypes_, DTYPE_KEY):
        raise Exception(f"Retrieved dtypes from non-nan data do not match dtype key: "
                        f"\nRETRIEVED DTYPES = {TestCls.original_dtypes_}\nDTYPE KEY={DTYPE_KEY}")

    del TestCls

    TestCls = MinCountTransformer(*args, **kwargs)
    TestCls.fit(X, y)
    if not np.array_equiv(TestCls.original_dtypes_, DTYPE_KEY):
        raise Exception(f"Retrieved dtypes from nan data do not match dtype key: "
                        f"\nRETRIEVED DTYPES = {TestCls.original_dtypes_}\nDTYPE KEY={DTYPE_KEY}")

    del TestCls

    print(f"\033[92mTests for accuracy of dtypes passed.\033[0m")

# END TEST CORRECT DTYPES ARE RETRIEVED W/ OR W/O np.nan MIXED IN #######################################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST FOR EXCEPTS ON BAD count_threshold, max_recursions, set_output, n_jobs  #######################################
if not (bypass):

    def print_error(args, kwargs, output_type, WHOS_BAD):
        print(f'args:{args}')
        print(f'kwargs:')
        [print(f'   {k}: {v}') for k, v in kwargs.items()]
        print(f'set_output(): {output_type}')
        print(f'WHOS BAD = ', WHOS_BAD)


    TEST_KWARGS = ['count_threshold', 'max_recursions', 'output_transform', 'n_jobs']
    print(f'\n\033[92mTesting bad {", ".join(TEST_KWARGS)} inputs\033[0m')
    CT_THRESH = [-2, 1, 3, 2 * x_rows, np.pi, np.nan, 'junk']
    RECURSIONS = [-1, 0, 1, 10, np.pi, np.nan, True, 'junk']
    OUTPUT_TYPES = [None, 'default', 'dask_array', 'wrong_junk', 'NUMPY_ARRAY', 'Pandas_Dataframe', 'pandas_series',
                    np.pi, np.nan]
    N_JOBS = [-2, -1, 0, 1, np.nan, 10, float('inf')]

    total_input_trials = 1
    for trial_set in [CT_THRESH, RECURSIONS, OUTPUT_TYPES, N_JOBS]:
        total_input_trials *= len(trial_set)

    ctr = 0
    for ct_thresh in CT_THRESH:
        args = [ct_thresh]
        for recursions in RECURSIONS:
            kwargs['max_recursions'] = recursions
            for output_type in OUTPUT_TYPES:  # THIS IS SET BY THE MinCountTransformer.set_output() METHOD
                for _n_jobs in N_JOBS:
                    kwargs['n_jobs'] = _n_jobs

                    ctr += 1
                    # print(f"Running bad input trial {ctr} of {total_input_trials}")

                    TestCls = MinCountTransformer(*args, **kwargs)

                    some_bad = 0
                    WHOS_BAD = []
                    try:
                        assert int(ct_thresh) == ct_thresh
                        assert ct_thresh >= 2
                        assert ct_thresh < x_rows
                    except:
                        some_bad += 1
                        WHOS_BAD.append('ct_thresh')

                    try:
                        assert not isinstance(recursions, bool)
                        assert int(recursions) == recursions
                        assert recursions >= 1
                    except:
                        some_bad += 1
                        WHOS_BAD.append('recursions')

                    try:
                        if output_type is None:
                            pass
                        else:
                            assert output_type.lower() in ['default', 'numpy_array', 'pandas_dataframe',
                                                           'pandas_series']
                            assert output_type != 'pandas_series' and X.shape[0] != 1
                    except:
                        some_bad += 1
                        WHOS_BAD.append('output_type')

                    try:
                        assert int(_n_jobs) == _n_jobs
                        assert _n_jobs in (list(range(1, 17)) + [-1])
                    except:
                        some_bad += 1
                        WHOS_BAD.append('n_jobs')

                    try:
                        TestCls.set_output(transform=output_type)
                        TestCls.fit_transform(X, y)
                        assert some_bad == 0, f"\033[91mMinCountTransformer did not except with bad {', '.join(WHOS_BAD)} inputs\033[0m"
                    except ValueError:
                        if not some_bad > 0:
                            print_error(args, kwargs, output_type, WHOS_BAD)
                            raise AssertionError(
                                f"\033[91mMinCountTransformer raised ValueError with no bad {', '.join(TEST_KWARGS)} inputs\033[0m")
                    except AssertionError as e1:
                        print_error(args, kwargs, output_type, WHOS_BAD)
                        raise AssertionError(e1)
                    except Exception as e2:
                        print_error(args, kwargs, output_type, WHOS_BAD)
                        raise Exception(
                            f"{', '.join(TEST_KWARGS)} input test excepted for reason other than Value, NotImplemented, or Assertion Error --- {e2}")

    del CT_THRESH, RECURSIONS, OUTPUT_TYPES, N_JOBS, total_input_trials, trial_set, ctr, print_error
    del ct_thresh, args, kwargs, recursions, output_type, _n_jobs, TestCls, some_bad, WHOS_BAD

    print(f"\033[92mAll bad {', '.join(TEST_KWARGS)} input tests passed.\033[0m")

    del TEST_KWARGS
# END TEST FOR EXCEPTS ON BAD count_threshold, max_recursions, set_output, n_jobs #######################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST FOR EXCEPTS ON BAD ignore_columns / handle_as_bool ###############################################################
if not (bypass):
    print(f'\n\033[92mTesting ignore_columns / handle_as_bool inputs\033[0m')


    def print_error(args, kwargs, input_dtype, kwarg_input):
        print(f'\033[91m')
        print(f'\nargs:{args}')
        print(f'kwargs:')
        for k, v in kwargs.items():
            print(5 * f" " + f"{k}: {v}")
        print(f"input_dtype: {input_dtype}")
        print(f"kwarg_input: {kwarg_input}\n")


    INPUT_DTYPE = ['numpy', 'numpy_recarray', 'numpy_masked_array', 'pandas_dataframe', 'pandas_series']
    KWARG_INPUT = [[], [1, 3, 5], [COLUMNS[2], COLUMNS[4], COLUMNS[6]], [True, False, None, 'junk'], [1, 3, COLUMNS[6]],
                   True, False, None, 'junk', np.nan, [np.nan], {0: 'a', 1: 'b'}, [1000, 1001, 1002], 'good_callable',
                   'bad_callable']
    KWARG = ['ignore_columns', 'handle_as_bool', 'both']

    total_input_trials = 1
    for trial_set in [KWARG, INPUT_DTYPE, KWARG_INPUT]:
        total_input_trials *= len(trial_set)

    _y = y[:, 0].copy()

    ctr = 0
    for input_dtype in INPUT_DTYPE:
        for kwarg_input in KWARG_INPUT:
            for _kwarg in KWARG:

                ctr += 1
                # print(f"Running bad ignore_columns / handle_as_bool input trial {ctr} of {total_input_trials}")

                if _kwarg == 'ignore_columns':
                    if kwarg_input == 'good_callable':
                        kwargs['ignore_columns'] = lambda X: [0] if 1 in X.shape else list(range(3 * _cols, x_cols))
                    elif kwarg_input == 'bad_callable':
                        kwargs['ignore_columns'] = lambda X: 'unrecognizable junk'
                    else:
                        kwargs['ignore_columns'] = kwarg_input
                    kwargs['handle_as_bool'] = None
                elif _kwarg == 'handle_as_bool':
                    if kwarg_input == 'good_callable':
                        kwargs['handle_as_bool'] = lambda X: [0] if 1 in X.shape else list(range(_cols, 2 * _cols))
                    elif kwarg_input == 'bad_callable':
                        kwargs['handle_as_bool'] = lambda X: 'unrecognizable junk'
                    else:
                        kwargs['handle_as_bool'] = kwarg_input
                    kwargs['ignore_columns'] = None
                elif _kwarg == 'both':
                    if kwarg_input == 'good_callable':
                        kwargs['ignore_columns'] = lambda X: [0] if 1 in X.shape else list(range(3 * _cols, x_cols))
                        kwargs['handle_as_bool'] = lambda X: [0] if 1 in X.shape else list(range(_cols, 2 * _cols))
                    elif kwarg_input == 'bad_callable':
                        kwargs['ignore_columns'] = lambda X: 'unrecognizable junk'
                        kwargs['handle_as_bool'] = lambda X: 'unrecognizable junk'
                    else:
                        kwargs['ignore_columns'] = kwarg_input
                        kwargs['handle_as_bool'] = kwarg_input
                else:
                    raise Exception(f"logic managing ignore_columns and/or handle_as_bool failed")

                TestCls = MinCountTransformer(*args, **kwargs)

                if input_dtype == 'numpy':
                    X_NEW, y_NEW = X.copy(), _y
                elif input_dtype == 'numpy_recarray':
                    X_NEW = np.recarray((x_rows,), names=COLUMNS, formats=[list(
                        zip(COLUMNS, [np.uint8 for _ in range(x_cols // 2)] + ['<U1' for _ in range(x_cols // 2)]))])
                    X_NEW.data = X
                    y_NEW = _y
                elif input_dtype == 'numpy_masked_array':
                    X_NEW = np.ma.array(X, mask=False)
                    y_NEW = np.ma.array(y, mask=False)
                elif input_dtype == 'pandas_dataframe':
                    X_NEW = pd.DataFrame(data=X, columns=COLUMNS, dtype=object)
                    y_NEW = pd.DataFrame(data=_y, columns=['y'], dtype=object)
                elif input_dtype == 'pandas_series':
                    X_NEW = pd.Series(data=X[:, _cols], dtype=np.float64)
                    y_NEW = pd.Series(data=_y, dtype=object)

                type_error = 0
                value_error = 0
                try:
                    if isinstance(kwarg_input, type(None)):
                        raise UnicodeError
                    if kwarg_input == 'good_callable':  # <== THIS MUST BE AHEAD OF isinstance(kwarg_input, str)
                        raise UnicodeError  # <===
                    elif kwarg_input == 'bad_callable':  # <===
                        raise TypeError  # <===
                    if isinstance(kwarg_input, (dict, str)):
                        raise TypeError
                    try:
                        list(kwarg_input)
                    except:
                        raise TypeError
                    if isinstance(kwarg_input, list) and len(kwarg_input) == 0:
                        raise UnicodeError  # ESCAPE OUT WITHOUT INCREMENTING type_error OR value_error
                    _dtypes_ = list(map(str, map(type, kwarg_input)))
                    unq_dtypes = np.unique(_dtypes_)
                    if len(unq_dtypes) > 1:
                        raise ValueError
                    if True in [_ in unq_dtypes[0] for _ in ['int', 'float']]:
                        if False in [int(__) == __ for __ in kwarg_input]:
                            raise ValueError
                        try:
                            if max(kwarg_input) > X_NEW.shape[1]:
                                raise Exception
                        except:
                            raise ValueError
                        if _kwarg in ['handle_as_bool', 'both']:
                            for c_idx in kwarg_input:
                                if c_idx in range(3 * _cols, x_cols):
                                    raise ValueError
                    elif isinstance(kwarg_input[0], str):
                        if input_dtype == 'numpy':
                            raise ValueError
                        if True in [_ not in COLUMNS for _ in kwarg_input]:
                            raise ValueError
                        try:
                            X_NEW.columns
                            if _kwarg in ['handle_as_bool', 'both']:
                                for col_name in kwarg_input:
                                    if col_name in np.array(COLUMNS)[3 * _cols: x_cols]:
                                        raise ValueError
                        except:
                            raise ValueError
                    else:
                        raise UnicodeError
                except UnicodeError:
                    pass
                except TypeError:
                    type_error += 1
                except ValueError:
                    value_error += 1
                except Exception:
                    raise Exception(f"classifying {_kwarg if _kwarg != 'both' else ', '.join(KWARG[:2])} "
                                    f"as good or bad excepted for uncontrolled reason")

                if isinstance(X_NEW, np.recarray):
                    type_error += 1

                if isinstance(X_NEW, np.ma.core.MaskedArray):
                    type_error += 1

                try:
                    TestCls.fit_transform(X_NEW, y_NEW)
                    assert type_error + value_error == 0, f"\033[91mMinCountTransformer did not except with bad input for {_kwarg}\033[0m"
                except TypeError:
                    if not type_error > 0:
                        print_error(args, kwargs, input_dtype, kwarg_input)
                        raise AssertionError(
                            f"\033[91mMinCountTransformer excepted for TypeError but no TypeError inducing inputs\033[0m")
                except ValueError:
                    if not value_error > 0:
                        print_error(args, kwargs, input_dtype, kwarg_input)
                        raise AssertionError(
                            f"\033[91mMinCountTransformer excepted for ValueError but no ValueError inducing inputs\033[0m")
                except AssertionError as e1:
                    print_error(args, kwargs, input_dtype, kwarg_input)
                    raise AssertionError from e1
                except Exception as e2:
                    print_error(args, kwargs, input_dtype, kwarg_input)
                    raise Exception(
                        f"input test for {_kwarg} excepted for reason other than Type, Value, or Assertion Error --- {e2}")

    del print_error, INPUT_DTYPE, KWARG_INPUT, KWARG, total_input_trials, trial_set, _y, ctr, input_dtype, kwarg_input, \
        _kwarg, X_NEW, y_NEW, type_error, value_error, _dtypes_, unq_dtypes, c_idx, col_name

    print(f'\033[92mAll ignore_column / handle_as_bool input tests passed.\033[0m')
# END TEST FOR EXCEPTS ON BAD ignore_columns / handle_as_bool ###############################################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# ALWAYS ACCEPTS y==None IS PASSED TO fit(), partial_fit(), AND transform() #############################################
if not (bypass):
    print(f"\n\033[92mTest accepts y==None anytime sent to fit(), partial_fit(), and transform()\033[0m")

    try:
        TestCls = MinCountTransformer(*args, **kwargs)
        TestCls.fit(X, None)
    except:
        raise Exception(f"\033[91mfit() excepted with y==None\033[0m")

    try:
        TestCls = MinCountTransformer(*args, **kwargs)
        TestCls.partial_fit(X, y)
        TestCls.partial_fit(X, None)
    except:
        raise Exception(f"\033[91mpartial_fit() excepted with y==None\033[0m")

    try:
        TestCls = MinCountTransformer(*args, **kwargs)
        TestCls.fit(X, y)
        TestCls.partial_fit(X, None)
    except:
        raise Exception(f"\033[91mpartial_fit() excepted with y==None\033[0m")

    try:
        TestCls = MinCountTransformer(*args, **kwargs)
        TestCls.fit(X, y)
        TestCls.transform(X, None)
    except:
        raise Exception(f"\033[transform() excepted with y==None\033[0m")

    try:
        TestCls = MinCountTransformer(*args, **kwargs)
        TestCls.fit_transform(X, None)
    except:
        raise Exception(f"\033[91fit_transform() excepted with y==None\033[0m")

    del TestCls

    print(f"\033[92mAll tests passed for y==None sent to fit(), partial_fit(), and transform()\033[0m")

# END ALWAYS ACCEPTS y==None IS PASSED TO fit(), partial_fit(), AND transform() #############################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST FOR EXCEPTS ON BAD y ROWS FOR DF, SERIES, ARRAY ###################################################################
if not (bypass):
    print(f'\n\033[92mTesting bad y rows with pandas dataframe, pandas series, and numpy arrays \033[0m')

    FIRST_FIT_y_DTYPE = ['numpy', 'pandas_dataframe', 'pandas_series']
    SECOND_FIT_y_DTYPE = ['numpy', 'pandas_dataframe', 'pandas_series']
    SECOND_FIT_y_DIFF_ROWS = ['good', 'less_row', 'more_row']
    TRANSFORM_y_DTYPE = ['numpy', 'pandas_dataframe', 'pandas_series']
    TRANSFORM_y_DIFF_ROWS = ['good', 'less_row', 'more_row']

    total_input_trials = 1
    for trial_set in [FIRST_FIT_y_DTYPE, SECOND_FIT_y_DTYPE, SECOND_FIT_y_DIFF_ROWS,
                      TRANSFORM_y_DTYPE, TRANSFORM_y_DIFF_ROWS
                      ]:
        total_input_trials *= len(trial_set)


    def row_checker(y_obj, X_rows):
        if y_obj is None:
            return False
        elif y_obj.shape[0] != X_rows:
            return True
        else:
            return False


    def y_builder(y_rows: int, y_cols: int, y: np.ndarray, new_dtype: str, diff_rows: str):

        if 'int' not in str(type(y_rows)).lower() or y_rows < 1:
            raise Exception(f"y_builder(): y_rows must be int > 0")
        if 'int' not in str(type(y_cols)).lower() or y_cols < 1:
            raise Exception(f"y_builder(): y_cols must be int > 0")
        if not isinstance(y, np.ndarray):
            raise Exception("y_builder(): y going in should be a numpy array")
        if new_dtype not in ['numpy', 'pandas_dataframe', 'pandas_series', None]:
            raise Exception(f"y_builder() new_dtype invalid --- '{new_dtype}'")
        if diff_rows not in ['good', 'less_row', 'more_row']:
            raise Exception(
                f"y_builder(): invalid diff_rows '{diff_rows}', must be in {', '.join(['good', 'less_row', 'more_row'])}")

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


    variables = [
        'fst_fit_y_dtype', 'scd_fit_y_dtype', 'scd_fit_y_cols',
        'scd_fit_y_rows', 'trfm_y_dtype', 'trfm_y_cols', 'trfm_y_rows'
    ]

    ctr = 0
    for fst_fit_y_dtype in FIRST_FIT_y_DTYPE:
        for scd_fit_y_dtype in SECOND_FIT_y_DTYPE:
            for scd_fit_y_rows in SECOND_FIT_y_DIFF_ROWS:
                for trfm_y_dtype in TRANSFORM_y_DTYPE:
                    for trfm_y_rows in TRANSFORM_y_DIFF_ROWS:

                        ctr += 1
                        # if ctr % 50 == 0:
                        #     print(f"Running bad y rows trial {ctr} of {total_input_trials}")

                        TestCls = MinCountTransformer(*args, **kwargs)

                        # first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                        fst_fit_Y = y_builder(y_rows, y_cols, y.copy(), new_dtype=fst_fit_y_dtype, diff_rows='good')
                        # end first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

                        # second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                        scd_fit_Y = y_builder(y_rows, y_cols, y.copy(), new_dtype=scd_fit_y_dtype,
                                              diff_rows=scd_fit_y_rows)
                        # end second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

                        # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                        trfm_Y = y_builder(y_rows, y_cols, y.copy(), new_dtype=trfm_y_dtype, diff_rows=trfm_y_rows)
                        # end transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

                        value_error = 0
                        # True only if y rows != X rows ** ** ** ** ** ** ** ** ** ** ** ** ** **
                        value_error += row_checker(fst_fit_Y, x_rows)
                        value_error += row_checker(scd_fit_Y, x_rows)
                        value_error += row_checker(trfm_Y, x_rows)
                        # END True only if y rows != X rows ** ** ** ** ** ** ** ** ** ** ** ** ** **

                        try:
                            TestCls.partial_fit(X, fst_fit_Y)
                            TestCls.partial_fit(X, scd_fit_Y)
                            TestCls.transform(X, trfm_Y)
                            assert np.sum((
                                              value_error)) == 0, f"\033[91mMinCountTransformer did not except with bad shape/hdr inputs\033[0m"
                        except ValueError:
                            if not value_error > 0:
                                ERR_OUTPUT = dict((zip(variables,
                                                       [fst_fit_y_dtype, scd_fit_y_dtype, scd_fit_y_rows, trfm_y_dtype,
                                                        trfm_y_rows])))
                                [print(f"{var}: {value}") for var, value in ERR_OUTPUT.items()]
                                raise AssertionError(
                                    f"\033[91mMinCountTransformer raised ValueError but no ValueError inducing inputs\033[0m")
                        except AssertionError as e1:
                            ERR_OUTPUT = dict((zip(variables,
                                                   [fst_fit_y_dtype, scd_fit_y_dtype, scd_fit_y_rows, trfm_y_dtype,
                                                    trfm_y_rows])))
                            [print(f"{var}: {value}") for var, value in ERR_OUTPUT.items()]
                            raise AssertionError(e1)
                        except Exception as e2:
                            ERR_OUTPUT = dict((zip(variables,
                                                   [fst_fit_y_dtype, scd_fit_y_dtype, scd_fit_y_rows, trfm_y_dtype,
                                                    trfm_y_rows])))
                            [print(f"{var}: {value}") for var, value in ERR_OUTPUT.items()]
                            raise Exception(
                                f"y shapes & hdrs input test excepted for reason other than ValueError or AssertionError --- {e2}")

    del FIRST_FIT_y_DTYPE, SECOND_FIT_y_DTYPE, SECOND_FIT_y_DIFF_ROWS, TRANSFORM_y_DTYPE, TRANSFORM_y_DIFF_ROWS
    del total_input_trials, trial_set, row_checker, y_builder, ctr, fst_fit_y_dtype
    del scd_fit_y_dtype, scd_fit_y_rows, trfm_y_dtype, trfm_y_rows
    del TestCls, fst_fit_Y, scd_fit_Y, trfm_Y, value_error, variables

    print(f'\033[92mAll y rows input tests passed.\033[0m')
# END TEST FOR EXCEPTS ON BAD y ROWS FOR DF, SERIES, ARRAY ###################################################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST EXCEPTS ANYTIME X==None IS PASSED TO fit(), partial_fit(), AND transform() #############################################
if not (bypass):
    print(f"\n\033[92mTest TypeError for X==None is sent to fit(), partial_fit(), and transform()\033[0m")

    try:
        TestCls = MinCountTransformer(*args, **kwargs)
        TestCls.fit(None)
    except TypeError:
        pass
    except:
        raise Exception(f"\033[91mfit() excepted for a reason other than TypeError\033[0m")

    try:
        TestCls = MinCountTransformer(*args, **kwargs)
        TestCls.partial_fit(X)
        TestCls.partial_fit(None)
    except TypeError:
        pass
    except:
        raise Exception(f"\033[91mpartial_fit() excepted for a reason other than TypeError\033[0m")

    try:
        TestCls = MinCountTransformer(*args, **kwargs)
        TestCls.fit(X)
        TestCls.partial_fit(None)
    except TypeError:
        pass
    except:
        raise Exception(f"\033[91mpartial_fit() excepted for a reason other than TypeError\033[0m")

    try:
        TestCls = MinCountTransformer(*args, **kwargs)
        TestCls.fit(X)
        TestCls.transform(None)
    except TypeError:
        pass
    except:
        raise Exception(f"\033[transform() excepted for a reason other than TypeError\033[0m")

    try:
        TestCls = MinCountTransformer(*args, **kwargs)
        TestCls.fit_transform(None)
    except TypeError:
        pass
    except:
        raise Exception(f"\033[91mfit_transform() excepted for a reason other than TypeError\033[0m")

    del TestCls

    print(f"\033[92mAll tests passed for X==None sent to fit(), partial_fit(), and transform()\033[0m")

# END TEST EXCEPTS ANYTIME X==None IS PASSED TO fit(), partial_fit(), OR transform() #########################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# VERIFY ACCEPTS X AS SINGLE COLUMN / SERIES #################################################################################
if not (bypass):

    print(f'\n\033[92mTesting X is accepted as single column / series \033[0m')

    fst_fit_y = y[:, 0].copy()
    NEW_X = X[:, 0].copy()

    FIRST_FIT_X_DTYPE = ['numpy', 'pandas_dataframe', 'pandas_series']
    FIRST_FIT_X_HDR = [True, None]

    for fst_fit_x_dtype in FIRST_FIT_X_DTYPE:
        for fst_fit_x_hdr in FIRST_FIT_X_HDR:

            if fst_fit_x_dtype == 'numpy':
                if fst_fit_x_hdr:
                    continue
                else:
                    fst_fit_X = NEW_X.copy()

            if 'pandas' in fst_fit_x_dtype:
                if fst_fit_x_hdr:
                    fst_fit_X = pd.DataFrame(data=NEW_X, columns=COLUMNS[:1])
                else:
                    fst_fit_X = pd.DataFrame(data=NEW_X)
            if fst_fit_x_dtype == 'pandas_series':
                fst_fit_X = fst_fit_X.squeeze()

            TestCls = MinCountTransformer(*args, **kwargs)

            TestCls.fit_transform(fst_fit_X, fst_fit_y)

    del fst_fit_y, NEW_X, FIRST_FIT_X_DTYPE, FIRST_FIT_X_HDR, fst_fit_x_dtype, fst_fit_x_hdr, TestCls

    print(f'\033[92mAll X as single column / series input tests passed.\033[0m')
# END VERIFY ACCEPTS X AS SINGLE COLUMN / SERIES #############################################################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST FOR EXCEPTS ON BAD X SHAPES ON SERIES & ARRAY ###################################################################
if not (bypass):
    print(f'\n\033[92mTesting bad X shapes with pandas series and numpy arrays \033[0m')

    fst_fit_y = y.copy()[:, 0].ravel()
    scd_fit_y = fst_fit_y.copy()
    trfm_y = fst_fit_y.copy()

    FIRST_FIT_X_DTYPE = list(reversed(['numpy', 'pandas_series']))
    SECOND_FIT_X_DTYPE = list(reversed(['numpy', 'pandas_series']))
    SECOND_FIT_X_SAME_DIFF_COLUMNS = list(reversed(['good', 'less_col', 'more_col']))
    SECOND_FIT_X_SAME_DIFF_ROWS = list(reversed(['good', 'less_row', 'more_row']))
    TRANSFORM_X_DTYPE = list(reversed(['numpy', 'pandas_series']))
    TRANSFORM_X_SAME_DIFF_COLUMNS = list(reversed(['good', 'less_col', 'more_col']))
    TRANSFORM_X_SAME_DIFF_ROWS = list(reversed(['good', 'less_row', 'more_row']))


    def X_builder(X_rows: int, X_cols: int, X: np.ndarray, new_dtype: str, diff_cols: str, diff_rows: str):

        if 'int' not in str(type(X_rows)).lower() or X_rows < 1:
            raise Exception(f"X_builder(): X_rows must be int > 0")
        if 'int' not in str(type(X_cols)).lower() or X_cols < 1:
            raise Exception(f"X_builder(): X_cols must be int > 0")
        if not isinstance(X, np.ndarray):
            raise Exception("X_builder(): X going in should be a numpy array")
        if not isinstance(COLUMNS, list):
            raise Exception("X_builder(): COLUMNS going in should be a py list")
        if new_dtype not in ['numpy', 'pandas_dataframe', 'pandas_series', None]:
            raise Exception(
                f"X_builder(): invalid new_dtype '{new_dtype}', must be in {', '.join(['numpy', 'pandas_dataframe', 'pandas_series', None])}")
        if diff_cols not in ['good', 'less_col', 'more_col']:
            raise Exception(
                f"X_builder(): invalid diff_cols '{diff_cols}', must be in {', '.join(['good', 'less_col', 'more_col'])}")
        if diff_rows not in ['good', 'less_row', 'more_row']:
            raise Exception(
                f"X_builder(): invalid diff_rows '{diff_rows}', must be in {', '.join(['good', 'less_row', 'more_row'])}")

        if new_dtype is None:
            NEW_X = None
        else:
            if diff_cols == 'good':
                NEW_X = X.copy()
                assert NEW_X.shape[0] == X_rows
                assert NEW_X.shape[1] == X_cols
            elif diff_cols == 'less_col':
                NEW_X = X.copy()[:, :X_cols // 2]
                assert NEW_X.shape[0] == X_rows
                assert NEW_X.shape[1] == X_cols // 2
            elif diff_cols == 'more_col':
                NEW_X = np.hstack((X.copy(), X.copy()))
                assert NEW_X.shape[0] == X_rows
                assert NEW_X.shape[1] == 2 * X_cols

            if diff_rows == 'good':
                pass
            elif diff_rows == 'less_row':
                if len(NEW_X.shape) == 1:
                    NEW_X = NEW_X[:X_rows // 2]
                elif len(NEW_X.shape) == 2:
                    NEW_X = NEW_X[:X_rows // 2, :]
                else:
                    raise Exception(f"X_builder() NEW_X.shape logic failed")
                assert NEW_X.shape[0] == X_rows // 2
            elif diff_rows == 'more_row':
                if len(NEW_X.shape) == 1:
                    NEW_X = np.hstack((NEW_X, NEW_X))
                elif len(NEW_X.shape) == 2:
                    NEW_X = np.vstack((NEW_X, NEW_X))
                else:
                    raise Exception(f"X_builder() NEW_X.shape logic failed")
                assert NEW_X.shape[0] == 2 * X_rows

            if 'pandas' in new_dtype:
                NEW_X = pd.DataFrame(data=NEW_X, columns=None, dtype=object)
            if new_dtype == 'pandas_series':
                NEW_X = NEW_X.iloc[:, 0].squeeze()

        return NEW_X


    # end X_builder ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    total_input_trials = 1
    for trial_set in [FIRST_FIT_X_DTYPE, SECOND_FIT_X_DTYPE, SECOND_FIT_X_SAME_DIFF_COLUMNS,
                      SECOND_FIT_X_SAME_DIFF_ROWS,
                      TRANSFORM_X_DTYPE, TRANSFORM_X_SAME_DIFF_COLUMNS, TRANSFORM_X_SAME_DIFF_ROWS
                      ]:
        total_input_trials *= len(trial_set)

    variables = ['fst_fit_x_dtype', 'scd_fit_x_dtype', 'scd_fit_x_cols',
                 'scd_fit_x_rows', 'trfm_x_dtype', 'trfm_x_cols', 'trfm_x_rows']

    ctr = 0
    for fst_fit_x_dtype in FIRST_FIT_X_DTYPE:
        for scd_fit_x_dtype in SECOND_FIT_X_DTYPE:
            for scd_fit_x_cols in SECOND_FIT_X_SAME_DIFF_COLUMNS:
                for scd_fit_x_rows in SECOND_FIT_X_SAME_DIFF_ROWS:
                    for trfm_x_dtype in TRANSFORM_X_DTYPE:
                        for trfm_x_cols in TRANSFORM_X_SAME_DIFF_COLUMNS:
                            for trfm_x_rows in TRANSFORM_X_SAME_DIFF_ROWS:

                                ctr += 1
                                # if ctr % 50 == 0:
                                #     print(f"Running bad X rows / columns trial {ctr} of {total_input_trials}")

                                # CANT HAVE 'more_col' or 'less_col' WHEN X IS A SERIES
                                if scd_fit_x_dtype == 'pandas_series' and scd_fit_x_cols in ['less_col',
                                                                                             'more_col']: continue
                                if trfm_x_dtype == 'pandas_series' and trfm_x_cols in ['less_col', 'more_col']: continue

                                TestCls = MinCountTransformer(*args, **kwargs)

                                # first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                                fst_fit_X = X_builder(x_rows, x_cols, X.copy(), new_dtype=fst_fit_x_dtype,
                                                      diff_cols='good', diff_rows='good')
                                # end first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

                                # second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                                scd_fit_X = X_builder(x_rows, x_cols, X.copy(), new_dtype=scd_fit_x_dtype,
                                                      diff_cols=scd_fit_x_cols, diff_rows=scd_fit_x_rows)
                                # end second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

                                # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                                trfm_X = X_builder(x_rows, x_cols, X.copy(), new_dtype=trfm_x_dtype,
                                                   diff_cols=trfm_x_cols, diff_rows=trfm_x_rows)
                                # end transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

                                not_fitted_error = 0
                                value_error = 0
                                type_error = 0

                                # ValueError WHEN ROWS OF y != X ROWS UNDER ALL CIRCUMSTANCES
                                if fst_fit_X is not None and fst_fit_y.shape[0] != fst_fit_X.shape[0]:
                                    value_error += 1
                                if scd_fit_X is not None and scd_fit_y.shape[0] != scd_fit_X.shape[0]:
                                    value_error += 1
                                if trfm_X is not None and trfm_y.shape[0] != trfm_X.shape[0]:
                                    value_error += 1

                                # ValueError WHEN n_features_in_ != FIRST FIT n_features_in_ UNDER ALL OTHER CIRCUMSTANCES
                                value_error += True in [__ in ['more_col', 'less_col'] for __ in
                                                        [scd_fit_x_cols, trfm_x_cols]]

                                # ValueError WHEN COLUMNS PASSED TO i) A LATER {partial_}fit() OR 2) TRANSFORM DO NOT MATCH COLUMNS SEEN ON ANY PREVIOUS {partial_}fit()
                                # COLUMNS CANNOT BE "BAD" WHEN SEEN FOR THE FIRST TIME

                                if scd_fit_X is None:
                                    pass
                                elif scd_fit_X.shape != fst_fit_X.shape:
                                    value_error += 1
                                if trfm_X is None:
                                    pass
                                elif trfm_X.shape != fst_fit_X.shape:
                                    value_error += 1

                                try:
                                    TestCls.partial_fit(fst_fit_X, fst_fit_y)
                                    TestCls.partial_fit(scd_fit_X, scd_fit_y)
                                    TestCls.transform(trfm_X, trfm_y)
                                    assert np.sum((not_fitted_error, value_error,
                                                   type_error)) == 0, f"\033[91mMinCountTransformer did not except with bad shape/hdr inputs\033[0m"
                                except NotFittedError:
                                    if not not_fitted_error > 0:
                                        ERR_OUTPUT = dict((zip(variables,
                                                               [fst_fit_x_dtype, scd_fit_x_dtype, scd_fit_x_cols,
                                                                scd_fit_x_rows, trfm_x_dtype, trfm_x_cols,
                                                                trfm_x_rows])))
                                        [print(f"{var}: {value}") for var, value in ERR_OUTPUT.items()]
                                        raise AssertionError(
                                            f"\033[91mMinCountTransformer raised NotFittedError but no NotFittedError inducing inputs\033[0m")
                                except ValueError:
                                    if not value_error > 0:
                                        ERR_OUTPUT = dict((zip(variables,
                                                               [fst_fit_x_dtype, scd_fit_x_dtype, scd_fit_x_cols,
                                                                scd_fit_x_rows, trfm_x_dtype, trfm_x_cols,
                                                                trfm_x_rows])))
                                        [print(f"{var}: {value}") for var, value in ERR_OUTPUT.items()]
                                        raise AssertionError(
                                            f"\033[91mMinCountTransformer raised ValueError but no ValueError inducing inputs\033[0m")
                                except TypeError:
                                    if not type_error > 0:
                                        ERR_OUTPUT = dict((zip(variables,
                                                               [fst_fit_x_dtype, scd_fit_x_dtype, scd_fit_x_cols,
                                                                scd_fit_x_rows, trfm_x_dtype, trfm_x_cols,
                                                                trfm_x_rows])))
                                        [print(f"{var}: {value}") for var, value in ERR_OUTPUT.items()]
                                        raise AssertionError(
                                            f"\033[91mMinCountTransformer raised TypeError but no TypeError inducing inputs\033[0m")
                                except AssertionError as e1:
                                    print(f'not_fitted_error, value_error, type_error = ', not_fitted_error,
                                          value_error, type_error)
                                    ERR_OUTPUT = dict((zip(variables, [fst_fit_x_dtype, scd_fit_x_dtype, scd_fit_x_cols,
                                                                       scd_fit_x_rows, trfm_x_dtype, trfm_x_cols,
                                                                       trfm_x_rows])))
                                    [print(f"{var}: {value}") for var, value in ERR_OUTPUT.items()]
                                    raise AssertionError(e1)
                                except Exception as e2:
                                    ERR_OUTPUT = dict((zip(variables, [fst_fit_x_dtype, scd_fit_x_dtype, scd_fit_x_cols,
                                                                       scd_fit_x_rows, trfm_x_dtype, trfm_x_cols,
                                                                       trfm_x_rows])))
                                    [print(f"{var}: {value}") for var, value in ERR_OUTPUT.items()]
                                    raise Exception(
                                        f"X shapes & hdrs input test excepted for reason other than Value, Type, NotFitted or Assertion Error --- {e2}")

    del FIRST_FIT_X_DTYPE, SECOND_FIT_X_DTYPE, SECOND_FIT_X_SAME_DIFF_COLUMNS, SECOND_FIT_X_SAME_DIFF_ROWS
    del TRANSFORM_X_DTYPE, TRANSFORM_X_SAME_DIFF_COLUMNS, TRANSFORM_X_SAME_DIFF_ROWS
    del fst_fit_y, scd_fit_y, trfm_y, X_builder, total_input_trials, variables, TestCls, fst_fit_X, scd_fit_X, trfm_X
    del ctr, fst_fit_x_dtype, scd_fit_x_dtype, scd_fit_x_cols, scd_fit_x_rows, trfm_x_dtype, trfm_x_cols, trfm_x_rows

    print(f'\033[92mAll bad X shape input tests passed.\033[0m')
# END TEST FOR EXCEPTS ON BAD X SHAPES, ON SERIES & ARRAY ###################################################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST ValueError ANYTIME SEES A DF HEADER THAT IS DIFFERENT FROM A PREVIOUSLY-SEEN HEADER #################################
if not (bypass):
    print(f"\n\033[92mTesting ValueError anytime seeing 2 different DF headers\033[0m")
    GOOD_DF = pd.DataFrame(data=X, columns=np.char.lower(COLUMNS))
    BAD_DF = pd.DataFrame(data=X, columns=np.char.upper(COLUMNS))
    NO_HDR_DF = pd.DataFrame(data=X, columns=None)

    NAMES = ['GOOD_DF', 'BAD_DF', 'NO_HDR_DR']
    FST_FIT_X = [GOOD_DF, BAD_DF, NO_HDR_DF]
    SCD_FIT_X = [GOOD_DF, BAD_DF, NO_HDR_DF]
    TRFM_X = [GOOD_DF, BAD_DF, NO_HDR_DF]

    for fst_fit_name, fst_fit_X in zip(NAMES, SCD_FIT_X):
        for scd_fit_name, scd_fit_X in zip(NAMES, SCD_FIT_X):
            for trfm_name, trfm_X in zip(NAMES, TRFM_X):

                value_error = 0

                value_error += (scd_fit_name != fst_fit_name)
                value_error += (trfm_name != fst_fit_name)
                value_error += (trfm_name != scd_fit_name)

                try:
                    TestCls = MinCountTransformer(*args, **kwargs)
                    TestCls.partial_fit(fst_fit_X)
                    TestCls.partial_fit(scd_fit_X)
                    TestCls.transform(trfm_X)

                    TestCls = MinCountTransformer(*args, **kwargs)
                    TestCls.fit(fst_fit_X)
                    TestCls.transform(trfm_X)

                    TestCls = MinCountTransformer(*args, **kwargs)
                    TestCls.fit_transform(fst_fit_X)  # SHOULD NOT EXCEPT

                    assert value_error == 0, f'different header test did not except with ValueError inducing args'

                except ValueError:
                    if not value_error > 0:
                        print(f'fst_fit_name, scd_fit_name, trfm_name = ', fst_fit_name, scd_fit_name, trfm_name)
                        raise AssertionError(f"header tests excepted with ValueError but no ValueError inducing args")
                except AssertionError as e1:
                    print(f'fst_fit_name, scd_fit_name, trfm_name = ', fst_fit_name, scd_fit_name, trfm_name)
                    raise AssertionError(e1)
                except:
                    raise Exception(f"different header tests raised exception other than ValueError")

    del GOOD_DF, BAD_DF, NO_HDR_DF, NAMES, FST_FIT_X, SCD_FIT_X, TRFM_X, fst_fit_name, scd_fit_name, scd_fit_X
    del trfm_name, trfm_X, value_error, TestCls

    print(f"\033[92mAll tests passed for ValueError anytime seeing 2 different DF headers\033[0m")

# END TEST ValueError ANYTIME SEES A DF HEADER THAT IS DIFFERENT FROM A PREVIOUSLY-SEEN HEADER #############################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST FOR ValueError ON BAD X DF COLUMNS, ROWS ##########################################################################
if not (bypass):
    print(f'\n\033[92mTesting bad X rows / columns with pandas dataframe\033[0m')

    fst_fit_y = y.copy()[:, 0].ravel()
    scd_fit_y = fst_fit_y.copy()
    trfm_y = fst_fit_y.copy()

    SECOND_FIT_X_SAME_DIFF_COLUMNS = ['good', 'less_col', 'more_col']
    SECOND_FIT_X_SAME_DIFF_ROWS = ['good', 'less_row', 'more_row']
    TRANSFORM_X_SAME_DIFF_COLUMNS = ['good', 'less_col', 'more_col']
    TRANSFORM_X_SAME_DIFF_ROWS = ['good', 'less_row', 'more_row']


    def X_builder(X_rows: int, X_cols: int, X: np.ndarray, COLUMNS: list, diff_cols: str, diff_rows: str):

        if 'int' not in str(type(X_rows)).lower() or X_rows < 1:
            raise Exception(f"X_builder(): X_rows must be int > 0")
        if 'int' not in str(type(X_cols)).lower() or X_cols < 1:
            raise Exception(f"X_builder(): X_cols must be int > 0")
        if not isinstance(X, np.ndarray):
            raise Exception("X_builder(): X going in should be a numpy array")
        if not isinstance(COLUMNS, list):
            raise Exception("X_builder(): COLUMNS going in should be a py list")
        if diff_cols not in ['good', 'less_col', 'more_col']:
            raise Exception(
                f"X_builder(): invalid diff_cols '{diff_cols}', must be in {', '.join(['good', 'less_col', 'more_col'])}")
        if diff_rows not in ['good', 'less_row', 'more_row']:
            raise Exception(
                f"X_builder(): invalid diff_rows '{diff_rows}', must be in {', '.join(['good', 'less_row', 'more_row'])}")

        if diff_cols == 'good':
            NEW_X = X.copy()
            NEW_X_HDR = COLUMNS.copy()
            assert NEW_X.shape[0] == X_rows
            assert NEW_X.shape[1] == X_cols
        elif diff_cols == 'less_col':
            NEW_X = X.copy()[:, :X_cols // 2]
            NEW_X_HDR = COLUMNS.copy()[:X_cols // 2]
            assert NEW_X.shape[0] == X_rows
            assert NEW_X.shape[1] == X_cols // 2
        elif diff_cols == 'more_col':
            NEW_X = np.hstack((X.copy(), X.copy()))
            NEW_X_HDR = np.hstack((COLUMNS.copy(), np.array([str(uuid.uuid4())[:4] for _ in range(x_cols)])))
            assert NEW_X.shape[0] == X_rows
            assert NEW_X.shape[1] == 2 * X_cols

        if diff_rows == 'good':
            # KEEP X & HDR FROM COLUMN SECTION
            pass
        elif diff_rows == 'less_row':
            if len(NEW_X.shape) == 1:
                NEW_X = NEW_X[:X_rows // 2]
            elif len(NEW_X.shape) == 2:
                NEW_X = NEW_X[:X_rows // 2, :]
            else:
                raise Exception(f"X_builder() NEW_X.shape logic failed")
            # KEEP HDR FROM COLUMN SECTION
            assert NEW_X.shape[0] == X_rows // 2
        elif diff_rows == 'more_row':
            if len(NEW_X.shape) == 1:
                NEW_X = np.hstack((NEW_X, NEW_X))
            elif len(NEW_X.shape) == 2:
                NEW_X = np.vstack((NEW_X, NEW_X))
            else:
                raise Exception(f"X_builder() NEW_X.shape logic failed")
            # KEEP HDR FROM COLUMN SECTION
            assert NEW_X.shape[0] == 2 * X_rows

        NEW_X = pd.DataFrame(data=NEW_X, columns=NEW_X_HDR, dtype=object)

        return NEW_X


    # end X_builder ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    total_input_trials = 1
    for trial_set in [
        SECOND_FIT_X_SAME_DIFF_COLUMNS, SECOND_FIT_X_SAME_DIFF_ROWS, TRANSFORM_X_SAME_DIFF_COLUMNS,
        TRANSFORM_X_SAME_DIFF_ROWS
    ]:
        total_input_trials *= len(trial_set)

    variables = ['scd_fit_x_cols', 'scd_fit_x_rows', 'trfm_x_cols', 'trfm_x_rows']

    ctr = 0
    for scd_fit_x_cols in SECOND_FIT_X_SAME_DIFF_COLUMNS:
        for scd_fit_x_rows in SECOND_FIT_X_SAME_DIFF_ROWS:
            for trfm_x_cols in TRANSFORM_X_SAME_DIFF_COLUMNS:
                for trfm_x_rows in TRANSFORM_X_SAME_DIFF_ROWS:

                    ctr += 1
                    # if ctr % 20 == 0:
                    #     print(f"Running bad X DF rows / columns trial {ctr} of {total_input_trials}")

                    TestCls = MinCountTransformer(*args, **kwargs)

                    # first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                    fst_fit_X = X_builder(x_rows, x_cols, X.copy(), COLUMNS, diff_cols='good', diff_rows='good')
                    # end first fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

                    # second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                    scd_fit_X = X_builder(x_rows, x_cols, X.copy(), COLUMNS, diff_cols=scd_fit_x_cols,
                                          diff_rows=scd_fit_x_rows)
                    # end second fit ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

                    # transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                    trfm_X = X_builder(x_rows, x_cols, X.copy(), COLUMNS, diff_cols=trfm_x_cols, diff_rows=trfm_x_rows)
                    # end transform ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

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

                    # ValueError WHEN n_features_in_ != FIRST FIT n_features_in_ UNDER ALL OTHER CIRCUMSTANCES
                    value_error += True in [__ in ['more_col', 'less_col'] for __ in [scd_fit_x_cols, trfm_x_cols]]

                    try:
                        TestCls.partial_fit(fst_fit_X, fst_fit_y)
                        TestCls.partial_fit(scd_fit_X, scd_fit_y)
                        TestCls.transform(trfm_X, trfm_y)
                        assert value_error == 0, f"\033[91mMinCountTransformer did not except with bad shape/hdr inputs\033[0m"
                    except ValueError:
                        if not value_error > 0:
                            ERR_OUTPUT = dict(
                                (zip(variables, [scd_fit_x_cols, scd_fit_x_rows, trfm_x_cols, trfm_x_rows])))
                            [print(f"{var}: {value}") for var, value in ERR_OUTPUT.items()]
                            raise AssertionError(
                                f"\033[91mMinCountTransformer raised ValueError but no ValueError inducing inputs\033[0m")
                    except AssertionError as e1:
                        print(f'value_error, type_error = ', value_error, type_error)
                        ERR_OUTPUT = dict((zip(variables, [scd_fit_x_cols, scd_fit_x_rows, trfm_x_cols, trfm_x_rows])))
                        [print(f"{var}: {value}") for var, value in ERR_OUTPUT.items()]
                        raise AssertionError(e1)
                    except Exception as e2:
                        ERR_OUTPUT = dict((zip(variables, [scd_fit_x_cols, scd_fit_x_rows, trfm_x_cols, trfm_x_rows])))
                        [print(f"{var}: {value}") for var, value in ERR_OUTPUT.items()]
                        raise Exception(
                            f"X shapes & hdrs input test excepted for reason other than Value or Assertion Error --- {e2}")

    del SECOND_FIT_X_SAME_DIFF_COLUMNS, SECOND_FIT_X_SAME_DIFF_ROWS, TRANSFORM_X_SAME_DIFF_COLUMNS, TRANSFORM_X_SAME_DIFF_ROWS
    del fst_fit_y, scd_fit_y, trfm_y, X_builder, total_input_trials, trial_set, variables, ctr, scd_fit_x_cols, scd_fit_x_rows,
    del trfm_x_cols, trfm_x_rows, TestCls, fst_fit_X, scd_fit_X, trfm_X, value_error

    print(f'\033[92mAll bad X DF rows / columns input tests passed.\033[0m')
# END TEST FOR ValueError ON BAD X DF COLUMNS, ROWS ##########################################################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST ignore_float_columns WORKS ###################################################################################
if not (bypass):
    print(f'\n\033[92mTesting ignore_float_columns works correctly\033[0m')

    # FLOAT ONLY COLUMNS SHOULD BE 3rd GROUP OF COLUMNS
    FLOAT_ONLY_X = X[:, (2 * _cols):(3 * _cols)]

    # ignore_float_columns = False SHOULD delete all columns and rows
    kwargs['ignore_float_columns'] = False
    TestCls = MinCountTransformer(*args, **kwargs)
    try:
        TestCls.fit_transform(FLOAT_ONLY_X, y)
    except ValueError:
        pass
    except:
        raise Exception(f"ignore_float_columns=False test excepted for reason other than ValueError")

    # ignore_float_columns = True SHOULD not delete anything
    kwargs['ignore_float_columns'] = True
    TestCls = MinCountTransformer(*args, **kwargs)
    try:
        OUTPUT_FLOAT_ONLY_X, OUTPUT_FLOAT_ONLY_y = TestCls.fit_transform(FLOAT_ONLY_X, y)
        if not np.array_equiv(OUTPUT_FLOAT_ONLY_X[np.logical_not(np.isnan(OUTPUT_FLOAT_ONLY_X.astype(np.float64)))],
                              FLOAT_ONLY_X[np.logical_not(np.isnan(FLOAT_ONLY_X.astype(np.float64)))]):
            raise AssertionError(f"ignore_float_columns test fit_transform altered X when should have ignored")
        if not np.array_equiv(OUTPUT_FLOAT_ONLY_y[np.logical_not(np.isnan(OUTPUT_FLOAT_ONLY_y.astype(np.float64)))],
                              y[np.logical_not(np.isnan(y.astype(np.float64)))]):
            raise AssertionError(f"ignore_float_columns test fit_transform altered y when should have ignored")
        del OUTPUT_FLOAT_ONLY_X, OUTPUT_FLOAT_ONLY_y
    except AssertionError as e1:
        raise AssertionError(e1)
    except ValueError:
        raise Exception(f"ignore_float_columns=True excepted for ValueError when should have ignored float columns")
    except:
        raise Exception(f"ignore_float_columns=False test excepted for reason other than ValueError")

    del TestCls, FLOAT_ONLY_X

    print(f'\033[92mAll ignore_float_columns tests passed.\033[0m')
# END TEST ignore_float_columns WORKS ################################################################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST ignore_non_binary_integer_columns WORKS ###################################################################################
if not (bypass):
    print(f'\n\033[92mTesting ignore_non_binary_integer_columns works correctly\033[0m')

    # NON-BINARY INTEGER COLUMNS SHOULD BE 2nd GROUP OF COLUMNS
    NON_BIN_INT_ONLY_X = X[:, _cols:(2 * _cols)]

    # ignore_non_binary_integer_columns = False may or may not delete some rows
    kwargs['ignore_non_binary_integer_columns'] = False
    TestCls = MinCountTransformer(*args, **kwargs)
    try:
        OUTPUT_NON_BIN_INT_ONLY_X = TestCls.fit_transform(NON_BIN_INT_ONLY_X, y)[0]
        if np.array_equiv(OUTPUT_NON_BIN_INT_ONLY_X[np.logical_not(np.isnan(OUTPUT_NON_BIN_INT_ONLY_X.astype(np.float64)))],
                          NON_BIN_INT_ONLY_X[np.logical_not(np.isnan(NON_BIN_INT_ONLY_X.astype(np.float64)))]):
            raise AssertionError(
                f"fit_transform did not alter non-binary int only X when ignore_non_binary_integer_columns=False")
        del OUTPUT_NON_BIN_INT_ONLY_X
    except ValueError as e1:
        raise Exception(
            f"ignore_non_binary_integer_columns=False test excepted for no rows/columns left when should have passed test --- {e1}")
    except:
        raise Exception(f"ignore_non_binary_integer_columns=False test excepted for reason other than ValueError")

    # ignore_non_binary_integer_columns = True, ignore_nan = True SHOULD not delete anything
    kwargs['ignore_non_binary_integer_columns'] = True
    TestCls = MinCountTransformer(*args, **kwargs)
    try:
        OUTPUT_NON_BIN_INT_ONLY_X = TestCls.fit_transform(NON_BIN_INT_ONLY_X, y)[0]
        if not np.array_equiv(OUTPUT_NON_BIN_INT_ONLY_X[np.logical_not(np.isnan(OUTPUT_NON_BIN_INT_ONLY_X.astype(np.float64)))],
                              NON_BIN_INT_ONLY_X[np.logical_not(np.isnan(NON_BIN_INT_ONLY_X.astype(np.float64)))]):
            raise AssertionError(
                f"ignore_non_binary_integer_columns test fit_transform altered X when should have ignored")
        del OUTPUT_NON_BIN_INT_ONLY_X
    except ValueError:
        raise Exception(
            f"ignore_non_binary_integer_columns=True excepted for ValueError when should have ignored non-bin int columns")
    except:
        raise Exception(f"ignore_non_binary_integer_columns=False test excepted for reason other than ValueError")

    del TestCls, NON_BIN_INT_ONLY_X

    print(f'\033[92mAll ignore_non_binary_integer_columns tests passed.\033[0m')
# END TEST ignore_non_binary_integer_columns WORKS ################################################################################



args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)


# TEST ignore_nan WORKS ###########################################################################################################
if not (bypass):
    print(f'\n\033[92mTesting ignore_nan works correctly\033[0m')

    def assertion_handle(_assertion, err_msg, has_nan, nan_type, _ignore_nan):
        if not _assertion:
            print(f'\033[91m')
            print(f'has_nan = {has_nan}')
            print(f'nan_type = {nan_type}')
            print(f'_ignore_nan = {_ignore_nan}')
            raise AssertionError(err_msg)

    HAS_NAN = [False, True]
    NAN_TYPE = ['np_nan', 'str_nan']
    IGNORE_NAN = [False, True]

    kwargs['ignore_float_columns'] = False

    for has_nan in HAS_NAN:
        for nan_type in NAN_TYPE:
            # RIG A VECTOR SO THAT ONE CAT WOULD BE KEPT, ANOTHER CAT WOULD BE DELETED, AND nan WOULD BE DELETED
            NOT_DEL_VECTOR = np.random.choice([2, 3], x_rows - args[0] + 1, replace=True).astype(np.float64)

            # SPRINKLE nan INTO THIS VECTOR
            if has_nan:
                NOT_DEL_VECTOR[np.random.choice(range(x_rows - args[0] + 1), args[0] - 1, replace=False)] = np.nan
            if nan_type == 'np_nan':
                pass
            elif nan_type == 'str_nan':
                NOT_DEL_VECTOR = NOT_DEL_VECTOR.astype(object)
            else:
                raise Exception

            # STACK ON A VECTOR OF VALUES THAT WILL BE ALWAYS BE DELETED
            TEST_X = np.hstack((NOT_DEL_VECTOR, [2.5 for _ in range(args[0] - 1)])).ravel()

            del NOT_DEL_VECTOR

            TEST_Y = np.random.randint(0, 2, len(TEST_X))

            for _ignore_nan in IGNORE_NAN:

                kwargs['ignore_nan'] = _ignore_nan

                TestCls = MinCountTransformer(*args, **kwargs)

                TestCls.fit(TEST_X, TEST_Y)

                try:
                    TRFM_X, TRFM_Y = TestCls.transform(TEST_X, TEST_Y)
                except ValueError as e1:
                    assertion_handle(1 == 2, e1, has_nan, nan_type, _ignore_nan)
                except Exception as e1:
                    raise AssertionError(f"TestCls.transform(TEST_X, TEST_Y) "
                                         f"raise error other than ValueError")

                correct_x_and_y_len = x_rows - (args[0] - 1) - has_nan * np.logical_not(_ignore_nan) * (args[0] - 1)

                assertion_handle(len(TRFM_X) == correct_x_and_y_len,
                                 f"TRFM_X is not the correct length after transform",
                                 has_nan, nan_type, _ignore_nan)

                assertion_handle(len(TRFM_Y) == correct_x_and_y_len,
                                 f"TRFM_X is not the correct length after transform",
                                 has_nan, nan_type, _ignore_nan)

                if TestCls._ignore_nan == True:
                    # 2.5's SHOULD BE DELETED, BUT NOT nan
                    if nan_type == 'str_nan':
                        MASK = (TEST_X != 2.5).astype(bool)
                    elif nan_type == 'np_nan':
                        MASK = (TEST_X != 2.5).astype(bool)
                elif TestCls._ignore_nan == False:
                    # 2.5's AND nan SHOULD BE DELETED
                    if nan_type == 'str_nan':
                        MASK = ((TEST_X != 2.5) * (TEST_X.astype(str) != f'{np.nan}')).astype(bool)
                    elif nan_type == 'np_nan':
                        MASK = ((TEST_X != 2.5) * np.logical_not(np.isnan(TEST_X))).astype(bool)

                REF_X = TEST_X[MASK]
                REF_Y = TEST_Y[MASK]

                assertion_handle(len(REF_X) == correct_x_and_y_len,
                                 f"REF_X is not the correct length",
                                 has_nan, nan_type, _ignore_nan)

                assertion_handle(len(REF_Y) == correct_x_and_y_len,
                                 f"REF_X is not the correct length",
                                 has_nan, nan_type, _ignore_nan)

                del correct_x_and_y_len

                if has_nan and TestCls._ignore_nan == True:
                    NAN_MASK = np.logical_not(np.isnan(REF_X.astype(np.float64)))

                    assertion_handle(
                                     np.array_equiv(TRFM_X.ravel()[NAN_MASK].astype(np.float64),
                                          REF_X[NAN_MASK].ravel().astype(np.float64)),
                                     f"TRFM_X != EXPECTED X",
                                     has_nan, nan_type, _ignore_nan
                    )

                    assertion_handle(
                                     np.array_equiv(TRFM_Y.ravel()[NAN_MASK],
                                                    REF_Y.ravel()[NAN_MASK]),
                                     f"TRFM_Y != EXPECTED Y",
                                     has_nan, nan_type, _ignore_nan
                    )
                    del NAN_MASK
                else:
                    assertion_handle(
                                     np.array_equiv(TRFM_X.ravel().astype(np.float64),
                                              REF_X.ravel().astype(np.float64)),
                                     f"TRFM_X != EXPECTED X",
                                     has_nan, nan_type, _ignore_nan
                    )

                    assertion_handle(
                                     np.array_equiv(TRFM_Y.ravel(), REF_Y.ravel()),
                                     f"TRFM_Y != EXPECTED Y",
                                     has_nan, nan_type, _ignore_nan
                    )

    del HAS_NAN, NAN_TYPE, IGNORE_NAN, has_nan, nan_type, TEST_X, TEST_Y, _ignore_nan, TestCls
    del TRFM_X, TRFM_Y, MASK, REF_X, REF_Y, assertion_handle

    print(f'\033[92mAll ignore_nan tests passed.\033[0m')
# END TEST ignore_nan WORKS ###########################################################################################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST ignore_columns WORKS ###########################################################################################################
if not (bypass):
    print(f'\n\033[92mTesting ignore_columns works correctly\033[0m')

    # USE FLOAT AND STR COLUMNS
    NEW_X = NO_NAN_X[:, 2 * _cols:].copy()

    args = [2 * args[0]]

    # DEMONSTRATE THAT THIS THRESHOLD WILL ALTER X (AND y)
    try:
        TestCls = MinCountTransformer(*args, **kwargs)
        OUTPUT_X, OUTPUT_y = TestCls.fit_transform(NEW_X, y)
        if np.array_equiv(OUTPUT_X, NEW_X):
            raise AssertionError(f"ignore_columns X was not altered when high threshold on str columns")
        if np.array_equiv(OUTPUT_y, y):
            raise AssertionError(f"ignore_columns y was not altered when high threshold on str columns")
        del OUTPUT_X, OUTPUT_y
    except ValueError as e1:
        pass  # MANY OR ALL STR ROWS SHOULD BE DELETED
    except:
        raise Exception(f"ignore_columns test excepted for reason other than ValueError")

    kwargs['ignore_columns'] = np.arange(_cols, NEW_X.shape[1], 1)

    # DEMONSTRATE THAT WHEN THE COLUMNS ARE IGNORED THAT X (AND y) ARE NOT ALTERED
    try:
        TestCls = MinCountTransformer(*args, **kwargs)
        OUTPUT_X, OUTPUT_y = TestCls.fit_transform(NEW_X, y)
        if not np.array_equiv(OUTPUT_X, NEW_X):
            raise AssertionError(f"ignore_columns X was altered when the only columns that could change were ignored")
        if not np.array_equiv(OUTPUT_y, y):
            raise AssertionError(f"ignore_columns y was altered when the only columns that could change were ignored")
        del OUTPUT_X, OUTPUT_y
    except AssertionError as e1:
        raise AssertionError(e1)
    except ValueError:
        raise ValueError(f"ignore_columns ValueError was raised when no rows/columns should have changed")
    except:
        raise Exception(f"ignore_columns test excepted for reason other than ValueError")

    del NEW_X, TestCls

    print(f'\033[92mAll ignore_columns tests passed.\033[0m')
# END TEST ignore_columns WORKS ###########################################################################################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST handle_as_bool WORKS ##############################################################################################
if not (bypass):
    print(f'\n\033[92mTesting handle_as_bool works correctly\033[0m')

    kwargs['ignore_non_binary_integer_columns'] = False

    # USE NON_BINARY_INT COLUMNS
    NEW_X = NO_NAN_X[:, _cols:2 * _cols].copy()

    # RIG ONE OF THE COLUMNS WITH ENOUGH ZEROS THAT IT WOULD BE DELETED WHEN HANDLED AS AN INT --- BECAUSE
    # EACH INT WOULD BE < count_threshold, DELETING THEM, LEAVING A COLUMN OF ALL ZEROS, WHICH WOULD THEN BE DELETED
    RIGGED_INTEGERS = np.zeros(x_rows, dtype=np.uint32)
    for row_idx in range(1, args[0] + 2):
        RIGGED_INTEGERS[row_idx] = row_idx
    NEW_X[:, -1] = RIGGED_INTEGERS
    del RIGGED_INTEGERS

    # DEMONSTRATE THAT ONE CHOP WHEN NOT HANDLED AS BOOL WILL SHRINK ROWS AND ALSO DELETE 1 COLUMN FROM X
    kwargs['handle_as_bool'] = None
    TestCls = MinCountTransformer(*args, **kwargs)
    TRFM_X = TestCls.fit_transform(NEW_X)
    if TRFM_X.shape[1] != NEW_X.shape[1] - 1:
        raise AssertionError(f"handle_as_bool X was not shrunk by 1 column by test chop")
    if not TRFM_X.shape[0] < NEW_X.shape[0]:
        raise AssertionError(f"handle_as_bool X rows was not shortened by test chop")
    del TRFM_X, TestCls

    # DEMONSTRATE THAT WHEN ZERO-PEPPERED COLUMN IS HANDLED AS A BOOL, THE COLUMN IS RETAINED
    kwargs['handle_as_bool'] = [NEW_X.shape[1] - 1]
    TestCls = MinCountTransformer(*args, **kwargs)
    TRFM_X = TestCls.fit_transform(NEW_X)
    if not TRFM_X.shape[1] == NEW_X.shape[1]:
        raise AssertionError(f"handle_as_bool X --- a column was deleted when shouldnt have")

    # TEST handle_as_bool CANNOT BE USED ON STR ('obj') COLUMNS ** ** ** ** ** ** ** ** ** ** ** ** **
    # STR COLUMNS SHOULD BE [:, 3*_cols:] ON ORIGINAL X
    for col_idx in [_cols - 1, 2 * _cols - 1, 3 * _cols - 1,
                    3 * _cols]:  # PICK ONE COLUMN IS STR; ONE EACH FROM BIN-INT, INT, AND FLOAT
        kwargs['handle_as_bool'] = [col_idx]
        TestCls = MinCountTransformer(*args, **kwargs)
        try:
            TestCls.fit(X, y)
            if col_idx in range(3 * _cols, x_cols):
                raise AssertionError(f"handle_as_bool should have raised ValueError for string column but did not")
        except AssertionError as e1:
            raise AssertionError(e1)
        except ValueError:
            if col_idx in range(3 * _cols, x_cols):
                pass
            else:
                raise AssertionError(f"handle_as_bool raised ValueError for non-string column but should not")
        except Exception as e2:
            raise Exception(
                f"handle_as_bool wrongly allowed for str columns should have raised ValueError but raised --- {e2}")

        del TestCls

    # DEMONSTRATE THAT AFTER fit() WITH VALID handle_as_bool, IF handle_as_bool IS CHANGED TO INVALID, RAISES ValueError
    kwargs['handle_as_bool'] = [_cols + 1]  # A NON-BINARY INT COLUMN
    TestCls = MinCountTransformer(*args, **kwargs)
    try:
        TestCls.partial_fit(X)
    except:
        raise AssertionError(f"handle_as_bool with valid columns excepted when fitting and shouldnt have")

    try:
        TestCls.set_params(handle_as_bool=[x_cols - 1])  # STR COLUMN
        raise AssertionError(
            f"handle_as_bool accepted invalid columns on partial_fit() did not except with ValueError --- {e1}")
    except ValueError:
        pass
    except Exception as e1:
        raise AssertionError(
            f"handle_as_bool with invalid columns on partial_fit() excepted for reason other than ValueError --- {e1}")

    TestCls._handle_as_bool = [x_cols - 1]  # STR COLUMN

    try:
        TestCls.partial_fit(X)
        raise AssertionError(
            f"handle_as_bool accepted invalid columns on partial_fit() did not except with ValueError --- {e1}")
    except ValueError:
        pass
    except Exception as e1:
        raise AssertionError(
            f"handle_as_bool with invalid columns on partial_fit() excepted for reason other than ValueError --- {e1}")

    TestCls._handle_as_bool = [x_cols - 1]  # STR COLUMN

    try:
        TestCls.transform(X)
        raise AssertionError(
            f"handle_as_bool accepted invalid columns on transform() and did not except with ValueError --- {e1}")
    except ValueError:
        pass
    except Exception as e1:
        raise AssertionError(
            f"handle_as_bool with invalid columns on transform() excepted for reason other than ValueError --- {e1}")

    del NEW_X, TestCls, TRFM_X, col_idx

    print(f'\033[92mAll handle_as_bool tests passed.\033[0m')
# END TEST handle_as_bool WORKS #########################################################################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST delete_axis_0 WORKS ###########################################################################################################
if not (bypass):
    print(f'\n\033[92mTesting delete_axis_0 works correctly\033[0m')

    # USE FLOAT AND STR COLUMNS, DUMMY THE STRS AND RUN delete_axis_0 True, DO THE SAME WHEN STRS ARE NOT DUMMIED,
    # BUT ONLY USE FLOAT COLUMNS TO SEE IF THE RESULTS ARE EQUAL; PROVE NO ROWS ARE DELETED FROM DUMMIED WHEN delete_axis_0 is False
    FLOAT_STR_X = NO_NAN_X[:, 2 * _cols:4 * _cols].copy()
    FLOAT_STR_COLUMNS = COLUMNS[2 * _cols:4 * _cols].copy()
    FLOAT_STR_DF = pd.DataFrame(data=FLOAT_STR_X, columns=FLOAT_STR_COLUMNS)
    FLOAT_DF = pd.DataFrame(data=FLOAT_STR_X[:, :_cols],
                            columns=FLOAT_STR_COLUMNS[:_cols])  # "TRUTH" for when delete_axis_0 = False
    STR_DF = pd.DataFrame(data=FLOAT_STR_X[:, _cols:], columns=FLOAT_STR_COLUMNS[_cols:])
    del FLOAT_STR_X, FLOAT_STR_COLUMNS

    # get remaining float rows after strs are chopped with MinCountTransformer
    # THIS IS SUPPOSED TO BE THE "TRUTH" FOR WHEN delete_axis_0 = True
    ChopStrTestCls = MinCountTransformer(*args, **kwargs)
    STR_MIN_COUNTED_X = ChopStrTestCls.fit_transform(FLOAT_STR_DF)
    STR_MIN_COUNTED_X_DF = pd.DataFrame(data=STR_MIN_COUNTED_X, columns=ChopStrTestCls.get_feature_names_out(None))
    del ChopStrTestCls, STR_MIN_COUNTED_X, _, FLOAT_STR_DF
    STR_MIN_COUNTED_FLOAT_DF = STR_MIN_COUNTED_X_DF.iloc[:, :_cols]  # "TRUTH" for when delete_axis_0 = True
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

    DUMMIED_STR_DF = pd.DataFrame(data=onehot.transform(STR_DF), columns=onehot.get_feature_names_out())
    FULL_DUMMIED_STR_DF = pd.concat((FLOAT_DF, DUMMIED_STR_DF), axis=1)
    del onehot, STR_DF, DUMMIED_STR_DF

    kwargs['delete_axis_0'] = True
    ChopDummyDeleteAxis0TestCls = MinCountTransformer(*args, **kwargs)
    FULL_DUM_MIN_COUNTED_DELETE_0_X, _ = ChopDummyDeleteAxis0TestCls.fit_transform(FULL_DUMMIED_STR_DF, y)
    FULL_DUMMIED_DELETE_0_DF = pd.DataFrame(data=FULL_DUM_MIN_COUNTED_DELETE_0_X,
                                            columns=ChopDummyDeleteAxis0TestCls.get_feature_names_out(None))
    del ChopDummyDeleteAxis0TestCls, FULL_DUM_MIN_COUNTED_DELETE_0_X, _
    DUM_MIN_COUNTED_DELETE_0_FLOAT_DF = FULL_DUMMIED_DELETE_0_DF.iloc[:,
                                        :_cols]  # COMPARE AGAINST STR_MIN_COUNTED_FLOAT_DF
    del FULL_DUMMIED_DELETE_0_DF

    kwargs['delete_axis_0'] = False
    ChopDummyDontDeleteAxis0TestCls = MinCountTransformer(*args, **kwargs)
    FULL_DUM_MIN_COUNTED_DONT_DELETE_0_X, _ = ChopDummyDontDeleteAxis0TestCls.fit_transform(FULL_DUMMIED_STR_DF, y)
    FULL_DUMMIED_DONT_DELETE_0_DF = pd.DataFrame(data=FULL_DUM_MIN_COUNTED_DONT_DELETE_0_X,
                                                 columns=ChopDummyDontDeleteAxis0TestCls.get_feature_names_out(None))
    del ChopDummyDontDeleteAxis0TestCls, FULL_DUM_MIN_COUNTED_DONT_DELETE_0_X, _
    DUM_MIN_COUNTED_DONT_DELETE_0_FLOAT_DF = FULL_DUMMIED_DONT_DELETE_0_DF.iloc[:, :_cols]  # COMPARE AGAINST FLOAT_DF
    del FULL_DUMMIED_DONT_DELETE_0_DF

    # COMPARE:
    # 1) Ensure some rows were actually deleted by comparing STR_MIN_COUNTED_FLOAT_DF against FLOAT_DF
    # 2) Compare DUM_MIN_COUNTED_DELETE_0_FLOAT_DF against STR_MIN_COUNTED_FLOAT_DF
    # 3) Compare DUM_MIN_COUNTED_DONT_DELETE_0_FLOAT_DF against FLOAT_DF

    assert not STR_MIN_COUNTED_FLOAT_DF.equals(FLOAT_DF), \
        f"MinCountTransform of FLOAT_STR_DF did not delete any rows"
    assert DUM_MIN_COUNTED_DELETE_0_FLOAT_DF.equals(STR_MIN_COUNTED_FLOAT_DF), \
        f"rows after MinCount on dummies with delete_axis_0=True do not equal rows from MinCount on strings"
    assert DUM_MIN_COUNTED_DONT_DELETE_0_FLOAT_DF.equals(FLOAT_DF), \
        f"rows after MinCount on dummies with delete_axis_0=False do not equal original rows"

    del FLOAT_DF, FULL_DUMMIED_STR_DF, STR_MIN_COUNTED_FLOAT_DF, DUM_MIN_COUNTED_DELETE_0_FLOAT_DF
    del DUM_MIN_COUNTED_DONT_DELETE_0_FLOAT_DF

    print(f'\033[92mAll delete_axis_0 tests passed.\033[0m')
# END TEST delete_axis_0 WORKS ###########################################################################################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST OUTPUT TYPES ##################################################################################################################
if not (bypass):
    print(f'\n\033[92mRunning output types tests\033[0m')
    NEW_X = X[:, 0].copy()
    NEW_COLUMNS = X[:1].copy()
    NEW_Y = y[:, 0].copy()

    X_INPUT_TYPES = ['numpy_array', 'pandas_dataframe', 'pandas_series']
    Y_INPUT_TYPES = ['numpy_array', 'pandas_dataframe', 'pandas_series']
    OUTPUT_TYPES = [None, 'default', 'numpy_array', 'pandas_dataframe', 'pandas_series']

    total_tests = 1
    for trial_set in [X_INPUT_TYPES, Y_INPUT_TYPES, OUTPUT_TYPES]:
        total_tests *= len(trial_set)

    ctr = 0
    for x_input_type in X_INPUT_TYPES:
        if x_input_type == 'numpy_array':
            TEST_X = NEW_X.copy()
        elif 'pandas' in x_input_type:
            TEST_X = pd.DataFrame(data=NEW_X, columns=NEW_COLUMNS)
            if x_input_type == 'pandas_series':
                TEST_X = TEST_X.squeeze()

        for y_input_type in Y_INPUT_TYPES:
            if y_input_type == 'numpy_array':
                TEST_Y = NEW_Y.copy()
            elif 'pandas' in y_input_type:
                TEST_Y = pd.DataFrame(data=NEW_Y, columns=['y'])
                if y_input_type == 'pandas_series':
                    TEST_Y = TEST_Y.squeeze()

            for output_type in OUTPUT_TYPES:

                ctr += 1
                # if ctr % 10 == 0:
                #     print(f'Running output type test {ctr} of {total_tests}')

                TestCls = MinCountTransformer(*args, **kwargs)
                TestCls.set_output(transform=output_type)

                TRFM_X, TRFM_Y = TestCls.fit_transform(TEST_X, TEST_Y)

                if output_type is None:
                    assert type(TRFM_X) == type(
                        TEST_X), f"output_type is None, X output type ({type(TRFM_X)}) != X input type ({type(TEST_X)})"
                    assert type(TRFM_Y) == type(
                        TEST_Y), f"output_type is None, Y output type ({type(TRFM_Y)}) != Y input type ({type(TEST_Y)})"
                elif output_type in ['default', 'numpy_array']:
                    assert isinstance(TRFM_X,
                                      np.ndarray), f"output_type is default or numpy_array, TRFM_X is {type(TRFM_X)}"
                    assert isinstance(TRFM_Y,
                                      np.ndarray), f"output_type is default or numpy_array, TRFM_Y is {type(TRFM_Y)}"
                elif output_type == 'pandas_dataframe':
                    # pandas.core.frame.DataFrame
                    assert isinstance(TRFM_X,
                                      pd.core.frame.DataFrame), f"output_type is pandas dataframe, TRFM_X is {type(TRFM_X)}"
                    assert isinstance(TRFM_Y,
                                      pd.core.frame.DataFrame), f"output_type is pandas dataframe, TRFM_Y is {type(TRFM_Y)}"
                elif output_type == 'pandas_series':
                    # pandas.core.series.Series
                    assert isinstance(TRFM_X,
                                      pd.core.series.Series), f"output_type is pandas series, TRFM_X is {type(TRFM_X)}"
                    assert isinstance(TRFM_Y,
                                      pd.core.series.Series), f"output_type is pandas sereis, TRFM_Y is {type(TRFM_Y)}"

    del NEW_X, NEW_COLUMNS, NEW_Y, X_INPUT_TYPES, Y_INPUT_TYPES, OUTPUT_TYPES, total_tests, trial_set, ctr
    del x_input_type, TEST_X, y_input_type, TEST_Y, output_type, TestCls, TRFM_X, TRFM_Y
    print(f'\033[92mAll output types tests passed.\033[0m')

# TEST OUTPUT TYPES ##################################################################################################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST CONDITIONAL ACCESS TO partial_fit() AND fit() ###########################################################################
# 1) partial_fit() should allow unlimited number of subsequent partial_fits()
# 2) one call to fit() should allow subsequent attempts to partial_fit()
# 3) one call to fit() should allow later attempts to fit() (2nd fit will reset)
# 4) calls to partial_fit() should allow later attempt to fit() (fit will reset)
# 5) fit_transform() should allow calls ad libido
if not (bypass):
    print(f'\n\033[92mTesting conditional access to fit() and partial_fit()\033[0m')

    TestCls = MinCountTransformer(*args, **kwargs)
    TEST_X = X.copy()
    TEST_Y = y.copy()

    # 1)
    try:
        for _ in range(5):
            TestCls.partial_fit(TEST_X, TEST_Y)
    except:
        raise AssertionError(
            f"multiple attempts to perform partial_fit() starting with a clean instance was disallowed")

    del TestCls

    # 2)
    TestCls = MinCountTransformer(*args, **kwargs)
    TestCls.fit(TEST_X, TEST_Y)
    try:
        TestCls.partial_fit(TEST_X, TEST_Y)
    except ValueError:
        pass
    except Exception as e1:
        raise AssertionError(
            f"attempt to partial_fit() after a fit() was already done raised an error other than Value or Assertion Error --- {e1}")

    del TestCls

    # 3)
    TestCls = MinCountTransformer(*args, **kwargs)
    TestCls.fit(TEST_X, TEST_Y)
    try:
        TestCls.fit(TEST_X, TEST_Y)
    except ValueError:
        raise AssertionError(f"fit() was disallowed after a fit() was already done on the same instance")
    except Exception as e1:
        raise AssertionError(
            f"attempt to fit() again after a fit() was already done raised an error other than ValueError --- {e1}")

    del TestCls

    # 4) a call to fit() after a previous partial_fit() should be denied
    TestCls = MinCountTransformer(*args, **kwargs)
    TestCls.partial_fit(TEST_X, TEST_Y)
    try:
        TestCls.fit(TEST_X, TEST_Y)
    except ValueError:
        raise AssertionError(f"fit() was disallowed after a partial_fit() was already done on the same instance")
    except Exception as e1:
        raise AssertionError(
            f"attempt to fit() after a partial_fit() was already done raised an error other than ValueError --- {e1}")

    # 5) fit transform should allow calls ad libido
    try:
        for _ in range(5):
            TestCls.fit_transform(TEST_X, TEST_Y)
    except:
        raise AssertionError(f"multiple access to fit_transform() on the same instance was disallowed")

    del TEST_X, TEST_Y, TestCls

    print(f'\033[92mConditional access to fit() and partial_fit() tests passed\033[0m')
# END TEST CONDITIONAL ACCESS TO partial_fit() AND fit() #######################################################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST CONDITIONAL ACCESS TO RECURSION ###############################################################################
# 1) access to partial_fit(), fit() or transform() when max_recursions > 1 is blocked
# 2) access fit & transform when max_recursions > 1 can only be through fit_transform()
# 3) access to partial_fit(), fit() or transform() when max_recursions == 1 is not blocked
if not bypass:
    print(f'\n\033[92mTesting conditional access to recursion\033[0m')
    kwargs['max_recursions'] = 3

    TEST_X = X.copy()
    TEST_Y = y.copy()

    TestCls = MinCountTransformer(*args, **kwargs)

    # 1)
    for _name, _method in zip(['partial_fit', 'fit', 'transform'],
                      [TestCls.partial_fit, TestCls.fit, TestCls.transform]):
        try:
            _method(TEST_X, TEST_Y)
            raise AssertionError
        except ValueError:
            pass
        except AssertionError:
            raise AssertionError(
                f"{_name}() should have raised ValueError for accessing with "
                f"max_recursions > 1 but did not"
            )
        except Exception as e1:
            raise AssertionError(
                f"max_recursions > 1 --- {_name}() should have raised "
                f"ValueError but raised a different error"
            ) from e1

    # 2)
    try:
        for _ in range(5):
            TestCls.fit_transform(TEST_X, TEST_Y)
    except ValueError:
        raise AssertionError(
            f"max_recursions > 1 --- fit_transform() raised a ValueError when "
            f"should have passed the test")
    except Exception as e1:
        raise AssertionError(f"max_recursions > 1 --- fit-transform() raised "
                             f"an error other than ValueError ") from e1

    # 3)
    kwargs['max_recursions'] = 1
    TestCls = MinCountTransformer(*args, **kwargs)
    for _name, cls_method in zip(['fit', 'partial_fit', 'transform'],
                     [TestCls.fit, TestCls.partial_fit, TestCls.transform]):
        try:
            cls_method(TEST_X, TEST_Y)
        except ValueError:
            raise AssertionError(
                f"max_recursions == 1 --- fit_transform() raised a ValueError "
                f"when should have passed the test"
            )
        except Exception as e1:
            raise AssertionError(
                f"max_recursions == 1 --- fit-transform() raised an error other "
                f"than ValueError "
            ) from e1

    del TEST_X, TEST_Y, TestCls, _method, _name, cls_method, _

    print(f'\033[92mConditional access to recursion tests passed\033[0m')
# END TEST CONDITIONAL ACCESS TO RECURSION ####################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST ALL COLUMNS WILL BE DELETED ############################################
if not bypass:
    print(f'\n\033[92mTesting when all columns will be deleted\033[0m')

    # CREATE VERY SPARSE DATA
    TEST_X = np.zeros((x_rows, x_cols), dtype=np.uint8)
    TEST_Y = np.random.randint(0, 2, x_rows)

    for col_idx in range(x_cols):
        TEST_X[np.random.choice(range(x_rows), 2, replace=False), col_idx] = 1

    TestCls = MinCountTransformer(*args, **kwargs)
    TestCls.fit(TEST_X, TEST_Y)

    TestCls.test_threshold()
    print(f'^^^ mask building instructions should be displayed above ^^^')

    try:
        TestCls.transform(TEST_X, TEST_Y)
        raise AttributeError
    except ValueError:
        pass
    except AttributeError:
        raise AttributeError(f"transform should have raised ValueError for all "
                             f"columns will be deleted"
        )
    except Exception as e1:
        raise AttributeError(
            f"transform should have raised ValueError for all columns will be "
            f"deleted but instead raised --- {e1}"
        )

    del TEST_X, TEST_Y, col_idx, TestCls

    print(f'\033[92mAll tests when all columns will be deleted passed.\033[0m')

# TEST ALL COLUMNS WILL BE DELETED ############################################


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST ALL ROWS WILL BE DELETED ###############################################
if not bypass:
    print(f'\n\033[92mTesting when all rows will be deleted\033[0m')

    # ALL FLOATS
    TEST_X = np.random.uniform(0, 1, (x_rows, x_cols))
    TEST_Y = np.random.randint(0, 2, x_rows)

    kwargs['ignore_float_columns'] = False
    TestCls = MinCountTransformer(*args, **kwargs)
    TestCls.fit(TEST_X, TEST_Y)

    TestCls.test_threshold()
    print(f'^^^ mask building instructions should be displayed above ^^^')

    try:
        TestCls.transform(TEST_X, TEST_Y)
        raise AttributeError
    except ValueError:
        pass
    except AttributeError:
        raise AttributeError(
            f"transform should have raised ValueError for all rows will be deleted"
        )
    except Exception as e1:
        raise AttributeError(
            f"transform should have raised ValueError for all rows will be "
            f"deleted but instead raised --- {e1}"
        )

    del TEST_X, TEST_Y, TestCls

    print(f'\033[92mAll tests when all rows will be deleted passed.\033[0m')

# TEST ALL ROWS WILL BE DELETED ###############################################



args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)


# TEST BIN INT COLUMN WITH ALL ABOVE THRESHOLD NOT DELETED ####################
if not bypass:
    print(f'\n\033[92mTesting bin-int column with all above threshold is not deleted\033[0m')
    args = [2]
    TestCls = MinCountTransformer(*args, **kwargs)

    NEW_X = np.array([['a',0], ['b',0], ['a',1], ['b',1], ['c',0]], dtype=object)
    NEW_Y = np.array([0, 1, 0, 1, 1], dtype=np.uint8)

    TestCls.fit(NEW_X, NEW_Y)

    TRFM_X, TRFM_Y = TestCls.transform(NEW_X, NEW_Y)

    assert TRFM_X.shape[1]==2, f"bin int column with all values above threshold was deleted"
    assert TRFM_X.shape[0]==4, f"TRFM_X should have 4 rows but has {TRFM_X.shape[0]}"

    print(f'\033[92mbin-int test passed\033[0m')
# END TEST BIN INT COLUMN WITH ALL ABOVE THRESHOLD NOT DELETED ################




args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)


# TEST ACCURACY ***************************************************************
if True:
    print(f'\n\033[92mRunning accuracy tests\033[0m')

    CT_THRESH = [2, 3]
    IGN_FLOAT_COLS = [True, False]
    IGN_NON_BIN_INT_COLS = [True, False]
    IGN_COLS = [None, [0, 1, 2, 3]]
    IGN_NAN = [True, False]
    HANDLE_AS_BOOL = [None, list(range(_cols, 2 * _cols)), lambda X: list(range(_cols, 2 * _cols))]
    DEL_AX_0 = [False, True]
    RECURSIONS = [1, 2]

    total_tests = 1
    for trial_set in [CT_THRESH, IGN_FLOAT_COLS, IGN_NON_BIN_INT_COLS, IGN_COLS,
                      IGN_NAN, HANDLE_AS_BOOL, DEL_AX_0, RECURSIONS]:
        total_tests *= len(trial_set)

    TEST_X = X.copy()
    TEST_Y = y.copy()

    ctr = 0
    for count_threshold in CT_THRESH:
        args = [count_threshold]
        for ignore_float_columns in IGN_FLOAT_COLS:
            kwargs['ignore_float_columns'] = ignore_float_columns
            for ignore_non_binary_integer_columns in IGN_NON_BIN_INT_COLS:
                kwargs['ignore_non_binary_integer_columns'] = \
                    ignore_non_binary_integer_columns
                for ignore_columns in IGN_COLS:
                    kwargs['ignore_columns'] = ignore_columns
                    for ignore_nan in IGN_NAN:
                        kwargs['ignore_nan'] = ignore_nan
                        for handle_as_bool in HANDLE_AS_BOOL:
                            kwargs['handle_as_bool'] = handle_as_bool
                            for delete_axis_0 in DEL_AX_0:
                                kwargs['delete_axis_0'] = delete_axis_0
                                for max_recursions in RECURSIONS:
                                    kwargs['max_recursions'] = max_recursions

                                    ctr += 1
                                    if ctr % 30 == 0:
                                        print(f'Running accuracy test {ctr} of {total_tests}')

                                    TestCls = MinCountTransformer(*args, **kwargs)
                                    TRFM_X, TRFM_Y = TestCls.fit_transform(TEST_X, TEST_Y)

                                    ###########################################
                                    # MANUALLY OPERATE ON MOCK_X & MOCK_Y #####
                                    MOCK_X = X.copy()
                                    MOCK_Y = y.copy()

                                    try:
                                        _ignore_columns = ignore_columns(TEST_X)
                                    except:
                                        _ignore_columns = ignore_columns
                                    try:
                                        _handle_as_bool = handle_as_bool(TEST_X)
                                    except:
                                        _handle_as_bool = handle_as_bool

                                    if max_recursions == 1:
                                        MOCK_X, MOCK_Y = mmct().trfm(
                                            MOCK_X, MOCK_Y, _ignore_columns, ignore_nan,
                                            ignore_non_binary_integer_columns,
                                            ignore_float_columns, _handle_as_bool,
                                            delete_axis_0, count_threshold
                                        )

                                    elif max_recursions == 2:

                                        mmct_first_rcr = mmct()
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
                                            new_ignore_columns = np.arange(len(NEW_COLUMN_MASK))[OG_IGN_COL_MASK[NEW_COLUMN_MASK]]

                                            OG_H_A_B_MASK = np.zeros(x_cols).astype(bool)
                                            OG_H_A_B_MASK[_handle_as_bool] = True
                                            new_handle_as_bool = np.arange(len(NEW_COLUMN_MASK))[OG_H_A_B_MASK[NEW_COLUMN_MASK]]

                                            del NEW_COLUMN_MASK, OG_IGN_COL_MASK, OG_H_A_B_MASK

                                        MOCK_X, MOCK_Y = mmct().trfm(
                                            MOCK_X1, MOCK_Y1, new_ignore_columns, ignore_nan,
                                            ignore_non_binary_integer_columns,
                                            ignore_float_columns, new_handle_as_bool,
                                            delete_axis_0, count_threshold
                                        )

                                        del MOCK_X1, MOCK_Y1, new_ignore_columns, new_handle_as_bool

                                    else:
                                        raise Exception(f"Test is not designed "
                                            f"to handle more than 2 recursions")
                                    # END MANUALLY OPERATE ON MOCK_X & MOCK_Y #
                                    ###########################################

                                    assertion_error = 0
                                    X_error, Y_error = 0, 0
                                    if not np.array_equiv(TRFM_X.astype(str), MOCK_X.astype(str)):
                                        assertion_error += 1
                                        X_error += 1

                                    if max_recursions > 1 and assertion_error:
                                        kwargs['max_recursions'] = 1
                                        NewTestCls_1 = MinCountTransformer(*args,**kwargs)
                                        NEW_TRFM_X, NEW_TRFM_Y = NewTestCls_1.fit_transform(TEST_X, TEST_Y)

                                        # ADJUST ign_columns & handle_as_bool FOR 2ND RECURSION
                                        NEW_COLUMN_MASK = NewTestCls_1.get_support(True)

                                        OG_IGN_COL_MASK = np.zeros(x_cols).astype(bool)
                                        OG_IGN_COL_MASK[_ignore_columns] = True
                                        new_ignore_columns = np.arange(len(NEW_COLUMN_MASK))[OG_IGN_COL_MASK[NEW_COLUMN_MASK]]

                                        OG_H_A_B_MASK = np.zeros(x_cols).astype(bool)
                                        OG_H_A_B_MASK[_handle_as_bool] = True
                                        new_handle_as_bool = np.arange(len(NEW_COLUMN_MASK))[OG_H_A_B_MASK[NEW_COLUMN_MASK]]

                                        del NEW_COLUMN_MASK, OG_IGN_COL_MASK, OG_H_A_B_MASK

                                        kwargs['ignore_columns'] = new_ignore_columns
                                        kwargs['handle_as_bool'] = new_handle_as_bool
                                        NewTestCls_2 = MinCountTransformer(*args,**kwargs)
                                        NEW_TRFM_X, NEW_TRFM_Y = NewTestCls_2.fit_transform(NEW_TRFM_X, NEW_TRFM_Y)

                                        if not np.array_equiv(NEW_TRFM_X[np.logical_not(np.char.lower(NEW_TRFM_X.astype(str)) == 'nan')],
                                                              MOCK_X[np.logical_not(np.char.lower(MOCK_X.astype(str)) == 'nan')]
                                                              ):
                                            print(f'\n\033[91m*** {max_recursions}X PASSES THRU TestCls WITH max_recursions=1 ALSO FAILED ***\033[0m')
                                        else:
                                            print(f'\n\033[93m*** {max_recursions}X PASSES THRU TestCls WITH max_recursions=1 DID NOT FAIL ***\033[0m')

                                        del NEW_TRFM_X, NEW_TRFM_Y

                                    del _ignore_columns, _handle_as_bool

                                    if not np.array_equiv(TRFM_Y[np.logical_not(np.char.lower(TRFM_Y.astype(str))=='nan')],
                                                          MOCK_Y[np.logical_not(np.char.lower(MOCK_Y.astype(str))=='nan')]):
                                        assertion_error += 1
                                        Y_error += 1

                                    if assertion_error:
                                        INFO_DICT = {
                                            f'iteration': ctr,
                                            f'count_threshold': count_threshold,
                                            f'ignore_float_columns': ignore_float_columns,
                                            f'ignore_non_binary_integer_columns': ignore_non_binary_integer_columns,
                                            f'ignore_columns': ignore_columns,
                                            f'ignore_nan': ignore_nan,
                                            f'handle_as_bool': str(handle_as_bool),
                                            f'delete_axis_0': delete_axis_0,
                                            f'max_recursions': max_recursions,
                                            f'TRFM_X.shape': str(TRFM_X.shape),
                                            f'MOCK_X.shape': str(MOCK_X.shape),
                                            f'TRFM_Y.shape': str(TRFM_Y.shape),
                                            f'MOCK_Y.shape': str(MOCK_Y.shape)
                                        }

                                        INFO_DF = pd.DataFrame(data=np.array(list(INFO_DICT.items())),
                                                               columns=['(kw)arg', 'value'])

                                        print(f'\033[91m')
                                        [print(f'{k} = {v}') for k, v in INFO_DICT.items()]

                                        RAW_X = pd.DataFrame(data=X, columns=COLUMNS)
                                        RAW_Y = pd.DataFrame(data=y, columns=['y1', 'y2'])

                                        print(f'\nRAW_X = ')
                                        print(RAW_X)

                                        print(f'\nTRFM_X = ')
                                        try:
                                            TRFM_X_DF = pd.DataFrame(
                                                data=TRFM_X,
                                                columns=TestCls.get_feature_names_out(
                                                input_features=COLUMNS
                                                )
                                            )
                                        except:
                                            print(f'\n\033[91m* * * * * TRFM_X_DF EXCEPTED * * * * * *\n')
                                            TRFM_X_DF = pd.DataFrame(data=TRFM_X)

                                        print(TRFM_X_DF.head(10))
                                        print(f'\nMOCK_X = ')
                                        MOCK_X_DF = pd.DataFrame(data=MOCK_X, columns=None)
                                        print(MOCK_X_DF.head(10))
                                        print(f'\nTRFM_Y = ')
                                        TRFM_Y_DF = pd.DataFrame(data=TRFM_Y, columns=['y1', 'y2'])
                                        print(TRFM_Y_DF.head(10))
                                        print(f'\nMOCK_Y = ')
                                        MOCK_Y_DF = pd.DataFrame(data=MOCK_Y, columns=['y1', 'y2'])
                                        print(MOCK_Y_DF.head(10))

                                        print(f'\nLAST INSTRUCTIONS:')
                                        print(TestCls.test_threshold(threshold=count_threshold, clean_printout=True))
                                        print()

                                        # _bp = r'/home/bear/Desktop/'
                                        # INFO_DF.to_csv(_bp + r'INFO_DF.ods')
                                        # RAW_X.to_csv(_bp + r'RAW_X.ods')
                                        # MOCK_X_DF.to_csv(_bp + r'MOCK_X.ods')
                                        # TRFM_X_DF.to_csv(_bp + r'TRFM_X.ods')

                                        raise AssertionError(f"Accuracy test failed for "
                                                             f"{'X' if X_error else ''}{' and ' if X_error and Y_error else ''}{'Y' if Y_error else ''}")

    del CT_THRESH, IGN_FLOAT_COLS, IGN_NON_BIN_INT_COLS, IGN_COLS, IGN_NAN, DEL_AX_0, RECURSIONS, total_tests, trial_set
    del TEST_X, TEST_Y, ctr, count_threshold, args, ignore_float_columns, ignore_non_binary_integer_columns, ignore_columns,
    del ignore_nan, delete_axis_0, max_recursions, TestCls, TRFM_X, TRFM_Y, MOCK_X, MOCK_Y, assertion_error, X_error, Y_error

    print(f'\033[92mAll accuracy tests passed.\033[0m')

# END TEST ACCURACY **************************************************************************************************



args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)


# TEST FIT->SET_PARAMS->TRFM == SET_PARAMS->FIT_TRFM *****************************************************************
if not bypass:
    print(f'\033[92mTesting fit->set_params->trfm == set_params->fit->trfm\033[0m')


    # DEFAULT ARGS/KWARGS
    # orig_args = [_rows // 20]
    # orig_kwargs = {
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

    alt_args = [_rows // 25]
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
    FSPTCls = MinCountTransformer(*args, **kwargs)
    FSPTCls.fit(TEST_X, TEST_Y)
    FSPTCls.set_params(count_threshold=alt_args[0], **alt_kwargs)
    FSPT_TRFM_X, FSPT_TRFM_Y = FSPTCls.transform(TEST_X, TEST_Y)

    if not np.array_equiv(SPFT_TRFM_X.astype(str), FSPT_TRFM_X.astype(str)):
        print(f'\033[91m')
        print(f"SPFT_TRFM_X:")
        print(pd.DataFrame(SPFT_TRFM_X).head(20))
        print()
        print(f"FSPT_TRFM_X:")
        print(pd.DataFrame(FSPT_TRFM_X).head(20))
        raise AssertionError(f"SPFT_TRFM_X != FSPT_TRFM_X")

    if not np.array_equiv(SPFT_TRFM_Y, FSPT_TRFM_Y):
        print(f'\033[91m')
        print(f"SPFT_TRFM_Y:")
        print(pd.DataFrame(SPFT_TRFM_Y).head(20))
        print()
        print(f"FSPT_TRFM_Y:")
        print(pd.DataFrame(FSPT_TRFM_Y).head(20))
        raise AssertionError(f"SPFT_TRFM_Y != FSPT_TRFM_Y")

    MOCK_X = mmct().trfm(TEST_X, None, alt_kwargs['ignore_columns'],
            alt_kwargs['ignore_nan'], alt_kwargs['ignore_non_binary_integer_columns'],
            alt_kwargs['ignore_float_columns'], alt_kwargs['handle_as_bool'],
            alt_kwargs['delete_axis_0'], alt_args[0]
    )

    if not np.array_equiv(FSPT_TRFM_X.astype(str), MOCK_X.astype(str)):
        print(f'\033[91m')
        print(f"FSPT_TRFM_X, shape = {FSPT_TRFM_X.shape}:")
        print(pd.DataFrame(FSPT_TRFM_X).head(20))
        print()
        print(f"MOCK_X, shape = {MOCK_X.shape}:")
        print(pd.DataFrame(MOCK_X).head(20))
        print()
        print(f'Different values:')
        MASK = (FSPT_TRFM_X==MOCK_X).astype(bool)
        AXIS_0_MASK = (MASK.sum(axis=1) < FSPT_TRFM_X.shape[1])
        AXIS_1_MASK = (MASK.sum(axis=0) < FSPT_TRFM_X.shape[0])
        CHOPPED_FSPT = FSPT_TRFM_X[AXIS_0_MASK, :]
        CHOPPED_FSPT = CHOPPED_FSPT[:, AXIS_1_MASK]
        CHOPPED_MOCK = MOCK_X[AXIS_0_MASK, :]
        CHOPPED_MOCK = CHOPPED_MOCK[:, AXIS_1_MASK]
        del MASK, AXIS_0_MASK, AXIS_1_MASK
        print(f"FSPT_TRFM_X:")
        print(pd.DataFrame(CHOPPED_FSPT).head(20))
        print()
        print(f"MOCK_X:")
        print(pd.DataFrame(CHOPPED_MOCK).head(20))

        raise AssertionError(f"FSPT_TRFM_X != MOCK_X")

    del TEST_X, TEST_Y, MOCK_X

    print(f'\033[92mfit->set_params->trfm == set_params->fit->trfm test passed.\033[0m')
# END TEST FIT->SET_PARAMS->TRFM == SET_PARAMS->FIT_TRFM *********************************************************




args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)


# TEST MANY PARTIAL FITS == ONE BIG FIT ******************************************************************************
if True:
    print(f'\n\033[92mTesting many partial fits == one big fit\033[0m')

    # TEST THAT ONE-SHOT partial_fit() / transform() == ONE-SHOT fit() / transform() ** ** ** ** ** ** ** **
    OneShotPartialFitTestCls = MinCountTransformer(*args, **kwargs)
    OneShotPartialFitTestCls.partial_fit(X, y)
    ONE_SHOT_PARTIAL_FIT_TRFM_X, ONE_SHOT_PARTIAL_FIT_TRFM_Y = OneShotPartialFitTestCls.transform(X, y)

    OneShotFullFitTestCls = MinCountTransformer(*args, **kwargs)
    OneShotFullFitTestCls.partial_fit(X, y)
    ONE_SHOT_FULL_FIT_TRFM_X, ONE_SHOT_FULL_FIT_TRFM_Y = OneShotFullFitTestCls.transform(X, y)

    if not np.array_equiv(ONE_SHOT_PARTIAL_FIT_TRFM_X.astype(str), ONE_SHOT_FULL_FIT_TRFM_X.astype(str)):
        print(f'\033[91m')
        print(f'ONE_SHOT_PARTIAL_FIT_TRFM_X.shape = ', ONE_SHOT_PARTIAL_FIT_TRFM_X.shape)
        print(f'ONE_SHOT_FULL_FIT_TRFM_X.shape = ', ONE_SHOT_FULL_FIT_TRFM_X.shape)
        raise AssertionError(f"one shot partial fit trfm X != one shot full fit trfm X")

    if not np.array_equiv(ONE_SHOT_PARTIAL_FIT_TRFM_Y, ONE_SHOT_FULL_FIT_TRFM_Y):
        print(f'\033[91m')
        print(f'ONE_SHOT_PARTIAL_FIT_TRFM_Y.shape = ', ONE_SHOT_PARTIAL_FIT_TRFM_Y.shape)
        print(f'ONE_SHOT_FULL_FIT_TRFM_Y.shape = ', ONE_SHOT_FULL_FIT_TRFM_Y.shape)
        raise AssertionError(f"one shot partial fit trfm Y != one shot full fit trfm Y")
    # END TEST THAT ONE-SHOT partial_fit() / transform() == ONE-SHOT fit() / transform() ** ** ** ** ** ** **

    # TEST PARTIAL FIT COUNTS ARE DOUBLED WHEN FULL DATA IS partial_fit() 2X ** ** ** ** ** ** ** ** **
    SingleFitTestClass = MinCountTransformer(*args, **kwargs)
    DoublePartialFitTestClass = MinCountTransformer(*args, **kwargs)

    SingleFitTestClass.fit(X, y)
    DoublePartialFitTestClass.partial_fit(X, y)

    # def print_unqs_cts_from_fits(instance_name, total_cts_by_column_dict):
    #     print_w_pad = lambda words, mult: print(5*mult*' ' + words)
    #
    #     print_w_pad(f'\n{instance_name}:', 0)
    #     for column, unq_ct_dict in total_cts_by_column_dict.items():
    #         print_w_pad(f'Column {column}:', 1)
    #         for unq, ct in unq_ct_dict.items():
    #             print_w_pad(f'{unq}'.ljust(20) + f'{ct}', 2)

    # print_unqs_cts_from_fits(f'\nSingleFitTestClass', SingleFitTestClass._total_counts_by_column)
    #
    # print_unqs_cts_from_fits(f'\nDoublePartialFitTestClass first pass', DoublePartialFitTestClass._total_counts_by_column)

    DoublePartialFitTestClass.partial_fit(X, y)

    # print_unqs_cts_from_fits(f'\nDoublePartialFitTestClass second pass', DoublePartialFitTestClass._total_counts_by_column)
    #
    # del print_unqs_cts_from_fits

    # END TEST PARTIAL FIT COUNTS ARE DOUBLED WHEN FULL DATA IS partial_fit() 2X ** ** ** ** ** ** **

    # STORE CHUNKS TO ENSURE THEY STACK BACK TO THE ORIGINAL X/y
    _chunks = 5
    X_CHUNK_HOLDER = []
    Y_CHUNK_HOLDER = []
    for row_chunk in range(_chunks):
        X_CHUNK_HOLDER.append(X[row_chunk * x_rows // _chunks:(row_chunk + 1) * x_rows // _chunks, :])
        Y_CHUNK_HOLDER.append(y[row_chunk * x_rows // _chunks:(row_chunk + 1) * x_rows // _chunks, :])

    assert np.array_equiv(np.vstack(X_CHUNK_HOLDER).astype(str), X.astype(str)), \
        f"agglomerated X chunks != original X"
    assert np.array_equiv(np.vstack(Y_CHUNK_HOLDER), y), \
        f"agglomerated Y chunks != original Y"

    PartialFitPartialTrfmTestCls = MinCountTransformer(*args, **kwargs)
    PartialFitOneShotTrfmTestCls = MinCountTransformer(*args, **kwargs)
    OneShotFitTransformTestCls = MinCountTransformer(*args, **kwargs)

    # PIECEMEAL PARTIAL FIT
    for X_CHUNK, Y_CHUNK in zip(X_CHUNK_HOLDER, Y_CHUNK_HOLDER):
        PartialFitPartialTrfmTestCls.partial_fit(X_CHUNK, Y_CHUNK)
        PartialFitOneShotTrfmTestCls.partial_fit(X_CHUNK, Y_CHUNK)

    # PIECEMEAL TRANSFORM ***********************************************************
    # THIS MUST BE IN ITS OWN LOOP, ALL FITS MUST BE DONE BEFORE DOING ANY TRFMS
    PARTIAL_TRFM_X_HOLDER = []
    PARTIAL_TRFM_Y_HOLDER = []
    for X_CHUNK, Y_CHUNK in zip(X_CHUNK_HOLDER, Y_CHUNK_HOLDER):
        PARTIAL_TRFM_X, PARTIAL_TRFM_Y = PartialFitPartialTrfmTestCls.transform(X_CHUNK, Y_CHUNK)
        PARTIAL_TRFM_X_HOLDER.append(PARTIAL_TRFM_X)
        PARTIAL_TRFM_Y_HOLDER.append(PARTIAL_TRFM_Y)

    del PartialFitPartialTrfmTestCls, PARTIAL_TRFM_X, PARTIAL_TRFM_Y

    # AGGLOMERATE PARTIAL TRFMS FROM PARTIAL FIT
    FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM = np.vstack(PARTIAL_TRFM_X_HOLDER)
    FULL_TRFM_Y_FROM_PARTIAL_FIT_PARTIAL_TRFM = np.vstack(PARTIAL_TRFM_Y_HOLDER)
    # END PIECEMEAL TRANSFORM ***********************************************************

    # DO ONE-SHOT TRANSFORM OF X,y ON THE PARTIALLY FIT INSTANCE
    FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM, FULL_TRFM_Y_FROM_PARTIAL_FIT_ONESHOT_TRFM = PartialFitOneShotTrfmTestCls.transform(
        X, y)

    # ONE-SHOT FIT TRANSFORM
    FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM, FULL_TRFM_Y_ONE_SHOT_FIT_TRANSFORM = OneShotFitTransformTestCls.fit_transform(X,
                                                                                                                      y)

    # ASSERT ALL AGGLOMERATED X AND Y TRFMS ARE EQUAL
    if not np.array_equiv(FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM.astype(str), FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM.astype(str)):
        print(f'\033[91m')
        print(f'FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM.shape = ', FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM.shape)
        print(f'FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM.shape = ', FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM.shape)
        raise AssertionError(f"compiled trfm X from partial fit / partial trfm != one-shot fit/trfm X")

    if not np.array_equiv(FULL_TRFM_Y_ONE_SHOT_FIT_TRANSFORM, FULL_TRFM_Y_FROM_PARTIAL_FIT_PARTIAL_TRFM):
        print(f'\033[91m')
        print(f'FULL_TRFM_Y_ONE_SHOT_FIT_TRANSFORM.shape = ', FULL_TRFM_Y_ONE_SHOT_FIT_TRANSFORM.shape)
        print(f'FULL_TRFM_Y_FROM_PARTIAL_FIT_PARTIAL_TRFM.shape = ', FULL_TRFM_Y_FROM_PARTIAL_FIT_PARTIAL_TRFM.shape)
        raise AssertionError(f"compiled trfm y from partial fit / partial trfm != one-shot fit/trfm y")

    if not np.array_equiv(FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM.astype(str), FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM.astype(str)):
        print(f'\033[91m')
        print(f'FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM.shape = ', FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM.shape)
        print(f'FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM.shape = ', FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM.shape)
        raise AssertionError(f"trfm X from partial fits / one-shot trfm != one-shot fit/trfm X")

    if not np.array_equiv(FULL_TRFM_Y_ONE_SHOT_FIT_TRANSFORM, FULL_TRFM_Y_FROM_PARTIAL_FIT_ONESHOT_TRFM):
        print(f'\033[91m')
        print(f'FULL_TRFM_Y_ONE_SHOT_FIT_TRANSFORM.shape = ', FULL_TRFM_Y_ONE_SHOT_FIT_TRANSFORM.shape)
        print(f'FULL_TRFM_Y_FROM_PARTIAL_FIT_ONESHOT_TRFM.shape = ', FULL_TRFM_Y_FROM_PARTIAL_FIT_ONESHOT_TRFM.shape)
        raise AssertionError(f"trfm y from partial fits / one-shot trfm != one-shot fit/trfm y")

    del _chunks, X_CHUNK_HOLDER, Y_CHUNK_HOLDER, row_chunk, PartialFitOneShotTrfmTestCls, OneShotFitTransformTestCls
    del PARTIAL_TRFM_X_HOLDER, PARTIAL_TRFM_Y_HOLDER, X_CHUNK, Y_CHUNK, FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM
    del FULL_TRFM_Y_FROM_PARTIAL_FIT_PARTIAL_TRFM, FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM
    del FULL_TRFM_Y_FROM_PARTIAL_FIT_ONESHOT_TRFM, FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM, FULL_TRFM_Y_ONE_SHOT_FIT_TRANSFORM

    print(f'\033[92mpassed many partial fits == one big fit tests\033[0m')

# END TEST MANY PARTIAL FITS == ONE BIG FIT **************************************************************************


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST LATER PARTIAL FITS ACCEPT NEW UNIQUES **************************************************************************
if True:
    print(f'\n\033[92mTesting later partial fits accept new uniques')
    X1 = NO_NAN_X[:, _cols:(2 * _cols)].copy().astype(np.float64).astype(np.int32)
    y1 = y.copy()
    # 10X THE VALUES IN THE COPY OF DATA TO INTRODUCE NEW UNIQUE VALUES
    X2 = (10 * X1.astype(np.float64)).astype(np.int32)
    y2 = y.copy()

    STACKED_X = np.vstack((X1, X2)).astype(np.float64).astype(np.int32)
    STACKED_Y = np.vstack((y1, y2)).astype(np.uint8)

    args = [2 * args[0]]
    kwargs['ignore_non_binary_integer_columns'] = False

    PartialFitTestCls = MinCountTransformer(*args, **kwargs)

    PartialFitTestCls.partial_fit(X1, y1)
    PartialFitTestCls.partial_fit(X2, y2)
    PARTIAL_FIT_X, PARTIAL_FIT_Y = PartialFitTestCls.transform(STACKED_X, STACKED_Y)

    # VERIFY SOME ROWS WERE ACTUALLY DELETED
    assert not np.array_equiv(PARTIAL_FIT_X, np.vstack((X1, X2))), \
        f'later partial fits accept new uniques --- transform did not delete any rows'

    SingleFitTestCls = MinCountTransformer(*args, **kwargs)
    SingleFitTestCls.fit(STACKED_X, STACKED_Y)
    SINGLE_FIT_X, SINGLE_FIT_Y = SingleFitTestCls.transform(STACKED_X, STACKED_Y)

    assert np.array_equiv(PARTIAL_FIT_X,
                          SINGLE_FIT_X), f"new uniques in partial fits -- partial fitted X does not equal single fitted X"
    assert np.array_equiv(PARTIAL_FIT_Y,
                          SINGLE_FIT_Y), f"new uniques in partial fits -- partial fitted y does not equal single fitted y"

    del X1, y1, X2, y2, STACKED_X, STACKED_Y, PartialFitTestCls, PARTIAL_FIT_X, PARTIAL_FIT_Y, SingleFitTestCls
    del SINGLE_FIT_X, SINGLE_FIT_Y

    print(f'\033[92mLater partial fits accept new uniques tests passed.')

# END TEST LATER PARTIAL FITS ACCEPT NEW UNIQUES **********************************************************************


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST TRANSFORM CONDITIONALLY ACCEPTS NEW UNIQUES ***********************************************************************
if True:
    print(f'\n\033[92mTesting transform() conditionally accepts new uniques')
    # USE STR COLUMNS
    X1 = NO_NAN_X[:, (3 * _cols):(4 * _cols)].copy()
    y1 = y.copy()

    # fit() & transform() ON X1 TO PROVE X1 PASSES transform()
    TestCls = MinCountTransformer(*args, **kwargs)
    TestCls.fit(X1, y1)
    TestCls.transform(X1, y1)
    del TestCls

    # PEPPER ONE OF THE STR COLUMNS WITH A UNIQUE THAT WAS NOT SEEN DURING fit()
    X2 = X1.copy()
    X2[np.random.choice(range(x_rows), 10, replace=False), 0] = list('1234567890')
    TestCls = MinCountTransformer(*args, **kwargs)
    TestCls.fit(X1, y1)

    # DEMONSTRATE NEW VALUES ARE ACCEPTED WHEN reject_unseen_values = False
    try:
        TestCls.set_params(reject_unseen_values=False)
        TestCls.transform(X2, y1)
    except Exception as e1:
        raise AssertionError(f"transform() should have have accepted unseen data instead raised --- {e1}")

    # DEMONSTRATE NEW VALUES ARE REJECTED WHEN reject_unseen_values = True
    try:
        TestCls.set_params(reject_unseen_values=True)
        TestCls.transform(X2, y1)
        raise AssertionError
    except ValueError:
        pass
    except AssertionError:
        raise AssertionError(f"transform() accepted data with unseen uniques but should have raised ValueError")
    except Exception as e1:
        raise AssertionError(f"tranform() should have raised ValueError but instead raised --- {e1}")

    del X1, y1, X2, TestCls

    print(f'\033[92mtransform() conditionally accepts new uniques tests passed.')

# END TEST TRANSFORM CONDITIONALLY ACCEPT NEW UNIQUES *******************************************************************


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# TEST DASK Incremental + ParallelPostFit == ONE BIG sklearn fit_transform() ********************************************
if True:
    print(f'\n\033[92mTesting dask Incremental + ParallelPostFit == one big sklearn fit_transform()\033[0m')

    # USE NUMERICAL COLUMNS ONLY 24_03_27_11_45_00
    # NotImplementedError: Can not use auto rechunking with object dtype.
    # We are unable to estimate the size in bytes of object data

    NEW_X = X.copy()[:, :3*_cols].astype(np.float64)

    DaskTestCls = ParallelPostFit(estimator=Incremental(MinCountTransformer(*args, **kwargs)))

    da_X = da.array(NEW_X).rechunk((x_rows // 5, x_cols))

    da_y1 = da.array(y).rechunk((y_rows // 5, 2))
    da_y2 = da.array(y).transpose().rechunk((2, y_rows // 5))
    da_y3 = da.array(y[:, 0].ravel()).rechunk((y_rows // 5,))

    for y_version, da_y in zip(['2D_as_row', '2D_as_col', '1D'], [da_y1, da_y2, da_y3]):
        try:
            DaskTestCls.fit(da_X, da_y)
            try:
                DA_TRFM_X, DA_TRFM_Y = DaskTestCls.transform(da_X, da_y)
            except:
                DA_TRFM_X = DaskTestCls.transform(da_X)
            assert isinstance(DA_TRFM_X, da.core.Array), f"transform()ed dask input was not returned as dask object"
            DA_TRFM_X = DA_TRFM_X.compute()
            try:
                DA_TRFM_Y = DA_TRFM_Y.compute()
            except:
                pass
        except AssertionError:
            if y_version == '2D_as_col':
                continue
            else:
                raise AssertionError(f"with given y, transform() should not have raised dask AssertionError")
        except ValueError:
            if y_version == '2D_as_row':
                continue
            else:
                raise AssertionError(f"with given y, transform() should not have raised MinCountTransformer ValueError")
        except Exception as e1:
            raise AssertionError(
                f"dask wrapper fit/transform excepted with something other than Assertion or Value Error")

        FitTransformTestCls = MinCountTransformer(*args, **kwargs)
        FT_TRFM_X, FT_TRFM_Y = FitTransformTestCls.fit_transform(NEW_X, y)

        assert np.array_equiv(DA_TRFM_X.astype(str), FT_TRFM_X.astype(str)), \
            f"transformed X from dask != transformed X from single sklearn fit/transform"

        try:
            assert np.array_equiv(DA_TRFM_Y, FT_TRFM_Y), \
                f"transformed y from dask != transformed y from single sklearn fit/transform"
        except:
            pass

    del DaskTestCls, NEW_X, da_X, da_y, DA_TRFM_X, FitTransformTestCls, FT_TRFM_X, FT_TRFM_Y
    try:
        del DA_TRFM_Y
    except:
        pass

    print(f'\n\033[92mpassed dask Incremental + ParallelPostFit == one big sklearn fit_transform() test\033[0m')

# END TEST DASK Incremental + ParallelPostFit == ONE BIG sklearn fit_transform() ****************************************


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM, ATTR ACCURACY; FOR 1 RECURSION ******************************************
if True:
    print(f'\n\033[92mTesting ATTR access and accuracy before and after fit and transform for 1 recursion\033[0m')

    NEW_X = X.copy()
    NEW_Y = y.copy()
    NEW_X_DF = pd.DataFrame(data=X, columns=COLUMNS)
    NEW_Y_DF = pd.DataFrame(data=y, columns=['y1', 'y2'])

    # BEFORE FIT **********************************************************

    TestCls = MinCountTransformer(*args, **kwargs)

    # ALL OF THESE SHOULD GIVE AttributeError
    err_msg = lambda _name: f"accessing {_name} before fit() excepted for reason other than AttributeError"
    try:
        TestCls.feature_names_in_
    except AttributeError:
        pass
    except:
        raise Exception(err_msg('feature_names_in_'))

    try:
        TestCls.n_features_in_
    except AttributeError:
        pass
    except:
        raise Exception(err_msg('n_features_in_'))

    try:
        TestCls.original_dtypes_
    except AttributeError:
        pass
    except:
        raise Exception(err_msg('original_dtypes_'))

    try:
        TestCls.original_dtypes_ = list('abcde')
        raise AssertionError(
            f"original_dtypes_ was allowed to be set by assignment when is read-only and should have raised AttributeError")
    except AttributeError:
        pass
    except Exception as e1:
        raise AssertionError(
            f"original_dtypes_ disallowed set by assignment but raised an error other than AttributeError --- {e1}")

    del err_msg, TestCls
    # END BEFORE FIT **********************************************************

    # AFTER FIT **********************************************************
    for data_dtype in ['np', 'pd']:
        if data_dtype == 'np':
            TEST_X, TEST_Y = NEW_X.copy(), NEW_Y.copy()
        elif data_dtype == 'pd':
            TEST_X, TEST_Y = NEW_X_DF.copy(), NEW_Y_DF.copy()

        TestCls = MinCountTransformer(*args, **kwargs)
        TestCls.fit(TEST_X, TEST_Y)

        # ONLY EXCEPTION SHOULD BE feature_names_in_ IF NUMPY
        if data_dtype == 'pd':
            assert np.array_equiv(TestCls.feature_names_in_, COLUMNS), \
                f"feature_names_in_ after fit() != originally passed columns"
        elif data_dtype == 'np':
            try:
                TestCls.feature_names_in_
                raise AssertionError
            except AttributeError:
                pass
            except AssertionError:
                print(f'\033[91mfeature_names_in_ = ')
                print(TestCls.feature_names_in_)
                raise Exception(f"feature_names_in_ was accessed even though fit was done with a numpy array")
            except:
                raise Exception(
                    f"accessing feature_names_in_ with numpy fit() excepted for reason other than AttributeError")

        assert TestCls.n_features_in_ == x_cols, f"n_features_in_ after fit() != number of originally passed columns"

        assert np.array_equiv(TestCls._original_dtypes,
                              DTYPE_KEY), f"_original_dtypes after fit() != originally passed dtypes"

    del data_dtype, TEST_X, TEST_Y, TestCls

    # END AFTER FIT **********************************************************

    # AFTER TRANSFORM *********************************************************

    for data_dtype in ['np', 'pd']:
        if data_dtype == 'np':
            TEST_X, TEST_Y = NEW_X.copy(), NEW_Y.copy()
        elif data_dtype == 'pd':
            TEST_X, TEST_Y = NEW_X_DF.copy(), NEW_Y_DF.copy()

        TestCls = MinCountTransformer(*args, **kwargs)
        TRFM_X, TRFM_Y = TestCls.fit_transform(TEST_X, TEST_Y)

        # ONLY EXCEPTION SHOULD BE feature_names_in_ WHEN NUMPY
        if data_dtype == 'pd':
            assert np.array_equiv(TestCls.feature_names_in_, COLUMNS), \
                f"feature_names_in_ after fit() != originally passed columns"
        elif data_dtype == 'np':
            try:
                TestCls.feature_names_in_
                raise AssertionError
            except AttributeError:
                pass
            except AssertionError:
                print(f'\033[91mfeature_names_in_ = ')
                print(TestCls.feature_names_in_)
                raise Exception(f"feature_names_in_ was accessed even though fit was done with a numpy array")
            except:
                raise Exception(
                    f"accessing feature_names_in_ with numpy fit() excepted for reason other than AttributeError")

        assert TestCls.n_features_in_ == x_cols, f"n_features_in_ after fit() != number of originally passed columns"

        assert np.array_equiv(TestCls._original_dtypes,
                              DTYPE_KEY), f"_original_dtypes after fit() != originally passed dtypes"

    del data_dtype, TEST_X, TEST_Y, TestCls, TRFM_X, TRFM_Y
    # END AFTER TRANSFORM *********************************************************

    del NEW_X, NEW_Y, NEW_X_DF, NEW_Y_DF

    print(f'\033[92mAll ATTR access and accuracy tests passed for 1 recursion\033[0m')
# END ACCESS ATTR BEFORE AND AFTER FIT AND TRANSFORM, ATTR ACCURACY; FOR 1 RECURSION **************************************


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM; FOR 1 RECURSION *******************************************************
if True:
    print(f'\n\033[92mTest METHOD access and accuracy before and after fit and transform for 1 recursion\033[0m')


    def not_fitted_tester(_name, cls_method, *args, **kwargs):

        try:
            cls_method(*args, **kwargs)
        except NotFittedError:
            pass
        except:
            raise Exception(f"{_name} excepted for reason other than NotFittedError")


    TestCls = MinCountTransformer(*args, **kwargs)

    # ******************************************************************************************************************
    # vvv BEFORE FIT vvv *******************************************************************************************

    # ** _base_fit()
    # ** _check_is_fitted()

    # ** test_threshold()
    not_fitted_tester('test_threshold', TestCls.test_threshold)

    # fit()
    # fit_transform()

    # get_feature_names_out()
    not_fitted_tester('get_feature_names_out', TestCls.get_feature_names_out, None)

    # get_metadata_routing()
    try:
        TestCls.get_metadata_routing()
    except NotImplementedError:
        pass
    except Exception as e1:
        raise AssertionError(
            f"get_metadata_routing() should have excepted with NotImplementedError but excepted with --- {e1} ")

    # get_params()
    TestCls.get_params(True)

    # get_row_support()
    not_fitted_tester('get_row_support', TestCls.get_row_support, True)

    # get_support()
    not_fitted_tester('get_support', TestCls.get_support, True)

    # ** _handle_X_y()

    # inverse_transform()
    not_fitted_tester('inverse_transform', TestCls.inverse_transform, X)

    # ** _make_instructions()
    # ** _must_be_fitted()
    # partial_fit()
    # ** _reset()

    # set_output()
    TestCls.set_output(transform='pandas_dataframe')

    # set_params()
    KEYS = ['count_threshold', 'ignore_float_columns', 'ignore_non_binary_integer_columns', 'ignore_columns',
            'ignore_nan', 'handle_as_bool', 'delete_axis_0', 'reject_unseen_values', 'max_recursions', 'n_jobs']
    VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
    test_kwargs = dict((zip(KEYS, VALUES)))
    TestCls.set_params(**test_kwargs)
    ATTRS = [TestCls._count_threshold, TestCls._ignore_float_columns, TestCls._ignore_non_binary_integer_columns,
             TestCls._ignore_columns, TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
             TestCls._reject_unseen_values, TestCls._max_recursions, TestCls._n_jobs]
    for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
        assert _attr == _value, f'set_params() did not set {_key}'

    # DEMONSTRATE EXCEPTS FOR UNKNOWN PARAM
    try:
        TestCls.set_params(pizza=1)
        raise AssertionError
    except ValueError:
        pass
    except AssertionError:
        raise AssertionError(f"set_params() should have raised ValueError for unknown param but did not")
    except Exception as e1:
        raise AssertionError(
            f"set_params() should have raised ValueError for unknown param but instead raised --- {e1}")

    del TestCls, KEYS, VALUES, ATTRS

    TestCls = MinCountTransformer(*args, **kwargs)
    # transform()
    not_fitted_tester('transform', TestCls.transform, X, y)

    # ** _validate_delete_instr()
    # ** _validate_feature_names()
    # ** _validate()

    del not_fitted_tester

    # END ^^^ BEFORE FIT ^^^ *******************************************************************************************
    # ******************************************************************************************************************

    TestCls.fit(X, y)

    # ******************************************************************************************************************
    # vvv AFTER FIT vvv *******************************************************************************************

    # ** _base_fit()
    # ** _check_is_fitted()

    # ** test_threshold()
    TestCls.test_threshold()
    print(f'^^^ mask building instructions should be displayed above ^^^')

    # fit()
    # fit_transform()

    # get_feature_names_out() ******************************************************
    # vvv NO COLUMN NAMES PASSED (NP) vvv
    # **** CAN ONLY TAKE LIST-TYPE OF STRS OR None
    for junk_arg in [float('inf'), np.pi, 'garbage', {'junk': 3}, [*range(len(COLUMNS))]]:
        try:
            TestCls.get_feature_names_out(np.pi)
        except TypeError:
            pass
        except Exception as e1:
            raise AssertionError(
                f"get_feature_names_out() should have raised TypeError for '{junk_arg}' arg but raised --- {e1}")

    # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN ['x0', ..., 'x(n-1)][COLUMN MASK]
    if not np.array_equiv(TestCls.get_feature_names_out(None),
                          np.array([f"x{i}" for i in range(len(COLUMNS))])[TestCls.get_support(False)]):
        raise AssertionError(f"get_feature_names_out(None) after fit() != sliced array of generic headers")

    # WITH NO HEADER PASSED, SHOULD RAISE ValueError IF len(input_features) != n_features_in_
    try:
        TestCls.get_feature_names_out([f"x{i}" for i in range(2 * len(COLUMNS))])
    except ValueError:
        pass
    except Exception as e1:
        raise AssertionError(
            f"get_feature_names_out(TOO_MANY_COLUMNS) after fit() should have raised ValueError but raised --- {e1}")

    # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features, SHOULD RETURN SLICED PASSED COLUMNS
    RETURNED_FROM_GFNO = TestCls.get_feature_names_out(COLUMNS)
    if not isinstance(RETURNED_FROM_GFNO, np.ndarray):
        raise AssertionError(
            f"get_feature_names_out should return numpy.ndarray, but returned {type(RETURNED_FROM_GFNO)}")
    if not np.array_equiv(RETURNED_FROM_GFNO, np.array(COLUMNS)[TestCls.get_support(False)]):
        raise AssertionError(f"get_feature_names_out() did not return original columns")

    del junk_arg, RETURNED_FROM_GFNO, TestCls

    # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

    # vvv COLUMN NAMES PASSED (PD) vvv

    TestCls = MinCountTransformer(*args, **kwargs)
    TestCls.fit(pd.DataFrame(data=X, columns=COLUMNS), y)

    # WITH HEADER PASSED AND input_features=None, SHOULD RETURN SLICEDORIGINAL COLUMNS
    if not np.array_equiv(TestCls.get_feature_names_out(None), np.array(COLUMNS)[TestCls.get_support(False)]):
        raise AssertionError(f"get_feature_names_out(None) after fit() != originally passed columns")

    # WITH HEADER PASSED, SHOULD RAISE TypeError IF input_features FOR DISALLOWED TYPES

    for junk_col_names in [[*range(len(COLUMNS))], [*range(2 * len(COLUMNS))], {'a': 1, 'b': 2}]:
        try:
            TestCls.get_feature_names_out(junk_col_names)
        except TypeError:
            pass
        except Exception as e1:
            raise AssertionError(
                f"get_feature_names_out(INVALID_TYPES) after fit() should have raised TypeError but raised --- {e1}")

    # WITH HEADER PASSED, SHOULD RAISE ValueError IF input_features DOES NOT EXACTLY MATCH ORIGINALLY FIT COLUMNS
    for junk_col_names in [np.char.upper(COLUMNS), np.hstack((COLUMNS, COLUMNS)), []]:
        try:
            TestCls.get_feature_names_out(junk_col_names)
        except ValueError:
            pass
        except Exception as e1:
            raise AssertionError(
                f"get_feature_names_out(INVALID_NAMES) after fit() should have raised ValueError but raised --- {e1}")

    # WHEN HEADER PASSED TO (partial_)fit() AND input_features IS THAT HEADER, SHOULD RETURN SLICED VERSION OF THAT HEADER

    RETURNED_FROM_GFNO = TestCls.get_feature_names_out(COLUMNS)
    if not isinstance(RETURNED_FROM_GFNO, np.ndarray):
        raise AssertionError(
            f"get_feature_names_out should return numpy.ndarray, but returned {type(RETURNED_FROM_GFNO)}")
    if not np.array_equiv(RETURNED_FROM_GFNO, np.array(COLUMNS)[TestCls.get_support(False)]):
        raise AssertionError(f"get_feature_names_out() did not return original columns")

    del junk_col_names, RETURNED_FROM_GFNO
    # END ^^^ COLUMN NAMES PASSED (PD) ^^^

    # END get_feature_names_out() *************************************************

    # get_metadata_routing()
    try:
        TestCls.get_metadata_routing()
    except NotImplementedError:
        pass
    except Exception as e1:
        raise AssertionError(
            f"get_metadata_routing() should have excepted with NotImplementedError but excepted with --- {e1} ")

    # get_params()
    TestCls.get_params(True)

    # get_row_support()
    try:
        TestCls.get_row_support(False)
    except AttributeError:
        pass
    except Exception as e1:
        raise AssertionError(f"get_row_support() should have excepted with AttributeError but excepted with --- {e1} ")

    # get_support()
    for _indices in [True, False]:
        __ = TestCls.get_support(_indices)
        assert isinstance(__, np.ndarray), f"get_support() did not return numpy.ndarray"

        if not _indices:
            assert __.dtype == 'bool', f"get_support with indices=False did not return a boolean array"
            assert len(__) == TestCls.n_features_in_, f"len(get_support(False)) != n_features_in_"
            assert sum(__) == len(TestCls.get_feature_names_out(None))
        elif _indices:
            assert 'int' in str(__.dtype).lower(), f"get_support with indices=True did not return an array of integers"
            assert len(__) == len(TestCls.get_feature_names_out(None))

    del TestCls, _indices, __,

    # ** _handle_X_y()

    # inverse_transform() ********************
    TestCls = MinCountTransformer(*args, **kwargs)
    TestCls.fit(X, y)  # X IS NP ARRAY

    # SHOULD RAISE ValueError IF X IS NOT A 2D ARRAY
    for junk_x in [[], [[]]]:
        try:
            TestCls.inverse_transform(junk_x)
        except ValueError:
            pass
        except Exception as e1:
            raise AssertionError(
                f"inverse_transform() should have excepted with ValueError but excepted with --- {e1} ")

    # SHOULD RAISE TypeError IF X IS NOT A LIST-TYPE
    for junk_x in [None, 'junk_string', 3, np.pi]:
        try:
            TestCls.inverse_transform(junk_x)
        except TypeError:
            pass
        except Exception as e1:
            raise AssertionError(f"inverse_transform() should have excepted with TypeError but excepted with --- {e1} ")

    # SHOULD RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF RETAINED COLUMNS
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
                    TEST_X = pd.DataFrame(data=TEST_X, columns=np.hstack((__[TRFM_MASK], np.char.upper(__[TRFM_MASK]))))

            try:
                TestCls.inverse_transform(TEST_X)
                assert diff_cols == 'same', \
                    f"inverse_transform() did not raise any exception for X with invalid number of columns"
            except ValueError:
                pass
            except Exception as e1:
                raise AssertionError(
                    f"inverse_transform() invalid number of columns should have raised ValueError but raised --- {e1}")

    INV_TRFM_X = TestCls.inverse_transform(TRFM_X)

    assert isinstance(INV_TRFM_X, np.ndarray), f"output of inverse_transform() is not a numpy array"
    assert INV_TRFM_X.shape[0] == TRFM_X.shape[0], f"rows in output of inverse_transform() do not match input rows"
    assert INV_TRFM_X.shape[
               1] == TestCls.n_features_in_, f"columns in output of inverse_transform() do not match originally fitted columns"

    __ = np.logical_not(TestCls.get_support(False))
    assert np.array_equiv(INV_TRFM_X[:, __], np.zeros((TRFM_X.shape[0], sum(__)))), \
        f"back-filled parts of inverse_transform() output do not slice to a zero array"
    del __

    assert np.array_equiv(TRFM_X.astype(str), INV_TRFM_X[:, TestCls.get_support(False)].astype(str)), \
        f"output of inverse_transform() does not reduce back to the output of transform()"

    del junk_x, TRFM_X, TRFM_MASK, obj_type, diff_cols, TEST_X, INV_TRFM_X, TestCls

    # END inverse_transform() **********

    TestCls = MinCountTransformer(*args, **kwargs)

    # ** _make_instructions()
    # ** _must_be_fitted()
    # partial_fit()
    # ** _reset()

    # set_output()
    TestCls.set_output(transform='pandas_dataframe')

    # set_params()
    KEYS = ['count_threshold', 'ignore_float_columns', 'ignore_non_binary_integer_columns', 'ignore_columns',
            'ignore_nan', 'handle_as_bool', 'delete_axis_0', 'reject_unseen_values', 'max_recursions', 'n_jobs']
    VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
    test_kwargs = dict((zip(KEYS, VALUES)))

    TestCls.set_params(**test_kwargs)
    ATTRS = [TestCls._count_threshold, TestCls._ignore_float_columns, TestCls._ignore_non_binary_integer_columns,
             TestCls._ignore_columns, TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
             TestCls._reject_unseen_values, TestCls._max_recursions, TestCls._n_jobs]
    for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
        assert _attr == _value, f'set_params() did not set {_key}'

    del TestCls, KEYS, VALUES, ATTRS

    # transform()
    # ** _validate_delete_instr()
    # ** _validate_feature_names()
    # ** _validate()

    # END ^^^ AFTER FIT ^^^ *******************************************************************************************
    # ******************************************************************************************************************

    # ******************************************************************************************************************
    # vvv AFTER TRANSFORM vvv **************************************************************************************
    FittedTestCls = MinCountTransformer(*args, **kwargs).fit(X, y)
    TransformedTestCls = MinCountTransformer(*args, **kwargs).fit(X, y)
    TRFM_X, TRFM_Y = TransformedTestCls.transform(X, y)

    # ** _base_fit()
    # ** _check_is_fitted()

    # ** test_threshold()
    # SHOULD BE THE SAME AS AFTER FIT
    TransformedTestCls.test_threshold()
    print(f'^^^ mask building instructions should be displayed above ^^^')

    # fit()
    # fit_transform()

    # get_feature_names_out() ******************************************************
    # vvv NO COLUMN NAMES PASSED (NP) vvv

    # # WHEN NO HEADER PASSED TO (partial_)fit() AND VALID input_features, SHOULD RETURN ORIGINAL (UNSLICED) COLUMNS
    try:
        RETURNED_FROM_GFNO = TransformedTestCls.get_feature_names_out(COLUMNS)
    except:
        raise ValueError(
            f"transformed instance that had not seen header did not accept a header with len==n_features_in_")

    if not np.array_equiv(RETURNED_FROM_GFNO, np.array(COLUMNS)[TransformedTestCls.get_support(False)]):
        raise AssertionError(f"get_feature_names_out() after transform did not return sliced original columns")

    del RETURNED_FROM_GFNO
    # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

    # vvv COLUMN NAMES PASSED (PD) vvv
    PDTransformedTestCls = MinCountTransformer(*args, **kwargs)
    _, _ = PDTransformedTestCls.fit_transform(pd.DataFrame(data=X, columns=COLUMNS), y)
    del _
    # WITH HEADER PASSED AND input_features=None, SHOULD RETURN SLICED ORIGINAL COLUMNS
    if not np.array_equiv(PDTransformedTestCls.get_feature_names_out(None),
                          np.array(COLUMNS)[PDTransformedTestCls.get_support(False)]):
        raise AssertionError(f"get_feature_names_out(None) after transform() != originally passed columns")

    del PDTransformedTestCls
    # END ^^^ COLUMN NAMES PASSED (PD) ^^^

    # END get_feature_names_out() *************************************************

    # get_metadata_routing()
    try:
        TransformedTestCls.get_metadata_routing()
    except NotImplementedError:
        pass
    except Exception as e1:
        raise AssertionError(
            f"get_metadata_routing() should have excepted with NotImplementedError but excepted with --- {e1} ")

    # get_params()
    assert TransformedTestCls.get_params(True) == FittedTestCls.get_params(True), \
        f"get_params() after transform() != before transform()"

    # get_row_support()
    for _indices in [True, False]:
        __ = TransformedTestCls.get_row_support(_indices)
        assert isinstance(__, np.ndarray), f"get_row_support() did not return numpy.ndarray"

        if not _indices:
            assert __.dtype == 'bool', f"get_row_support with indices=False did not return a boolean array"
        elif _indices:
            assert 'int' in str(
                __.dtype).lower(), f"get_row_support with indices=True did not return an array of integers"

    del __

    # get_support()
    assert np.array_equiv(FittedTestCls.get_support(False), TransformedTestCls.get_support(False)), \
        f"get_support(False) after transform() != get_support(False) before"

    # ** _handle_X_y()

    # inverse_transform() ************

    assert np.array_equiv(FittedTestCls.inverse_transform(TRFM_X).astype(str),
                          TransformedTestCls.inverse_transform(TRFM_X).astype(str)), \
        f"inverse_transform(TRFM_X) after transform() != inverse_transform(TRFM_X) before transform()"

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
    TestCls = MinCountTransformer(*args, **kwargs)
    KEYS = ['count_threshold', 'ignore_float_columns', 'ignore_non_binary_integer_columns', 'ignore_columns',
            'ignore_nan', 'handle_as_bool', 'delete_axis_0', 'reject_unseen_values', 'max_recursions', 'n_jobs']
    VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
    test_kwargs = dict((zip(KEYS, VALUES)))

    TestCls.set_params(**test_kwargs)
    TestCls.fit_transform(X, y)
    ATTRS = [TestCls._count_threshold, TestCls._ignore_float_columns, TestCls._ignore_non_binary_integer_columns,
             TestCls._ignore_columns, TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
             TestCls._reject_unseen_values, TestCls._max_recursions, TestCls._n_jobs]
    for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
        assert _attr == _value, f'set_params() did not set {_key}'

    del KEYS, VALUES, ATTRS

    # transform()
    # ** _validate_delete_instr()
    # ** _validate_feature_names()
    # ** _validate()

    del FittedTestCls, TestCls, TRFM_X, TRFM_Y

    # END ^^^ AFTER TRANSFORM ^^^ **************************************************************************************
    # ******************************************************************************************************************

    print(f'\033[92mAll METHOD access and accuracy tests passed for 1 recursion\033[0m')

# END ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM; FOR 1 RECURSION ************************************************


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# ACCESS ATTR BEFORE fit() AND AFTER fit_transform(), ATTR ACCURACY; FOR 2 RECURSIONS ******************************************
if True:
    print(f'\n\033[92mTesting ATTR access and accuracy before fit() and after fit_transform() for 2 recursion\033[0m')

    NEW_X = X.copy()
    NEW_Y = y.copy()
    NEW_X_DF = pd.DataFrame(data=X, columns=COLUMNS)
    NEW_Y_DF = pd.DataFrame(data=y, columns=['y1', 'y2'])

    args = [3]
    OneRecurTestCls = MinCountTransformer(*args, **kwargs)

    kwargs['max_recursions'] = 2

    # BEFORE FIT **********************************************************

    TwoRecurTestCls = MinCountTransformer(*args, **kwargs)

    # ALL OF THESE SHOULD GIVE AttributeError
    err_msg = lambda _name: f"accessing {_name} before fit() excepted for reason other than AttributeError"
    try:
        TwoRecurTestCls.feature_names_in_
    except AttributeError:
        pass
    except:
        raise Exception(err_msg('feature_names_in_'))

    try:
        TwoRecurTestCls.n_features_in_
    except AttributeError:
        pass
    except:
        raise Exception(err_msg('n_features_in_'))

    try:
        TwoRecurTestCls.original_dtypes_
    except AttributeError:
        pass
    except:
        raise Exception(err_msg('original_dtypes_'))

    try:
        TwoRecurTestCls.original_dtypes_ = list('abcde')
        raise AssertionError(
            f"original_dtypes_ was allowed to be set by assignment when is read-only and should have raised AttributeError")
    except AttributeError:
        pass
    except Exception as e1:
        raise AssertionError(
            f"original_dtypes_ disallowed set by assignment but raised an error other than AttributeError --- {e1}")

    del err_msg
    # END BEFORE FIT **********************************************************

    # AFTER fit_transform() **********************************************************
    for data_dtype in ['np', 'pd']:
        if data_dtype == 'np':
            TEST_X, TEST_Y = NEW_X.copy(), NEW_Y.copy()
        elif data_dtype == 'pd':
            TEST_X, TEST_Y = NEW_X_DF.copy(), NEW_Y_DF.copy()

        OneRecurTestCls.fit_transform(TEST_X, TEST_Y)
        TRFM_X, TRFM_Y = TwoRecurTestCls.fit_transform(TEST_X, TEST_Y)

        assert OneRecurTestCls.n_features_in_ == x_cols, f"OneRecur.n_features_in_ after fit_transform() != number of originally passed columns"
        assert TwoRecurTestCls.n_features_in_ == x_cols, f"TwoRecur.n_features_in_ after fit_transform() != number of originally passed columns"

        # ONLY EXCEPTION SHOULD BE feature_names_in_ WHEN NUMPY
        if data_dtype == 'pd':
            assert np.array_equiv(TwoRecurTestCls.feature_names_in_, COLUMNS), \
                f"2 recurrence feature_names_in_ after fit_transform() != originally passed columns"

            assert np.array_equiv(TwoRecurTestCls.feature_names_in_, OneRecurTestCls.feature_names_in_), \
                f"2 recurrence feature_names_in_ after fit_transform() != 1 recurrence feature_names_in_ after fit_transform()"
        elif data_dtype == 'np':
            try:
                TwoRecurTestCls.feature_names_in_
                raise AssertionError
            except AttributeError:
                pass
            except AssertionError:
                print(f'\033[91mfTwoRecur.eature_names_in_ = ')
                print(TwoRecurTestCls.feature_names_in_)
                raise Exception(
                    f"2 recurrence feature_names_in_ was accessed even though fit was done with a numpy array")
            except:
                raise Exception(
                    f"accessing TwoRecur.feature_names_in_ with numpy fit() excepted for reason other than AttributeError")

        # n_features_in_ SHOULD BE EQUAL FOR OneRecurTestCls AND TwoRecurTestCls
        _, __ = OneRecurTestCls.n_features_in_, TwoRecurTestCls.n_features_in_
        assert _ == __, f"OneRecurTestCls.n_features_in_ ({_}) != TwoRecurTestcls.n_features_in_ ({__})"
        del _, __

        assert np.array_equiv(TwoRecurTestCls._original_dtypes, DTYPE_KEY), \
            f"_original_dtypes after fit_transform() != originally passed dtypes"

    # END AFTER fit_transform() *********************************************************

    del NEW_X, NEW_Y, NEW_X_DF, NEW_Y_DF, data_dtype, TEST_X, TEST_Y, OneRecurTestCls, TwoRecurTestCls, TRFM_X, TRFM_Y

    print(f'\033[92mAll ATTR access and accuracy tests passed for 2 recursions\033[0m')
# END ACCESS ATTR BEFORE fit() AND AFTER fit_transform(), ATTR ACCURACY; FOR 2 RECURSIONS **************************************


args = deepcopy(orig_args)
kwargs = deepcopy(orig_kwargs)

# ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM; FOR 2 RECURSIONS *******************************************************
if True:
    print(f'\n\033[92mTest METHOD access and accuracy before and after fit and transform for 2 recursions\033[0m')

    # CREATE AN INSTANCE WITH ONLY 1 RECURSION TO COMPARE 1X-TRFMED OBJECTS AGAINST 2X-TRFMED OBJECTS
    args = [3]
    kwargs['ignore_columns'] = None
    kwargs['ignore_nan'] = False
    kwargs['ignore_non_binary_integer_columns'] = False
    kwargs['ignore_float_columns'] = False
    kwargs['delete_axis_0'] = True
    kwargs['max_recursions'] = 1

    OneRecurTestCls = MinCountTransformer(*args, **kwargs)

    kwargs['max_recursions'] = 2

    TwoRecurTestCls = MinCountTransformer(*args, **kwargs)


    def not_fitted_tester(_name, cls_method, *args, **kwargs):

        try:
            cls_method(*args, **kwargs)
        except NotFittedError:
            pass
        except:
            raise Exception(f"{_name} excepted for reason other than NotFittedError")


    # ******************************************************************************************************************
    # vvv BEFORE fit_transform() vvv ***********************************************************************************

    # ** _base_fit()
    # ** _check_is_fitted()

    # ** test_threshold()
    not_fitted_tester('test_threshold', TwoRecurTestCls.test_threshold)

    # fit()
    # fit_transform()

    # get_feature_names_out()
    not_fitted_tester('get_feature_names_out', TwoRecurTestCls.get_feature_names_out, None)

    # get_metadata_routing()
    try:
        TwoRecurTestCls.get_metadata_routing()
    except NotImplementedError:
        pass
    except Exception as e1:
        raise AssertionError(
            f"get_metadata_routing() should have excepted with NotImplementedError but excepted with --- {e1} ")

    # get_params()
    # ALL PARAMS SHOULD BE THE SAME EXCEPT FOR max_recursions
    _ = OneRecurTestCls.get_params(True)
    del _['max_recursions']
    __ = TwoRecurTestCls.get_params(True)
    del __['max_recursions']
    assert _ == __, f"pre-fit 1 recursion instance get_params() != get_params() from 2 recursion instance"
    del _, __

    # get_row_support()
    not_fitted_tester('get_row_support', TwoRecurTestCls.get_row_support, True)

    # get_support()
    not_fitted_tester('get_support', TwoRecurTestCls.get_support, True)

    # ** _handle_X_y()

    # inverse_transform()
    not_fitted_tester('inverse_transform', TwoRecurTestCls.inverse_transform, X)

    # ** _make_instructions()
    # ** _must_be_fitted()
    # partial_fit()
    # ** _reset()

    # set_output()
    TwoRecurTestCls.set_output(transform='pandas_dataframe')

    # set_params()
    TestCls = MinCountTransformer(*args, **kwargs)
    KEYS = ['count_threshold', 'ignore_float_columns', 'ignore_non_binary_integer_columns', 'ignore_columns',
            'ignore_nan', 'handle_as_bool', 'delete_axis_0', 'reject_unseen_values', 'max_recursions', 'n_jobs']
    VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
    test_kwargs = dict((zip(KEYS, VALUES)))

    TestCls.set_params(**test_kwargs)
    ATTRS = [TestCls._count_threshold, TestCls._ignore_float_columns, TestCls._ignore_non_binary_integer_columns,
             TestCls._ignore_columns, TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
             TestCls._reject_unseen_values, TestCls._max_recursions, TestCls._n_jobs]
    for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
        assert _attr == _value, f'set_params() did not set {_key}'

    del TestCls, KEYS, VALUES, ATTRS

    TwoRecurTestCls = MinCountTransformer(*args, **kwargs)
    # transform()
    not_fitted_tester('transform', TwoRecurTestCls.transform, X, y)

    # ** _validate_delete_instr()
    # ** _validate_feature_names()
    # ** _validate()

    del not_fitted_tester

    # END ^^^ BEFORE fit_transform() ^^^ *******************************************************************************
    # ******************************************************************************************************************

    del TwoRecurTestCls
    TwoRecurTestCls = MinCountTransformer(*args, **kwargs)

    ONE_RCR_TRFM_X, ONE_RCR_TRFM_Y = OneRecurTestCls.fit_transform(X, y)
    TWO_RCR_TRFM_X, TWO_RCR_TRFM_Y = TwoRecurTestCls.fit_transform(X, y)

    # ******************************************************************************************************************
    # vvv AFTER fit_transform() vvv ************************************************************************************

    # ** _base_fit()
    # ** _check_is_fitted()

    # ** test_threshold()
    assert not np.array_equiv(ONE_RCR_TRFM_X, TWO_RCR_TRFM_X), \
        f"ONE_RCR_TRFM_X == TWO_RCR_TRFM_X when it shouldnt"

    assert OneRecurTestCls._total_counts_by_column != TwoRecurTestCls._total_counts_by_column, \
        f"OneRecurTestCls._total_counts_by_column == TwoRecurTestCls._total_counts_by_column when it shouldnt"

    _ONE_delete_instr = OneRecurTestCls._make_instructions(args[0])
    _TWO_delete_instr = TwoRecurTestCls._make_instructions(args[0])
    # THE FOLLOWING MUST BE TRUE BECAUSE TEST DATA BUILD VALIDATION REQUIRES 2 RECURSIONS W CERTAIN KWARGS DOES DELETE SOMETHING
    assert _TWO_delete_instr != _ONE_delete_instr, \
        f"fit-trfmed 2 recursion delete instr == fit-trfmed 1 recursion delete instr and should not"

    # THE NUMBER OF COLUMNS IN BOTH delete_instr DICTS ARE EQUAL
    assert len(_TWO_delete_instr) == len(_ONE_delete_instr), \
        f"number of columns in TwoRecurTestCls delete instr != number of columns in OneRecurTestCls delete instr"

    # LEN OF INSTRUCTIONS IN EACH COLUMN FOR TWO RECUR MUST BE >= INSTRUCTIONS FOR ONE RECUR BECAUSE THEYVE BEEN MELDED
    for col_idx in _ONE_delete_instr:
        _, __ = len(_TWO_delete_instr[col_idx]), len(_ONE_delete_instr[col_idx])
        assert _ >= __, \
            f"number of instruction in TwoRecurTestCls count is not >= number of instruction in OneRecurTestCls"

    # ALL THE ENTRIES FROM 1 RECURSION ARE IN THE MELDED INSTRUCTION DICT OUTPUT OF MULTIPLE RECURSIONS
    for col_idx in _TWO_delete_instr:
        for unq in _ONE_delete_instr[col_idx]:
            if unq in ['INACTIVE', 'DELETE COLUMN']:
                continue
            assert unq in _TWO_delete_instr[col_idx], f"{unq} is in two recursion delete instructions but not one recur"

    del _ONE_delete_instr, _TWO_delete_instr, _, __, col_idx, unq

    TwoRecurTestCls.test_threshold(clean_printout=True)
    print(f'^^^ mask building instructions should be displayed above ^^^')

    try:
        TwoRecurTestCls.test_threshold(2 * args[0])
        raise AssertionError(f"threshold != original threshold with multiple recursions should have raised ValueError")
    except ValueError:
        pass
    except Exception as e1:
        raise AssertionError(f"threshold != original threshold with multiple recursions should have raised ValueError "
                             f"but instead raised --- {e1}")

    # fit()
    # fit_transform()

    # get_feature_names_out() ******************************************************
    # vvv NO COLUMN NAMES PASSED (NP) vvv

    # WITH NO HEADER PASSED AND input_features=None, SHOULD RETURN SLICED ['x0', ..., 'x(n-1)]
    if not np.array_equiv(TwoRecurTestCls.get_feature_names_out(None),
                          np.array([f"x{i}" for i in range(len(COLUMNS))])[TwoRecurTestCls.get_support(False)]):
        raise AssertionError(f"get_feature_names_out(None) after fit_transform() != sliced array of generic headers")

    # WITH NO HEADER PASSED, SHOULD RAISE ValueError IF len(input_features) != n_features_in_
    try:
        TwoRecurTestCls.get_feature_names_out([f"x{i}" for i in range(2 * len(COLUMNS))])
    except ValueError:
        pass
    except Exception as e1:
        raise AssertionError(
            f"TwoRecur.get_feature_names_out(TOO_MANY_COLUMNS) after fit() should have raised ValueError but raised --- {e1}")

    # WHEN NO HEADER PASSED TO fit_transform() AND VALID input_features, SHOULD RETURN SLICED PASSED COLUMNS
    RETURNED_FROM_GFNO = TwoRecurTestCls.get_feature_names_out(COLUMNS)
    if not isinstance(RETURNED_FROM_GFNO, np.ndarray):
        raise AssertionError(
            f"TwoRecur.get_feature_names_out should return numpy.ndarray, but returned {type(RETURNED_FROM_GFNO)}")
    if not np.array_equiv(RETURNED_FROM_GFNO, np.array(COLUMNS)[TwoRecurTestCls.get_support(False)]):
        raise AssertionError(f"TwoRecur.get_feature_names_out() did not return original columns")

    del RETURNED_FROM_GFNO, TwoRecurTestCls

    # END ^^^ NO COLUMN NAMES PASSED (NP) ^^^

    # vvv COLUMN NAMES PASSED (PD) vvv
    OneRecurTestCls = MinCountTransformer(*args, **kwargs)
    ONE_RCR_TRFM_X, ONE_RCR_TRFM_Y = OneRecurTestCls.fit_transform(pd.DataFrame(data=X, columns=COLUMNS), y)

    TwoRecurTestCls = MinCountTransformer(*args, **kwargs)
    TWO_RCR_TRFM_X, TWO_RCR_TRFM_Y = TwoRecurTestCls.fit_transform(pd.DataFrame(data=X, columns=COLUMNS), y)

    # WITH HEADER PASSED AND input_features=None:
    # SHOULD RETURN SLICED ORIGINAL COLUMNS
    if not np.array_equiv(TwoRecurTestCls.get_feature_names_out(None),
                          np.array(COLUMNS)[TwoRecurTestCls.get_support(False)]):
        raise AssertionError(
            f"TwoRecur.get_feature_names_out(None) after fit_transform() != sliced originally passed columns")

    # WHEN HEADER PASSED TO fit_transform() AND input_features IS THAT HEADER, SHOULD RETURN SLICED VERSION OF THAT HEADER
    RETURNED_FROM_GFNO = TwoRecurTestCls.get_feature_names_out(COLUMNS)
    if not isinstance(RETURNED_FROM_GFNO, np.ndarray):
        raise AssertionError(
            f"get_feature_names_out should return numpy.ndarray, but returned {type(RETURNED_FROM_GFNO)}")
    if not np.array_equiv(RETURNED_FROM_GFNO, np.array(COLUMNS)[TwoRecurTestCls.get_support(False)]):
        raise AssertionError(f"get_feature_names_out() did not return original columns")

    del RETURNED_FROM_GFNO
    # END ^^^ COLUMN NAMES PASSED (PD) ^^^
    # END get_feature_names_out() *************************************************

    # get_metadata_routing()
    try:
        TwoRecurTestCls.get_metadata_routing()
    except NotImplementedError:
        pass
    except Exception as e1:
        raise AssertionError(
            f"get_metadata_routing() should have excepted with NotImplementedError but excepted with --- {e1} ")

    # get_params()
    # ALL PARAMS SHOULD BE THE SAME EXCEPT FOR max_recursions
    _ = OneRecurTestCls.get_params(True)
    del _['max_recursions']
    __ = TwoRecurTestCls.get_params(True)
    del __['max_recursions']
    assert _ == __, f"pre-fit 1 recursion instance get_params() != get_params() from 2 recursion instance"
    del _, __

    # get_row_support()
    for _indices in [True, False]:
        _ONE = OneRecurTestCls.get_row_support(_indices)
        _TWO = TwoRecurTestCls.get_row_support(_indices)

        assert isinstance(_ONE, np.ndarray), f"get_row_support() for 1 recursion did not return numpy.ndarray"
        assert isinstance(_TWO, np.ndarray), f"get_row_support() for 2 recursions did not return numpy.ndarray"

        if not _indices:
            assert _ONE.dtype == 'bool', f"get_row_support with indices=False for 1 recursion did not return a boolean array"
            assert _TWO.dtype == 'bool', f"get_row_support with indices=False for 2 recursions did not return a boolean array"
            # len(ROW SUPPORT TWO RECUR) AND len(ROW SUPPORT ONE RECUR) MUST EQUAL NUMBER OF ROWS IN X
            assert len(_ONE) == x_rows, f"row_support vector length for 1 recursion != rows in passed data"
            assert len(_TWO) == x_rows, f"row_support vector length for 2 recursions != rows in passed data"
            # NUMBER OF Trues in ONE RECUR MUST == NUMBER OF ROWS IN ONE RCR TRFM X; SAME FOR TWO RCR
            assert sum(_ONE) == ONE_RCR_TRFM_X.shape[0], f"one rcr Trues IN row_support != TRFM X rows"
            assert sum(_TWO) == TWO_RCR_TRFM_X.shape[0], f"two rcr Trues IN row_support != TRFM X rows"
            # NUMBER OF Trues IN ONE RECUR MUST BE >= NUMBER OF Trues IN TWO RECUR
            assert sum(_ONE) >= sum(_TWO), f"two recursion has more rows kept in it that one recursion"
            # ANY Trues IN TWO RECUR MUST ALSO BE True IN ONE RECUR
            assert np.unique(_ONE[_TWO])[
                       0] == True, f"Rows that are to be kept in 2nd recur (True) were False in 1st recur"
        elif _indices:
            assert 'int' in str(
                _ONE.dtype).lower(), f"get_row_support with indices=True for 1 recursion did not return an array of integers"
            assert 'int' in str(
                _TWO.dtype).lower(), f"get_row_support with indices=True for 2 recursions did not return an array of integers"
            # len(row_support) ONE RECUR MUST == NUMBER OF ROWS IN ONE RCR TRFM X; SAME FOR TWO RCR
            assert len(_ONE) == ONE_RCR_TRFM_X.shape[0], f"one rcr len(row_support) as idxs does not equal TRFM X rows"
            assert len(_TWO) == TWO_RCR_TRFM_X.shape[0], f"two rcr len(row_support) as idxs does not equal TRFM X rows "
            # NUMBER OF ROW IDXS IN ONE RECUR MUST BE >= NUM ROW IDXS IN TWO RECUR
            assert len(_ONE) >= len(_TWO), f"two recursion has more rows kept in it that one recursion"
            # INDICES IN TWO RECUR MUST ALSO BE IN ONE RECUR
            for row_idx in _TWO:
                assert row_idx in _ONE, f"Rows that are to be kept by 2nd recur were not kept by 1st recur"

    del _ONE, _TWO, row_idx, _indices, ONE_RCR_TRFM_X, ONE_RCR_TRFM_Y, TWO_RCR_TRFM_X, TWO_RCR_TRFM_Y

    # get_support()
    for _indices in [True, False]:
        _ = OneRecurTestCls.get_support(_indices)
        __ = TwoRecurTestCls.get_support(_indices)
        assert isinstance(_, np.ndarray), f"2 recursion get_support() did not return numpy.ndarray"
        assert isinstance(__, np.ndarray), f"2 recursion get_support() did not return numpy.ndarray"

        if not _indices:
            assert _.dtype == 'bool', f"1 recursion get_support with indices=False did not return a boolean array"
            assert __.dtype == 'bool', f"2 recursion get_support with indices=False did not return a boolean array"
            # len(ROW SUPPORT TWO RECUR) AND len(ROW SUPPORT ONE RECUR) MUST EQUAL NUMBER OF COLUMNS IN X
            assert len(_) == x_cols, f"1 recursion len(get_support({_indices})) != X columns"
            assert len(__) == x_cols, f"2 recursion len(get_support({_indices})) != X columns"
            # NUM COLUMNS IN 1 RECURSION MUST BE <= NUM COLUMNS IN X
            assert sum(_) <= x_cols, f"impossibly, number of columns kept by 1 recursion > number of columns in X"
            # NUM COLUMNS IN 2 RECURSION MUST BE <= NUM COLUMNS IN 1 RECURSION
            assert sum(__) <= sum(
                _), f"impossibly, number of columns kept by 2 recursion > number of columns kept by 1 recursion"
            # INDICES IN TWO RECUR MUST ALSO BE IN ONE RECUR
            assert np.unique(_[__])[
                       0] == True, f"Columns that are to be kept in 2nd recur (True) were False in 1st recur"
        elif _indices:
            assert 'int' in str(
                _.dtype).lower(), f"1 recursion get_support with indices=True did not return an array of integers"
            assert 'int' in str(
                __.dtype).lower(), f"2 recursion get_support with indices=True did not return an array of integers"
            # ONE RECURSION COLUMNS MUST BE <= n_features_in_
            assert len(_) <= x_cols, f"impossibly, 1 recursion len(get_support({_indices})) > X columns"
            # TWO RECURSION COLUMNS MUST BE <= ONE RECURSION COLUMNS
            assert len(__) <= len(
                _), f"2 recursion len(get_support({_indices})) > 1 recursion len(get_support({_indices}))"
            # INDICES IN TWO RECUR MUST ALSO BE IN ONE RECUR
            for col_idx in __:
                assert col_idx in _, f"Columns that are to be kept by 2nd recur were not kept by 1st recur"

    del TwoRecurTestCls, _, __, _indices, col_idx

    # ** _handle_X_y()

    # inverse_transform() ********************
    TwoRecurTestCls = MinCountTransformer(*args, **kwargs)
    TRFM_X, TRFM_Y = TwoRecurTestCls.fit_transform(X, y)  # X IS NP ARRAY

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
                    TEST_X = pd.DataFrame(data=TEST_X, columns=np.hstack((__[TRFM_MASK], np.char.upper(__[TRFM_MASK]))))

            try:
                TwoRecurTestCls.inverse_transform(TEST_X)
                assert diff_cols == 'same', \
                    f"inverse_transform() did not raise any exception for X with invalid number of columns"
            except ValueError:
                pass
            except Exception as e1:
                raise AssertionError(
                    f"inverse_transform() invalid number of columns should have raised ValueError but raised --- {e1}")

    INV_TRFM_X = TwoRecurTestCls.inverse_transform(TRFM_X)

    assert isinstance(INV_TRFM_X, np.ndarray), f"output of inverse_transform() is not a numpy array"
    assert INV_TRFM_X.shape[0] == TRFM_X.shape[0], f"rows in output of inverse_transform() do not match input rows"
    assert INV_TRFM_X.shape[
               1] == TwoRecurTestCls.n_features_in_, f"columns in output of inverse_transform() do not match originally fitted columns"

    __ = np.logical_not(TwoRecurTestCls.get_support(False))
    assert np.array_equiv(INV_TRFM_X[:, __], np.zeros((TRFM_X.shape[0], sum(__)))), \
        f"back-filled parts of inverse_transform() output do not slice to a zero array"
    del __

    assert np.array_equiv(TRFM_X.astype(str), INV_TRFM_X[:, TwoRecurTestCls.get_support(False)].astype(str)), \
        f"output of inverse_transform() does not reduce back to the output of transform()"

    del TwoRecurTestCls, TRFM_X, TRFM_Y, TRFM_MASK, obj_type, diff_cols, TEST_X, INV_TRFM_X

    # END inverse_transform() **********

    # ** _make_instructions()
    # ** _must_be_fitted()
    # partial_fit()
    # ** _reset()

    # set_output()
    TwoRecurTestCls = MinCountTransformer(*args, **kwargs)
    TwoRecurTestCls.set_output(transform='pandas_dataframe')
    assert TwoRecurTestCls._output_transform == 'pandas_dataframe'
    TwoRecurTestCls.fit_transform(X, y)
    assert TwoRecurTestCls._output_transform == 'pandas_dataframe'

    del TwoRecurTestCls

    # set_params()
    TestCls = MinCountTransformer(*args, **kwargs)
    KEYS = ['count_threshold', 'ignore_float_columns', 'ignore_non_binary_integer_columns', 'ignore_columns',
            'ignore_nan', 'handle_as_bool', 'delete_axis_0', 'reject_unseen_values', 'max_recursions', 'n_jobs']
    VALUES = [4, False, False, [0], False, [2], True, True, 2, 4]
    test_kwargs = dict((zip(KEYS, VALUES)))

    TestCls.set_params(**test_kwargs)
    ATTRS = [TestCls._count_threshold, TestCls._ignore_float_columns, TestCls._ignore_non_binary_integer_columns,
             TestCls._ignore_columns, TestCls._ignore_nan, TestCls._handle_as_bool, TestCls._delete_axis_0,
             TestCls._reject_unseen_values, TestCls._max_recursions, TestCls._n_jobs]
    for _key, _attr, _value in zip(KEYS, ATTRS, VALUES):
        assert _attr == _value, f'set_params() did not set {_key}'

    del TestCls, KEYS, VALUES, ATTRS

    # transform()
    # ** _validate_delete_instr()
    # ** _validate_feature_names()
    # ** _validate()

    del OneRecurTestCls,

    # END ^^^ AFTER fit_transform() ^^^ ********************************************************************************
    # ******************************************************************************************************************

    print(f'\033[92mAll METHOD access and accuracy tests passed for 2 recursions\033[0m')

# END ACCESS METHODS BEFORE AND AFTER FIT AND TRANSFORM; FOR 2 RECURSIONS ************************************************


del NO_NAN_X, DTYPE_KEY

print(f'\n\nTotal test time = {time.perf_counter() - test_start_time: ,.0f} s')
















