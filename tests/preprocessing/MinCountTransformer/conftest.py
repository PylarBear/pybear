# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import numpy.typing as npt
import joblib
import uuid

from typing_extensions import Union, Iterable, Callable



# at the bottom are fixtures used for testing mmct in
# conftest__mmct__test.py


# Build y for MCT tests (not mmct test!) ** * ** * ** * ** * ** * ** * **

@pytest.fixture(scope='session')
def y(y_rows, y_cols):
    return np.random.randint(0, 2, (y_rows, y_cols), dtype=np.uint8)

# Build y for MCT tests (not mmct test!) ** * ** * ** * ** * ** * ** * **

# build X, NO_NAN_X, DTYPE_KEY, x_rows, x_cols for MCT test (not mmct test!)

@pytest.fixture(scope='session')
def build_test_objects_for_MCT(mmct, _rows, _cols, _args):

    # This constructs a test array "X" of randomly filled vectors that have
    # certain criteria like a certain number of certain types of columns,
    # certain amounts of uniques, certain proportions of uniques, to make
    # X manipulable with certain outcomes across all tests. The vectors
    # are filled randomly and may not always be generated with the
    # expected characteristics in one shot, so this iterates over and
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
        # CREATE _cols COLUMNS OF BINARY INTEGERS
        _X = np.random.randint(0, 2, (_rows, _cols)).astype(object)
        # CREATE _cols COLUMNS OF NON-BINARY INTEGERS
        _X = np.hstack((
            _X, np.random.randint(0, _rows // 15, (_rows, _cols)).astype(object)
        ))
        # CREATE _cols COLUMNS OF FLOATS
        _X = np.hstack(
            (_X, np.random.uniform(0, 1, (_rows, _cols)).astype(object)))
        # CREATE _cols COLUMNS OF STRS
        _alpha = 'abcdefghijklmnopqrstuvwxyz'
        _alpha = _alpha + _alpha.upper()
        for _ in range(_cols):
            _X = np.hstack((_X,
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

        _X = np.hstack((_X, DUM_STR_COL.reshape((-1, 1)).astype(object)))
        del DUM_STR_COL

        # _X SHAPE SHOULD BE (x_rows, 4 * x_cols + 1)
        x_rows = _rows
        x_cols = 4 * _cols + 1

        _DTYPE_KEY = [k for k in ['int', 'int', 'float', 'obj'] for j in
                      range(_cols)]
        _DTYPE_KEY += ['obj']

        # KEEP THIS FOR TESTING IF DTYPES RETRIEVED CORRECTLY WITH np.nan MIXED IN
        _NO_NAN_X = _X.copy()

        # FLOAT/STR ONLY --- NO_NAN_X MUST BE REDUCED WHEN STR COLUMNS ARE
        # TRANSFORMED
        FLOAT_STR_X = _NO_NAN_X[:, 2 * _cols:4 * _cols].copy()
        # mmct() args = MOCK_X, MOCK_Y, ign_col, ign_nan, ignore_flt_col,
        # ignore_non_binary_int_col, handle_as_bool, delete_axis_0, ct_thresh
        _X1 = mmct(_X, None, None, True, True, True, None, True, *_args)
        if np.array_equiv(_X1, FLOAT_STR_X):
            del _X1
            continue
        del _X1

        del FLOAT_STR_X

        # PEPPER 10% OF CORE DATA WITH np.nan
        for _ in range(x_rows * x_cols // 10):
            row_coor = np.random.randint(0, x_rows)
            col_coor = np.random.randint(0, x_cols - 1)
            if col_coor < 3 * _cols:
                _X[row_coor, col_coor] = np.nan
            elif col_coor >= 3 * _cols:
                _X[row_coor, col_coor] = 'nan'
        del row_coor, col_coor

        # MAKE EVERY CORE COLUMN HAVE 2 VALUES THAT CT FAR EXCEEDS
        # count_threshold SO DOESNT ALLOW FULL DELETE
        _repl = x_rows // 3
        _get_idxs = lambda: np.random.choice(range(x_rows), _repl, replace=False)
        # 24_06_05_13_16_00 the assignments here cannot be considated using
        # lambda functions - X is being passed to mmct and it is saying cannot
        # pickle
        for idx in range(_cols):
            _X[_get_idxs(), _cols + idx] = \
                int(np.random.randint(0, x_rows // 20) + idx)
            _X[_get_idxs(), _cols + idx] = \
                int(np.random.randint(0, x_rows // 20) + idx)
            _X[_get_idxs(), 2 * _cols + idx] = np.random.uniform(0, 1) + idx
            _X[_get_idxs(), 2 * _cols + idx] = np.random.uniform(0, 1) + idx
            _X[_get_idxs(), 3 * _cols + idx] = _alpha[:x_rows // 15][idx]
            _X[_get_idxs(), 3 * _cols + idx] = _alpha[:x_rows // 15][idx + 1]

        del idx, _repl, _alpha

        # VERIFY ONE RECURSION OF mmct DELETED THE SACRIFICIAL LAST COLUMN
        # (CORE COLUMNS ARE RIGGED TO NOT BE DELETED)
        # MOCK_X, MOCK_Y, ign_col, ign_nan, ignore_non_binary_int_col,
        # ignore_flt_col, handle_as_bool, delete_axis_0, ct_thresh
        _X1 = mmct(_X, None, None, False, False, False, None, True, *_args)
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
        for flt_col_idx in range(2 * _cols, 3 * _cols, 1):
            _max_ct = np.unique(
                _X[:, flt_col_idx], return_counts=True
            )[1].max(axis=0)
            if _max_ct < _args[0]:
                continue
        del _max_ct

        # TRFM OF NON-BINARY INTEGER COLUMNS MUST NOT DELETE EVERYTHING,
        # BUT MUST DELETE SOMETHING
        try:
            _X1 = mmct(
                _X[:, _cols:(2 * _cols)].copy(),
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
            if np.array_equiv(_X1, _X[:, _cols:(2 * _cols)].copy()):
                continue
        except:
            continue

        try_again = False
        # IF ALL CTS OF EVERY STR UNIQUE IS >= THRESHOLD, BUILD NEW X
        for str_col_idx in range(x_cols - 1, x_cols - _cols - 1, -1):
            _min_ct = min(np.unique(_X[:, str_col_idx], return_counts=True)[1])
            if _min_ct >= _args[0]:
                try_again = True
                break
        if try_again:
            continue
        del _min_ct

        # IF X CANNOT TAKE 2 RECURSIONS WITH THRESHOLD==3, BUILD NEW X
        try_again = False
        _X1 = mmct(_X, None, None, False, False, False, False, True, 3)
        # MOCK_X, MOCK_Y, ign_col, ign_nan, ignore_non_binary_int_col,
        # ignore_flt_col, handle_as_bool, delete_axis_0, ct_thresh
        try:
            # THIS SHOULD EXCEPT IF ALL ROWS/COLUMNS WOULD BE DELETED
            _X2 = mmct(_X1, None, None, False, False, False, False, True, 3)
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






# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **



# this is a fixture used to referee against MinCountTransformer in
# MinCountTransformer_test. It also is used to validate that X meets
# certain rigged conditions for testing MinCountTransformer.
# Tests for this fixture are in
# conftest__mmct__test.py. The fixtures used to run the tests in
# conftest__mmct__test.py are at the bottom of this module.

@pytest.fixture(scope='session')
def mmct():

    def trfm(
        MOCK_X: npt.NDArray[any],
        MOCK_Y: Union[npt.NDArray[int], None],
        ignore_columns: Union[Iterable[int], None],
        ignore_nan: bool,
        ignore_non_binary_integer_columns: bool,
        ignore_float_columns: bool,
        handle_as_bool: Union[Iterable[int], None],
        delete_axis_0: bool,
        count_threshold: int
    ) -> Union[tuple[npt.NDArray[any], npt.NDArray[int]], npt.NDArray[any]]:

        # GET UNIQUES:
        @joblib.wrap_non_picklable_objects
        def get_uniques(X_COLUMN: np.ndarray) -> np.ndarray:
            # CANT HAVE X_COLUMN AS DTYPE object!
            og_dtype = X_COLUMN.dtype
            WIP_X_COLUMN = X_COLUMN.copy()
            try:
                WIP_X_COLUMN = WIP_X_COLUMN.astype(np.float64)
            except:
                WIP_X_COLUMN = WIP_X_COLUMN.astype(str)

            UNIQUES, COUNTS = np.unique(WIP_X_COLUMN, return_counts=True)
            del WIP_X_COLUMN
            UNIQUES = UNIQUES.astype(og_dtype)
            del og_dtype

            return UNIQUES, COUNTS


        ACTIVE_C_IDXS = [i for i in range(MOCK_X.shape[1])]
        if ignore_columns:
            for i in ignore_columns:
                try:
                    ACTIVE_C_IDXS.remove(i)
                except ValueError:
                    raise ValueError(f'ignore_columns column index {i} out '
                        f'of bounds for data with {MOCK_X.shape[1]} columns')
                except Exception as e1:
                    raise Exception(f'ignore_columns remove from ACTIVE_C_IDXS '
                        f'except for reason other than ValueError --- {e1}')

        # DONT HARD-CODE backend, ALLOW A CONTEXT MANAGER TO SET
        UNIQUES_COUNTS_TUPLES = joblib.Parallel(return_as='list', n_jobs=-1)(
            joblib.delayed(get_uniques)(MOCK_X[:, c_idx]) for c_idx in ACTIVE_C_IDXS
        )

        for col_idx in range(MOCK_X.shape[1]):
            if col_idx not in ACTIVE_C_IDXS:
                UNIQUES_COUNTS_TUPLES.insert(col_idx, None)

        del ACTIVE_C_IDXS

        get_support_ = np.ones(MOCK_X.shape[1]).astype(bool)

        # GET DTYPES ** ** ** ** ** **
        DTYPES = [None for _ in UNIQUES_COUNTS_TUPLES]
        for col_idx in range(len(UNIQUES_COUNTS_TUPLES)):
            if UNIQUES_COUNTS_TUPLES[col_idx] is None:
                continue

            UNIQUES_COUNTS_TUPLES[col_idx] = list(UNIQUES_COUNTS_TUPLES[col_idx])
            UNIQUES, COUNTS = UNIQUES_COUNTS_TUPLES[col_idx]

            try:
                MASK = np.logical_not(np.isnan(UNIQUES.astype(np.float64)))
                NO_NAN_UNIQUES = UNIQUES[MASK]
                del MASK
                NO_NAN_UNIQUES_AS_FLT = NO_NAN_UNIQUES.astype(np.float64)
                NO_NAN_UNIQUES_AS_INT = NO_NAN_UNIQUES_AS_FLT.astype(np.int32)
                if np.array_equiv(NO_NAN_UNIQUES_AS_INT, NO_NAN_UNIQUES_AS_FLT):
                    if len(NO_NAN_UNIQUES) == 1:
                        DTYPES[col_idx] = 'constant'
                    elif len(NO_NAN_UNIQUES) == 2:
                        DTYPES[col_idx] = 'bin_int'
                    elif len(NO_NAN_UNIQUES) > 2:
                        DTYPES[col_idx] = 'non_bin_int'
                        if ignore_non_binary_integer_columns:
                            UNIQUES_COUNTS_TUPLES[col_idx] = None
                            continue
                else:
                    DTYPES[col_idx] = 'float'
                    if ignore_float_columns:
                        UNIQUES_COUNTS_TUPLES[col_idx] = None
                        continue

                del NO_NAN_UNIQUES, NO_NAN_UNIQUES_AS_INT, NO_NAN_UNIQUES_AS_FLT

            except:
                DTYPES[col_idx] = 'obj'

            del UNIQUES, COUNTS
        # END GET DTYPES ** ** ** ** ** **


        for col_idx in range(len(UNIQUES_COUNTS_TUPLES) - 1, -1, -1):

            if UNIQUES_COUNTS_TUPLES[col_idx] is None:
                continue

            UNIQUES, COUNTS = UNIQUES_COUNTS_TUPLES[col_idx]

            if handle_as_bool and col_idx in handle_as_bool:

                if DTYPES[col_idx] == 'obj':
                    raise ValueError(
                        f"MOCK X trying to do handle_as_bool on str column"
                    )

                NEW_UNQ_CT_DICT = {0:0, 1:0}
                for u,c in zip(UNIQUES, COUNTS):
                    if str(u).lower() == 'nan':
                        NEW_UNQ_CT_DICT[u] = c
                    elif u != 0:
                        NEW_UNQ_CT_DICT[1] += c
                    elif u == 0:
                        NEW_UNQ_CT_DICT[0] = c
                UNIQUES = list(NEW_UNQ_CT_DICT.keys())
                COUNTS = list(NEW_UNQ_CT_DICT.values())
                del NEW_UNQ_CT_DICT


            ROW_MASK = np.zeros(MOCK_X.shape[0])
            for u_idx, unq, ct in zip(
                    range(len(UNIQUES) - 1, -1, -1),
                    np.flip(UNIQUES),
                    np.flip(COUNTS)
                ):

                if ignore_nan and str(unq).lower()=='nan':
                    continue

                if ct < count_threshold:
                    try:
                        NAN_MASK = \
                            np.isnan(MOCK_X[:, col_idx].astype(np.float64))
                    except:
                        NAN_MASK = \
                            (np.char.lower(MOCK_X[:, col_idx].astype(str)) == 'nan')

                    if str(unq).lower()=='nan':
                        ROW_MASK += NAN_MASK
                    else:
                        try:
                            if col_idx in handle_as_bool:
                                NOT_NAN_MASK = np.logical_not(NAN_MASK)
                                _ = (MOCK_X[NOT_NAN_MASK, col_idx].astype(bool) == unq)
                                ROW_MASK[NOT_NAN_MASK] += _
                                del NOT_NAN_MASK, _
                            else:
                                # JUST TO SEND INTO not handle_as_bool CODE
                                raise Exception
                        except:
                            ROW_MASK += (MOCK_X[:, col_idx] == unq)

                    del NAN_MASK

                    # vvv USE LEN LATER TO INDICATE TO DELETE COLUMN
                    UNIQUES = np.delete(UNIQUES, u_idx)
                    COUNTS = np.delete(COUNTS, u_idx)

            if DTYPES[col_idx] == 'constant':
                pass
            elif DTYPES[col_idx] == 'bin_int' and not delete_axis_0:
                pass
            elif (handle_as_bool and col_idx in handle_as_bool) \
                    and not delete_axis_0:
                pass
            else:
                ROW_MASK = np.logical_not(ROW_MASK)
                MOCK_X = MOCK_X[ROW_MASK, :]
                if MOCK_Y is not None:
                    MOCK_Y = MOCK_Y[ROW_MASK]

                del ROW_MASK

            if len([_ for _ in UNIQUES if str(_).lower() != 'nan']) <= 1:
                get_support_[col_idx] = False
                MOCK_X = np.delete(MOCK_X, col_idx, axis=1)

        if MOCK_Y is not None:
            return MOCK_X, MOCK_Y
        else:
            return MOCK_X


    return trfm



# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **



# Build vectors for mmct test. The process below builds random, but rigged,
# vectors that are used to test mmct (mock_min_count_transformer, above).
# The test for mmct is in conftest__mmct__test.py. mmct is then used to
# test MinCountTransformer. There are 7 fixtures below. The first five
# fixtures build 5 different types of test vectors (non-binary integer,
# bin, float, str, bool). The process in the 7th fixture
# (build_vectors_for_mock_mct_test) that generates valid vectors for all
# the types except float is trial-and-error. Because of this, the fixtures
# for those 4 other types of vectors must generate a new vector each time
# called and must stay as fixture factories. The 6th fixture,
# trfm_validator, is used within build_vectors_for_mock_mct_test to
# validate if each of the vectors meet the qualifications to be used to
# test mmct. If not, another iteration of vectors is generated and
# validated. Once a set of 5 valid vectors are found, they are used in
# conftest__mmct__test. The iterations take place over allowable
# construction parameters and once the set of 5 is found, the construction
# parameters are also returned. The vectors must be constructed
# simultaneously because they must share a common valid min cutoff
# threshold and number of rows. The five vectors and the construction
# parameters are then split into individual fixtures to be passed to
# conftest__mmct__test.


# MOCK_X_BIN ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
# binary integer vector

@pytest.fixture(scope='session')
def build_MOCK_X_BIN() -> Callable:

    """
    build MOCK_X_BIN for use below in build_vectors_for_mock_mct_test()
    This must be a fixture factory to return something different at each
    call.

    """

    def foo(_rows: int, _source_len: int) -> npt.NDArray[int]:

        _p = [1 - 1 / _source_len, 1 / _source_len]

        return np.random.choice([0, 1], _rows, p=_p).reshape((-1, 1))

    return foo

# END MOCK_X_BIN ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# MOCK_X_NBI ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
# non-binary integer vector

@pytest.fixture(scope='session')
def build_MOCK_X_NBI() -> Callable:

    """
    build MOCK_X_NBI for use below in build_vectors_for_mock_mct_test()
    This must be a fixture factory to return something different at each
    call.

    """

    def foo(_rows: int, _source_len: int) -> npt.NDArray[int]:

        return np.random.randint(0, _source_len, (_rows,)).reshape((-1, 1))

    return foo

# END MOCK_X_NBI ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# MOCK_X_FLT ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
# float vector

@pytest.fixture(scope='session')
def build_MOCK_X_FLT() -> Callable:

    """
    build MOCK_X_FLT for use below in build_vectors_for_mock_mct_test()
    This must be a factory to receive the _rows arg.

    """

    def foo(_rows: int) -> npt.NDArray[float]:

        return np.random.uniform(0, 1, (_rows,)).reshape((-1, 1))

    return foo


# END MOCK_X_FLT ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# MOCK_X_STR ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
# string vector

@pytest.fixture(scope='session')
def build_MOCK_X_STR() -> Callable:

    """
    build MOCK_X_STR for use below in build_vectors_for_mock_mct_test().
    This must be a fixture factory to return something different at each
    call.

    """

    def foo(_rows:int, _source_len:int) -> npt.NDArray[str]:

        _alpha = 'qwertyuiopasdfghjklzxcvbnm'
        _alpha += _alpha.upper()
        _STRS = list(_alpha[:_source_len])
        del _alpha

        return np.random.choice(_STRS, _rows, replace=True).reshape((-1, 1))

    return foo

# END MOCK_X_STR ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# MOCK_X_BOOL ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

@pytest.fixture(scope='session')
def build_MOCK_X_BOOL() -> Callable:

    """
    build MOCK_X_BOOL for use below in build_vectors_for_mock_mct_test().
    This must be a fixture factory to return something different at each
    call.

    """

    def foo(_rows:int, _source_len:int) -> npt.NDArray[bool]:
        MOCK_X_BOOL = np.zeros(_rows)
        _idx = np.random.choice(range(_rows), _source_len, replace=False)
        MOCK_X_BOOL[_idx] = np.arange(1, _source_len + 1)
        del _idx
        MOCK_X_BOOL = MOCK_X_BOOL.reshape((-1, 1))

        return MOCK_X_BOOL

    return foo

# END MOCK_X_BOOL ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


@pytest.fixture(scope='session')
def trfm_validator() -> Callable:

    """
    For use below in build_vectors_for_mock_mct_test(). This determines
    if any of the 4 randomly generated vectors is suitable to test mmct.
    Float is always valid and does not need validation. This must be a
    function factory.

    """

    def foo(
        VECTOR: npt.NDArray[Union[int, str, bool]],
        _thresh: int,
        _source_len: int
    ) -> bool:

        # STR & NBI: MOCK_MCT MUST DELETE ALL @ 2X THRESH, NONE @ 1/2 THRESH,
        # AND AT LEAST 1 BUT LEAVE 2+ BEHIND AT THRESH
        _low = _thresh // 2
        _mid = _thresh
        _high = 2 * _thresh

        UNQ_CT_DICT = dict((zip(*np.unique(VECTOR, return_counts=True))))
        CTS = list(UNQ_CT_DICT.values())

        # ALWAYS MUST HAVE AT LEAST 2 UNQS IN THE COLUMN
        if len(CTS) < 2:
            return False

        # IF IS BIN INT
        if len(CTS) == 2 and min(UNQ_CT_DICT) == 0 and max(UNQ_CT_DICT) == 1:
            if min(CTS) not in range(_low, _high):
                return False
            return True

        # FOR handle_as_bool
        if sum(sorted(CTS)[:-1]) == _source_len:
            if not max(CTS) >= 2 * _thresh:
                return False
            if sum(sorted(CTS)[:-1]) >= 2 * _thresh:
                return False
            return True

        if max(CTS) >= _high:
            return False

        if min(CTS) < _low + 1:
            return False

        if not sorted(CTS)[-2] >= _mid:
            return False

        if not min(CTS) < _mid:
            return False

        return True

    return foo


@pytest.fixture(scope='session')
def build_vectors_for_mock_mct_test(
    build_MOCK_X_BIN,
    build_MOCK_X_NBI,
    build_MOCK_X_FLT,
    build_MOCK_X_STR,
    build_MOCK_X_BOOL,
    trfm_validator
) -> tuple[
    npt.NDArray[int], npt.NDArray[int], npt.NDArray[float], npt.NDArray[str],
    npt.NDArray[bool], int, int, int
]:

    """
    Use the vector generator factories and the validator fixture from
    above to produce 5 vectors suitably rigged to test mmct. This fixture
    iterates over various allowable construction parameters of rows,
    thresholds, and number of uniques to use. Also return the
    construction parameters under which the valid vectors were generated.

    """

    _quit = False
    for _rows in range(10, 101):
        for _thresh in range(6, _rows):
            for _source_len in range(_thresh, _rows):

                # _thresh must be less than _rows
                # _source_len must be less than _thresh

                _MOCK_X_BIN = build_MOCK_X_BIN(_rows, _source_len)
                _MOCK_X_NBI = build_MOCK_X_NBI(_rows, _source_len)
                _MOCK_X_FLT = build_MOCK_X_FLT(_rows)
                _MOCK_X_STR = build_MOCK_X_STR(_rows, _source_len)
                _MOCK_X_BOOL = build_MOCK_X_BOOL(_rows, _source_len)

                _good = 0
                _good += trfm_validator(_MOCK_X_BIN, _thresh, _source_len)
                _good += trfm_validator(_MOCK_X_NBI, _thresh, _source_len)
                _good += trfm_validator(_MOCK_X_STR, _thresh, _source_len)
                _good += trfm_validator(_MOCK_X_BOOL, _thresh, _source_len)
                if _good == 4:
                    _quit = True

                if _quit:
                    break
            if _quit:
                break
        if _quit:
            break
    else:
        raise ValueError(f'Unable to find suitable test vectors')


    return _MOCK_X_BIN, _MOCK_X_NBI, _MOCK_X_FLT, _MOCK_X_STR, _MOCK_X_BOOL, \
        _thresh, _source_len, _rows


# split the vectors and build parameters from build_vectors_for_mock_mct_test
# into individual fixtures

@pytest.fixture(scope='session')
def MOCK_X_BIN(build_vectors_for_mock_mct_test):
    return build_vectors_for_mock_mct_test[0]


@pytest.fixture(scope='session')
def MOCK_X_NBI(build_vectors_for_mock_mct_test):
    return build_vectors_for_mock_mct_test[1]


@pytest.fixture(scope='session')
def MOCK_X_FLT(build_vectors_for_mock_mct_test):
    return build_vectors_for_mock_mct_test[2]


@pytest.fixture(scope='session')
def MOCK_X_STR(build_vectors_for_mock_mct_test):
    return build_vectors_for_mock_mct_test[3]


@pytest.fixture(scope='session')
def MOCK_X_BOOL(build_vectors_for_mock_mct_test):
    return build_vectors_for_mock_mct_test[4]


@pytest.fixture(scope='session')
def _mmct_test_thresh(build_vectors_for_mock_mct_test):
    return build_vectors_for_mock_mct_test[5]


@pytest.fixture(scope='session')
def _source_len(build_vectors_for_mock_mct_test):
    return build_vectors_for_mock_mct_test[6]


@pytest.fixture(scope='session')
def _mmct_test_rows(build_vectors_for_mock_mct_test):
    return build_vectors_for_mock_mct_test[7]










