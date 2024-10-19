# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import numpy.typing as npt
import joblib
from joblib import Parallel, delayed

from typing_extensions import Union, Iterable, Callable

from pybear.utilities._nan_masking import (
    nan_mask_numerical,
    nan_mask_string,
    nan_mask
)


# at the bottom are fixtures used for testing mmct in
# conftest__mmct__test.py



# this is a fixture used to referee against MinCountTransformer in
# MinCountTransformer_test. It also is used to validate that X meets
# certain rigged conditions for testing MinCountTransformer.
# Tests for this fixture are in
# conftest__mmct__test.py. The fixtures used to run the tests in
# conftest__mmct__test.py are at the bottom of this module.


@pytest.fixture(scope='session')
def mmct():

    class _mmct:
        # this must be a class, cannot be a function. need to access
        # get_support_ during MinCountTransformer_test

        def trfm(
            self,
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
            def get_unq(X_COLUMN: np.ndarray) -> np.ndarray:
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
            if ignore_columns is not None and len(ignore_columns) > 0:
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
            UNIQUES_COUNTS_TUPLES = \
                Parallel(return_as='list', n_jobs=-1)(
                    delayed(get_unq)(MOCK_X[:, c_idx]) for c_idx in ACTIVE_C_IDXS
            )

            for col_idx in range(MOCK_X.shape[1]):
                if col_idx not in ACTIVE_C_IDXS:
                    UNIQUES_COUNTS_TUPLES.insert(col_idx, None)

            del ACTIVE_C_IDXS, get_unq

            self.get_support_ = list(map(bool, np.ones(MOCK_X.shape[1]).astype(bool)))

            # GET DTYPES ** ** ** ** ** **
            DTYPES = [None for _ in UNIQUES_COUNTS_TUPLES]
            for col_idx in range(len(UNIQUES_COUNTS_TUPLES)):
                if UNIQUES_COUNTS_TUPLES[col_idx] is None:
                    continue

                UNIQUES_COUNTS_TUPLES[col_idx] = list(UNIQUES_COUNTS_TUPLES[col_idx])
                UNIQUES, COUNTS = UNIQUES_COUNTS_TUPLES[col_idx]

                try:
                    MASK = np.logical_not(nan_mask_numerical(
                        UNIQUES.astype(np.float64))  # <=== if float64 excepts
                    )
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

                if handle_as_bool is not None and col_idx in handle_as_bool:

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

                        NAN_MASK = nan_mask(MOCK_X[:, col_idx])

                        if str(unq).lower()=='nan':
                            ROW_MASK += NAN_MASK
                        else:
                            try:
                                if col_idx in handle_as_bool:
                                    NOT_NAN_MASK = np.logical_not(NAN_MASK)
                                    _ = MOCK_X[NOT_NAN_MASK, col_idx].astype(bool)
                                    ROW_MASK[NOT_NAN_MASK] += (_ == unq)
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
                elif (handle_as_bool is not None and col_idx in handle_as_bool) \
                        and not delete_axis_0:
                    pass
                else:
                    ROW_MASK = np.logical_not(ROW_MASK)
                    MOCK_X = MOCK_X[ROW_MASK, :]
                    if MOCK_Y is not None:
                        MOCK_Y = MOCK_Y[ROW_MASK]

                    del ROW_MASK

                if len([_ for _ in UNIQUES if str(_).lower() != 'nan']) <= 1:
                    self.get_support_[col_idx] = False
                    MOCK_X = np.delete(MOCK_X, col_idx, axis=1)

            if MOCK_Y is not None:
                return MOCK_X, MOCK_Y
            else:
                return MOCK_X

    return _mmct



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










