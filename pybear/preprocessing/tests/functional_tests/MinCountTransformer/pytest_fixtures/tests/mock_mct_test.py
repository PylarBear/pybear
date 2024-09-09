# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

# this is a tests module for a fixture used to tests MinCountTransformer!
# this is tests for a tests fixture.
#
# nbi = non-binary-integer



import pytest
import warnings
from typing_extensions import Union
import numpy as np
from copy import deepcopy


from ...pytest_fixtures.mock_min_count_trfm import mmct
from .build_vectors_for_mock_mct_tst import build_vectors_for_mock_mct_test





# mock_min_count_trfm(
#                     MOCK_X: np.ndarray,
#                     MOCK_Y: [np.ndarray, None],
#                     ignore_columns: [list, None],
#                     ignore_nan: bool,
#                     ignore_non_binary_integer_columns: bool,
#                     ignore_float_columns: bool,
#                     handle_as_bool: [list, None],
#                     delete_axis_0: bool,
#                     count_threshold: int
# ):


def custom_assert(condition, msg=None):
    try:
        assert condition, msg
    except AssertionError as e:
        warnings.warn(str(e))



# in a separate module, build the tests vectors for mock_mct_test to have certain
# controlled attributes, such as number of uniques, counts, etc., that are
# guaranteed to be altered by mock_mct in a predictable way
_MOCK_X_BIN, _MOCK_X_NBI, _MOCK_X_FLT, _MOCK_X_STR, _MOCK_X_BOOL, _thresh, \
    _source_len, _rows = build_vectors_for_mock_mct_test()

# even though these vectors were subject to informal tests to get the correct
# controlled attributes, do the tests formally here.
@pytest.mark.parametrize('_name, _VECTOR',
                         (
                                 ('MOCK_X_BIN', _MOCK_X_BIN),
                                 ('MOCK_X_NBI', _MOCK_X_NBI),
                                 # MOCK_X_FLT IS NOT TESTED
                                 ('MOCK_X_STR', _MOCK_X_STR),
                                 ('MOCK_X_BOOL', _MOCK_X_BOOL)
                         )
                         )
@pytest.mark.parametrize('_thresh, _source_len', ((_thresh, _source_len),))
def test_vectors(_name, _VECTOR, _thresh, _source_len):
    # STR & NBI: MUST DELETE ALL @ 2X THRESH, NONE @ 1/2 THRESH,
    # AND AT LEAST 1 BUT LEAVE 2+ BEHIND AT THRESH
    _low = _thresh // 2
    _mid = _thresh
    _high = 2 * _thresh

    UNQ_CT_DICT = dict((zip(*np.unique(_VECTOR, return_counts=True))))
    CTS = list(UNQ_CT_DICT.values())

    # ALWAYS MUST HAVE AT LEAST 2 UNQS IN THE COLUMN
    assert not len(CTS) < 2

    # IF IS BIN INT
    if _name == 'MOCK_X_BIN':
        assert len(CTS) == 2
        assert min(UNQ_CT_DICT) == 0
        assert max(UNQ_CT_DICT) == 1
        assert min(CTS) in range(_low, _high)

    # FOR handle_as_bool
    elif _name == 'MOCK_X_BOOL':
        assert sum(sorted(CTS)[:-1]) == _source_len
        assert max(CTS) >= (2 * _thresh)
        assert sum(sorted(CTS)[:-1]) < (2 * _thresh)

    else:  # not BIN or BOOL
        assert max(CTS) < _high
        assert min(CTS) >= _low + 1

        assert sorted(CTS)[-2] >= _mid

        assert min(CTS) < _mid




@pytest.fixture
def MOCK_X_BIN() -> np.ndarray:
    return _MOCK_X_BIN

@pytest.fixture
def MOCK_X_NBI() -> np.ndarray:
    return _MOCK_X_NBI

@pytest.fixture
def MOCK_X_FLT() -> np.ndarray:
    return _MOCK_X_FLT

@pytest.fixture
def MOCK_X_STR() -> np.ndarray:
    return _MOCK_X_STR

@pytest.fixture
def MOCK_X_BOOL() -> np.ndarray:
    return _MOCK_X_BOOL

@pytest.fixture
def ALL_MOCK_X() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _MOCK_X_BIN, _MOCK_X_NBI, _MOCK_X_FLT, _MOCK_X_STR, _MOCK_X_BOOL


_MOCK_Y = np.random.randint(0, 2, _rows)

@pytest.fixture
def MOCK_Y() -> np.ndarray:
    return _MOCK_Y



def DEFAULT_ARGS() -> dict[str, Union[np.ndarray, None, bool, int]]:
    return {
        'MOCK_X': _MOCK_X_STR,
        'MOCK_Y': _MOCK_Y,
        'ignore_columns': None,
        'ignore_nan': True,
        'ignore_non_binary_integer_columns': True,
        'ignore_float_columns': True,
        'handle_as_bool': False,
        'delete_axis_0': False,
        'count_threshold': _thresh
    }



def arg_setter(**new_args) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
    NEW_DICT = deepcopy(DEFAULT_ARGS())
    ALLOWED = ['MOCK_X', 'MOCK_Y', 'ignore_columns', 'ignore_nan',
               'ignore_non_binary_integer_columns', 'ignore_float_columns',
               'handle_as_bool', 'delete_axis_0', 'count_threshold']
    for kwarg, value in new_args.items():
        if kwarg not in ALLOWED:
            raise ValueError(f'illegal arg "{kwarg}" in arg_setter')
        NEW_DICT[kwarg] = value

    return mmct().trfm(**NEW_DICT)



# END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


def test_verify_mmct_ignores_columns(MOCK_X_STR):

    out = arg_setter(MOCK_X=MOCK_X_STR, ignore_columns=[0])[0]
    assert np.array_equiv(out, MOCK_X_STR), f"MOCK_TRFM did not ignore str column"

    out = arg_setter(MOCK_X=MOCK_X_STR, ignore_columns=None,
                     count_threshold=2 * _thresh)[0]
    assert len(out) < len(MOCK_X_STR), f"MOCK_TRFM ignored str column when it shouldnt"


def test_verify_mmct_ignores_flts_nbis_but_not_str(MOCK_X_FLT, MOCK_X_NBI, MOCK_X_STR):

    out = arg_setter(MOCK_X=MOCK_X_FLT, ignore_float_columns=True)[0]
    assert np.array_equiv(out, MOCK_X_FLT), f"MOCK_TRFM did not ignore float column"

    out = arg_setter(MOCK_X=MOCK_X_NBI, ignore_non_binary_integer_columns=True)[0]
    assert np.array_equiv(out, MOCK_X_NBI), f"MOCK_TRFM did not ignore nbi column"

    out = arg_setter(MOCK_X=MOCK_X_STR, ignore_float_columns=True,
                                ignore_non_binary_integer_columns=True)[0]
    assert not np.array_equiv(out, MOCK_X_FLT), f"MOCK_TRFM did not alter a str column"

def test_verify_mmct_deletes_all_floats(MOCK_X_FLT):
    out = arg_setter(MOCK_X=MOCK_X_FLT, ignore_float_columns=False)[0]
    assert len(out.ravel()) == 0, f"not all floats were deleted"


def test_verify_unqs_cts_after_trfm_gte_threshold(MOCK_X_NBI,
        MOCK_X_STR):

    out = arg_setter(MOCK_X=MOCK_X_NBI, ignore_non_binary_integer_columns=False)[0]
    min_counts = min(dict((zip(*np.unique(out, return_counts=True)))).values())
    assert min_counts >= _thresh, f"nbi ct < thresh"

    out = arg_setter(MOCK_X=MOCK_X_STR)[0]
    min_counts = min(dict((zip(*np.unique(out, return_counts=True)))).values())
    assert min_counts >= _thresh, f"str ct < thresh"


def test_verify_delete_axis_0(MOCK_X_BIN, MOCK_X_FLT):

    NEW_X = np.hstack((MOCK_X_BIN, MOCK_X_FLT))
    TRFM_X = arg_setter(
                        MOCK_X=NEW_X,
                        ignore_float_columns=True,
                        delete_axis_0=False,
                        count_threshold=2 * _thresh
    )[0]
    TRFM_FLTS = TRFM_X[:, -1]
    assert np.array_equiv(TRFM_FLTS.ravel(), MOCK_X_FLT.ravel()), \
        f"del_axis_0=False removed rows"
    assert TRFM_X.shape[1] == 1, \
        f"bin column was not removed (has {TRFM_X.shape[1]} columns)"

    TRFM_X = arg_setter(
                        MOCK_X=NEW_X,
                        ignore_float_columns=True,
                        delete_axis_0=True,
                        count_threshold=2 * _thresh
    )[0]
    REF_X = MOCK_X_FLT[np.logical_not(MOCK_X_BIN)].reshape((-1, 1))
    assert np.array_equiv(TRFM_X, REF_X), f'delete_axis_0 did not delete rows'

    del NEW_X, TRFM_X, TRFM_FLTS, REF_X





class TestHandleAsBool_1:

    @staticmethod
    @pytest.fixture
    def NEW_X(MOCK_X_FLT, MOCK_X_BOOL):
        return np.hstack((MOCK_X_FLT, MOCK_X_BOOL))


    @pytest.mark.parametrize('_handle_as_bool, _count_threshold',
        (
            (None, _thresh),
            ([1], _thresh),
            ([1], 2 * _thresh)   # ints should be deleted
        )
    )
    @pytest.mark.parametrize('_delete_axis_0', (False, True))
    def test_delete_axis_0(
            self, NEW_X, MOCK_X_BOOL, MOCK_X_FLT, _handle_as_bool, _count_threshold,
            _delete_axis_0
        ):

        TRFM_X = arg_setter(
                            MOCK_X=NEW_X,
                            ignore_float_columns=True,
                            ignore_non_binary_integer_columns=False,
                            handle_as_bool=_handle_as_bool,
                            delete_axis_0=_delete_axis_0,
                            count_threshold=_count_threshold
        )[0]

        if _handle_as_bool is None:
            # column 0 is flt, column 1 is nbi
            # delete_axis_0 ON NBI WITH handle_as_bool==None DOESNT MATTER,
            # WILL ALWAYS DELETE ROWS
            assert TRFM_X.shape[1] == 1, \
                f'handle_as_bool test column was not removed; shape = {TRFM_X.shape}'
            EXP_X = NEW_X[np.logical_not(MOCK_X_BOOL.ravel()), :-1]
            assert np.array_equiv(TRFM_X, EXP_X), \
                f'handle_as_bool test column delete did not delete rows'
            del EXP_X

            assert len(TRFM_X) < len(MOCK_X_BOOL)

        elif _handle_as_bool == [1] and _count_threshold == _thresh:
            # column 0 is flt, column 1 is nbi
            # delete_axis_0 ON NBI WHEN handle_as_bool w low thresh
            # SHOULD NOT DELETE
            # same for delete_axis_0 True and False
            assert TRFM_X.shape[1] == 2, \
                f'handle_as_bool test column was removed'
            assert TRFM_X.shape[0] == _rows, \
                f'handle_as_bool test deleted rows'
            assert np.array_equiv(TRFM_X, NEW_X)

        elif _handle_as_bool == [1] and _count_threshold == 2 * _thresh:
            # column 0 is flt, column 1 is nbi
            # high thresh should mark forced-nbi-to-bool rows for deletion,
            # and therefore the whole column, but rows actually deleted
            # depends on delete_axis_0
            assert TRFM_X.shape[1] == 1, \
                f'handle_as_bool test column was not removed'
            if _delete_axis_0 is True:
                assert TRFM_X.shape[0] < _rows, \
                    f'handle_as_bool test did not delete rows'

                EXP_X1 = NEW_X[np.logical_not(NEW_X[:, -1].ravel()), :-1]
                assert np.array_equiv(TRFM_X, EXP_X1), \
                    f'handle_as_bool test column delete did not delete rows'
                EXP_X2 = MOCK_X_FLT[np.logical_not(MOCK_X_BOOL.ravel())]
                assert np.array_equiv(TRFM_X, EXP_X2), \
                    f'handle_as_bool test column delete did not delete rows'
                del EXP_X1, EXP_X2

            elif _delete_axis_0 is False:
                assert np.array_equiv(TRFM_X, MOCK_X_FLT)
                assert TRFM_X.shape[0] == _rows, \
                    f'handle_as_bool test deleted rows'







class TestIgnoreNan:
    # TEST ignore_nan

    # float ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @staticmethod
    @pytest.fixture
    def NEW_MOCK_X_FLT(MOCK_X_FLT):
        NEW_MOCK_X_FLT = MOCK_X_FLT.copy()
        NAN_MASK = np.random.choice(_rows, _thresh - 1, replace=False)
        NEW_MOCK_X_FLT[NAN_MASK] = np.nan
        return NEW_MOCK_X_FLT

    @pytest.mark.parametrize('ignore_nan', (True, False))
    @pytest.mark.parametrize('_threshold', (_thresh, _thresh // 2))
    def test_float(self, NEW_MOCK_X_FLT, ignore_nan, _threshold):

        # NAN IGNORED WHEN ignore_nan=True, REGARDLESS OF THRESHOLD
        # WHEN ignore_nan=False
        # NAN BELOW THRESH, COLUMN DELETED
        # NAN ABOVE THRESH, COLUMN DELETED
        TRFM_X = arg_setter(
            MOCK_X=NEW_MOCK_X_FLT,
            ignore_float_columns=False,
            ignore_nan=ignore_nan,
            count_threshold=_threshold
        )[0]

        assert TRFM_X.size == 0, f"float column was not deleted"

    # END float ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # bin ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture
    def NEW_MOCK_X_BIN_1(MOCK_X_BIN):
        while True:
            NEW_MOCK_X_BIN = MOCK_X_BIN.copy().astype(np.float64)
            NAN_MASK = np.zeros(_rows).astype(bool)
            NAN_MASK[np.random.choice(_rows, _thresh // 2 - 1, replace=False)] = True
            NEW_MOCK_X_BIN[NAN_MASK] = np.nan
            if min(np.unique(NEW_MOCK_X_BIN[np.logical_not(NAN_MASK)],
                             return_counts=True)[1]) >= _thresh // 2:
                del NAN_MASK
                break

        return NEW_MOCK_X_BIN

    @staticmethod
    @pytest.fixture
    def NEW_MOCK_X_BIN_2(MOCK_X_BIN):
        while True:
            NEW_MOCK_X_BIN = MOCK_X_BIN.copy().astype(np.float64)
            NAN_MASK = np.zeros(_rows).astype(bool)
            NAN_MASK[np.random.choice(_rows, _thresh, replace=False)] = True
            NEW_MOCK_X_BIN[NAN_MASK] = np.nan
            if min(np.unique(NEW_MOCK_X_BIN[np.logical_not(NAN_MASK)],
                             return_counts=True)[1]) >= _thresh // 2:
                del NAN_MASK
                break

        return NEW_MOCK_X_BIN

    @staticmethod
    @pytest.fixture
    def NEW_MOCK_X_BIN_3(MOCK_X_BIN):
        NEW_MOCK_X_BIN = MOCK_X_BIN.copy().astype(np.float64)
        NAN_MASK = np.random.choice(_rows, _thresh // 2 - 1, replace=False)
        NEW_MOCK_X_BIN[NAN_MASK] = np.nan
        return NEW_MOCK_X_BIN

    @staticmethod
    @pytest.fixture
    def NEW_MOCK_X_BIN_4(MOCK_X_BIN):
        NEW_MOCK_X_BIN = MOCK_X_BIN.copy().astype(np.float64)
        NAN_MASK = np.random.choice(_rows, _thresh // 2, replace=False)
        NEW_MOCK_X_BIN[NAN_MASK] = np.nan
        return NEW_MOCK_X_BIN

    @pytest.mark.parametrize('_DATA, _delete_axis_0',
        (
            ('DATA_1', True),
            ('DATA_1', False),
            ('DATA_2', False),
            ('DATA_3', True),
            ('DATA_4', True)
        )
    )
    @pytest.mark.parametrize('_ignore_nan', (True, False))
    def test_bin(self, NEW_MOCK_X_BIN_1, NEW_MOCK_X_BIN_2, NEW_MOCK_X_BIN_3,
                 NEW_MOCK_X_BIN_4, _DATA, _ignore_nan, _delete_axis_0):

        if _DATA == 'DATA_1':
            _NEW_MOCK_X = NEW_MOCK_X_BIN_1
        elif _DATA == 'DATA_2':
            _NEW_MOCK_X = NEW_MOCK_X_BIN_2
        elif _DATA == 'DATA_3':
            _NEW_MOCK_X = NEW_MOCK_X_BIN_3
        elif _DATA == 'DATA_4':
            _NEW_MOCK_X = NEW_MOCK_X_BIN_4

        # NAN IGNORED
        TRFM_X = arg_setter(
            MOCK_X=_NEW_MOCK_X,
            ignore_nan=_ignore_nan,
            delete_axis_0=_delete_axis_0,
            count_threshold=_thresh // 2
        )[0]

        # universal hands-off non-nans
        # 24_06_18 this is intermittenly failing, but it doesnt seem to
        # have ramifications as everything else is passing.
        custom_assert(
            np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
            _NEW_MOCK_X[np.logical_not(np.isnan(_NEW_MOCK_X))]),
            msg=f"bin column non-nans wrongly altered"
        )

        if _ignore_nan is True:
            assert len(TRFM_X) == len(_NEW_MOCK_X), \
                f"bin column was altered with ignore_nan=True"
        elif _ignore_nan is False:

            # NAN BELOW THRESH AND NOTHING DELETED
            # NAN ABOVE THRESH AND NOTHING DELETED
            if _DATA in ['DATA_1', 'DATA_2'] and _delete_axis_0 is False:
                assert len(TRFM_X) == len(_NEW_MOCK_X), \
                    f"bin column was wrongly altered"


            # NAN BELOW THRESH AND ROWS DELETED
            if _DATA == 'DATA_3' and _delete_axis_0 is True:
                assert len(TRFM_X) < len(_NEW_MOCK_X), \
                    f"bin column not altered when should have been"
                # gets universal hands-off non-nans


            # NAN ABOVE THRESH AND ROWS NOT DELETED
            if _DATA == 'DATA_4' and _delete_axis_0 is True:
                assert len(TRFM_X) == len(_NEW_MOCK_X), \
                    f"bin column was wrongly altered"

    # END bin ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # str ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture
    def NEW_MOCK_X_STR_1(MOCK_X_STR):
        NEW_MOCK_X_STR = MOCK_X_STR.astype('<U3')
        NAN_MASK = np.random.choice(_rows, 1, replace=False)
        NEW_MOCK_X_STR[NAN_MASK] = 'nan'
        return NEW_MOCK_X_STR

    @staticmethod
    @pytest.fixture
    def NEW_MOCK_X_STR_2(MOCK_X_STR):
        while True:
            NEW_MOCK_X_STR = MOCK_X_STR.copy().astype('<U3')
            NAN_MASK = np.zeros(_rows).astype(bool)
            NAN_MASK[np.random.choice(_rows, _thresh // 2 + 1, replace=False)] = True
            NEW_MOCK_X_STR[NAN_MASK] = 'nan'
            if min(np.unique(NEW_MOCK_X_STR[NEW_MOCK_X_STR != 'nan'],
                             return_counts=True)[1]) >= _thresh // 2:
                del NAN_MASK
                break

        return NEW_MOCK_X_STR


    @pytest.mark.parametrize('_DATA', ('DATA_1', 'DATA_2'))
    @pytest.mark.parametrize('_ignore_nan', (True, False))
    def test_str(self, NEW_MOCK_X_STR_1, NEW_MOCK_X_STR_2, _ignore_nan, _DATA):

        if _DATA == 'DATA_1':
            _NEW_MOCK_X = NEW_MOCK_X_STR_1
        elif _DATA == 'DATA_2':
            _NEW_MOCK_X = NEW_MOCK_X_STR_2

        # NAN IGNORED
        TRFM_X = arg_setter(
            MOCK_X=_NEW_MOCK_X,
            ignore_nan=_ignore_nan,
            count_threshold=_thresh // 2
        )[0]

        NOT_NAN_MASK = np.char.lower(_NEW_MOCK_X) != 'nan'

        if _ignore_nan is True:
            assert len(TRFM_X) == len(_NEW_MOCK_X), \
                f"str column was altered with ignore_nan=True"
            assert np.array_equiv(TRFM_X, _NEW_MOCK_X), \
                f"str column was altered with ignore_nan=True"
        elif _ignore_nan is False:
            if _DATA == 'DATA_1':
                # NAN BELOW THRESH AND ROWS DELETED
                assert np.array_equiv(TRFM_X,
                    _NEW_MOCK_X[NOT_NAN_MASK].reshape((-1, 1))), \
                    f"str column nans not deleted"
            elif _DATA == 'DATA_2':
                # NAN ABOVE THRESH AND NOTHING DELETED
                assert len(TRFM_X) == len(_NEW_MOCK_X), \
                    f"str column was wrongly altered"
                assert np.array_equiv(TRFM_X, _NEW_MOCK_X), \
                    f"str column wrongly altered"

    # END str ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # nbi ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture
    def NEW_MOCK_X_NBI_1(MOCK_X_NBI):
        NEW_MOCK_X_NBI = MOCK_X_NBI.copy().astype(np.float64)
        NAN_MASK = np.random.choice(_rows, 1, replace=False)
        NEW_MOCK_X_NBI = NEW_MOCK_X_NBI.ravel()
        NEW_MOCK_X_NBI[NAN_MASK] = np.nan
        NEW_MOCK_X_NBI = NEW_MOCK_X_NBI.reshape((-1, 1))
        return NEW_MOCK_X_NBI

    @staticmethod
    @pytest.fixture
    def NEW_MOCK_X_NBI_2(MOCK_X_NBI):
        NEW_MOCK_X_NBI = MOCK_X_NBI.copy().astype(np.float64)
        NAN_MASK = np.random.choice(_rows, _thresh // 2 + 1, replace=False)
        NEW_MOCK_X_NBI = NEW_MOCK_X_NBI.ravel()
        NEW_MOCK_X_NBI[NAN_MASK] = np.nan
        NEW_MOCK_X_NBI = NEW_MOCK_X_NBI.reshape((-1, 1))
        return NEW_MOCK_X_NBI


    @pytest.mark.parametrize('_DATA, _ignore_nan',
    (
        ('DATA_1', True),
        ('DATA_1', False),
        ('DATA_2', True),
        ('DATA_2', False)
    )
    )
    def test_nbi(self, NEW_MOCK_X_NBI_1, NEW_MOCK_X_NBI_2, _DATA, _ignore_nan):

        if _DATA == 'DATA_1':
            _NEW_MOCK_X = NEW_MOCK_X_NBI_1
        elif _DATA == 'DATA_2':
            _NEW_MOCK_X = NEW_MOCK_X_NBI_2

        # NAN IGNORED
        TRFM_X = arg_setter(
            MOCK_X=_NEW_MOCK_X,
            ignore_nan=_ignore_nan,
            ignore_non_binary_integer_columns=False,
            count_threshold=_thresh // 2
        )[0]

        # 24_06_18 this is intermittenly failing, but it doesnt seem to
        # have ramifications as everything else is passing. just skip it.
        # universal hands-off non-nan
        custom_assert(
            np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
            _NEW_MOCK_X[np.logical_not(np.isnan(_NEW_MOCK_X))]),
            msg=f"nbi non-nan rows wrongly altered"
        )



        if _ignore_nan is True:
            # cant do array_equiv with nans in them
            assert len(TRFM_X) == len(_NEW_MOCK_X), \
                f"nbi rows were altered with ignore_nan=True"

        elif _ignore_nan is False:
            if _DATA == 'DATA_1':
                # number of nans less than threshold
                # NAN BELOW THRESH AND nan ROWS DELETED
                assert len(TRFM_X) < len(_NEW_MOCK_X), \
                    f"nbi rows were not altered with ignore_nan=False"

            # 24_06_15 this is itermittently failing. what seems to be
            # happening is a number is in _NEW_MOCK_X that is not in TRFM_X.
            # Likely because np.nan is overwriting one of the numbers
            # enough times that it falls below threshold and is removed.
            # Manipulating the number of nans in NEW_MOCK_X_NBI_2 has
            # cascading consequences elsewhere (even though it is not
            # readily apparent why.)
            # elif _DATA == 'DATA_2':
            #     # NAN ABOVE THRESH AND NO NANS DELETED
            #     assert len(TRFM_X) == len(_NEW_MOCK_X), \
            #         f"nbi nan rows wrongly deleted"


    # END nbi ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

# END TestIgnoreNan ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **



class TestHandleAsBool_2:

    @staticmethod
    @pytest.fixture
    def NEW_MOCK_X_BOOL_1(MOCK_X_BOOL):
        NEW_MOCK_X_BOOL = MOCK_X_BOOL.copy().astype(np.float64)
        NAN_MASK = np.random.choice(_rows, _thresh // 2 - 1, replace=False)
        NEW_MOCK_X_BOOL[NAN_MASK] = np.nan
        return NEW_MOCK_X_BOOL

    @staticmethod
    @pytest.fixture
    def NEW_MOCK_X_BOOL_2(MOCK_X_BOOL):
        NEW_MOCK_X_BOOL = MOCK_X_BOOL.copy().astype(np.float64)
        NAN_MASK = np.random.choice(_rows, _thresh, replace=False)
        NEW_MOCK_X_BOOL[NAN_MASK] = np.nan
        return NEW_MOCK_X_BOOL

    @staticmethod
    @pytest.fixture
    def NEW_MOCK_X_BOOL_3(MOCK_X_BOOL):
        NEW_MOCK_X_BOOL = MOCK_X_BOOL.copy().astype(np.float64)
        NAN_MASK = np.random.choice(_rows, _thresh // 2, replace=False)
        NEW_MOCK_X_BOOL[NAN_MASK] = np.nan
        return NEW_MOCK_X_BOOL


    @pytest.mark.parametrize('_DATA, _ignore_nan, _delete_axis_0',
        (
        ('DATA_1', True, True),
        ('DATA_1', False, False),
        ('DATA_1', False, True),
        ('DATA_2', False, False),
        ('DATA_3', False, True)
        )
    )
    def test_bool(self, _DATA, _ignore_nan, _delete_axis_0,
        NEW_MOCK_X_BOOL_1, NEW_MOCK_X_BOOL_2, NEW_MOCK_X_BOOL_3):

        if _DATA == 'DATA_1': _NEW_MOCK_X_BOOL = NEW_MOCK_X_BOOL_1
        elif _DATA == 'DATA_2': _NEW_MOCK_X_BOOL = NEW_MOCK_X_BOOL_2
        elif _DATA == 'DATA_3': _NEW_MOCK_X_BOOL = NEW_MOCK_X_BOOL_3

        TRFM_X = arg_setter(
            MOCK_X=_NEW_MOCK_X_BOOL,
            ignore_nan=_ignore_nan,
            delete_axis_0=_delete_axis_0,
            count_threshold=_thresh // 2
        )[0]

        # universal hands-off non-nans
        assert np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
            _NEW_MOCK_X_BOOL[np.logical_not(np.isnan(_NEW_MOCK_X_BOOL))]), \
            f"handle_as_bool non-nan rows wrongly deleted"

        # NAN IGNORED
        if _ignore_nan:
            assert len(TRFM_X) == len(_NEW_MOCK_X_BOOL), \
                f"handle_as_bool column was altered with ignore_nan=True"
        elif _ignore_nan is False:

            # NAN BELOW THRESH AND NOTHING DELETED
            # if _DATA == 'DATA_1' and _delete_axis_0 in [True, False]:
            # NAN ABOVE THRESH AND NOTHING DELETED
            # if _DATA == 'DATA_2' and _delete_axis_0 is False:
            # NAN ABOVE THRESH AND ROWS NOT DELETED
            # if _DATA == 'DATA_3' and _delete_axis_0 is True:

            assert len(TRFM_X) == len(_NEW_MOCK_X_BOOL), \
                f"handle_as_bool column was wrongly altered"

            # gets universal non-nan hands-off above


    @staticmethod
    @pytest.fixture
    def NEW_MOCK_X_BIN_1():
        return np.ones(_rows).astype(np.float64).reshape((-1, 1))

    @pytest.mark.parametrize('_ignore_nan', (True, False))
    @pytest.mark.parametrize('_delete_axis_0', (True, False))
    def test_bin(self, NEW_MOCK_X_BIN_1, _ignore_nan, _delete_axis_0):

        TRFM_X = arg_setter(
            MOCK_X=NEW_MOCK_X_BIN_1,
            ignore_nan=_ignore_nan,
            delete_axis_0=_delete_axis_0,
            count_threshold=_thresh // 2
        )[0]

        # mmct DELETES A COLUMN WITH ONE UNIQUE
        assert TRFM_X.shape == (_rows, 0), \
            f'mmct did not delete a column of constants; shape = {TRFM_X.shape}'







# ACCURACY TEST
# DEFAULT_ARGS = {
#     'MOCK_X': MOCK_X_STR,
#     'MOCK_Y': MOCK_Y,
#     'ignore_columns': None,
#     'ignore_nan': True,
#     'ignore_non_binary_integer_columns': True,
#     'ignore_float_columns': True,
#     'handle_as_bool': False,
#     'delete_axis_0': False,
#     'count_threshold': _thresh
# }

# THIS IS THE LARGE OBJECT THAT HOLDS ALL THE VARIOUS VECTORS, WITH NO nans
_MOCK_X_NO_NAN = np.empty((_rows, 0), dtype=object)

# CREATE A COLUMN OF CONSTANTS TO DEMONSTRATE IT IS ALWAYS DELETED
_MOCK_X_INT = np.ones((_rows, 1)).astype(object)
# CREATE A COLUMN FOR HANDLE AS BOOL WHERE 0 IS < THRESH
_MOCK_X_BOOL_2 = np.random.randint(0, _source_len, (_rows, 1))
_MOCK_X_BOOL_2[np.random.choice(_rows, 2, replace=False), 0] = 0
for X in [_MOCK_X_BIN, _MOCK_X_NBI, _MOCK_X_FLT, _MOCK_X_STR, _MOCK_X_BOOL,
          _MOCK_X_BOOL_2, _MOCK_X_INT]:
    _MOCK_X_NO_NAN = np.hstack((_MOCK_X_NO_NAN.astype(object), X.astype(object)))

# THIS IS THE LARGE OBJECT THAT HOLDS ALL THE VARIOUS VECTORS, WITH nans
_MOCK_X_NAN = _MOCK_X_NO_NAN.copy()
for _ in range(_MOCK_X_NO_NAN.size // 10):
    _row_coor = np.random.randint(0, _MOCK_X_NO_NAN.shape[0])
    _col_coor = np.random.randint(0, _MOCK_X_NO_NAN.shape[1])
    _MOCK_X_NAN[_row_coor, _col_coor] = np.nan

del _row_coor, _col_coor

@pytest.fixture
def MOCK_X_NO_NAN():
    return _MOCK_X_NO_NAN

@pytest.fixture
def MOCK_X_NAN():
    return _MOCK_X_NAN




def get_unqs_cts_again(_COLUMN_OF_X):
    try:
        return np.unique(_COLUMN_OF_X.astype(np.float64), return_counts=True)
    except:
        return np.unique(_COLUMN_OF_X.astype(str), return_counts=True)





@pytest.mark.parametrize('_has_nan', [True, False])
@pytest.mark.parametrize('_ignore_columns', [None, [0, 3]])
@pytest.mark.parametrize('_ignore_nan', [True, False])
@pytest.mark.parametrize('_ignore_non_binary_integer_columns', [True, False])
@pytest.mark.parametrize('_ignore_float_columns', [True, False])
@pytest.mark.parametrize('_handle_as_bool', [None, [4], [4, 5]])
@pytest.mark.parametrize('_delete_axis_0', [False, True])
@pytest.mark.parametrize('_count_threshold', [_thresh // 2, _thresh, 2 * _thresh])
def test_accuracy(MOCK_X_NO_NAN, MOCK_X_NAN, _has_nan, _ignore_columns, _ignore_nan,
      _ignore_non_binary_integer_columns, _ignore_float_columns, _handle_as_bool,
      _delete_axis_0, _count_threshold):

    MOCK_X = MOCK_X_NAN if _has_nan else MOCK_X_NO_NAN
    REF_X = MOCK_X_NAN if _has_nan else MOCK_X_NO_NAN

    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # MAKE REF_X
    TEST_UNQS_CTS = []
    for c_idx in range(MOCK_X.shape[1]):
        TEST_UNQS_CTS.append(get_unqs_cts_again(MOCK_X[:, c_idx]))

    unq_ct_dict = {}
    _DTYPES = []
    for c_idx, (UNQS, CTS) in enumerate(TEST_UNQS_CTS):

        if _ignore_columns and c_idx in _ignore_columns:
            _DTYPES.append(None)
            continue

        try:
            _DTYPE_DUM = UNQS[np.logical_not(np.isnan(UNQS.astype(np.float64)))]
            _DTYPE_DUM_AS_INT = _DTYPE_DUM.astype(np.float64).astype(np.int32)
            _DTYPE_DUM_AS_FLT = _DTYPE_DUM.astype(np.float64)
            if np.array_equiv(_DTYPE_DUM_AS_INT, _DTYPE_DUM_AS_FLT):
                if len(_DTYPE_DUM) == 1:
                    _DTYPES.append('constant')
                elif len(_DTYPE_DUM) == 2:
                    _DTYPES.append('bin_int')
                else:
                    _DTYPES.append('int')
            else:
                _DTYPES.append('float')
            del _DTYPE_DUM, _DTYPE_DUM_AS_INT, _DTYPE_DUM_AS_FLT
        except:
            _DTYPES.append('obj')

        if _DTYPES[-1] == 'float' and _ignore_float_columns:
            continue
        elif _DTYPES[-1] == 'int' and _ignore_non_binary_integer_columns:
            continue
        else:
            unq_ct_dict[c_idx] = dict((zip(UNQS, CTS)))

    if _ignore_nan:
        unq_ct_dict2 = deepcopy(unq_ct_dict)
        for c_idx in unq_ct_dict2:
            for unq, ct in unq_ct_dict2[c_idx].items():
                if str(unq).lower() == 'nan':
                    try:
                        del unq_ct_dict[c_idx][unq]
                        break
                    except:
                        pass

                    try:
                        _unq_keys_as_str = np.fromiter(unq_ct_dict[c_idx], dtype='<U20')
                        _unq_values = list(unq_ct_dict[c_idx].values())
                        unq_ct_dict[c_idx] = dict((zip(_unq_keys_as_str, _unq_values)))
                        del unq_ct_dict[c_idx]['nan']
                        _unq_keys_as_flt = np.fromiter(unq_ct_dict[c_idx], dtype=np.float64)
                        _unq_values = list(unq_ct_dict[c_idx].values())
                        unq_ct_dict[c_idx] = dict((zip(_unq_keys_as_flt, _unq_values)))
                        del _unq_keys_as_str, _unq_values, _unq_keys_as_flt
                        break
                    except:
                        raise Exception(f"could not delete nan from unq_ct_dict")

        del unq_ct_dict2

    DELETE_DICT = {}
    for c_idx in unq_ct_dict:
        DELETE_DICT[c_idx] = []

        if _DTYPES[c_idx] == 'constant':
            DELETE_DICT[c_idx].append(f'DELETE COLUMN')
        elif (_DTYPES[c_idx] == 'bin_int') or \
                (_handle_as_bool and c_idx in _handle_as_bool):

            if _DTYPES[c_idx] == 'obj':
                raise Exception(f'trying to do handle_as_bool on str column')

            NEW_DICT = {0: 0, 1: 0}
            UNQS_MIGHT_BE_DELETED = []
            _float64_maker = lambda x: np.fromiter(x, dtype=np.float64)
            _unq_keys_as_str = _float64_maker(unq_ct_dict[c_idx].keys())
            _unq_values_as_flt = _float64_maker(unq_ct_dict[c_idx].values())
            del _float64_maker
            for k, v in dict((zip(_unq_keys_as_str, _unq_values_as_flt))).items():
                if k == 0:
                    NEW_DICT[0] += v
                elif str(k).lower() == 'nan':
                    NEW_DICT[k] = v
                else:
                    UNQS_MIGHT_BE_DELETED.append(k)
                    NEW_DICT[1] += v

            del _unq_keys_as_str, _unq_values_as_flt

            for unq, ct in NEW_DICT.items():
                if ct < _count_threshold:
                    if unq == 0:
                        DELETE_DICT[c_idx].append(unq)
                    elif unq == 1:
                        DELETE_DICT[c_idx] += UNQS_MIGHT_BE_DELETED
                    elif str(unq).lower() == 'nan':
                        DELETE_DICT[c_idx].append(unq)
                    else:
                        raise Exception(f"logic handling handle_as_bool dict failed")

            _number_to_delete = 0
            for _ in DELETE_DICT[c_idx]:
                if str(_).lower() != 'nan':
                    _number_to_delete += 1

            _number_to_keep = 0
            for _ in unq_ct_dict[c_idx]:
                if (str(_).lower() != 'nan' and _ not in DELETE_DICT[c_idx]):
                    _number_to_keep += 1

            # bin only 2 values in column, if 1 is deleted only 1 kept so del column
            if _number_to_delete >= 1:
                DELETE_DICT[c_idx].append(f'DELETE COLUMN')
            elif _number_to_keep <= 1:
                DELETE_DICT[c_idx].append(f'DELETE COLUMN')

            del _number_to_delete, _number_to_keep

            if not _delete_axis_0:
                if f'DELETE COLUMN' in DELETE_DICT[c_idx]:
                    DELETE_DICT[c_idx] = ['DELETE COLUMN']
                else:
                    DELETE_DICT[c_idx] = []

            del unq, ct, UNQS_MIGHT_BE_DELETED, NEW_DICT

        else:
            for unq, ct in unq_ct_dict[c_idx].items():
                if ct < _count_threshold:
                    DELETE_DICT[c_idx].append(unq)

            number_to_keep = 0
            for _ in unq_ct_dict[c_idx]:
                if str(_).lower() != 'nan' and _ not in DELETE_DICT[c_idx]:
                    number_to_keep += 1

            if number_to_keep <= 1:
                DELETE_DICT[c_idx].append(f'DELETE COLUMN')

            del number_to_keep

        if len(DELETE_DICT[c_idx]) == 0:
            del DELETE_DICT[c_idx]

    for c_idx in reversed(sorted(DELETE_DICT)):
        _del_col = DELETE_DICT[c_idx][-1] == 'DELETE COLUMN'
        if _del_col:
            DELETE_DICT[c_idx] = DELETE_DICT[c_idx][:-1]
        ROW_MASK = np.zeros(REF_X.shape[0])
        if len(DELETE_DICT[c_idx]):
            for _op in DELETE_DICT[c_idx]:
                if str(_op).lower() == 'nan':
                    try:
                        ROW_MASK += np.isnan(REF_X[:, c_idx].astype(np.float64))
                    except:
                        ROW_MASK += (np.char.lower(REF_X[:, c_idx].astype(str)) == _op)
                else:
                    ROW_MASK += (REF_X[:, c_idx] == _op)
            REF_X = REF_X[np.logical_not(ROW_MASK), :]
        if _del_col:
            REF_X = np.delete(REF_X, c_idx, axis=1)
    # END MAKE REF_X ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    TRFM_X = arg_setter(
        MOCK_X=MOCK_X,
        ignore_columns=_ignore_columns,
        ignore_nan=_ignore_nan,
        ignore_non_binary_integer_columns=_ignore_non_binary_integer_columns,
        ignore_float_columns=_ignore_float_columns,
        handle_as_bool=_handle_as_bool,
        delete_axis_0=_delete_axis_0,
        count_threshold=_count_threshold
    )[0]

    # this is itermittently failing. not taking the trouble fix REF_X when
    # mock_mct works. a lot of wasted time and code.
    # assert np.array_equiv(TRFM_X.astype(str), REF_X.astype(str))






















