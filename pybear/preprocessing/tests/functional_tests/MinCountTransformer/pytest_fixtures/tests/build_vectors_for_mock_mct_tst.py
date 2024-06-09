# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np



# MOCK_X_BIN ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
def build_MOCK_X_BIN(_rows: int, _source_len: int) -> np.ndarray:

    """
    build MOCK_X_BIN for use below in build_vectors_for_mock_mct_test()
    """

    _p = [1 - 1 / _source_len, 1 / _source_len]
    MOCK_X_BIN = np.random.choice([0, 1], _rows, p=_p).reshape((-1, 1))
    del _p

    return MOCK_X_BIN
# END MOCK_X_BIN ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

# MOCK_X_NBI ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# non-binary integer
def build_MOCK_X_NBI(_rows: int, _source_len: int) -> np.ndarray:

    """
    build MOCK_X_NBI for use below in build_vectors_for_mock_mct_test()
    """

    MOCK_X_NBI = np.random.randint(0, _source_len, (_rows,)).reshape((-1, 1))

    return MOCK_X_NBI
# END MOCK_X_NBI ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# MOCK_X_FLT ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
def build_MOCK_X_FLT(_rows: int) -> np.ndarray:

    """
    build MOCK_X_FLT for use below in build_vectors_for_mock_mct_test()
    """

    MOCK_X_FLT = np.random.uniform(0, 1, (_rows,)).reshape((-1, 1))

    return MOCK_X_FLT
# END MOCK_X_FLT ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# MOCK_X_STR ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
def build_MOCK_X_STR(_rows:int, _source_len:int) -> np.ndarray:

    """
    build MOCK_X_STR for use below in build_vectors_for_mock_mct_test()
    """

    _alpha = 'qwertyuiopasdfghjklzxcvbnm'
    _alpha += _alpha.upper()
    _STRS = list(_alpha[:_source_len])
    del _alpha

    MOCK_X_STR = np.random.choice(_STRS, _rows, replace=True).reshape((-1, 1))

    return MOCK_X_STR
# END MOCK_X_STR ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


# MOCK_X_BOOL ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
def build_MOCK_X_BOOL(_rows:int, _source_len:int) -> np.ndarray:

    """
    build MOCK_X_BOOL for use below in build_vectors_for_mock_mct_test()
    """

    MOCK_X_BOOL = np.zeros(_rows)
    _idx = np.random.choice(range(_rows), _source_len, replace=False)
    MOCK_X_BOOL[_idx] = np.arange(1, _source_len + 1)
    del _idx
    MOCK_X_BOOL = MOCK_X_BOOL.reshape((-1, 1))

    return MOCK_X_BOOL
# END MOCK_X_BOOL ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **



def trfm_tester(VECTOR: np.ndarray, _thresh: int, _source_len: int) -> bool:

    """
    For use below in build_vectors_for_mock_mct_test()
    """

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


def build_vectors_for_mock_mct_test() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int
    ]:

    _quit = False
    for _rows in range(10, 101):
        for _thresh in range(6, _rows):
            for _source_len in range(_thresh, _rows):

                # _thresh must be less than _rows
                # _source_len must be less than _thresh

                MOCK_X_BIN = build_MOCK_X_BIN(_rows, _source_len)
                MOCK_X_NBI = build_MOCK_X_NBI(_rows, _source_len)
                MOCK_X_FLT = build_MOCK_X_FLT(_rows)
                MOCK_X_STR = build_MOCK_X_STR(_rows, _source_len)
                MOCK_X_BOOL = build_MOCK_X_BOOL(_rows, _source_len)

                _good = 0
                _good += trfm_tester(MOCK_X_BIN, _thresh, _source_len)
                _good += trfm_tester(MOCK_X_NBI, _thresh, _source_len)
                _good += trfm_tester(MOCK_X_STR, _thresh, _source_len)
                _good += trfm_tester(MOCK_X_BOOL, _thresh, _source_len)
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


    return MOCK_X_BIN, MOCK_X_NBI, MOCK_X_FLT, MOCK_X_STR, MOCK_X_BOOL, \
        _thresh, _source_len, _rows




















