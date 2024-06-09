import numpy as np
import pandas as pd
import joblib
from copy import deepcopy
import time


class mmct:

    def __init__(self):
        pass


    def trfm(
            self,
            MOCK_X: np.ndarray,
            MOCK_Y: [np.ndarray, None],
            ignore_columns: [list, None],
            ignore_nan: bool,
            ignore_non_binary_integer_columns: bool,
            ignore_float_columns: bool,
            handle_as_bool: [list, None],
            delete_axis_0: bool,
            count_threshold: int
        ):

        # GET UNIQUES:
        @joblib.wrap_non_picklable_objects
        def get_uniques(X_COLUMN):
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
                    raise ValueError(f'ignore_columns column index {i} out of bounds for data with {MOCK_X.shape[1]} columns')
                except Exception as e1:
                    raise Exception(f'ignore_columns remove from ACTIVE_C_IDXS except for reason other than ValueError --- {e1}')

        # DONT HARD-CODE backend, ALLOW A CONTEXT MANAGER TO SET
        UNIQUES_COUNTS_TUPLES = joblib.Parallel(return_as='list', n_jobs=-1)(
            joblib.delayed(get_uniques)(MOCK_X[:, c_idx]) for c_idx in ACTIVE_C_IDXS)

        for col_idx in range(MOCK_X.shape[1]):
            if col_idx not in ACTIVE_C_IDXS:
                UNIQUES_COUNTS_TUPLES.insert(col_idx, None)

        del ACTIVE_C_IDXS

        self.get_support_ = np.ones(MOCK_X.shape[1]).astype(bool)

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
                if np.array_equiv(NO_NAN_UNIQUES.astype(np.float64).astype(np.int32),
                                  NO_NAN_UNIQUES.astype(np.float64)):
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

                del MASK, NO_NAN_UNIQUES

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
                    raise ValueError(f"MOCK X trying to do handle_as_bool on str column")

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
                        NAN_MASK = np.isnan(MOCK_X[:, col_idx].astype(np.float64))
                    except:
                        NAN_MASK = (np.char.lower(MOCK_X[:, col_idx].astype(str)) == 'nan')

                    if str(unq).lower()=='nan':
                        ROW_MASK += NAN_MASK
                    else:
                        try:
                            if col_idx in handle_as_bool:
                                NOT_NAN_MASK = np.logical_not(NAN_MASK)
                                ROW_MASK[NOT_NAN_MASK] += (MOCK_X[NOT_NAN_MASK, col_idx].astype(bool) == unq)
                                del NOT_NAN_MASK
                            else:
                                raise Exception  # JUST TO SEND INTO not handle_as_bool CODE
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
            elif (handle_as_bool and col_idx in handle_as_bool) and not delete_axis_0:
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

# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *





if __name__ == '__main__':

    _alpha = 'qwertyuiopasdfghjklzxcvbnm'
    _alpha += _alpha.upper()

    def build_viable_test_objects(_rows, _thresh, _source_len):

        if _thresh >= _rows:
            raise ValueError(f"_thresh cannot be >= _rows")

        MOCK_X_BIN = np.random.choice([0,1],_rows,p=[1-1/_source_len, 1/_source_len]).reshape((-1,1))
        MOCK_X_NBI = np.random.randint(0,_source_len,(_rows,)).reshape((-1,1))
        MOCK_X_FLT = np.random.uniform(0,1,(_rows,)).reshape((-1,1))
        MOCK_X_STR = np.random.choice(list(_alpha[:_source_len]), _rows, replace=True).reshape((-1,1))
        MOCK_X_BOOL = np.zeros(_rows)
        MOCK_X_BOOL[np.random.choice(range(_rows), _source_len, replace=False)] = np.arange(1, _source_len+1)
        MOCK_X_BOOL = MOCK_X_BOOL.reshape((-1, 1))

        return MOCK_X_BIN, MOCK_X_NBI, MOCK_X_FLT, MOCK_X_STR, MOCK_X_BOOL


    def trfm_tester(VECTOR, _thresh):
        # STR & NBI: MUST DELETE ALL @ 2X THRESH, NONE @ 1/2 THRESH,
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
        if len(CTS) == 2 and min(UNQ_CT_DICT)==0 and max(UNQ_CT_DICT)==1:
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

        if not max(CTS) < _high:
            return False

        if min(CTS) < _low + 1:
            return False

        if not sorted(CTS)[-2] >= _mid:
            return False

        if not min(CTS) < _mid:
            return False

        return True


    _quit = False
    for _rows in range(10, 101):
        for _thresh in range(6,_rows):
            for _source_len in range(_thresh, _rows):

                MOCK_X_BIN, MOCK_X_NBI, MOCK_X_FLT, MOCK_X_STR, MOCK_X_BOOL = \
                    build_viable_test_objects(_rows, _thresh, _source_len)

                _good = 0
                _good += trfm_tester(MOCK_X_BIN, _thresh)
                _good += trfm_tester(MOCK_X_NBI, _thresh)
                _good += trfm_tester(MOCK_X_STR, _thresh)
                _good += trfm_tester(MOCK_X_BOOL, _thresh)
                if _good == 4:
                    print(f'rows = {_rows}, threshold = {_thresh}')
                    _quit = True

                if _quit:
                    break
            if _quit:
                break
        if _quit:
            break
    else:
        raise ValueError(f'Unable to find suitable test vectors')


    MOCK_Y = np.random.randint(0, 2, _rows)


    print(f'MOCK_Y = {MOCK_Y}')
    print(f'MOCK_X_BIN = {MOCK_X_BIN.ravel()}')
    print(f'MOCK_X_NBI = {MOCK_X_NBI.ravel()}')
    print(f'MOCK_X_FLT = {MOCK_X_FLT.ravel()}')
    print(f'MOCK_X_STR = {MOCK_X_STR.ravel()}')
    print(f'MOCK_X_BOOL = {MOCK_X_BOOL.ravel()}')





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

    DEFAULT_ARGS = {
                    'MOCK_X': MOCK_X_STR,
                    'MOCK_Y': MOCK_Y,
                    'ignore_columns': None,
                    'ignore_nan': True,
                    'ignore_non_binary_integer_columns': True,
                    'ignore_float_columns': True,
                    'handle_as_bool': False,
                    'delete_axis_0': False,
                    'count_threshold': _thresh
    }


    def arg_setter(**new_args):

        NEW_DICT = deepcopy(DEFAULT_ARGS)
        ALLOWED = ['MOCK_X', 'MOCK_Y', 'ignore_columns', 'ignore_nan',
                   'ignore_non_binary_integer_columns', 'ignore_float_columns',
                   'handle_as_bool', 'delete_axis_0', 'count_threshold']
        for kwarg, value in new_args.items():
            if kwarg not in ALLOWED:
                raise ValueError(f'illegal arg "{kwarg}" in arg_setter')
            NEW_DICT[kwarg] = value

        return mmct().trfm(**NEW_DICT)



    # VERIFY ignore_columns WORKS
    assert np.array_equiv(arg_setter(MOCK_X=MOCK_X_STR, ignore_columns=[0])[0], MOCK_X_STR), \
        f"MOCK_TRFM did not ignore str column"
    assert len(arg_setter(MOCK_X=MOCK_X_STR, ignore_columns=None, count_threshold=2*_thresh)[0]) < len(MOCK_X_STR), \
        f"MOCK_TRFM ignored str column when it shouldnt"

    # MOCK_TRFM IGNORES FLTS AND N-B-INTS BUT NOT STR
    assert np.array_equiv(arg_setter(MOCK_X=MOCK_X_FLT,
                         ignore_float_columns=True)[0], MOCK_X_FLT), \
        f"MOCK_TRFM did not ignore float column"
    assert np.array_equiv(arg_setter(MOCK_X=MOCK_X_NBI,
                         ignore_non_binary_integer_columns=True)[0], MOCK_X_NBI), \
        f"MOCK_TRFM did not ignore nbi column"
    assert not np.array_equiv(arg_setter(MOCK_X=MOCK_X_STR, ignore_float_columns=True,
                         ignore_non_binary_integer_columns=True)[0], MOCK_X_FLT), \
        f"MOCK_TRFM did not alter a str column"

    # VERIFY ALL FLOATS ARE DELETED
    assert len(arg_setter(MOCK_X=MOCK_X_FLT, ignore_float_columns=False)[0].ravel()) == 0, \
        f"not all floats were deleted"

    # VERIFY UNQ/CTS AFTER TRFM ARE >= threshold
    assert min(dict((zip(*np.unique(arg_setter(MOCK_X=MOCK_X_NBI,
               ignore_non_binary_integer_columns=False)[0], return_counts=True)))).values()) >= _thresh, \
        f"nbi ct < thresh"
    assert min(dict((zip(*np.unique(arg_setter(MOCK_X=MOCK_X_STR)[0], return_counts=True)))).values()) >= _thresh, \
        f"str ct < thresh"

    # VERIFY delete_axis_0
    NEW_X = np.hstack((MOCK_X_BIN, MOCK_X_FLT))
    TRFM_X = arg_setter(MOCK_X=NEW_X, ignore_float_columns=True, delete_axis_0=False, count_threshold=2*_thresh)[0]
    TRFM_FLTS = TRFM_X[:, -1]
    assert np.array_equiv(TRFM_FLTS.ravel(), MOCK_X_FLT.ravel()), f"del_axis_0=False removed rows"
    assert TRFM_X.shape[1] == 1, f"bin column was not removed (has {TRFM_X.shape[1]} columns)"
    TRFM_X = arg_setter(MOCK_X=NEW_X, ignore_float_columns=True, delete_axis_0=True, count_threshold=2*_thresh)[0]
    REF_X = MOCK_X_FLT[np.logical_not(MOCK_X_BIN)].reshape((-1,1))
    assert np.array_equiv(TRFM_X, REF_X), f'delete_axis_0 did not delete rows'
    del NEW_X, TRFM_X, TRFM_FLTS, REF_X

    # VERIFY handle_as_bool
    NEW_X = np.hstack((MOCK_X_FLT, MOCK_X_BOOL))
    # delete_axis_0 ON NBI WITH handle_as_bool==None DOESNT MATTER, WILL ALWAYS DELETE ROWS
    TRFM_X = arg_setter(MOCK_X=NEW_X, ignore_float_columns=True, ignore_non_binary_integer_columns=False,
                        handle_as_bool=None, delete_axis_0=False, count_threshold=_thresh)[0]
    assert TRFM_X.shape[1] == 1, f'handle_as_bool test column was not removed; shape = {TRFM_X.shape}'
    assert np.array_equiv(TRFM_X, NEW_X[np.logical_not(MOCK_X_BOOL).ravel(), :-1]), f'handle_as_bool test column delete did not delete rows'
    TRFM_X = arg_setter(MOCK_X=NEW_X, ignore_float_columns=True, ignore_non_binary_integer_columns=False,
                        handle_as_bool=None, delete_axis_0=True, count_threshold=_thresh)[0]
    assert TRFM_X.shape[1] == 1, f'handle_as_bool test column was not removed'
    assert np.array_equiv(TRFM_X, NEW_X[np.logical_not(MOCK_X_BOOL).ravel(), :-1]), f'handle_as_bool test column delete did not delete rows'
    TRFM_X = arg_setter(MOCK_X=NEW_X, ignore_float_columns=True, ignore_non_binary_integer_columns=False,
                        handle_as_bool=[1], delete_axis_0=False, count_threshold=_thresh)[0]
    assert TRFM_X.shape[1] == 2, f'handle_as_bool test column was removed'
    assert TRFM_X.shape[0] == _rows, f'handle_as_bool test deleted rows'
    TRFM_X = arg_setter(MOCK_X=NEW_X, ignore_float_columns=True, ignore_non_binary_integer_columns=False,
                        handle_as_bool=[1], delete_axis_0=True, count_threshold=_thresh)[0]
    assert TRFM_X.shape[1] == 2, f'handle_as_bool test column was removed'
    assert TRFM_X.shape[0] == _rows, f'handle_as_bool test deleted rows'
    TRFM_X = arg_setter(MOCK_X=NEW_X, ignore_float_columns=True, ignore_non_binary_integer_columns=False,
                        handle_as_bool=[1], delete_axis_0=False, count_threshold=2*_thresh)[0]
    assert TRFM_X.shape[1] == 1, f'handle_as_bool test column was not removed'
    assert TRFM_X.shape[0] == _rows, f'handle_as_bool test deleted rows'
    TRFM_X = arg_setter(MOCK_X=NEW_X, ignore_float_columns=True, ignore_non_binary_integer_columns=False,
                        handle_as_bool=[1], delete_axis_0=True, count_threshold=2*_thresh)[0]
    assert TRFM_X.shape[1] == 1, f'handle_as_bool test column was not removed'
    assert np.array_equiv(TRFM_X, NEW_X[np.logical_not(MOCK_X_BOOL).ravel(), :-1]), \
                            f'handle_as_bool test column delete did not delete rows'


    # TEST ignore_nan
    # FLT
    NEW_MOCK_X_FLT = MOCK_X_FLT.copy()
    NAN_MASK = np.random.choice(_rows, _thresh-1, replace=False)
    NEW_MOCK_X_FLT[NAN_MASK] = np.nan
    # NAN IGNORED
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_FLT, ignore_float_columns=False, ignore_nan=True)[0]
    assert TRFM_X.size == 0, f"float column was not deleted"
    # NAN BELOW THRESH AND COLUMN DELETED
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_FLT, ignore_float_columns=False, ignore_nan=False)[0]
    assert TRFM_X.size == 0, f"float column was not deleted"
    # NAN ABOVE THRESH AND COLUMN DELETED
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_FLT, ignore_float_columns=False, ignore_nan=False, count_threshold=_thresh//2)[0]
    assert TRFM_X.size == 0, f"float column was not deleted"

    # BIN
    while True:
        NEW_MOCK_X_BIN = MOCK_X_BIN.copy().astype(np.float64)
        NAN_MASK = np.zeros(_rows).astype(bool)
        NAN_MASK[np.random.choice(_rows, _thresh//2-1, replace=False)] = True
        NEW_MOCK_X_BIN[NAN_MASK] = np.nan
        if min(np.unique(NEW_MOCK_X_BIN[np.logical_not(NAN_MASK)], return_counts=True)[1]) >= _thresh//2:
            del NAN_MASK
            break
    # NAN IGNORED
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_BIN, ignore_nan=True, delete_axis_0=True, count_threshold=_thresh//2)[0]
    assert len(TRFM_X) == len(NEW_MOCK_X_BIN), f"bin column was altered with ignore_nan=True"
    assert np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
                NEW_MOCK_X_BIN[np.logical_not(np.isnan(NEW_MOCK_X_BIN))]), \
                f"bin column was altered with ignore_nan=True"
    # NAN BELOW THRESH AND NOTHING DELETED
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_BIN, ignore_nan=False, delete_axis_0=False, count_threshold=_thresh//2)[0]
    assert np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
                NEW_MOCK_X_BIN[np.logical_not(np.isnan(NEW_MOCK_X_BIN))]), f"bin column wrongly altered"
    # NAN ABOVE THRESH AND NOTHING DELETED
    while True:
        NEW_MOCK_X_BIN = MOCK_X_BIN.copy().astype(np.float64)
        NAN_MASK = np.zeros(_rows).astype(bool)
        NAN_MASK[np.random.choice(_rows, _thresh, replace=False)] = True
        NEW_MOCK_X_BIN[NAN_MASK] = np.nan
        if min(np.unique(NEW_MOCK_X_BIN[np.logical_not(NAN_MASK)], return_counts=True)[1]) >= _thresh//2:
            del NAN_MASK
            break

    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_BIN, ignore_nan=False, delete_axis_0=False, count_threshold=_thresh//2)[0]
    assert len(TRFM_X) == len(NEW_MOCK_X_BIN), f"bin column was wrongly altered"
    assert np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
                NEW_MOCK_X_BIN[np.logical_not(np.isnan(NEW_MOCK_X_BIN))]), f"bin column wrongly altered"

    NEW_MOCK_X_BIN = MOCK_X_BIN.copy().astype(np.float64)
    NAN_MASK = np.random.choice(_rows, _thresh//2-1, replace=False)
    NEW_MOCK_X_BIN[NAN_MASK] = np.nan
    # NAN BELOW THRESH AND ROWS DELETED
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_BIN, ignore_nan=False, delete_axis_0=True, count_threshold=_thresh//2)[0]
    assert np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
                NEW_MOCK_X_BIN[np.logical_not(np.isnan(NEW_MOCK_X_BIN))]), \
                    f"incorrect bin nan rows deleted"
    # NAN ABOVE THRESH AND ROWS NOT DELETED
    NEW_MOCK_X_BIN = MOCK_X_BIN.copy().astype(np.float64)
    NAN_MASK = np.random.choice(_rows, _thresh//2, replace=False)
    NEW_MOCK_X_BIN[NAN_MASK] = np.nan
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_BIN, ignore_nan=False, delete_axis_0=True, count_threshold=_thresh//2)[0]
    assert len(TRFM_X) == len(NEW_MOCK_X_BIN), f"bin column was altered"
    assert np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
                NEW_MOCK_X_BIN[np.logical_not(np.isnan(NEW_MOCK_X_BIN))]), f"bin column was altered"

    # STR
    NEW_MOCK_X_STR = MOCK_X_STR.copy().astype('<U3')
    NAN_MASK = np.random.choice(_rows, 1, replace=False)
    NEW_MOCK_X_STR[NAN_MASK] = 'nan'
    NOT_NAN_MASK = np.char.lower(NEW_MOCK_X_STR) != 'nan'
    # NAN IGNORED
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_STR, ignore_nan=True, count_threshold=_thresh//2)[0]
    assert len(TRFM_X) == len(NEW_MOCK_X_STR), f"str column was altered with ignore_nan=True"
    assert np.array_equiv(TRFM_X, NEW_MOCK_X_STR), \
                f"str column was altered with ignore_nan=True"
    # NAN BELOW THRESH AND ROWS DELETED
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_STR, ignore_nan=False, count_threshold=_thresh//2)[0]
    assert np.array_equiv(TRFM_X, NEW_MOCK_X_STR[NOT_NAN_MASK].reshape((-1,1))), f"str column nans not deleted"
    # NAN ABOVE THRESH AND NOTHING DELETED
    while True:
        NEW_MOCK_X_STR = MOCK_X_STR.copy().astype('<U3')
        NAN_MASK = np.zeros(_rows).astype(bool)
        NAN_MASK[np.random.choice(_rows, _thresh//2+1, replace=False)] = True
        NEW_MOCK_X_STR[NAN_MASK] = 'nan'
        if min(np.unique(NEW_MOCK_X_STR[NEW_MOCK_X_STR!='nan'], return_counts=True)[1]) >= _thresh//2:
            del NAN_MASK
            break
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_STR, ignore_nan=False, count_threshold=_thresh//2)[0]
    assert len(TRFM_X) == len(NEW_MOCK_X_STR), f"str column was wrongly altered"
    assert np.array_equiv(TRFM_X, NEW_MOCK_X_STR), f"str column wrongly altered"

    # NBI
    NEW_MOCK_X_NBI = MOCK_X_NBI.copy().astype(np.float64)
    NAN_MASK = np.random.choice(_rows, 1, replace=False)
    NEW_MOCK_X_NBI[NAN_MASK] = np.nan
    # NAN IGNORED
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_NBI, ignore_nan=True, count_threshold=_thresh//2)[0]
    assert len(TRFM_X) == len(NEW_MOCK_X_NBI), f"nbi column was altered with ignore_nan=True"
    assert np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
                          NEW_MOCK_X_NBI[np.logical_not(np.isnan(NEW_MOCK_X_NBI))]), \
                                    f"nbi column was altered with ignore_nan=True"
    # NAN BELOW THRESH AND ROWS DELETED
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_NBI, ignore_nan=False, count_threshold=_thresh//2)[0]
    assert np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
                          NEW_MOCK_X_NBI[np.logical_not(np.isnan(NEW_MOCK_X_NBI))]), \
                          f"nbi column nans not deleted"
    # NAN ABOVE THRESH AND NOTHING DELETED
    NEW_MOCK_X_NBI = MOCK_X_NBI.copy().astype(np.float64)
    NAN_MASK = np.random.choice(_rows, _thresh//2+1, replace=False)
    NEW_MOCK_X_NBI[NAN_MASK] = np.nan
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_NBI, ignore_nan=False, count_threshold=_thresh//2)[0]
    assert len(TRFM_X) == len(NEW_MOCK_X_NBI), f"nbi column was wrongly altered"
    assert np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
          NEW_MOCK_X_NBI[np.logical_not(np.isnan(NEW_MOCK_X_NBI))]), f"nbi column wrongly altered"


    # HANDLE AS BOOL
    NEW_MOCK_X_BOOL = MOCK_X_BOOL.copy().astype(np.float64)
    NAN_MASK = np.random.choice(_rows, _thresh//2-1, replace=False)
    NEW_MOCK_X_BOOL[NAN_MASK] = np.nan
    # NAN IGNORED
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_BOOL, ignore_nan=True, delete_axis_0=True, count_threshold=_thresh//2)[0]
    assert len(TRFM_X) == len(NEW_MOCK_X_BOOL), f"handle_as_bool column was altered with ignore_nan=True"
    assert np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
                NEW_MOCK_X_BOOL[np.logical_not(np.isnan(NEW_MOCK_X_BOOL))]), \
                f"handle_as_bool column was altered with ignore_nan=True"
    # NAN BELOW THRESH AND NOTHING DELETED
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_BOOL, ignore_nan=False, delete_axis_0=False, count_threshold=_thresh//2)[0]
    assert np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
                NEW_MOCK_X_BOOL[np.logical_not(np.isnan(NEW_MOCK_X_BOOL))]), f"handle_as_bool column wrongly altered"
    # NAN ABOVE THRESH AND NOTHING DELETED
    NEW_MOCK_X_BOOL = MOCK_X_BOOL.copy().astype(np.float64)
    NAN_MASK = np.random.choice(_rows, _thresh, replace=False)
    NEW_MOCK_X_BOOL[NAN_MASK] = np.nan
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_BOOL, ignore_nan=False, delete_axis_0=False, count_threshold=_thresh//2)[0]
    assert len(TRFM_X) == len(NEW_MOCK_X_BOOL), f"handle_as_bool column was wrongly altered"
    assert np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
                NEW_MOCK_X_BOOL[np.logical_not(np.isnan(NEW_MOCK_X_BOOL))]), f"handle_as_bool column wrongly altered"

    NEW_MOCK_X_BOOL = MOCK_X_BOOL.copy().astype(np.float64)
    NAN_MASK = np.random.choice(_rows, _thresh//2-1, replace=False)
    NEW_MOCK_X_BOOL[NAN_MASK] = np.nan
    # NAN BELOW THRESH AND ROWS DELETED
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_BOOL, ignore_nan=False, delete_axis_0=True, count_threshold=_thresh//2)[0]
    assert np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
                NEW_MOCK_X_BOOL[np.logical_not(np.isnan(NEW_MOCK_X_BOOL))]), \
                    f"incorrect bin nan rows deleted"
    # NAN ABOVE THRESH AND ROWS NOT DELETED
    NEW_MOCK_X_BOOL = MOCK_X_BOOL.copy().astype(np.float64)
    NAN_MASK = np.random.choice(_rows, _thresh//2, replace=False)
    NEW_MOCK_X_BOOL[NAN_MASK] = np.nan
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_BOOL, ignore_nan=False, delete_axis_0=True, count_threshold=_thresh//2)[0]
    assert len(TRFM_X) == len(NEW_MOCK_X_BOOL), f"bin column was altered"
    assert np.array_equiv(TRFM_X[np.logical_not(np.isnan(TRFM_X))],
                NEW_MOCK_X_BOOL[np.logical_not(np.isnan(NEW_MOCK_X_BOOL))]), f"bin column was altered"


    # mmct DELETES A COLUMN WITH ONE UNIQUE
    NEW_MOCK_X_INT = np.ones(_rows).astype(np.float64).reshape((-1, 1))
    TRFM_X = arg_setter(MOCK_X=NEW_MOCK_X_INT, ignore_nan=False, delete_axis_0=False,
                                                count_threshold=_thresh // 2)[0]
    assert TRFM_X.shape == (_rows, 0), f'mmct did not delete a column of constants; shape = {TRFM_X.shape}'



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

    # CREATE A COLUMN OF CONSTANTS TO DEMONSTRATE IT IS ALWAYS DELETED
    MOCK_X_INT = np.ones((_rows, 1)).astype(object)
    MOCK_X_NO_NAN = np.empty((_rows, 0), dtype=object)
    # CREATE A COLUMN FOR HANDLE AS BOOL WHERE 0 IS < THRESH
    MOCK_X_BOOL_2 = np.random.randint(0, _source_len, (_rows, 1))
    MOCK_X_BOOL_2[np.random.choice(_rows, 2, replace=False), 0] = 0
    for X in [MOCK_X_BIN, MOCK_X_NBI, MOCK_X_FLT, MOCK_X_STR, MOCK_X_BOOL, MOCK_X_BOOL_2, MOCK_X_INT]:
        MOCK_X_NO_NAN = np.hstack((MOCK_X_NO_NAN.astype(object), X.astype(object)))

    MOCK_X_NAN = MOCK_X_NO_NAN.copy()
    for _ in range(MOCK_X_NO_NAN.size//10):
        _row_coor = np.random.randint(0, MOCK_X_NO_NAN.shape[0])
        _col_coor = np.random.randint(0, MOCK_X_NO_NAN.shape[1])
        MOCK_X_NAN[_row_coor, _col_coor] = np.nan

    del _row_coor, _col_coor

    HAS_NAN = [True, False]
    IGN_COLS = [None, [0, 3]]
    IGN_NAN = [True, False]
    IGN_NBI = [True, False]
    IGN_FLT = [True, False]
    HAB = [None, [4], [4,5]]
    DA0 = [False, True]
    THRESH = [_thresh // 2, _thresh, 2 * _thresh]


    def get_unqs_cts_again(_COLUMN_OF_X):
        try:
            return np.unique(_COLUMN_OF_X.astype(np.float64), return_counts=True)
        except:
            return np.unique(_COLUMN_OF_X.astype(str), return_counts=True)


    _total_trials = 1
    for _obj in (HAS_NAN, IGN_COLS, IGN_NAN, IGN_NBI, IGN_FLT, HAB, DA0, THRESH):
        _total_trials *= len(_obj)
    del _obj

    ctr = 0
    for _has_nan in HAS_NAN:
        for _ignore_columns in IGN_COLS:
            for _ignore_nan in IGN_NAN:
                for _ignore_non_binary_integer_columns in IGN_NBI:
                    for _ignore_float_columns in IGN_FLT:
                        for _handle_as_bool in HAB:
                            for _delete_axis_0 in DA0:
                                for _count_threshold in THRESH:

                                    ctr += 1
                                    print(f'\033[92mRunning accuracy trial {ctr} of {_total_trials}\033[0m')

                                    if _has_nan:
                                        MOCK_X = MOCK_X_NAN.copy()
                                        REF_X = MOCK_X_NAN.copy()
                                    elif not _has_nan:
                                        MOCK_X = MOCK_X_NO_NAN.copy()
                                        REF_X = MOCK_X_NO_NAN.copy()
                                    else:
                                        raise Exception(f'_has_nan logic failed')

                                    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                                    # MAKE REF_X
                                    TEST_UNQS_CTS = joblib.Parallel(return_as='list')(
                                        joblib.delayed(get_unqs_cts_again)(MOCK_X[:, c_idx]) for c_idx in range(MOCK_X.shape[1]))

                                    unq_ct_dict = {}
                                    _DTYPES = []
                                    for c_idx, (UNQS, CTS) in enumerate(TEST_UNQS_CTS):

                                        if _ignore_columns and c_idx in _ignore_columns:
                                            _DTYPES.append(None)
                                            continue

                                        try:
                                            _DTYPE_DUM = UNQS[np.logical_not(np.isnan(UNQS.astype(np.float64)))]
                                            if np.array_equiv(_DTYPE_DUM.astype(np.float64).astype(np.int32), _DTYPE_DUM.astype(np.float64)):
                                                if len(_DTYPE_DUM)==1:
                                                    _DTYPES.append('constant')
                                                elif len(_DTYPE_DUM)==2:
                                                    _DTYPES.append('bin_int')
                                                else:
                                                    _DTYPES.append('int')
                                            else:
                                                _DTYPES.append('float')
                                            del _DTYPE_DUM
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
                                                if str(unq).lower()=='nan':
                                                    try:
                                                        del unq_ct_dict[c_idx][unq]
                                                    except:
                                                        try:
                                                            unq_ct_dict[c_idx] = dict((zip(np.fromiter(unq_ct_dict[c_idx].keys(), dtype='<U20'), list(unq_ct_dict[c_idx].values()))))
                                                            del unq_ct_dict[c_idx]['nan']
                                                            unq_ct_dict[c_idx] = dict((zip(np.fromiter(unq_ct_dict[c_idx].keys(), dtype=np.float64), list(unq_ct_dict[c_idx].values()))))
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

                                            NEW_DICT = {0:0, 1:0}
                                            UNQS_MIGHT_BE_DELETED = []
                                            for k,v in dict((zip(np.fromiter(unq_ct_dict[c_idx].keys(), dtype=np.float64),
                                                                 np.fromiter(unq_ct_dict[c_idx].values(), dtype=np.float64))
                                                )).items():
                                                if k == 0:
                                                    NEW_DICT[0] += v
                                                elif str(k).lower()=='nan':
                                                    NEW_DICT[k] = v
                                                else:
                                                    UNQS_MIGHT_BE_DELETED.append(k)
                                                    NEW_DICT[1] += v

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

                                            if len([_ for _ in DELETE_DICT[c_idx] if str(_).lower() != 'nan']) >= 1:
                                                DELETE_DICT[c_idx].append(f'DELETE COLUMN')
                                            elif len([_ for _ in unq_ct_dict[c_idx] if (str(_).lower() != 'nan' and _ not in DELETE_DICT[c_idx])]) <= 1:
                                                DELETE_DICT[c_idx].append(f'DELETE COLUMN')

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

                                            if len([_ for _ in unq_ct_dict[c_idx] if (str(_).lower() != 'nan' and _ not in DELETE_DICT[c_idx])]) <= 1:
                                                DELETE_DICT[c_idx].append(f'DELETE COLUMN')

                                        if len(DELETE_DICT[c_idx])==0:
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

                                    if not np.array_equiv(TRFM_X.astype(str), REF_X.astype(str)):
                                        print(f'\n\033[91m')
                                        print(f'MOCK_X - {MOCK_X.shape} :')
                                        print(pd.DataFrame(MOCK_X).head(10))
                                        print(f'TRFM_X - {TRFM_X.shape} :')
                                        print(pd.DataFrame(TRFM_X).head(10))
                                        print(f'REF_X - {REF_X.shape}:')
                                        print(pd.DataFrame(REF_X).head(10))
                                        print(f'_ignore_columns = {_ignore_columns}')
                                        print(f'_ignore_nan = {_ignore_nan}')
                                        print(f'_ignore_non_binary_integer_columns = {_ignore_non_binary_integer_columns}')
                                        print(f'_ignore_float_columns = {_ignore_float_columns}')
                                        print(f'_handle_as_bool = {_handle_as_bool}')
                                        print(f'_delete_axis_0 = {_delete_axis_0}')
                                        print(f'_count_threshold = {_count_threshold}')
                                        time.sleep(1)
                                        raise AssertionError(f"TRFM_X != REF_X")









    print(f'\n\033[92mAll tests passed.')


