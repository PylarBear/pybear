# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
import joblib


from typing import Union, Iterable


# this is fixture for MinCountTransformer_test
# test for this is in pytest_fixtures.tests.mock_mct_test


class mmct:

    # y did u do it like this with __init__?
    # 24_06_04_11_13_00 this is to be able to pass everything all at once
    # to trfm, and not have to re-instantiate the entire class when wanting
    # to change certain things.... maybe it could have been a function?
    # is there a class behavior that u wanted, other than just be a class
    # like MinCountTransformer?

    def __init__(self):
        pass


    def trfm(
            self,
            MOCK_X: np.ndarray,
            MOCK_Y: Union[np.ndarray, None],
            ignore_columns: Union[Iterable[int], None],
            ignore_nan: bool,
            ignore_non_binary_integer_columns: bool,
            ignore_float_columns: bool,
            handle_as_bool: Union[Iterable[int], None],
            delete_axis_0: bool,
            count_threshold: int
        ) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:

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
                    raise ValueError(f'ignore_columns column index {i} out of '
                             f'bounds for data with {MOCK_X.shape[1]} columns')
                except Exception as e1:
                    raise Exception(f'ignore_columns remove from ACTIVE_C_IDXS '
                            f'except for reason other than ValueError --- {e1}')

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
















