# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal
from typing_extensions import Union
import numpy.typing as npt
from .._type_aliases import (
    DataType,
    InternalXContainer,
    TotalCountsByColumnType
)

import itertools
import numbers

import joblib
import numpy as np

from ._parallelized_row_masks import _parallelized_row_masks
from .._partial_fit._columns_getter import _columns_getter



def _make_row_and_column_masks(
    _X: InternalXContainer,
    _total_counts_by_column: TotalCountsByColumnType,
    _delete_instr: dict[
        int,
        Union[Literal['INACTIVE', 'DELETE ALL', 'DELETE COLUMN'], DataType]
    ],
    _reject_unseen_values: bool,
    _n_jobs: Union[numbers.Integral, None]
) -> tuple[npt.NDArray[bool], npt.NDArray[bool]]:

    """
    Make a mask that indicates which columns to keep and another mask
    that indicates which rows to keep from X. Columns that are to be
    deleted are already flagged in _delete_instr with 'DELETE COLUMN'.
    For rows, iterate over all columns, and within each column iterate
    over its respective uniques in _delete_instr, to identify which rows
    are to be deleted.


    Parameters
    ----------
    _X:
        Union[np.ndarray, pd.DataFrame, scipy.sparse] - the data to be
        transformed.
    _total_counts_by_column:
        dict[int, dict[DataType, int]] - dictionary holding the uniques
        and their counts for each column.
    _delete_instr:
        dict[
            int,
            Union[Literal['INACTIVE', 'DELETE ALL', 'DELETE COLUMN'], DataType]
        ] - a dictionary that is keyed by column index and the values are
        lists. Within the lists is information about operations to
        perform with respect to values in the column. The following items
        may be in the list:

        -'INACTIVE' - ignore the column and carry it through for all
            other operations

        -Individual values - (in raw datatype format, not converted to
            string) indicates to delete the rows on axis 0 that contain
            that value in that column, including nan-like values

        -'DELETE ALL' - delete every value in the column.

        -'DELETE COLUMN' - perform any individual row deletions that
            need to take place while the column is still in the data,
            then delete the column from the data.
    _reject_unseen_values: bool - If False, do not even look to see if
        there are unknown uniques in the column. If True, compare uniques
        in the column against uniques in _COLUMN_UNQ_CT_DICT and raise
        exception if there is a value not previously seen.
    _n_jobs:
        Union[int, None] - Number of CPU cores/threads used when doing
        parallel search of features during fit. -1 means using all
        processors.


    Return
    ------
    -
        tuple[ROW_KEEP_MASK, COLUMN_KEEP_MASK]:
            tuple[np.ndarray[bool], np.ndarray[bool] - the masks for the
            rows and columns to keep in binary integer format.

    """


    # MAKE COLUMN DELETE MASK ** * ** * ** * ** * ** * ** * ** * ** * **

    _delete_columns_mask = np.zeros(_X.shape[1], dtype=np.uint32)

    for col_idx, _instr in _delete_instr.items():
        if 'DELETE COLUMN' in _instr:
            _delete_columns_mask[col_idx] += 1
            _instr = _instr[:-1]

    # END MAKE COLUMN DELETE MASK ** * ** * ** * ** * ** * ** * ** * **

    # MAKE ROW DELETE MASK ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # if there is a 'DELETE ALL' just make a mask of all ones and forget
    # about the loop.
    for _instr in _delete_instr.values():
        if 'DELETE ALL' in _instr:
            _delete_rows_mask = np.ones(_X.shape[0], dtype=np.uint8)
            break
    # otherwise build the mask from individual row masks.
    else:
        # pizza axe this if joblib w chunks works
        # _delete_rows_mask = np.zeros(_X.shape[0], dtype=np.uint8)
        # for col_idx, _instr in _delete_instr.items():
        #     if 'INACTIVE' in _instr or len(_instr) == 0:
        #         continue
        #     else:
        #         _delete_rows_mask += _parallelized_row_masks(
        #             _columns_getter(_X, col_idx).ravel(),
        #             _total_counts_by_column[col_idx],
        #             _delete_instr[col_idx],
        #             _reject_unseen_values,
        #             col_idx
        #         )


        # pizza probably should make a joblib test module like for partial_fit
        _ACTIVE_COL_IDXS = []
        for col_idx, _instr in _delete_instr.items():
            if 'INACTIVE' in _instr or len(_instr) == 0:
                continue
            else:
                _ACTIVE_COL_IDXS.append(col_idx)

        # pizza set this to whatever is set in partial_fit
        _n_cols = 1000

        if len(_ACTIVE_COL_IDXS) == 0:
            _delete_rows_mask = np.zeros(_X.shape[0], dtype=np.uint8)
        elif len(_ACTIVE_COL_IDXS) < 2 * _n_cols:
            _delete_rows_mask = _parallelized_row_masks(
                _columns_getter(_X, tuple(_ACTIVE_COL_IDXS)),
                {k:v for k,v in _total_counts_by_column.items() if k in _ACTIVE_COL_IDXS},
                {k:v for k,v in _delete_instr.items() if k in _ACTIVE_COL_IDXS},
                _reject_unseen_values
            )
        else:
            # DONT HARD-CODE backend, ALLOW A CONTEXT MANAGER TO SET
            with joblib.parallel_config(prefer='processes', n_jobs=_n_jobs):
                ROW_MASKS = joblib.Parallel(return_as='list')(
                    joblib.delayed(_parallelized_row_masks)(
                        _columns_getter(
                            _X,
                            tuple(range(i, min(i + _n_cols, len(_ACTIVE_COL_IDXS))))
                        ),
                        {k:v for k,v in _total_counts_by_column.items() if k in np.array(_ACTIVE_COL_IDXS)[list(range(i, min(i + _n_cols, len(_ACTIVE_COL_IDXS))))]},
                        {k:v for k,v in _delete_instr.items() if k in np.array(_ACTIVE_COL_IDXS)[list(range(i, min(i + _n_cols, len(_ACTIVE_COL_IDXS))))]},
                        _reject_unseen_values
                    ) for i in range(0, len(_ACTIVE_COL_IDXS), _n_cols))

                # vstack will work as long as their is at least 1 thing
                # in ROW_MASKS
                _delete_rows_mask = \
                    np.sum(np.vstack(ROW_MASKS, dtype=np.uint8)).astype(np.uint8)

                del ROW_MASKS

        del _ACTIVE_COL_IDXS

    # END MAKE ROW DELETE MASK ** * ** * ** * ** * ** * ** * ** * ** * **


    ROW_KEEP_MASK = np.logical_not(_delete_rows_mask)
    del _delete_rows_mask
    COLUMN_KEEP_MASK = np.logical_not(_delete_columns_mask)
    del _delete_columns_mask

    delete_all_msg = \
        lambda j: f"this threshold and recursion depth will delete all {j}"

    if not any(ROW_KEEP_MASK):
        raise ValueError(delete_all_msg('rows'))

    if not any(COLUMN_KEEP_MASK):
        raise ValueError(delete_all_msg('columns'))

    del delete_all_msg

    # ^^^ END BUILD ROW_KEEP & COLUMN_KEEP MASKS ** ** ** ** ** ** ** **

    return ROW_KEEP_MASK, COLUMN_KEEP_MASK




