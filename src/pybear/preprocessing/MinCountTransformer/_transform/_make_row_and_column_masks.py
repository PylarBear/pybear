# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from .._type_aliases import DataType, TotalCountsByColumnType
import numpy.typing as npt

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as ss

from ._parallelized_row_masks import _parallelized_row_masks
from .._partial_fit._column_getter import _column_getter



def _make_row_and_column_masks(
    X: Union[npt.NDArray[DataType], pd.DataFrame, ss.csc_matrix, ss.csc_array],
    _total_counts_by_column: TotalCountsByColumnType,
    _delete_instr: dict[int, Union[str, DataType]],
    _reject_unseen_values: bool,
    _n_jobs: Union[int, None]
) -> Union[npt.NDArray[bool], npt.NDArray[bool]]:

    """
    Make a mask that indicates which columns to keep and another mask
    that indicates which rows to keep from X. Columns that are to be
    deleted are already flagged in _delete_instr with 'DELETE COLUMN'.
    For rows, iterate over all columns, and within each column iterate
    over its respective uniques in _delete_instr, to identify which rows
    are to be deleted.

    Parameters
    ----------
    X:
        np.ndarray[DataType] - the data to be transformed
    _total_counts_by_column:
        dict[int, dict[DataType, int]] - dictionary holding the uniques
        and their counts for each column
    _delete_instr:
        dict[int, Union[str, DataType]] - _delete_instr is a dictionary
        that is keyed by column index and the values are lists. Within
        the lists is information about operations to perform with respect
        to values in the column. The following items may be in the list:

        -'INACTIVE' - ignore the column and carry it through for all
            other operations
        -Individual values - (in raw datatype format, not converted to
            string) indicates to delete the rows on axis 0 that contain
            that value in that column, including 'nan' or np.nan values
        -'DELETE COLUMN' - perform any individual row deletions that
            need to take place while the column is still in the data,
            then delete the column from the data.
    _reject_unseen_values: bool - If False, do not even look to see if
        there are unknown uniques in the column. If True, compare uniques
        in the column against uniques in _COLUMN_UNQ_CT_DICT and raise
        exception if there is a value not previously seen.
    _n_jobs: int, default=None
        Number of CPU cores used when parallelizing over features during
        fit. None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.


    Return
    ------
    -
        tuple[ROW_KEEP_MASK, COLUMN_KEEP_MASK]:
            tuple[np.ndarray[np.uint8], np.ndarray[np.uint8] - the masks
            for the rows and columns to keep in binary integer format.

    """


    # MAKE COLUMN DELETE MASK ** * ** * ** * ** * ** * ** * ** * ** * **

    _delete_columns_mask = np.zeros(X.shape[1], dtype=np.uint32)

    for col_idx, _instr in _delete_instr.items():
        if 'DELETE COLUMN' in _instr:
            _delete_columns_mask[col_idx] += 1
            _instr = _instr[:-1]

    # END MAKE COLUMN DELETE MASK ** * ** * ** * ** * ** * ** * ** * **

    # MAKE ROW DELETE MASK ** * ** * ** * ** * ** * ** * ** * ** * ** *

    _ACTIVE_COL_IDXS = []
    for col_idx, _instr in _delete_instr.items():
        if 'INACTIVE' in _instr or len(_instr) == 0:
            continue
        else:
            _ACTIVE_COL_IDXS.append(col_idx)

    # DONT HARD-CODE backend, ALLOW A CONTEXT MANAGER TO SET
    joblib_kwargs = {'prefer': 'processes', 'return_as': 'list',
                     'n_jobs': _n_jobs}
    ROW_MASKS = joblib.Parallel(**joblib_kwargs)(
        joblib.delayed(_parallelized_row_masks)(
            _column_getter(X, _idx).reshape((-1, 1)),
            _total_counts_by_column[_idx],
            _delete_instr[_idx],
            _reject_unseen_values,
            _idx
        ) for _idx in _ACTIVE_COL_IDXS)

    del _ACTIVE_COL_IDXS, joblib_kwargs


    # sum the individual masks in ROW_MASKS
    # _delete_rows_mask = np.sum(ROW_MASKS, axis=0).astype(np.uint8)
    # 24_06_15 dont use np.sum(ROW_MASKS, axis=0)!!! This works fine as
    # long there are any non-zero in the masks. But when all the masks are
    # full of zeroes, even with specifying axis=0 numpy is reducing this
    # to a single value instead of a vector. Sum the vectors individually.

    _delete_rows_mask = np.zeros(X.shape[0], dtype=np.uint8)
    for _MASK in ROW_MASKS:
        _delete_rows_mask += _MASK

    # END MAKE ROW DELETE MASK ** * ** * ** * ** * ** * ** * ** * ** * **


    ROW_KEEP_MASK = np.logical_not(_delete_rows_mask)
    del _delete_rows_mask
    COLUMN_KEEP_MASK = np.logical_not(_delete_columns_mask)
    del _delete_columns_mask

    delete_all_msg = \
        lambda x: f"this threshold and recursion depth will delete all {x}"

    if True not in ROW_KEEP_MASK:
        raise ValueError(delete_all_msg('rows'))

    if True not in COLUMN_KEEP_MASK:
        raise ValueError(delete_all_msg('columns'))

    del delete_all_msg

    # ^^^ END BUILD ROW_KEEP & COLUMN_KEEP MASKS ** ** ** ** ** ** ** **

    return ROW_KEEP_MASK, COLUMN_KEEP_MASK

















