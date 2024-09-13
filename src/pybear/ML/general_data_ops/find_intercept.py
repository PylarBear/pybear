import numpy as np
from data_validation import arg_kwarg_validater as akv
from general_data_ops import get_shape as gs


def find_intercept(DATA_AS_LISTTYPE, orientation):
    '''Finds column(s) of constants. Returns (dict/empty dict of non-zero constant indices, list/em of zero idxs.'''
    # RETURNS COLUMNS OF ZEROS FOR SUBSEQUENT HANDLING. len(COLUMNS OF CONSTANTS) SHOULD BE 1, BUT RETURN FOR HANDLING IF OTHERWISE.

    orientation = akv.arg_kwarg_validater(orientation, 'orientation', ['ROW','COLUMN'], 'find_intercept', 'find_intercept')

    _rows, _cols = gs.get_shape('DATA', DATA_AS_LISTTYPE, orientation)

    # COLUMNS_OF_CONSTANTS IS RETURNED AS A DICT WITH KEY AS COLUMN IDX AND VALUE AS CONSTANT VALUE
    COLUMNS_OF_CONSTANTS, COLUMNS_OF_ZEROS = {}, []
    for col_idx in range(_cols):

        if orientation=='ROW':
            _min, _max = np.min(DATA_AS_LISTTYPE[:, col_idx]), np.max(DATA_AS_LISTTYPE[:, col_idx])

        elif orientation=='COLUMN':
            _min, _max = np.min(DATA_AS_LISTTYPE[col_idx]), np.max(DATA_AS_LISTTYPE[col_idx])

        if _min != _max: continue
        elif _min == _max and _min == 0: COLUMNS_OF_ZEROS.append(col_idx)
        elif _min == _max: COLUMNS_OF_CONSTANTS = COLUMNS_OF_CONSTANTS | {col_idx: _min}

    del orientation, _rows, _cols

    return COLUMNS_OF_CONSTANTS, COLUMNS_OF_ZEROS










