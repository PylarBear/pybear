# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from pybear.preprocessing._MinCountTransformer._transform._make_row_and_column_masks \
    import _make_row_and_column_masks

import numpy as np




# run a "big" X to see if n_jobs is using all processors



_rows = 20_000
_cols = 1_000
_max = _rows // 20
_thresh = _rows // _max

# build fixtures ** * ** * ** * ** * ** * ** * ** * **
print(f'building test objects... ', end='')
ctr = 0
while True:

    if ctr > 100:
        raise Exception(f"unable to build a good X in 100 tries")

    ctr += 1

    X = np.random.randint(0, _max, (_rows, _cols), dtype=np.uint16)

    TCBC = {}
    for _idx, _column in enumerate(X.transpose()):
        TCBC[_idx] = dict((zip(*np.unique(_column, return_counts=True))))


    _INSTR = {}
    for col_idx, column_unq_ct_dict in TCBC.items():

        _INSTR[col_idx] = []
        for unq, ct in column_unq_ct_dict.items():
            if ct < _thresh:
                _INSTR[col_idx].append(unq)

        if len(_INSTR[col_idx]) >= len(column_unq_ct_dict) - 1:
            _INSTR[col_idx].append('DELETE COLUMN')

    for col_idx in _INSTR:
        if not len(_INSTR[col_idx]) > 0:
            continue
    else:
        break
print(f'Done.')
# END build fixtures ** * ** * ** * ** * ** * ** * ** * **

print(f"running _make_row_and_column_masks (n_jobs should peg the CPUs)... ", end='')
ROW_KEEP_MASK, COLUMN_KEEP_MASK = _make_row_and_column_masks(
    X,
    TCBC,
    _INSTR,
    _reject_unseen_values=True,
    _n_jobs=-1
)

print(f"Done.")


# notes 24_06_14_09_59_00.... n_jobs is working, but it took about 5 seconds
# after starting _make_row_and_column_masks to kick in













