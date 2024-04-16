import sys, inspect
import numpy as np, pandas as pd






def build_empty_mlr_train_results(DATA_HEADER):

    DATA_HEADER = DATA_HEADER.reshape((1,-1))[0]

    TRAIN_RESULTS_HEADER = pd.MultiIndex(
        levels=[['COLUMN', '      ', 'OVERALL'],
                ['NAME', 'p VALUE', 'COEFFS', 'R', 'R2', 'ADJ R2', 'F']],
        codes=[[1, 1, 2, 2, 2, 2], [1, 2, 3, 4, 5, 6]])

    # TRAIN_RESULTS_HEADER = [
    #     ['COLUMN', '      ',  '      ', 'OVERALL', 'OVERALL', 'OVERALL', 'OVERALL'],
    #     [' NAME ', 'p VALUE', 'COEFFS', '   R   ', '   R2  ', ' ADJ R2', '   F   ']
    #     ]

    TRAIN_RESULTS = pd.DataFrame(index=DATA_HEADER, columns=TRAIN_RESULTS_HEADER, dtype=np.float64).fillna('-')


    del DATA_HEADER, TRAIN_RESULTS_HEADER

    return TRAIN_RESULTS























