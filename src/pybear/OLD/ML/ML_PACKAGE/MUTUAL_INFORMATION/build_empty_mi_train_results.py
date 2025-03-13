import sys, inspect
import numpy as np, pandas as pd


def build_empty_mi_train_results(DATA_HEADER):

    # CREATE "TRAIN_RESULTS" OBJECT & HEADER, TO BE RETURNED TO MIRun FOR PRINT TO SCREEN & DUMP TO FILE

    DATA_HEADER = DATA_HEADER.reshape((1,-1))[0]

    TRAIN_RESULTS_HEADER = pd.MultiIndex(
        levels=[ ['COLUMN', 'INDIV', 'FINAL', 'CUMUL'],
                 ['NAME', 'MI SCORE', 'R', 'R2', 'ADJ R2', 'F', 'COEFFS', 'p VALUE'] ],
        codes=[[1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3], [1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5]] )

    TRAIN_RESULTS = pd.DataFrame(index=DATA_HEADER, columns=TRAIN_RESULTS_HEADER, dtype=np.float64).fillna('-')

    del DATA_HEADER, TRAIN_RESULTS_HEADER

    return TRAIN_RESULTS









