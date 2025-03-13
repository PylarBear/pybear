import numpy as np


# THIS IS A FUNCTION USED IN TEST OF MLRegression() FOR ACCURACY, AS REFEREE CORRECT VALUES
# TESTED AND VERIFIED 6/1/23

def exp_f_calc(DATA, TARGET, rglztn_fctr):   # FROM SCRATCH VIA matmul
    # DATA & TARGET AS ROW
    # USE np.array INSTEAD OF astype SO THAT SINGLE NUMBERS ARE RETURNED AS []


    COEFFS = np.matmul(
        np.matmul(XTX_INV.astype(np.float64), DATA.transpose().astype(np.float64)).astype(np.float64),
        TARGET.astype(np.float64)
    ).astype(np.float64)#.reshape((1 ,-1))

    return COEFFS   # AS ROW