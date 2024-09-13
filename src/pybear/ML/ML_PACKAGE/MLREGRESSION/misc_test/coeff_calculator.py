import numpy as np


# THIS IS A FUNCTION USED IN TEST OF MLRegression() FOR ACCURACY, AS REFEREE CORRECT VALUES
# TESTED AND VERIFIED 6/1/23

def exp_coeff_calc(DATA, TARGET, rglztn_fctr):   # FROM SCRATCH VIA matmul
    # DATA & TARGET AS ROW
    # USE np.array INSTEAD OF astype SO THAT SINGLE NUMBERS ARE RETURNED AS []

    XTX = np.array(np.matmul(DATA.transpose().astype(np.float64), DATA.astype(np.float64)), dtype=np.float64)
    XTX += rglztn_fctr * np.identity(len(XTX)).astype(np.float64)
    XTX_INV = np.linalg.inv(XTX)
    COEFFS = np.matmul(
        np.matmul(XTX_INV.astype(np.float64), DATA.transpose().astype(np.float64)).astype(np.float64),
        TARGET.astype(np.float64)
    ).astype(np.float64)#.reshape((1 ,-1))

    return COEFFS   # AS ROW




