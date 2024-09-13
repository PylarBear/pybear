import numpy as np


# THIS IS A FUNCTION USED IN TEST OF MLRegression() FOR ACCURACY, AS REFEREE CORRECT VALUES
# TESTED AND VERIFIED 6/1/23

def exp_r2_calc(DATA, TARGET, COEFFS, has_intercept=None):   # RELIES ON np.corrcoeff FOR has_intercept=True
    # DATA & TARGET AS ROW, COEFFS AS ROW
    PREDICTED = np.matmul(DATA.astype(np.float64), COEFFS.astype(np.float64)).astype(np.float64)
    if has_intercept:
        # MUST PASS VECTORS TO corrcoeff AS [[]=COLUMN], GIVING DIV ERRORS AND THE LIKE FOR [[]=ROW]
        exp_r2 = np.corrcoef(PREDICTED.transpose(), TARGET.transpose())[0][1]**2
    elif not has_intercept:
        SSR = np.sum(np.power(PREDICTED, 2))
        SSE = np.sum(np.power(TARGET - PREDICTED, 2))
        exp_r2 = SSR / (SSR + SSE)


    return exp_r2