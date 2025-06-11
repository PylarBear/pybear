# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import numbers

import numpy as np
from scipy.special import logsumexp



def mutual_information(
    DATA: Sequence[numbers.Real],
    TARGET: Sequence[numbers.Real]
) -> float:

    """


    Parameters
    ----------
    DATA:
        Sequence[numbers.Real] - 
    TARGET:
        Sequence[numbers.Real] -


    Returns
    -------
    -
        float - pizza

    """


    TARGET_VECTOR = np.array(TARGET, dtype=np.float64).ravel()

    _y_len = len(TARGET_VECTOR[0])

    TARGET_UNIQUES = np.unique(TARGET_VECTOR)

    DATA = np.array(DATA, dtype=np.float64)

    DATA_UNIQUES = np.unique(DATA)
    data_len = DATA.size


    total_score = 0
    for x_idx in range(len(DATA_UNIQUES)):

        X_OCCUR = (
            DATA.astype(np.float64) == DATA_UNIQUES[x_idx].astype(np.float64)
        ).astype(np.uint8)
        x_freq = np.sum(X_OCCUR) / data_len

        for y_idx in range(len(TARGET_UNIQUES)):

            Y_OCCUR = (
                np.int8(TARGET_VECTOR[0].astype(np.float64) == TARGET_UNIQUES[y_idx].astype(np.float64))
            ).astype(np.uint8)

            Y_SUM = np.sum(Y_OCCUR)

            y_freq = Y_SUM / _y_len



            p_x_y = np.matmul(
                X_OCCUR.astype(float),
                Y_OCCUR.astype(float),
                dtype=np.float64
            )

            p_x_y /= Y_SUM

            try:
            # CONVERT TO logsumexp FOR DEALING WITH EXTREMELY BIG OR SMALL NUMBERS
            # GIVING RuntimeWarning AS A REGULAR FXN IF p_x_y GOES TO 0
            # (NUMERATOR IN LOG10) THEN BLOWUP, SO except AND pass IF p_x_y IS 0
                with np.errstate(divide='raise'):
                    total_score += p_x_y * (np.log10(logsumexp(p_x_y)) - np.log10(logsumexp(x_freq)) - np.log10(logsumexp(y_freq)))
            except FloatingPointError:
                pass
                # RuntimeWarning pizza
            except Exception as e:
                raise e from None


    return total_score










if __name__ == '__main__':

    DATA = np.random.randint(1,100,(50,))
    # DATA = sd.unzip_to_ndarray_float64(DATA)[0][0]
    TARGET = np.random.randint(0,2,(1,100), dtype=np.int8)

    total_score = mutual_information(DATA, TARGET).run()

    print(total_score)




