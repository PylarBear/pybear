# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numbers



def _cond_total_passes(
    _total_passes: numbers.Integral
) -> int:

    """
    Standardize `total_passes` to python integer.


    Parameters
    ----------
    _total_passes:
        numbers.Integral - the number of passes of grid search to perform.


    Returns
    -------
    -
        total_passes: int - total_passes as python integer.

    """


    return int(_total_passes)



