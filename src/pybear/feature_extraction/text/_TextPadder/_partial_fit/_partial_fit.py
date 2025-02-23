# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import XContainer

import numbers


def _partial_fit(
    X: XContainer
) -> numbers.Integral:


    """
    Get the number of features in X.


    Parameters
    ----------
    X:
        XContainer - the data.


    Return
    ------
    -
        numbers.Integral

    """


    return max(map(len, X))







