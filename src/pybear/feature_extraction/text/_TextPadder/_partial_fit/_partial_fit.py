# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import XContainer

import numbers


def _partial_fit(
    _X: XContainer
) -> numbers.Integral:
    """Get the number of features in `X`.

    Parameters
    ----------
    X : XContainer
        The data.

    Returns
    -------
    n_features : numbers.Integral
        The number of features in `X`.

    """


    if hasattr(_X, 'shape'):
        return _X.shape[1]
    else:
        return max(map(len, _X))







