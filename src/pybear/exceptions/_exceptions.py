# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




class NotFittedError(ValueError, AttributeError):

    """
    Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help
    with exception handling.

    Examples
    --------
    >>> from pybear.preprocessing import ColumnDeduplicateTransformer as CDT
    >>> from pybear.utilities.exceptions import NotFittedError
    >>> import numpy as np
    >>> X = np.random.randint(0, 10, (5,3))
    >>> try:
    ...     CDT().transform(X)
    ... except NotFittedError as e:
    ...     print(repr(e))
    NotFittedError("This LinearSVC instance is not fitted yet. Call 'fit'
    with appropriate arguments before using this estimator."...)


    """
