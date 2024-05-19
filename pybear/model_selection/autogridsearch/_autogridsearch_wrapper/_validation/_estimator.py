# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import inspect

def _estimator(_estimator):

    """
    Validate estimator is a class, at least

    """

    if not inspect.isclass(_estimator):
        raise TypeError(f"'estimator' must be a class")


