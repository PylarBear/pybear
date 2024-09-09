# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing_extensions import Union



def _validate_n_jobs(
    _n_jobs: Union[int, None]
) -> Union[int, None]:

    """

    Validate that n_jobs is an integer in [-1, 1, 2, 3,...] or None.
    If None, do not overwrite, return as None.


    Parameters
    ----------
    _n_jobs: Union[int, None] -
        Number of jobs to run in parallel.


    Return
    ------
    -
        _n_jobs: Union[int, None] - Validated n_jobs


    """


    if _n_jobs is None:
        return

    try:
        float(_n_jobs)
        if isinstance(_n_jobs, bool):
            raise
        if int(_n_jobs) != _n_jobs:
            raise
        if _n_jobs < -1 or _n_jobs == 0:
            raise
    except:
        raise ValueError(f"n_jobs must be None, -1, or an integer greater than 0")


    return _n_jobs












