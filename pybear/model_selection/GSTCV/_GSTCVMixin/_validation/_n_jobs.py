# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing import Union



def _validate_n_jobs(_n_jobs: Union[int, None]) -> Union[int, None]:

    if _n_jobs is None:
        return

    try:
        float(_n_jobs)
        if isinstance(_n_jobs, bool):
            raise Exception
        if int(_n_jobs) != _n_jobs:
            raise Exception
        if _n_jobs < -1 or _n_jobs == 0:
            raise Exception
    except:
        raise ValueError(f"n_jobs must be None, -1, or an integer greater than 0")


    return _n_jobs












