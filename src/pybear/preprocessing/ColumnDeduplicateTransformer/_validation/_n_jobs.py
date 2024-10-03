# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing_extensions import Union



def _val_n_jobs(_n_jobs: Union[int, None]) -> None:

    """
    n_jobs must be -1 or integer >= 1

    """


    err_msg = f"n_jobs must be None, -1, or an integer greater than 0"

    if _n_jobs is None:
        return

    try:
        float(_n_jobs)
        if isinstance(_n_jobs, bool):
            raise Exception
        if int(_n_jobs) != _n_jobs:
            raise Exception
        _n_jobs = int(_n_jobs)
    except:
        raise TypeError(err_msg)

    if _n_jobs == -1 or _n_jobs >= 1:
        pass
    else:
        raise ValueError(err_msg)

    del err_msg






