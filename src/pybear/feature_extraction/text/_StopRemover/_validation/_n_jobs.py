# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import numbers



def _val_n_jobs(_n_jobs: Union[numbers.Integral, None]) -> None:

    """
    Validate n_jobs. Must be None or integer >= -1 and != 0.


    Parameters
    ----------
    _n_jobs:
        Union[numbers.Integral, None]] - the number of cores/threads to
        use when parallelizing the search for stop words in the rows of
        X. The default is to use processes but can be set by running
        StopRemover under a joblib parallel_config context manager. None
        uses the joblib's default number of cores/threads. -1 uses all
        available cores/threads.


    Returns
    -------
    -
        None

    """


    if _n_jobs is None:
        return


    err_msg = f"'n_jobs' must be None or an integer >= -1 but not equal to 0."

    if not isinstance(_n_jobs, numbers.Integral):
        raise TypeError(err_msg)

    if isinstance(_n_jobs, bool):
        raise TypeError(err_msg)

    if not _n_jobs >= -1 or _n_jobs == 0:
        raise ValueError(err_msg)




