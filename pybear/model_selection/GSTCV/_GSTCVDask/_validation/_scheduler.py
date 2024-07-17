# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#





from typing import Union

from distributed import Client

from model_selection.GSTCV._type_aliases import (
    SchedulerType
)




def _validate_scheduler(
        _scheduler: SchedulerType
        ) -> SchedulerType:

    """
    As of 24_07_07_13_40_00, the scheduler is being instantiated here
    if "None" was passed to the scheduler kwarg of GSTCVDask. Prior to
    this, the scheduler was instantiated with hard n_workers and
    threads_per_workers. The hard kwargs have been removed to allow for
    the client to be manipulated by a context manager (presumably, but
    not pizza verified as of 24_07_07). If a scheduler was passed, this module
    does not perform any validation but allows that to be handled by dask
    at compute time.

    Parameters
    ----------
    _scheduler:
        _scheduler to be validated

    Return
    ------
    -
        _scheduler

    """


    if _scheduler is None:
        # If no special scheduler is passed, use a n_jobs local cluster
        _scheduler = Client()

            # 24_07_06_16_01_00 originally was passing these to Client,
            # now thinking that in order for Client to accept input from
            # context manager, need to leave it empty. pizza verify.

            # n_workers=_n_jobs,
            # threads_per_worker=1,
            # set_as_default=True

    else:
        # self.scheduler ONLY FLOWS THRU TO compute(), SO LET compute()
        # HANDLE VALIDATION
        pass

    return _scheduler




















