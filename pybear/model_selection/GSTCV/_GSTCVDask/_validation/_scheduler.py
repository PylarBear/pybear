# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from distributed import Client, get_client

import contextlib
from typing import Union
from model_selection.GSTCV._type_aliases import SchedulerType



def _validate_scheduler(
        _scheduler: SchedulerType,
        _n_jobs: Union[int, None]
        ) -> SchedulerType:

    """
    24_08_08_17_51_00. The passed scheduler supersedes all other external
    schedulers. If "None" was passed to the scheduler kwarg of GSTCVDask
    (the default), look for an external context manager or global
    scheduler using get_client. If one exists, use nullcontext as an
    internal context manager to not interfere with the external scheduler.
    If "None" was passed to the scheduler kwarg of GSTCVDask and there is
    no external scheduler, instantiate distributed.Client(), which
    defaults to LocalCluster, with n_workers=n_jobs and 1 thread per
    worker. If n_jobs is None, uses the defaults distributed.Client
    behavior when n_workers is set to None.

    If a scheduler is passed, this module does not perform any validation
    but allows that to be handled by dask at compute time.

    This module intentionally deviates from the dask_ml API, and disallows
    any shorthand methods for setting up a scheduler (such as strings
    like 'threading' and 'multiprocessing', which are ultimately passed
    to dask.base.get_scheduler.) All of these types of configurations
    should be handled by the user external to the GSTCVDask module. Given
    the evolving development of dask and dask_ml, the goal here has been,
    as much as possible, to allow dask and distributed objects to flow
    through without any hard-coded input.

    Prior to this (24_08_08) in cases where an external scheduler is not
    available and one is not passed, Client was instantiated with hard
    n_workers and threads_per_worker. The hard kwargs have been removed
    to operate with the dask defaults.

    ******
    From the dask_ml.GridSearchCV docs 24_08_08_07_56_00:
    scheduler: string, callable, Client, or None, default=None
    The dask scheduler to use. Default is to use the global scheduler if
    set, and fallback to the threaded scheduler otherwise. To use a
    different scheduler either specify it by name (either “threading”,
    “multiprocessing”, or “synchronous”), pass in a dask.distributed.Client,
    or provide a scheduler get function.
    ******


    Parameters
    ----------
    _scheduler:
        _scheduler to be validated and used for compute

    Return
    ------
    -
        _scheduler:
            validated, instantiated scheduler

    """


    if _scheduler is None:
    # if there is no hard scheduler...
        try:
            # ...try to get an existing client...
            get_client()
            # if a client is available (either an external context manager
            # or a default scheduler in an outer scope) let that scheduler
            # take precedence. set _scheduler to nullcontext for empty
            # internal context managers.
            _scheduler = contextlib.nullcontext()
        except ValueError:
            # ...if no external client and no hard scheduler (client),
            # create a new one
            _scheduler = Client(n_workers=_n_jobs, threads_per_worker=1)


    else:
    # if there is a hard scheduler, that supersedes all.
        pass


    return _scheduler




















