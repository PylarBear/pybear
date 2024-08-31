# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



"""

    scheduler:
        distributed.Client, distributed.scheduler.Scheduler, or None,
        default=None -

        A passed scheduler supersedes all other external schedulers.
        When a scheduler is explicitly passed, GSTCVDask does not perform
        any validation or verification but allows that to be handled by
        dask at compute time.

        If "None" was passed to the scheduler kwarg (the default),
        GSTCVDask looks for an external context manager or global
        scheduler using get_client. If one exists, GSTCVDask uses that
        as the scheduler. If an external scheduler does not exist,
        GSTCVDask instantiates a multiprocessing distributed.Client()
        (which defaults to LocalCluster) with n_workers=n_jobs and 1
        thread per worker. If n_jobs is None, GSTCVDask uses the default
        distributed.Client behavior when n_workers is set to None.

        This module intentionally disallows any shorthand methods for
        internally setting up a scheduler (such as strings like 'thread-
        ing' and 'multiprocessing', which are ultimately passed to
        dask.base.get_scheduler.) All of these types of configurations
        should be handled by the user external to the GSTCVDask module.
        As much as possible, dask and distributed objects are allowed to
        flow through without any hard-coded input.


"""

