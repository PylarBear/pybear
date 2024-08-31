# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



"""

    n_jobs:
        Union[int, None], default=None - Active only if no scheduler
        is available. That is, if a scheduler is not passed to the
        scheduler kwarg, if no global scheduler is available, and if
        there is no scheduler context manager, only then does n_jobs
        become effectual. In this case, GSTCVDask creates a distributed
        Client multiprocessing instance with n_workers=n_jobs.



"""



