# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Union



def _validate_pre_dispatch(
        _pre_dispatch: Union[None, int, str]
    ) -> Union[None, int, str]:

    """
    THIS IS VERBATIM FROM SKLEARN GSCV DOCS 24_07_07_08_26_00

    Controls the number of jobs that get dispatched during parallel
    execution. Reducing this number can be useful to avoid an explosion
    of memory consumption when more jobs get dispatched than CPUs can
    process. This parameter can be:
    1) None, in which case all the jobs are immediately created and
    spawned. Use this for lightweight and fast-running jobs, to avoid
    delays due to on-demand spawning of the jobs
    2) int, giving the exact number of total jobs that are spawned
    3) str, giving an expression as a function of n_jobs, as in ‘2*n_jobs’

    Parameters
    ----------
    _pre_dispatch: None, int, or str, default=’2*n_jobs’ - the number of
        jobs that get dispatched during parallel execution

    Return
    ------
    -
        _pre_dispatch - None, int, or str, default=’2*n_jobs’

    """

    err_msg = (f"'pre_dispatch' must be None, an integer >= 1, or a string "
        f"giving an expression as a function of n_jobs, as in ‘2*n_jobs’")

    if _pre_dispatch is None:
        pass

    elif isinstance(_pre_dispatch, str):
        n_jobs = 4
        try:
            eval(_pre_dispatch)
        except:
            raise ValueError(err_msg)

    else:
        try:
            float(_pre_dispatch)
            if isinstance(_pre_dispatch, bool):
                raise UnicodeError
            if int(_pre_dispatch) != float(_pre_dispatch):
                raise BrokenPipeError
            _pre_dispatch = int(_pre_dispatch)
            if _pre_dispatch < 1:
                raise BrokenPipeError
        except UnicodeError:
            raise TypeError(err_msg)
        except BrokenPipeError:
            raise ValueError(err_msg)
        except:
            raise TypeError(err_msg)




    return _pre_dispatch







