# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import inspect
from sklearn.pipeline import Pipeline



def check_pipeline(
    pipeline: Pipeline
) -> None:

    """
    Validate a pipeline setup, in particular, the construction of the
    steps attribute. Validate that 'steps' is a list of tuples. In the
    first position of each tuple must be a string. The second position
    of each tuple must contain a class instance that has a 'fit' method.


    Parameters
    ----------
    pipeline:
        sklearn.pipeline.Pipeline - a Pipeline instance.


    Return
    ------
    -
        None


    """

    if inspect.isclass(pipeline):
        raise ValueError(f"'pipeline' must be an instance, not the class")



    err_msg = (f"pipeline steps must be in the format "
               f"[(str1, cls1()), (str2, cls2()), ...]")

    _steps = pipeline.steps


    if not isinstance(_steps, list):
        raise ValueError(err_msg)


    if not all(map(isinstance, _steps, (tuple for _ in _steps))):
        raise ValueError(err_msg)


    if len(_steps) == 0:
        raise ValueError(f"estimator pipeline has empty steps")


    for step in _steps:

        if len(step) != 2:
            raise ValueError(err_msg)
        if not isinstance(step[0], str):
            raise ValueError(err_msg)
        if not hasattr(step[1], 'fit'):
            raise ValueError(err_msg)

        try:
            step[1]()
            raise UnicodeError
        except UnicodeError:
            raise ValueError(err_msg)
        except:
            pass


    return

