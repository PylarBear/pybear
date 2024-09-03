# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import ParamsType




def _estimator(_params: ParamsType, _estimator_instance) -> None:

    """

    Validate estimator is a class instance that has fit, get_params,
    and set_params methods. Also serendipitously check if the parameter
    names in params are an attr of the estimator instance, if not, raise.

    """

    # verify methods exist
    if not hasattr(_estimator_instance, 'fit'):
        raise TypeError(f"'estimator' must have a 'fit' method")

    # GSTCV does not require estimator have score() method. if GSCV
    # 'scoring' is not None, then the estimator need not have a scoring
    # method.
    # if not hasattr(_estimator_instance, 'score'):
    #     raise TypeError(f"'estimator' must have a 'score' method")

    if not hasattr(_estimator_instance, 'get_params'):
        raise TypeError(f"'estimator' must have a 'get_params' method")

    if not hasattr(_estimator_instance, 'set_params'):
        raise TypeError(f"'estimator' must have a 'set_params' method")


    # verify params in _params are attrs of estimator
    # inspect.signature() does not recognize an instance, only the class
    BAD_PARAM_NAMES = []
    for _param in _params:
        if not hasattr(_estimator_instance, str(_param)):
            BAD_PARAM_NAMES.append(_param)

    if len(BAD_PARAM_NAMES) == 1:
        raise AttributeError(f"param '{', '.join(BAD_PARAM_NAMES)}' is "
            f"not an attribute or method of {type(_estimator_instance).__name__}. "
            f"Did you pass the estimator class and not an instance of the "
            f"class?")
    elif len(BAD_PARAM_NAMES) > 1:
        raise AttributeError(f"params {', '.join(BAD_PARAM_NAMES)} are "
            f"not attributes or methods of {type(_estimator_instance).__name__}. "
            f"\nDid you pass the estimator class and not an instance of the "
            f"class?")

    del BAD_PARAM_NAMES





