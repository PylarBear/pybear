# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable, Callable
from typing_extensions import Union, TypeAlias

AllFunc: TypeAlias = Callable[[Iterable], bool]
AnyFunc: TypeAlias = Callable[[Iterable], bool]


def _is_fitted(
    estimator,
    attributes: Union[str, Iterable[str], None]=None,
    all_or_any: Union[AllFunc, AnyFunc]=all
):

    """
    Determine if an estimator/transformer is fitted and return a boolean.
    'True' means fitted and 'False' means not fitted.

    This algorithm looks for 3 things, in the presented order.

    The estimator/transformer is fitted if it:\\\n
    1) has a __pybear_is_fitted__ dunder method and it returns boolean
    True\\\n
    2) has any or all attributes given by :param: 'attributes', if it is
    passed; if not passed, this step is skipped\\\n
    3) has an attribute that ends with an underscore and does not start
    with double underscore.\\\n


    Parameters
    ----------
    estimator:
        estimator/transformer instance - Estimator/transformer instance
        for which the check is performed.

    attributes:
        Union[str, Iterable[str], None], default=None - Attribute name(s)
        given as string or a list/tuple of strings Eg.: "coef_" or
        ["coef_", "estimator_", ...]

    all_or_any : callable, {all, any}, default=all
        Specifies whether all or any of the given attributes must exist.


    Returns
    -------
    -
        fitted : bool - Whether the estimator/transformer is fitted.


    Examples
    --------
    >>> from pybear.utilities._is_fitted import _is_fitted
    >>> from pybear.preprocessing import InterceptManager as IM
    >>> trf = IM()
    >>> _is_fitted(trf)
    False
    >>> import numpy as np
    >>> X = np.random.uniform(0, 1, (5,3))
    >>> trf.fit(X)
    InterceptManager()
    >>> _is_fitted(trf)
    True


    """


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    if not hasattr(estimator, 'fit'):
        raise ValueError(
            f"'estimator' must be a valid estimator or transformer that at "
            f"least has a 'fit' method."
        )

    try:
        if isinstance(attributes, (str, type(None))):
            raise UnicodeError
        iter(attributes)
        if isinstance(attributes, dict):
            raise Exception
        if not all(map(isinstance, attributes, (str for _ in attributes))):
            raise Exception
    except UnicodeError:
        pass
    except:
        raise ValueError(
            f"'attributes' must be a string, Iterable[str], or None"
        )

    if not (all_or_any is all or all_or_any is any):
        raise ValueError(
            f"'all_or_any' must be python built-in function 'all' or "
            f"python built-in function 'any'."
        )

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    if hasattr(estimator, "__pybear_is_fitted__"):
        return estimator.__pybear_is_fitted__()

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        return all_or_any([hasattr(estimator, attr) for attr in attributes])

    fitted_attrs = [
        v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
    ]
    return len(fitted_attrs) > 0













