# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from ..base._is_fitted import is_fitted
from ..exceptions import NotFittedError

from inspect import isclass

from typing import Iterable, Callable
from typing_extensions import Union, TypeAlias


AllFunc: TypeAlias = Callable[[Iterable], bool]
AnyFunc: TypeAlias = Callable[[Iterable], bool]



def check_is_fitted(
    estimator,
    attributes:Union[str, Iterable[str], None]=None,
    *,
    msg: Union[str, None]=None,
    all_or_any: Union[AllFunc, AnyFunc]=all
):

    """
    Perform _is_fitted validation on an estimator/transformer.

    Checks if the estimator/transformer is fitted by looking for 3
    things, in the presented order, via the pybear _is_fitted function.

    The estimator/transformer is fitted if it:\\\n
    1) has a __pybear_is_fitted__ dunder method and it returns boolean
    True\\\n
    2) has any or all attributes given by :param: 'attributes', if it is
    passed; if not passed, this step is skipped\\\n
    3) has an attribute that ends with an underscore and does not start
    with double underscore.\\\n

    If none of these things are true, the estimator/transformer is not
    fitted and raises a :class: pybear.exceptions.NotFittedError with
    the message given by :param: msg or the default message if :param:
    msg is not passed.


    Parameters
    ----------
    estimator:
        estimator/transformer instance - Estimator/tranformer instance
        for which the validation is performed.

    attributes : Union[str, Iterable[str], None], default=None -
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: "coef_" or ["coef_", "estimator_", ...].

    msg : str, default=None
        The default error message is, "This {estimator} instance is
        not fitted yet. Call 'fit' with appropriate arguments before
        using this estimator."

        For custom messages if "%(name)s" is present in the message
        string, it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.


    Raises
    ------
    ValueError
        If the passed estimator/transformer is invalid or is valid but
        is not instantiated.
        If the value passed to :param: attributes is invalid.
        If the value passed to :param: msg is not a string or None.
        If the function passed to :param: all_or_any is not one of the
        built-in python all() or any() functions.

    NotFittedError
        If the estimator/transformer fails all 3 checks for being fit.


    Examples
    --------
    >>> import numpy as np
    >>> from pybear.preprocessing import InterceptManager as IM
    >>> from pybear.utilities import check_is_fitted
    >>> from pybear.exceptions import NotFittedError
    >>> trf = IM()
    >>> try:
    ...     check_is_fitted(trf)
    ... except NotFittedError as exc:
    ...     print(f"Model is not fitted yet.")
    Model is not fitted yet.
    >>> X = np.random.randint(0,4,(2,2))
    >>> y = np.random.randint(0,2,(2,))
    >>> trf.fit(X, y)
    InterceptManager()
    >>> check_is_fitted(trf)

    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    if isclass(estimator):
        raise ValueError(f"{estimator} is a class, not an instance.")

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

    if not isinstance(msg, (str, type(None))):
        raise ValueError(
            f":param: 'msg' must be a string or None"
        )

    if not (all_or_any is all or all_or_any is any):
        raise ValueError(
            f"'all_or_any' must be python built-in function 'all' or "
            f"python built-in function 'any'."
        )

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **



    default_msg = (
        f"This {type(estimator).__name__} instance is not fitted yet. "
        f"Call 'fit' with appropriate arguments before using this "
        f"estimator."
    )

    if not is_fitted(estimator, attributes, all_or_any):
        raise NotFittedError(msg or default_msg)













