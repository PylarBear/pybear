# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


from typing import Iterable

import numpy as np


def arg_kwarg_validater(arg: any,
                        name: str,
                        allowed: Iterable[any],
                        module: str,
                        function: str,
                        return_if_none: any=None
    ) -> any:

    """ Validate a parameter, param: arg, against allowable entries as
    listed in param: allowed. An exception is raised if there is no match.
    Integers and floats require exact matching. Strings, however, are not
    case sensitive. Strings are searched against param: allowed in a non-
    case-sensitive manner and if there is a match, the entry as it is in
    param: allowed is returned, not the original param: arg (unless they
    were already equivalent.) If an array-like is passed to param: arg,
    each term in the array-like is compared against param: allowed,
    subject to the same rules stated previously. If a single term in the
    array-like is not in the allowable terms, an exception is raised; all
    terms in the array-like must be in param: allowed.

    Parameters
    ----------
    arg:
        any - the parameter to be validated against allowed entries
    name:
        str - deprecated - the name of the parameter
    allowed:
        array-like - the list of allowed entries
    module:
        str - deprecated - the name of the calling module
    function:
        str - deprecated - the name of the calling function
    return_if_none:
        any, default = None - value to return if param: arg passed
        validation and is a None-type.

    Return
    ------
    arg -
        arg: validated arg

    Notes
    ----
    # PIZZA
    # pybear Note 24_04_13_18_03_00 - 'name', 'module' and 'function' are
    # relics and are not currently used inside this function. The
    # arguments remain in place because of the numerous dependents on
    # this funciton. Reinstitute them if more clarity in tracebacks is
    # needed again.


    Examples
    -------
    >>> ARGS = ('my_param', [1, 2, 3, None], 'some_module', 'calling_function')
    >>> out1 = arg_kwarg_validater(3, *ARGS)
    >>> out1
    3

    >>> out3 = arg_kwarg_validater(None, *ARGS, return_if_none=72)
    >>> out3
    72

    """


    if not isinstance(name, str):
     raise TypeError(f"'name_as_str' must be a str")

    if not isinstance(module, str):
     raise TypeError(f"'module_name_as_str' must be a str")

    if not isinstance(function, str):
     raise TypeError(f"'function_name_as_str' must be a str")

    try:
        iter(allowed)
        if isinstance(allowed, (dict, str)):
            raise Exception
    except:
        raise TypeError(f"'allowed' must be an array-like")

    try:
        allowed = np.array(list(allowed), dtype=object).ravel()
    except:
        raise TypeError(f"'allowed' must be an array-like that can be converted"
                        f"to a numpy array")

    if len(allowed)==0:
        raise ValueError(f"'allowed' cannot be empty")

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    err_msg = f"'{arg}' is not in allowed, must be {', '.join(map(str, allowed))}"

    if isinstance(arg, str):
        for item in allowed:
            try:
                if arg.upper() == item.upper():
                    arg = item
                    break
            except:
                pass
        else:
            raise ValueError(err_msg)

    elif isinstance(arg, dict):
        raise TypeError("arg_kwarg_validater cannot validate dictionaries")

    elif callable(arg):
        raise TypeError("arg_kwarg_validater cannot validate callables")

    elif isinstance(arg, type(None)):
        if arg not in allowed:
            raise ValueError(err_msg)

    elif isinstance(arg, bool):
        # if True in [0, 1] returns True, so does False, must do it out
        for __ in allowed:
            if arg is __:
                break
        else:
            raise ValueError(err_msg)

    elif any([_ in str(type(arg)).upper() for _ in ('INT','FLOAT')]):
        # if 0 in [True, False] returns True, so does 1, must do it out
        for __ in allowed:
            if isinstance(__, bool) and arg is __:
                break

            if not isinstance(__, bool) and arg == __:
                break
        else:
            raise ValueError(err_msg)

    elif isinstance(arg, (np.ndarray, list, set, tuple)):
        arg = np.array(list(arg), dtype=object).ravel()
        for idx, item in enumerate(arg):

            try:
                iter(item)
                raise TypeError(f"arg_kwarg_validater cannot validate iterables "
                    f"directly, only the contents of iterables can be validated "
                    f"(iterable 'arg' and 'allowed' must be one iterable deep)")
            except:
                pass

            arg[idx] = arg_kwarg_validater(
                    item,
                    name,
                    allowed,
                    module,
                    function,
                    return_if_none=return_if_none
            )
    else:
        raise TypeError(f"arg_kwarg_validater CANNOT VALIDATE (kw)arg '{arg}'")

    del err_msg

    if arg is None:
        return return_if_none
    else:
        return arg






