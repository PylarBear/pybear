# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union



def _val_sep(
    _sep: Union[None, str, set[str], list[Union[str, set[str], None]]],
    _X: Sequence[str]
) -> None:

    """
    Validate the sep parameter. Must be None, a single string, a python
    set of strings, or a python list of those three objects. If passed
    as a list of the three allowed types, the number of entries in the
    list must equal the number of strings in X.


    Parameters
    ----------
    _sep:
        Union[None, str, set[str], list[Union[None, str, set[str]]]] -
        the separator character(s) to split the strings in X on. If None,
        apply the default splitting for str.split() to all the strings
        in X. If a single character string, all strings in X are split
        on that single character string. If passed as a python set of
        strings, all the strings in X are split on all the values in
        the set. If passed as a python list of any or all of those three
        types, the number of entries must equal the number of strings in
        X, and each entry is applied to the corresponding string in X.
        The string entries themselves are not validated, any exceptions
        would be raised by str.split().


    Return
    ------
    -
        None


    Notes
    -----
    see str.split()

    """


    err_msg = (
        f"'sep' must be None, a single string, a python set of strings, "
        f"or a python list of Union[None, str, set[str]]. If passed as a "
        f"list, the number of entries must equal the number of strings "
        f"in X."
    )

    try:
        if _sep is None:
            raise UnicodeError
        iter(_sep)
        if isinstance(_sep, str):
            raise UnicodeError
        if isinstance(_sep, set):
            raise NotImplementedError
        if isinstance(_sep, list):
            raise TimeoutError
        # if get to this point, not None, str, set[str], list[of the 3]
        raise Exception
    except UnicodeError:
        # if is a single string or None
        pass
    except NotImplementedError:
        # if is a set of strings
        if not all(map(isinstance, _sep, (str for _ in _sep))):
            raise TypeError(err_msg)
    except TimeoutError:
        # if is a list of Union[None, str, set[str]]
        # --- len must match len(X)
        if len(_sep) != len(_X):
            raise ValueError(err_msg)
        # --- if no sets, just ensure are strings and/or None
        if all(map(isinstance, _sep, ((str, type(None)) for _ in _sep))):
            pass
        # --- but if has sets in it
        if not all(map(isinstance, _sep, ((str, type(None), set) for _ in _sep))):
            raise TypeError(err_msg)
        for __ in _sep:
            if isinstance(__, set) \
                    and not all(map(isinstance, __, (str for _ in __))):
                raise TypeError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)



    del err_msg









