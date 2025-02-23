# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from .._type_aliases import StrSepType



def _val_str_sep(
    _str_sep: StrSepType,
    _X: Sequence[str]
) -> None:

    """
    Validate the sep parameter for str.split(). Must be None, a single
    character string, a python set of character strings, or a python
    list of Nones, strings, sets of strings, and Falses. If passed as a
    list, the number of entries in the list must equal the number of
    strings in X.


    Parameters
    ----------
    _sep:
        StrSepType - the separator character(s) to split the strings in
        X on. If None, apply the default splitting for str.split() to
        all the strings in X. If a single character string, all strings
        in X are split on that single character string. If passed as a
        python set of strings, all the strings in X are split on all the
        values in the set. If passed as a python list of Nones, Falses,
        strings, or sets of strings, the number of entries must equal
        the number of strings in X, and each entry is applied to the
        corresponding string in X. The string entries themselves are not
        validated, any exceptions would be raised by str.split().


    Return
    ------
    -
        None


    Notes
    -----
    see str.split()

    """


    if _str_sep is None:
        return


    err_msg = (
        f"'str_sep' must be None, a single string, a python set of "
        f"strings, or a python list of Nones, Falses, strings, and sets "
        f"of strings. \nIf passed as a list, the number of entries must "
        f"equal the number of strings in X."
    )

    try:
        iter(_str_sep)
        if isinstance(_str_sep, str):
            raise UnicodeError
        if isinstance(_str_sep, set):
            raise NotImplementedError
        if isinstance(_str_sep, list):
            raise TimeoutError
        # if get to this point, not None, str, set[str], list[of the 4]
        raise Exception
    except UnicodeError:
        # if is a single string
        pass
    except NotImplementedError:
        # if is a set of strings
        if not all(map(isinstance, _str_sep, (str for _ in _str_sep))):
            raise TypeError(err_msg)
    except TimeoutError:
        # if is a list of Union[None, str, set[str], False]
        # --- len must match len(X)
        if len(_str_sep) != len(_X):
            raise ValueError(err_msg)
        for _ in _str_sep:
            if isinstance(_, (str, bool, type(None))):
                if _ is True:
                    raise TypeError(err_msg)
            elif isinstance(_, set):
                if not all(map(isinstance, _, (str for x in _))):
                    raise TypeError(err_msg)
            else:
                raise TypeError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)



    del err_msg









