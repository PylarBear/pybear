# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union



def _val_sep(
    _sep: Union[str, set[str]]
) -> None:

    """
    Validate 'sep'. Must be a non-empty string or a non-empty python set
    of non-empty strings.


    Parameters
    ----------
    _sep:
        Union[str, set[str]] - the non-empty character string sequence(s)
        that indicate to TextJustifier where it is allowed to wrap a
        line. When passed as a non-empty set of strings, TextJustifier
        will consider any of those non-empty strings as a place where it
        can wrap a line. TextJustifier processes all data in 1D form (as
        list of strings), with all data given as 2D converted to 1D.


    Return
    ------
    -
        None

    """


    err_msg = (f"'sep' must be a non-empty string or a non-empty python "
               f"set of non-empty strings. ")

    try:
        if not isinstance(_sep, (str, set)):
            raise UnicodeError
        if isinstance(_sep, str):
            if len(_sep) == 0:
                raise TimeoutError
        elif isinstance(_sep, set):
            if len(_sep) == 0:
                raise MemoryError
            for _ in _sep:
                if not isinstance(_, str):
                    raise UnicodeError
                if len(_) == 0:
                    raise TimeoutError
    except UnicodeError:
        raise TypeError(err_msg + f"Got {type(_sep)}.")
    except TimeoutError:
        raise ValueError(err_msg + f"Got empty string.")
    except MemoryError:
        raise ValueError(err_msg + f"Got empty set.")
    except Exception as e:
        raise e








