# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union



def _val_line_break(
    _line_break: Union[str, set[str], None]
) -> None:

    """
    Validate 'line_break'. Must be None, a non-empty string or a
    non-empty python set of non-empty strings.


    Parameters
    ----------
    _line_break:
        Union[str, set[str], None] - When passed as a single string,
        TextJustifier will start a new line immediately AFTER all
        occurrences of the character string sequence regardless of the
        number of characters in the line. When passed as a set of
        strings, TextJustifier will start a new line immediately after
        all occurrences of the character strings given. If None, do not
        force any line breaks. If the there are no string sequences in
        the data that match the given strings, then there are no forced
        line breaks. If a line_break string is in the middle of a
        sequence that might otherwise be expected to be contiguous, TJ
        will force a new line AFTER the line_break indiscriminately.
        Cannot be an empty string. Cannot be an empty set. No line_breaks
        can be identical and one cannot be a substring of another. No
        line_break can be identical to a sep entry and no line_break can
        be a substring of a sep.


    Return
    ------
    -
        None

    """


    if _line_break is None:
        return


    err_msg = (f"'line_break' must be a non-empty string or a non-empty "
               f"python set of non-empty strings. ")

    try:
        if not isinstance(_line_break, (str, set)):
            raise UnicodeError
        if isinstance(_line_break, str):
            if len(_line_break) == 0:
                raise TimeoutError
        elif isinstance(_line_break, set):
            if len(_line_break) == 0:
                raise MemoryError
            for _ in _line_break:
                if not isinstance(_, str):
                    raise UnicodeError
                if len(_) == 0:
                    raise TimeoutError
    except UnicodeError:
        raise TypeError(err_msg + f"Got {type(_line_break)}.")
    except TimeoutError:
        raise ValueError(err_msg + f"Got empty string.")
    except MemoryError:
        raise ValueError(err_msg + f"Got empty set.")
    except Exception as e:
        raise e






