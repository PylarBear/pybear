# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import re



def _val_replace_always(
    _REPLACE_ALWAYS: None | dict[str | re.Pattern[str], str]
) -> None:
    """Validate `REPLACE_ALWAYS`.

    Must be None or a dictionary with strings and/or re.compile objects
    as keys and strings as values. Cannot be empty.

    Parameters
    ----------
    _REPLACE_ALWAYS : None | dict[str | re.Pattern[str], str]
        A non-empty dictionary with strings and/or re.compile objects for
        keys and strings for values. When the key is match against a
        word in the text then the respective value is put in place of
        the word in the text body.

    Raises
    ------
    ValueError
    TypeError

    Returns
    -------
    None

    """


    if _REPLACE_ALWAYS is None:
        return

    _err_msg = (f"'REPLACE_ALWAYS' must be None or a dictionary with keys as "
                f"strings and/or re.compile objects and values as strings.")

    _addon = ""


    try:
        if not isinstance(_REPLACE_ALWAYS, dict):
            raise Exception
        if len(_REPLACE_ALWAYS) == 0:
            _addon = "Got empty."
            raise UnicodeError
        if not all(map(
            isinstance,
            _REPLACE_ALWAYS.keys(),
            ((str, re.Pattern) for _ in _REPLACE_ALWAYS)
        )):
            _addon = f"Got bad key."
            raise Exception
        if not all(map(
            isinstance,
            _REPLACE_ALWAYS.values(),
            (str for _ in _REPLACE_ALWAYS.values())
        )):
            _addon = f"Got bad value."
            raise Exception
    except UnicodeError:
        raise ValueError(_err_msg + _addon)
    except Exception as e:
        raise TypeError(_err_msg + _addon)




