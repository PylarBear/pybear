# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union



def _val_sep(
    _sep: Union[str, Sequence[str]]
) -> None:

    """
    Validate sep. Must be string or a 1D sequence of strings. Neither
    can be empty.


    Parameters
    ----------
    _sep:
        Union[str, Sequence[str]] - the string sequence(s) to split on.


    Return
    ------
    -
        None


    """



    try:
        iter(_sep)
        if isinstance(_sep, dict):
            raise Exception
        if len(_sep) == 0:
            raise Exception
        if isinstance(_sep, str):
            raise UnicodeError
        if not all(map(isinstance, _sep, (str for _ in _sep))):
            raise Exception
    except UnicodeError:
        pass
    except Exception as e:
        raise TypeError(
            f"'sep' must be a string or sequence of strings. neither can "
            f"be empty."
        )



        # pizza a sep cannot be equal to or be a substring of another sep









