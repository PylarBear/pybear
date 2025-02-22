# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union

import re



def _val_regexp(
    _regexp: Union[str, re.Pattern, Sequence[Union[str, re.Pattern]], None],
    _X: Sequence[str]
) -> None:

    """
    Validate the regexp parameter. Must be None, a single regexp string,
    a re.Pattern object, or a sequence of Union[str, re.Pattern]. If
    passed as a sequence, the number of entries must equal the number of
    strings in X.


    Parameters
    ----------
    _regexp:
        Union[str, re.Pattern, Sequence[Union[str, re.Pattern]], None] -
        the regular expression string or the re.Pattern object to split
        the strings in X on. If a single expression or pattern, that is
        applied to all strings in X. If passed as a sequence, the number
        of entries must equal the number of strings in X, and the entries
        are applied to the corresponding string in X. The regular
        expressions and the re.Pattern objects themselves are not
        validated for legitimacy, any exceptions would be raised by
        re.split().


    Return
    ------
    -
        None


    Notes
    -----
    see re.split()

    """


    if _regexp is None:
        return


    err_msg = (
        f"'regexp' must be None, a single regular expression, a "
        f"re.Pattern object, or a sequence of Union[str, re.Pattern]. If "
        f"passed as a sequence, the number of entries must equal the "
        f"number of strings in X."
    )

    try:
        if isinstance(_regexp, re.Pattern):
            raise UnicodeError
        iter(_regexp)
        if isinstance(_regexp, dict):
            raise Exception
        if isinstance(_regexp, str):
            raise UnicodeError
        raise TimeoutError
    except UnicodeError:
        # if is a single string or re.Pattern
        pass
    except TimeoutError:
        # if is a sequence of somethings
        if len(_regexp) != len(_X):
            raise ValueError(err_msg)
        if not all(map(isinstance, _regexp, ((str, re.Pattern) for _ in _regexp))):
            raise TypeError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)



    del err_msg









