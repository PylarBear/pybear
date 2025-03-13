# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import re



def _val_sep(
    _sep: Union[str, re.Pattern]
) -> None:

    """
    Validate 'sep'. Must be a regexp pattern or a re.Pattern object.


    Parameters
    ----------
    _sep:
        Union[str, re.Pattern] - the regexp pattern that indicates to
        TJRE where it is allowed to wrap a line. If a pattern match is
        in the middle of a sequence that might otherwise be expected to
        be contiguous, TJRE will wrap a new line AFTER the pattern
        indiscriminately if proximity to the n_chars limit dictates to
        do so. This parameter is only validated by TJRE to be an instance
        of str or re.Pattern. TJRE does not assess the validity of the
        expression itself. Any exceptions would be raised by re.search.


    Return
    ------
    -
        None

    """


    err_msg = (f"'sep' must be a regexp string or re.Pattern object. ")


    if not isinstance(_sep, (str, re.Pattern)):
        raise TypeError(err_msg)








