# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union

import numbers



def _condition_sep(
    _sep: Union[str, Sequence[str]],
    _n_rows: numbers.Integral
) -> list[str]:
    """Condition the `sep` parameter into a Python list of strings whose
    length equals the number of rows in X.

    If already a 1D sequence of strings, simply return.

    Parameters
    ----------
    _sep : Union[str, Sequence[str]]
        The string sequence(s) to use to join each row of text strings
        in the data.
    _n_rows : numbers.Integral
        The number of sub-containers of text in the data.

    Returns
    -------
    seps : list[str]
        A single Python list of strings holding `sep` for each row in
        the data.

    """


    if isinstance(_sep, str):
        return [_sep for _ in range(_n_rows)]
    else:
        # must be sequence of str
        return list(_sep)





