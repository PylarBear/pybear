# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union

import numbers



def _val_maxsplit(
    _maxsplit: Union[numbers.Integral, Sequence[numbers.Integral], None],
    _X: Sequence[str]
) -> None:

    """
    Validate the maxsplit parameter.


    Parameters
    ----------
    _maxsplit:
        Union[numbers.Integral, Sequence[numbers.Integral], None] - the
        maximum number of splits to perform on each string in X. If a
        single integer, that integer is applied to all strings in X. If
        passed as a sequence, the number of integers must equal the
        number of strings in X, and the integers are applied to the
        corresponding string in X. The values are only validated for
        being integers; any expections raised beyond that are raised by
        str.split() or re.split().

    _X:
        Sequence[str] - the data.


    Return
    ------
    -
        None


    """


    if _maxsplit is None:
        return


    err_msg = (
        f"'maxsplit' must be a single integer or a sequence of integers "
        f"whose length equals the number of strings in X"
    )


    try:
        if isinstance(_maxsplit, numbers.Real):
            raise UnicodeError
        iter(_maxsplit)
        if isinstance(_maxsplit, (str, dict)):
            raise Exception
        raise TimeoutError
    except UnicodeError:
        # if is a single number
        if not isinstance(_maxsplit, numbers.Integral):
            raise TypeError(err_msg)
        if isinstance(_maxsplit, bool):
            raise TypeError(err_msg)
    except TimeoutError:
        # if is sequence
        if len(_maxsplit) != len(_X):
            raise ValueError(err_msg)
        if not all(map(isinstance, _maxsplit, (numbers.Real for _ in _maxsplit))):
            raise TypeError(err_msg)
        if not all(map(isinstance, _maxsplit, (numbers.Integral for _ in _maxsplit))):
            raise TypeError(err_msg)
        if any(map(isinstance, _maxsplit, (bool for _ in _maxsplit))):
            raise TypeError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)


    del err_msg


