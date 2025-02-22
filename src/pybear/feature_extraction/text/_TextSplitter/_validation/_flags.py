# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union

import numbers



def _val_flags(
    _flags: Union[numbers.Integral, Sequence[numbers.Integral], None],
    _X: Sequence[str]
) -> None:

    """
    Validate the 'flags' parameter for re.split().


    Parameters
    ----------
    _flags:
         Union[numbers.Integral, Sequence[numbers.Integral], None] - the
         flags arguments for re.split(), if regular expressions are being
         used. Must be None, an instance of numbers.Integral, or a
         sequence of such integers whose sequence length must match the
         number of strings in the data. The values of the integers are
         not validated for legitimacy, any exceptions would be raised by
         re.split().


    Return
    ------
    -
        None

    """


    if _flags is None:
        return


    err_msg = (
        f"'flags' must be None, a single integer, or a sequence of "
        f"integers whose length matches the number of strings in X."
    )

    try:
        if isinstance(_flags, numbers.Integral):
            raise UnicodeError
        iter(_flags)
        if isinstance(_flags, (str, dict)):
            raise Exception
        if len(_flags) != len(_X):
            raise TimeoutError
        if not all(map(isinstance, _flags, (numbers.Integral for _ in _flags))):
            raise Exception
        if any(map(isinstance, _flags, (bool for _ in _flags))):
            raise Exception
    except UnicodeError:
        # if a single integer
        if isinstance(_flags, bool):
            raise TypeError(err_msg)
    except TimeoutError:
        raise ValueError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)







