# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import CountThresholdType

import numbers
import numpy as np



def _val_count_threshold(
    _count_threshold: CountThresholdType,
    _n_features_in: int
) -> None:

    """
    Validate count_threshold is non-bool integer >= 2, or a 1D list-like
    of non-bool integers >= 1 with at least one value >= 2 and length
    that equals '_n_features_in'.


    Parameters
    ----------
    _count_threshold:
        Union[int, Iterable[int]]: the minimum frequency a value must
        have within a column in order to not be removed.
    _n_features_in:
        int: the number of features in the data.


    Return
    ------
    -
        None


    """

    err_msg1 = (f"\nwhen 'count_threshold' is passed as a single integer it "
               f"must be >= 2. ")

    try:
        float(_count_threshold)
        if isinstance(_count_threshold, bool):
            raise MemoryError
        if not int(_count_threshold) == _count_threshold:
            raise IndexError
        if not _count_threshold >= 2:
            raise IndexError
        # if get to this point, we are good to go as integer
        return
    except MemoryError:
        # # if MemoryError, means bad number
        raise TypeError(err_msg1)
    except IndexError:
        # if IndexError, means bad integer
        raise ValueError(err_msg1)
    except:
        # if not MemoryError or IndexError, excepted for not number,
        # must be Iterable to pass, but could be other junk
        pass


    err_msg2 = (
        f"\nwhen 'count_threshold' is passed as an iterable it must be a 1D "
        f"list-like of integers where any value can be >= 1, but at least "
        f"one value must be >= 2. \nthe length of the iterable also must "
        f"match the number of features in the data. "
    )

    try:
        iter(_count_threshold)
        if isinstance(_count_threshold, (str, dict)):
            raise UnicodeError
        _count_threshold = np.array(_count_threshold)
        if len(_count_threshold.shape) != 1:
            raise MemoryError
        if len(_count_threshold) != _n_features_in:
            raise MemoryError
        if not all(map(
            isinstance, _count_threshold, (numbers.Integral for _ in _count_threshold)
        )):
            raise MemoryError
        if not all(map(lambda x: x >= 1, _count_threshold)):
            raise MemoryError
        if not any(map(lambda x: x >= 2, _count_threshold)):
            raise MemoryError
    except UnicodeError:
        # if UnicodeError is str or dict iterable
        raise TypeError(err_msg2)
    except MemoryError:
        # if MemoryError, then bad shape or not integers, not all >=1, not one >= 2
        raise ValueError(err_msg2)
    except:
        # if not UnicodeError or MemoryError, then is not a float and is not iterable
        raise TypeError(err_msg1 + err_msg2 + f"got {type(_count_threshold)}.")











