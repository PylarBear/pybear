# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from typing import Iterable
from typing_extensions import Union

import numbers

from .._validation._count_threshold import _val_count_threshold



def _threshold_listifier(
    _n_features_in: int,
    *_threshold: Union[int, Iterable[int]]
) -> Union[list[int], tuple[list[int], ...]]:

    """
    Return '_threshold' as list-like(s) of integers with number of entries
    equaling the number of features in the data. Any number of threshold
    values can be passed as positional arguments to be converted to a
    list, if not already a list. This module will return the number of
    threshold values that are passed to it.


    Parameters
    ----------
    _n_features_in:
        int - the number of features in the data.
    *_threshold:
        Union[int, Iterable[int]] - the threshold value(s) to be
        converted to list[int]. Any number of threshold values can be
        passed as positional arguments.


    Return
    ------
    -
        _threshold_lists: Union[list[int], tuple[list[int], ...]] -
        a single list[int] or a tuple of list[int]s that indicate the
        threshold for each feature in the data.


    """


    _threshold_lists = []
    for _threshold_entry in _threshold:

        # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        # _n_features_in is validated by _val_count_threshold

        _val_count_threshold(
            _threshold_entry,
            ['int', 'Iterable[int]'],
            _n_features_in
        )

        # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        if isinstance(_threshold_entry, numbers.Integral):
            _threshold_lists.append(
                [int(_threshold_entry) for _ in range(_n_features_in)]
            )
        else:
            _threshold_lists.append(list(map(int, _threshold_entry)))


    if len(_threshold_lists) == 1:
        return _threshold_lists[0]
    else:
        return tuple(_threshold_lists)









