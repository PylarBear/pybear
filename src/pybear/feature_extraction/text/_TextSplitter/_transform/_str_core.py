# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import numbers

import numpy as np

from pybear.feature_extraction.text._TextSplitter._type_aliases import (
    StrSepType,
    StrMaxSplitType
)



def _str_core(
    _X: Sequence[str],
    _str_sep: StrSepType,
    _str_maxsplit: StrMaxSplitType
) -> list[list[str]]:

    """
    Split the strings in X based on the criteria in _str_sep.


    Parameters
    ----------
    _X:
        Sequence[str] - the data.
    _str_sep:
        Union[str, None, set[str]] - the splitting criteria for
        str.split().


    Return
    ------
    -
        list[list[str]] - the split data.

    """


    # get the False mask -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # if more than 1 parameter was passed as list we know the Falses
    # match from validation.

    if isinstance(_str_sep, list):
        _false_mask = np.array(list(map(lambda x: x is False, _str_sep)))
    elif isinstance(_str_maxsplit, list):
        _false_mask = np.array(list(map(lambda x: x is False, _str_maxsplit)))
    else:
        _false_mask = np.zeros(len(_X)).astype(bool)
    # END get the False mask -- -- -- -- -- -- -- -- -- -- -- -- -- --


    # turn _str_sep into list -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if isinstance(_str_sep, list):
        # can be None, str, set[str], False inside
        _seps = _str_sep
    elif isinstance(_str_sep, str):
        _seps = [_str_sep for _ in range(len(_X))]
    elif isinstance(_str_sep, set):
        _seps = [_str_sep for _ in range(len(_X))]
    elif isinstance(_str_sep, type(None)):
        _seps = [_str_sep for _ in range(len(_X))]
    else:
        raise TypeError(f"unexpected str_sep type {type(_str_sep)}")

    # map the Falses in case this one wasnt passed as a list but another was
    if np.any(_false_mask):
        _seps = np.array(_seps, dtype=object)
        _seps[_false_mask] = False
        _seps = _seps.tolist()
    # END turn _str_sep into list -- -- -- -- -- -- -- -- -- -- -- -- --


    # turn _str_maxsplit into list -- -- -- -- -- -- -- -- -- -- -- --
    if isinstance(_str_maxsplit, list):
        # can be None, integer, or False inside
        _maxsplits = _str_maxsplit
    elif isinstance(_str_maxsplit, numbers.Integral):
        _maxsplits = [_str_maxsplit for _ in range(len(_X))]
    elif isinstance(_str_maxsplit, type(None)):
        _maxsplits = [_str_maxsplit for _ in range(len(_X))]
    else:
        raise TypeError(f"unexpected str_maxsplit type {type(_str_maxsplit)}")

    # map the Falses in case this one wasnt passed as a list but another was
    if np.any(_false_mask):
        _maxsplits = np.array(_maxsplits, dtype=object)
        _maxsplits[_false_mask] = False
        _maxsplits = _maxsplits.tolist()
    # END turn _str_maxsplit into list -- -- -- -- -- -- -- -- -- -- --

    for _idx, _str in enumerate(_X):

        if _seps[_idx] is False:
            # even though it is not split, it still needs to go from str
            # to list[str]
            _X[_idx] = [_str]
            continue


        if isinstance(_seps[_idx], set):
            # need to do this the long way because of complications with
            # replace, split, and maxsplit. cant just replace all instances
            # of things in the set and then split on them when maxsplit
            # is specified.
            ctr = 0
            _last_idx = 0
            _X[_idx] = []
            for _char_idx, _char in enumerate(_str):
                if _char in _seps[_idx]:
                    _X[_idx].append(_str[_last_idx:_char_idx])
                    _last_idx = _char_idx + 1
                    ctr += 1
                    if _maxsplits[_idx] is not None and ctr == _maxsplits[_idx]:
                        _X[_idx].append(_str[_last_idx:])
                        break
            else:
                _X[_idx].append(_str[_last_idx:])

            del ctr, _last_idx, _char_idx, _char
        else:
            # must be str or None

            if _maxsplits[_idx] is None:
                _split_kwargs = {}
            else:
                # must be int
                _split_kwargs = {'maxsplit': _maxsplits[_idx]}

            _X[_idx] = _str.split(
                _seps[_idx],
                **_split_kwargs
            )


    del _false_mask, _seps, _maxsplits


    return _X








