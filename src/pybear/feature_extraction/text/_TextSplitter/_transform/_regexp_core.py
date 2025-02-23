# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import numbers
import re

import numpy as np

from pybear.feature_extraction.text._TextSplitter._type_aliases import (
    RegExpSepType,
    RegExpMaxSplitType,
    RegExpFlagsType
)



def _regexp_core(
    _X: Sequence[str],
    _regexp_sep: RegExpSepType,
    _regexp_maxsplit: RegExpMaxSplitType,
    _regexp_flags: RegExpFlagsType
) -> list[list[str]]:

    """
    Split the strings in X based on the criteria in _regexp_sep.


    Parameters
    ----------
    _X:
        Sequence[str] - the data.
    _regexp_sep:
        RegExpSepType - the regular expression or re.compile object that
        determines the splits.
    _regexp_maxsplit:
        RegExpMaxSplitType - the maximum number of splits to do on each
        line, working from left to right.
    _regexp_flags:
        RegExpFlagsType - modifiers for the split criteria.


    Return
    ------
    -
        list[list[str]] - the split data.

    """


    # get the False mask -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # if more than 1 parameter was passed as list we know the Falses
    # match from validation.

    if isinstance(_regexp_sep, list):
        _false_mask = np.array(list(map(lambda x: x is False, _regexp_sep)))
    elif isinstance(_regexp_maxsplit, list):
        _false_mask = np.array(list(map(lambda x: x is False, _regexp_maxsplit)))
    elif isinstance(_regexp_flags, list):
        _false_mask = np.array(list(map(lambda x: x is False, _regexp_flags)))
    else:
        _false_mask = np.zeros(len(_X)).astype(bool)
    # END get the False mask -- -- -- -- -- -- -- -- -- -- -- -- -- --


    # turn _regexp_sep into list -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if isinstance(_regexp_sep, list):
        # can be str, re.compile, False inside
        _seps = _regexp_sep
    elif isinstance(_regexp_sep, str):
        _seps = [_regexp_sep for _ in range(len(_X))]
    elif isinstance(_regexp_sep, re.Pattern):
        _seps = [_regexp_sep for _ in range(len(_X))]
    else:
        raise TypeError(f"unexpected regexp_sep type {type(_regexp_sep)}")

    # map the Falses in case this one wasnt passed as a list but another was
    if np.any(_false_mask):
        _seps = np.array(_seps, dtype=object)
        _seps[_false_mask] = False
        _seps = _seps.tolist()
    # END turn _regexp_sep into list -- -- -- -- -- -- -- -- -- -- -- -- --


    # turn _regexp_maxsplit into list -- -- -- -- -- -- -- -- -- -- -- --
    if isinstance(_regexp_maxsplit, list):
        # can be None, integer, or False inside
        _maxsplits = _regexp_maxsplit
    elif isinstance(_regexp_maxsplit, numbers.Integral):
        _maxsplits = [_regexp_maxsplit for _ in range(len(_X))]
    elif isinstance(_regexp_maxsplit, type(None)):
        _maxsplits = [_regexp_maxsplit for _ in range(len(_X))]
    else:
        raise TypeError(f"unexpected regexp_maxsplit type {type(_regexp_maxsplit)}")

    # map the Falses in case this one wasnt passed as a list but another was
    if np.any(_false_mask):
        _maxsplits = np.array(_maxsplits, dtype=object)
        _maxsplits[_false_mask] = False
        _maxsplits = _maxsplits.tolist()
    # END turn _regexp_maxsplit into list -- -- -- -- -- -- -- -- -- -- --


    # turn _regexp_flags into list -- -- -- -- -- -- -- -- -- -- -- --
    if isinstance(_regexp_flags, list):
        # can be None, integer, or False inside
        _flags = _regexp_flags
    elif isinstance(_regexp_flags, numbers.Integral):
        _flags = [_regexp_flags for _ in range(len(_X))]
    elif isinstance(_regexp_flags, type(None)):
        _flags = [_regexp_flags for _ in range(len(_X))]
    else:
        raise TypeError(f"unexpected str_maxsplit type {type(_regexp_flags)}")

    # map the Falses in case this one wasnt passed as a list but another was
    if np.any(_false_mask):
        _flags = np.array(_flags, dtype=object)
        _flags[_false_mask] = False
        _flags = _flags.tolist()
    # END turn _regexp_flags into list -- -- -- -- -- -- -- -- -- -- --


    for _idx, _str in enumerate(_X):

        if _seps[_idx] is False:
            # even though it is not split, it still needs to go from str
            # to list[str]
            _X[_idx] = [_str]
            continue

        if _maxsplits[_idx] is None:
            _split_kwargs = {}
        else:
            # must be int
            _split_kwargs = {'maxsplit': _maxsplits[_idx]}

        if _flags[_idx] is None:
            _split_kwargs |= {}
        else:
            # must be int
            _split_kwargs |= {'flags': _flags[_idx]}

        if isinstance(_seps[_idx], str):  # must be a regexp

            _X[_idx] = re.split(
                _seps[_idx],
                _str,
                **_split_kwargs
            )

        else:
            # must be re.compile

            _X[_idx] = re.split(
                _seps[_idx],
                _str,
                **_split_kwargs
            )


    del _false_mask, _seps, _maxsplits, _flags


    return _X





