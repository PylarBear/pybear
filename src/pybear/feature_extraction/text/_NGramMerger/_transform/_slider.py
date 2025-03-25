# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable, Sequence
from typing_extensions import Union

import re



def _slider(
    _line: list[str],
    _ngram: Sequence[Union[str, re.Pattern]],
    _ngcallable: Union[Callable[[Sequence[str]], str], None],
    _sep: Union[str, None]
) -> list[str]:

    """
    Slide along a sequence of strings backwards looking for matches
    against an n-gram pattern. When one is found, replace the words with
    the contiguous string mapped from the words.

    Merge ngrams that match ngram patterns using the following hierarchy:

    by the given callable

    by the given separator

    by the default separator


    Parameters
    ----------
    _line:
        list[str] - A single row index from the data.
    _ngram:
        Sequence[Union[str, re.Pattern]] - A single n-gram sequence
        containing string literals and/or re.Pattern objects that
        specify an n-gram pattern. Cannot be empty.
    _ngcallable:
        Union[Callable[[Sequence[str]], str], None] - the callable
        applied to sequences that match an n-gram pattern to produce a
        single contiguous string.
    _sep:
        Union[str, None] - the user defined separator to join the words
        with. If no separator is defined by the user, use the pybear
        default separator.


    Returns
    -------
    -
        list[str] - the sequence of strings with all matching n-gram
        patterns replaced with contiguous strings.


    """


    # validation wont allow empty ngram
    # but there may be empty lines
    _n_len = len(_ngram)

    if _n_len > len(_line):
        return _line


    _sep = _sep or '_'

    if _ngcallable is None:
        _ngcallable = lambda _matches: _sep.join(_matches)


    _idx = len(_line) - _n_len
    while _idx >= 0:

        _block = _line[_idx: _idx + _n_len]

        # _sp = sub_pattern
        if all(re.fullmatch(_sp, _word) for _sp, _word in zip(_ngram, _block)):
            del _line[_idx: _idx + _n_len]
            _str = _ngcallable(_block)
            if not isinstance(_str, str):
                raise TypeError(f"'ngcallable' must return a single string")
            _line.insert(_idx, _str)
            _idx -= _n_len
        else:
            _idx -= 1


    return _line





