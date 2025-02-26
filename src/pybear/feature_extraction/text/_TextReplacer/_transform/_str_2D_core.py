# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from .._type_aliases import XContainer

import operator



def _str_2D_core(
    _X: XContainer,
    _new_str_replace: list[set[Union[tuple[str, str], tuple[str, str, int]]]]
) -> XContainer:

    """
    Search and replace strings in a 2D dataset using exact string
    matching.


    Parameters
    ----------
     _X:
        XContainer - the data.
    _new_str_replace:
        list[set[Union[tuple[str, str], tuple[str, str, int]]]] - the
        search and replace criteria. must have been conditioned by
        _param_conditioner before being passed here.


    Return
    ------
    -
        XContainer - the 2D array-like with string substitutions made.


    """


    assert isinstance(_X, list)
    for _row in _X:
        assert isinstance(_row, list)
        assert all(map(isinstance, _row, (str for _ in _row)))

    assert isinstance(_new_str_replace, list)
    assert all(map(isinstance, _new_str_replace, (set for _ in _X)))

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    for _idx in range(len(_X)):

        # if set is empty will skip (False in og list becomes empty set)
        for _tuple in _new_str_replace[_idx]:   # set of tuples (or empty)

            # for str_replace, the tuples are already the exact args for str.replace
            _X[_idx] = \
                list(map(operator.methodcaller('replace', *_tuple), _X[_idx]))


    return _X











