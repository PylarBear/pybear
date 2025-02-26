# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from .._type_aliases import XContainer



def _str_1D_core(
    _X: XContainer,
    _new_str_replace: list[set[Union[tuple[str, str], tuple[str, str, int]]]]
) -> XContainer:

    """
    Search and replace strings in a 1D dataset using exact string
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
        XContainer - the 1D vector with string substitutions made.


    """


    assert isinstance(_X, list)
    assert all(map(isinstance, _X, (str for _ in _X)))

    assert isinstance(_new_str_replace, list)
    assert all(map(isinstance, _new_str_replace, (set for _ in _X)))

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    for _idx in range(len(_X)):

        # if set is empty will skip (False in og list becomes empty set)
        for _tuple in _new_str_replace[_idx]:   # set of tuples (or empty)

            # for str_replace, the tuples are already the exact args for str.replace
            _X[_idx] = _X[_idx].replace(*_tuple)


    return _X







