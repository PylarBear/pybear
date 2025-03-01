# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal



def _val_keep(
    _keep: Literal['first', 'last', 'random']
) -> None:

    """
    Validate keep - must be 'first', 'last', or 'random'.


    Parameters
    ----------
    _keep:
        Literal['first', 'last', 'random'], default = 'first' -
        The strategy for keeping a single representative from a set of
        identical columns. 'first' retains the column left-most in the
        data; 'last' keeps the column right-most in the data; 'random'
        keeps a single randomly-selected column from the set of
        duplicates.


    Return
    ------
    -
        None


    """




    _err_msg = lambda _required: f"'keep' must be one of {', '.join(_required)}"
    _required = ('first', 'last', 'random')

    if not isinstance(_keep, str):
        raise TypeError(_err_msg(_required))

    if sum([_ == _keep for _ in _required]) != 1:
        raise ValueError(_err_msg(_required))
    del _err_msg, _required












