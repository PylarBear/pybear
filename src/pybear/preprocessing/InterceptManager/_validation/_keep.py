# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import KeepType



def _val_keep(
    _keep:KeepType
) -> None:

    """
    Validate keep - must be any of:
        Literal['first', 'last', 'random'],
        dict[str, any],
        None


    Parameters
    ----------
    _keep:
        Literal['first', 'last', 'random', 'none'], dict[str, any],
        default = 'last' -
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




    _err_msg = (
        f"'keep' must be one of: "
        f"\nLiteral['first', 'last', 'random', 'none'], or "
        f"\ndict[str, any]"
    )

    if isinstance(_keep, str):
        _keep = _keep.lower()
        if _keep in ('first', 'last', 'random', 'none'):
            return

    if isinstance(_keep, dict):
        if len(_keep) == 1 and isinstance(list(_keep.keys())[0], str):
            # pizza is leaving dict value unvalidated for the time being
            return

    raise ValueError(_err_msg)












