# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import UpperType



def _val_upper(
    _upper: UpperType
) -> None:

    """
    Validate the 'upper' parameter. Must be None or bool.


    Parameters
    ----------
    _upper:
        UpperType - If True, set all text to upper-case; if False set
        all text to lower-case; if None, do a no-op.


    Return
    ------
    -
        None


    Notes
    -----
    see str.upper(), str.lower()

    """


    if _upper is None:
        return


    if not isinstance(_upper, bool):
        raise TypeError(f"'upper' must be None or boolean.")







