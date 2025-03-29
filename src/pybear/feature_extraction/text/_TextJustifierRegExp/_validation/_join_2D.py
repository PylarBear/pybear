# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_join_2D(
    _join_2D: str
) -> None:

    """
    Validate 'join_2D'. Must be a string.


    Parameters
    ----------
    _join_2D:
        str - Ignored if the data is given as a 1D sequence. For 2D
        containers of strings, this is the character string sequence
        that is used to join the strings across rows. The single string
        value is used to join for all rows.


    Return
    ------
    -
        None

    """



    if not isinstance(_join_2D, str):
        raise TypeError(f"'join_2D' must be a string.")




