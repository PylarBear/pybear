# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#





def _val_fill(_fill: str) -> None:

    """
    Validate the fill parameter. Must be a string.


    Parameter
    ---------
    _fill:
        str - the character string to fill void space with.


    Return
    ------
    -
        None


    """


    if not isinstance(_fill, str):
        raise TypeError(f"'fill' must be a string")











