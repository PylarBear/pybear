# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_skip_numbers(_skip_numbers: bool) -> None:

    """
    Validate skip_numbers. Must be boolean.


    Parameters
    ----------
    _skip_numbers:
        bool - Whether to skip numbers while processing text. If True,
        TextLookup tries float(word) to see if python can cast the text
        to a number. If the text can be cast, TextLookup considers it a
        number and skips it. If False, TextLookup does not perform this
        check and treats something that may otherwise be a number as a
        word.



    Return
    ------
    -
        None

    """


    if not isinstance(_skip_numbers, bool):
        raise TypeError(f"'skip_numbers' must be boolean")











