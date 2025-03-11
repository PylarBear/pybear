# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_backfill_sep(_backfill_sep: str) -> None:

    """
    Validate backfill_sep. Must be a string.


    Parameters
    ----------
    _backfill_sep:
        str - when justifying text and there is a shortfall of characters
        in a line, TJ will look to the next line to backfill strings. In
        that case, this character string will divide the text from the
        two lines.


    Returns
    -------
    -
        None

    """


    if not isinstance(_backfill_sep, str):
        raise TypeError(f"'backfill_sep' must be a string.")




