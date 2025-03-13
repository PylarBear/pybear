# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_backfill_sep(
    _backfill_sep: str
) -> None:

    """
    Validate backfill_sep. Must be a string.


    Parameters
    ----------
    _backfill_sep:
        str - Some lines in the text may not have any of the given wrap
        separators or line breaks at the end of the line. When justifying
        text and there is a shortfall of characters in a line, TJ will
        look to the next line to backfill strings. In the case where the
        line being backfilled onto does not have a separator or line
        break at the end of the string, this character string will
        separate the otherwise separator-less strings from the strings
        being backfilled onto them.


    Returns
    -------
    -
        None

    """


    if not isinstance(_backfill_sep, str):
        raise TypeError(f"'backfill_sep' must be a string.")




