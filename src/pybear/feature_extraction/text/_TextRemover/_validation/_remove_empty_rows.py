# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_remove_empty_rows(
    _remove_empty_rows: bool
) -> None:

    """
    Validate 'remove_empty_rows'. Must be boolean.


    Parameters
    ----------
    _remove_empty_rows:
        bool - whether to remove any empty rows from 2D data. This does
        not apply to 1D data. By definition, rows are always removed
        from 1D data, because the strings are the rows.


    Returns
    -------
    -
        None


    """


    if not isinstance(_remove_empty_rows, bool):
        raise TypeError("'remove_empty_rows' must be boolean")



