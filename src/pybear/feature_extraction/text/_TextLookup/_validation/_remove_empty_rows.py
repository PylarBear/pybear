# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




def _val_remove_empty_rows(
    _remove_empty_rows: bool
) -> bool:

    """
    Validate 'remove_empty_rows'. Must be boolean.
    
    
    Parameters
    ----------
    _remove_empty_rows:
        bool - whether to delete any rows that become empty during 
        processing by TextLookup.
    
    
    Returns
    -------
    -
        None
    
    
    """


    if not isinstance(_remove_empty_rows, bool):
        raise TypeError(f"'remove_empty_rows' must be boolean")






