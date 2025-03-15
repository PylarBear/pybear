# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_verbose(_verbose: bool) -> None:

    """
    Validate verbose. Must be boolean.


    Parameters
    ----------
    _verbose:
        bool - Whether to display helpful information during processing.


    Return
    ------
    -
        None

    """


    if not isinstance(_verbose, bool):
        raise TypeError(f"'verbose' must be boolean")











