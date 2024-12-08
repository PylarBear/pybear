# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_interaction_only(
    _interaction_only: bool
) -> None:

    """
    Validate interaction_only; must be bool


    Parameters
    ----------
    _interaction_only:
        bool, default = False -


    Return
    ------
    -
        None


    """


    if not isinstance(_interaction_only, bool):
        raise TypeError(f"'interaction_only' must be bool")







