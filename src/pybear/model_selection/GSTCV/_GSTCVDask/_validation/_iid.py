# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_iid(
    _iid: bool
) -> None:

    """
    `iid` can only be boolean. Indicates whether the data is believed to
    have random distribution of examples (True) or if the data is
    organized non-randomly in some way (False).


    Parameters
    ----------
    _iid:
        bool - to be validated

    Return
    ------
    -
        _iid: bool - validated boolean _iid

    """


    if not isinstance(_iid, bool):
        raise TypeError(f"'iid' must be a bool")







