# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _validate_iid(_iid: bool) -> bool:

    """
    iid can only be boolean. Indicates whether the data is believed to
    have random distribution of examples (True) or if the data is
    organized non-randomly in some way (False).


    Parameters
    ----------
    _iid:
        bool - to be validated

    Return
    ------
    -
        _iid - validated boolean _iid


    """


    if not isinstance(_iid, bool):
        raise TypeError(f'iid must be a bool')


    return _iid






