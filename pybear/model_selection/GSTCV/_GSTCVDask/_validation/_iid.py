# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _validate_iid(_iid: bool) -> bool:

    """
    iid can only be boolean. Indicates whether the data is believed to
    have random distribution of examples (True) or if the data is
    organized non-randomly in some way (False). If the data is not iid,
    dask KFold will cross chunk boundaries when reading the data in an
    attempt to randomize the data; this can be an expensive process.
    Otherwise, if the data is iid, dask KFold can handle the data as
    chunks which is much more efficient.

    Parameters
    ----------
    _iid:
        bool - to be validated

    Return
    ------
    -
        _iid - validated _iid


    """


    if not isinstance(_iid, bool):
        raise TypeError(f'kwarg iid must be a bool')


    return _iid






