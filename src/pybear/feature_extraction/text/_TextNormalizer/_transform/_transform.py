# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    UpperType
)



def _transform(
    _X: XContainer,
    _upper: UpperType
) -> XContainer:

    """
    Convert all string characters to upper-case, lower-case, or do a
    no-op.


    Parameters
    ----------
    _X:
        XContainer - the data.
    _upper:
        Union[bool, None] - what case to set the type to. If True, set
        to upper-case; if False, set to lower-case; if None, do a no-op.


    Returns
    -------
    _X
        XContainer - the data with normalized text.


    """


    if _upper is None:
        return _X


    if all(map(isinstance, _X, (str for _ in _X))):

        if _upper is True:
            _X = list(map(str.upper, _X))
        elif _upper is False:
            _X = list(map(str.lower, _X))

    elif all(map(isinstance, _X, (list for _ in _X))):

        if _upper is True:
            _X = list(map(list, map(lambda x: map(str.upper, x), _X)))
        elif _upper is False:
            _X = list(map(list, map(lambda x: map(str.lower, x), _X)))

    else:
        raise Exception(f'unrecognized X format in transform')


    return _X





