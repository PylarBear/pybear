# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XWipContainer,
    UpperType
)



def _transform(
    _X: XWipContainer,
    _upper: UpperType
) -> XWipContainer:

    """
    Convert all string characters to upper-case, lower-case, or do a
    no-op.


    Parameters
    ----------
    _X:
        XWipContainer - the data.
    _upper:
        Union[bool, None] - what case to set the type to. If True, set
        to upper-case; if False, set to lower-case; if None, do a no-op.


    Returns
    -------
    _X
        XContainer - the data with normalized text.


    """


    if all(map(isinstance, _X, (str for _ in _X))):

        if _upper is None:
            return _X

        if _upper is True:
            _X = list(map(str.upper, _X))
        elif _upper is False:
            _X = list(map(str.lower, _X))

    else:
        raise ValueError(f'unrecognized X format in transform')


    return _X





