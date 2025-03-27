# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import XWipContainer



def _transform(
    _X: XWipContainer
) -> XWipContainer:

    """
    Strip leading and trailing spaces from 1D and 2D text data.


    Parameters
    ----------
    _X:
        XWipContainer - the data whose text will be stripped.


    Return
    ------
    _X:
        XWipContainer - the data with leading and trailing spaces removed.


    """


    if all(map(isinstance, _X, (str for _ in _X))):
        _X = list(map(str.strip, _X))

    elif all(map(isinstance, _X, (list for _ in _X))):
        _X = list(map(list, map(lambda x: map(str.strip, x), _X)))

    else:
        raise Exception(f'unrecognized X format in transform')


    return _X


