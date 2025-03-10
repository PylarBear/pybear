# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import OutputContainer


def _transform(
    _X: list[list[str]],
    _sep: list[str]
) -> OutputContainer:

    """
    Convert each row of strings in X to a single string, joining on
    the string character sequence(s) provided by the 'sep' parameter.
    Returns a python list of strings.


    Parameters
    ----------
    _X:
        list[list[str]] - the (possibly ragged) 2D container of text
        to be joined along rows using the 'sep' character string(s).
        _X should have been converted to a list-of-lists in the transform
        method of the TextJoiner main module.
    _sep:
        list[str] - the 1D python list of strings to use to join the
        strings in the data. The length is identical to the number of
        rows in the data, and each string in _sep is used to join the
        corresponding sequence of strings in the data.


    Return
    ------
    -
        list[str] - A single list containing strings, one string
        for each row in the original X.

    """


    assert isinstance(_X, list)
    assert all(map(isinstance, _X, (list for _ in _X)))
    assert isinstance(_sep, list)
    assert all(map(isinstance, _sep, (str for _ in _sep)))


    for r_idx in range(len(_X)):

        _X[r_idx] = _sep[r_idx].join(_X[r_idx])


    return _X







