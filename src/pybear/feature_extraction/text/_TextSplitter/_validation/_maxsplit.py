# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import MaxSplitsType

import numbers



def _val_maxsplit(
    _maxsplit: MaxSplitsType,
    _n_rows: numbers.Integral
) -> None:

    """
    Validate the 'maxsplit' parameter.


    Parameters
    ----------
    _maxsplit:
        MaxSplitsType - the maximum number of splits to perform on
        each row in the data, working from left to right. Can be None,
        an integer, or a list of Nones and/or integers. If None, the
        default maxsplit for re.split() is used. If a single integer,
        that integer is passed to re.split() for all rows in the data.
        If passed as a list, the number of entries must equal the
        number of rows in the data, and the values are applied to the
        corresponding row in the data. The values are only validated
        for being None or integers; any exceptions raised beyond that
        are raised by re.split().
    _n_rows:
        numbers.Integral - the number of strings (rows) in the data.


    Return
    ------
    -
        None


    Notes
    -----
    re.split()


    """


    if _maxsplit is None:
        return


    err_msg = (
        f"'maxsplit' must be None, a single integer, or a list of "
        f"Nones and/or integers, whose length equals the number of "
        f"strings in the data."
    )


    try:
        if isinstance(_maxsplit, numbers.Integral):
            raise UnicodeError
        if isinstance(_maxsplit, list):
            raise TimeoutError
        raise Exception
    except UnicodeError:
        # if is a single number
        if isinstance(_maxsplit, bool):
            raise TypeError(err_msg)
    except TimeoutError:
        # if is a list, len must == len(_X) and can contain Nones or ints
        if len(_maxsplit) != _n_rows:
            raise ValueError(err_msg)
        for _ in _maxsplit:
            # numbers.Integral covers integers and bool
            if not isinstance(_, (numbers.Integral, type(None))):
                raise TypeError(err_msg)
            if _ is True:
                raise TypeError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)


    del err_msg


