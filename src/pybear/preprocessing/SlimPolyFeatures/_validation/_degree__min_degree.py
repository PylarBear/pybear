# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from numbers import Integral



def _val_degree__min_degree(
    _degree:int,
    _min_degree:int
) -> None:

    """

    min_degree, max_degree must be integers greater than 1 that fulfil "
    min_degree <= max_degree


    Parameters
    ----------
    _degree:
        int, default=2 - The maximum polynomial degree of the generated
        features.

    _min_degree:
        int, default=0 - The minimum polynomial degree of the generated
        features. Polynomial terms with degree below 'min_degree' are
        not included in the final output array. pizza say something about trivial cases.


    Return
    ------
    -
        None


    """

    err_msg = \
        (f"\n'min_degree' must be an integer >= 1.\n'degree' must "
        f"be an integer >= 2.\n'degree' must be greater than or equal "
        f"to 'min_degree'. \ngot {_min_degree=}, {_degree=}.")

    _value_error = 0
    _value_error += isinstance(_degree, bool)
    _value_error += isinstance(_min_degree, bool)
    _value_error += not isinstance(_min_degree, Integral)
    _value_error += not isinstance(_degree, Integral)

    if _value_error:
        raise ValueError(err_msg)

    _value_error += _min_degree < 1
    _value_error += _degree < 2
    _value_error += _min_degree > _degree

    if _value_error:
        raise ValueError(err_msg)











