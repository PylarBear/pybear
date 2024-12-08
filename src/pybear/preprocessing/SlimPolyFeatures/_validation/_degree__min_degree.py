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

    min_degree, max_degree must be non-negative integers that fulfil "
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


    if isinstance(_degree, bool) or \
        isinstance(_min_degree, bool) or not \
        isinstance(_min_degree, Integral) or not \
        isinstance(_degree, Integral) or \
        _min_degree < 0 or \
        _min_degree > _degree:

        raise ValueError(
            "'min_degree' and 'degree', must be non-negative integers "
            f"that fulfil min_degree <= max_degree, got {_min_degree=}, "
            f"{_degree=}."
        )











