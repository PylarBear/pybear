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
        not included in the final output array, except for zero-degree
        terms (a column of ones), which is controlled by :param: include_bias.
        Note that `min_degree=0`
        and `min_degree=1` are equivalent as outputting the degree zero term is
        determined by `include_bias`.


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











