# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_interaction_only(
    _interaction_only: bool
) -> None:

    """
    Validate interaction_only; must be bool.


    Parameters
    ----------
    _interaction_only:
        bool - If True, only interaction features are produced, that is,
        polynomial features that are products of 'degree' distinct input
        features. Terms with power of 2 or higher for any feature are
        excluded. Consider 3 features 'a', 'b', and 'c'. If
        'interaction_only' is True, 'min_degree' is 1, and 'degree' is
        2, then only the first degree interaction terms ['a', 'b', 'c']
        and the second degree interaction terms ['ab', 'ac', 'bc'] are
        returned in the polynomial expansion.


    Return
    ------
    -
        None


    """


    if not isinstance(_interaction_only, bool):
        raise TypeError(f"'interaction_only' must be bool")







