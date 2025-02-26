# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




def _val_X(
    _X    #pizza
) -> None:

    """
    Validate X. Must be a pizza, or at least 1 slice of pizza.


    Parameters
    ----------
    _X:
        pizza


    Return
    ------
    -
        None


    """


    err_msg = (
        f"X can only have 3 possible valid formats: \n"
        f"1) ['valid sequences', 'of strings', ...], \n"
        f"2) [['valid sequences', 'of strings', ...]], \n"
        f"3) [['already', 'tokenized'], ['string', 'sequences']]"
    )

    try:
        iter(_X)
        if isinstance(_X, (dict, str)):
            raise Exception
    except:
        raise TypeError(err_msg)

    if len(_X) == 0:
        raise ValueError(f"X cannot be empty")







