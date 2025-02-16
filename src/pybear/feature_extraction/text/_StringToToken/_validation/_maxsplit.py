# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numbers



def _val_maxsplit(_maxsplit: numbers.Integral) -> None:

    """
    Validate 'maxsplit'. Must be an integer. Anything else pizza?


    Parameters
    ----------
    _maxsplit:
        numbers.Integral - The maximum number of splits made, working
        from left to right.


    """


    try:
        float(_maxsplit)
        if int(_maxsplit) != _maxsplit:
            raise Exception
    except:
        raise TypeError(f"'maxsplit' must be an integer")






