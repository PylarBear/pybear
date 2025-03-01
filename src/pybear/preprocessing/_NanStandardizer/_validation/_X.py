# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import XContainer



def _val_X(
    _X: XContainer
) -> None:

    """
    Validate the data.


    Parameters
    ----------
    _X:
        XContainer - the data to have nan-like values replaced.


    Returns
    -------
    -
        None

    """


    _err_msg = (
        f"'X' must be an array-like with a copy() or clone() method, "
        f"such as numpy arrays, scipy sparse matrices or arrays, pandas "
        f"dataframes/series, polars dataframes/series. \nif passing a "
        f"scipy sparse object, it cannot be dok or lil. \npython built-in "
        f"containers, such as lists, sets, and tuples, are not allowed."
    )

    try:
        iter(_X)
        if isinstance(_X, (str, dict, list, tuple, set)):
            raise Exception
        if not hasattr(_X, 'copy') and not hasattr(_X, 'clone'):
            # copy for numpy, pandas, and scipy; clone for polars
            raise Exception
        if hasattr(_X, 'toarray'):
            if not hasattr(_X, 'data'): # ss dok
                raise Exception
            elif all(map(isinstance, _X.data, (list for _ in _X.data))):
                # ss lil
                raise Exception
            else:
                _X = _X.data
    except:
        raise TypeError(_err_msg)







