# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
import numpy.typing as npt

import numpy as np

from .._validation._2D_str_array import _val_2D_str_array

from ..._TextStatistics.TextStatistics import TextStatistics as TS



def _return_overall_uniques(
    _X_as_list_of_lists: Union[list[list[str]], npt.NDArray[str]],
    _return_counts: bool
) -> Union[
         npt.NDArray[str],
         tuple[npt.NDArray[str], npt.NDArray[np.int32]]
     ]:

    """
    Return the unique words in the entire CLEANED_TEXT object. Optionally
    return the frequencies also.


    Parameters
    ----------
    _X_as_list_of_lists:
        Union[list[list[str]], npt.NDArray[str]] - the data in list of
        lists format (strings are tokenized in the inner lists.)
    _return_counts:
        bool - return the frequency of the uniques along with the
        uniques.


    Return
    ------
    -
        Union[npt.NDArray[str], tuple[npt.NDArray[str], npt.NDArray[np.int32]]]


    """


    _val_2D_str_array(_X_as_list_of_lists)

    if not isinstance(_return_counts, bool):
        raise TypeError(f"'return_counts' must be boolean")

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    ts = TS(store_uniques=True)
    for _list in _X_as_list_of_lists:
        ts.partial_fit(_list)

    if not _return_counts:

        return np.array(sorted(ts.uniques_))

    elif _return_counts:

        __ = ts.string_frequency_

        _argsort = np.argsort(list(__.keys()))

        return tuple((
            np.fromiter(__.keys(), dtype=f'<U{max(map(len, __))}')[_argsort],
            np.fromiter(__.values(), dtype=np.int32)[_argsort]
        ))







