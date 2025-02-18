# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ..._type_aliases import XContainer

from ._1D_str_sequence import _val_1D_str_sequence



def _val_X(
    X: XContainer
) -> None:

    """
    Validate the data. Must be a 1D vector of strings.


    Parameters
    ----------
    X:
        Sequence[str] -


    Return
    ------
    -
        None

    """


    _val_1D_str_sequence(X)


    # pizza this is from the original TextCleaner
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #
    #     LIST_OF_STRINGS = np.array(LIST_OF_STRINGS)
    #     if len(LIST_OF_STRINGS.shape) == 1:
    #         LIST_OF_STRINGS = LIST_OF_STRINGS.reshape((1, -1))
    #
    # try:
    #     self.LIST_OF_STRINGS = \
    #         np.fromiter(map(str, LIST_OF_STRINGS[0]), dtype='<U10000')
    # except:
    #     raise TypeError(f"LIST_OF_STRINGS MUST CONTAIN DATA THAT CAN "
    #                     f"BE CONVERTED TO str", fxn)





