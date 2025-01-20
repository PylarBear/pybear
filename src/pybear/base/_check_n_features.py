# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

from ._num_features import num_features



def check_n_features(
    X,
    n_features_in_: Union[int, None],
    reset: bool
) -> int:

    """
    Set the 'n_features_in_' attribute, or check against it.

    pybear recommends calling reset=True in 'fit' and in the first call
    to 'partial_fit'. All other methods that validate 'X' should set
    'reset=False'.


    Parameters
    ----------
    X:
        array-like of shape (n_samples, n_features) or (n_samples,) with
        a 'shape' attribute - The input data.
    n_features_in_:
        Union[int, None] - the number of features in the data. If this
        attribute exists, it is integer. If it does not exist, it is
        None.
    reset:
        bool -
        If True, the 'n_features_in_' attribute is set to 'X.shape[1]'
        If False:
            if n_features_in_ exists check it is equal to 'X.shape[1]'
            if n_features_in_ does *not* exist the check is skipped


    Return
    ------
    -
        n_features: int - the number of features in X.

    """

    n_features = num_features(X)

    # this is somewhat arbitrary, in that there is nothing following in
    # this module that requires this. there is nothing in the near
    # periphery that will be impacted if this is changed / removed.
    if n_features == 0:
        raise ValueError("X does not contain any features")

    if reset:
        return n_features

    # reset must be False for all below v v v v v v v v v v v
    if n_features_in_ is None:
        return

    if n_features != n_features_in_:
        raise ValueError(
            f"X has {n_features} feature(s), but expected {n_features_in_}."
        )

    # if get to here, n_features must == n_features_in__
    return n_features










