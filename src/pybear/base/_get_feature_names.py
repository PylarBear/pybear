# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import warnings
import numpy as np



# this parallels sklearn.utils.validation._get_feature_names(X), which is
# called by the _check_feature_names method of BaseEstimator, which in
# turn is called by the _validate_data method of BaseEstimator
def get_feature_names(X):

    """
    Get feature names from X. X must have a 'columns' attribute or a
    __dataframe__ dunder, i.e., follows the dataframe interchange
    protocol. Otherwise, feature names are not retrieved and None is
    returned. If the dataframe does not have a header comprised of
    strings (the dataframe was constructed without passing a header and
    a default non-string header is used), a warning is raised and None
    is returned.


    Parameters
    ----------
    X:
        {array-like} of shape (n_samples, n_features) of (n_samples, ) -
        Array container from which to extract feature names.

        Objects that have known compatibility with this module:
        pandas dataframe, dask series, dask dataframe, polars dataframe.
        The columns will be considered to be feature names. If the
        dataframe contains non-string feature names, None is returned.

        All other array containers will return 'None'. Examples of
        containers known to not yield feature names: numpy array, dask
        array, scipy sparse matrices / arrays.


    Returns
    -------
    -
        feature_names: Union[NDArray[object], None] - The feature names
        of 'X'. Unrecognized array containers return None.

    """



    feature_names = None

    # extract feature names from containers
    if hasattr(X, "columns"):
        # the 'columns' attr at least covers pandas, dask, and polars dataframes
        feature_names = np.asarray(X.columns, dtype=object)
    elif hasattr(X, "__dataframe__"):
        # as a fall-back, look to the dataframe interchange protocol.
        # this is not the primary implementation because of:
        # - the potential for copy overhead
        # - backward compatibility concerns
        # - library-specific optimizations
        # but if hasattr 'columns' fails, give this a shot.
        df_protocol = X.__dataframe__()
        feature_names = np.asarray(
            list(df_protocol.column_names()),
            dtype=object
        )

    if feature_names is None or len(feature_names) == 0:
        return

    if not all(map(
        isinstance, feature_names, (str for _ in feature_names)
    )):
        # non-string is not supported
        warnings.warn(
            "Feature names are only supported if all input features are "
            "strings, but the input has non-string column name types. "
            "\nIf you want feature names to be stored and validated, you "
            "must convert them all to strings."
        )
        return
    else:
        # Only feature names of all strings are supported
        return feature_names























