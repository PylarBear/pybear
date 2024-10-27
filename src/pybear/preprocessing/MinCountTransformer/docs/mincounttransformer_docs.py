# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def mincounttransformer_docs():

    """
    Remove examples (and/or features) that contain values whose feature-
    wise frequencies fall below the specified count threshold.

    MinCountTransformer (MCT) is useful in cases where interest is only
    in the events (values) that happen most frequently, ignoring
    infrequent occurrences that may distort relationships that govern
    the more frequent events.

    Totalling frequencies of unique values happens independently on each
    feature, not across the entire data set. At fit time, the uniques and
    their frequencies are generated for every feature, regardless of
    whether the user has made any indications to ignore certain features.
    These results are stored to later find and remove pertinent values
    from data undergoing :term: transform. At transform time, the ignore
    policies are applied and for the remaining features the frequencies
    found during fit are compared against the count threshold to determine
    which values are to be removed. Values are removed by deleting the
    entire example (row) or feature (column) that contains it.

    For all cases except binary integer features (string, float, and non-
    binary integer features), the entire example is removed from the
    dataset when the frequency of a value within is below the count
    threshold. This by nature also removes values in other features that
    were not necessarily below the threshold. For features of these
    datatypes, the example will always be deleted, and the feature will
    not be deleted for simply containing a value whose frequency is below
    the count threshold, unless only one unique value remains. See below
    for rules regarding removal of a feature.

    For binary integers, however, the default action (when the
    delete_axis_0 parameter is set to the default of False) is not to
    delete examples, but to delete the entire feature. However, if the
    delete_axis_0 parameter is set to True, both the examples containing
    the minority value and the feature are removed from the data set.
    This makes binary integer columns mimic the behavior of all other
    datatypes. As an example, consider binary integer columns that arose
    by dummying (one-hot encoding) a feature that was originally
    categorical as strings. Had that column not been dummied, the
    examples containing marked values for that feature would have been
    deleted, but not the feature as well. But as dummies, the default
    behavior of binary integer features is to delete the feature and
    retain the example. The delete_axis_0 parameter allows the user to
    force removal of marked examples as would happen under any other case.
    The delete_axis_0 parameter is a global setting and cannot be toggled
    for individual features.

    In the case that removing examples from a feature has left only one
    unique value in that feature, those remaining same-valued examples
    are not deleted (reducing the dataset to empty) but the entire
    feature is removed instead. This is always the case and cannot be
    toggled. The analyst must be wary that MCT will delete any feature
    containing a single value, including an intercept column. If the user
    wishes to retain such a feature, a workaround is to remove and retain
    the feature separately before using MCT and reappend to the data
    afterwards.

    As MCT removes examples that contain values below the count threshold,
    it also collaterally removes values in other features as well,
    possibly causing those values' frequencies to fall below the count
    threshold. Another pass through MCT would then mark the rows/features
    associated with those values and remove them. MCT can perform this
    recursive action with a single call by appropriate settings to the
    max_recursions parameter. This functionality is only available with
    :meth: fit_transform and not with :meths: partial_fit, fit, and
    transform. This ensures that the recursive functionality is working
    with the entire set of data so that the rules developed as the
    recursion proceeds are uniform across all the data.
    Recursion continues until it is stopped for any of these four reasons:
        1) the max_recursions specified by the user is reached (default=1)
        2) all values in all features in the data set appear at least
            count_threshold number of times
        3) all rows would be deleted
        4) all columns would be deleted

    By default, MCT ignores float features (default :param:
    ignore_float_columns=True), meaning they are excluded from application
    of the frequency rules; any impact on these features could only
    happen when an example is removed in enacting rules made for other
    features. The user can override this behavior and allow the float
    feature's values to be handled as if they were categorical, i.e.,
    counted and possibly removed based on insufficient frequency as would
    happen with a categorical column.

    MCT also defaults to handling non-binary integer columns in the same
    way (:param: ignore_non_binary_integer_columns=True.) When set to
    False, the ignore_non_binary_integer_columns parameter allows these
    type of features to be handled as categorical as well.

    :param: ignore_nan allows or disallows the counting and removal of
    nan values just like any other discrete value. The default behavior
    (ignore_nan=True) will count nan values during the fit process, but
    overlook their frequency counts during the transform process and not
    develop any rules for removal.

    :params: ignore_float_columns, ignore_non_binary_integer_columns, and
    ignore_nan policies' are global settings and these behaviors cannot
    be toggled for individual features.

    :param: handle_as_bool (default=None) allows the user to handle a
    feature as if it were boolean, i.e., in the same manner as binary
    integer columns. Consider a bag-of-words TextVectorizer operation
    which results in a column that is sparse except for a few non-zero
    integer values (which may be different.) handle_as_bool allows for
    the non-zero values as to handled as if they are the same value. In
    that way, handle_as_bool can be used to indicate the frequency of
    presence (or absence) as opposed to the frequency of each unique
    value. :params: ignore_columns, ignore_float_columns, and
    ignore_non_binary_integer_columns, when True, will supercede
    handle_as_bool and the feature will be ignored.

    :param: reject_unseen_values requires all values within data passed
    to :meth: transform to have been seen during fit. When False, values
    not seen during fit are ignored and no operations take place for
    those values because rules were not generated for them. This may
    lead to the transformed data containing values that violate the count
    threshold. When True, any value within the data not seen during
    training will terminate MCT. :meth: fit_transform operations see all
    the fitted and transformed data in a single step, therefore :param:
    reject_unseen_values is irrelevant in that case.

    :params: ignore_columns and handle_as_bool accept:
        1) a single vector of features names if the fit data is passed in
            a format that contains feature names (e.g., pandas dataframe)
        2) a single vector of indices that indicate feature positions, or
        3) a callable that returns 1) or 2).
    If data is passed as a dataframe with strings as column names during
    fit, MCT will recognize those names when passed to these parameters
    in an array-like. In all cases, column indices are recognized, as
    long as they are within range. The callable functionality affords the
    luxury of identifying features to ignore or handle as boolean when
    the ultimate name or index of the feature is not known beforehand,
    as in a sci-kit learn pipeline operation. The callables must accept
    a single argument, which is the X parameter passed to :meth: transform,
    whereby column indices can be found based on characteristics of X.
    Consider a pipeline process that includes some operations that act
    on the features of the data, e.g. TextVectorizer or OneHotEncoder.
    In that case, the desired columns can be identified as ignored or
    handled as boolean by running an appropriate algorithm on X delivered
    via the callable. Additional care must be taken when using callables.
    The safest use is with :meth: fit_transform, however, use is not
    limited to only that case to allow for use with dask Incremental and
    ParallelPostFit wrappers. Upon every call to :meth: transform, the
    callable is executed on the currently-passed data X, generating
    column indices. In a serialized data processing operation, the
    callable must generate the same indices for each X seen or the
    algorithm will return nonsensical results.

    :meth: transform and :meth: fit_transform do nothing beyond execute
    the rules prescribed by applying the count threshold to the
    frequencies discovered during :term: fit. In some circumstances,
    transforming new unseen data by these rules may cause the transformed
    data to contain one or more features that only contain a single value,
    and this one value could possibly be the 'nan' value. (If the rules
    would have left only one value in the feature during :term: fit, then
    there would be instruction to delete the feature entirely. However,
    proper rules may still induce these and other undesirable effects on
    unseen data during transform.) In these cases, no further action is
    taken by the :term: transform operation to diagnose or address any
    such conditions. The analyst must take care to discover if such
    conditions exist in the transformed data and address it appropriately.

    In all cases, :term: "ignore" parameters (:params: ignore_columns,
    ignore_float_columns, ignore_non_binary integer columns) override the
    behavior of other parameters. For example, if column index 0 was
    indicated in :param: ignore_columns but is also indicated in :param:
    handle_as_bool, ignore_columns supercedes and the column is ignored.

    See the "Notes" section for additional discussion of less frequent
    use cases.


    Parameters
    ----------
    count_threshold : int
        The threshold that determines whether a value is removed
        from the data (frequency is below threshold) or retained
        (frequency is greater than or equal to threshold.)

    ignore_float_columns : bool, default=True
        If True, values and frequencies within float features are ignored
        and the feature is retained through transform. If False, the
        feature is handled as if it is categorical and unique values are
        subject to count threshold rules and possible removal.

    ignore_non_binary_integer_columns : bool, default=True
        If True, values and frequencies within non-binary integer features
        are ignored and the feature is retained through transform. If
        False, the feature is handled as if it is categorical and unique
        values are subject to count threshold rules and possible removal.

    ignore_columns : list, callable, or None, default=None
        A one-dimensional vector of integer index positions or feature
        names (if data formats containing column names were passed to
        :meth: fit.) Also accepts a callable that creates such vectors
        when passed the 'X' argument that was passed to :meth: transform.
        Excludes indicated features from the thresholding operation.

    ignore_nan : bool, default=True
        If True, nan is ignored in all features and passes through the
        :term: transform operation; it would only be removed collateraly
        by removal of examples for causes dictated by other features. If
        False, frequency for both numpy.nan and 'nan' as string (not case
        sensitive) are calculated and assessed against count_threshold.

    handle_as_bool : list, callable, or None, default=None
        A one-dimensional vector of integer index positions or feature
        names (if data formats containing column names were passed to
        :meth: fit.) For the indicated features, non-zero values within
        the feature are treated as if they are the same value.

    delete_axis_0 : bool, default=False
        Only applies to features indicated in :param: handle_as_bool or
        binary integer features such as those generated by OneHotEncoder.
        Under normal operation of MCT, when the frequency of one of the
        two values in the binary feature is below :param: count_threshold,
        the minority-class examples would be removed and would leave
        only one value in the feature, at which point the feature would
        be also be removed for containing only one value. :param:
        delete_axis_0 overrides this behavior. When :param: delete_axis_0
        is False under the above conditions, the feature is removed
        without deleting examples, preserving the data in the other
        features. If True, however, the default behavior is used and
        examples associated with the minority value are removed and the
        feature is then also removed for having only one value.

    reject_unseen_data : bool, default=False
        If False (default), new values encountered during :term: transform
        that were not seen during :term: fit are ignored. If True, MCT
        will terminate when a value that was not seen during :term: fit
        is encountered while transforming data.

    max_recursions : int, default=1
        The number of times MCT repeats its algorithm on passed data.
        Only available for :meth: fit_transform.

    n_jobs: int, default=None
        Number of CPU cores used when parallelizing over features while
        gathering uniques and counts during :term: fit and when
        parallelizing over features while building masks during :term:
        transform. None means 1 unless in a joblib.parallel_backend
        context. -1 means using all processors.


    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term: fit.

    feature_names_in_ :
        Names of features seen during :term: fit. Defined only when X
        is passed in a format that contains feature names and has feature
        names that are all strings.

    original_dtypes_ : ndarray of shape (n_features_in,)
        The datatypes discovered in each feature at first :term: fit.
        np.nan or 'nan' values are ignored while discovering datatypes
        and the collective datatype of the non-nan values is reported.


    Notes
    -----
    MCT recognizes n/a values in the numpy.nan format or in the 'nan'
    string format (not case sensitive.) See :param: ignore_nan.

    The analyst is cautioned that this transformer:
        1) modifies data dimensionality along the example axis, and
        2) necessarily forces such an operation on a target object,
            which requires that MCT methods accept target arguments.
    In supervised learning, if the data dimensionality along the example
    axis is changed, the target must also correspondingly change along
    the example axis. These two characteristics of MCT violate at least
    four APIs:
        1) the scikit-learn transformer API,
        2) the scikit-learn pipeline API,
        3) the dask Incremental API, and
        4) the dask ParallelPostFit API.

    For pipeline applications, there are some options available beyond
    the scikit-learn pipeline implementation.

    https://stackoverflow.com/questions/25539311/
    custom-transformer-for-sklearn-pipeline-that-alters-both-x-and-y
    The package imblearn, which is built on top of sklearn, contains an
    estimator FunctionSampler that allows manipulating both the features
    array, X, and target array, y, in a pipeline step. Note that using
    it in a pipeline step requires using the Pipeline class in imblearn
    that inherits from the one in sklearn.

    For dask Incremental and ParallelPostFit applications, a workaround
    for the API constraint is to merge the data and the target into a
    single X object, use the ignore_columns parameter of MCT to ignore
    the target column, perform the :term: fit and :term: transform, then
    split the X object back into data and target. dask_ml Incremental
    and ParallelPostFit wrappers also preclude the use of multiple
    recursions, unless the data/target hybrid object can be
    passed as a single chunk to :meth: fit_transform.

    When wrapping MCT with dask_ml Incremental or ParallelPostFit, MCT
    is bound to any constraints imposed by dask_ml. dask_ml (currently)
    does not accept a mix of dask and non-dask objects passed
    simultaneously as X and y to :meths: fit and partial_fit. Therefore,
    if passing y to these methods of a wrapped instance of MCT, both X
    and y must be both dask objects or non-dask objects. Dask objects
    include arrays, dataframes, and series, and non-dask objects include
    numpy arrays, pandas dataframes, and pandas series. MCT does not
    require that y be passed to any method, and simply not passing a y
    argument circumvents these constraints. The dask_ml wrappers
    (currently) do not accept a y argument to :meth: transform, and MCT
    is bound to this condition. However, the need to mask y identically
    to the masking of X still exists. A workaround is to :term: fit on X
    only, use the get_row_support method of MCT to get the mask generated
    for X, and apply the mask to y separately.


    Examples
    --------
    >>> from pybear.preprocessing import MinCountTransformer
    >>> data = [['a', 0], ['a', 1], ['b', 0], ['c', 1], ['b', 2], ['d', 0]]
    >>> mincount = MinCountTransformer(2, ignore_non_binary_integer_columns=False)
    >>> out = mincount.fit(data)
    >>> print(mincount.original_dtypes_)
    ['obj' 'int']
    >>> print(mincount.transform(data))
    [['a' 0]
     ['a' 1]
     ['b' 0]]
    >>> print(mincount.transform([['a', 2], ['b', 0], ['c', 1], ['b', 1]]))
    [['b' 0]
     ['b' 1]]

    """





































