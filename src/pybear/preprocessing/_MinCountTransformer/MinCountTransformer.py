# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence, Optional
from typing_extensions import Union, Self
from ._type_aliases import (
    CountThresholdType,
    OriginalDtypesType,
    TotalCountsByColumnType,
    IgnoreColumnsType,
    HandleAsBoolType,
    XContainer,
    YContainer
)

import warnings
import numbers
import numpy as np
import pandas as pd
import scipy.sparse as ss
import joblib

from ._make_instructions._make_instructions import _make_instructions
from ._partial_fit._column_getter import _column_getter
from ._partial_fit._original_dtypes_merger import _original_dtypes_merger
from ._partial_fit._parallel_dtypes_unqs_cts import _parallel_dtypes_unqs_cts
from ._partial_fit._tcbc_merger import _tcbc_merger
from ._print_instructions._repr_instructions import _repr_instructions
from ._transform._ic_hab_condition import _ic_hab_condition
from ._transform._make_row_and_column_masks import _make_row_and_column_masks
from ._transform._tcbc_update import _tcbc_update
from ._validation._validation import _validation
from ._validation._y import _val_y

from ...base import (
    FeatureMixin,
    # FitTransformMixin, not used, fit_transform needs special handling
    GetParamsMixin,
    ReprMixin,
    SetOutputMixin,
    SetParamsMixin,
    is_fitted,
    check_is_fitted,
    get_feature_names_out as _get_feature_names_out,
    validate_data
)



class MinCountTransformer(
    FeatureMixin,
    # FitTransformMixin,  # do not use this, need custom code
    GetParamsMixin,
    ReprMixin,
    SetOutputMixin,
    SetParamsMixin
):

    """
    Remove examples and/or features that contain values whose frequencies
    within their respective feature fall below the specified count
    threshold.

    MinCountTransformer (MCT) is useful in cases where interest is only
    in the events (values) that happen most frequently, removing
    infrequent occurrences that may distort relationships that govern
    the more frequent events.

    Totalling frequencies of unique values happens independently on each
    feature, not across the entire data set. At fit time, the uniques and
    their frequencies are generated for every feature, regardless of
    whether the user has made any indications to ignore certain features.
    These results are stored to later find and remove pertinent values
    from data undergoing transform. At transform time, the ignore
    policies are applied and for the remaining features the frequencies
    found during fit are compared against the count threshold to
    determine which values are to be removed. Values are removed by
    deleting the entire example (row) or feature (column) that contains
    it.

    For all cases except binary integer features (string, float, and
    non-binary integer features), the entire example is removed from the
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
    the sub-threshold value and the feature are removed from the data
    set. This makes binary integer columns mimic the behavior of all
    other datatypes. As an example, consider binary integer columns that
    arose by dummying (one-hot encoding) a feature that was originally
    categorical as strings. Had that column not been dummied, the
    examples containing marked values for that feature would have been
    deleted, but not the entire feature. But as dummies, the default
    behavior of binary integer features is to delete the feature and
    retain the example. The delete_axis_0 parameter allows the user to
    force removal of marked examples as would happen under any other
    case. The delete_axis_0 parameter is a global setting and cannot be
    toggled for individual features.

    In the case that removing examples from a feature has left only one
    unique value in that feature, those remaining same-valued examples
    are not deleted (reducing the dataset to empty) but the entire
    feature is removed instead. This is always the case and cannot be
    toggled. The analyst must be wary that MCT will delete any feature
    containing a single value, including an intercept column that the
    user may want to keep. If the user wishes to retain such features,
    a workaround is to process the data with MCT, which will remove the
    columns of constants, then re-append them afterward using pybear
    InterceptManager.

    By default, MCT ignores floats (:param: `ignore_float_columns`,
    default value is True), meaning they are excluded from application
    of the frequency rules. When True, any impact on these features could
    only happen when an example is removed in enacting rules made for
    other features. The user can override ignoring float columns and
    allow the float features' values to be handled as if they were
    categorical, i.e., counted and possibly removed based on insufficient
    frequency as would happen with a categorical column. See the notes
    for more discussion about float columns.

    MCT also defaults to ignoring non-binary integer columns in the same
    way (:param: `ignore_non_binary_integer_columns` default is True).
    When set to False, these type of features to be handled as
    categorical as well.

    The :param: `ignore_nan` parameter allows or disallows the counting
    and removal of nan values just like any other discrete value. The
    default behavior (ignore_nan=True) will count nan values during the
    fit process, but overlook their frequency counts during the transform
    process and not develop any rules for removal. See the notes section
    for in-depth discussion about how MCT handles nan-like values.

    The `ignore_float_columns`, `ignore_non_binary_integer_columns`, and
    `ignore_nan` policies are global settings and these behaviors cannot
    be toggled for individual features.

    The `handle_as_bool` parameter (default=None) allows the user to
    handle a feature as if it were boolean, i.e., in the same manner as
    binary integer columns. Consider a bag-of-words TextVectorizer
    operation which results in a column that is sparse except for a few
    non-zero integer values (which may be different.) 'handle_as_bool'
    allows for the non-zero values as to handled as if they are the same
    value. In that way, handle_as_bool can be used to indicate the
    frequency of presence (or absence) as opposed to the frequency of
    each unique value. `ignore_columns`, `ignore_float_columns`, and
    `ignore_non_binary_integer_columns`, when True, will supercede
    `handle_as_bool` and the feature will be ignored.

    The :param: `reject_unseen_values` parameter requires all values
    within data passed to :meth: `transform` to have been seen during
    fit. When False, values not seen during fit are ignored and no
    operations take place for those values because rules were not
    generated for them. This may lead to the transformed data containing
    values that violate the count threshold. When True, any value in
    data passed to :meth: `transform` not seen during training will raise
    an exception. By its nature, data passed to :meth: `fit_transform`
    must see :meth: `fit` before being passed to :meth: `transform`,
    which makes `reject_unseen_values` is irrelevant in that case.

    The `ignore_columns` and `handle_as_bool` parameters accept:
        1) a single vector of features names if the fit data is passed
            in a format that contains feature names (e.g., a pandas
            dataframe)
        2) a single vector of indices that indicate feature positions, or
        3) a callable that returns 1) or 2) when passed X.
    If data is passed as a dataframe with strings as column names during
    fit, MCT will recognize those names when passed to these parameters
    in a 1D array-like. In all cases, column indices are recognized, as
    long as they are within range. The callable functionality affords the
    luxury of identifying features to ignore or handle as boolean when
    the ultimate name or index of the target feature(s) is/are not known
    beforehand, such as if the data undergoes another transformation
    prior to MCT. The callables must accept a single argument, the X
    parameter passed to methods `partial_fit`, `fit`, and `transform`,
    whereby column indices can be found based on characteristics of X.
    Consider a pipeline-like process that includes some operations that
    act on the features of the data, e.g. TextVectorizer or OneHotEncoder.
    In that case, the target columns can be identified as ignored or
    handled as boolean by running an appropriate algorithm on X via the
    callable. Additional care must be taken when using callables. The
    safest use is with :meth: `fit_transform`, however, use is not
    limited to only that case to allow for use with batch-wise operations
    like dask Incremental and ParallelPostFit wrappers. Upon every call
    to :meth: `partial_fit` and :meth: `transform`, the callable is
    executed on the currently-passed data X, generating column names or
    indices. In a serialized data processing operation, the callable
    must generate the same indices for each X seen or the algorithm will
    return nonsensical results.

    The `transform` and `fit_transform` methods do nothing beyond
    execute the rules prescribed by applying the count threshold to the
    frequencies discovered during :term: fit. In some circumstances,
    transforming new unseen data by these rules may cause the transformed
    data to contain one or more features that only contain a single value,
    and this one value could possibly be the 'nan' value. (If the rules
    would have left only one value in the feature during :term: fit, then
    there would be instruction to delete the feature entirely. However,
    proper rules may still induce these and other undesirable effects on
    unseen data during transform.) In these cases, no furtheraction is
    taken by the transform operation to diagnose or address any such
    conditions. The analyst must take care to discover if such conditions
    exist in the transformed data and address it appropriately.

    In all cases, the "ignore" parameters (:params: ignore_columns,
    ignore_float_columns, ignore_non_binary integer columns) override
    the behavior of other parameters. For example, if column index 0 was
    indicated in :param: ignore_columns but is also indicated in :param:
    handle_as_bool, ignore_columns supercedes and the column is ignored.

    As MCT removes examples that contain values below the count threshold,
    it also collaterally removes values in other features as well,
    possibly causing those values' frequencies to fall below the count
    threshold. Another pass through MCT would then mark the rows/features
    associated with those values and remove them. MCT can perform this
    recursive action with a single call by appropriate settings to
    the :param: `max_recursions` parameter. This functionality is only
    available for :meth: `fit_transform` and not with `partial_fit`,
    `fit`, and `transform`. This ensures that the recursive functionality
    is working with the entire set of data so that the rules developed
    as the recursion proceeds are uniform across all the data.
    Recursion continues until it is stopped for any of these four reasons:
        1) the max_recursions specified by the user is reached (default=1)
        2) all values in all features in the data set appear at least
            'count_threshold' number of times
        3) all rows would be deleted
        4) all columns would be deleted

    See the "Notes" section for additional discussion of less frequent
    use cases.

    MCT has a convenient :meth: `print_instructions` method that allows
    the analyst to view the recipes for deleting values and columns
    before transforming any data. MCT gets uniques and counts during fit,
    and then applies the rules dictated by the parameter settings
    (including minimum frequency threshold) to formulate instructions for
    ignoring features, deleting rows, and deleting columns. These
    instructions are data-agnostic after fit and can be viewed ad libido
    for any valid parameter settings. In the fitted state, the analyst
    can experiment with different settings via :meth: `set_params` to
    see the impact on the transformation. See the documentation for
    the :meth: `print_instructions` method for more information.

    MCT can process X in multiple formats. All MCT 'X' arguments must be
    2D and can be passed as numpy array, pandas DataFrame, or any
    scipy sparse matrix/array. All MCT 'y' arguments can be 1 or 2D and
    can be passed as numpy array, pandas DataFrame, or pandas Series.


    Parameters
    ----------
    count_threshold:
        Optional[Union[int, Sequence[int]]], default = 3 -  The threshold
        that determines whether a value is removed from the data
        (frequency is below threshold) or retained (frequency is greater
        than or equal to threshold.) When passed as a single integer, it
        must be >= 2 and that threshold value is applied to all features.
        If passed as a 1D vector, it must have the same length as the
        number of the features in the data and each value is applied to
        its respective feature. All thresholds must be >= 1, and at least
        one value must be >= 2. Setting the threshold for a feature to 1
        is the same as ignoring the feature.

    ignore_float_columns:
        Optional[bool], default=True - If True, values and frequencies
        within float features are ignored and the feature is retained
        through transform. If False, the feature is handled as if it is
        categorical and unique values are subject to count threshold
        rules and possible removal. See the notes section for more
        discussion about float features.

    ignore_non_binary_integer_columns:
        Optional[bool], default=True - If True, values and frequencies
        within non-binary integer features are ignored and the feature
        is retained through transform. If False, the feature is handled
        as if it is categorical and unique values are subject to count
        threshold rules and possible removal.

    ignore_columns:
        Optional[Union[Sequence[str], Sequence[int], Callable[X], None]],
        default=None - Excludes indicated features from the thresholding
        operation. A one-dimensional vector of integer index positions
        or feature names (if data formats containing column names were
        used during fitting.) Also accepts a callable that creates such
        vectors when passed the data (the 'X' argument). THERE ARE NO
        PROTECTIONS IN PLACE IF THE CALLABLE GENERATES DIFFERENT
        PLAUSIBLE OUTPUTS ON DIFFERENT BATCHES IN AN EPOCH OF DATA. IF
        CONSISTENCY OF IGNORED COLUMNS IS REQUIRED, THEN THE USER MUST
        ENSURE THAT THE CALLABLE PRODUCES IDENTICAL OUTPUTS FOR EACH
        BATCH OF DATA WITHIN AN EPOCH.

    ignore_nan:
        Optional[bool], default=True - If True, nan-like values are
        ignored in all features and pass through the transform operation;
        one could only be removed collateraly by removal of examples for
        causes dictated by other features. If False, frequencies for
        nan-likes are calculated and compared against 'count_threshold'.
        See the Notes section for more on how MCT handles nan-like
        values.

    handle_as_bool:
        Optional[Union[Sequence[str], Sequence[int], callable(X), None]],
        default=None - For the indicated features, non-zero values within
        the feature are treated as if they are the same value. Accepts a
        1D vector of integer index positions or feature names (if data
        formats containing column names were used during fitting.) Also
        accepts a callable that creates such vectors when passed the
        data (the 'X' argument). THERE ARE NO PROTECTIONS IN PLACE IF
        THE CALLABLE GENERATES DIFFERENT PLAUSIBLE OUTPUTS ON DIFFERENT
        BATCHES IN AN EPOCH OF DATA. IF CONSISTENCY OF COLUMNS TO BE
        HANDLED AS BOOLEAN IS REQUIRED, THEN THE USER MUST ENSURE
        THAT THE CALLABLE PRODUCES IDENTICAL OUTPUTS FOR EACH BATCH OF
        DATA WITHIN AN EPOCH.

    delete_axis_0:
        Optional[bool], default=False - Only applies to features
        indicated in :param: `handle_as_bool` or binary integer features
        such as those generated by OneHotEncoder. Under normal operation
        of MCT, when the frequency of one of the two values in a binary
        feature is below :param: `count_threshold`, the minority-class
        examples would not be removed along the example axis, but the
        entire feature would be removed, leaving all other data intact.
        The `delete_axis_0` parameter overrides this behavior. When
        `delete_axis_0` is False, the default behavior for binary columns
        is used, as described above. If True, however, the default
        behavior is overrided and examples associated with the minority
        value are removed along the example axis which would leave only
        one value in the feature, at which point the feature would be
        also be removed for containing only one value.

    reject_unseen_data:
        Optional[bool], default=False - If False, new values encountered
        during transform that were not seen during fit are ignored. If
        True, MCT will terminate when a value that was not seen during
        fit is encountered while transforming data.

    max_recursions:
        Optional[int], default=1 - The number of times MCT repeats its
        algorithm on passed data. Only available for `fit_transform`.

    n_jobs:
        Optional[Union[int, None]], default=None - Number of CPU cores
        or threads used when parallelizing over features while gathering
        uniques and counts during fit. The default is to use cores, but
        an external joblib.parallel_backend context can override this to
        use threads. -1 means using maximum threads/processors.


    Attributes
    ----------
    n_features_in_:
        int - the number of features seen during fit.

    feature_names_in_:
        NDArray[str] of shape (n_features_in_,) - Names of features seen
        during fit. Defined only when X is passed in a container that
        has feature names and the feature names are all strings. If
        accessed when not defined, MCT will raise an AttributeError.

    original_dtypes_ :
        NDArray[Literal['bin_int', 'int', 'float', 'obj']] of shape
        (n_features_in,) - The datatype assigned by MCT to each feature.
        nan-like values are ignored while discovering datatypes
        and the collective datatype of the non-nan values is reported.

    total_counts_by_column_:
        dict[int, dict[any, int]] - A dictionary of the uniques and their
        frequencies found in each column of the fitted data. The keys are
        the zero-based column indices and the values are dictionaries.
        The inner dictionaries are keyed by the unique values found in
        that respective column and the values are their counts. All
        nan-like values are represented by numpy.nan.

    instructions_:
        dict[
            int,
            list[Union['INACTIVE', 'DELETE ALL', 'DELETE COLUMN', any]]
        ] - a dictionary that is keyed by column index and the values are
        lists. Within the lists is information about operations to
        perform with respect to values in the column. The following items
        may be in the list:

        -'INACTIVE' - ignore the column and carry it through for all
            other operations

        -Individual values - indicates to delete the rows along the
            example axis that contain that value in that column,
            possibly including nan-like values.

        -'DELETE ALL' - delete every value in the column.

        -'DELETE COLUMN' - perform any individual row deletions that
            need to take place while the column is still in the data,
            then delete the column from the data.


    Notes
    -----
    Concerning the handling of nan-like values. MCT can recognize various
    nan-like formats, such as numpy.nan, pandas.NA, str(nan), None, and
    others. When collecting uniques and counts during :term: fit, MCT
    extracts each column one-by-one and converts it to a numpy 1D vector.
    Then MCT casts all nan-like values to numpy.nan. The user should be
    wary that regardless of what type of nan-like values were passed
    during fit, MCT will report them as all numpy.nan in the attributes
    `total_counts_by_column_` and `instructions_`. If you are unlucky
    enough to have multiple types of nan-like values in your data, be a
    pro and use pybear.utilities.nan_mask to cast them all to the same
    format. See :param: `ignore_nan`.

    Concerning float features. MCT was never really intended to perform
    thresholding on float columns, but there are use cases where float
    columns have repeating values. So the functionality exists, on the
    off-chance of a legitimate application. pybear typically recommends
    ignoring all float columns. Internally, when MCT gathers uniques and
    counts, it builds a dictionary keyed by column indices and the values
    are dictionaries that are keyed by the uniques in each respective
    column, and the values are the respective counts for those uniques.
    For a float column, in most applications every value in the column
    is unique, and the dictionary fills as such. The user is advised that
    a float column with, say, 100,000,000 unique values will generate an
    equally sized python dictionary, which has immense carrying-cost, and
    will be a pinch-point for MCT and your RAM.

    The analyst is cautioned that this transformer:
        1) modifies data dimensionality along the example axis, and
        2) necessarily forces such an operation on a target object,
            which MCT methods accommodate by accepting target arguments.
    In supervised learning, if the data dimensionality along the example
    axis is changed, the target must also correspondingly change along
    the example axis. These two characteristics of MCT violate at least
    four APIs:
        1) the scikit-learn transformer API,
        2) the scikit-learn pipeline API,
        3) the dask_ml Incremental API, and
        4) the dask_ml ParallelPostFit API.

    For pipeline applications, there are some options available beyond
    the scikit-learn pipeline implementation.

    https://stackoverflow.com/questions/25539311/
    custom-transformer-for-sklearn-pipeline-that-alters-both-x-and-y
    The package imblearn, which is built on top of sklearn, contains an
    estimator FunctionSampler that allows manipulating both the features
    array, X, and target array, y, in a pipeline step. Note that using
    it in a pipeline step requires using the Pipeline class in imblearn
    that inherits from the one in sklearn.

    The dask_ml wrappers (currently) do not accept a 'y' argument
    to :meth: `transform`, and MCT is bound to this condition. However,
    the need to mask 'y' identically to the masking of X still exists.
    For dask Incremental and ParallelPostFit applications, one workaround
    for the API constraint is to merge the data and the target into a
    single X object, use the `ignore_columns` parameter of MCT to ignore
    the target column, perform the fit and transform, then split the X
    object back into data and target. A second workaround is to fit on X
    only, use MCT :meth: `get_row_support` to get the row mask generated
    for X, and apply the mask to y separately.

    dask_ml Incremental and ParallelPostFit wrappers also preclude the
    use of multiple recursions, unless the data/target hybrid object can
    be passed as a single chunk to :meth: `fit_transform`.


    Examples
    --------
    >>> from pybear.preprocessing import MinCountTransformer
    >>> import numpy as np
    >>> column1 = np.array(['a', 'a', 'b', 'c', 'b', 'd'])
    >>> column2 = np.array([0, 1, 0, 1, 2, 0])
    >>> data = np.vstack((column1, column2)).transpose().astype(object)
    >>> data
    array([['a', '0'],
           ['a', '1'],
           ['b', '0'],
           ['c', '1'],
           ['b', '2'],
           ['d', '0']], dtype=object)
    >>> MCT = MinCountTransformer(2, ignore_non_binary_integer_columns=False)
    >>> MCT.fit(data)
    MinCountTransformer(count_threshold=2, ignore_non_binary_integer_columns=False)
    >>> print(MCT.original_dtypes_)
    ['obj' 'int']
    >>> tcbc = MCT.total_counts_by_column_
    >>> tcbc[0]
    {np.str_('a'): 2, np.str_('b'): 2, np.str_('c'): 1, np.str_('d'): 1}
    >>> tcbc[1]
    {np.str_('0'): 3, np.str_('1'): 2, np.str_('2'): 1}
    >>> print(MCT.instructions_)
    {0: [np.str_('c'), np.str_('d')], 1: [np.str_('2')]}
    >>> print(MCT.transform(data))
    [['a' '0']
     ['a' '1']
     ['b' '0']]


    """


    _original_dtypes: OriginalDtypesType
    _total_counts_by_column: TotalCountsByColumnType


    def __init__(
        self,
        count_threshold:Optional[CountThresholdType]=3,
        *,
        ignore_float_columns:Optional[bool]=True,
        ignore_non_binary_integer_columns:Optional[bool]=True,
        ignore_columns:Optional[IgnoreColumnsType]=None,
        ignore_nan:Optional[bool]=True,
        handle_as_bool:Optional[HandleAsBoolType]=None,
        delete_axis_0:Optional[bool]=False,
        reject_unseen_values:Optional[bool]=False,
        max_recursions:Optional[int]=1,
        n_jobs:Optional[Union[int, None]]=None
    ):


        self.count_threshold = count_threshold
        self.ignore_float_columns = ignore_float_columns
        self.ignore_non_binary_integer_columns = ignore_non_binary_integer_columns
        self.ignore_columns = ignore_columns
        self.ignore_nan = ignore_nan
        self.handle_as_bool = handle_as_bool
        self.delete_axis_0 = delete_axis_0
        self.reject_unseen_values = reject_unseen_values
        self.max_recursions = max_recursions
        self.n_jobs = n_jobs


    @property
    def original_dtypes_(self):

        check_is_fitted(self)

        return self._original_dtypes


    @property
    def total_counts_by_column_(self):

        check_is_fitted(self)

        return self._total_counts_by_column


    @property
    def instructions_(self):

        check_is_fitted(self)

        return self._make_instructions()


    def __pybear_is_fitted__(self) -> bool:

        # must have this because there are no trailing-underscore attrs
        # generated by {partial_}fit(). all the trailing-underscore attrs
        # are accessed via @property.
        return hasattr(self, '_total_counts_by_column')


    def partial_fit(
        self,
        X: XContainer,
        y: Optional[YContainer]=None
    ) -> Self:

        """
        Batch-wise accrual of uniques and counts for thresholding.

        All of X is processed in multiple batches. This is intended for
        cases when :meth: `fit` is not feasible due to very large number
        of `n_samples` or because X is read from a continuous stream.


        Parameters
        ----------
        X:
            Union[numpy.ndarray, pandas.DataFrame, scipy sparse] of shape
            (n_samples, n_features) - The data used to determine the
            uniques and their frequencies, used for later thresholding
            along the feature axis.
        y:
            Optional[Union[numpy.ndarray, pandas.DataFrame, pandas.Series,
            None]] of shape (n_samples, n_outputs), or (n_samples,),
            default= None - Target relative to X for classification or
            regression; None for unsupervised learning.


        Return
        ------
        -
            self : object -
                Fitted MinCountTransformer instance.

        """


        self._recursion_check()


        X = validate_data(
            X,
            copy_X=False,
            cast_to_ndarray=False,
            accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
            dtype='any',
            require_all_finite=False,
            cast_inf_to_nan=False,
            standardize_nan=False,
            allowed_dimensionality=(2,),
            ensure_2d=False,
            order='F',
            ensure_min_features=1,
            ensure_max_features=None,
            ensure_min_samples=3,
            sample_check=None
        )

        # GET n_features_in_, feature_names_in_, _n_rows_in_ ** * ** *
        # do not make assignments! let the functions handle it.
        self._check_feature_names(X, reset=not is_fitted(self))
        self._check_n_features(X, reset=not is_fitted(self))

        # IF WAS PREVIOUSLY FITTED, THEN self._n_rows_in EXISTS
        if hasattr(self, '_n_rows_in'):
            self._n_rows_in += X.shape[0]
        else:
            self._n_rows_in = X.shape[0]

        # END GET n_features_in_, feature_names_in_, _n_rows_in_ ** * **

        _validation(
            X,
            self.count_threshold,
            self.ignore_float_columns,
            self.ignore_non_binary_integer_columns,
            self.ignore_columns,
            self.ignore_nan,
            self.handle_as_bool,
            self.delete_axis_0,
            self.reject_unseen_values,
            self.max_recursions,
            self.n_jobs,
            getattr(self, 'n_features_in_'),
            getattr(self, 'feature_names_in_', None)
        )


        # GET TYPES, UNQS, & CTS FOR ACTIVE COLUMNS ** ** ** ** ** ** **

        # scipy coo, dia, and bsr cant be sliced by columns, need to be
        # converted to another format. standardize all scipy sparse to
        # csc, makes for quicker column scans when getting unqs/cts.
        # need to change it back after the scan. dont mutate X, avoid
        # copies of X.
        if hasattr(X, 'toarray'):
            _og_dtype = type(X)
            X = ss.csc_array(X)


        # need to run all columns to get the dtypes; no columns are ignored,
        # for this operation, so any ignore inputs do not matter. getting
        # dtypes on columns that are ignored is needed to validate new
        # partial fits have appropriate data.

        # DONT HARD-CODE backend, ALLOW A CONTEXT MANAGER TO SET
        with joblib.parallel_config(prefer='processes', n_jobs=self.n_jobs):
            DTYPE_UNQS_CTS_TUPLES = \
                joblib.Parallel(return_as='list')(
                    joblib.delayed(_parallel_dtypes_unqs_cts)(
                        _column_getter(X,_idx),
                        X.shape[0],
                        _idx
                    ) for _idx in range(self.n_features_in_)
                )

        # if scipy sparse, change back to the original format. do this
        # before going into the ic/hab callables below, possible that the
        # callable may have some dependency on the container.
        if hasattr(X, 'toarray'):
            X = _og_dtype(X)
            del _og_dtype



        _col_dtypes = np.empty(self.n_features_in_, dtype='<U8')
        # DOING THIS for LOOP 2X TO KEEP DTYPE CHECK SEPARATE AND PRIOR TO
        # MODIFYING self._total_counts_by_column, PREVENTS INVALID DATA FROM
        # INVALIDATING ANY VALID UNQS/CT IN THE INSTANCE'S
        # self._total_counts_by_column
        for col_idx, (_dtype, UNQ_CT_DICT) in enumerate(DTYPE_UNQS_CTS_TUPLES):
            _col_dtypes[col_idx] = _dtype

        self._original_dtypes = _original_dtypes_merger(
            _col_dtypes,
            getattr(self, '_original_dtypes', None),
            self.n_features_in_
        )

        del _col_dtypes

        self._total_counts_by_column: dict[int, dict[any, int]] = \
            _tcbc_merger(
                DTYPE_UNQS_CTS_TUPLES,
                getattr(self, '_total_counts_by_column', {})
            )

        del DTYPE_UNQS_CTS_TUPLES

        # END GET TYPES, UNQS, & CTS FOR ACTIVE COLUMNS ** ** ** ** ** *

        self._ignore_columns, self._handle_as_bool = \
            _ic_hab_condition(
                X,
                self.ignore_columns,
                self.handle_as_bool,
                self.ignore_float_columns,
                self.ignore_non_binary_integer_columns,
                self._original_dtypes,
                self.count_threshold,
                self.n_features_in_,
                getattr(self, 'feature_names_in_', None),
                _raise=True
            )

        X = np.ascontiguousarray(X)

        return self


    def fit(
        self,
        X: XContainer,
        y: Optional[YContainer]=None
    ) -> Self:

        """
        Determine the uniques and their frequencies to be used for
        thresholding.


        Parameters
        ----------
        X:
            Union[numpy.ndarray, pandas.DataFrame, scipy sparse] of shape
            (n_samples, n_features) - The data used to determine the
            uniques and their frequencies, used for later thresholding
            along the feature axis.
        y:
            Optional[Union[numpy.ndarray, pandas.DataFrame, pandas.Series,
            None]] of shape (n_samples, n_outputs), or (n_samples,),
            default=None - Target relative to X; None for unsupervised
            learning.


        Return
        ------
        -
            self : object -
                Fitted MinCountTransformer.


        """

        self.reset()

        return self.partial_fit(X, y)


    def fit_transform(
        self,
        X: XContainer,
        y: Optional[YContainer]=None
    ) -> Union[tuple[XContainer, YContainer], XContainer]:

        """
        Fits MinCountTransformer to X and returns a transformed version
        of X.


        Parameters
        ----------
        X:
            Union[numpy.ndarray, pandas.DataFrame, scipy.sparse] of shape
            (n_samples, n_features) - The data used to determine the
            uniques and their frequencies and to be transformed by rules
            created from those frequencies.
        y:
            Optional[Union[None, numpy.ndarray, pandas.DataFrame,
            pandas.Series]] of shape (n_samples, n_outputs), or
            (n_samples,), default=None - Target values for the data.
            None for unsupervised transformations.


        Return
        ------
        -
            X_tr: Union[numpy.ndarray, pandas.DataFrame, scipy.sparse]
                of shape (n_samples_new, n_features_new) - the transformed
                data.

            y_tr: Union[numpy.ndarray, pandas.DataFrame, pandas.Series]
                of shape (n_samples_new,) or (n_samples_new, n_outputs)
                Transformed target, if provided.

        """

        # cant use FitTransformMixin, need custom code to handle _recursion_check

        # this temporarily creates an attribute that is only looked at by
        # self._recursion_check. recursion check needs to be disabled
        # when calling transform() from this method, but otherwise the
        # recursion check in transform() must always be operative.
        self.recursion_check_disable = True

        __ = self.fit(X, y).transform(X, y)

        delattr(self, 'recursion_check_disable')

        return __


    def get_feature_names_out(
        self,
        input_features:Optional[Union[Sequence[str], None]]=None
    ):
        """
        Get the feature names for the output of :meth: `transform`.


        Parameters
        ----------
        input_features:
            Optional[Sequence[str], None]], default=None - Externally
            provided feature names for the fitted data, not the
            transformed data.

            If input_features is None:

            - if feature_names_in_ is defined, then feature_names_in_ is
                used as the input features.

            - if feature_names_in_ is not defined, then the following
                input feature names are generated:
                ["x0", "x1", ..., "x(n_features_in_ - 1)"].

            If input_features is not None:

            - if feature_names_in_ is not defined, then input_features is
                used as the input features.

            - if feature_names_in_ is defined, then input_features must
                exactly match the features in feature_names_in_.


        Return
        ------
        -
            feature_names_out: NDArray[object] - The feature names of
            the transformed data.

        """

        check_is_fitted(self)

        feature_names_out = _get_feature_names_out(
            input_features,
            getattr(self, 'feature_names_in_', None),
            self.n_features_in_
        )

        return feature_names_out[self.get_support(indices=False)]


    def get_metadata_routing(self):
        """Get metadata routing of this object - Not implemented."""
        __ = type(self).__name__
        raise NotImplementedError(f"get_metadata_routing is not available in {__}")


    # def get_params inherited from GetParamsMixin


    def get_row_support(self, indices:bool=False):
        """Get a mask, or integer index, of the rows selected.


        Parameters
        ----------
        indices:
            bool, default=False - If True, the return value will be an
            array of integers, rather than a boolean mask.


        Return
        ------
        -
            support: ndarray - A slicer that selects the retained rows
            from the X most recently seen by transform. If indices is
            False, this is a boolean array of shape (n_input_features, )
            in which an element is True if its corresponding row is
            selected for retention. If indices is True, this is an
            integer array of shape (n_output_features, ) whose values
            are indices into the sample axis.

        """

        check_is_fitted(self)

        if not hasattr(self, '_row_support'):
            raise AttributeError(
                f"get_row_support() can only be accessed after some data "
                f"has been transformed"
            )

        if indices is False:
            return self._row_support
        elif indices is True:
            return np.arange(len(self._row_support))[self._row_support]


    def get_support(self, indices:Optional[bool]=False):

        """
        Get a boolean mask or the integer indices of the features
        retained in the data.


        Parameters
        ----------
        indices:
            Optional[bool], default=False - If True, the return value
            will be a 1D array of integers; if False, the return will be
            a 1D boolean mask.


        Return
        ------
        -
            support:
                Union[numpy.NDArray[bool], numpy.NDArray[np.uint32]] -
                A mask that selects the retained features from a feature
                vector. If indices is False, this is a boolean array of
                shape (n_features_in_,) in which an element is True if
                its corresponding feature is selected for retention. If
                indices is True, this is an integer array of shape
                (n_features_in_, ) whose values are indices into the
                input feature vector.

        """

        check_is_fitted(self)

        # must use _make_instructions() in order to construct the column
        # support mask after fit and before a transform. otherwise, if an
        # attribute _column_support were assigned to COLUMN_KEEP_MASK in
        # transform() like _row_support is assigned to ROW_KEEP_MASK, then
        # a transform would have to be done before being able to access
        # get_support().

        COLUMNS = np.array(
            ['DELETE COLUMN' not in v for v in self._make_instructions().values()]
        )

        if indices is False:
            return np.array(COLUMNS)
        elif indices is True:
            return np.arange(self.n_features_in_)[COLUMNS].astype(np.uint32)


    def print_instructions(
        self,
        *,
        clean_printout: Optional[bool] = True,
        max_char: Optional[numbers.Integral] = 99
    ):

        """
        Display instructions generated for the current fitted state,
        subject to the current settings of the parameters. The printout
        will indicate what values and columns will be deleted, and if
        all columns or all rows will be deleted. Use :meth: `set_params`
        after finishing fits to change MCTs parameters and see the impact
        on the transformation.

        If the instance has multiple recursions (i.e., :param:
        max_recursions is > 1), parameters cannot be changed via method
        set_params, but the net effect of the actual transformation that
        was performed is displayed (remember that multiple recursions
        can only be accessed through :meth: `fit_transform`). The results
        are displayed as a single set of instructions, as if to perform
        the cumulative effect of the recursions in a single step.

        This print utility can only report the instructions and outcomes
        that can be directly inferred from the information learned about
        uniques and counts during fitting. It cannot predict any
        interaction effects that occur during transform of a dataset that
        may ultimately cause all rows to be deleted. It also cannot
        capture the effects of previously unseen values that may be
        passed during transform.


        Parameters
        ----------
        clean_printout:
            bool - Truncate printout to fit on screen.
        max_char:
            numbers.Integral, default = 99 - the maximum number of
            characters to display per line if `clean_printout` is set to
            True. Ignored if `clean_printout` is False. Must be an
            integer in range [72, 120].


        Return
        ------
        -
            None

        """

        check_is_fitted(self)

        # params can be changed after fit & before calling this by
        # set_params(). need to validate params. _make_instructions()
        # handles the validation of almost all the params in __init__
        # except max_recursions, reject_unseen_params, and n_jobs.
        # max_recursions cannot be changed in set_params once fitted.
        # None of these 3 are used here.

        # after fit, ic & hab are blocked from being set to callable
        # (but is OK if they were already a callable when fit.) that
        # means that the mapping of the callables used during fit lives
        # in self._ignore_columns and/or self._handle_as_bool. but,
        # set_params does not block setting ic/hab to Sequence[str] or
        # Sequence[int]. so if self.ignore_columns and/or
        # self.handle_as_bool are callable, we need to pass the output
        # that lives in _ic & _hab. but if not callable, need to use the
        # (perhaps changed) ic/hab in self.ignore_columns &
        # self.handle_as_bool. if ic/hab were changed to Sequence[str]
        # in set_params, need to map to Sequence[int].

        # _ic_hab_condition takes X, but we dont have it so we need to
        # spoof it. X doesnt matter here, X is only for ic/hab callable
        # in partial_fit() and transform(). since we are always passing
        # ic/hab as vectors, dont need to worry about the callables.


        if callable(self.ignore_columns):
            _wip_ic = self._ignore_columns
        else:
            _wip_ic = self.ignore_columns

        if callable(self.handle_as_bool):
            _wip_hab = self._handle_as_bool
        else:
            _wip_hab = self.handle_as_bool

        self._ignore_columns, self._handle_as_bool = \
            _ic_hab_condition(
                None,     # placehold X
                _wip_ic,
                _wip_hab,
                self.ignore_float_columns,
                self.ignore_non_binary_integer_columns,
                self._original_dtypes,
                self.count_threshold,
                self.n_features_in_,
                getattr(self, 'feature_names_in_', None),
                _raise=True
            )

        del _wip_ic, _wip_hab

        _repr_instructions(
            _delete_instr=self._make_instructions(),
            _total_counts_by_column=self._total_counts_by_column,
            _thresholds=self.count_threshold,
            _n_features_in=self.n_features_in_,
            _feature_names_in=getattr(self, 'feature_names_in_', None),
            _clean_printout=clean_printout,
            _max_char=max_char
        )


    def reset(self) -> Self:
        """
        Reset the internal data state of MinCountTransformer.

        """

        if hasattr(self, 'n_features_in_'):
            delattr(self, 'n_features_in_')

        if hasattr(self, 'feature_names_in_'):
            delattr(self, 'feature_names_in_')

        if hasattr(self, '_ignore_columns'):
            delattr(self, '_ignore_columns')

        if hasattr(self, '_handle_as_bool'):
            delattr(self, '_handle_as_bool')

        if hasattr(self, '_n_rows_in'):
            delattr(self, '_n_rows_in')

        if hasattr(self, '_original_dtypes'):
            delattr(self, '_original_dtypes')

        if hasattr(self, '_total_counts_by_column'):
            delattr(self, '_total_counts_by_column')

        if hasattr(self, '_row_support'):
            delattr(self, '_row_support')

        return self


    def score(self, X: XContainer, y:Optional[YContainer]=None) -> None:
        """
        Dummy method to spoof dask Incremental and ParallelPostFit
        wrappers. Verified must be here for dask wrappers.
        """

        check_is_fitted(self)

        return


    # def set_output(self, transform) - inherited from SetOutputMixin


    def set_params(self, **params):
        """
        Set the parameters of the MCT instance.

        Pass the exact parameter name and its value as a keyword argument
        to the `set_params` method call. Or use ** dictionary unpacking
        on a dictionary keyed with exact parameter names and the new
        parameter values as the dictionary values. Valid parameter keys
        can be listed with :meth: `get_params`. Note that you can
        directly set the parameters of MinCountTransformer.

        Once MCT is fitted, MCT :meth: `set_params` blocks some
        parameters from being set to ensure the validity of results.
        In these cases, to use different parameters without creating a
        new instance of MCT, call MCT :meth: `reset` on the instance.
        Otherwise, create a new MCT instance.

        'max_recursions' is always blocked when MCT is in a fitted state.

        If MCT was fit with :param: 'max_recursions' >= 2  (only
        a :meth: `fit_transform` could be done), all parameters are
        blocked. To break the block, call :meth: `reset` before
        calling :meth: `set_params`. All information learned from any
        prior `fit_transform` will be lost.

        Also, when MCT has been fitted, :params: `ignore_columns` and
        `handle_as_bool` cannot be set to a callable (they can, however,
        be changed to None, Sequence[int], or Sequence[str]). To set
        these parameters to a callable when MCT is in a fitted state,
        call :meth: `reset` then use :meth: `set_params` to set them
        to a callable. All information learned from any prior fit(s)
        will be lost when calling 'reset'.


        Parameters
        ----------
        **params:
            dict[str, any] - MinCountTransformer parameters.


        Return
        ------
        -
            self: the MinCountTransformer instance.

        """

        
        # if MCT is fitted with max_recursions==1, allow everything but
        # block 'ignore_columns' and 'handle_as_bool' from being set to
        # callables.
        # if max_recursions is >= 2, block everything.
        # if not fitted, allow everything to be set.
        if is_fitted(self):

            if self.max_recursions > 1:
                raise ValueError(
                    f":method: 'set_params' blocks all parameters from being "
                    f"set when MCT is fitted with :param: 'max_recursions' "
                    f">= 2. \nto set new parameter values, call :method: "
                    f"'reset' then call :method: 'set_params'."
                )

            _valid_params = {}
            _invalid_params = {}
            _garbage_params = {}
            _spf_params = self.get_params()
            for param in params:
                if param not in _spf_params:
                    _garbage_params[param] = params[param]
                elif param == 'max_recursions':
                    _invalid_params[param] = params[param]
                elif param in ['ignore_columns', 'handle_as_bool'] \
                        and callable(params[param]):
                    _invalid_params[param] = params[param]
                else:
                    _valid_params[param] = params[param]

            if any(_garbage_params):
                # let super.set_params raise
                super().set_params(**params)

            if 'max_recursions' in _invalid_params:
                warnings.warn(
                    "Once MCT is fitted, :param: 'max_recursions' cannot be "
                    "changed. To change this setting, call :method: 'reset' or "
                    "create a new instance of MCT. 'max_recursions' has not "
                    "been changed but any other valid parameters passed have "
                    "been set."
                )

            if any(
                [_ in _invalid_params for _ in ['ignore_columns', 'handle_as_bool']]
            ):
                warnings.warn(
                    "Once MCT is fitted, :params: 'ignore_columns' and "
                    "'handle_as_bool' cannot be set to callables. \nThe "
                    "currently passed parameter(s) "
                    f"{', '.join(list(_invalid_params))} has/have been "
                    "skipped, but any other valid parameters that were "
                    "passed have been set. \nTo set "
                    "ignore_columns/handle_as_bool to callables without "
                    "creating a new instance of MCT, call :method: 'reset' "
                    "on this instance then set the callable parameter "
                    "values (all results from previous fits will be lost). "
                    "Otherwise, create a new instance of MCT."
                )

            super().set_params(**_valid_params)

            del _valid_params, _invalid_params, _garbage_params, _spf_params

        else:

            super().set_params(**params)


        return self


    @SetOutputMixin._set_output_for_transform
    def transform(
        self,
        X:XContainer,
        y:Optional[YContainer]=None,
        *,
        copy:Optional[Union[bool, None]]=None
    ) -> Union[tuple[XContainer, YContainer], XContainer]:

        """
        Reduce X by the thresholding rules found during fit.


        Parameters
        ----------
        X:
            Union[numpy.ndarray, pandas.DataFrame, scipy.sparse] of shape
            (n_samples, n_features) - The data that is to be reduced
            according to the thresholding rules found during fit.

        y:
            Optional[Union[None, numpy.ndarray, pandas.DataFrame,
            pandas.Series]] of shape (n_samples, n_outputs), or
            (n_samples,), default = None - Target values for the data.
            None for unsupervised transformations.

        copy:
            Optional[Union[bool, None]], default = None - whether to
            copy X (and y, if provided) before doing the transformation.


        Return
        ------
        -
            X_tr : Union[numpy.ndarray, pandas.DataFrame, scipy.sparse]
                    of shape (n_samples_new, n_features_new) -
                    Transformed array.
            y_tr : Union[numpy.ndarray, pandas.DataFrame, pandas.Series]
                    of shape (n_samples_new, n_outputs) or
                    (n_samples_new,) - Transformed target.

        """

        check_is_fitted(self)

        self._recursion_check()

        X_tr = validate_data(
            X,
            copy_X=copy or False,
            cast_to_ndarray=False,
            accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
            dtype='any',
            require_all_finite=False,
            cast_inf_to_nan=False,
            standardize_nan=False,
            allowed_dimensionality=(2,),
            ensure_2d=False,
            order='F',
            ensure_min_features=1,
            ensure_max_features=None,
            ensure_min_samples=3,
            sample_check=None
        )

        self._check_feature_names(X_tr, reset=False)
        self._check_n_features(X_tr, reset=False)

        _validation(
            X_tr,
            self.count_threshold,
            self.ignore_float_columns,
            self.ignore_non_binary_integer_columns,
            self.ignore_columns,
            self.ignore_nan,
            self.handle_as_bool,
            self.delete_axis_0,
            self.reject_unseen_values,
            self.max_recursions,
            self.n_jobs,
            getattr(self, 'n_features_in_'),
            getattr(self, 'feature_names_in_', None)
        )

        # END X handling ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # y handling ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        if y is not None:

            # When dask_ml Incremental and ParallelPostFit (versions 2024.4.4
            # and 2023.5.27 at least) are passed y = None, they are putting
            # y = ('', order[i]) into the dask graph for y and sending that
            # as the value of y to the wrapped partial fit method.
            # All use cases where y=None are like this, it will always happen
            # and there is no way around it. To get around this, look to
            # see if y is of the form tuple(str, int). If that is the
            # case, override y with y = None.

            # Determine if the MCT instance is wrapped by dask_ml Incremental
            # or ParallelPostFit by checking the stack and looking for the
            # '_partial' method used by those modules. Wrapping with these
            # modules imposes limitations on passing a value to y.
            # as of 25_02_04 bypassing the wrapper diagnosis. an obscure
            # object like ('', order[i]) hopefully is enough to determine
            # that its wrapped by dask.
            # _using_dask_ml_wrapper = False
            # for frame_info in inspect.stack():
            #     _module = inspect.getmodule(frame_info.frame)
            #     if _module:
            #         if _module.__name__ == 'dask_ml._partial':
            #             _using_dask_ml_wrapper = True
            #             break
            # del _module

            if isinstance(y, tuple) and isinstance(y[0], str) \
                    and isinstance(y[1], int):
                y = None

            # END accommodate dask_ml junk y ** * ** * ** * ** * ** * ** * *

            y_tr = validate_data(
                y,
                copy_X=copy or False,
                cast_to_ndarray=False,
                accept_sparse=None,
                dtype='any',
                require_all_finite=False,
                cast_inf_to_nan=False,
                standardize_nan=False,
                allowed_dimensionality=(1, 2),
                ensure_2d=False,
                order='C',
                ensure_min_features=1,
                ensure_max_features=None,
                ensure_min_samples=3,
                sample_check=X.shape[0]
            )

            _val_y(y_tr)
        else:
            y_tr = None
        # END y handling ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


        # extra count_threshold val
        _base_err = (
            f":param: 'count_threshold' is >= the total number of rows "
            f"seen during fitting. this is a degenerate condition. "
            f"\nfit more data or set a lower count_threshold."
        )
        if isinstance(self.count_threshold, numbers.Integral) \
                and self.count_threshold >= self._n_rows_in:
            raise ValueError(_base_err)
        # must be list-like
        elif np.any(self.count_threshold) >= self._n_rows_in:
            raise ValueError(f"at least one value in " + _base_err)
        del _base_err


        # VALIDATE _ignore_columns & handle_as_bool; CONVERT TO IDXs **

        # PERFORM VALIDATION & CONVERT ic/hab callables to IDXS.
        # _ignore_columns MUST BE BEFORE _make_instructions

        self._ignore_columns, self._handle_as_bool = \
            _ic_hab_condition(
                X_tr,
                self.ignore_columns,
                self.handle_as_bool,
                self.ignore_float_columns,
                self.ignore_non_binary_integer_columns,
                self._original_dtypes,
                self.count_threshold,
                self.n_features_in_,
                getattr(self, 'feature_names_in_', None),
                _raise=True
            )
        # END handle_as_bool -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # END VALIDATE _ignore_columns & handle_as_bool; CONVERT TO IDXs ** **

        _delete_instr = self._make_instructions()

        # if scipy sparse, dia, coo, and bsr cannot be indexed, need to convert
        # to an indexable sparse. since this needs to be done, might as well
        # convert all scipy sparse to csc for fast column operations.
        # need to change this back later.
        # do this after the ignore_columns/handle_as_bool callables, the
        # callables may depend on the container.
        if hasattr(X_tr, 'toarray'):
            _og_dtype = type(X_tr)
            X_tr = X_tr.tocsc()

        # BUILD ROW_KEEP & COLUMN_KEEP MASKS ** ** ** ** ** ** ** ** **

        ROW_KEEP_MASK, COLUMN_KEEP_MASK = \
            _make_row_and_column_masks(
                X_tr,
                self._total_counts_by_column,
                _delete_instr,
                self.reject_unseen_values,
                self.n_jobs
            )

        # END BUILD ROW_KEEP & COLUMN_KEEP MASKS ** ** ** ** ** ** ** **

        self._row_support = ROW_KEEP_MASK.copy()

        if all(ROW_KEEP_MASK) and all(COLUMN_KEEP_MASK):
            # Skip all recursion code.
            # if all rows and all columns are kept, the data has converged
            # to a point where all the values in every column appear in
            # their respective column at least count_threshold times.
            # There is no point to performing any more possible recursions.
            pass
        else:
            # X must be 2D, np, pd, or csc
            if isinstance(X_tr, np.ndarray):
                X_tr = X_tr[ROW_KEEP_MASK, :]
                X_tr = X_tr[:, COLUMN_KEEP_MASK]
            elif isinstance(X_tr, pd.core.frame.DataFrame):
                X_tr = X_tr.loc[ROW_KEEP_MASK, :]
                X_tr = X_tr.loc[:, COLUMN_KEEP_MASK]
            elif isinstance(X_tr, (ss.csc_matrix, ss.csc_array)):
                # ensure bool mask for ss
                X_tr = X_tr[ROW_KEEP_MASK.astype(bool), :]
                X_tr = X_tr[:, COLUMN_KEEP_MASK.astype(bool)]
            else:
                raise Exception(
                    f"expected X as ndarray, pd df, or csc. got {type(X_tr)}."
                )

            if y_tr is not None:
                # y can be only np or pd, 1 or 2D
                if isinstance(y_tr, np.ndarray):
                    y_tr = y_tr[ROW_KEEP_MASK]
                elif isinstance(y_tr, pd.core.series.Series):
                    y_tr = y_tr.loc[ROW_KEEP_MASK]
                elif isinstance(y_tr, pd.core.frame.DataFrame):
                    y_tr = y_tr.loc[ROW_KEEP_MASK, :]
                else:
                    raise Exception(
                        f"expected y as ndarray or pd df or series, got {type(y_tr)}."
                    )

            # v v v everything below here is for recursion v v v v v v
            if self.max_recursions > 1:

                # NEED TO RE-ALIGN _ignore_columns, _handle_as_bool AND
                # count_threshold FROM WHAT THEY WERE FOR self TO WHAT
                # THEY ARE FOR THE CURRENT (POTENTIALLY COLUMN MASKED)
                # DATA GOING INTO THIS RECURSION

                if callable(self.ignore_columns):
                    NEW_IGN_COL = self.ignore_columns # pass the function!
                else:
                    IGN_COL_MASK = np.zeros(self.n_features_in_).astype(bool)
                    IGN_COL_MASK[self._ignore_columns.astype(np.uint32)] = True
                    NEW_IGN_COL = np.arange(sum(COLUMN_KEEP_MASK))[
                                            IGN_COL_MASK[COLUMN_KEEP_MASK]
                    ]
                    del IGN_COL_MASK

                if callable(self.handle_as_bool):
                    NEW_HDL_AS_BOOL_COL = self.handle_as_bool # pass the function!
                else:
                    HDL_AS_BOOL_MASK = np.zeros(self.n_features_in_).astype(bool)
                    HDL_AS_BOOL_MASK[self._handle_as_bool.astype(np.uint32)] = True
                    NEW_HDL_AS_BOOL_COL = np.arange(sum(COLUMN_KEEP_MASK))[
                                            HDL_AS_BOOL_MASK[COLUMN_KEEP_MASK]]
                    del HDL_AS_BOOL_MASK

                if isinstance(self.count_threshold, numbers.Integral):
                    NEW_COUNT_THRESHOLD = self.count_threshold
                else:
                    NEW_COUNT_THRESHOLD = self.count_threshold[COLUMN_KEEP_MASK]

                # END RE-ALIGN _ic, _hab, count_threshold ** * ** * ** * **

                RecursiveCls = MinCountTransformer(
                    NEW_COUNT_THRESHOLD,
                    ignore_float_columns=self.ignore_float_columns,
                    ignore_non_binary_integer_columns=
                        self.ignore_non_binary_integer_columns,
                    ignore_columns=NEW_IGN_COL,
                    ignore_nan=self.ignore_nan,
                    handle_as_bool=NEW_HDL_AS_BOOL_COL,
                    delete_axis_0=self.delete_axis_0,
                    max_recursions=self.max_recursions-1,
                    n_jobs=self.n_jobs
                )

                del NEW_IGN_COL, NEW_HDL_AS_BOOL_COL, NEW_COUNT_THRESHOLD

                if y_tr is None:
                    X_tr = RecursiveCls.fit_transform(X_tr, y_tr)
                else:
                    X_tr, y_tr = RecursiveCls.fit_transform(X_tr, y_tr)

                # vvv tcbc update vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
                MAP_DICT = dict((
                    zip(
                        list(range(RecursiveCls.n_features_in_)),
                        sorted(list(map(int, self.get_support(indices=True))))
                    )
                ))

                self._total_counts_by_column = \
                    _tcbc_update(
                        self._total_counts_by_column,
                        RecursiveCls._total_counts_by_column,
                        MAP_DICT
                )

                # ^^^ tcbc update ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                self._row_support[self._row_support] = RecursiveCls._row_support

                del RecursiveCls, MAP_DICT

            del ROW_KEEP_MASK, COLUMN_KEEP_MASK


        if hasattr(X_tr, 'toarray'):
            X_tr = _og_dtype(X_tr)
            del _og_dtype
        elif isinstance(X_tr, np.ndarray):
            X_tr = np.ascontiguousarray(X_tr)

        if y_tr is None:
            return X_tr
        else:
            return X_tr, y_tr


    def _make_instructions(self):

        """
        Make the instructions dictionary for the current uniques and
        counts stored in the instance and the current settings of the
        parameters.


        """

        check_is_fitted(self)

        return _make_instructions(
            self.count_threshold,
            self.ignore_float_columns,
            self.ignore_non_binary_integer_columns,
            self._ignore_columns,
            self.ignore_nan,
            self._handle_as_bool,
            self.delete_axis_0,
            self._original_dtypes,
            self.n_features_in_,
            getattr(self, 'feature_names_in_', None),
            self._total_counts_by_column
        )


    def _recursion_check(self) -> None:
        """
        Raise exception if attempting to use recursion with partial_fit,
        fit, and transform. Only allow fit_transform.

        """
        if getattr(self, 'recursion_check_disable', False) is False:

            if self.max_recursions > 1:
                raise ValueError(
                    f"partial_fit(), fit(), and transform() are not "
                    f"available if max_recursions > 1. fit_transform() only."
                )



































