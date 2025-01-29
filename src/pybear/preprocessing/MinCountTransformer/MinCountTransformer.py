# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numbers
from typing import Iterable, Optional
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

import inspect
from copy import deepcopy
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as ss
import joblib

from ._ic_hab_condition import _ic_hab_condition
from ._make_instructions._make_instructions import _make_instructions
from ._partial_fit._column_getter import _column_getter
from ._partial_fit._original_dtypes_merger import _original_dtypes_merger
from ._partial_fit._parallel_dtypes_unqs_cts import _parallel_dtypes_unqs_cts
from ._partial_fit._tcbc_merger import _tcbc_merger
from ._print_instructions._repr_instructions import _repr_instructions
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

# pizza, after set_params_test is done, when all the tests there prove out that
# the og params are not mutated, see if all the deepcopy can come out.



class MinCountTransformer(
    FeatureMixin,
    # FitTransformMixin,  # do not use this, need custom code
    GetParamsMixin,
    ReprMixin,
    SetOutputMixin,
    SetParamsMixin
):

    # pizza say something about the load of huge float columns on _unq_ct_dict

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

    ignore_columns:
        Optional[Union[Iterable[str], Iterable[int], callable(X), None]],
        default=None - Excludes indicated features from the thresholding
        operation. A one-dimensional vector of integer index positions
        or feature names (if data formats containing column names were
        used during fitting.) Also accepts a callable that creates such
        vectors when passed the data (the 'X' argument). THERE ARE NO
        PROTECTIONS IN PLACE IF THE CALLABLE GENERATES DIFFERENT
        PLAUSIBLE OUTPUTS ON DIFFERENT BATCHES IN AN EPOCH OF DATA. IF
        CONSISTENCY OF IGNORED COLUMNS IS REQUIRED, THEN THE USER MUST
        ENSURE THAT THE CALLABLE PRODUCES CONSISTENT OUTPUTS FOR EACH
        BATCH OF DATA.

    ignore_nan:
        Optional[bool], default=True -
        If True, nan is ignored in all features and passes through the
        :term: transform operation; it would only be removed collateraly
        by removal of examples for causes dictated by other features. If
        False, frequency for both numpy.nan and 'nan' as string (not case
        sensitive) are calculated and assessed against count_threshold.

    handle_as_bool:
        Optional[Union[Iterable[str], Iterable[int], callable(X), None]],
        default=None - For the indicated features, non-zero values within
        the feature are treated as if they are the same value. A
        one-dimensional vector of integer index positions or feature
        names (if data formats containing column names were used during
        fitting.) Also accepts a callable that creates such vectors when
        passed the data (the 'X' argument). THERE ARE NO PROTECTIONS IN
        PLACE IF THE CALLABLE GENERATES DIFFERENT PLAUSIBLE OUTPUTS ON
        DIFFERENT BATCHES IN AN EPOCH OF DATA. IF CONSISTENCY OF COLUMNS
        TO BE HANDLED AS BOOLEAN IS REQUIRED, THEN THE USER MUST ENSURE
        THAT THE CALLABLE PRODUCES CONSISTENT OUTPUTS FOR EACH BATCH OF
        DATA.

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

    n_jobs:
        Optional[Union[int, None]], default=None
        Number of CPU cores used when parallelizing over features while
        gathering uniques and counts during :term: fit and when
        parallelizing over features while building masks during :term:
        transform. None means 1 unless in a joblib.parallel_backend
        context. -1 means using all processors.


    Attributes
    ----------
    n_features_in_:
        int - the number of features seen during fit.

    feature_names_in_:
        NDArray[str] of shape (n_features_in_,) - Names of features seen
        during fit. Defined only when X is passed in a container that
        has feature names and has feature names that are all strings. If
        not defined, MCT will raise an AttributeError.

    original_dtypes_ :
        NDArray[Literal['bin_int', 'int', 'float', 'obj']] of shape
        (n_features_in,) - The datatypes assigned by MCT to each feature.
        nan-like values are ignored while discovering datatypes
        and the collective datatype of the non-nan values is reported.

    total_counts_by_column_:
        dict[int, dict[any, int]] - A dictionary of the uniques and their
        frequencies found in each column of the fitted data. The keys are
        the zero-based column indices and the values are dictionaries.
        The inner dictionaries are keyed by the unique values found in
        that respective column and the values are their counts. All
        nan-like values are represented by numpy.nan.

    delete_instructions_:
        dict[
            int,
            Union[Literal['INACTIVE', 'DELETE ALL', 'DELETE COLUMN'], any]
        ] - a dictionary that is keyed by column index and the values are
        lists. Within the lists is information about operations to
        perform with respect to values in the column. The following items
        may be in the list:

        -'INACTIVE' - ignore the column and carry it through for all
            other operations

        -Individual values - (in raw datatype format, not converted to
            string) indicates to delete the rows on axis 0 that contain
            that value in that column, including nan-like values

        -'DELETE ALL' - delete every value in the column.

        -'DELETE COLUMN' - perform any individual row deletions that
            need to take place while the column is still in the data,
            then delete the column from the data.


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
    {np.float64(0.0): 3, np.float64(1.0): 2, np.float64(2.0): 1}
    >>> print(MCT.instructions_)
    {0: [np.str_('c'), np.str_('d')], 1: [np.float64(2.0)]}
    >>> print(MCT.transform(data))
    [['a' '0']
     ['a' '1']
     ['b' '0']
     ['b' '2']]

    # pizza this is bad!  tests need to catch this!
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


    @original_dtypes_.setter
    def original_dtypes_(self, value):
        raise AttributeError(f'original_dtypes_ attribute is read-only')


    @property
    def total_counts_by_column_(self):

        check_is_fitted(self)

        return self._total_counts_by_column


    @total_counts_by_column_.setter
    def total_counts_by_column_(self, value):
        raise AttributeError(f'total_counts_by_column_ attribute is read-only')


    @property
    def instructions_(self):

        check_is_fitted(self)

        return self._make_instructions()


    @instructions_.setter
    def instructions_(self, value):
        raise AttributeError(f'instructions_ attribute is read-only')


    def __pybear_is_fitted__(self):

        """
        If an estimator/transformer does not set any attributes with a
        trailing underscore, it can define a '__pybear_is_fitted__' method
        returning a boolean to specify if the estimator/transformer is
        fitted or not.

        """

        return hasattr(self, 'n_features_in_')


    def partial_fit(
        self,
        X: XContainer,
        y: Optional[YContainer]=None
    ) -> Self:

        """Online accrual of uniques and counts for later thresholding.

        All of X is processed as a single batch. This is intended for
        cases when :meth:`fit` is not feasible due to very large number
        of `n_samples` or because X is read from a continuous stream.

        Parameters
        ----------
        X : Union[numpy.ndarray, pandas.DataFrame, pandas.Series,
            scipy sparse] of shape (n_samples,
            n_features) - The data used to determine the uniques and
            their frequencies, used for later thresholding along the
            feature axis.

        y : Union[numpy.ndarray, pandas.DataFrame, pandas.Series, None]
            of shape (n_samples, n_outputs), or (n_samples,), default=
            None - Target relative to X for classification or regression;
            None for unsupervised learning.

        Return
        ------
        -
            self : object -
                Fitted min count transformer.

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
        joblib_kwargs = {
            'prefer': 'processes', 'return_as':'list', 'n_jobs':self.n_jobs
        }
        DTYPE_UNQS_CTS_TUPLES = \
            joblib.Parallel(**joblib_kwargs)(
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

        self._total_counts_by_column: dict[int, dict[any, int]] = _tcbc_merger(
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

        return self


    def fit(
        self,
        X: XContainer,
        y: Optional[YContainer]=None
    ) -> Self:

        """Determine the uniques and their frequencies to be used for
        later thresholding.

        Parameters
        ----------
        X:
            Union[numpy.ndarray, pandas.DataFrame, pandas.Series,
            dask.array, dask.DataFrame, dask.Series] of shape (n_samples,
            n_features) - The data used to determine the uniques and
            their frequencies, used for later thresholding along the
            feature axis.

        y:
            Optional[Union[numpy.ndarray, pandas.DataFrame, pandas.Series, None]]
            of shape (n_samples, n_outputs), or (n_samples,),
            default=None - Target relative to X; None for unsupervised
            learning.


        Return
        ------
        -
            self : object -
                Fitted min count transformer.


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
        X : Union[numpy.ndarray, pandas.DataFrame, pandas.Series,
            dask.array, dask.DataFrame, dask.Series] of shape (n_samples,
            n_features) - The data used to determine the uniques and
            their frequencies and to be transformed by rules created
            from those frequencies.

        y : Optional[Union[numpy.ndarray, pandas.DataFrame, pandas.Series]]
            of shape (n_samples, n_outputs), or (n_samples,), default=None - Target values
            (None for unsupervised transformations).

        Return
        ------
        -
            X_tr : {ndarray, pandas.DataFrame, pandas.Series} of shape
                    (n_samples_new, n_features_new)
                Transformed data.

            y_tr : {ndarray, pandas.DataFrame, pandas.Series} of shape
                    (n_samples_new,) or (n_samples_new, n_outputs)
                Transformed target.
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
        input_features:Optional[Union[Iterable[str], None]]=None
    ):
        """
        Get the feature names for the output of :method: transform.


        Parameters
        ----------
        input_features :
            Optional[Iterable[str], None]], default=None - Externally provided
            feature names for the fitted data, not the transformed data.

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
            feature_names_out : NDArray[object] - The feature names of
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
        indices : bool, default=False - If True, the return value will
            be an array of integers, rather than a boolean mask.

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


    def get_support(self, indices:bool=False):
        """Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        indices : bool, default=False - If True, the return value will
            be an array of integers rather than a boolean mask.

        Return
        ------
        -
            support : ndarray - An index that selects the retained
            features from a feature vector. If indices is False, this is
            a boolean array of shape (n_input_features,) in which an
            element is True if its corresponding feature is selected for
            retention. If indices is True, this is an integer array of
            shape (n_output_features, ) whose values are indices into
            the input feature vector.
        """

        check_is_fitted(self)

        # must use _make_instructions() in order to construct the column
        # support mask after fit and before a transform. otherwise, if an
        # attribute _column_support were assigned to COLUMN_KEEP_MASK in
        # transform() like _row_support is assigned to ROW_KEEP_MASK, then
        # a transform would have to be done before being able to access
        # get_support().

        COLUMNS = np.array(
            ['DELETE COLUMN' not in v for k, v in self._make_instructions().items()]
        )

        if indices is False:
            return np.array(COLUMNS)
        elif indices is True:
            return np.arange(self.n_features_in_)[COLUMNS].astype(np.uint32)


    def print_instructions(
        self,
        *,
        clean_printout:bool=True,
        max_print_len:int=100
    ):

        # pizza add this method to the main docs

        # pizza make test for this

        """
        Display instructions generated for the current fitted state,
        subject to the current settings of the parameters. The printout
        will indicate what rows / columns will be deleted, and if all
        columns or all rows will be deleted. Use :method: set_params to
        change MCTs parameters and see the impact on the transformation.

        If the instance has multiple recursions (i.e., :param:
        max_recursions is > 1, parameters cannot be changed via :method:
        set_params, but the net effect of the actual transformation that
        was performed is displayed (remember that multiple recursions
        can only be accessed through :method: fit_transform). The results
        are displayed as a single set of instructions, as if to perform
        the cumulative effect of the recursions in a single step.


        Parameters
        ----------
        _clean_printout:
            bool - Truncate printout to fit on screen.
        _max_print_len:
            int - the maximum number of characters to display per line.


        Return
        ------
        -
            None


        """


        check_is_fitted(self)

        # params can be changed after fit & before calling this by set_params().
        # need to validate params. _make_instructions() handles the validation
        # of almost all the params in __init__ except max_recursions,
        # reject_unseen_params, and n_jobs. max_recursions cannot be changed
        # in set_params once fitted. None of these 3 are used here.

        # after fit, ic & hab are blocked from being set to callable (but is OK
        # if was already a callable when fit.) that means that the mapping of
        # the callable used during fit lives in self._ignore_columns and
        # self._handle_as_bool. but, set_params does not block setting ic/hab
        # to Iterable[str] or Iterable[int]. so if self.ignore_columns and/or
        # self.handle_as_bool are callable, we need to pass the output that
        # lives in _ic & _hab. but if not callable, need to use the (perhaps
        # changed) ic/hab in self.ignore_columns & self.handle_as_bool.
        # if ic/hab were changed to Iterable[str] in set_params, need to map
        # to Iterable[int].

        # _ic_hab_condition takes X, but we dont have it so we need to spoof it.
        # X doesnt matter here, X is only for ic/hab callable in partial_fit()
        # and transform(). since we are always passing ic/hab as vectors, dont
        # need to worry about the callables.


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

        _repr_instructions(
            _delete_instr=self._make_instructions(),
            _total_counts_by_column=self._total_counts_by_column,
            _thresholds=self.count_threshold,
            _n_features_in=self.n_features_in_,
            _feature_names_in=getattr(self, 'feature_names_in_', None),
            _clean_printout=clean_printout,
            _max_print_len=max_print_len
        )


    def reset(self):
        """
        Reset the internal data state of MinCountTransformer.

        """

        try: del self.n_features_in_
        except: pass

        try: del self.feature_names_in_
        except: pass

        try: del self._ignore_columns
        except: pass

        try: del self._handle_as_bool
        except: pass

        try: del self._n_rows_in
        except: pass

        try: del self._original_dtypes
        except: pass

        try: del self._row_support
        except: pass


    def score(self, X, y=None) -> None:
        """
        Dummy method to spoof dask Incremental and ParallelPostFit
        wrappers. Verified must be here for dask wrappers.
        """

        pass


    # def set_output(self, transform) - inherited from SetOutputMixin


    def set_params(self, **params):
        """
        Set the parameters of the MCT instance.

        Pass the exact parameter name and its value as a keyword argument
        to the 'set_params' method call. Or use ** dictionary unpacking
        on a dictionary keyed with exact parameter names and the new
        parameter values as the dictionary values. Valid parameter keys
        can be listed with get_params(). Note that you can directly set
        the parameters of MinCountTransformer.

        Once MCT is fitted, MCT :method: set_params blocks some
        parameters from being set to ensure the validity of results.
        In these cases, to use different parameters without creating a
        new instance of MCT, call MCT :method: 'reset' on the instance.
        Otherwise, create a new MCT instance.

        'max_recursions' is always blocked when MCT is in a fitted state.

        If MCT was fit with :param: 'max_recursions' >= 2  (only a
        fit_transform could be done), all parameters are blocked. To
        break the block, call :method: 'reset' before calling :method:
        'set_params'. All information learned from any prior
        'fit_transform' will be lost.

        Also, when MCT has been fitted, :params: 'ignore_columns' and
        'handle_as_bool' cannot be set to a callable (they can, however,
        be changed to None, Iterable[int], or Iterable[str]). To set
        these parameters to a callable when MCT is in a fitted state,
        call :method: 'reset' then use :method: 'set_params' to set them
        to a callable. All information learned from any prior fit(s)
        will be lost when calling :method: 'reset'.


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
        X: XContainer,
        y: Optional[YContainer]=None,
        copy=False
    ) -> Union[tuple[XContainer, YContainer], XContainer]:

        """
        Reduce X by the thresholding rules found during fit.

        Parameters
        ----------
        X:
            Union[numpy.ndarray, pandas.DataFrame, scipy.sparse] of shape
            (n_samples, n_features) - The data that is to be reduced
            according to the thresholding rules found during :term: fit.

        y:
            Union[numpy.ndarray, pandas.DataFrame, pandas.Series, None]
            of shape (n_samples, n_outputs), or (n_samples,), default =
            None - Target values (None for unsupervised transformations).


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

        # this was for diagnosing why dask_ml wrappers are changing a
        # dask 200x20 array to 1x20 in dask_wrappers_test
        # print(f'pizza print in main very top of transform {X.shape=}')
        # print(f'pizza print in main very top of transform {type(X)=}')
        # print(f'pizza print in main very top of transform {X=}')
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

            # 24_06_17 When dask_ml Incremental and ParallelPostFit (versions
            # 2024.4.4 and 2023.5.27 at least) are passed y = None, they are
            # putting y = ('', order[i]) into the dask graph for y and sending
            # that junk as the value of y to the wrapped partial fit method.
            # All use cases where y=None are like this, it will always happen
            # and there is no way around it. To get around this, diagnose if
            # MCT is wrapped with Incr and/or PPF, then look to see if y is
            # of the form tuple(str, int). If both are the case, override y
            # with y = None.

            # Determine if the MCT instance is wrapped by dask_ml Incremental
            # or ParallelPostFit by checking the stack and looking for the
            # '_partial' method used by those modules. Wrapping with these
            # modules imposes limitations on passing a value to y.
            # pizza do we even need this? isnt an obscure object like
            # ('', order[i]) enough to convince u that its wrapped by dask?
            _using_dask_ml_wrapper = False
            for frame_info in inspect.stack():
                _module = inspect.getmodule(frame_info.frame)
                if _module:
                    if _module.__name__ == 'dask_ml._partial':
                        _using_dask_ml_wrapper = True
                        break
            del _module

            _is_garbage_y_from_dask_ml = False
            if isinstance(y, tuple) and isinstance(y[0], str) and isinstance(y[1], int):
                _is_garbage_y_from_dask_ml = True

            if _using_dask_ml_wrapper and _is_garbage_y_from_dask_ml:
                y = None

            del _using_dask_ml_wrapper, _is_garbage_y_from_dask_ml
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
        # MUST VALIDATE ON ALL PASSES BECAUSE _ignore_columns AND/OR
        # _handle_as_bool CAN BE CALLABLE BASED ON X AND X IS NEW EVERY
        # TIME SO THE CALLABLES MUST BE RECOMPUTED AND RE-VALIDATED
        # BECAUSE THEY MAY (UNDESIRABLY) GENERATE DIFFERENT IDXS.
        # _ignore_columns MUST BE BEFORE _make_instructions

        # pizza remember to test callables that spew differing output
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
                        deepcopy(self._total_counts_by_column),
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

        if y_tr is not None:
            return X_tr, y_tr
        else:
            return X_tr


    def _make_instructions(
        self,
        _threshold:Optional[Union[int, Iterable[int], None]]=None
    ):

        # must be before _make_instructions()
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



































