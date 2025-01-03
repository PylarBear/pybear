# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import Self, Union
from ._type_aliases import (
    KeepType, DataFormatType
)

from numbers import Real, Integral
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as ss

from ._validation._validation import _validation
from ._validation._X import _val_X
from ._partial_fit._find_constants import _find_constants
from ._shared._make_instructions import _make_instructions
from ._shared._set_attributes import _set_attributes
from ._shared._manage_keep import _manage_keep
from ._inverse_transform._inverse_transform import _inverse_transform
from ._transform._transform import _transform

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_array

from ...base import check_is_fitted, get_feature_names_out




class InterceptManager(BaseEstimator, TransformerMixin):

    """
    InterceptManager (IM) is a scikit-style transformer that identifies
    and manages the constant columns in a dataset.

    A dataset may contain columns with constant values for a variety of
    reasons, some intentional, some circumstantial. The use of a column
    of constants in a dataset may be a design consideration for some
    data analytics algorithms, such as multiple linear regression.
    Therefore, the presence of one such column may be desirable.

    The presence of multiple constant columns is generally a degenerate
    condition. In many data analytics learning algorithms, such a
    condition can cause convergence problems, inversion problems, or
    other undesirable effects. The analyst is often forced to address
    the issue to perform a meaningful analysis of the data.

    IM has several key characteristics that make it a versatile and
    powerful tool that can help fix this condition.

    IM...
    1) handles numerical and non-numerical data
    2) accepts nan-like values, and has flexibility in dealing with them
    3) has a partial fit method for block-wise fitting and transforming
    4) uses joblib for parallelized discovery of constant columns
    5) has parameters that give flexibility to how 'constant' is defined
    6) can remove all, selectively keep one, or append a column of
        constants to a dataset

    The methodology that IM uses to identify a constant column is
    different for numerical and non-numerical data.

    In the simplest situation with non-numerical data where nan-like
    values are not involved, the computation is simply to determine the
    number of unique values in the column. If there is only one unique
    value, then the column is constant.

    The computation for numerical columns is slightly more complex.
    IM calculates the mean of the column then compares it against the
    individual values via numpy.allclose. allclose has rtol and atol
    parameters that give latitude to the definition of 'equal'. They
    provide a tolerance window whereby numerical data that are not
    exactly equal are considered equal if their difference falls within
    the tolerance. IM affords some flexibility in defining 'equal' for
    the purpose of identifying constants by providing direct access to
    the numpy.allclose rtol and atol parameters through its own rtol and
    atol parameters. IM requires rtol and atol be non-boolean,
    non-negative real numbers, in addition to any other restrictions
    enforced by numpy.allclose. See the numpy docs for clarification of
    the technical details.

    The equal_nan parameter controls how IM handles nan-like values. If
    equal_nan is True, exclude any nan-like values from the allclose
    comparison. This essentially assumes that the nan values are equal
    to the mean of the non-nan values within their column. nan-like
    values will not in and of themselves cause a column to be considered
    non-constant when equal_nan is True. If equal_nan is False, IM does
    not make the same assumption that the nan values are implicitly equal
    to the mean of the non-nan values, thus making the column not
    constant. This is in line with the normal numpy handling of nan-like
    values. See the Notes section below for a discussion on the handling
    of nan-like values.

    IM also has a 'keep' parameter that allows the user to manage the
    constant columns that are identified. 'keep' accepts several types
    of arguments. The 'Keep' discussion section has a list of all the
    options that can be passed to :param: 'keep', what they do, and how
    to use them.

    The partial_fit, fit, fit_transform, and inverse_transform methods
    of IM accept data as numpy arrays, pandas dataframes, and scipy
    sparse matrices/arrays. IM has a set_output method, whereby the user
    can set the type of output container for :method: transform. This
    behavior is managed by scikit-learn functionality adopted into IM,
    and is subject to change at their discretion. :method: set_output
    can return transformed outputs as numpy arrays, pandas dataframes,
    or polars dataframes. When :method: set_output is None, the output
    container is the same as the input, that is, numpy array, pandas
    dataframe, or scipy sparse matrix/array.

    The partial_fit method allows for incremental fitting of data. This
    makes IM suitable for use with packages that do batch-wise fitting
    and transforming, such as dask_ml via the Incremental and
    ParallelPostFit wrappers.

    There are no safeguards in place to prevent the user from changing
    rtol, atol, or equal_nan between partial fits. These 3 parameters
    have strong influence over whether IM classifies a column as
    constant, and therefore is instrumental in dictating what IM learns
    during fitting. Changes to these parameters between partial fits can
    drastically change IM's understanding of the constant columns in
    the data versus what would otherwise be learned under constant
    settings. pybear recommends against this practice, however, it is
    not strictly blocked.

    When performing multiple batch-wise transformations of data, that is,
    making sequential calls to :method: transform, it is critical that
    the same column indices be kept / removed at each call. This issue
    manifests when :param: keep is set to 'random'; the random index
    to keep must be the same at all calls to :method: transform, and
    cannot be dynamically randomized within :method: transform. IM
    handles this by generating a static random index to keep at fit time,
    and does not change this number during transform time. This number
    is dynamic with each call to :method: partial_fit, and will likely
    change at each call. Fits performed after calls to :method: transform
    will change the random index away from that used in the previous
    transforms, causing IM to perform entirely different transformations
    than those previously being done. IM cannot block calls to :method:
    partial_fit after calls to :method: transform, but pybear strongly
    discourages this practice because the output will be nonsensical.
    pybear recommends doing all partial fits consecutively, then doing
    all transformations consecutively.


    The 'keep' Parameter
    --------------------
    IM learns which columns are constant during fitting. At transform,
    IM applies the instruction given to it via the 'keep' parameter.
    The 'keep' parameter takes several types of arguments, providing
    various ways to manage the columns of constants within a dataset.
    Below is a comprehensive list of all the arguments that can be
    passed to :param: keep.

    Literal 'first':
        Retains the constant column left-most in the data (if any) and
        deletes any others. Must be lower case. Does not except if there
        are no constant columns.
    Literal 'last':
        The default setting, keeps the constant column right-most in the
        data (if any) and deletes any others. Must be lower case. Does
        not except if there are no constant columns.
    Literal 'random':
        Keeps a single randomly-selected constant column (if any) and
        deletes any others. Must be lower case. Does not except if there
        are no constant columns.
    Literal 'none':
        Removes all constant columns (if any). Must be lower case. Does
        not except if there are no constant columns.
    integer:
        An integer indicating the column index in the original data to
        keep, while removing all other columns of constants. IM will
        raise an exception if this passed index is not a column of
        constants.
    string: A string indicating feature name to keep if a pandas
        dataframe is passed, while deleting all other constant columns.
        Case sensitive. IM will except if 1) a string is passed that is
        not an allowed string literal ('first', 'last', 'random', 'none')
        and a pandas dataframe is not passed to :method: fit, 2) a pandas
        dataframe is passed to :method: fit but the given feature name
        is not valid, 3) the feature name is valid but the column is not
        constant.
    callable(X): a callable that returns a valid column index when the
        data is passed to it, indicating the index of the column of
        constants to keep while deleting all other columns of constants.
        This enables the analyst to use characteristics of the data
        being transformed to determine which column of constants to
        keep. IM passes the data as-is directly to the callable without
        any preprocessing. The callable needs to operate on the data
        object directly.
        IM will except if 1) the callable does not return an integer,
        2) the integer returned is out of the range of columns in the
        passed data, 3) the integer that is returned does not correspond
        to a constant column.
        IM does not retain state information about what indices have
        been returned from the callable during transforms. IM cannot
        catch if the callable is returning different indices for
        different blocks of data within a sequence of calls to :method:
        transform. When doing multiple batch-wise transforms, it is up
        to the user to ensure that the callable returns the same index
        for each transform. If the callable returns a different index
        for any of the blocks of data passed in a sequence of transforms
        then the results will be nonsensical.
    dictionary[str, any]: dictionary of {feature name:str, constant
        value:any}. A column of constants is appended to the right
        end of the data, with the constant being the value in the
        dictionary. The :param: 'keep' dictionary requires a single
        key:value pair.
        The key must be a string indicating feature name. This applies
        to any format of data that is transformed. If the data is a
        pandas dataframe, then this string will become the feature name
        of the new constant feature. If the fitted data is a numpy array
        or scipy sparse, then this column name is ignored.
        The dictionary value is the constant value for the new feature.
        This value has only two restrictions: it cannot be a non-string
        iterable (e.g. list, tuple, etc.) and it cannot be a callable.
        Essentially, the constant value is restricted to being integer,
        float, string, or boolean.
        When appending a constant value to a pandas df, if the constant
        is numeric it is appended as float64; if it is not numeric it is
        appended as object. Otherwise, if the constant is being appended
        to a numpy array or scipy sparse it will be forced to the dtype
        of the transformed data (with some caveats.)
        When transforming a pandas dataframe and the new feature name is
        already a feature in the data, there are two possible outcomes.
        1) If the original feature is not constant, the new constant
        values will overwrite in the old column (generally an undesirable
        outcome.) 2) If the original feature is constant, the original
        column will be removed and a new column with the same name will
        be appended with the new constant values. IM will warn about
        this condition but not terminate the program. It is up to the
        user to manage the feature names in this situation.
        :attr: column_mask_ is not adjusted for the new feature appended
        by the :param: 'keep' dictionary (see the discussion on :attr:
        column_mask_.) But the :param: 'keep' dictionary does make
        adjustment to :method: get_feature_names_out. As :method:
        get_feature_names_out reflects the characteristics of transformed
        data, and the :param: 'keep' dictionary modifies the data at
        transform time, then :method: get_feature_names_out also reflects
        this modification.

    To access the :param: keep literals ('first', 'last', 'random',
    'none'), these MUST be passed as lower-case. If a pandas dataframe
    is passed and there is a conflict between a literal that has been
    passed and a feature name, IM will raise because it is not clear to
    IM whether you want to indicate the literal or the feature name. To
    afford a little more flexibility with feature names, IM does not
    normalize case for this parameter. This means that if :param: keep
    is passed as 'first',  feature names such as 'First', 'FIRST',
    'FiRsT', etc. will not raise, only 'first' will.

    The only value that removes all constant columns is 'none'. All other
    valid arguments for :param: 'keep' leave one column of constants
    behind. All other constant columns are removed from the dataset.
    If IM does not find any constant columns, 'first', 'last', 'random',
    and 'none' will not raise an exception. It is like telling IM: "I
    dont know if there are any constant columns, but if you find some,
    then apply this rule." However, if using an integer, feature name,
    or callable, IM will raise an exception if it does not find a
    constant column at that index. It is like telling IM: "I know that
    this column is constant, and you need to keep it and remove any
    others." If IM finds that it is not constant, it will raise an
    exception because you lied to it.


    Parameters
    ----------
    keep:
        Optional[Union[Literal['first', 'last', 'random', 'none'],
        dict[str, any], int, str, callable[[X], int], default='last' -
        The strategy for handling the constant columns. See 'The keep
        Parameter' section for a lengthy explanation of the 'keep'
        parameter.
    equal_nan:
        Optional[bool], default=True - If equal_nan is True, exclude
        nan-likes from computations that discover constant columns.
        This essentially assumes that the nan value would otherwise be
        equal to the mean of the non-nan values in the same column.
        If equal_nan is False and any value in a column is nan, do not
        assume that the nan value is equal to the mean of the non-nan
        values in the same column, thus making the column non-constant.
        This is in line with the normal numpy handling of nan values.
    rtol:
        Optional[numbers.Real], default = 1e-5 - The relative difference
        tolerance for equality. Must be a non-boolean, non-negative,
        real number. See numpy.allclose.
    atol:
        Optional[numbers.Real], default = 1e-8 - The absolute difference
        tolerance for equality. Must be a non-boolean, non-negative,
        real number. See numpy.allclose.
    n_jobs:
        Optional[Union[numbers.Integral, None]], default=-1 - The number
        of joblib Parallel jobs to use when scanning the data for columns
        of constants. The default is to use processes, but can be
        overridden externally using a joblib parallel_config context
        manager. The default number of jobs is -1 (all processors). To
        get maximum speed benefit, pybear recommends using the default
        setting.


    Notes
    -----
    Concerning the handling of nan-like representations. While IM
    accepts data in the form of numpy arrays, pandas dataframes, and
    scipy sparse matrices/arrays, during the search for constants
    each column is separately converted to a 1D numpy array (see below
    for more detail about how scipy sparse is handled.) After the
    conversion to numpy 1D array and prior to calculating the mean and
    applying numpy.allclose, IM identifies any nan-like representations
    in the numpy array and standardizes all of them to numpy.nan. The
    user needs to be wary that whatever is used to indicate 'not-a-number'
    in the original data must first survive the conversion to numpy
    array, then be recognizable by IM as nan-like, so that IM can
    standardize it to numpy.nan. nan-like representations that are
    recognized by IM include, at least, numpy.nan, pandas.NA, None (of
    type None, not string 'None'), and string representations of 'nan'
    (not case sensitive).

    Concerning the handling of infinity. IM has no special handling
    for numpy.inf, -numpy.inf, float('inf') or float('-inf'). IM falls
    back to the native handling of these values for python and numpy.
    Specifically, numpy.inf==numpy.inf and float('inf')==float('inf').

    Concerning the handling of scipy sparse arrays. When searching for
    constant columns, the columns are converted to dense 1D numpy arrays
    one at a time. Each column is sliced from the data in sparse form
    and is converted to numpy ndarray via the 'toarray' method. This a
    compromise that causes some memory expansion but allows for efficient
    handling of constant column calculations that would otherwise
    involve implicit non-dense values.


    Attributes
    ----------
    n_features_in_:
        int - number of features in the fitted data before transform.

    feature_names_in_:
        NDArray[str] - The names of the features as seen during fitting.
        Only accessible if X is passed to :methods: partial_fit or fit
        as a pandas dataframe that has a header.

    constant_columns_:
        dict[int, any] - A dictionary whose keys are the indices of the
        constant columns found during fit, indexed by their column
        location in the original data. The dictionary values are the
        constant values in those columns. For example, if a dataset has
        two constant columns, the first in the third index and the
        constant value is 1, and the other is in the tenth index and the
        constant value is 0, then constant_columns_ will be {3:1, 10:0}.
        If there are no constant columns, then constant_columns_ is an
        empty dictionary.

    kept_columns_:
        dict[int, any] - A subset of the constant_columns_ dictionary,
        constructed with the same format. This holds the subset of
        constant columns that are retained in the data. If a constant
        column is kept, then this contains one key:value pair from
        constant_columns_. If there are no constant columns or no columns
        are kept, then this is an empty dictionary. When :param: 'keep'
        is a dictionary, all the original constant columns are removed
        and a new constant column is appended to the data. That column
        is NOT included in kept_columns_.

    removed_columns_:
        dict[int, any] - A subset of the constant_columns_ dictionary,
        constructed with the same format. This holds the subset of
        constant columns that are removed from the data. If there are
        no constant columns or no constant columns are removed, then
        this is an empty dictionary.

    column_mask_:
        NDArray[bool] - shape (n_features_,) - Indicates which columns
        of the fitted data are kept (True) and which are removed (False)
        during transform. When :param: keep is a dictionary, all original
        constant columns are removed and a new column of constants is
        appended to the data. This new column is NOT appended to
        column_mask_. This mask is intended to be applied to data of the
        same dimension as that seen during fit, and the new column of
        constants is a feature added after transform.


    See Also
    --------
    numpy.ndarray
    pandas.core.frame.DataFrame
    scipy.sparse
    numpy.allclose
    numpy.isclose
    numpy.unique


    Examples
    --------
    >>> from pybear.preprocessing import InterceptManager as IM
    >>> import numpy as np
    >>> np.random.seed(99)
    >>> X = np.random.randint(0, 10, (5, 5))
    >>> X[:, 1] = 0
    >>> X[:, 2] = 1
    >>> print(X)
    [[1 0 1 8 9]
     [8 0 1 5 4]
     [1 0 1 7 1]
     [1 0 1 4 7]
     [2 0 1 8 4]]
    >>> trf = IM(keep='first', equal_nan=True)
    >>> trf.fit(X)
    InterceptManager(keep='first')
    >>> out = trf.transform(X)
    >>> print(out)
    [[1 0 8 9]
     [8 0 5 4]
     [1 0 7 1]
     [1 0 4 7]
     [2 0 8 4]]
    >>> print(trf.n_features_in_)
    5
    >>> print(trf.constant_columns_)
    {1: np.float64(0.0), 2: np.float64(1.0)}
    >>> print(trf.removed_columns_)
    {2: np.float64(1.0)}
    >>> print(trf.column_mask_)
    [ True  True False  True  True]


    """


    _parameter_constraints: dict = {
        "keep": [
            StrOptions({"first", "last", "random", "none"}),
            dict, Integral, str, callable
        ],
        "equal_nan": ["boolean"],
        "rtol": [Real],
        "atol": [Real],
        "n_jobs": [Integral, None]
    }


    def __init__(
        self,
        *,
        keep: Optional[KeepType]='last',
        equal_nan: Optional[bool]=True,
        rtol: Optional[Real]=1e-5,
        atol: Optional[Real]=1e-8,
        n_jobs: Optional[Union[Integral, None]]=-1
    ):

        self.keep = keep
        self.equal_nan = equal_nan
        self.rtol = rtol
        self.atol = atol
        self.n_jobs = n_jobs



    def _reset(self):

        """
        Reset internal data-dependent state of InterceptManager.
        __init__ parameters are not changed.

        """

        if hasattr(self, 'constant_columns_'):
            del self.constant_columns_
            del self.kept_columns_
            del self.removed_columns_
            del self.column_mask_


    def get_feature_names_out(self, input_features=None):

        """
        Return the feature names for the output of :method: transform.
        When :param: 'keep' is a dictionary, the appended column of
        constants is included in the outputted feature name vector.


        Parameters
        ----------
        input_features :
            array-like of str or None, default=None - Externally provided
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

        # get_feature_names_out() would otherwise be provided by
        # pybear.base.GFNOMixin, but since this transformer deletes
        # and/or adds columns, must build a one-off.

        # when there is a {'Intercept': 1} in :param: keep, need to make
        # sure that that column is accounted for here, and the dropped
        # columns are also accounted for.

        check_is_fitted(self)

        feature_names_out = get_feature_names_out(
            input_features,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            self.n_features_in_
        )

        feature_names_out = feature_names_out[self.column_mask_]

        if isinstance(self.keep, dict):
            feature_names_out = np.hstack((
                  feature_names_out,
                  list(self.keep.keys())[0]
            )).astype(object)

        return feature_names_out


    def get_metadata_routing(self):
        """
        Get metadata routing is not implemented in InterceptManager.

        """
        __ = type(self).__name__
        raise NotImplementedError(
            f"get_metadata_routing is not implemented in {__}"
        )


    # def get_params - inherited from BaseEstimator
    # if ever needed, hard code that can be substituted for the
    # BaseEstimator get/set_params can be found in GSTCV_Mixin


    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(
        self,
        X:DataFormatType,
        y:any=None
    ) -> Self:

        """
        Perform incremental fitting on one or more blocks of data.
        Determine the constant columns in the given data, subject to the
        criteria defined in :params: rtol, atol, and equal_nan.


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - Data to find constant columns in.
        y:
            {vector-like of shape (n_samples,) or None}, default = None -
            ignored. The target for the data.


        Return
        ------
        -
            self: the fitted InterceptManager instance.




        """

        # keep this before _validate_data. when X is junk, _validate_data
        # and check_array except for varying reasons. this standardizes
        # the error message for non-np/pd/ss X.
        _val_X(X)

        # validation of X must be done here, not in a separate module
        # BaseEstimator has _validate_data method, which when called
        # exposes n_features_in_ and feature_names_in_.
        X = self._validate_data(
            X=X,
            reset=not hasattr(self, "constant_columns_"),
            cast_to_ndarray=False,
            # vvv become **check_params, and get fed to check_array()
            accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
            dtype=None,
            force_all_finite="allow-nan",
            ensure_2d=True,
            ensure_min_features=1,
            order='F'
        )

        # reset – Whether to reset the n_features_in_ attribute. If False,
        # the input will be checked for consistency with data provided when
        # reset was last True.
        # It is recommended to call reset=True in fit and in the first call
        # to partial_fit. All other methods that validate X should set
        # reset=False.
        #
        # cast_to_ndarray – Cast X and y to ndarray with checks in
        # check_params. If False, X and y are unchanged and only
        # feature_names_in_ and n_features_in_ are checked.


        # this must be after _validate_data, needs feature_names_in_ to
        # be exposed, if available.

        _validation(
            X,
            self.feature_names_in_ if \
                hasattr(self, 'feature_names_in_') else None,
            self.keep,
            self.equal_nan,
            self.rtol,
            self.atol,
            self.n_jobs
        )


        err_msg = (
            f"'X' must be a valid 2 dimensional numpy ndarray, pandas "
            f"dataframe, or scipy sparce matrix or array, with at least "
            f"1 column and 1 example."
        )

        # sklearn _validate_data & check_array are not catching this
        if len(X.shape) != 2:
            raise ValueError(err_msg)


        # sklearn _validate_data & check_array are not catching this
        if X.shape[1] < 1:
            raise ValueError(err_msg)


        # ss sparse that cant be sliced
        if isinstance(
            X,
            (ss.coo_matrix, ss.dia_matrix, ss.bsr_matrix, ss.coo_array,
             ss.dia_array, ss.bsr_array)
        ):

            warnings.warn(
                f"pybear works hard to avoid mutating or creating copies of "
                f"your original data. \nyou have passed your data as {type(X)}, "
                f"which cannot be sliced by columns. \npybear needs to create "
                f"a copy. \nto avoid this, pass your sparse data as csr, csc, "
                f"lil, or dok."
            )
            _X = X.copy().tocsc()
        else:
            _X = X


        # if IM has already been fitted and constant_columns_ is empty
        # (meaning there are no constant columns) dont even bother to
        # scan more data, cant possibly have constant columns
        if hasattr(self, 'constant_columns_') and self.constant_columns_ == {}:
            self.constant_columns_ = {}
        else:
            # dictionary of column indices and respective constant values
            self.constant_columns_: dict[int, any] = \
                _find_constants(
                    _X,
                    self.constant_columns_ if \
                        hasattr(self, 'constant_columns_') else None,
                    self.equal_nan,
                    self.rtol,
                    self.atol,
                    self.n_jobs
                )


        # Create an instance attribute that specifies the random column index
        # to keep when 'keep' is 'random'. This value must be static on calls
        # to :method: transform (meaning sequential calls to transform get the
        # same random index every time.) This  value is generated and retained
        # even if :param: 'keep' != 'random', in case :param: 'keep' should be
        # set to 'random' at any point via set_params().
        if len(self.constant_columns_):
            self._rand_idx = int(np.random.choice(list(self.constant_columns_)))
        else:
            self._rand_idx = None

        _keep = _manage_keep(
            self.keep,
            _X,
            self.constant_columns_,
            self.n_features_in_,
            self.feature_names_in_ if \
                hasattr(self, 'feature_names_in_') else None,
            self._rand_idx
        )

        self._instructions = _make_instructions(
            _keep,
            self.constant_columns_,
            self.n_features_in_
        )

        self.kept_columns_, self.removed_columns_, self.column_mask_ = \
            _set_attributes(
                self.constant_columns_,
                self._instructions,
                self.n_features_in_
            )

        return self


    def fit(
        self,
        X:DataFormatType,
        y:any=None
    ) -> Self:

        """
        Perform a single fitting on a data set. Determine the constant
        columns in the given data, subject to the criteria defined
        in :params: rtol, atol, and equal_nan.


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - Data to find constant columns in.
        y:
            {vector-like of shape (n_samples,) or None}, default = None -
            ignored. The target for the data.


        Return
        ------
        -
            self: the fitted InterceptManager instance.



        """

        self._reset()
        return self.partial_fit(X, y=y)


    def inverse_transform(
        self,
        X: DataFormatType,
        *,
        copy: bool = None
    ) -> DataFormatType:

        """
        Revert transformed data back to its original state. :method:
        set_output does not control the output container here, the output
        container is always the same as passed. This operation cannot
        restore any nan-like values that may have been in the original
        untransformed data.

        Very little validation is possible to ensure that the passed
        data is valid for the current state of IM. It is only possible
        to ensure that the number of columns in the passed data match
        the number of columns that are expected to be outputted by
        :method: transform for the current state of IM. It is up to the
        user to ensure the state of IM aligns with the state of the data
        that is to undergo inverse transform. Otherwise the output will
        be nonsensical.


        Parameters
        ----------
        X :
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_transform_features) - A transformed data set.
        copy:
            Union[bool, None], default=None - Whether to make a copy of
            X before the inverse transform.


        Return
        ------
        -
            X_tr : {array-like, scipy sparse matrix} of shape (n_samples,
                n_features) - Transformed data reverted to its original
                untransformed state.


        """

        check_is_fitted(self)

        if not isinstance(copy, (bool, type(None))):
            raise TypeError(f"'copy' must be boolean or None")

        # keep this before check_array. when X is junk, _validate_data
        # and check_array except for varying reasons. this standardizes
        # the error message for non-np/pd/ss X.
        _val_X(X)

        # dont assign to X, check_array always converts to ndarray
        check_array(
            array=X,
            accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
            dtype=None,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=copy or False,
            order='F',
            ensure_min_samples=1,  # this is doing squat, validated in _val_X
            ensure_min_features=1,
        )

        # if _keep is a dict, a column of constants was stacked to the right
        # side of the data. check that _keep is valid (may have changed via
        # set_params()), the passed data matches against _keep, and remove
        # the column
        if isinstance(self.keep, dict):

            _name = list(self.keep.keys())[0]

            err_msg = (
                f"when passed as a dictionary, :param: 'keep' must have "
                f"one key:value pair. The key must be a string and the "
                f"value must not be callable or list-like iterable."
            )
            # must be one entry and key must be str
            if len(self.keep) != 1 or not isinstance(_name, str):
                raise ValueError(err_msg)

            try:
                # if is callable, except
                if callable(self.keep[_name]):
                    raise BrokenPipeError
                iter(self.keep[_name])
                if isinstance(self.keep[_name], str):
                    raise Exception
                # if is any iterable beside string, except
                raise UnicodeError
            except BrokenPipeError:
                raise ValueError(
                    f"The 'keep' dictionary value is a callable, which IM "
                    f"does not allow. " + err_msg
                )
            except UnicodeError:
                raise ValueError(
                    f"The 'keep' dictionary value is a non-string iterable, "
                    f"which IM does not allow. " + err_msg
                )
            except:
                # accept anything that is string or not an iterable
                pass

            if isinstance(X, np.ndarray):
                _unqs = np.unique(X[:, -1])
            elif isinstance(X, pd.core.frame.DataFrame):
                _unqs = np.unique(X.iloc[:, -1].to_numpy())
            elif hasattr(X, 'toarray'):
                _unqs = np.unique(X.tocsc().getcol(-1).toarray())

            _key = list(self.keep.keys())[0]
            _value_matches = False
            try:
                _value = float(self.keep[_key])
                assert _value == _unqs[0]
                _value_matches = True
            except:
                if _unqs[0] == self.keep[_key]:
                    _value_matches = True

            if len(_unqs) == 1 and _value_matches:
                if isinstance(X, np.ndarray):
                    X = np.delete(X, -1, axis=1)
                elif isinstance(X, pd.core.frame.DataFrame):
                    X = X.drop(columns=[_key], inplace=False)
                elif hasattr(X, 'toarray'):
                    _og_dtype = type(X)
                    X = X.tocsc()[:, list(range(X.shape[1]-1))]
                    X = _og_dtype(X)
                    del _og_dtype
            else:
                raise ValueError(
                    f":param: 'keep' is a dictionary but the last column "
                    f"of the data to be inverse transformed does not match "
                    f"{_unqs[0]=}, {self.keep[_key]=}"
                )

            del _value_matches

        # the number of columns in X must be equal to the number of features
        # remaining in column_mask_
        if X.shape[1] != np.sum(self.column_mask_):
            raise ValueError(
                f"the number of columns in X must be equal to the number of "
                f"columns kept from the fitted data after removing constants "
                f"({np.sum(self.column_mask_)})"
            )

        X = _inverse_transform(
            X,
            self.removed_columns_,
            self.feature_names_in_ if \
                hasattr(self, 'feature_names_in_') else None
        )

        if isinstance(X, np.ndarray):
            X = np.ascontiguousarray(X)


        return X


    def score(self, X, y=None):
        """
        Dummy method to spoof dask Incremental and ParallelPostFit
        wrappers. Verified must be here for dask wrappers.
        """

        pass


    # def set_params(self) - inherited from TransformerMixin
    # if ever needed, hard code that can be substituted for the
    # BaseEstimator get/set_params can be found in GSTCV_Mixin

    # def set_output(self) - inherited from TransformerMixin


    def transform(
        self,
        X: DataFormatType,
        copy: bool=None
    ) -> DataFormatType:

        """
        Manage the constant columns in X. Apply the removal criteria
        given by :param: keep to the constant columns found during fit.


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - The data to be transformed.
        copy:
            Union[bool, None], default=None - Whether to make a copy of
            X before the transform.


        Return
        ------
        -
            X_tr: {array-like, scipy sparse matrix} of shape (n_samples,
                n_transformed_features) - The transformed data.


        """

        check_is_fitted(self)

        if not isinstance(copy, (bool, type(None))):
            raise TypeError(f"'copy' must be boolean or None")

        # keep this before _validate_data. when X is junk, _validate_data
        # and check_array except for varying reasons. this standardizes
        # the error message for non-np/pd/ss X.
        _val_X(X)

        X_tr = self._validate_data(
            X=X,
            reset=False,
            cast_to_ndarray=False,
            copy=copy or False,
            accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
            dtype=None,
            force_all_finite="allow-nan",
            ensure_2d=True,
            ensure_min_features=1,
            ensure_min_samples=1,  # this is doing squat, validated in _val_X
            order ='F'
        )

        _validation(
            X_tr,
            self.feature_names_in_ if \
                hasattr(self, 'feature_names_in_') else None,
            self.keep,
            self.equal_nan,
            self.rtol,
            self.atol,
            self.n_jobs
        )


        # everything below needs to be redone every transform in case 'keep'
        # was changed via set params after fit

        _keep = _manage_keep(
            self.keep,
            X_tr,
            self.constant_columns_,
            self.n_features_in_,
            self.feature_names_in_ if \
                hasattr(self, 'feature_names_in_') else None,
            self._rand_idx
        )

        self._instructions = _make_instructions(
            _keep,
            self.constant_columns_,
            self.n_features_in_
        )

        self.kept_columns_, self.removed_columns_, self.column_mask_ = \
            _set_attributes(
                self.constant_columns_,
                self._instructions,
                self.n_features_in_
            )

        X_tr = _transform(X_tr, self._instructions)

        if isinstance(X_tr, np.ndarray):
            X_tr = np.ascontiguousarray(X_tr)

        return X_tr















