# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# pizza dont forget to clean up these imports!!
from typing import Optional
from typing_extensions import Self
from ._type_aliases import (
    KeepType, DataFormatType
)
from numbers import Real, Integral

import numpy as np

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
from sklearn.utils.validation import check_is_fitted, check_array




class InterceptManager(BaseEstimator, TransformerMixin):

    """

    InterceptManager (IM) is a scikit-style transformer that identifies
    and manages the constant columns in a dataset.

    Columns with constant values within the same dataset may occur
    for a variety of reasons, some intentional, some circumstantial.
    The use of a column of constants in a dataset may be a design
    consideration for some data analytics algorithms, such as multiple
    linear regression. Therefore, the existence of one such column may
    be desirable.

    The presence of multiple constant columns is generally a degenerate
    condition. In many data analytics learning algorithms, such a
    condition can cause convergence problems, inversion problems, or
    other undesirable effects. The analyst is often forced to address
    the issue to perform a meaningful analysis of the data.

    IM is a tool that can help fix this condition, and has several key
    characteristics that make it versatile and powerful.

    IM...
    1) handles numerical and non-numerical data
    2) accepts nan-like values, and has flexibility in dealing with them
    3) has a partial fit method for block-wise fitting and transforming
    4) uses joblib for parallelized discovery of constant columns
    5) has parameters that give flexibility to the definition of 'constant'
    6) can remove all, selectively keep one, or append a column of constants to a dataset

    IM affords parameters that give some flexibility to the definition
    of 'equal' for the sake of identifying constants. Namely, the
    'rtol', 'atol', and 'equal_nan' parameters.

    The methodology that IM uses to identify constant columns is to
    calculate the mean of a column then compare it against the individual
    values for equality via numpy.allclose with respect to the rtol and
    atol parameters.

    The rtol and atol parameters provide a tolerance window whereby
    numerical data that are not exactly equal are considered equal if their
    difference falls within the tolerance. See the numpy docs for
    clarification of the technical details. If
    all values fall within the tolerance window then the column is considered constant.
    IM requires rtol and atol be non-boolean, non-negative real numbers,
    in addition to any other restrictions
    IM requires that rtol and atol be non-boolean,
    enforced by numpy.allclose.

    The equal_nan parameter controls how IM handles nan-like
    representations. If equal_nan is True, exclude
    any nan-like values from the computation; this essentially assumes
    that the nan values are equal to the mean of the non-nan values
    within their column.  nan-like values will
    not in and of themselves cause a pair of columns to be marked as unequal.
    If equal_nan is False, IM does not make the same equality assumption, thus making
    the column pair not constant. This is in line
    with the normal numpy handling of nan values. See the Notes section
    below for a discussion on the handling of nan-like values.

    IM also has a 'keep' parameter that allows the user to manage the
    constant columns that are identified. 'keep' accepts several different
    types of values. 'first', 'last', 'random', 'none', an integer indicating
    column index, a string indicating feature name if a pandas dataframe
    is passed, a callable that returns a valid column index when the
    fitted data is passed to it, or a dictionary of {column name, constant value}.
    The default setting is 'last'.

    That's a lot of options! Some general rules to help simplify the
    situation.
    The only value that removes all constant columns is 'none'. All other
    valid values leave one column of constants behind.
    If IM does not find any constant columns, 'first', 'last', 'random', and
    'none' will not raise an exception. When using these values you are
    saying to IM: "I dont know if there are any constant columns, but if you
    find some, then apply this rule. However, if using the integer, feature name,
    of callable, IM will raise an exception if IM does not find a constant
    column there. What you are saying to IM is: "I know that this column is
    constant, and you need to keep it and remove any others." If IM finds
    that it is not constant, it will raise an exception because you lied to it.


    'first': retains the constant column left-most in the data (if any)
    'last': keeps the constant column right-most in the data (if any)
    'random': keeps a single randomly-selected column of the set of constants.
    All other constant columns are removed from the dataset.

    pizza talk about callable keep, possible variable output
    pizza dict keep does not make adjustment to column_mask_, but does
    make adjustment to get_feature_names_out.
    # pizza, this is for :param: keep
    to access the :param: keep literals ('first', 'last', 'random', 'none'),
    these MUST be passed as lower-case. If a pandas dataframe is passed
    and there is a conflict between the literals and a feature name, IM
    will raise because it is not clear to IM whether you want to indicate
    the literal or the feature name. to afford a little more flexibility
    with feature names, IM does not normalize case for this
    parameter. This means that if :param: keep is 'first',  feature names such as
    'First', 'FIRST', 'FiRsT', etc. will not raise, only 'first' will.

    The partial_fit, fit, fit_transform, and inverse_transform methods
    of CDT accept data as numpy arrays, pandas dataframes, and scipy
    sparse matrices/arrays. CDT has a set_output method,
    whereby the user can set the type of output container. This behavior
    is managed by scikit-learn functionality adopted into CDT, and is
    subject to change at their discretion. As of first publication,
    :method: set_output can return transformed outputs as numpy arrays,
    pandas dataframes, or polars dataframes. When :method: set_output is
    None, the output container is the same as the input, that is, numpy
    array, pandas dataframe, or scipy sparse matrix/array.

    The partial fit method allows for batch-wise fitting of data. This
    makes CDT suitable for use with packages that do batch-wise fitting
    and transforming, such as dask_ml via the Incremental and
    ParallelPostFit wrappers.

    pizza talk about setting equal_nan, rtol, atol between partial fits
    There are no safeguards in place to prevent the user from changing
    rtol, atol, or equal_nan between partial fits. These 3 parameters
    have strong influence over whether CDT classifies two columns as
    equal, and therefore are instrumental in dictating what CDT learns
    during fitting. Changes to these parameters between partial fits can
    drastically change CDT's understanding of the duplicate columns in
    the data versus what would otherwise be learned under constant
    settings. pybear recommends against this practice, however, it is
    not strictly blocked.


    Parameters
    ----------
    keep:
        Optional[Union[Literal['first', 'last', 'random', 'none'], dict[str,any], None]], int, str, callable]
        default='last' -         The strategy for keeping a single representative from a set of
        identical columns. 'first' retains the column left-most in the
        data; 'last' keeps the column right-most in the data; 'random'
        keeps a single randomly-selected column of the set of duplicates.
    equal_nan:
        Optional[bool], default=True -         If equal_nan is True, exclude from comparison any rows where
        one or both of the values is/are nan. If one value is nan, this
        essentially assumes that the nan value would otherwise be the
        same as its non-nan counterpart. When both are nan, this
        considers the nans as equal (contrary to the default numpy
        handling of nan, where np.nan does not equal np.nan) and will
        not in and of itself cause a pair of columns to be marked as
        unequal. If equal_nan is False and either one or both of the
        values in the compared pair of values is/are nan, consider the
        pair to be not equivalent, thus making the column pair not equal.
        This is in line with the normal numpy handling of nan values.
    rtol:
        numbers.Real, default = 1e-5 - The relative difference tolerance
            for equality. See numpy.allclose.
    atol:
        numbers.Real, default = 1e-8 - The absolute tolerance parameter
            for equality. See numpy.allclose.
    n_jobs:
        # pizza finalize this based on benchmarking
        Optional[Integral], default=-1 - The number of joblib Parallel
        jobs to use when scanning the data for columns of constants. The
        default is to use processes, but can be overridden externally
        using a joblib parallel_config context manager. The default
        number of jobs is -1 (all processors). To get maximum speed
        benefit, pybear recommends using the default setting.
    n_jobs:
        Union[int, None], default = -1 - The number of joblib Parallel
        jobs to use when comparing columns. The default is to use
        processes, but can be overridden externally using a joblib
        parallel_config context manager. The default number of jobs is
        -1 (all processors). To get maximum speed benefit, pybear
        recommends using the default setting.




    Notes
    -----
    pizza, this is straight from CDT 24_11_11. review this.
    Concerning the handling of nan-like representations. While CDT
    accepts data in the form of numpy arrays, pandas dataframes, and
    scipy sparse matrices/arrays, at the time of column comparison both
    columns of data are converted to numpy arrays (see below for more
    detail about how scipy sparse is handled.) After the conversion
    and prior to the comparison, CDT identifies any nan-like
    representations in both of the numpy arrays and standardizes all of
    them to np.nan. The user needs to be wary that whatever is used to
    indicate 'not-a-number' in the original data must first survive the
    conversion to numpy array, then be recognizable by CDT as nan-like,
    so that CDT can standardize it to np.nan. nan-like representations
    that are recognized by CDT include, at least, np.nan, pandas.NA,
    None (of type None, not string 'None'), and string representations
    of 'nan' (not case sensitive).

    Concerning the handling of infinity. CDT has no special handling
    for np.inf, -np.inf, float('inf') or float('-inf'). CDT falls back
    to the native handling of these values for python and numpy.
    Specifically, numpy.inf==numpy.inf and float('inf')==float('inf').

    Concerning the handling of scipy sparse arrays. When comparing
    columns for equality, the columns are not converted to dense numpy
    arrays. Each column is sliced from the data in sparse form and the
    'indices' and 'data' attributes of this slice are stacked together.
    The single vector holding the indices and dense values is used to
    make equality comparisons.








    Attributes
    ----------
    n_features_in_:
        int - number of features in the fitted data before transform.

    feature_names_in_:
        NDArray[str] - The names of the features as seen during fitting.
        Only accessible if X is passed to :methods: partial_fit or fit
        as a pandas dataframe that has a header.

    constant_columns_:
        dict[int, any] - pizza!

    kept_columns_:
        dict[int, any] - pizza!

    removed_columns_:
        dict[int, any] - a dictionary whose keys are the
        indices of duplicate columns removed from the original data,
        indexed by their column location in the original data; the values
        are the column index in the original data of the respective
        duplicate that was kept.

    column_mask_:
        NDArray[bool] - shape (n_features_,) - Indicates which
        columns of the fitted data are kept (True) and which are removed
        (False) during transform.


    Attributes:
    -----------


    column_mask_: list[bool],





    See Also
    --------
    numpy.ndarray
    pandas.core.frame.DataFrame
    scipy.sparse
    numpy.isclose
    numpy.unique


    """


    _parameter_constraints: dict = {
        "keep": [StrOptions({"first", "last", "random", "none"}), dict, Integral, str, callable],
        "equal_nan": ["boolean"],
        "rtol": [Real],
        "atol": [Real],
        "n_jobs": [Integral, None]
    }


    def __init__(
        self,
        *,
        keep: KeepType='last',
        equal_nan: Optional[bool]=True,
        rtol: Optional[Real]=1e-5,
        atol: Optional[Real]=1e-8,
        n_jobs: Optional[Integral]=-1  # pizza benchmark what is the best setting
    ):

        self.keep = keep
        self.equal_nan = equal_nan
        self.rtol = rtol
        self.atol = atol
        self.n_jobs = n_jobs



    def _reset(self):

        """
        Reset internal data-dependent state of the transformer.
        __init__ parameters are not changed.

        """

        if hasattr(self, 'constant_columns_'):
            del self.constant_columns_
            del self.kept_columns_
            del self.removed_columns_
            del self.column_mask_


    def get_feature_names_out(self, input_features=None):

        """
        Get remaining feature names after transform.


        Parameters
        ----------
        input_features :
            array-like of str or None, default=None - Externally provided
            feature names.

            If input_features is None:
            if feature_names_in_ is defined, then feature_names_in_ is
            used as the input features.
            If feature_names_in_ is not defined, then the following input
            feature names are generated:
                ["x0", "x1", ..., "x(n_features_in_ - 1)"].

            If input_features is not None:
            if feature_names_in_ is not defined, then input_features is
            used as the input features.
            if feature_names_in_ is defined, then input_features must be
            an array-like whose feature names exactly match those in
            feature_names_in_.

        Return
        ------
        -
            feature_names_out : NDArray[str] - The feature names in the
            transformed data.

        """

        # get_feature_names_out() would otherwise be provided by
        # OneToOneFeatureMixin, but since this transformer deletes
        # and/or adds columns, must build a one-off.

        # when there is a {'Intercept': 1} in :param: keep, need to make sure
        # that that column is accounted for here, and the dropped columns are
        # also accounted for.


        check_is_fitted(self)

        try:
            # input_features can be None
            if isinstance(input_features, type(None)):
                raise UnicodeError
            # must be iterable
            iter(input_features)
            # cannot be dict or str
            if isinstance(input_features, (str, dict)):
                raise Exception
            # iterable must contain strings
            if not all(map(
                isinstance, input_features, (str for _ in input_features)
            )):
                raise Exception
        except UnicodeError:
            pass
        except:
            raise ValueError(
                f"'input_features' must be a vector-like containing strings, or None"
            )


        # if input_features is passed, check against n_features_in_ &
        # features_names_in_, apply column_mask_, optionally append intcpt
        # from keep dict
        if input_features is not None:

            if len(input_features) != self.n_features_in_:
                raise ValueError("input_features should have length equal")

            if hasattr(self, 'feature_names_in_'):
                if not np.array_equal(input_features, self.feature_names_in_):
                    raise ValueError(
                        f"input_features is not equal to feature_names_in_"
                    )

            # column_mask_ is always shaped against num features in fitted data,
            # regardless of if keep is a dictionary. apply mask before adding
            # keep dict intercept.
            input_features = \
                np.array(input_features)[self.column_mask_].astype(object)

            # adjust if appending a new intercept column
            if isinstance(self.keep, dict):
                input_features = np.hstack((
                    input_features,
                    list(self.keep.keys())[0]
                )).astype(object)

            return input_features

        # if input_features is not passed, but feature_names_in_ is available,
        # apply column_mask_ to feature_names_in_, optionally append intcpt
        # from keep dict
        elif hasattr(self, 'feature_names_in_'):  # and input_features is None

            # column_mask_ is always shaped against num features in fitted data,
            # regardless of if keep is a dictionary. apply mask before adding
            # keep dict intercept.
            out = self.feature_names_in_[self.column_mask_].astype(object)

            if isinstance(self.keep, dict):
                out = np.hstack((
                    out,
                    list(self.keep.keys())[0]
                )).astype(object)

            return out

        # if input_features is not passed and no attr feature_names_in_,
        # build a dummy vector of headers, apply column_mask_, optionally
        # append intcpt from keep dict
        else:  # feature_names_in_ not available and input_features is None

            # column_mask_ is always shaped against num features in fitted data,
            # regardless of if keep is a dictionary. apply mask before adding
            # keep dict intercept.
            _dum_header = [f"x{i}" for i in range(self.n_features_in_)]

            out = np.array(_dum_header, dtype=object)[self.column_mask_]

            if isinstance(self.keep, dict):
                out = np.hstack((out, list(self.keep.keys())[0])).astype(object)

            return out


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
        Perform incremental fitting on one or more data sets. Determine
        the constant columns in the given data, subject to the criteria
        defined in :params: rtol, atol, and equal_nan.


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - Data to remove constant columns from.
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

        # ^^^^^^ pizza be sure to review _validate_data! ^^^^^^^

        # this must be after _validate_data, needs feature_names_in_ to
        # be exposed, if available.

        _validation(
            X,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
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


        # dictionary of column indices and respective constant values
        self.constant_columns_:dict[int, any] = \
            _find_constants(
                X,
                self.constant_columns_ if hasattr(self, 'constant_columns_') else {},
                self.equal_nan,
                self.rtol,
                self.atol,
                self.n_jobs
            )

        _keep = _manage_keep(
            self.keep,
            X,
            self.constant_columns_,
            self.n_features_in_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None
        )


        self._instructions = _make_instructions(
            _keep,
            self.constant_columns_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            X.shape
        )

        self.kept_columns_, self.removed_columns_, self.column_mask_ = \
            _set_attributes(
                self.constant_columns_,
                self._instructions,
                self.n_features_in_
            )

        # pizza take these training wheels off when done
        assert len(self.column_mask_)== X.shape[1]

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
            n_features) - Data to remove constant columns from.
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
        Revert transformed data back to its original state. This operation
        cannot restore any nan-like values that may have been in the
        original untransformed data.


        Parameters
        ----------
        X :
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features - n_features_removed) - A transformed data set.
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

        # pizza what if set_params is changed?
        # think on this
        # if we cant come up with a way to validate/ensure that the data
        # being inverted back matches with the current state of the params,
        # then will have to put some disclaimers in the docs.

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

        # the number of columns in X must be equal to the number of features
        # remaining in column_mask_
        if X.shape[1] != np.sum(self.column_mask_):
            raise ValueError(
                f"the number of columns in X must be equal to the number of "
                f"columns kept from the fitted data after removing constants "
                f"{np.sum(self.column_mask_)}"
            )

        # if _keep is a dict, a column of constants was stacked to the right
        # side of the data. check that the passed data matches against _keep,
        # and remove the column
        if isinstance(self.keep, dict):
            _unqs = np.unique(X[:, -1])
            if len(_unqs) == 1 and _unqs[0] == self.keep[list(self.keep.keys())[0]]:
                # pizza, this needs to be for np, df, and ss
                X = np.delete(X, -1, axis=1)
            else:
                raise ValueError(
                    f":param: 'keep' is a dictionary but the last column of the "
                    f"data to be inverse transformed does not match."
                )

        # pizza take these training wheels off when done
        assert sum(self.column_mask_)== X.shape[1], \
            f"{sum(self.column_mask_)=}, {X.shape[1]=}"

        X = _inverse_transform(
            X,
            self.removed_columns_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None
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


    # def set_params(self):
        # pizza! dont forget! once the instance is fitted, cannot change equal_nan, rtol, and atol!
        # ... or maybe u can.... its just that new fits will be fitted subject to
        # different rules than prior fits
    # if ever needed, hard code that can be substituted for the
    # BaseEstimator get/set_params can be found in GSTCV_Mixin

    # def set_output(self) - inherited from TransformerMixin


    def transform(
        self,
        X: DataFormatType,
        copy: bool=None
    ) -> DataFormatType:

        """
        Manage the constant columns from X. Apply the criteria given
        by :param: keep to the sets of constant columns found during fit.


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - The data.
        copy:
            Union[bool, None], default=None - Whether to make a copy of
            X before the transform.


        Return
        ------
        -
            X: {array-like, scipy sparse matrix} of shape (n_samples,
                n_features - n_removed_features) - The transformed data.




        """

        check_is_fitted(self)

        if not isinstance(copy, (bool, type(None))):
            raise TypeError(f"'copy' must be boolean or None")

        # keep this before _validate_data. when X is junk, _validate_data
        # and check_array except for varying reasons. this standardizes
        # the error message for non-np/pd/ss X.
        _val_X(X)

        X = self._validate_data(
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
            X,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            self.keep,
            self.equal_nan,
            self.rtol,
            self.atol,
            self.n_jobs
        )


        # everything below needs to be redone every transform in case 'keep' was
        # changed via set params after fit

        _keep = _manage_keep(
            self.keep,
            X,
            self.constant_columns_,
            self.n_features_in_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None
        )

        self._instructions = _make_instructions(
            _keep,
            self.constant_columns_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            X.shape
        )

        self.kept_columns_, self.removed_columns_, self.column_mask_ = \
            _set_attributes(
                self.constant_columns_,
                self._instructions,
                self.n_features_in_
            )

        # pizza take these training wheels off when done
        assert len(self.column_mask_)== X.shape[1]

        X = _transform(X, self._instructions)

        if isinstance(X, np.ndarray):
            X = np.ascontiguousarray(X)

        return X















