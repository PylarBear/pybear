# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
import numpy.typing as npt
import pandas as pd

import numbers
from typing import Iterable, Literal, Optional
from typing_extensions import Union, Self
from ._type_aliases import SparseTypes

from ._validation._validation import _validation
from ._validation._X import _val_X
from ._partial_fit._dupl_idxs import _dupl_idxs
from ._partial_fit._identify_idxs_to_delete import _identify_idxs_to_delete
from ._transform._transform import _transform
from ._inverse_transform._inverse_transform import _inverse_transform

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted, check_array




class ColumnDeduplicateTransformer(BaseEstimator, TransformerMixin):

    """
    ColumnDeduplicateTransformer (CDT) is a scikit-style transformer
    that removes duplicate columns from data, leaving behind one column
    out of a set of duplicate columns.

    Duplicate columns are a point of concern for analysts. In many data
    analytics learning algorithms, such a condition can cause convergence
    problems, inversion problems, or other undesirable effects. The
    analyst is often forced to address the issue to perform a meaningful
    analysis of the data.

    Columns with identical values within the same dataset may occur
    coincidentally in a sampling of data, may occur during one-hot
    encoding of categorical data, or may occur during polynomial feature
    expansion.

    CDT is a tool that can help fix this problem. CDT identifies
    duplicate columns and selectively keeps one from a group of
    duplicates based on the configuration set by the user.

    CDT affords parameters that give some flexibility to the definition
    of "equal" for the sake of identifying duplicates. Namely, the
    'rtol', 'atol', and 'equal_nan' parameters.

    The rtol and atol parameters provide tolerance windows within which
    numerical data that is not exactly equal but within the tolerance
    are considered equal. See the numpy docs for deeper clarification of
    the technical details.

    The equal_nan parameter controls how CDT handles nan-like
    representations during comparisons. If equal_nan is True, exclude
    from comparison any rows where one or both of the values is/are nan.
    If one value is nan, this essentially assumes that the nan value
    would otherwise be the same as its non-nan counterpart. When both
    are nan, this considers the nans as equal (contrary to the default
    numpy handling of nan, where np.nan does not equal np.nan) and will
    not in and of itself cause a pair of columns to be marked as unequal.
    If equal_nan is False and either one or both of the values in the
    compared pair of values is/are nan, consider the pair to be not
    equivalent, thus making the column pair not equal. This is in line
    with the normal numpy handling of nan values. See the Notes section
    below for a discussion on the handling of nan-like values.

    CDT has parameters that allow the user to control which column is
    retained out of a set of duplicates: the 'keep', 'do_not_drop', and
    'conflict' parameters.

    The parameter 'keep' sets the strategy for keeping a single
    representative from a set of identical columns. It accepts one of
    three values: 'first', 'last', or 'random'. The default setting is
    'first'. 'first' retains the column left-most in the data; 'last'
    keeps the column right-most in the data; 'random' keeps a single
    randomly-selected column of the set of duplicates. All other columns
    in the set of duplicates are removed from the dataset.

    The parameter 'do_not_drop' allows the user to indicate columns not
    to be removed from the data. This is to be given as a list-like of
    integers or strings. If fitting is done on a pandas dataframe that
    has a header, a list of feature names may be provided; the values
    within must match exactly the features as named in the dataframe
    header (case sensitive.) Otherwise, a list of column indices must be
    provided. The :param: do_not_drop instructions may conflict with
    the :param: keep instructions. If such a conflict arises, such as
    two columns specified in :param: do_not_drop are duplicates of each
    other, the behavior is managed by :param: conflict.

    The 'conflict' parameter accepts two possible values: 'raise' or
    'ignore'. When :param: do_not_drop is not passed, :param: conflict
    is ignored. This parameter instructs CDT how to deal with a conflict
    between the instructions in :param: keep and :param: do_not_drop. A
    conflict arises when the instruction in :param: keep ('first',
    'last', 'random') is applied and a column in :param: do_not_drop is
    found to be a member of the columns to be removed. When :param:
    conflict is 'raise', an exception is raised in this case. When
    :param: conflict is 'ignore', there are 2 possible scenarios:

        1) when only one column in :param: do_not_drop is among the
        columns to be removed, the :param: keep instruction is overruled
        and the do_not_drop column is kept

        2) when multiple columns in :param: do_not_drop are among the
        duplicates, the :param: keep instruction ('first', 'last',
        'random') is applied to the set of do-not-delete columns
        that are amongst the duplicates --- this may not give the same
        result as applying the :param: keep instruction to the entire
        set of duplicate columns. This also causes at least one member
        of the columns not to be dropped to be removed.

    The partial_fit, fit, fit_transform, and inverse_transform methods
    of CDT accept data as numpy arrays, pandas dataframes, and scipy
    sparse matrices/arrays. CDT has a set_output method, whereby the user
    can set the type of output container. This behavior is managed by
    scikit-learn functionality adopted into CDT, and is subject to change
    at their discretion. As of first publication, :method: set_output
    can return transformed outputs as numpy arrays, pandas dataframes,
    or polars dataframes. When :method: set_output is None, the output
    container is the same as the input, that is, numpy array, pandas
    dataframe, or scipy sparse matrix/array.

    The partial fit method allows for incremental fitting of data sets.
    This makes CDT suitable for use with packages that do batch-wise
    fitting and transforming, such as dask_ml via the Incremental and
    ParallelPostFit wrappers.


    Parameters
    ----------
    keep:
        Literal['first', 'last', 'random'], default = 'first' -
        The strategy for keeping a single representative from a set of
        identical columns. 'first' retains the column left-most in the
        data; 'last' keeps the column right-most in the data; 'random'
        keeps a single randomly-selected column of the set of duplicates.
    _do_not_drop:
        Union[Iterable[int], Iterable[str], None], default=None - A list
        of columns not to be dropped. If fitting is done on a pandas
        dataframe that has a header, a list of feature names may be
        provided. Otherwise, a list of column indices must be provided.
        If a conflict arises, such as two columns specified in :param:
        do_not_drop are duplicates of each other, the behavior is managed
        by :param: conflict.
    conflict:
        Literal['raise', 'ignore'] - Ignored when :param: do_not_drop is
        not passed. Instructs CDT how to deal with a conflict between
        the instructions in :param: keep and :param: do_not_drop. A
        conflict arises when the instruction in :param: keep ('first',
        'last', 'random') is applied and a column in :param: do_not_drop
        is found to be a member of the columns to be removed. When
        :param: conflict is 'raise', an exception is raised in this case.
        When :param: conflict is 'ignore', there are 2 possible scenarios:

        1) when only one column in :param: do_not_drop is among the
        columns to be removed, the :param: keep instruction is overruled
        and the do_not_drop column is kept

        2) when multiple columns in :param: do_not_drop are among the
        columns to be removed, the :param: keep instruction ('first',
        'last', 'random') is applied to the set of do-not-delete columns
        that are amongst the duplicates --- this may not give the same
        result as applying the :param: keep instruction to the entire
        set of duplicate columns. This also causes at least one member
        of the columns not to be dropped to be removed.
    rtol:
        float, default = 1e-5 - The relative difference tolerance for
            equality. See numpy.allclose.
    atol:
        float, default = 1e-8 - The absolute tolerance parameter for .
            equality. See numpy.allclose.
    equal_nan:
        bool, default = False - When comparing pairs of columns row by
        row:
        If equal_nan is True, exclude from comparison any rows where
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
    n_jobs:
        Union[int, None], default = -1 - The number of joblib Parallel
        jobs to use when comparing columns. The default is to use
        processes, but can be overridden externally using a joblib
        parallel_config context manager. The default number of jobs is
        -1 (all processors). To get maximum speed benefit, pybear
        recommends using the default setting.


    Attributes:
    -----------
    n_features_in_:
        int - number of features in the fitted data before deduplication.

    feature_names_in_:
        NDArray[str] - The names of the features as seen during fitting.
        Only accessible if X is passed to :methods: partial_fit or fit
        as a pandas dataframe that has a header.

    duplicates_: list[list[int]] - a list of the groups of identical
        columns, indicated by their zero-based column index positions
        in the originally fit data.

    removed_columns_: dict[int, int] - a dictionary whose keys are the
        indices of duplicate columns removed from the original data,
        indexed by their column location in the original data; the values
        are the column index in the original data of the respective
        duplicate that was kept.

    column_mask_: list[bool], shape (n_features_,) - Indicates which
        columns of the fitted data are kept (True) and which are removed
        (False) during transform.


    Notes
    -----
    Concerning the handling of nan-like representations. While CDT
    accepts data in the form of numpy arrays, pandas dataframes, and
    scipy sparse matrices/arrays, at the time of column comparison both
    columns of data are converted to numpy arrays. After the conversion
    and prior to the comparison, CDT identifies any nan-like
    representations in both of the numpy arrays and standardizes all of
    them to np.nan. The user needs to be wary that whatever is used to
    indicate 'not-a-number' in the original data must first survive the
    conversion to numpy array, then be recognizable by CDT as nan-like,
    so that CDT can standardize it to np.nan. nan-like representations
    that are recognized by CDT include, at least, np.nan, pandas.NA,
    None (of type None, not string 'None'), and string representations
    of "nan" (not case sensitive).


    See Also
    --------
    numpy.ndarray
    pandas.core.frame.DataFrame
    scipy.sparse
    numpy.allclose
    numpy.array_equal


    Examples
    --------
    >>> from pybear.preprocessing import ColumnDeduplicateTransformer as CDT
    >>> import numpy as np
    >>> np.random.seed(99)
    >>> X = np.random.randint(0, 10, (5, 5))
    >>> X[:, 2] = X[:, 0]
    >>> X[:, 4] = X[:, 1]
    >>> print(X)
    [[1 3 1 8 3]
     [8 2 8 5 2]
     [1 7 1 7 7]
     [1 0 1 4 0]
     [2 0 2 8 0]]
    >>> trf = CDT(keep='first', do_not_drop=None)
    >>> trf.fit(X)
    ColumnDeduplicateTransformer()
    >>> out = trf.transform(X)
    >>> print(out)
    [[1 3 8]
     [8 2 5]
     [1 7 7]
     [1 0 4]
     [2 0 8]]
    >>> print(trf.n_features_in_)
    5
    >>> print(trf.duplicates_)
    [[0, 2], [1, 4]]
    >>> print(trf.removed_columns_)
    {2: 0, 4: 1}
    >>> print(trf.column_mask_)
    [True, True, False, True, False]


    """

    _parameter_constraints: dict = {
        "keep": [StrOptions({"first", "last", "random"})],
        "do_not_drop": [list, tuple, set, None,],
        "conflict": [StrOptions({"raise", "ignore"})],
        "rtol": [numbers.Real],
        "atol": [numbers.Real],
        "equal_nan": ["boolean"],
        "n_jobs": [numbers.Integral, None],
    }

    def __init__(
        self,
        *,
        keep: Optional[Literal['first', 'last', 'random']] = 'first',
        do_not_drop: Optional[Union[Iterable[str], Iterable[int], None]] = None,
        conflict: Optional[Literal['raise', 'ignore']] = 'raise',
        rtol: Optional[float] = 1e-5,
        atol: Optional[float] = 1e-8,
        equal_nan: Optional[bool] = False,
        n_jobs: Optional[Union[int, None]] = None
    ) -> None:


        self.keep = keep
        self.do_not_drop = do_not_drop
        self.conflict = conflict
        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan
        self.n_jobs = n_jobs


    def _reset(self):
        """
        Reset internal data-dependent state of the transformer.
        __init__ parameters are not changed.

        """

        if hasattr(self, "duplicates_"):
            del self.duplicates_
            del self.removed_columns_
            del self.column_mask_


    def get_feature_names_out(self, input_features=None):

        """
        Get remaining feature names after deduplication.


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
            deduplicated data after transformation.

        """

        # get_feature_names_out() would otherwise be provided by
        # OneToOneFeatureMixin, but since this transformer deletes
        # columns, must build a one-off.


        try:
            if isinstance(input_features, type(None)):
                raise UnicodeError
            iter(input_features)
            if isinstance(input_features, (str, dict)):
                raise Exception
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


        if input_features is not None:

            if len(input_features) != self.n_features_in_:
                raise ValueError("input_features should have length equal")

            if hasattr(self, 'feature_names_in_'):

                if not np.array_equal(input_features, self.feature_names_in_):
                    raise ValueError(
                        f"input_features is not equal to feature_names_in_"
                    )

            out = np.array(input_features, dtype=object)[self.column_mask_]

            return out

        elif hasattr(self, 'feature_names_in_'):
            return self.feature_names_in_[self.column_mask_].astype(object)

        else:
            try:
                input_features = \
                    np.array([f"x{i}" for i in range(self.n_features_in_)])
                return input_features.astype(object)[self.column_mask_]
            except:
                raise NotFittedError(
                    f"This {type(self).__name__} instance is not fitted yet. "
                    f"Call 'fit' with appropriate arguments before using this "
                    f"estimator."
                )


    # def get_params - inherited from BaseEstimator
    # if ever needed, hard code that can be substituted for the
    # BaseEstimator get/set_params can be found in GSTCV_Mixin


    def get_metadata_routing(self):
        """
        Get metadata routing is not implemented in ColumnDeduplicateTransformer.

        """
        __ = type(self).__name__
        raise NotImplementedError(
            f"get_metadata_routing is not implemented in {__}"
        )


    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(
        self,
        X: Union[npt.NDArray[any], pd.DataFrame, SparseTypes],
        y: any=None
    ) -> Self:

        """
        Perform incremental fitting on one or more data sets. Determine
        the duplicate columns in the given data, subject to the criteria
        defined in :params: rtol, atol, and equal_nan.


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - Data to remove duplicate columns from.
        y:
            {vector-like of shape (n_samples,) or None}, default = None -
            ignored. The target for the data.


        Return
        ------
        -
            self - the fitted ColumnDeduplicateTransformer instance.


        """


        # validation of X must be done here (with reset=True), not in a
        # separate module
        # BaseEstimator has _validate_data method, which when called
        # exposes n_features_in_ and feature_names_in_.
        X = self._validate_data(
            X=X,
            reset=not hasattr(self, "duplicates_"),
            cast_to_ndarray=False,
            # vvv become **check_params, and get fed to check_array()
            accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
            dtype=None,
            force_all_finite="allow-nan",
            ensure_2d=True,
            ensure_min_features=2,
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


        _validation(
            X,  # Not validated, used for validation of other objects
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            self.conflict,
            self.do_not_drop,
            self.keep,
            self.rtol,
            self.atol,
            self.equal_nan,
            self.n_jobs
        )


        # find the duplicate columns
        self.duplicates_ = \
            _dupl_idxs(
                X,
                self.duplicates_ if hasattr(self, 'duplicates_') else None,
                self.rtol,
                self.atol,
                self.equal_nan,
                self.n_jobs
        )

        self.removed_columns_ = \
            _identify_idxs_to_delete(
                self.duplicates_,
                self.keep,
                self.do_not_drop,
                self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
                self.conflict
            )

        self.column_mask_ = np.ones(self.n_features_in_).astype(bool)
        self.column_mask_[list(self.removed_columns_)] = False
        self.column_mask_ = list(map(bool, self.column_mask_))

        return self


    def fit(
        self,
        X: Union[npt.NDArray[any], pd.DataFrame, SparseTypes],
        y: any=None
    ) -> Self:

        """
        Perform a single fitting on a data set. Determine the duplicate
        columns in the given data, subject to the criteria defined
        in :params: rtol, atol, and equal_nan.


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - Data to remove duplicate columns from.
        y:
            {vector-like of shape (n_samples,) or None}, default = None -
            ignored. The target for the data.


        Return
        ------
        -
            self - the fitted ColumnDeduplicateTransformer instance.


        """

        self._reset()
        return self.partial_fit(X, y=y)


    def inverse_transform(
        self,
        X: Union[npt.NDArray[any], pd.DataFrame, SparseTypes],
        *,
        copy: bool = None
        ) -> Union[npt.NDArray[any], pd.DataFrame, SparseTypes]:

        """
        Revert deduplicated data back to its original state.


        Parameters
        ----------
        X :
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features - n_features_removed) - A deduplicated data set.
        copy:
            Union[bool, None], default=None - Whether to make a copy of
            X before the inverse transform.


        Returns
        -------
        -
            X_tr : {array-like, scipy sparse matrix} of shape (n_samples,
                n_features) - Deduplicated data reverted to its original
                state.

        """

        check_is_fitted(self)

        if not isinstance(copy, (bool, type(None))):
            raise TypeError(f"'copy' must be boolean or None")


        # the number of columns in X must be equal to the number of features
        # remaining in column_mask_
        _n_remaining = np.sum(self.column_mask_)
        err_msg = (f"the number of columns in X must be equal to the number of "
               f"columns kept from the fitted data after removing duplicates "
               f"{_n_remaining}"
        )
        if X.shape[1] != _n_remaining:
            raise ValueError(err_msg)


        X = check_array(
            array=X,
            accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
            dtype=None,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=copy or False,
            order='F'
        )

        out = _inverse_transform(
            X,
            self.duplicates_,
            self.removed_columns_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None
        )

        if isinstance(out, np.ndarray):
            out = np.ascontiguousarray(out)

        return out


    def score(self, X, y=None):
        """
        Dummy method to spoof dask Incremental and ParallelPostFit
        wrappers. Verified must be here for dask wrappers.
        """

        pass


    # def set_params - inherited from BaseEstimator
    # if ever needed, hard code that can be substituted for the
    # BaseEstimator get/set_params can be found in GSTCV_Mixin


    # def set_output(self) - inherited from TransformerMixin


    def transform(
        self,
        X: Union[npt.NDArray[any], pd.DataFrame, SparseTypes],
        *,
        copy: bool = None
    ) -> Union[npt.NDArray[any], pd.DataFrame, SparseTypes]:

        """
        Remove the duplicate columns from X. Apply the criteria given
        by :params: keep, do_not_drop, and conflict to the sets of
        duplicate columns found during fit.


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - The data to be deduplicated.
        copy:
            Union[bool, None], default=None - Whether to make a copy of
            X before the transform.


        Return
        ------
        -
            X: {array-like, scipy sparse matrix} of shape (n_samples,
                n_features - n_removed_features) - The deduplicated data.

        """

        check_is_fitted(self)

        if not isinstance(copy, (bool, type(None))):
            raise TypeError(f"'copy' must be boolean or None")

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
            ensure_min_features=2,
            order ='F'
        )


        _validation(
            X,  # Not validated, used for validation of other objects
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            self.conflict,
            self.do_not_drop,
            self.keep,
            self.rtol,
            self.atol,
            self.equal_nan,
            self.n_jobs
        )


        # redo these here in case set_params() was changed after (partial_)fit
        # determine the columns to remove based on given parameters.
        self.removed_columns_ = \
            _identify_idxs_to_delete(
                self.duplicates_,
                self.keep,
                self.do_not_drop,
                self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
                self.conflict
            )

        self.column_mask_ = np.ones(self.n_features_in_).astype(bool)
        self.column_mask_[list(self.removed_columns_)] = False
        self.column_mask_ = list(map(bool, self.column_mask_))
        # end redo

        out = _transform(X, self.column_mask_)

        if isinstance(out, np.ndarray):
            out = np.ascontiguousarray(out)

        return out














