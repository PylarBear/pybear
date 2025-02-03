# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from numbers import Real
from typing import Sequence, Literal, Optional
import numpy.typing as npt
from typing_extensions import Union, Self
from ._type_aliases import DataContainer

import numpy as np

from ._validation._validation import _validation
from ._validation._X import _val_X
from ._partial_fit._dupl_idxs import _dupl_idxs
from ._partial_fit._lock_in_random_idxs import _lock_in_random_idxs
from ._partial_fit._identify_idxs_to_delete import _identify_idxs_to_delete
from ._inverse_transform._inverse_transform import _inverse_transform
from ._transform._transform import _transform

from ...base import (
    FeatureMixin,
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetOutputMixin,
    SetParamsMixin,
    check_is_fitted,
    get_feature_names_out,
    validate_data
)



class ColumnDeduplicateTransformer(
    FeatureMixin,
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetOutputMixin,
    SetParamsMixin
):

    """
    ColumnDeduplicateTransformer (CDT) is a scikit-style transformer
    that removes duplicate columns from data, leaving behind one column
    out of a set of duplicate columns.

    Duplicate columns are a point of concern for analysts. In many data
    analytics learning algorithms, such a condition can cause convergence
    problems, inversion problems, or other undesirable effects. The
    analyst is often forced to address this issue to perform a meaningful
    analysis of the data.

    Columns with identical values within the same dataset may occur
    coincidentally in a sampling of data, during one-hot encoding of
    categorical data, or during polynomial feature expansion.

    CDT is a tool that can help fix this problem. CDT identifies
    duplicate columns and selectively keeps one from a group of
    duplicates based on the configuration set by the user.

    CDT affords parameters that give some flexibility to the definition
    of 'equal' for the sake of identifying duplicates. Namely, the
    'rtol', 'atol', and 'equal_nan' parameters.

    The rtol and atol parameters provide a tolerance window whereby
    numerical data that are not exactly equal are considered equal if
    their difference falls within the tolerance. See the numpy docs for
    clarification of the technical details. CDT requires that rtol and
    atol be non-boolean, non-negative real numbers, in addition to any
    other restrictions enforced by numpy.allclose.

    The equal_nan parameter controls how CDT handles nan-like
    representations during comparisons. If equal_nan is True, exclude
    from comparison any rows where one or both of the values is/are nan.
    If one value is nan, this essentially assumes that the nan value
    would otherwise be the same as its non-nan counterpart. When both
    are nan, this considers the nans as equal (contrary to the default
    numpy handling of nan, where numpy.nan does not equal numpy.nan) and
    will not in and of itself cause a pair of columns to be marked as
    unequal. If equal_nan is False and either one or both of the values
    in the compared pair of values is/are nan, consider the pair to be
    not equivalent, thus making the column pair not equal. This is in
    line with the normal numpy handling of nan values. See the Notes
    section below for a discussion on the handling of nan-like values.

    CDT has parameters that allow the user to control which column is
    retained out of a set of duplicates: the 'keep', 'do_not_drop', and
    'conflict' parameters.

    The parameter 'keep' sets the strategy for keeping a single
    representative from a set of identical columns. It accepts one of
    three values: 'first', 'last', or 'random'. The default setting is
    'first'. 'first' retains the column left-most in the data; 'last'
    keeps the column right-most in the data; 'random' keeps a single
    randomly-selected column from the set of duplicates. All other
    columns in the set of duplicates are removed from the dataset.

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
        columns to be removed, the :param: keep instruction ('first',
        'last', 'random') is applied to that subset of do-not-drop
        columns --- this may not give the same result as applying the
        :param: keep instruction to the entire set of duplicate columns.
        This also causes at least one member of the columns not to be
        dropped to be removed.

    The partial_fit, fit, fit_transform, and inverse_transform methods
    of CDT accept data as numpy arrays, pandas dataframes, and scipy
    sparse matrices/arrays. CDT has a set_output method, whereby the
    user can set the type of output container for :method: transform.
    This behavior is managed by scikit-learn functionality adopted into
    CDT, and is subject to change at their discretion. :method:
    set_output can return transformed outputs as numpy arrays, pandas
    dataframes, or polars dataframes. When :method: set_output is None,
    the output container is the same as the input, that is, numpy array,
    pandas dataframe, or scipy sparse matrix/array.

    The partial_fit method allows for incremental fitting of data. This
    makes CDT suitable for use with packages that do batch-wise fitting
    and transforming, such as dask_ml via the Incremental and
    ParallelPostFit wrappers.

    There are no safeguards in place to prevent the user from changing
    rtol, atol, or equal_nan between partial fits. These 3 parameters
    have strong influence over whether CDT classifies two columns as
    equal, and therefore are instrumental in dictating what CDT learns
    during fitting. Changes to these parameters between partial fits can
    drastically change CDT's understanding of the duplicate columns in
    the data versus what would otherwise be learned under constant
    settings. pybear recommends against this practice, however, it is
    not strictly blocked.

    When performing multiple batch-wise transformations of data, that is,
    making sequential calls to :method: transform, it is critical that
    the same column indices be kept / removed at each call. This issue
    manifests when :param: keep is set to 'random'; the random indices
    to keep must be the same at all calls to :method: transform, and
    cannot be dynamically randomized within :method: transform. CDT
    handles this by generating a static list of random indices to keep
    at fit time, and does not mutate this list during transform time.
    This list is dynamic with each call to :method: partial_fit, and
    will likely change at each call. Fits performed after calls to
    :method: transform will change the random indices away from those
    used in the previous transforms, causing CDT to perform entirely
    different transformations than those previously being done. CDT
    cannot block calls to :method: partial_fit after calls to :method:
    transform, but pybear strongly discourages this practice because the
    output will be nonsensical. pybear recommends doing all partial fits
    consecutively, then doing all transformations consecutively.


    Parameters
    ----------
    keep:
        Optional[Literal['first', 'last', 'random']], default = 'first' -
        The strategy for keeping a single representative from a set of
        identical columns. 'first' retains the column left-most in the
        data; 'last' keeps the column right-most in the data; 'random'
        keeps a single randomly-selected column of the set of duplicates.
    do_not_drop:
        Optional[Union[Sequence[int], Sequence[str], None]], default=None
        - A list of columns not to be dropped. If fitting is done on a
        pandas dataframe that has a header, a list of feature names may
        be provided. Otherwise, a list of column indices must be provided.
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
        Real, default = 1e-5 - The relative difference tolerance for
        equality. Must be a non-boolean, non-negative, real number.
        See numpy.allclose.
    atol:
        Real, default = 1e-8 - The absolute difference tolerance for
        equality. Must be a non-boolean, non-negative, real number.
        See numpy.allclose.
    equal_nan:
        bool, default = False - When comparing pairs of columns row by
        row:
        If equal_nan is True, exclude from comparison any rows where
        one or both of the values is/are nan. If one value is nan, this
        essentially assumes that the nan value would otherwise be the
        same as its non-nan counterpart. When both are nan, this
        considers the nans as equal (contrary to the default numpy
        handling of nan, where numpy.nan does not equal numpy.nan) and
        will not in and of itself cause a pair of columns to be marked
        as unequal. If equal_nan is False and either one or both of the
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

    column_mask_: NDArray[bool], shape (n_features_,) - Indicates which
        columns of the fitted data are kept (True) and which are removed
        (False) during transform.


    Notes
    -----
    Concerning the handling of nan-like representations. While CDT
    accepts data in the form of numpy arrays, pandas dataframes, and
    scipy sparse matrices/arrays, at the time of column comparison both
    columns of data are converted to numpy arrays (see below for more
    detail about how scipy sparse is handled.) After the conversion
    and prior to the comparison, CDT identifies any nan-like
    representations in both of the numpy arrays and standardizes all of
    them to numpy.nan. The user needs to be wary that whatever is used
    to indicate 'not-a-number' in the original data must first survive
    the conversion to numpy array, then be recognizable by CDT as
    nan-like, so that CDT can standardize it to numpy.nan. nan-like
    representations that are recognized by CDT include, at least,
    numpy.nan, pandas.NA, None (of type None, not string 'None'), and
    string representations of 'nan' (not case sensitive).

    Concerning the handling of infinity. CDT has no special handling for
    the various infinity-types, e.g, numpy.inf, -numpy.inf, float('inf'),
    float('-inf'), etc. This is a design decision to not force infinity
    values to numpy.nan to avoid mutating or making copies of passed
    data. SPF falls back to the native handling of these values for
    python and numpy. Specifically, numpy.inf==numpy.inf and
    float('inf')==float('inf').

    Concerning the handling of scipy sparse arrays. When comparing
    columns for equality, the columns are not converted to dense numpy
    arrays. Each column is sliced from the data in sparse form and the
    'indices' and 'data' attributes of this slice are stacked together.
    The single vector holding the indices and dense values is used to
    make equality comparisons.


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


    def __init__(
        self,
        *,
        keep: Optional[Literal['first', 'last', 'random']] = 'first',
        do_not_drop: Optional[Union[Sequence[str], Sequence[int], None]] = None,
        conflict: Optional[Literal['raise', 'ignore']] = 'raise',
        rtol: Optional[Real] = 1e-5,
        atol: Optional[Real] = 1e-8,
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


    def _reset(self) -> Self:
        """
        Reset internal data-dependent state of the transformer.
        __init__ parameters are not changed.

        """

        if hasattr(self, "duplicates_"):

            delattr(self, 'duplicates_')
            delattr(self, 'removed_columns_')
            delattr(self, 'column_mask_')
            delattr(self, 'n_features_in_')

            if hasattr(self, 'feature_names_in_'):
                delattr(self, 'feature_names_in_')

        return self


    def get_feature_names_out(
        self,
        input_features:Optional[Union[Sequence[str], None]]=None
    ):

        """
        Get the feature names for the deduplicated data.


        Parameters
        ----------
        input_features :
            Optional[Union[Sequence[str], None], default=None -
            Externally provided feature names for the fitted data, not
            the transformed data.

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
            the deduplicated data after transformation.

        """

        # get_feature_names_out() would otherwise be provided by
        # pybear.base.FeatureMixin, but since this transformer deletes
        # columns, must build a one-off.

        check_is_fitted(self)

        feature_names_out = get_feature_names_out(
            input_features,
            getattr(self, 'feature_names_in_', None),
            self.n_features_in_
        )

        return feature_names_out[self.column_mask_]


    def get_metadata_routing(self):
        """
        Get metadata routing is not implemented in ColumnDeduplicateTransformer.

        """
        __ = type(self).__name__
        raise NotImplementedError(
            f"get_metadata_routing is not implemented in {__}"
        )


    # def get_params - inherited from GetParamsMixin


    def partial_fit(
        self,
        X: DataContainer,
        y: Optional[any]=None
    ) -> Self:

        """
        Perform incremental fitting on one or more data sets. Determine
        the duplicate columns in the given data, subject to the criteria
        defined in :params: rtol, atol, and equal_nan.


        Parameters
        ----------
        X:
            Union[numpy.ndarray, pandas.DataFrame, scipy.sparse] of shape
            (n_samples, n_features) - Data to remove duplicate columns
            from.
        y:
            Optional[any], default = None - ignored. The target for the
            data.


        Return
        ------
        -
            self - the fitted ColumnDeduplicateTransformer instance.


        """

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
            ensure_min_features=2,
            ensure_max_features=None,
            ensure_min_samples=1,
            sample_check=None
        )

        
        # reset – Whether to reset the n_features_in_ attribute. If False,
        # the input will be checked for consistency with data provided when
        # reset was last True.
        # It is recommended to call reset=True in fit and in the first call
        # to partial_fit. All other methods that validate X should set
        # reset=False.

        # do not make an assignment! let the function handle it.
        self._check_n_features(
            X,
            reset=not hasattr(self, "duplicates_")
        )

        # do not make an assignment! let the function handle it.
        self._check_feature_names(
            X,
            reset=not hasattr(self, "duplicates_")
        )


        # this must be after _check_feature_names() needs feature_names_in_
        # to be exposed, if available.
        _validation(
            X,
            getattr(self, 'feature_names_in_', None),
            self.conflict,
            self.do_not_drop,
            self.keep,
            self.rtol,
            self.atol,
            self.equal_nan,
            self.n_jobs
        )

        # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        # ss sparse that cant be sliced
        # avoid copies of X, do not mutate X. if X is coo, dia, bsr, it cannot
        # be sliced. must convert to another ss. so just convert all of them
        # to csc for faster column slicing. need to change it back later.
        if hasattr(X, 'toarray'):
            _og_dtype = type(X)
            X = X.tocsc()

        # find the duplicate columns
        self.duplicates_: list[list[int]] = \
            _dupl_idxs(
                X,
                getattr(self, 'duplicates_', None),
                self.rtol,
                self.atol,
                self.equal_nan,
                self.n_jobs
        )

        # all scipy sparse were converted to csc before _dupl_idxs.
        # change it back to original state. do not mutate X!
        if hasattr(X, 'toarray'):
            X = _og_dtype(X)
            del _og_dtype

        # if 'keep' == 'random', _transform() must pick the same random
        # columns every time. need to set an instance attribute here
        # that doesnt change when _transform() is called. must set a
        # random idx for every set of dupls.
        self._rand_idxs: tuple[int, ...] = _lock_in_random_idxs(
            _duplicates=self.duplicates_,
            _do_not_drop=self.do_not_drop,
            _columns=self.feature_names_in_ if \
                hasattr(self, 'feature_names_in_') else None
        )

        self.removed_columns_: dict[int, int] = \
            _identify_idxs_to_delete(
                self.duplicates_,
                self.keep,
                self.do_not_drop,
                self.feature_names_in_ if \
                    hasattr(self, 'feature_names_in_') else None,
                self.conflict,
                self._rand_idxs
            )

        self.column_mask_: npt.NDArray[bool] = \
            np.ones(self.n_features_in_).astype(bool)
        self.column_mask_[list(self.removed_columns_)] = False

        return self


    def fit(
        self,
        X: DataContainer,
        y: Optional[any]=None
    ) -> Self:

        """
        Perform a single fitting on a data set. Determine the duplicate
        columns in the given data, subject to the criteria defined
        in :params: rtol, atol, and equal_nan.


        Parameters
        ----------
        X:
            Union[numpy.ndarray, pandas.DataFrame, scipy.sparse] of shape
            (n_samples, n_features) - Data to remove duplicate columns
            from.
        y:
            Optional[any], default = None - ignored. The target for the
            data.


        Return
        ------
        -
            self - the fitted ColumnDeduplicateTransformer instance.


        """

        self._reset()
        return self.partial_fit(X, y=y)


    # def fit_transform(self, X, y=None, **fit_params):
    # inherited from FitTransformMixin


    def inverse_transform(
        self,
        X: DataContainer,
        *,
        copy: Optional[Union[bool, None]] = None
        ) -> DataContainer:

        """
        Revert deduplicated data back to its original state. :method:
        set_output does not control the output container here, the output
        container is always the same as passed. This operation cannot
        restore any nan-like values that may have been in the original
        untransformed data.

        Very little validation is possible to ensure that the passed
        data is valid for the current state of CDT. It is only possible
        to ensure that the number of columns in the passed data match
        the number of columns that are expected to be outputted by
        :method: transform for the current state of CDT. It is up to the
        user to ensure the state of CDT aligns with the state of the
        data that is to undergo inverse transform. Otherwise the output
        will be nonsensical.


        Parameters
        ----------
        X:
            Union[numpy.ndarray, pandas.DataFrame, scipy.sparse] of shape
            (n_samples, n_features - n_features_removed) - A deduplicated
            data set.
        copy:
            Optional[Union[bool, None]], default=None - Whether to make
            a copy of X before the inverse transform.


        Returns
        -------
        -
            X_inv: {array-like, scipy sparse matrix} of shape (n_samples,
                n_features) - Deduplicated data reverted to its original
                state.

        """

        check_is_fitted(self)

        if not isinstance(copy, (bool, type(None))):
            raise TypeError(f"'copy' must be boolean or None")

        X_inv = validate_data(
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
            ensure_min_samples=1,
            sample_check=None
        )

        _val_X(X_inv)

        # the number of columns in X must be equal to the number of features
        # remaining in column_mask_
        if X_inv.shape[1] != np.sum(self.column_mask_):
            raise ValueError(
                f"the number of columns in X must be equal to the number of "
                f"columns kept from the fitted data after removing duplicates "
                f"{np.sum(self.column_mask_)}"
            )

        # dont need to do any other validation here, none of the parameters
        # that could be changed by set_params are used here

        # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        # ss sparse that cant be sliced
        # if X_inv is coo, dia, bsr, it cannot be sliced. must convert to
        # another ss. so just convert all of them to csc for faster column
        # slicing. need to change it back later.
        if hasattr(X_inv, 'toarray'):
            _og_format = type(X_inv)
            X_inv = X_inv.tocsc()

        X_inv = _inverse_transform(
            X_inv,
            self.removed_columns_,
            getattr(self, 'feature_names_in_', None)
        )

        # all scipy sparse were converted to csc before _inverse_transform().
        # change it back to original state.
        if hasattr(X_inv, 'toarray'):
            X_inv = _og_format(X_inv)
            del _og_format

        if isinstance(X_inv, np.ndarray):
            X_inv = np.ascontiguousarray(X_inv)

        return X_inv


    def score(
        self,
        X: DataContainer,
        y: Optional[any]=None
    ) -> None:

        """
        Dummy method to spoof dask Incremental and ParallelPostFit
        wrappers. Verified must be here for dask wrappers.
        """

        pass


    # def set_params(self) - inherited from SetParamsMixin


    # def set_output(self) - inherited from SetOutputMixin


    @SetOutputMixin._set_output_for_transform
    def transform(
        self,
        X: DataContainer,
        *,
        copy: Optional[Union[bool, None]] = None
    ) -> DataContainer:

        """
        Remove the duplicate columns from X. Apply the criteria given
        by :params: keep, do_not_drop, and conflict to the sets of
        duplicate columns found during fit.


        Parameters
        ----------
        X:
            Union[numpy.ndarray, pandas.DataFrame, scipy.sparse] of shape
            (n_samples, n_features) - The data to be deduplicated.
        copy:
            Optional[Union[bool, None]], default=None - Whether to make
            a copy of X before the transform.


        Return
        ------
        -
            X_tr: Union[numpy.ndarray, pandas.DataFrame, scipy.sparse]
            of shape (n_samples, n_features - n_removed_features) - The
            deduplicated data.

        """

        check_is_fitted(self)

        if not isinstance(copy, (bool, type(None))):
            raise TypeError(f"'copy' must be boolean or None")

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
            ensure_min_features=2,
            ensure_max_features=None,
            ensure_min_samples=1,
            sample_check=None
        )

        _validation(
            X_tr,
            getattr(self, 'feature_names_in_', None),
            self.conflict,
            self.do_not_drop,
            self.keep,
            self.rtol,
            self.atol,
            self.equal_nan,
            self.n_jobs
        )

        self._check_n_features(X_tr, reset=False)

        self._check_feature_names(X_tr, reset=False)

        # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        # redo these here in case set_params() was changed after (partial_)fit
        # determine the columns to remove based on given parameters.
        self.removed_columns_ = \
            _identify_idxs_to_delete(
                self.duplicates_,
                self.keep,
                self.do_not_drop,
                getattr(self, 'feature_names_in_', None),
                self.conflict,
                self._rand_idxs
            )

        self.column_mask_ = np.ones(self.n_features_in_).astype(bool)
        self.column_mask_[list(self.removed_columns_)] = False
        # end redo

        # ss sparse that cant be sliced
        # if X is coo, dia, bsr, it cannot be sliced. must convert to another
        # ss. so just convert all of them to csc for faster column slicing.
        # need to change it back later.
        if hasattr(X_tr, 'toarray'):
            _og_format = type(X_tr)
            X_tr = X_tr.tocsc()

        X_tr = _transform(X_tr, self.column_mask_)

        # all scipy sparse were converted to csc right before the _transform
        # method. change it back to original state.
        if hasattr(X_tr, 'toarray'):
            X_tr = _og_format(X_tr)
            del _og_format

        if isinstance(X_tr, np.ndarray):
            X_tr = np.ascontiguousarray(X_tr)

        return X_tr














