# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np

from joblib import Parallel

from typing import Iterable, Literal, Optional
from typing_extensions import Union, Self
from ._type_aliases import DataType

from numbers import Real, Integral

from ._validation._validation import _validation
from ._validation._X import _val_X
from pybear.preprocessing.NoDupPolyFeatures._base_fit._constant_handling. \
    _get_constant_columns import _get_constant_columns
from ._base_fit._combination_builder import _combination_builder
from ._transform._transform import _transform
from ._inverse_transform._inverse_transform import _inverse_transform

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, check_array










class NoDupPolyFeatures(BaseEstimator, TransformerMixin):

    """
    make pizza


    Parameters
    ----------
    degree:
        int, default=2 - The maximum polynomial degree of the generated
        features.

    min_degree:
        int, default=0 - The minimum polynomial degree of the generated
        features. Polynomial terms with degree below 'min_degree' are
        not included in the final output array, except for zero-degree
        terms (a column of ones), which is controlled by :param: include_bias.
        Note that `min_degree=0`
        and `min_degree=1` are equivalent as outputting the degree zero term is
        determined by `include_bias`.

    drop_duplicates:
        bool - pizza!

    keep:
        Literal['first', 'last', 'random'], default = 'first' -
        The strategy for keeping a single representative from a set of
        identical columns. 'first' retains the column left-most in the
        data; 'last' keeps the column right-most in the data; 'random'
        keeps a single randomly-selected column of the set of duplicates.
    do_not_drop:
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
    interaction_only:
        bool - pizza!
    include_bias:
        bool - pizza!
    drop_constants:
        bool - pizza!
    output_sparse:
        bool - pizza!
    order:
        Literal['C', 'F'], default = 'C' - Order of output array in the
        dense cases. 'C' means the data is stored in memory in row-major
        order, 'F' means column-major order. NoDup processes all dense
        arrays in 'F' order and defaults to returning them in 'C' order.
        'F' order may slow down subsequent estimators, because 'C' order
        is the standard numpy layout.
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


    Attributes
    ----------
    n_features_in_:
        int - number of features in the fitted data before deduplication.

    feature_names_in_:
        NDArray[str] - The names of the features as seen during fitting.
        Only accessible if X is passed to :methods: partial_fit or fit
        as a pandas dataframe that has a header.

    duplicates_:
        list[list[int]] - a list of the groups of identical
        columns, indicated by their zero-based column index positions
        in the originally fit data.

    dropped_duplicates_:
        dict[int, int] - a dictionary whose keys are the indices of
        duplicate columns removed from the original data, indexed by
        their column location in the original data; the values are the
        column index in the original data of the respective duplicate
        that was kept.

    column_mask_:
        list[bool], shape (n_features_,) - Indicates which
        columns of the fitted data are kept (True) and which are removed
        (False) during transform.

    constants_:
        dict[tuple[int], any]] - put words about how the only constant,
        in this unforgiving world, is good pizza.

    dropped_constants_:
        dict[int, any] - if :param: drop_constants is True, columns of
        constants other than NoDup's bias column are removed from the data.
        In that case, information about the removed constants is stored in the
        dropped_constants_ attribute. There are two scenarios:
        If the column of constants is in the originally passed data, then
        the key in the dictionary is the zero-based index of that column
        in the original data and the value is the value of the constant.
        If the column of constants is in the polynomial expansion, then
        the key in the dictionary is a tuple of the column indices in the
        originally passed data whose product yields the constants, and the
        value is the value of the constant. If :param: drop_constants is False,
        or there are no columns of constants, then :attr: dropped_constants_ is
        an empty dictionary.


    Notes
    -----


    See Also
    --------
    numpy.ndarray
    pandas.core.frame.DataFrame
    scipy.sparse
    numpy.allclose
    numpy.array_equal


    Examples
    --------






    """

    _parameter_constraints: dict = {
        "degree": [Interval(Integral, 0, None, closed="left")],
        "min_degree": [Interval(Integral, 0, None, closed="left")],
        "keep": [StrOptions({"first", "last", "random"})],
        "do_not_drop": [list, tuple, set, None, ],
        "conflict": [StrOptions({"raise", "ignore"})],
        "drop_duplicates": ["boolean"],
        "interaction_only": ["boolean"],
        "include_bias": ["boolean"],
        "drop_constants": ["boolean"],
        "output_sparse": ["boolean"],
        "order": [StrOptions({"C", "F"})],
        "rtol": [Real],
        "atol": [Real],
        "equal_nan": ["boolean"],
        "n_jobs": [Integral, None],
    }


    def __init__(
        self,
        degree:Optional[int]=2,
        *,
        min_degree:Optional[int]=0,
        drop_duplicates: Optional[bool] = True,
        keep: Optional[Literal['first', 'last', 'random']] = 'first',
        do_not_drop: Optional[Union[Iterable[int], Iterable[str], None]] = None,
        conflict: Optional[Literal['raise', 'ignore']] = 'raise',
        interaction_only: Optional[bool] = False,
        include_bias: Optional[bool] = True,
        drop_constants: Optional[bool] = True,
        output_sparse: Optional[bool] = False,
        order: Optional[Literal['C', 'F']] = 'C',
        rtol: Optional[float] = 1e-5,
        atol: Optional[float] = 1e-8,
        equal_nan: Optional[bool] = False,
        n_jobs: Optional[Union[int, None]] = None
    ):

        self.degree = degree
        self.min_degree = min_degree
        self.drop_duplicates = drop_duplicates
        self.keep = keep
        self.do_not_drop = do_not_drop
        self.conflict = conflict
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.drop_constants = drop_constants
        self.output_sparse = output_sparse
        self.order = order
        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan
        self.n_jobs = n_jobs

    # END init ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    def _reset(self):
        """
        Reset internal data-dependent state of the transformer.
        __init__ parameters are not changed.

        """

        if hasattr(self, "duplicates_"):
            del self.duplicates_
            del self.dropped_duplicates_
            del self.dropped_constants_
            del self.column_mask_  # pizza?


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


    def get_metadata_routing(self):
        """
        Get metadata routing is not implemented in ColumnDeduplicateTransformer.

        """
        __ = type(self).__name__
        raise NotImplementedError(
            f"get_metadata_routing is not implemented in {__}"
        )



    # def get_params(self, deep=?:bool) -> dict[str: any]:
    # if ever needed, hard code that can be substituted for the
    # BaseEstimator get/set_params can be found in GSTCV_Mixin



    def _base_fit(
        self,
        X: DataType,
        return_poly:bool=False
    ):

        """


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples, n_features) -
            The data to undergo polynomial expansion.
        return_poly:
            bool - Whether to store in memory the polynomial expansion object
            created while finding the columns of constants and duplicates.


        Return
        ------
        -
            self - the fitted NoDupPolyFeatures instance.

        """


        X = self._validate_data(
            X=X,
            reset=not hasattr(self, "duplicates_"),
            cast_to_ndarray=False,
            # vvv become **check_params, and get fed to check_array()
            accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
            dtype=None,  # do not use 'numeric' here, sk will force to float64
            # check for numeric in supplemental X validation
            force_all_finite=True, # blocks nans here
            ensure_2d=True,
            ensure_min_features=2,
            order='F'
        )


        _validation(
            X,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            self.degree,
            self.min_degree,
            self.drop_duplicates,
            self.keep,
            self.do_not_drop,
            self.conflict,
            self.interaction_only,
            self.include_bias,
            self.drop_constants,
            self.output_sparse,
            self.order,
            self.rtol,
            self.atol,
            self.equal_nan,
            self.n_jobs
        )
        # the only thing that exists at this point is the data and holders.
        # the holders may not be empty.




        # Identify constants in X v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        # This is just to keep them out of the
        # expansion. They will not be removed from X.


        # cannot overwrite self.constants_! may have previous fits in it
        _constant_columns = _get_constant_columns(
            X,
            self.equal_nan,
            self.rtol,
            self.atol,
            _as_indices=True,
            _n_jobs=self.n_jobs
        )
        # END Identify constants in X v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v

        # need to get the permutations to run, based on the size of x,
        # constant columns, min degree, max degree, and interaction_only.
        _combos = _combination_builder(
            _shape=X.shape,
            _constants=_constant_columns,
            _min_degree=self.min_degree,
            _max_degree=self.degree,
            _intx_only=self.interaction_only
        )

        # old constant stuff, see if any of this will be useful during expansion
        """
        # pizza, may need to get constants any time 'include_bias' is True,
        # because then if it already has constants, then it has bias column already
        # on second thought, need to get this every time, in case set_params
        # is changed after fit


        joblib_kwargs = \
            {'prefer': 'processes', 'n_jobs': self.n_jobs, 'return_as': 'list'}
        

        # pizza write test for _parallel_get_uniques
        # out is list[Union[any : value of the constant], Literal[False]]
        out = Parallel(**joblib_kwargs)(
            _parallel_get_uniques(X[:, c_idx]) for c_idx in X.shape[1]
        )

        # convert 'out' to a dict of idxs and values
        # constants_ is {c_idx1: value1, c_idx2: value2.....]
        _constants = {
            (idx, ): value for idx, value in enumerate(out) if idx
        }
        del out

        # pizza, need to meld _constants into self.constants_ -- self.constants_
        # would be holding the constants found in previous partial fits
        # remember that if there already is constants, it will have constants
        # from the polynomial terms too!
        """

        # duplicates v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
        # if removing duplicates, identify them in X --- need to do this
        # every time, set_params may be changed
        # cannot overwrite self.duplicates_! may have previous fits in it
        _duplicates = []

        joblib_kwargs = \
            {'prefer': 'processes', 'n_jobs': self.n_jobs, 'return_as': 'list'}

        # pizza write test for _find_duplicates
        # out is list[[i1, i2,..], [j1, j2, ...]]  GROUPS OF DUPLICATES
        out = Parallel(**joblib_kwargs)(
            _find_duplicates(X[:, c_idx]) for c_idx in X.shape[1]
        )

        # need to convert 'out' to

        # pizza, _duplicates needs to be melded into self.duplicates_ ---
        # which would be holding duplicates found in previous partial fits
        # remember that if there already is duplicates, it will have duplicates
        # from the polynomial terms too!
        # END duplicates v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v



        # pizza, if doing partial fit, then this object doesnt need to be stored,
        # all u need to know from each partial_fit is the columns of constants and
        # duplicate columns, then apply them in transform.
        # if doing just one fit(), then keep this object and just return it in
        # transform.
        if return_poly:
            return _pizza_data_object
        else:
            del _pizza_data_object
            return



    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(
        self,
        X: DataType,
        y: Union[Iterable[any], None]=None
    ) -> Self:

        """
        pizza


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples, n_features) -
            The data to undergo polynomial expansion.
        y:
            {array-like, None} - Always ignored. The target for the data.


        Return
        ------
        -
            self - the fitted NoDupPolyFeatures instance.

        """

        # only need to generate the constants and duplicates holder objects,
        # dont need to retain the data object constructed while making the
        # holder objects
        self._base_fit(X, return_poly=False)

        return self


    def fit(
        self,
        X: DataType,
        y: Union[Iterable[any], None]=None
    ) -> Self:

        """


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples, n_features) -
            The data to undergo polynomial expansion.
        y:
            {array-like, None} - Always ignored. The target for the data.



        Return
        ------
        -
            self - the fitted NoDupPolyFeatures instance.


        """

        self._stored_poly = self._base_fit(X, return_poly=True)

        return self


    def inverse_transform(
        self,
        X: DataType,
        *,
        copy: bool = None
        ) -> DataType:

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
            dtype=None,  # do not use 'numeric' here, sk will force to float64
            # check for numeric in supplemental X validation
            force_all_finite=True, # blocks nans here
            ensure_2d=True,
            copy=copy or False,
            order='F'
        )

        # since not enforcing dtype, need to do supplemental X validation
        _val_X(X)

        out = _inverse_transform(
            X,
            self.duplicates_,
            self.dropped_duplicates_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None
        )

        if isinstance(out, np.ndarray):
            out = np.ascontiguousarray(out)

        return out


    # def set_params
    # if ever needed, hard code that can be substituted for the
    # BaseEstimator get/set_params can be found in GSTCV_Mixin


    def score(
        self,
        X,
        y:Union[Iterable[any], None]=None
    ) -> None:
        """
        Dummy method to spoof dask Incremental and ParallelPostFit
        wrappers. Verified must be here for dask wrappers.
        """

        pass


    def transform(
        self,
        X: DataType,
        copy: bool
    ) -> DataType:

        """



        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples, n_features) -
            The data to undergo polynomial expansion.


        Return
        -------


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
            dtype=None,  # do not use 'numeric' here, sk will force to float64
            # check for numeric in supplemental X validation
            force_all_finite=True, # blocks nans here
            ensure_2d=True,
            ensure_min_features=2,
            order='F'
        )


        _validation(
            X,
            self.columns,
            self.degree,
            self.min_degree,
            self.drop_duplicates,
            self.keep,
            self.do_not_drop,
            self.conflict,
            self.interaction_only,
            self.include_bias,
            self.drop_constants,
            self.output_sparse,
            self.order,
            self.rtol,
            self.atol,
            self.equal_nan,
            self.n_jobs
        )

        out = _transform(pizza)

        if isinstance(out, np.ndarray):
            out = np.ascontiguousarray(out)

        return out
























