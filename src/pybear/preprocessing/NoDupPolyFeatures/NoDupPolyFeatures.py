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
from ._base_fit._combination_builder import _combination_builder
from ._base_fit._column_getter import _column_getter
from ._base_fit._parallel_column_comparer import _parallel_column_comparer
from ._base_fit._parallel_constant_finder import _parallel_constant_finder
from ._base_fit._merge_constants import _merge_constants
from ._base_fit._merge_dupls import _merge_dupls
from ._transform._transform import _transform

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted










class NoDupPolyFeatures(BaseEstimator, TransformerMixin):

    """
    Perform a polynomial feature expansion on a set of data and optionally
    omit duplicate and / or constant columns. Generate a new feature matrix
    consisting of all polynomial combinations
    of the features with degree less than or equal to the specified degree.
    For example, if an input sample is two dimensional and of the form
    [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    Polynomial feature expansion is useful for finding non-linear relationships
    of the data against the target.

    Unfortunately, even modest-sized datasets can quickly expand to sizes
    beyond RAM. This limits the number of polynomial terms that can be analyzed.
    Many polynomial expansions generate columns that are duplicate,
    constant (think all zeros from interactions on a one-hot encoded feature),
    or directly proportional to another column (think columns being multiplied
    by a constant column.) NDPF finds these columns before the data is even
    expanded and prevents them from ever appearing in the final array.
    A conventional workflow would be to perform a polynomial expansion on
    data (that fits in memory!), remove duplicate, constant, and multicollinear
    columns, and perform an analysis. The memory occupied by the removed columns
    prevented us from doing a higher order expansion. NDPF affords the opportunity to
    (possibly) do higher order expansions than otherwise possible because the
    noisome irritant columns are never even created.


    NoDupPolyFeatures (NDPF) never returns the 0-degree column (a column of ones)
    and will except if :param: min_degree is set to zero (minimum allowed setting is 1).
    To append a zero-degree column to your data, use pybear InterceptManager.
    Also, NDPF does not allow the idempotent case of :param: degree = 1, where
    the original data is returned unchanged. The minimum setting for :param: degree is 2.
    NDPF prohibits the degenerate case of only returning the 0-degree column and
    the idempotent case of returning the original data.

    pybear strongly recommends removing duplicate columns from your data set
    with pybear ColumnDeduplicateTransformer. NDPF can find constant columns
    within your data, and withhold them from the expansion via the drop_constants
    parameter.


    NDPF has a partial_fit method that allows for incremental fitting of
    data. Through this method, NDPF is able to learn what columns in the
    original data are constant, and what columns in the expansion would be
    duplicate, constant, or multicollinear. At transform time, NDPF applies
    the rules it learned during fitting and only builds the columns that
    add value to the dataset. Through this partial_fit method NDPF is
    amenable to batch-wise fitting and transforming via dask_ml Incremental
    and ParallelPostFit wrappers.


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
        list[list[tuple[int, ...]]] - a list of the groups of identical
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
        min_degree:Optional[int]=1,
        interaction_only: Optional[bool] = False,
        drop_duplicates: Optional[bool] = True,
        drop_constants: Optional[bool] = True,
        keep: Optional[Literal['first', 'last', 'random']] = 'first',
        output_sparse: Optional[bool] = False,
        equal_nan: Optional[bool] = False,
        rtol: Optional[float] = 1e-5,
        atol: Optional[float] = 1e-8,
        n_jobs: Optional[Union[int, None]] = None
    ):

        self.degree = degree
        self.min_degree = min_degree
        self.interaction_only = interaction_only
        self.drop_duplicates = drop_duplicates
        self.drop_constants = drop_constants
        self.keep = keep
        self.output_sparse = output_sparse
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

        if hasattr(self, "n_features_in_"):
            del self.poly_duplicates_
            del self.dropped_poly_duplicates_
            del self.kept_poly_duplicates_
            del self.X_constants_
            del self.poly_constants_
            del self.dropped_poly_constants_
            del self.kept_poly_constants_


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



    # def get_params(self, deep:bool=?) -> dict[str: any]:
    # if ever needed, hard code that can be substituted for the
    # BaseEstimator get/set_params can be found in GSTCV_Mixin



    def _base_fit(
        self,
        X: DataType,
        return_poly:bool=False
    ):

        """
        pizza say something


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples, n_features) -
            The data to undergo polynomial expansion.
        return_poly:
            bool - Whether to cache the polynomial expansion object
            created while finding the columns of constants and duplicates.


        Return
        ------
        -
            self - the fitted NoDupPolyFeatures instance.

        """

        _duplicates: dict[tuple[int], list[tuple[int]]]
        _constants: dict[tuple[int], any]

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
            self.interaction_only,
            self.drop_constants,
            self.output_sparse,
            self.rtol,
            self.atol,
            self.equal_nan,
            self.n_jobs
        )

        if not hasattr(self, 'X_constants_'):
            self.X_constants_ = {}
        if not hasattr(self, 'poly_duplicates_'):
            self.poly_duplicates_ = {}
        if not hasattr(self, 'dropped_poly_duplicates_'):
            self.dropped_poly_duplicates_ = {}
        if not hasattr(self, 'kept_poly_duplicates_'):
            self.kept_poly_duplicates_ = {}
        if not hasattr(self, 'poly_constants_'):
            self.poly_constants_ = {}
        if not hasattr(self, 'dropped_poly_constants_'):
            self.dropped_poly_constants_ = {}
        if not hasattr(self, 'kept_poly_constants_'):
            self.kept_poly_constants_ = {}

        # the only thing that exists at this point is the data and possibly
        # holders. the holders may not be empty.




        # Identify constants in X v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        # This is just to know which input columns to take out of the expansion
        # if drop_constants=True.
        # They will not be removed from X.

        # cannot overwrite self.constants_! may have previous fits in it
        # pizza move this import
        from pybear.preprocessing.InterceptManager.InterceptManager import \
            InterceptManager as IM
        _X_current_constants: dict[int: any] = \
            list(
                IM(
                    keep=self.keep,
                    equal_nan=self.equal_nan,
                    rtol=self.rtol,
                    atol=self.atol,
                    n_jobs=self.n_jobs
                ).partial_fit(X).constant_columns_
            )


        # pizza, need to meld _constants into self.constants_ -- self.constants_
        # would be holding the constants found in previous partial fits
        # remember that if there already is constants, it might have constants
        # from the polynomial terms too!

        # merge _X_constant_idxs and _constants, then can be set to self.constants_
        self.X_constants_ = _merge_constants(
            self.X_constants_,
            _X_current_constants,
            _rtol=self.rtol,
            _atol=self.atol
        )

        # END Identify constants in X v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v


        # build an array that holds the unique polynomial columns that are discovered.
        # need to carry this to compare the next calculated polynomial term against
        # X and the known unique polynomial columns in this.
        POLY_ARRAY = np.empty((X.shape[0], 0)).astype(X.dtype)
        IDXS_IN_POLY_ARRAY = []
        _duplicates_current_partial_fit = []

        # what we know for the first partial fit is that X has certain constants,
        # and that the first expansion may have certain constants and certain
        # duplicates. since we havent seen future Xs, we dont know the global
        # constants in X nor the global constants and duplicates in the expansion.
        # Future Xs may cause past constants in X to no longer be constant, and
        # may cause past constants and duplicates in the expansion to no longer
        # be constants or duplicates. But not the other way around; future
        # constants and duplicates cannot make a whole column become a constant
        # or a duplicate if it wasnt already.
        # Therefore, it doesnt matter what is currently constant or
        # duplicate for partial fits, we cant exclude them because they may
        # expose differently in the future, so we need to keep track of
        # absolutely everything.

        # we need to get everything because set_params() might change after fit!

        # need to iterate over the combos and find what is constant or duplicate

        # GENERATE COMBINATIONS W/ CONSTANTS IN # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
        # need to get the permutations to run, based on the size of x,
        # in degree, max degree, and interaction_only.
        # since we dont know what the constants are in future Xs, need to keep the
        # constants in the current combinations
        _combos = _combination_builder(
            _shape=X.shape,
            _min_degree=self.min_degree,
            _max_degree=self.degree,
            _intx_only=self.interaction_only
        )
        # END GENERATE COMBINATIONS W/ CONSTANTS IN # v^v^v^v^v^v^v^v^v^v^v^v^v

        for combo in _combos:

            _COLUMN = _column_getter(X, combo).product(1)

            # there are no constants put into POLY_ARRAY but are recorded in
            # constant_columns_ (pizza?)
            # if we are looking at duplicates, there are no duplicates in POLY_ARRAY,
            # but may be in X

            # if we are looking at both, neither is in poly array. in this case,
            # we can look at it being constant first, that is less expensive to
            # get. we already know a column of constants cant be a duplicate of
            # POLY, but it could be a duplicate of X

            import uuid  # pizza move this
            _poly_is_constant: Union[uuid.UUID, any] = \
                _parallel_constant_finder(
                    _column=_COLUMN,
                    _equal_nan = self.equal_nan,
                    _rtol = self.rtol,
                    _atol = self.atol
                )

            if not isinstance(_poly_is_constant, uuid.UUID):
                self.poly_constants_[combo] = _poly_is_constant

            del _poly_is_constant

            # duplicates v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
            # if removing duplicates, identify them in X --- need to do this
            # every time, set_params may be changed
            # cannot overwrite self.duplicates_! may have previous fits in it

            joblib_kwargs = \
                {'prefer': 'processes', 'n_jobs': self.n_jobs, 'return_as': 'list'}
            args = (_COLUMN, self.rtol, self.atol, self.equal_nan)

            # def _parallel_column_comparer(
            #     _column1: npt.NDArray[any],
            #     _column2: npt.NDArray[any],
            #     _rtol: float,
            #     _atol: float,
            #     _equal_nan: bool
            # ) -> bool:

            # look_for_duplicates_in_X
            # _look_for_duplicates_in_X needs to return the idx in X that the
            # combo matches so both X idx and combo can be put into the
            # _duplicates dict

            # get 1 representative out of any dupl sets
            import itertools   # pizza move this
            INACTIVE_X_IDXS = [[i for i in _set if len(i)==1][1:] for _set in _duplicates_current_partial_fit]
            ACTIVE_X_IDXS = [i for i in range(X.shape[1]) if i not in itertools.chain(*INACTIVE_X_IDXS)]
            del INACTIVE_X_IDXS

            # there can be more than one hit for duplicates here
            _out_X = Parallel(**joblib_kwargs)(
                _parallel_column_comparer(X[:, c_idx_tuple], *args) for c_idx_tuple in ACTIVE_X_IDXS
            )
            del ACTIVE_X_IDXS


            # if there is a duplicate in X, there cannot be a duplicate here.
            # if there is no duplicate in X, there can only be zero or one duplicate here.

            # get 1 representative out of any dupl sets
            INACTIVE_POLY_IDXS = [[i for i in _set if len(i)==2][1:] for _set in _duplicates_current_partial_fit]
            ACTIVE_POLY_IDXS = [i for i in IDXS_IN_POLY_ARRAY if i not in itertools.chain(*INACTIVE_POLY_IDXS)]
            del INACTIVE_POLY_IDXS
            _out_poly = Parallel(**joblib_kwargs)(
                _parallel_column_comparer(POLY_ARRAY[:, c_idx_tuple], *args) for c_idx_tuple in ACTIVE_POLY_IDXS
            )
            del ACTIVE_POLY_IDXS

            if any(_out_X):
                assert not any(_out_poly)
            elif not any(_out_X):
                assert sum(_out_poly) in [0,1]

            _out = _out_X + _out_poly
            del _out_X, _out_poly
            _indices = list(range(X.shape[1])) + IDXS_IN_POLY_ARRAY

            # need to convert 'out' to
            # [(i1,), (i2,),..] SINGLE GROUP OF DUPLICATES
            _duplicates_for_this_combo = []
            for _idx_tuple, _is_dupl in zip(_indices, _out):
                if _is_dupl:
                    _duplicates_for_this_combo.append(_idx_tuple)
            if len(_duplicates_for_this_combo):
                _duplicates_for_this_combo.append(combo)


            # need to merge the current _duplicates_for_this_combo with
            # _duplicates_current_partial_fit. if the sequence in
            # _duplicates_for_this_combo[:-1] matches something, then
            # replace with _duplicates_for_this_combo. otherwise add it.
            if not len(_duplicates_current_partial_fit):
                _duplicates_current_partial_fit.append(_duplicates_for_this_combo)
            else:
                _hits = 0
                for _idx, _set in enumerate(_duplicates_current_partial_fit):
                    if np.array_equal(_set, _duplicates_for_this_combo[:-1]):
                        _duplicates_current_partial_fit[_idx] = _duplicates_for_this_combo
                        _hits += 1
                if _hits > 1:
                    raise Exception(
                        f"duplicate dupl set in _duplicates_current_partial_fit"
                    )
                del _hits

            # END duplicates v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

            # if we get to this point, then we cared if _COLUMN was a constant
            # or a duplicate, but it was neither, so _COLUMN goes into POLY_ARRAY
            POLY_ARRAY = np.hstack((POLY_ARRAY, _COLUMN))
            IDXS_IN_POLY_ARRAY.append(combo)

        # need to merge the current _duplicates_current_partial_fit with
        # self._duplicates.
        # pizza, _duplicates needs to be melded into self.duplicates_ ---
        # which would be holding duplicates found in previous partial fits
        # remember that if there already is duplicates, it will have duplicates
        # from the polynomial terms too!
        self.duplicates_ = _merge_dupls(self.duplicates_, _duplicates_current_partial_fit)


        # what do we have at this point?

        # _X_constant_idxs --- good as is since its one fit
        # combinations (w/o constants in X)
        # the original X  --- fit to be returned
        # POLY_ARRAY --- this is correctly expanded out and would be fit to be returned
        # IDXS_IN_POLY_ARRAY:list[tuple[int]] nuff ced
        # _constants: dict[tuple[int], any]
        # _duplicates: dict[tuple[int], list[tuple[int]]
        # _duplicates needs to be converted to duplicates_:list[list[tuple]]
        # see CDT _find_duplicates.






        if return_poly:
            # if transforming straight from here:
            # hstack((X, POLY_ARRAY)) and return
            # if doing just one fit(), then keep this object and just return it in
            # transform.
            return _pizza_data_object
        else:
            # pizza, if doing partial fit, then this object doesnt need to be stored,
            # all u need to know from each partial_fit is the columns of constants and
            # duplicate columns, then apply them in transform.
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
        # there is no point in retaining the data object constructed while making
        # the holder objects
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

        self._reset()

        self._stored_poly = self._base_fit(X, return_poly=True)

        return self



    # pizza verify this cant be called from the Mixins
    # def inverse_transform()
        # raise NotImplementedError


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

        out = _transform(X)

        if isinstance(out, np.ndarray):
            out = np.ascontiguousarray(out)

        return out
























