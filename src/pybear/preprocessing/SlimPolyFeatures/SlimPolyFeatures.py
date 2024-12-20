# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from typing import Iterable, Literal, Optional, Callable
from typing_extensions import Union, Self
from ._type_aliases import DataType
import numpy.typing as npt

import numbers
import uuid
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as ss

from ._validation._validation import _validation
from ._validation._X import _val_X
from ._attributes._build_kept_poly_duplicates import _build_kept_poly_duplicates
from ._attributes._build_dropped_poly_duplicates import _build_dropped_poly_duplicates
from ._get_feature_names_out._gfno_X import _gfno_X
from ._get_feature_names_out._gfno_poly import _gfno_poly
from ._partial_fit._combination_builder import _combination_builder
from ._partial_fit._columns_getter import _columns_getter
from ._partial_fit._parallel_constant_finder import _parallel_constant_finder
from ._partial_fit._get_dupls_for_combo_in_X_and_poly import _get_dupls_for_combo_in_X_and_poly
from ._partial_fit._merge_constants import _merge_constants
from ._partial_fit._merge_partialfit_dupls import _merge_partialfit_dupls
from ._partial_fit._merge_combo_dupls import _merge_combo_dupls
from ._partial_fit._lock_in_random_combos import _lock_in_random_combos
from ._shared._identify_combos_to_keep import _identify_combos_to_keep
from ._shared._get_active_combos import _get_active_combos
from ._transform._check_X_constants_dupls import _check_X_constants_dupls
from ._transform._build_poly import _build_poly


from pybear.preprocessing.InterceptManager.InterceptManager import \
    InterceptManager as IM
from pybear.preprocessing.ColumnDeduplicateTransformer.ColumnDeduplicateTransformer import \
    ColumnDeduplicateTransformer as CDT

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from ...base import check_is_fitted
from ...utilities import nan_mask



class SlimPolyFeatures(BaseEstimator, TransformerMixin):

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
    by a constant column.) SPF finds these columns before the data is even
    expanded and prevents them from ever appearing in the final array.
    A conventional workflow would be to perform a polynomial expansion on
    data (that fits in memory!), remove duplicate, constant, and multicollinear
    columns, and perform an analysis. The memory occupied by the removed columns
    prevented us from doing a higher order expansion. SPF affords the opportunity to
    (possibly) do higher order expansions than otherwise possible because the
    noisome irritant columns are never even created.


    SlimPolyFeatures (SPF) never returns the 0-degree column (a column of ones)
    and will except if :param: min_degree is set to zero (minimum allowed setting is 1).
    To append a zero-degree column to your data, use pybear InterceptManager.
    Also, SPF does not allow the idempotent case of :param: degree = 1, where
    the original data is returned unchanged. The minimum setting for :param: degree is 2.
    SPF prohibits the degenerate case of only returning the 0-degree column and
    the idempotent case of returning the original data.

    pybear strongly recommends removing duplicate columns from your data set
    with pybear ColumnDeduplicateTransformer. SPF can find constant columns
    within your data, and withhold them from the expansion via the drop_constants
    parameter.


    SPF has a partial_fit method that allows for incremental fitting of
    data. Through this method, SPF is able to learn what columns in the
    original data are constant, and what columns in the expansion would be
    duplicate, constant, or multicollinear. At transform time, SPF applies
    the rules it learned during fitting and only builds the columns that
    add value to the dataset. Through this partial_fit method SPF is
    amenable to batch-wise fitting and transforming via dask_ml Incremental
    and ParallelPostFit wrappers.

    Pizza, talk about SlimPoly tries to keep dtype of original data, instead
    of forcing everything over to 64 bit (unless pandas). This could be a
    disaster if 8 bit multiplies out of range. User take warning.


    Parameters
    ----------
    degree:
        int, default=2 - The maximum polynomial degree of the generated
        features.

    min_degree:
        int, default=0 - The minimum polynomial degree of the generated
        features. Polynomial terms with degree below 'min_degree' are
        not included in the final output array. pizza say something about
        trivial cases.
    scan_X:
        bool, default=True - pizza finish!
    keep:
        Literal['first', 'last', 'random'], default = 'first' -
        The strategy for keeping a single representative from a set of
        identical columns. 'first' retains the column left-most in the
        data; 'last' keeps the column right-most in the data; 'random'
        keeps a single randomly-selected column of the set of duplicates.
        pizza, say that keep is always overruled if there is a column from
        X in the duplicates, keep only applies for picking duplicates out of poly.
        X cannot be mutated by SlimPoly!
    interaction_only:
        bool - pizza!
    sparse_output:
        bool - pizza!
    feature_name_combiner:
        Union[
            Callable[[Iterable[str], tuple[int, ...]], str],
            Literal['as_feature_names', 'as_indices']]
        ], default='as_indices', - pizza!
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
    rtol:
        numbers.Real, default = 1e-5 - The relative difference tolerance for
            equality. See numpy.allclose.
    atol:
        numbers.Real, default = 1e-8 - The absolute tolerance parameter for .
            equality. See numpy.allclose.
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

    expansion_combinations_:
        tuple[tuple[int, ...], ...] - The polynomial column combinations
        from X that are in the polynomial expansion part of the output.
        An example might be ((0,0), (0,1), ...), where the each tuple
        holds the column indices from X that are multiplied to produce
        that feature in the polynomial expansion.

    poly_constants_:
        dict[tuple[int, ...], any] - A dictionary whose keys are
        tuples of the indices of the constant polynomial columns found
        during fit, indexed by their column
        location in the original data. The dictionary values are the
        constant values in those columns. For example, if a dataset has
        two constant columns, the first in the third index and the
        constant value is 1, and the other is in the tenth index and the
        constant value is 0, then constant_columns_ will be {(3,):1, (10,):0}.
        If there are no constant columns, then constant_columns_ is an
        empty dictionary.
        dict[tuple[int, ...], any] - if :param: drop_constants is True, columns of
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
        These are always dropped from the polynomial expansion.

    poly_duplicates_:
        list[list[tuple[int, ...]]] - a list of the groups of identical
        polynomial columns, indicated by tuples of their zero-based
        column index positions in the originally fit data. pizza clarify
        this.

    dropped_poly_duplicates_:
        dict[tuple[int, ...], tuple[int, ...]] = a list of the groups of
        identical polynomial columns, indicated by tuples of their
        zero-based column index positions in the originally fit data.
        pizza clarify this.

        dict[tuple[int, ...], tuple[int, ...]] - a dictionary whose keys are the indices of
        duplicate columns removed from the original data, indexed by
        their column location in the original data; the values are the
        column index in the original data of the respective duplicate
        that was kept.

    kept_poly_duplicates_:
        list[tuple[int, ...]] = []


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
        "degree": [Interval(numbers.Integral, 0, None, closed="left")],
        "min_degree": [Interval(numbers.Integral, 0, None, closed="left")],
        "scan_X": ["boolean"],
        "keep": [StrOptions({"first", "last", "random"})],
        "interaction_only": ["boolean"],
        "sparse_output": ["boolean"],
        "feature_name_combiner": [StrOptions({"as_feature_names", "as_indices"}), callable],
        "equal_nan": ["boolean"],
        "rtol": [numbers.Real],
        "atol": [numbers.Real],
        "n_jobs": [numbers.Integral, None]
    }


    def __init__(
        self,
        degree:Optional[int]=2,
        *,
        min_degree:Optional[int]=1,
        interaction_only: Optional[bool] = False,
        scan_X: Optional[bool] = True,
        keep: Optional[Literal['first', 'last', 'random']] = 'first',
        sparse_output: Optional[bool] = True,
        feature_name_combiner: Optional[Union[
            Callable[[Iterable[str], tuple[int, ...]], str],
            Literal['as_feature_names', 'as_indices']
        ]] = 'as_indices',
        equal_nan: Optional[bool] = True,
        rtol: Optional[numbers.Real] = 1e-5,
        atol: Optional[numbers.Real] = 1e-8,
        n_jobs: Optional[Union[int, None]] = None
    ):

        self.degree = degree
        self.min_degree = min_degree
        self.interaction_only = interaction_only
        self.scan_X = scan_X
        self.keep = keep
        self.sparse_output = sparse_output
        self.feature_name_combiner = feature_name_combiner
        self.equal_nan = equal_nan
        self.rtol = rtol
        self.atol = atol
        self.n_jobs = n_jobs

    # END init ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    def __pybear_is_fitted__(self):

        """
        If an estimator does not set any attributes with a trailing
        underscore,  it can define a '__pybear_is_fitted__' method
        returning a boolean to specify if the estimator is fitted or not.

        """

        # must have this because there are no trailing-underscore attrs
        # generated by {partial_}fit(). all the trailing-underscore attrs
        # are accessed via @property.
        return hasattr(self, '_poly_duplicates')


    def _check_X_constants_and_dupls(self):

        if self.scan_X and hasattr(self, '_IM') and hasattr(self, '_CDT'):
            # if X was scanned for constants and dupls, raise if any
            # were present.
            # its possible that scan_X could be set to True via set_params
            # but _IM and _CDT are not created yet because {partial_}fit()
            # hasnt been called since.
            _check_X_constants_dupls(
                self._IM.constant_columns_,
                self._CDT.duplicates_
            )


    def _attr_access_warning(self) -> str:
        return (f"there are duplicate and/or constant columns in the data. "
            f"the attribute you have requested cannot be returned "
            f"because it is not accurate when the data has duplicate or "
            f"constant columns. this warning is raised and the program "
            f"not terminated to allow for more partial fits.")


    @property
    def expansion_combinations_(self) -> Union[tuple[tuple[int, ...], ...], None]:

        check_is_fitted(self)

        try:
            self._check_X_constants_and_dupls()
            #     self._active_combos must be sorted asc len, then asc on idxs. if _combos is sorted
            #     then this is sorted correctly at construction.
            return self._active_combos
        except:
            warnings.warn(self._attr_access_warning())
            return


    @property
    def poly_duplicates_(self) -> Union[list[list[tuple[int, ...]]], None]:

        check_is_fitted(self)

        try:
            self._check_X_constants_and_dupls()

            return self._poly_duplicates
        except:
            warnings.warn(self._attr_access_warning())
            return

        """
        # pizza delete if not needed 24_12_07_16_19_00
        # need to get the single columns from X out of _poly_duplicates
        _holder: list[list[tuple[int, ...]]] = []
        for _dupl_idx, _dupls in enumerate(deepcopy(self._poly_duplicates)):
            assert len(_dupls) >= 2
            _holder.append([])
            for _tuple in _dupls:
                if len(_tuple) >= 2:
                    _holder[-1].append(_tuple)
            # it shouldnt be possible for a set of _dupls to go empty, there
            # should always be at least one combo term in it
            if len(_holder[-1]) < 1:
                raise AssertionError(
                    f'algorithm failure, _poly_duplicates dupl '
                    f'set does not have a combo tuple in it'
                )

        return _holder
        """


    @property
    def kept_poly_duplicates_(self) -> Union[dict[tuple[int, ...], list[tuple[int, ...]]], None]:

        check_is_fitted(self)

        try:
            self._check_X_constants_and_dupls()

            return _build_kept_poly_duplicates(
                    self._poly_duplicates,
                    self._kept_combos
                )
        except:
            warnings.warn(self._attr_access_warning())
            return


    @property
    def dropped_poly_duplicates_(self) -> Union[dict[tuple[int, ...], tuple[int, ...]], None]:

        check_is_fitted(self)

        try:

            self._check_X_constants_and_dupls()

            return _build_dropped_poly_duplicates(
                    self._poly_duplicates,
                    self._kept_combos
                )

        except:
            warnings.warn(self._attr_access_warning())
            return


    @property
    def poly_constants_(self) -> Union[dict[tuple[int, ...], any], None]:

        check_is_fitted(self)

        try:
            self._check_X_constants_and_dupls()

            return self._poly_constants
        except:
            warnings.warn(self._attr_access_warning())
            return


    def reset(self) -> Self:
        """
        Reset internal data-dependent state of the transformer.
        __init__ parameters are not changed.
        Pizza, this is part of the external API to allow for reset because
        setting most params after a fit has been done is blocked.
        Need to verify if this (or find out how this can) set
        :method: check_is_fitted() back to False.


        """

        if hasattr(self, "_poly_duplicates"):
            del self._poly_duplicates
            del self._poly_constants
            del self._combos

            # _rand_combos, _kept_combos, and _active_combos may not exist even if
            # _poly_duplicates exists because
            # partial_fit short circuits before making them if there
            # are dupls/constants in X. so ask for permission.
            if hasattr(self, '_rand_combos'):
                del self._rand_combos
                del self._kept_combos
                del self._active_combos

            if hasattr(self, '_IM'):
                del self._IM

            if hasattr(self, '_CDT'):
                del self._CDT

        return self


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

        check_is_fitted(self)

        try:
            self._check_X_constants_and_dupls()
        except:
            warnings.warn(self._attr_access_warning())
            return

        # pizza, _gfno_X will likely become something like 'get_feature_names'
        # mock sklearn function in pybear.base. so u will be coming back to this.
        # if did not except....
        _X_header: npt.NDArray[object] = _gfno_X(
            input_features,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            self.n_features_in_
        )

        # there must be a poly header, self.degree must be >= 2
        _poly_header: npt.NDArray[object] = _gfno_poly(
            _X_header,
            self._active_combos,
            self.feature_name_combiner
        )

        if self.min_degree == 1:
            _poly_header = np.hstack((_X_header, _poly_header)).astype(object)
        # else poly header is returned as is

        return _poly_header



    def get_metadata_routing(self):
        """
        Get metadata routing is not implemented in ColumnDeduplicateTransformer.

        """
        __ = type(self).__name__
        raise NotImplementedError(
            f"get_metadata_routing is not implemented in {__}"
        )


    # def get_params(self, deep:bool=?) -> dict[str, any]:
    # if ever needed, hard code that can be substituted for the
    # BaseEstimator get/set_params can be found in GSTCV_Mixin


    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(
        self,
        X: DataType,
        y: Union[Iterable[any], None]=None
    ) -> Self:

        """
        pizza say something

        pizza,
        partial_fit makes a copy of your originally passed X.
        talk about how all pd nan-likes are converted to np.nan.
        This is to allow for validation of datatypes with astype(np.float64) (pd nan-likes blow this up)
        and also to convert X to scipy sparse (pd nan-likes also blow this up!)
        partial_fit always converts the copy of your X to scipy sparse. This
        is to save memory because for SPF to learn what columns in the expansion are duplicate,
        SPF must retain all the unique columns that were found during the expansion process.
        So SPF actually does the expansion out brute force (which can be large), but does not retain the X.
        At transform, the polynomial expansion is built based on what was learned about
        constant and duplicate columns during fitting.


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

        # keep this before _validate_data. when X is junk, _validate_data
        # and check_array except for varying reasons. this standardizes
        # the error message for non-np/pd/ss X.

        _val_X(X, self.interaction_only, self.n_jobs)


        X = self._validate_data(
            X=X,
            reset=not hasattr(self, "_poly_duplicates"),
            # cast_to_ndarray=False,   # pizza takes this out 24_12_13_13_43_00,
            # "TypeError: check_array() got an unexpected keyword argument 'cast_to_ndarray'"
            # started doing this after some changes to _val_X and did poetry install.
            # come back at the end and see if this needs to stay out.

            # vvv become **check_params, and get fed to check_array() vvv
            accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
            dtype=None,  # do not use 'numeric' here, sk will force to float64
            # check for numeric in supplemental X validation
            force_all_finite=False,
            ensure_2d=True,
            ensure_min_features=1,
            order='F'
        )


        _validation(
            X,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            self.degree,
            self.min_degree,
            self.scan_X,
            self.keep,
            self.interaction_only,
            self.sparse_output,
            self.feature_name_combiner,
            self.rtol,
            self.atol,
            self.equal_nan,
            self.n_jobs
        )


        if not hasattr(self, '_poly_duplicates'):
            self._poly_duplicates: list[list[tuple[int, ...]]] = []
        if not hasattr(self, '_poly_constants'):
            self._poly_constants: Union[dict[tuple[int, ...], any], None] = None
            # this must be None on the first pass! _merge_constants needs
            # this to be None to recognize first pass.

        # the only thing that exists at this point is the data and
        # holders. the holders may not be empty.

        if self.scan_X and not hasattr(self, '_IM'):
            self._IM = IM(
                keep=self.keep,
                equal_nan=self.equal_nan,
                rtol=self.rtol,
                atol=self.atol,
                n_jobs=self.n_jobs
            )

        if self.scan_X and not hasattr(self, '_CDT'):
            self._CDT = CDT(
                keep=self.keep,
                do_not_drop=None,
                conflict='ignore',
                equal_nan=self.equal_nan,
                rtol=self.rtol,
                atol=self.atol,
                n_jobs=self.n_jobs
            )

        # Identify constants & duplicates in X v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        # This is just to know if we reject X for having constant or
        # duplicate columns.

        if self.scan_X:
            self._IM.partial_fit(X)
            self._CDT.partial_fit(X)

            try:
                self._check_X_constants_and_dupls()
            except:
                warnings.warn(
                    f"There are duplicate and/or constant columns in the data."
                )

        # END Identify constants & duplicates in X v^v^v^v^v^v^v^v^v^v^v^v^v^v^

        # _validation should have caught non-numeric X. X must only be numeric
        # throughout all of SPF.


        # pizza rewrite all this
        # build a ss csc that holds the unique polynomial columns that are
        # discovered. need to carry this to compare the next calculated
        # polynomial term against the known unique polynomial columns already held
        # in this.
        # ss cant take object dtype. want to keep the bits as low as possible,
        # and preserve whatever dtype may have been passed. this is
        # seeing object dtype when pandas has funky nan-likes. if object dtype,
        # create POLY as np.float64, otherwise keep the original dtype.
        # assigning the lowest bits to this csc is a hairy issue when trying to
        # accommodate pandas dfs because of multiple dtypes. no matter what,
        # if there are any nan-likes in X, POLY must be float64. keeping it
        # simple, just assign float64 to for any pandas, and carry over dtype
        # from ndarray and ss.
        _POLY_CSC = ss.csc_array(np.empty((X.shape[0], 0))).astype(np.float64)

        IDXS_IN_POLY_CSC: list[tuple[int, ...]] = []
        _poly_dupls_current_partial_fit: list[list[tuple[int, ...]]] = []
        _poly_constants_current_partial_fit: dict[tuple[int, ...], any] = {}



        # ss sparse that cant be sliced
        if isinstance(X, (ss.coo_matrix, ss.dia_matrix, ss.bsr_matrix, ss.coo_array,
                       ss.dia_array, ss.bsr_array)):
            warnings.warn(
                f"pybear works hard to avoid mutating or creating copies of your original data. "
                f"\nyou have passed your data as {type(X)}, which cannot be sliced by columns."
                f"pybear needs to create a copy. \nto avoid this, pass your sparse data "
                f"as csr, csc, lil, or dok."
            )
            _X = X.copy().tocsc()
        else:
            _X = X


        
        # GENERATE COMBINATIONS # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
        # need to get the permutations to run, based on the size of x,
        # min degree, max degree, and interaction_only.
        self._combos: list[tuple[int, ...]] = _combination_builder(
            _shape=_X.shape,
            _min_degree=self.min_degree,
            _max_degree=self.degree,
            _intx_only=self.interaction_only
        )
        # END GENERATE COMBINATIONS # v^v^v^v^v^v^v^v^v^v^v^v^v

        # need to iterate over the combos and find what is constant or duplicate
        for combo in self._combos:

            # combo must always be at least degree 2, degree 1 is just the
            # original data and should not be processed here
            assert len(combo) >= 2

            _COLUMN: npt.NDArray[int, float] = _columns_getter(_X, combo).prod(1).ravel()

            __ = _COLUMN.shape
            assert len(__) == 1 or (len(__)==2 and __[1]==1)
            del __

            # poly constants v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

            _poly_is_constant: Union[uuid.UUID, any] = \
                _parallel_constant_finder(
                    _column=_COLUMN,
                    _equal_nan = self.equal_nan,
                    _rtol = self.rtol,
                    _atol = self.atol
                )


            if not isinstance(_poly_is_constant, uuid.UUID):
                _poly_constants_current_partial_fit[combo] = _poly_is_constant

            # END poly constants v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

            if isinstance(_poly_is_constant, uuid.UUID):  # is not constant

                # constant columns do not need to go into _POLY_CSC to know if
                # they are also a member of duplicates because they are
                # automatically deleted anyway.

                # poly duplicates v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v

                # this function scans the combo column across the columns in X and
                # poly looking for dupls. it returns a vector of bools whose len is
                # X.shape[1] + POLY.shape[1]. if True, then the combo column is
                # a duplicate of the X or POLY column that corresponds to that slot
                # in the list
                _out: list[bool] = _get_dupls_for_combo_in_X_and_poly(
                    _COLUMN,
                    _X,
                    _POLY_CSC,
                    _equal_nan=self.equal_nan,
                    _rtol=self.rtol,
                    _atol=self.atol,
                    _n_jobs=self.n_jobs
                )


                # need to convert 'out' to
                # [(i1,), (i2,),..] SINGLE GROUP OF DUPLICATES
                _indices = [(i,) for i in range(_X.shape[1])] + IDXS_IN_POLY_CSC
                _dupls_for_this_combo = []
                for _idx_tuple, _is_dupl in zip(_indices, _out):
                    if _is_dupl:
                        _dupls_for_this_combo.append(_idx_tuple)
                if len(_dupls_for_this_combo):
                    _dupls_for_this_combo.append(combo)

                assert len(_dupls_for_this_combo) != 1

                # need to merge the current _dupls_for_this_combo with
                # _poly_dupls_current_partial_fit. if _dupls_for_this_combo[0]
                # == _dupl_set[0] for any of the dupl sets in _poly_dupls_current_partial_fit,
                # then append the current combo idxs to that list.
                # otherwise add the entire _dupls_for_this_combo to
                # _poly_dupls_current_partial_fit.
                _poly_dupls_current_partial_fit: list[list[tuple[int, ...]]] = \
                    _merge_combo_dupls(
                        _dupls_for_this_combo,
                        _poly_dupls_current_partial_fit
                    )
                # END poly duplicates v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v

                # if _dupls_for_this_combo is empty, then combo column is unique,
                # and if not constant, put it in _POLY_CSC
                if not len(_dupls_for_this_combo):
                    _POLY_CSC = ss.hstack((
                        _POLY_CSC,
                        ss.csc_array(_COLUMN.reshape((-1,1)))
                    ))
                    IDXS_IN_POLY_CSC.append(combo)

                del _poly_is_constant, _dupls_for_this_combo


        # what do we have at this point?

        # the original X as csc_array
        # X constants in _IM()
        # X duplicates in _CDT()
        # _combos
        # _poly_constants_current_partial_fit
        # _poly_dupls_current_partial_fit
        # _POLY_CSC
        # IDXS_IN_POLY_CSC: list[tuple[int, ...]]

        # poly_constants -----------------------
        # pizza, need to meld _poly_constants_current_partial_fit into
        # self.poly_constants_, which would be holding the constants
        # found in previous partial fits

        self._poly_constants = _merge_constants(
            self._poly_constants,
            _poly_constants_current_partial_fit,
            _rtol=self.rtol,
            _atol=self.atol
        )

        del _poly_constants_current_partial_fit
        # END poly_constants -----------------------

        # poly duplicates -----------------------
        # need to merge the current _poly_dupls_current_partial_fit with
        # self._poly_duplicates, which could be holding duplicates found
        # in previous partial fits.
        # need to leave X tuples in here, need to follow the
        # len(dupl_set) >= 2 rule to correctly merge
        # _poly_dupls_current_partial_fit into _poly_duplicates
        # pizza come back to this --- X tuples are removed when
        # @property poly_duplicates_ is called, leaving only poly tuples.
        self._poly_duplicates: list[list[tuple[int, ...]]] = \
            _merge_partialfit_dupls(
                self._poly_duplicates,
                _poly_dupls_current_partial_fit
            )

        del _poly_dupls_current_partial_fit

        # _merge_partialfit_dupls sorts _poly_duplicates on the way out
        # within dupl sets, sort on len asc, then within the same lens sort on values asc
        # across all dupl sets, only look at the first value in a dupl set, sort on len asc, then values asc
        # END poly duplicates -----------------------

        # iff self.poly_constants_ is None it is because @property for it
        # is excepting on self._check_X_constants_and_dupls() and returning
        # None. In that case, all @properties will also trip on that and return None
        # for everything. partial_fit and transform
        # will continue to warn and the @properties will continue to warn as long as the
        # dupl and/or constants condition exists.
        # so because all access points are a no-op when dupls or
        # constants in X, then the below hidden params are not needed. need to
        # skip them because while there are dupls/constants
        # in X, _get_active_combos is calling self.poly_constants_ and
        # self.dropped_poly_duplicates_ and they are returning None which is getting
        # caught in the validation for those modules. so dont even access _get_active_combos.
        # _rand_combos and _kept_combos arent blowing anything up but they arent needed
        # and are just filling with nonsense because of the degenerate state of X.
        if self.poly_constants_ is None:
            return self

        # if 'keep' == 'random', _transform() must pick the same random
        # columns every time. need to set an instance attribute here
        # that doesnt change when _transform() is called. must set a
        # random idx for every set of dupls.
        self._rand_combos: tuple[tuple[int, ...], ...] = \
            _lock_in_random_combos(
                _poly_duplicates=self._poly_duplicates,
                _combinations=self._combos
            )

        # this needs to be before _get_active_combos because _kept_combos
        # is an input into @property dropped_poly_duplicates_
        self._kept_combos: tuple[tuple[int, ...], ...] = \
            _identify_combos_to_keep(
                self._poly_duplicates,
                self.keep,
                self._rand_combos
            )


        self._active_combos = _get_active_combos(
            self._combos,
            self.poly_constants_,
            self.dropped_poly_duplicates_
        )


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

        self.reset()

        return self.partial_fit(X)



    # pizza verify this cant be called from the Mixins
    # def inverse_transform()
        # raise NotImplementedError


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


    def set_params(self, **params):

        """
        Pizza, this jargon was taken directly from BaseEstimator 24_12_10_16_56_00

        Set the parameters of this estimator.

        Once this transformer is fitted, only :params: 'sparse_output',
        'keep', 'feature_name_combiner', and 'n_jobs' can be changed via
        :method: set_params. All other parameters are blocked. To use
        different parameters without creating a new instance of this
        transformer class, call :method: reset on this instance,
        otherwise create a new instance of this transformer class."

        This method works on simple estimators as well as on nested objects
        (such as :class: `sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update parameters of nested objects.


        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        -
            self : transformer instance.


        """

        # if ever needed, hard code that can be substituted for the
        # BaseEstimator get/set_params can be found in GSTCV_Mixin

        try:

            # this check must stay under try! if this is fitted, then run
            # everything else in the try which imposes blocks on some params.
            # if not fitted, except out and allow everything to be set
            check_is_fitted(self)

            allowed_params = (
                'keep', 'sparse_output', 'feature_name_combiner', 'n_jobs'
            )

            _valid_params = {}
            _invalid_params = {}
            for param in params:
                if param in allowed_params:
                    _valid_params[param] = params[param]
                else:
                    _invalid_params[param] = params[param]

            warnings.warn(
                "Once this transformer is fitted, only :params: 'keep', "
                "'sparse_output', 'feature_name_combiner', and 'n_jobs' "
                "can be changed via :method: set_params. All other parameters "
                f"are blocked. The currently passed parameters {', '.join(list(_invalid_params))} have "
                f"been blocked, but any valid parameters that were passed have been set."
                "To use different parameters without creating a "
                "new instance of this transformer class, call :method: reset "
                "on this instance, otherwise create a new instance of this "
                "transformer class."
            )

            super().set_params(**_valid_params)

        except:

            super().set_params(**params)


    def transform(
        self,
        X: DataType,
        copy: Union[bool, None]=None
    ) -> DataType:

        """



        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples, n_features) -
            The data to undergo polynomial expansion.
        copy:
            Union[bool, None] -


        Return
        -------
        -
            X_tr: {array-like, scipy sparse} - the polynomial feature expansion for X.

        """


        check_is_fitted(self)

        # this does a no-op if there are dupls or constants in X
        # returns None with a warning, allowing for more partial fits
        try:
            self._check_X_constants_and_dupls()
        except:
            warnings.warn(self._attr_access_warning())
            return


        if not isinstance(copy, (bool, type(None))):
            raise TypeError(f"'copy' must be boolean or None")


        # keep this before _validate_data. when X is junk, _validate_data
        # and check_array except for varying reasons. this standardizes
        # the error message for non-np/pd/ss X.
        # once _validata_data disappears, this can probably go back into _validation()
        _val_X(X, self.interaction_only, self.n_jobs)

        # _validation should have caught non-numeric X. X must only be numeric
        # throughout all of SPF.

        X = self._validate_data(
            X=X,
            reset=False,
            cast_to_ndarray=False,
            copy=copy or False,
            accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
            dtype=None,  # do not use 'numeric' here, sk will force to float64
            # check for numeric in supplemental X validation
            force_all_finite=False,
            ensure_2d=True,
            ensure_min_features=1,
            order='F'
        )


        _validation(
            X,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            self.degree,
            self.min_degree,
            self.scan_X,
            self.keep,
            self.interaction_only,
            self.sparse_output,
            self.feature_name_combiner,
            self.rtol,
            self.atol,
            self.equal_nan,
            self.n_jobs
        )


        _og_dtype = type(X)


        # SPF params may have changed via set_params. need to recalculate
        # some attributes.
        # poly_constants_ does not change no matter what params are
        # poly_duplicates_ does not change no matter what params are
        # kept_poly_duplicates_ and dropped_poly_duplicates_ might change
        # based on keep

        # this needs to be before _get_active_combos because _kept_combos
        # is an input into @property dropped_poly_duplicates_
        self._kept_combos: tuple[tuple[int, ...], ...] = \
            _identify_combos_to_keep(
                self._poly_duplicates,
                self.keep,
                self._rand_combos
            )

        # ss sparse that cant be sliced
        if isinstance(X, (ss.coo_matrix, ss.dia_matrix, ss.bsr_matrix, ss.coo_array,
                       ss.dia_array, ss.bsr_array)):
            warnings.warn(
                f"pybear works hard to avoid mutating or creating copies of your original data. "
                f"\nyou have passed your data as {type(X)}, which cannot be sliced by columns."
                f"\npybear needs to create a copy. \nto avoid this, pass your sparse data "
                f"as csr, csc, lil, or dok."
            )
            _X = X.copy().tocsc()
        # pd df with funky nan-likes that np and ss dont like
        elif self.min_degree == 1 and isinstance(X, pd.core.frame.DataFrame):
            try:
                X.astype(np.float64)
                # if excepts, there are pd nan-likes that arent recognized by numpy.
                # if passes, this df should just hstack with X_tr without a problem.
                _X = X
            except:
                warnings.warn(
                    f"pybear works hard to avoid mutating or creating copies of your original data."
                    f"\nyou have passed a dataframe that has nan-like values "
                    f"that are not recognized by numpy/scipy. \nbut to merge this data with "
                    f"the polynomial expansion, pybear must make a copy to replace all "
                    f"the nan-likes with numpy.nan. \nto avoid this copy, pass your dataframe "
                    f"with numpy.nan in place of any nan-likes that are only recognized by pandas."
                )
                _X = X.copy()
                _X[nan_mask(_X)] = np.nan
        else:
            _X = X


        self._active_combos = _get_active_combos(
            self._combos,
            self.poly_constants_,
            self.dropped_poly_duplicates_
        )

        X_tr: ss.csc_array = \
            _build_poly(
                _X,
                self._active_combos,
                self.n_jobs
            )

        # experiments show that if stacking with ss.hstack:
        # 1) at least one of the terms must be a scipy sparse
        # 2) if one is ss, and the other is not, always returns as COO
        #       regardless of what ss format was passed
        # 3) if both are ss, but are different types of ss, always returns as COO
        # 4) only when both are the same type of ss is that type of ss returned
        # 5) it is OK to mix ss array and ss matrix, array will trump matrix
        # so we need to convert X to whatever X_tr is to maintain X_tr format
        if self.min_degree == 1:
            # this is excepting when trying to do type(X_tr)(_X) when type(X_tr)
            # is ss and _X.dtype is object or str. we know from _validation that
            # X is numeric, if original X dtype is str or object set the dtype
            # of the merging X to float64
            X_tr = ss.hstack((type(X_tr)(_X.astype(np.float64)), X_tr))

        assert isinstance(X_tr, ss.csc_array)

        if self.sparse_output:
            return X_tr.tocsr()
        elif 'scipy' in str(_og_dtype).lower():
            # if input was scipy, but not 'sparse_output', return in the
            # original scipy format
            return _og_dtype(X_tr)
        else:
            # ndarray or pd df, return in the given format
            X_tr = X_tr.toarray()

            # pizza this will probably come out since abandoning sklearn _validate_data
            if _og_dtype is np.ndarray:
                return np.ascontiguousarray(X_tr)

            elif _og_dtype is pd.core.frame.DataFrame:
                return pd.DataFrame(
                    data=X_tr,
                    columns=self.get_feature_names_out()
                )
            else:
                raise Exception



























