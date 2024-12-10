# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# PIZZA...... decided what to do with this


from typing import Iterable, Literal, Optional, Callable
from typing_extensions import Union, Self
from ._type_aliases import DataType
import numpy.typing as npt

import numbers
import uuid
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.sparse as ss

from ._validation._validation import _validation
from ._validation._X import _val_X
from ._partial_fit._combination_builder import _combination_builder
from ._partial_fit._columns_getter import _columns_getter
from ._partial_fit._parallel_constant_finder import _parallel_constant_finder
from ._partial_fit._get_dupls_for_combo_in_X_and_poly import _get_dupls_for_combo_in_X_and_poly
from ._partial_fit._merge_constants import _merge_constants
from ._partial_fit._merge_partialfit_dupls import _merge_partialfit_dupls
from ._partial_fit._merge_combo_dupls import _merge_combo_dupls
from ._partial_fit._lock_in_random_combos import _lock_in_random_idxs
from ._partial_fit._build_kept_poly_duplicates import _build_attributes
from ._partial_fit._get_active_combos import _get_active_combos
from ._transform._check_X_constants_dupls import _check_X_constants_dupls
from ._transform._build_poly import _transform

from pybear.preprocessing.InterceptManager.InterceptManager import \
    InterceptManager as IM
from pybear.preprocessing.ColumnDeduplicateTransformer.ColumnDeduplicateTransformer import \
    ColumnDeduplicateTransformer as CDT

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted










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
    interaction_only:
        bool - pizza!
    sparse_output:
        bool - pizza!
    feature_name_combiner:
        Union[Callable[[Iterable[str], tuple[int, ...]], str]] - pizza!
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

    poly_constants_:
        dict[tuple[int, ...]: any] - A dictionary whose keys are
        tuples of the indices
        of the constant polynomial columns found during fit, indexed by their column
        location in the original data. The dictionary values are the
        constant values in those columns. For example, if a dataset has
        two constant columns, the first in the third index and the
        constant value is 1, and the other is in the tenth index and the
        constant value is 0, then constant_columns_ will be {(3,):1, (10,):0}.
        If there are no constant columns, then constant_columns_ is an
        empty dictionary.
        dict[tuple[int, ...]: any] - if :param: drop_constants is True, columns of
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

    poly_duplicates_:
        list[list[tuple[int, ...]]] - a list of the groups of identical
        polynomial columns, indicated by tuples of their zero-based
        column index positions in the originally fit data. pizza clarify
        this.

    dropped_poly_duplicates_:
        dict[tuple[int, ...]: tuple[int, ...]] = a list of the groups of
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
        "feature_name_combiner": [callable, None],
        "equal_nan": ["boolean"],
        "rtol": [numbers.Real],
        "atol": [numbers.Real],
        "n_jobs": [numbers.Integral, None],
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
            None
        ]] = None,
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


    @property
    def poly_duplicates_(self) -> list[list[tuple[int, ...]]]:

        check_is_fitted(self)

        self._check_X_constants_and_dupls()

        return self.poly_duplicates_

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
    def kept_poly_duplicates_(self):

        check_is_fitted(self)

        self._check_X_constants_and_dupls()




        return _kept_poly_duplicates




    def _reset(self):
        """
        Reset internal data-dependent state of the transformer.
        __init__ parameters are not changed.

        """

        if hasattr(self, "n_features_in_"):
            del self._poly_duplicates
            del self.dropped_poly_duplicates_
            del self.kept_poly_duplicates_
            del self.poly_constants_
            del self._IM
            del self._CDT


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


    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(
        self,
        X: DataType,
        y: Union[Iterable[any], None]=None
    ):

        """
        pizza say something


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


        X = self._validate_data(
            X=X,
            reset=not hasattr(self, "duplicates_"),
            cast_to_ndarray=False,
            # vvv become **check_params, and get fed to check_array() vvv
            accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
            dtype=None,  # do not use 'numeric' here, sk will force to float64
            # check for numeric in supplemental X validation
            force_all_finite=True, # blocks nans here.... pizza?
            ensure_2d=True,
            ensure_min_features=2,
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


        # pizza, revisit this at the end, see if we need to copy() X
        # convert X to csc to save memory
        # _validation should have caught non-numeric X. X must only be numeric
        # throughout all of SPF.
        if hasattr(X, 'toarray'):   # is scipy sparse
            X = X.tocsc()
        else: # is np or pd
            try:
                X = ss.csc_array(X)
            except:
                X = ss.csc_array(X.astype(np.float64))


        if not hasattr(self, '_poly_duplicates'):
            self._poly_duplicates: list[list[tuple[int, ...]]] = []
        if not hasattr(self, 'dropped_poly_duplicates_'):
            self.dropped_poly_duplicates_: dict[tuple[int, ...]: tuple[int, ...]] = {}
        if not hasattr(self, 'kept_poly_duplicates_'):
            self.kept_poly_duplicates_: dict[tuple[int, ...]: list[tuple[int, ...]]] = {}
        if not hasattr(self, 'poly_constants_'):
            self.poly_constants_: dict[tuple[int, ...]: any] = {}

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

        # Identify constants & duplicates in X v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        # This is just to know if we reject X for having constant or duplicate columns.

        if self.scan_X:
            self._IM.partial_fit(X)
            self._CDT.partial_fit(X)
        # END Identify constants & duplicates in X v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v


        # build a ss csc that holds the unique polynomial columns that are discovered.
        # need to carry this to compare the next calculated polynomial term against
        # the known unique polynomial columns held in this.
        _POLY_CSC = ss.csc_array(np.empty((X.shape[0], 0)).astype(X.dtype))
        IDXS_IN_POLY_CSC: list[tuple[int, ...]] = []
        _poly_dupls_current_partial_fit: list[list[tuple[int, ...]]] = []
        _poly_constants_current_partial_fit: dict[tuple[int, ...]: any] = {}

        
        # GENERATE COMBINATIONS # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
        # need to get the permutations to run, based on the size of x,
        # min degree, max degree, and interaction_only.
        self._combos: list[tuple[int, ...]] = _combination_builder(
            _shape=X.shape,
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

            _COLUMN: npt.NDArray[int, float] = _columns_getter(X, combo).prod(1)

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
                #  constant columns do not need to go into _POLY_CSC to know if they are also
                # a member of duplicates because they are automatically deleted anyway.

                # poly duplicates v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v

                # this function scans the combo column across the columns in X and
                # poly looking for dupls. it returns a vector of bools whose len is
                # X.shape[1] + POLY.shape[1]. if True, then the combo column is
                # a duplicate of the X or POLY column that corresponds to that slot
                # in the list
                _out: list[bool] = _get_dupls_for_combo_in_X_and_poly(
                    _COLUMN,
                    X,
                    _POLY_CSC,
                    _equal_nan=self.equal_nan,
                    _rtol=self.rtol,
                    _atol=self.atol,
                    _n_jobs=self.n_jobs
                )


                # need to convert 'out' to
                # [(i1,), (i2,),..] SINGLE GROUP OF DUPLICATES
                _indices = [(i,) for i in range(X.shape[1])] + IDXS_IN_POLY_CSC
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
                # put it in _POLY_CSC
                if not len(_dupls_for_this_combo):
                    _POLY_CSC = ss.hstack((_POLY_CSC, ss.csc_array(_COLUMN)))
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
        self.poly_constants_ = _merge_constants(
            self.poly_constants_,
            _poly_constants_current_partial_fit,
            _rtol=self.rtol,
            _atol=self.atol
        )
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
        # END poly duplicates -----------------------


        # if 'keep' == 'random', _transform() must pick the same random
        # columns every time. need to set an instance attribute here
        # that doesnt change when _transform() is called. must set a
        # random idx for every set of dupls.
        self._rand_idxs: tuple[tuple[int, ...], ...] = \
            _lock_in_random_idxs(
                poly_duplicates_=self._poly_duplicates,
                _combinations=self._combos
            )


        if len(self._poly_duplicates):
            self.dropped_poly_duplicates_, self.kept_poly_duplicates_ = \
                _build_attributes(
                    self._poly_duplicates,
                    self.keep,
                    self._rand_idxs
                )
        else:
            self.dropped_poly_duplicates_ = {}
            self.kept_poly_duplicates_ = {}


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

        self._reset()

        return self.partial_fit(X)



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
        -
            X_tr: {array-like, scipy sparse} - the polynomial feature expansion for X.

        """


        check_is_fitted(self)

        if not isinstance(copy, (bool, type(None))):
            raise TypeError(f"'copy' must be boolean or None")

        self._check_X_constants_and_dupls()

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

        # pizza, revisit this at the end, see if we need to copy() X
        # convert X to csc to save memory
        # _validation should have caught non-numeric X. X must only be numeric
        # throughout all of SPF.
        # pizza think on this, what about if sparse_output

        _og_dtype = type(X)

        if hasattr(X, 'toarray'):   # is scipy sparse
            X = X.tocsc()
        else: # is np or pd
            try:
                X = ss.csc_array(X)
            except:
                X = ss.csc_array(X.astype(np.float64))

        # SPF params may have changed via set_params. need to recalculate
        # some attributes.
        # poly_constants_ does not change no matter what params are
        # poly_duplicates_ does not change no matter what params are
        # kept_poly_duplicates_ and dropped_poly_duplicates_ might change
        if len(self._poly_duplicates):
            self.dropped_poly_duplicates_, self.kept_poly_duplicates_ = \
                _build_attributes(
                    self._poly_duplicates,
                    self.keep,
                    self._rand_idxs
                )
        else:
            self.dropped_poly_duplicates_ = {}
            self.kept_poly_duplicates_ = {}


        self._active_combos = _get_active_combos(
            self._combos,
            self.poly_constants_,
            self.dropped_poly_duplicates_
        )


        X_tr: Union[np.ndarray, pd.core.frame.DataFrame, ss.csc_array] = \
            _transform(
                X,
                self._active_combos,
                self.n_jobs
            )


        if self.min_degree == 1:
            X_tr = ss.hstack((X, X_tr))


        if self.sparse_output:
            return X_tr.tocsr()
        elif 'scipy' in str(_og_dtype).lower():
            return _og_dtype(X_tr)

        X_tr = X_tr.toarray()

        if _og_dtype is np.ndarray:

            X_tr = np.ascontiguousarray(X_tr)

        elif _og_dtype is pd.core.frame.DataFrame:
            # pizza what about the new feature names for df
            X_tr = pd.DataFrame(
                data=X_tr,
                columns=None# pizza fix this!
            )


        return X_tr
























