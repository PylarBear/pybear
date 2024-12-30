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
from ._attributes._build_kept_poly_duplicates import \
    _build_kept_poly_duplicates
from ._attributes._build_dropped_poly_duplicates import \
    _build_dropped_poly_duplicates
from ._get_feature_names_out._gfno_X import _gfno_X
from ._get_feature_names_out._gfno_poly import _gfno_poly
from ._partial_fit._combination_builder import _combination_builder
from ._partial_fit._columns_getter import _columns_getter
from ._partial_fit._deconstant_poly_dupls import _deconstant_poly_dupls
from ._partial_fit._parallel_constant_finder import _parallel_constant_finder
from ._partial_fit._get_dupls_for_combo_in_X_and_poly import \
    _get_dupls_for_combo_in_X_and_poly
from ._partial_fit._merge_constants import _merge_constants
from ._partial_fit._merge_partialfit_dupls import _merge_partialfit_dupls
from ._partial_fit._merge_combo_dupls import _merge_combo_dupls
from ._partial_fit._lock_in_random_combos import _lock_in_random_combos
from ._shared._identify_combos_to_keep import _identify_combos_to_keep
from ._shared._get_active_combos import _get_active_combos
from ._shared._check_X_constants_dupls import _check_X_constants_dupls
from ._transform._build_poly import _build_poly


from pybear.preprocessing.InterceptManager.InterceptManager import \
    InterceptManager as IM
from pybear.preprocessing.ColumnDeduplicateTransformer. \
    ColumnDeduplicateTransformer import ColumnDeduplicateTransformer as CDT


from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from ...base import check_is_fitted
from ...utilities import nan_mask



class SlimPolyFeatures(BaseEstimator, TransformerMixin):

    """
    SlimPolyFeatures (SPF) performs a polynomial feature expansion on a
    dataset, where any feature produced that is a column of constants or
    is a duplicate of another column is omitted from the final output.

    A polynomial feature expansion generates all possible multiplicative
    combinations of the original features, typically from the zero-degree
    term up to and including a specified maximum degree. For example, if
    a dataset has two features, called 'a' and 'b', then the degree-2
    polynomial features are [1, a, b, a^2, ab, b^2], where the column of
    ones is the zero-degree term, columns 'a' and 'b' are the first
    degree terms, and columns a^2, ab, and b^2 are the second degree
    terms. Generating these features for analysis is useful for finding
    non-linear relationships between the data and the target.

    Unfortunately, even modest-sized polynomial expansions can quickly
    grow to sizes beyond RAM. This limits the number of polynomial terms
    that can be analyzed. But there is opportunity to reduce memory
    footprint for a given expansion by eliminating features that do not
    add value. Many polynomial expansions generate columns that are
    duplicate or constant (think columns of all zeros from interactions
    on a one-hot encoded feature). SPF finds columns such as these before
    the data is fully expanded and prevents them from ever appearing in
    the final array (and occupying memory!)

    A typical workflow would be to perform a polynomial expansion on data
    (that fits in memory!), remove duplicate and constant columns, and
    perform an analysis. The memory occupied by the columns that were
    eventually removed may have prevented us from doing a higher order
    expansion, which may have provided useful information. SPF affords
    the opportunity to (possibly) do higher order expansions than
    otherwise possible because the non-value-added columns are never even
    created. SPF also saves the time of writing code to find and remove
    such columns after the expansion.

    To robustly and tractably do this, pybear requires that the dataset
    to undergo expansion have no duplicate or constant columns. There is
    further discussion in these documents about how SlimPolyFeatures
    handles these conditions in-situ during multiple partial fits, but
    ultimately SPF requires the totality of seen data have no constants
    and no duplicates. Remove duplicate columns from your data beforehand
    with pybear ColumnDeduplicateTransformer. Remove constant columns
    from your data with pybear InterceptManager.

    SPF is able to learn what columns in the original data are
    constant/duplicate and what columns in the expansion would be
    constant/duplicate. At transform time, SPF applies the rules it
    learned during fitting and only builds the columns that add value to
    the dataset.

    During the fitting process, partial_fit always generates the preliminary expansion as scipy sparse. This
    is to save memory because for SPF to learn what columns in the expansion are duplicate,
    SPF must retain all the unique columns that were found during the expansion process.
    So SPF actually does the expansion out brute force (which can be large), but does not retain the X.
    At transform, the polynomial expansion is built based on what was learned about
    constant and duplicate columns during fitting.

    The internal construction of the polynomial expansion in SPF is
    always as a scipy sparse csc array to maximize the amount of
    expansion that could be done. The expansion is also always done with
    float64 datatypes, regardless of datatype of the passed data, to
    prevent any overflow problems that might arise from multiplication
    of low bit number types. However, overflow is still possible with 64
    bit datatypes, especially when the data has large values or the
    degree of the expansion is high. SPF HAS NO PROTECTIONS FOR OVERFLOW.
    IT IS UP TO THE USER TO AVOID OVERFLOW CONDITIONS AND VERIFY RESULTS.

    SPF follows the standard scikit-learn transformer API, and makes
    the standard transformer methods available: fit, partial_fit,
    transform, set_params, get_params, and get_feature_names_out. SPF
    also has a reset method which is covered elsewhere in the docs.

    Numpy arrays, pandas dataframes, and all scipy sparse objects (csr,
    csc, coo, lil, dia, dok, bsr) are accepted by :methods: partial_fit,
    fit, and transform.

    SPF also has a 'sparse_output' parameter that can also increase the
    amount of useful expansion features held in memory. If :param:
    sparse_output is set to True, that expansion is returned as scipy
    sparse csr array. Only if :param: sparse_output is set to False is
    the data returned in the format passed to :method: transform.

    The SPF partial_fit method allows for incremental fitting. Through
    this method, even if the data is bigger-than-memory, SPF is able to
    learn what columns in X are constant/duplicate and what columns
    in the expansion are constant/duplicate, and carry out instructions
    to build the polynomial batch-wise.  This partial_fit method makes
    SPF amenable to batch-wise fitting and transforming, such as via
    dask_ml Incremental and ParallelPostFit wrappers.

    SPF takes parameters to set the minimum (:param: min_degree) and
    maximum degrees (:param: degree) of the polynomial terms produced
    during the expansion. The edge case of returning only the 0-degree
    column of ones is disallowed. SPF never returns the 0-degree column
    of ones under any circumstance. The lowest degree SPF ever returns
    is degree one (the original data in addition to whatever other order
    terms are required). SPF terminates if :param: min_degree is set to
    zero (minimum allowed setting is 1). To append a zero-degree column
    to your data, use pybear InterceptManager after using SPF. Also, SPF
    does not allow the no-op case of :param: degree = 1, where the
    original data would be returned unchanged. The minimum setting
    for :param: degree is 2.

    During fitting, SPF is able to tolerate constants and duplicates in
    the data. While this condition exists, however, SPF remains in a
    state where it waits for further partial fits to remedy the situation
    and does no-ops with warnings on most other actions (such as calls
    to attributes, the transform method, amongst others.) Only when the
    internal state of SPF is satisfied that there are no constant or
    duplicates are in the training data will SPF then allow access to
    the other functionality.

    SPF MUST ONLY TRANSFORM DATA IT HAS BEEN FITTED ON. TRANSFORMATION OF
    ANY OTHER DATA WILL PRODUCE OUTPUT THAT MAY CONTAIN CONSTANTS AND
    DUPLICATES.


    Parameters
    ----------
    degree:
        numbers.Integral, default=2 - The maximum polynomial degree of
        the generated features. The minimum value accepted by SPF is 2;
        the no-op case of simply returning the original degree-one data
        is not allowed.
    min_degree:
        numbers.Integral, default=1 - The minimum polynomial degree of
        the generated features. Polynomial terms with degree below
        'min_degree' are not included in the final output array. The
        minimum value accepted by SPF is 1; SPF cannot be used to
        generate a zero-degree column (a column of all ones).
    interaction_only:
        bool, default = False - If True, only interaction features are
        produced, that is, polynomial features that are products of
        'degree' distinct input features. Terms with power of 2 or higher
        for any feature are excluded. If False, produce the full
        polynomial expansion.
        Consider 3 features 'a', 'b', and 'c'. If 'interaction_only' is
        True, 'min_degree' is 1, and 'degree' is 2, then only the first
        degree interaction terms ['a', 'b', 'c'] and the second degree
        interaction terms ['ab', 'ac', 'bc'] are returned in the
        polynomial expansion.
    scan_X:
        bool, default=True - SPF requires that the data being fit has
        no columns of constants and no duplicate columns. When True, SPF
        does not assume that the user knows these states of the data
        and diagnoses them, which can be very expensive to do, especially
        finding duplicate columns. If it is known that there are no
        constant or duplicate columns in the data, setting this to False
        can greatly reduce the cost of the polynomial expansion. When in
        doubt, pybear recommends setting this to True (the default).
        When this is False, it is possible to pass columns of constants
        or duplicates, but pybear will continue to operate by the stated
        design requirement, and the output will be nonsensical.
    keep:
        Literal['first', 'last', 'random'], default = 'first' -
        The strategy for keeping a single representative from a set of
        identical columns in the polynomial expansion.
        This is over-ruled if a polynomial feature is a duplicate of one
        of the original features, and the original feature will always be
        kept and the polynomial duplicates will always be dropped.
        One of SPF's design rules is to never alter the originally passed
        data, so the original feature will always be kept. Under SPF's
        design rule that the original data has no duplicate columns, an
        expansion feature cannot be identical to 2 of the original
        features. In all cases where the duplicates are only within the
        polynomial expansion, 'first' retains the column left-most in the
        expansion (lowest degree); 'last' keeps the column right-most in
        the expansion (highest degree); 'random' keeps a single
        randomly-selected feature of the set of duplicates.
    sparse_output:
        bool, default = True - If set to True, the polynomial expansion
        is returned from :method: transform as a scipy sparse csr array.
        If set to False, the polynomial expansion is returned in the
        same format as passed to :method: transform.
    feature_name_combiner:
        Union[
            Callable[[Iterable[str], tuple[int, ...]], str],
            Literal['as_feature_names', 'as_indices']]
        ], default='as_indices' - Sets the naming convention for the
        created polynomial features. This does not set nor change any
        original feature names that may have been seen during fitting on
        pandas dataframes.

        feature_name_combiner must be:
        1) Literal 'as_feature_names',
        2) Literal 'as_indices',
        - or -
        3) a user-defined function (callable) for mapping polynomial
            column index combination tuples to polynomial feature names.

        If the default literal 'as_feature_names' is used, SPF generates
        new polynomial feature names based on the feature names in the
        original data. For example, if the feature names of X are
        ['x0', 'x1', ..., 'xn'] and the polynomial column index
        tuple is (1, 1, 3), then the polynomial feature name is
        'x1^2_x3'.

        If the default literal 'as_indices' is used, SPF generates new
        polynomial feature names based on the polynomial column index
        tuple itself. For example, if the polynomial column index tuple
        is (2, 2, 4), then the polynomial feature name is '(2, 2, 4)'.

        If a user-defined callable is passed, it must:
        A) Accept 2 arguments:
            1) a 1D vector of strings that contains the original feature
                names of X, as is used internally in SPF,
            2) the polynomial column indices combination tuple, which is
                a tuple of integers of variable length (min length is
                :param: min_degree, max length is :param: degree) with
                each integer falling in the range of [0, n_features_in_-1]
        B) Return a string that:
            1) is not a duplicate of any originally seen feature name
            2) is not a duplicate of any generated polynomial feature name
    equal_nan:
        bool, default = False -

        When comparing two columns for equality:

        If equal_nan is True, assume that a nan value would otherwise be
        the same as the compared non-nan counterpart, or if both compared
        values are nan, consider them as equal (contrary to the default
        numpy handling of nan, where numpy.nan != numpy.nan).
        If equal_nan is False and either one or both of the values in
        the compared pair of values is/are nan, consider the pair to be
        not equivalent, thus making the column pair not equal. This is
        in line with the normal numpy handling of nan values.

        When assessing if a column is constant:

        If equal_nan is True, assume any nan values are the mean of all
        non-nan values in the respective column. If equal_nan is False,
        any nan-values could never take the value of the mean of the
        non-nan values in the column, making the column not constant.
    rtol:
        numbers.Real, default = 1e-5 - The relative difference tolerance
        for equality. Must be a non-boolean, non-negative, real number.
        See numpy.allclose.
    atol:
        numbers.Real, default = 1e-8 - The absolute difference tolerance
        for equality. Must be a non-boolean, non-negative, real number.
        See numpy.allclose.
    n_jobs:
        Union[numbers.Integral, None], default = -1 - The number of
        joblib Parallel jobs to use when looking for duplicate columns
        or looking for columns of constants. The default is to use
        processes, but can be overridden externally using a joblib
        parallel_config context manager. The default number of jobs is
        -1 (all processors). To get maximum speed benefit, pybear
        recommends using the default setting.


    Attributes
    ----------
    n_features_in_:
        int - number of features in the fitted data, i.e., number of
        features before expansion.

    feature_names_in_:
        NDArray[object] - The names of the features as seen during
        fitting. Only accessible if X is passed to :methods: partial_fit
        or fit as a pandas dataframe that has a header.

    expansion_combinations_:
        tuple[tuple[int, ...], ...] - The polynomial column combinations
        from X that are in the polynomial expansion part of the final
        output. An example might be ((0,0), (0,1), ...), where each tuple
        holds the column indices from X that are multiplied to produce
        a feature in the polynomial expansion. This matches one-to-one
        with the created features, and similarly does not have any
        combinations that are excluded from the polynomial expansion for
        being a duplicate or constant.

    poly_constants_:
        dict[tuple[int, ...], any] - A dictionary whose keys are tuples
        of indices in the original data that produced a column of
        constants in the polynomial expansion. The dictionary values are
        the constant values in those columns. For example, if an
        expansion has a constant column that was produced by multiplying
        the second and third columns in X (index positions 1 and 2,
        respectively) and the constant value is 0, then constant_columns_
        entry will be {(1,2): 0}. If there are no constant columns, then
        constant_columns_ is an empty dictionary. These are always
        excluded from the polynomial expansion.

    poly_duplicates_:
        list[list[tuple[int, ...]]] - a list of the groups of identical
        polynomial features, indicated by tuples of their zero-based
        column index positions in the originally fit data. Columns from
        the original data itself can be in a group of duplicates, along
        with any duplicates from the polynomial expansion. For example,
        poly_duplicates_ for some dataset might look like:
        [[(1,), (2,3), (2,4)], [(5,6), (5,7), (6,7)]]

    dropped_poly_duplicates_:
        dict[tuple[int, ...], tuple[int, ...]] - A dictionary whose keys
        are the tuples that are removed from the polynomial expansion
        because they produced a duplicate of another column. The values
        of the dictionary are the tuples of indices of the respective
        duplicate that was kept.

    kept_poly_duplicates_:
        dict[tuple[int, ...], list[tuple[int, ...]]] - a dictionary whose
        keys are tuples of the indices of the columns of X that produced
        a polynomial column that was kept out of the sets of duplicates.
        The dictionary values are lists of the tuples of indices that
        created polynomial columns that were duplicate of the column
        indicated by the dictionary key, but were removed from the
        polynomial expansion.


    Notes
    -----
    pizza
    talk about how all pd nan-likes are converted to np.nan.
    talk about nan handling in general
    This is to allow for validation of datatypes with astype(np.float64) (pd nan-likes blow this up)
    and also to convert X to scipy sparse (pd nan-likes also blow this up!)

    v v v v taken directly from IM notes v v v v
    Concerning the handling of nan-like representations. While SPF
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
    ^ ^ ^ ^ taken directly from IM notes ^ ^ ^ ^


    See Also
    --------
    numpy.ndarray
    pandas.core.frame.DataFrame
    scipy.sparse
    numpy.allclose
    numpy.array_equal


    Examples
    --------
    >>> from pybear.preprocessing import SlimPolyFeatures as SPF
    >>> import numpy as np
    >>> trf = SPF(degree=2, min_degree=1, interaction_only=False)
    >>> X = [[0,1],[0,1],[1,1],[1,0]]
    >>> out = trf.fit_transform(X)
    >>> out
    pizza finish this!





    """

    # pizza this will ultimately come out at sklearn exorcism
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
        degree:Optional[numbers.Integral]=2,
        *,
        min_degree:Optional[numbers.Integral]=1,
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
        n_jobs: Optional[Union[numbers.Integral, None]] = None
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
        underscore, it can define a '__pybear_is_fitted__' method
        returning a boolean to specify if the estimator/transformer is
        fitted or not.

        """

        # must have this because there are no trailing-underscore attrs
        # generated by {partial_}fit(). all the trailing-underscore attrs
        # are accessed via @property.
        return hasattr(self, '_poly_duplicates')


    def _check_X_constants_and_dupls(self):

        """
        When SPF :param: scan_X is True, SPF uses pybear InterceptManager
        and pybear ColumnDeduplicateTransformer to scan X for constants
        and duplicates. If any are found, this method will raise an
        exception.

        """

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

        """
        Warning message for when duplicates and/or constants are found
        in X.

        """

        return (f"there are duplicate and/or constant columns in the data. "
            f"the attribute you have requested cannot be returned "
            f"because it is not accurate when the data has duplicate or "
            f"constant columns. this warning is raised and the program "
            f"not terminated to allow for more partial fits.")


    @property
    def expansion_combinations_(
        self
    ) -> Union[tuple[tuple[int, ...], ...], None]:

        check_is_fitted(self)

        try:
            self._check_X_constants_and_dupls()
            # self._active_combos must be sorted asc degree,
            # then asc on idxs. if _combos is sorted
            # then this is sorted correctly at construction.
            # if min_degree == 1, need to add original columns
            if self.min_degree == 1:
                _X_idxs = [(i,) for i in range(self.n_features_in_)]
                return tuple(_X_idxs + list(self._active_combos))
            else:
                return self._active_combos
        except:
            warnings.warn(self._attr_access_warning())
            return


    @property
    def poly_duplicates_(self) -> Union[list[list[tuple[int, ...]]], None]:

        # pizza keep these notes.
        # make some notes here that clarify how in-process _poly_duplicates
        # can have constants because they need to be tracked until the final
        # partial fit, but poly_duplicates_ cannot have the constants in it.
        # the builder functions at the end of partial_fit() and transform()
        # must use poly_duplicates_ and the user must get poly_duplicates_,
        # but _poly_duplicates must always have constants in it and must be
        # used for all other interal operations.

        check_is_fitted(self)

        try:
            self._check_X_constants_and_dupls()

            return _deconstant_poly_dupls(
                _poly_duplicates=self._poly_duplicates,
                _poly_constants=self._poly_constants
            )
        except:
            warnings.warn(self._attr_access_warning())
            return


    @property
    def kept_poly_duplicates_(
        self
    ) -> Union[dict[tuple[int, ...], list[tuple[int, ...]]], None]:

        check_is_fitted(self)

        try:
            self._check_X_constants_and_dupls()

            # must use poly_duplicates_ here not _poly_duplicates
            return _build_kept_poly_duplicates(
                    self.poly_duplicates_,
                    self._kept_combos
                )
        except:
            warnings.warn(self._attr_access_warning())
            return


    @property
    def dropped_poly_duplicates_(
        self
    ) -> Union[dict[tuple[int, ...], tuple[int, ...]], None]:


        check_is_fitted(self)

        try:

            self._check_X_constants_and_dupls()

            # must use poly_duplicates_ here not _poly_duplicates
            return _build_dropped_poly_duplicates(
                    self.poly_duplicates_,
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
        Resets the data-dependent state of the transformer. __init__
        parameters are not changed. This is part of the external API to
        allow for reset because setting most params after a partial fit
        has been done is blocked.

        """

        if hasattr(self, "_poly_duplicates"):
            del self._poly_duplicates
            del self._poly_constants
            del self._combos

            # _rand_combos, _kept_combos, and _active_combos may not exist
            # even if _poly_duplicates exists because
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
        Get feature names for the expanded feature set.


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
            expanded data after transformation.

        """

        # pizza, OneToOneFeatureMixin will become irrelevant after the exorcism
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

        # _X_header must exist as an actual header, and not None. if
        # self.features_names_in_ is None, the default boilerplate
        # feature names should have been generated for _X_header.
        assert _X_header is not None

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
        Get metadata routing is not implemented in SlimPolyFeatures.

        """
        __ = type(self).__name__
        raise NotImplementedError(
            f"get_metadata_routing is not implemented in {__}"
        )


    # pizza this will be redid at sklearn exorcism
    def get_params(self, deep:bool=True) -> dict[str, any]:
        # if ever needed, hard code that can be substituted for the
        # BaseEstimator get/set_params can be found in GSTCV_Mixin
        return super().get_params(deep)


    # pizza this decorator will come out, sklearn exorcism
    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(
        self,
        X: DataType,
        y: Union[Iterable[any], None]=None
    ) -> Self:

        """
        Incrementally train the SPF transformer instance on batches of
        data.


        Parameters
        ----------
        X:
            Union[numpy.ndarray, pandas.DataFrame, scipy.sparse] of shape
            (n_samples, n_features) - A batch of the dataset to undergo
            polynomial expansion.
        y:
            any - Always ignored. The target for the data.


        Return
        ------
        -
            self - the fitted SlimPolyFeatures instance.


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
            self._poly_duplicates: Union[list[list[tuple[int, ...]]], None] = None
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
                    f"There are duplicate and/or constant columns in X."
                )

        # END Identify constants & duplicates in X v^v^v^v^v^v^v^v^v^v^v^v^v^v^

        # _validation should have caught non-numeric X. X must only be numeric
        # throughout all of SPF.


        # build a ss csc that holds the unique polynomial columns that are
        # discovered. need to carry this to compare the next calculated
        # polynomial term against the known unique polynomial columns already
        # calculated for the expansion.
        _POLY_CSC = ss.csc_array(np.empty((X.shape[0], 0))).astype(np.float64)

        IDXS_IN_POLY_CSC: list[tuple[int, ...]] = []
        _poly_dupls_current_partial_fit: list[list[tuple[int, ...]]] = []
        _poly_constants_current_partial_fit: dict[tuple[int, ...], any] = {}



        # ss sparse that cant be sliced
        if isinstance(X,
            (ss.coo_matrix, ss.dia_matrix, ss.bsr_matrix, ss.coo_array,
             ss.dia_array, ss.bsr_array)
        ):
            warnings.warn(
                f"pybear works hard to avoid mutating or creating copies "
                f"of your original data. \nyou have passed your data as "
                f"{type(X)}, which cannot be sliced by columns. pybear "
                f"needs to create a copy. \nto avoid this, pass your "
                f"sparse data as csr, csc, lil, or dok."
            )
            _X = X.copy().tocsc()
        else:
            _X = X


        
        # GENERATE COMBINATIONS # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
        # need to get the permutations to run, based on the size of x,
        # min degree, max degree, and interaction_only.
        self._combos: list[tuple[int, ...]] = _combination_builder(
            n_features_in_=self.n_features_in_,
            _min_degree=self.min_degree,
            _max_degree=self.degree,
            _intx_only=self.interaction_only
        )
        # END GENERATE COMBINATIONS # v^v^v^v^v^v^v^v^v^v^v^v^v

        # need to iterate over the combos and find what is constant or duplicate
        for _combo in self._combos:

            # _combo must always be at least degree 2, degree 1 is just the
            # original data and should not be processed here
            assert len(_combo) >= 2

            _COLUMN: npt.NDArray[np.float64] = \
                _columns_getter(_X, _combo).prod(1).ravel()

            __ = _COLUMN.shape
            assert len(__) == 1 or (len(__)==2 and __[1]==1)
            del __

            # poly constants v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

            # if _poly_constants (the place that holds what poly columns are
            # constants across all partial fits) exists and it is empty, then
            # there cannot possibly be any columns of constants going forward,
            # so dont even expend the energy to check.
            if self._poly_constants and not len(self._poly_constants):
                _poly_is_constant = uuid.uuid4()
            else:
                _poly_is_constant: Union[uuid.UUID, any] = \
                    _parallel_constant_finder(
                        _column=_COLUMN,
                        _equal_nan = self.equal_nan,
                        _rtol = self.rtol,
                        _atol = self.atol
                    )

            if not isinstance(_poly_is_constant, uuid.UUID):
                _poly_constants_current_partial_fit[_combo] = _poly_is_constant

            # END poly constants v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

            # constant columns NEED TO GO INTO _POLY_CSC to know if
            # they are also a member of duplicates because even though
            # they are constants now, they might not be after more
            # partial fits, but they still might be duplicates.

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
            # [(i1,), (i2,),..] SINGLE 1D GROUP (list) OF DUPLICATES
            _indices = [(i,) for i in range(_X.shape[1])] + IDXS_IN_POLY_CSC
            _dupls_for_this_combo = []
            for _combo_tuple, _is_dupl in zip(_indices, _out):
                if _is_dupl:
                    _dupls_for_this_combo.append(_combo_tuple)
            if len(_dupls_for_this_combo):
                _dupls_for_this_combo.append(_combo)

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
            # put it in _POLY_CSC, even if it is constant
            if not len(_dupls_for_this_combo):
                _POLY_CSC = ss.hstack((
                    _POLY_CSC,
                    ss.csc_array(_COLUMN.reshape((-1,1)))
                ))
                IDXS_IN_POLY_CSC.append(_combo)

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
        # need to meld _poly_constants_current_partial_fit into
        # self.poly_constants_, which would be holding the constants
        # found in previous partial fits

        self._poly_constants: dict[tuple[int, ...], any] = \
            _merge_constants(
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
        self._poly_duplicates: list[list[tuple[int, ...]]] = \
            _merge_partialfit_dupls(
                self._poly_duplicates,
                _poly_dupls_current_partial_fit
            )

        del _poly_dupls_current_partial_fit

        # _merge_partialfit_dupls sorts _poly_duplicates on the way out
        # within dupl sets, sort on combo len (degree) asc, then sort asc
        # on combos, which should be sorted asc on idxs values
        # across all dupl sets, only look at the first value in a dupl
        # set, sort on len asc, then values asc
        # END poly duplicates -----------------------

        # iff self.poly_constants_ is None at this point it is because @property for it
        # is excepting on self._check_X_constants_and_dupls() and returning
        # None. In that case, all @properties will also trip on that and return None
        # for everything. partial_fit and transform
        # will continue to warn and the @properties will continue to warn as long as the
        # dupl and/or constants condition in X exists.
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

        # when doing partial fits, columns that are currently constant must be
        # tracked in self._poly_duplicates because in future partial fits
        # they may no longer be constant, but may still be duplicates. but this
        # creates a problem in that things that are constant in the present
        # state are mixed into self._poly_duplicates, which clouds the water
        # when building _rand_combos, _kept_combos, and _active_combos. need
        # a de-constanted _poly_duplicates for building these objects, while
        # preserving the state of _poly_duplicates.
        # all of this machinery is built into @property poly_duplicates_
        # ---------------------------------------------------------------------
        # BELOW THIS LINE USE poly_duplicates_ ONLY, DO NOT USE _poly_duplicates

        # if 'keep' == 'random', _transform() must pick the same random
        # duplicate columns at every call after fitting is completed. need
        # to set an instance attribute here
        # that doesnt change when _transform() is called. must set a
        # random idx for every set of dupls.
        self._rand_combos: tuple[tuple[int, ...], ...] = \
            _lock_in_random_combos(poly_duplicates_=self.poly_duplicates_)

        # this needs to be before _get_active_combos because _kept_combos
        # is an input into @property dropped_poly_duplicates_
        self._kept_combos: tuple[tuple[int, ...], ...] = \
            _identify_combos_to_keep(
                self.poly_duplicates_,
                self.keep,
                self._rand_combos
            )

        # @property dropped_poly_duplicates_ needs self._poly_duplicates
        # and self._kept_combos
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
        Train the SPF transformer instance on one batch of data. The
        internal state of the instance is reset with every call to this
        method.


        Parameters
        ----------
        X:
            Union[numpy.ndarray, pandas.DataFrame, scipy.sparse] of shape
            (n_samples, n_features) - The full dataset to undergo
            polynomial expansion.
        y:
            any - Always ignored. The target for the data.


        Return
        ------
        -
            self - the fitted SlimPolyFeatures instance.


        """

        self.reset()

        return self.partial_fit(X)


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


    def set_params(self, **params) -> Self:

        """
        Pizza, this jargon was taken directly from BaseEstimator 24_12_10_16_56_00

        Set the parameters of this estimator.

        This method works on simple estimators as well as on nested objects
        (such as :class: `sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update parameters of nested objects.

        ----------------------

        Once this transformer is fitted, only :params: 'sparse_output',
        'keep', 'feature_name_combiner', and 'n_jobs' can be changed via
        :method: set_params. All other parameters are blocked. To use
        different parameters without creating a new instance of this
        transformer class, call :method: reset on this instance,
        otherwise create a new instance of this transformer class."


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
                "can be changed via :method: set_params. \nAll other "
                "parameters are blocked. \nThe currently passed parameters "
                f"{', '.join(list(_invalid_params))} have been blocked, "
                "but any valid parameters that were passed have been set."
                "\nTo use different parameters without creating a new "
                "instance of this transformer class, call :method: reset "
                "on this instance, otherwise create a new instance of this "
                "transformer class."
            )

            super().set_params(**_valid_params)

        except:

            super().set_params(**params)

        return self


    def transform(
        self,
        X: DataType,
        copy: Union[bool, None]=None   # pizza, this is going to come out
    ) -> DataType:

        """
        Apply the expansion footprint that was learned during fitting to
        the given data. pybear strongly urges that only data that was
        seen during fitting be passed here.


        Parameters
        ----------
        X:
            Union[numpy.ndarray, pandas.DataFrame, scipy.sparse] of shape
            (n_samples, n_features) - The data to undergo polynomial
            expansion.
        copy:   # pizza this is going to come out
            Union[bool, None] -


        Return
        -------
        -
            X_tr: Union[numpy.ndarray, pandas.DataFrame, scipy.sparse] -
            the polynomial feature expansion for X.

        """

        check_is_fitted(self)

        # this does a no-op if there are dupls or constants in X
        # returns None with a warning, allowing for more partial fits
        try:
            self._check_X_constants_and_dupls()
        except:
            warnings.warn(self._attr_access_warning())
            return


        # pizza this is going to come out
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


        # ss sparse that cant be sliced
        if isinstance(X,
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
        # pd df with funky nan-likes that np and ss dont like
        elif self.min_degree == 1 and isinstance(X, pd.core.frame.DataFrame):
            try:
                X.astype(np.float64)
                # if excepts, there are pd nan-likes that arent recognized by numpy.
                # if passes, this df should just hstack with X_tr without a problem.
                _X = X
            except:
                warnings.warn(
                    f"pybear works hard to avoid mutating or creating copies of "
                    f"your original data. \nyou have passed a dataframe that has "
                    f"nan-like values that are not recognized by numpy/scipy. "
                    f"\nbut to merge this data with the polynomial expansion, "
                    f"pybear must make a copy to replace all the nan-likes with "
                    f"numpy.nan. \nto avoid this copy, pass your dataframe with "
                    f"numpy.nan in place of any nan-likes that are only recognized "
                    f"by pandas."
                )
                _X = X.copy()
                _X[nan_mask(_X)] = np.nan
        else:
            _X = X

        # ---------------------------------------------------------------------
        # BELOW THIS LINE USE poly_duplicates_ ONLY, DO NOT USE _poly_duplicates

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
                self.poly_duplicates_,
                self.keep,
                self._rand_combos
            )

        # @property dropped_poly_duplicates_ needs self._poly_duplicates and self._kept_combos
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



























