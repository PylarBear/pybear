# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pandas import SparseDtype
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import StrOptions

from typing import Iterable, Literal, Optional
from typing_extensions import Union, Self
import numpy.typing as npt
from ._type_aliases import SparseTypes
import numbers
import numpy as np
import pandas as pd

from ._validation._validation import _validation
from ._validation._X import _val_X
from ._partial_fit._dupl_idxs import _dupl_idxs
from ._partial_fit._identify_idxs_to_delete import _identify_idxs_to_delete
from ._transform._transform import _transform
from ._inverse_transform._inverse_transform import _inverse_transform

from sklearn.utils.validation import check_is_fitted, check_array




class ColumnDeduplicateTransformer(BaseEstimator, TransformerMixin):

    """
    pizza gibberish!

    pizza, talk about nan handling. at time of column comparison,
    both compared columns of data are converted to numpy array for the
    comparison. for numeric columns, whatever is used to indicate
    'not-a-number' must be recognizable by numpy as such, so that
    numpy can convert it to np.nan and handle it as such. for string
    data, CDT will recognize strings of the form 'nan' (not case sensitive.)



    Parameters
    ----------
    keep:
        Literal['first', 'last', 'random'], default = None -
        The strategy for keeping a single representative from a set
        of identical columns. 'first' retains the column left-most
        in the data; 'last' keeps the column right-most in the data;
        'random' keeps a single random column of the set of
        duplicates.
    do_not_drop:
        Union[Iterable[str], Iterable[int], None], default=None -
        Columns to never drop, overriding the positional 'keep'
        argument for the set of duplicates associated with the
        indicated column. If a conflict arises, such as two columns
        specified in 'do_not_drop' are duplicates of each other,
        the behavior is managed by :param: 'conflict'.
    conflict:
        Union[Literal['raise', 'ignore'] - Ignored when do_not_drop
        is not passed. Pizza say more words.
    columns:
        Union[Iterable[str], None], default=None - Externally
        supplied column names. If X is a dataframe, passing columns
        here will override those in the dataframe header; otherwise,
        if this is None, the dataframe header is retained.
    rtol:
        float, default = 1e-5 - The relative tolerance parameter.
            See numpy.allclose.
    atol:
        float, default = 1e-8 - The absolute tolerance parameter.
            See numpy.allclose.
    equal_nan:
        bool, default = False - When comparing pairs of columns row
        by row:
        If equal_nan is True, exclude from comparison any rows where
        one or both of the values is/are nan. If one value is nan,
        this essentially assumes that the nan value would otherwise
        be the same as its non-nan counterpart. When both are nan,
        this considers the nans as equal (contrary to the default
        numpy handling of nan, where np.nan != np.nan) and will not
        in and of itself cause a pair of columns to be marked as
        unequal.
        If equal_nan is False and either one or both of the values
        in the compared pair of values is/are nan, consider the pair
        to be not equivalent, thus making the column pair not equal.
        This is in line with the normal numpy handling of nan values.
    n_jobs:
        Union[int, None], default = None - The number of joblib
        Parallel jobs to use when comparing columns. The default is
        to use processes, but can be overridden externally using a
        joblib parallel_config context manager. The default number
        of jobs is -1. To get maximum speed benefit, pybear recommends
        using the default setting.    PIZZA disseminate!!!


    Attributes:
    -----------
    n_features_in_:
        int - number of features in the data before deduplication.

    feature_names_in_:
        Union[NDarray[str], None] - Only accessible if X is passed
            to :methods: partial_fit or fit as pandas dataframe.

    duplicates_: list[list[int]] - pizza

    removed_columns_: dict[int, int] - pizza

    column_mask_: NDArray[int] - pizza


    See Also
    --------
    numpy.ndarray
    pandas.core.frame.DataFrame
    scipy.sparse
    numpy.allclose (for numeric data)
    numpy.array_equal (for string data)

    """

    _parameter_constraints: dict = {
        "keep": [StrOptions({"first", "last", "random"})],
        "do_not_drop": [list, tuple, set, None,],   # pizza what about empty
        "conflict": [StrOptions({"raise", "ignore"})],
        "columns": [list, tuple, set, np.ndarray, None],
        "rtol": [numbers.Real],
        "atol": [numbers.Real],
        "equal_nan": ["boolean"],
        "n_jobs": [numbers.Integral, None],
    }

    def __init__(
        self,
        *,
        keep: Union[Literal['first'], Literal['last'], Literal['random']] = 'first',
        do_not_drop: Optional[Union[Iterable[str], Iterable[int], None]] = None,
        conflict: Optional[Literal['raise', 'ignore']] = 'raise',
        columns: Optional[Union[Iterable[str], None]] = None,
        rtol: Optional[float] = 1e-5,
        atol: Optional[float] = 1e-8,
        equal_nan: Optional[bool] = False,
        n_jobs: Optional[Union[int, None]] = None
    ) -> None:


        self.keep = keep
        self.do_not_drop = do_not_drop
        self.conflict = conflict
        self.columns = columns
        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan
        self.n_jobs = n_jobs


    def _reset(self):
        """
        Reset internal data-dependent state of the transformer.
        __init__ parameters are not touched.

        """

        if hasattr(self, "duplicates_"):
            del self.duplicates_
            del self.removed_columns_
            del self.column_mask_


    def get_feature_names_out(self):

        """Get output feature names after deduplication."""

        # get_feature_names_out() would otherwise be provided by
        # OneToOneFeatureMixin, but since this transformer deletes
        # columns, must build a one-off.

        if hasattr(self, 'feature_names_in_'):
            return self.feature_names_in_[self.column_mask_]
        else:
            return


    # def get_params!!! pizza dont forget about this! ESPECIALLY TEST!


    # def set_params!!! pizza dont forget about this! ESPECIALLY TEST!
    def set_params(self, **params):

        if 'columns' in params and not \
                np.array_equal(params['columns'], self.columns):
            raise ValueError(f"'columns' cannot be changed once instantiated.")

        super().set_params(**params)

    # def set_output!!! pizza this is being tested as of 24_10_13


    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(
        self,
        X: Union[npt.NDArray[any], pd.DataFrame, SparseTypes],
        y: any=None
    ) -> Self:

        """
        pizza gibberish


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - Data to remove duplicate columns from.
        y:
            {vector-like, None}, default = None - ignored.


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
        )


        """
        PIZZA
        
        reset – Whether to reset the n_features_in_ attribute. If False, 
        the input will be checked for consistency with data provided when 
        reset was last True.
        It is recommended to call reset=True in fit and in the first call 
        to partial_fit. All other methods that validate X should set 
        reset=False.
        
        cast_to_ndarray – Cast X and y to ndarray with checks in 
        check_params. If False, X and y are unchanged and only 
        feature_names_in_ and n_features_in_ are checked.
        
        """

        # do this before _validation so that self._columns
        # is correctly assigned
        if self.columns is None:
            if hasattr(self, 'feature_names_in_'):
                self._columns = self.feature_names_in_
            else:
                self._columns = None
        else:
            self._columns = list(map(str, np.array(list(self.columns)).ravel()))
            if hasattr(self, 'feature_names_in_') and not \
                np.array_equal(self.feature_names_in_, self._columns):
                raise ValueError(
                    f"conflict between column names passed via dataframe and "
                    f"the 'columns' kwarg."
                )

            self._columns = self.columns


        _validation(
            X,  # Not validated, used for validation of other objects
            self._columns,
            self.conflict,
            self.do_not_drop,
            self.keep,
            self.rtol,
            self.atol,
            self.equal_nan,
            self.n_jobs
        )

        if self._columns is not None:
            self.feature_names_in_ = np.array(self._columns, dtype=str)


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

        _validation(
            X,  # Not validated, used for validation of other objects
            self._columns,
            self.conflict,
            self.do_not_drop,
            self.keep,
            self.rtol,
            self.atol,
            self.equal_nan,
            self.n_jobs
        )


        # determine the columns to remove based on given parameters.
        # pizza this makes removed_columns_ and column_mask_ available after
        # (partial_)fit, but what if set_params() is used before transform?
        # redo these operations at the top of transform()
        self.removed_columns_ = \
            _identify_idxs_to_delete(
                self.duplicates_,
                self.keep,
                self.do_not_drop,
                self._columns,
                self.conflict
            )

        self.column_mask_ = np.ones(self.n_features_in_).astype(bool)
        self.column_mask_[list(self.removed_columns_)] = False

        return self


    def fit(
        self,
        X: Union[npt.NDArray[any], pd.DataFrame, SparseTypes],
        y: any=None
    ) -> Self:

        """
        pizza gibberish


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - Data to remove duplicate columns from.
        y:
            {vector-like, None}, default = None - ignored.


        Return
        ------
        -
            self - the fitted ColumnDeduplicateTransformer instance.


        """

        self._reset()
        return self.partial_fit(X, y=y)


    def transform(
        self,
        X: Union[npt.NDArray[any], pd.DataFrame, SparseDtype],
        *,
        copy: bool = None
    ) -> Union[npt.NDArray[any], pd.DataFrame, SparseDtype]:

        """
        Remove the duplicate columns from X.


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - The data to be deduplicated.
        copy:
            Union[bool, None], default=None - Whether or not to make a
            copy of X before the transform.


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
            ensure_min_features=2
        )


        _validation(
            X,  # Not validated, used for validation of other objects
            self._columns,
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
                self._columns,
                self.conflict
            )

        self.column_mask_ = np.ones(self.n_features_in_).astype(bool)
        self.column_mask_[list(self.removed_columns_)] = False
        # end redo

        return _transform(X, self.column_mask_)


    def inverse_transform(
        self,
        X: Union[npt.NDArray[any], pd.DataFrame, SparseDtype],
        *,
        copy: bool = None
        ) -> Union[npt.NDArray[any], pd.DataFrame, SparseDtype]:

        """
        Revert transformed data back to the original.


        Parameters
        ----------
        X : {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - A deduplicated data set.
        copy:
            Union[bool, None], default=None - Whether or not to make a
            copy of X before the inverse transform.


        Returns
        -------
        X_tr : {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - Deduplicated data reverted to original data.

        """

        check_is_fitted(self)

        if not isinstance(copy, (bool, type(None))):
            raise TypeError(f"'copy' must be boolean or None")

        # pizza, may want to put _val_X here

        copy = copy if copy is not None else self.copy
        X = check_array(
            array=X,
            accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
            dtype=None,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=copy or False
        )

        return _inverse_transform(
            X,
            self.duplicates_,
            self.removed_columns_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None
        )























