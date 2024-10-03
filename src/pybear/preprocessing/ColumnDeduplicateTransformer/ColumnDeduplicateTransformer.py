# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from sklearn.base import BaseEstimator, TransformerMixin # pizza OneToOneFeatureMixin

from typing import Iterable, Literal, Optional
from typing_extensions import Union, Self
import numpy.typing as npt

import numpy as np
import pandas as pd

from ._validation._validation import _validation
from ._partial_fit._dupl_idxs import _dupl_idxs
from ._partial_fit._identify_idxs_to_delete import _identify_idxs_to_delete

# pizza
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    _check_sample_weight,
    check_is_fitted,
    check_random_state,
)




class ColumnDeduplicateTransformer(BaseEstimator, TransformerMixin):


    def __init__(
        self,
        *,
        keep: Union[Literal['first'], Literal['last'], Literal['random']] = 'first',
        do_not_drop: Optional[Union[Iterable[str], Iterable[int], None]] = None,
        conflict: Optional[Union[Literal['raise'], Literal['ignore']]] = 'raise',
        columns: Optional[Union[Iterable[str], None]]=None,
        n_jobs: Optional[Union[int, None]]=None
    ) -> None:


        """
        pizza gibberish!

        Parameters
        ----------
        keep:
            Union[Literal['first'], Literal['last'], Literal['random']],
            default = None -
            The strategy for keeping a single representative from a set of
            identical columns. 'first' retains the column left-most in the
            data; 'last' keeps the column right-most in the data; 'random'
            keeps a single random column of the set of duplicates.
        do_not_drop:
            Union[Iterable[str], Iterable[int], None], default=None -
            Columns to never drop, overriding the positional 'keep' argument
            for the set of duplicates associated with the indicated column.
            If a conflict arises, such as two columns specified in
            'do_not_drop' are duplicates of each other, an error is raised.
        conflict:
            Union[Literal['raise'], Literal['ignore']] - Only matters when
            columns are passed to do_not_drop.
        columns:
            Union[Iterable[str], None], default=None - Externally supplied
            column names. If X is a dataframe, passing columns here will
            override those in the dataframe header; otherwise, if this is
            None, the dataframe header is retained.
        n_jobs:
            pizza finish

        Attributes:
        -----------
        n_features_in_: int - number of features in the data before deduplication.

        feature_names_in_: Union[NDarray[str], None] -

        duplicates_: list[list[int]] -

        removed_columns_: list[int] -

        """

        self._keep = keep
        self._do_not_drop = do_not_drop
        self._conflict = conflict
        self._columns = columns
        self._n_jobs = n_jobs


    def _reset(self):
        """
        Reset internal data-dependent state of the transformer.
        __init__ parameters are not touched.
        """
        pass


    def get_feature_names_out(self):

        """Get output feature names for the deduplication."""

        # get_feature_names_out() would otherwise be provided by
        # OneToOneFeatureMixin, but since this transformer deletes columns,
        # must build a one-off.

        # 24_10_02_12_49_00 pizza, finish this. the mask most likely will
        # be built from the object that indicates the duplicates.
        COLUMN_MASK = pizza


        return self._columns[COLUMN_MASK]


    # def get_params!!! pizza dont forget about this!


    # def set_params!!! pizza dont forget about this!


    def partial_fit(
        self,
        X: Union[npt.NDArray, pd.core.frame.DataFrame],
        y: Union[any, None]=None
    ) -> Self:

        """
        pizza gibberish


        Parameters
        ----------
        X:
            {array-like, sparse matrix} of shape (n_samples, n_features) -
            Data to remove duplicate columns from.
        y:
            {vector-like, None}, default = None - ignored.


        Return
        ------
        -
            self - the fitted ColumnDeduplicateTransformer instande.


        """

        # PIZZA! validation of X must be done here! not in a separate module!
        # BaseEstimator has _validate_data method, which when called exposes
        # n_features_in_ and feature_names_in!
        # first_call = not hasattr(self, "n_samples_seen_")
        # X = self._validate_data(
        #     X,
        #     accept_sparse=("csr", "csc"),
        #     dtype=FLOAT_DTYPES,
        #     force_all_finite="allow-nan",
        #     reset=first_call,
        # )

        _validation(X, self._keep, self._do_not_drop, self._conflict, self._columns, self._n_jobs)

        # if do_not_drop is strings, convert to idxs
        if isinstance(self._do_not_drop[0], str):
            self._do_not_drop = \
                [self._columns.index(col) for col in self._do_not_drop]

        # find the duplicate columns
        if hasattr(self, 'duplicates_'):
            self.duplicates_ = _dupl_idxs(X, self.duplicates_, self._n_jobs)
        else:
            self.duplicates_ = _dupl_idxs(X, None, self._n_jobs)

        # determine the columns to remove based on given parameters
        self.removed_columns = _identify_idxs_to_delete()

        return self


    def fit(
        self,
        X: Union[npt.NDArray, pd.core.frame.DataFrame],
        y: Union[any, None]=None
    ) -> Self:

        """
        pizza gibberish


        Parameters
        ----------
        X:
            {array-like, sparse matrix} of shape (n_samples, n_features) -
            Data to remove duplicate columns from.
        y:
            {vector-like, None}, default = None - ignored.


        Return
        ------
        -
            self - the fitted ColumnDeduplicateTransformer instande.


        """

        self._reset()
        return self.partial_fit(X, y=y)



    def transform(
        self,
        X
    ) -> #pizza figure out if always returning as ndarray or as given:


        check_is_fitted(self)


        """
        pizza gibberish


        Parameters
        ----------
        X:
            {array-like, sparse matrix} of shape (n_samples, n_features) -
            Data to remove duplicate columns from.


        Return
        ------
        -
            X: {array-like, sparse matrix} - Deduplicated data; data with
                duplicate columns removed based on the given parameters.


        """

        DELETE = []
        for group in GROUPS:
            if keep == 'left':

        X = np.delete(X_encoded_np, col_idx2, axis=0)
        COLUMNS = np.delete(COLUMNS, col_idx2, axis=0)
























