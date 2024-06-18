# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from copy import deepcopy
from typing import Union, Iterable, TypeAlias
from .._type_aliases import DataType, TotalCountsByColumnType

import numpy as np
import pandas as pd

from _make_row_and_column_masks import _make_row_and_column_masks





XType: TypeAlias = Union[Iterable[DataType], Iterable[Iterable[DataType]]]
YType: TypeAlias = Union[Iterable[DataType], Iterable[Iterable[DataType]]]


def transform(self, X, y=None):
    """
    Reduce X by the thresholding rules found during fit.

    Parameters
    ----------
    X : Union[numpy.ndarray, pandas.DataFrame, pandas.Series dask.array,
        dask.DataFrame, dask.Series] of shape (n_samples, n_features)}
        The data that is to be reduced according to the thresholding
        rules found during :term: fit.

    y : Union[numpy.ndarray, pandas.DataFrame, pandas.Series, dask.array,
        dask.DataFrame, dask.Series] of shape (n_samples, n_outputs),
        or (n_samples,), default=None - Target values (None for
        unsupervised transformations).

    Returns
    -------
    -
        X_tr : Union[numpy.ndarray, pandas.DataFrame, pandas.Series]
                of shape (n_samples_new, n_features_new)
                Transformed array.
        y_tr : Union[numpy.ndarray, pandas.DataFrame, pandas.Series]
                of shape (n_samples_new, n_outputs) or (n_samples_new,)
                Transformed target.

    """

    self._must_be_fitted()

    self._recursion_check()

    X, y, _columns = self._handle_X_y(X, y)
    # X & y ARE NOW np.array

    if _columns is None and hasattr(self, 'feature_names_in_'):
        # THIS INDICATES THAT DATA WAS FIT WITH A DF AT SOME POINT BUT
        # CURRENTLY PASSED DATA IS ARRAY
        pass
    elif _columns is None and not hasattr(self, 'feature_names_in_'):
        pass
    elif _columns is not None and hasattr(self, 'feature_names_in_'):
        _val_feature_names(_columns, self.feature_names_in_)
    elif _columns is not None and not hasattr(self, 'feature_names_in_'):
        # THIS INDICATES DATA WAS FIT WITH ARRAY BUT CURRENTLY PASSED IS DF
        pass

    if len(X.shape) == 1:
        _X_columns = 1
    elif len(X.shape) == 2:
        _X_columns = X.shape[1]
    if _X_columns != self.n_features_in_:
        raise ValueError(f"X has {_X_columns} columns, previously fit data "
                         f"had {self.n_features_in_} columns")

    del _X_columns

    # VALIDATE _ignore_columns & handle_as_bool; CONVERT TO IDXs ** ** ** **
    # _val_ignore_cols and _val_handle_as_bool INSIDE OF self._validate()
    # SKIPPED THE col_idx VALIDATE/CONVERT PART WHEN self.n_features_in_
    # DIDNT EXIST (I.E., UP UNTIL THE START OF FIRST FIT) BUT ON FIRST PASS
    # THRU HERE (AND EACH THEREAFTER) n_features_in_ (AND POSSIBLY
    # feature_names_in_) NOW EXISTS SO PERFORM THE VALIDATE TO CONVERT IDXS.
    # 24_03_18_16_42_00 MUST VALIDATE ON ALL PASSES NOW BECAUSE
    # _ignore_columns AND/OR _handle_as_bool CAN BE CALLABLE BASED ON X AND
    # X IS NEW EVERY TIME SO THE CALLABLES MUST BE RECOMPUTED AND
    # RE-VALIDATED BECAUSE THEY MAY (UNDESIRABLY) GENERATE DIFFERENT IDXS.
    # _ignore_columns MUST BE BEFORE _make_instructions

    if callable(self._ignore_columns):
        self._ignore_columns = self._ignore_columns(X)

    self._ignore_columns = _val_ignore_columns(
        self._ignore_columns,
        self._check_is_fitted(),
        self.n_features_in_,
        self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None
    )
    # _handle_as_bool MUST ALSO BE HERE OR WILL NOT CATCH  obj COLUMN
    if callable(self._handle_as_bool):
        self._handle_as_bool = self._handle_as_bool(X)

    self._handle_as_bool = _val_handle_as_bool(
        self._handle_as_bool,
        self._check_is_fitted(),
        self.n_features_in_,
        self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
        self._original_dtypes
    )
    # END VALIDATE _ignore_columns & handle_as_bool; CONVERT TO IDXs ** **

    _delete_instr = self._make_instructions()

    # BUILD ROW_KEEP & COLUMN_KEEP MASKS ** ** ** ** ** ** ** ** ** ** **

    ROW_KEEP_MASK, COLUMN_KEEP_MASK = \
        _make_row_and_column_masks(
            X,
            self._total_counts_by_column,
            _delete_instr,
            self._reject_unseen_values,
            self._n_jobs
        )

    # END BUILD ROW_KEEP & COLUMN_KEEP MASKS ** ** ** ** ** ** ** ** ** **

    self._row_support = ROW_KEEP_MASK.copy()

    FEATURE_NAMES = self.get_feature_names_out()

    if False not in ROW_KEEP_MASK and False not in COLUMN_KEEP_MASK:
        # skip
        pass
    else:
        X = X[ROW_KEEP_MASK, :]
        X = X[:, COLUMN_KEEP_MASK]

        if y is not None:
            y = y[ROW_KEEP_MASK]

        if self._max_recursions == 1:
            pass
        else:
            if self._max_recursions < 1:
                raise ValueError(f"max_recursions has fallen below 1")

            # NEED TO TRANSFORM _ignore_columns AND _handle_as_bool FROM
            # WHAT THEY WERE FOR self TO WHAT THEY ARE FOR THE CURRENT
            # (POTENTIALLY COLUMN MASKED) DATA GOING INTO THIS RECURSION
            if callable(self.ignore_columns):
                NEW_IGN_COL = self.ignore_columns
            else:
                IGN_COL_MASK = np.zeros(self.n_features_in_).astype(bool)
                IGN_COL_MASK[self._ignore_columns.astype(np.uint32)] = True
                NEW_IGN_COL = np.arange(sum(COLUMN_KEEP_MASK))[
                    IGN_COL_MASK[COLUMN_KEEP_MASK]
                ]
                del IGN_COL_MASK

            if callable(self.handle_as_bool):
                NEW_HDL_AS_BOOL_COL = self.handle_as_bool
            else:
                HDL_AS_BOOL_MASK = np.zeros(self.n_features_in_).astype(bool)
                HDL_AS_BOOL_MASK[self._handle_as_bool.astype(np.uint32)] = True
                NEW_HDL_AS_BOOL_COL = np.arange(sum(COLUMN_KEEP_MASK))[
                    HDL_AS_BOOL_MASK[COLUMN_KEEP_MASK]]
                del HDL_AS_BOOL_MASK

            RecursiveCls = MinCountTransformer(
                self.count_threshold,
                ignore_float_columns=self.ignore_float_columns,
                ignore_non_binary_integer_columns=
                self.ignore_non_binary_integer_columns,
                ignore_columns=NEW_IGN_COL,
                ignore_nan=self.ignore_nan,
                handle_as_bool=NEW_HDL_AS_BOOL_COL,
                delete_axis_0=self.delete_axis_0,
                max_recursions=self.max_recursions - 1,
                n_jobs=self._n_jobs
            )

            del NEW_IGN_COL, NEW_HDL_AS_BOOL_COL

            RecursiveCls.set_output(transform=None)

            # IF WAS PASSED WITH HEADER, REAPPLY TO DATA FOR RE-ENTRY
            if hasattr(self, 'feature_names_in_'):
                X = pd.DataFrame(data=X,
                                 columns=self.feature_names_in_[
                                     COLUMN_KEEP_MASK]
                                 )

            if y is not None:
                X, y = RecursiveCls.fit_transform(X, y)
            else:
                X = RecursiveCls.fit_transform(X)

            FEATURE_NAMES = RecursiveCls.get_feature_names_out(None)

            # ITERATE OVER RecursiveCls._total_counts_by_column AND COMPARE
            # THE COUNTS FROM self._total_counts_by_column, AND IF
            # RecursiveCls's VALUE IS LOWER PUT IT INTO self's.
            # _total_counts_by_column ONLY HOLDS ACTIVE COLUMNS (NOT IGNORED),
            # GIVES A DICT OF UNQ & CTS FOR COLUMN MUST MAP LOCATIONS IN
            # RecursiveCls._total_counts_by_column TO THEIR OLD LOCATIONS
            # IN self._total_counts_by_column

            MAP_DICT = dict((
                zip(np.arange(RecursiveCls.n_features_in_),
                    self.get_support(True))
            ))
            TCBC = deepcopy(self._total_counts_by_column)
            _ = RecursiveCls._total_counts_by_column
            for new_col_idx in _:
                old_col_idx = MAP_DICT[new_col_idx]
                for unq, ct in _[new_col_idx].items():
                    if unq not in TCBC[old_col_idx]:
                        if str(unq).lower() not in np.char.lower(
                                np.fromiter(TCBC[old_col_idx].keys(),
                                            dtype='<U100')
                        ):
                            raise AssertionError(f"algorithm failure, unique "
                                                 f"in a deeper recursion is not in the "
                                                 f"previous recursion")
                    try:
                        __ = TCBC[old_col_idx][unq]
                    except:
                        try:
                            __ = TCBC[old_col_idx][str(unq).lower()]
                        except:
                            try:
                                __ = dict((zip(list(
                                    map(str, TCBC[old_col_idx].keys())),
                                               list(
                                                   TCBC[old_col_idx].values()))
                                           ))['nan']
                            except:
                                raise ValueError(f"could not access key {unq} "
                                                 f"in _total_counts_by_column")
                    assert ct <= __, (f"algorithm failure, count of a unique "
                                      f"in a deeper recursion is > the count "
                                      f"of the same unique in a higher "
                                      f"recursion, can only be <=")
                    if ct < __:
                        self._total_counts_by_column[old_col_idx][unq] = ct

            self._row_support[self._row_support] = RecursiveCls._row_support

            del RecursiveCls, MAP_DICT, TCBC, _, new_col_idx, \
                old_col_idx, unq, ct, __

        del ROW_KEEP_MASK, COLUMN_KEEP_MASK

    # EVERYTHING WAS PROCESSED AS np.array
    __ = self._output_transform or self._x_original_obj_type
    if True in [j in __ for j in ['dataframe', 'series']]:
        X = pd.DataFrame(X, columns=FEATURE_NAMES)
        if 'series' in __:
            if self.n_features_in_ > 1:
                raise ValueError(
                    f"cannot return as Series when n_features_in_ is > 1.")
            X = X.squeeze()

    del FEATURE_NAMES

    __ = self._output_transform
    if y is not None:
        __ = __ or self._y_original_obj_type
        if True in [j in __ for j in ['dataframe', 'series']]:
            y = pd.DataFrame(y, columns=[f"y{k}" for k in
                                         range(y.shape[1] if len(
                                             y.shape) == 2 else 1)])
            if 'series' in __:
                if y.shape[1] > 1:
                    raise ValueError(
                        f"cannot return y as Series when y is multi-class.")
                y = y.squeeze()

        return X, y
    else:
        return X











