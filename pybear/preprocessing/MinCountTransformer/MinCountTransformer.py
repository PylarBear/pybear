# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from copy import deepcopy
from typing import Union, Iterable, TypeAlias
from ._type_aliases import OriginalDtypesDtype, TotalCountsByColumnType, \
    IgnColsHandleAsBoolDtype, DataType
import numpy as np
import pandas as pd
import joblib
from preprocessing.MinCountTransformer.docs.mincounttransformer_docs import \
    mincounttransformer_docs
from sklearn.exceptions import NotFittedError
from sklearn.base import check_array

from ._shared._validation._val_ignore_columns import _val_ignore_columns
from ._shared._validation._val_handle_as_bool import _val_handle_as_bool
from ._shared._validation._val_ign_cols_hab_callable import \
    _val_ign_cols_hab_callable
from ._validation._val_feature_names import _val_feature_names
from ._validation._mct_validation import _mct_validation

from ._shared._make_instructions._make_instructions import _make_instructions
from ._handle_X_y import _handle_X_y
from ._base_fit._parallel_dtypes_unqs_cts import _dtype_unqs_cts_processing
from ._test_threshold import _test_threshold
from ._transform._make_row_and_column_masks import _make_row_and_column_masks
from ._transform._tcbc_update import _tcbc_update


XType: TypeAlias = Iterable[Iterable[DataType]]
YType: TypeAlias = Union[Iterable[Iterable[DataType]], Iterable[DataType]]


class MinCountTransformer:

    __doc__ = mincounttransformer_docs.__doc__

    # pizza
    # def __doc__(self):
    #     return mincounttransformer_docs.__doc__


    _original_dtypes: OriginalDtypesDtype
    _total_counts_by_column: TotalCountsByColumnType


    def __init__(
                 self,
                 count_threshold:int,
                 *,
                 ignore_float_columns:bool=True,
                 ignore_non_binary_integer_columns:bool=True,
                 ignore_columns:IgnColsHandleAsBoolDtype=None,
                 ignore_nan:bool=True,
                 handle_as_bool:IgnColsHandleAsBoolDtype=None,
                 delete_axis_0:bool=False,
                 reject_unseen_values:bool=False,
                 max_recursions:int=1,
                 n_jobs:[int, None]=None
                 ):


        #pizza
        __doc__ = mincounttransformer_docs.__doc__

        self.count_threshold = count_threshold
        self.ignore_float_columns = ignore_float_columns
        self.ignore_non_binary_integer_columns = ignore_non_binary_integer_columns
        self.ignore_columns = ignore_columns
        self.ignore_nan = ignore_nan
        self.handle_as_bool = handle_as_bool
        self.delete_axis_0 = delete_axis_0
        self.n_jobs = n_jobs
        self.reject_unseen_values = reject_unseen_values
        self.max_recursions = max_recursions



    @property
    def original_dtypes_(self):
        return self._original_dtypes


    @original_dtypes_.setter
    def original_dtypes_(self, value):
        raise AttributeError(f'original_dtypes_ attribute is read-only')


    def _reset(self):
        """Reset the internal data state of MinCountTransformer."""

        if not hasattr(self, '_output_transform'):
            self._output_transform = None

        self._x_original_obj_type = None
        self._y_original_obj_type = None

        self._total_counts_by_column = {}

        self._recursion_block = self._max_recursions > 1

        try: del self.n_features_in_
        except: pass

        try: del self.feature_names_in_
        except: pass

        try: del self._n_rows_in
        except: pass

        try: del self._original_dtypes
        except: pass


    def _base_fit(self, X, y=None, **fit_kwargs):
        """
        Shared uniques and counts collection process for partial_fit() &
        fit().

        """

        X, y, _columns = self._handle_X_y(X, y)

        # GET _X_rows, _X_columns, n_features_in_, feature_names_in_, _n_rows_in_

        if _columns is None and hasattr(self, 'feature_names_in_'):
            # WAS FIT WITH A DF AT SOME POINT BUT CURRENTLY PASSED DATA
            # IS ARRAY
            pass
        elif _columns is None and not hasattr(self, 'feature_names_in_'):
            # DATA WITH COLUMNS HAS NEVER BEEN PASSED
            pass
        elif _columns is not None and hasattr(self, 'feature_names_in_'):
            # CURRENT DATA HAS COLUMNS AND FIT HAS SEEN COLUMNS PREVIOUSLY
            _val_feature_names(_columns, self.feature_names_in_)
        elif _columns is not None and not hasattr(self, 'feature_names_in_'):
            # FIRST PASS OR FIT WITH ARRAYS UP TO THIS POINT BUT CURRENTLY
            # PASSED IS DF
            self.feature_names_in_ = _columns

        _X_rows, _X_columns = X.shape

        # IF PREVIOUSLY FITTED, THEN self.n_features_in_ EXISTS
        if hasattr(self, 'n_features_in_') and _X_columns != self.n_features_in_:
            raise ValueError( f"X has {_X_columns} columns, previously "
                f"seen data had {self.n_features_in_} columns")
        else: # IF NOT PREVIOUSLY FITTED
            self.n_features_in_ = _X_columns

        try:
            # WAS PREVIOUSLY FITTED, THEN self._n_rows_in EXISTS
            self._n_rows_in += _X_rows
        except:
            self._n_rows_in = _X_rows

        del _X_columns
        # END GET _X_rows, _X_columns, n_features_in_, feature_names_in_, _n_rows_in_


        # GET TYPES, UNQS, & CTS FOR ACTIVE COLUMNS ** ** ** ** ** ** **

        # DONT HARD-CODE backend, ALLOW A CONTEXT MANAGER TO SET
        DTYPE_UNQS_CTS_TUPLES = \
            joblib.Parallel(return_as='list', n_jobs=self._n_jobs)(
                joblib.delayed(_dtype_unqs_cts_processing)(
                    X[:,_idx],
                    _idx,
                    self._ignore_float_columns,
                    self._ignore_non_binary_integer_columns
                    ) for _idx in range(self.n_features_in_)
            )

        _col_dtypes = np.empty(self.n_features_in_, dtype='<U8')
        # DOING THIS for LOOP 2X TO KEEP DTYPE CHECK SEPARATE AND PRIOR TO
        # MODIFYING self._total_counts_by_column, PREVENTS INVALID DATA FROM
        # INVALIDATING ANY VALID UNQS/CT IN THE INSTANCE'S
        # self._total_counts_by_column
        for col_idx, (_dtype, UNQ_CT_DICT) in enumerate(DTYPE_UNQS_CTS_TUPLES):
            _col_dtypes[col_idx] = _dtype

        if not hasattr(self, '_original_dtypes'):
            self._original_dtypes = _col_dtypes
        else:
            if not np.array_equiv(_col_dtypes, self._original_dtypes):
                raise TypeError(f"datatypes in most recently passed data do not "
                                f"match original dtypes")

        del _col_dtypes

        # _handle_as_bool MUST ALSO BE HERE OR WILL NOT CATCH obj COLUMN
        self._ignore_columns = _val_ignore_columns(
            self._ignore_columns,
            self._check_is_fitted(),
            self.n_features_in_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None
        )

        self._handle_as_bool = _val_handle_as_bool(
            self._handle_as_bool,
            self._check_is_fitted(),
            self.n_features_in_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            self._original_dtypes
        )

        for col_idx, (_dtype, UNQ_CT_DICT) in enumerate(DTYPE_UNQS_CTS_TUPLES):

            if col_idx not in self._total_counts_by_column:
                self._total_counts_by_column[col_idx] = UNQ_CT_DICT
            else:
                for unq, ct in UNQ_CT_DICT.items():
                    if unq in self._total_counts_by_column[col_idx]:
                        self._total_counts_by_column[col_idx][unq] += ct
                    else:
                        self._total_counts_by_column[col_idx][unq] = ct

        del DTYPE_UNQS_CTS_TUPLES, col_idx, _dtype, UNQ_CT_DICT

        # END GET TYPES, UNQS, & CTS FOR ACTIVE COLUMNS ** ** ** ** ** *

        return self


    def fit(self, X, y=None, **fit_kwargs):
        """Determine the uniques and their frequencies to be used for
        later thresholding.

        Parameters
        ----------
        X : {array-like, of shape (n_samples, n_features) The data used
            to determine the uniques and their frequencies, used for later
            thresholding along the feature axis.

        y : array-like of shape (n_samples, n_output) or (n_samples,),
            default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        self : object
            Fitted min count transformer.


        """

        self._validate()
        self._reset()

        self._recursion_check()

        return self._base_fit(X, y, **fit_kwargs)


    def partial_fit(self, X, y=None, **fit_kwargs):
        """Online accrual of uniques and counts for later thresholding.

        All of X is processed as a single batch. This is intended for
        cases when :meth:`fit` is not feasible due to very large number
        of `n_samples` or because X is read from a continuous stream.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to determine the uniques and their frequencies,
            used for later thresholding along the feature axis.

        y : array-like of shape (n_samples, n_output) or (n_samples,),
            default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        self : object
            Fitted min count transformer.
        """

        self._validate()

        if not self._check_is_fitted():
            self._reset()

        self._recursion_check()

        return self._base_fit(X, y, **fit_kwargs)


    def transform(
            self,
            X: XType,
            y: YType=None
        ) -> Union[tuple[XType, YType], XType]:

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
            # WAS FIT WITH A DF AT SOME POINT BUT CURRENTLY PASSED DATA
            # IS ARRAY
            pass
        elif _columns is None and not hasattr(self, 'feature_names_in_'):
            # DATA WITH COLUMNS HAS NEVER BEEN PASSED
            pass
        elif _columns is not None and hasattr(self, 'feature_names_in_'):
            # CURRENT DATA HAS COLUMNS AND FIT HAS SEEN COLUMNS PREVIOUSLY
            _val_feature_names(_columns, self.feature_names_in_)
        elif _columns is not None and not hasattr(self, 'feature_names_in_'):
            # FIRST PASS OR FIT WITH ARRAYS UP TO THIS POINT BUT CURRENTLY
            # PASSED IS DF
            self.feature_names_in_ = _columns


        if len(X.shape)==1:
            _X_columns = 1
        elif len(X.shape)==2:
            _X_columns = X.shape[1]
        if _X_columns != self.n_features_in_:
            raise ValueError(f"X has {_X_columns} columns, previously fit data "
                             f"had {self.n_features_in_} columns")

        del _X_columns


        # VALIDATE _ignore_columns & handle_as_bool; CONVERT TO IDXs **
        # _val_ignore_cols and _val_handle_as_bool INSIDE OF
        # self._validate() SKIPPED THE col_idx VALIDATE/CONVERT PART
        # WHEN self.n_features_in_ DIDNT EXIST (I.E., UP UNTIL THE START
        # OF FIRST FIT) BUT ON FIRST PASS THRU HERE (AND EACH THEREAFTER)
        # n_features_in_ (AND POSSIBLY feature_names_in_) NOW EXISTS SO
        # PERFORM VALIDATION TO CONVERT IDXS. 24_03_18_16_42_00 MUST
        # VALIDATE ON ALL PASSES NOW BECAUSE _ignore_columns AND/OR
        # _handle_as_bool CAN BE CALLABLE BASED ON X AND X IS NEW EVERY
        # TIME SO THE CALLABLES MUST BE RECOMPUTED AND RE-VALIDATED
        # BECAUSE THEY MAY (UNDESIRABLY) GENERATE DIFFERENT IDXS.
        # _ignore_columns MUST BE BEFORE _make_instructions

        if callable(self._ignore_columns):
            try:
                self._ignore_columns = self._ignore_columns(X)
            except Exception as e:
                raise Exception(f"ignore_columns callable excepted with "
                                f"this error --- {e}")

            _val_ign_cols_hab_callable(self._ignore_columns, 'ignore_columns')

        self._ignore_columns = _val_ignore_columns(
            self._ignore_columns,
            self._check_is_fitted(),
            self.n_features_in_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None
        )

        # _handle_as_bool MUST ALSO BE HERE OR WILL NOT CATCH obj COLUMN
        if callable(self._handle_as_bool):
            try:
                self._handle_as_bool = self._handle_as_bool(X)
            except Exception as e:
                raise Exception(f"handle_as_bool callable excepted with "
                                f"this error --- {e}")

            _val_ign_cols_hab_callable(self._handle_as_bool, 'handle_as_bool')

        self._handle_as_bool = _val_handle_as_bool(
            self._handle_as_bool,
            self._check_is_fitted(),
            self.n_features_in_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            self._original_dtypes
        )
        # END VALIDATE _ignore_columns & handle_as_bool; CONVERT TO IDXs ** **


        _delete_instr = self._make_instructions()

        # BUILD ROW_KEEP & COLUMN_KEEP MASKS ** ** ** ** ** ** ** ** **

        ROW_KEEP_MASK, COLUMN_KEEP_MASK = \
            _make_row_and_column_masks(
                X,
                self._total_counts_by_column,
                _delete_instr,
                self._reject_unseen_values,
                self._n_jobs
            )

        # END BUILD ROW_KEEP & COLUMN_KEEP MASKS ** ** ** ** ** ** ** **

        self._row_support = ROW_KEEP_MASK.copy()

        FEATURE_NAMES = self.get_feature_names_out()

        if all(ROW_KEEP_MASK) and all(COLUMN_KEEP_MASK):
            # Skip all recursion code and go directly to output formatting.
            # if all rows and all columns are kept, the data has converged
            # to a point where all the values in every column appear in
            # their respective column at least count_threshold times.
            # There is no point to performing any (more) (possible)
            # recursions.
            pass
        else:
            X = X[ROW_KEEP_MASK, :]
            X = X[:, COLUMN_KEEP_MASK]

            if y is not None:
                y = y[ROW_KEEP_MASK]

            if self._max_recursions == 1:
                # if reached last recursion, skip to output formatting
                pass
            else:
                assert self._max_recursions >= 1, f"max_recursions < 1"

                # NEED TO RE-ALIGN _ignore_columns AND _handle_as_bool
                # FROM WHAT THEY WERE FOR self TO WHAT THEY ARE FOR THE
                # CURRENT (POTENTIALLY COLUMN MASKED) DATA GOING INTO
                # THIS RECURSION
                if callable(self.ignore_columns):
                    NEW_IGN_COL = self.ignore_columns # pass the function!
                else:
                    IGN_COL_MASK = np.zeros(self.n_features_in_).astype(bool)
                    IGN_COL_MASK[self._ignore_columns.astype(np.uint32)] = True
                    NEW_IGN_COL = np.arange(sum(COLUMN_KEEP_MASK))[
                                            IGN_COL_MASK[COLUMN_KEEP_MASK]
                    ]
                    del IGN_COL_MASK

                if callable(self.handle_as_bool):
                    NEW_HDL_AS_BOOL_COL = self.handle_as_bool # pass the function!
                else:
                    HDL_AS_BOOL_MASK = np.zeros(self.n_features_in_).astype(bool)
                    HDL_AS_BOOL_MASK[self._handle_as_bool.astype(np.uint32)] = True
                    NEW_HDL_AS_BOOL_COL = np.arange(sum(COLUMN_KEEP_MASK))[
                                            HDL_AS_BOOL_MASK[COLUMN_KEEP_MASK]]
                    del HDL_AS_BOOL_MASK
                # END RE-ALIGN _ignore_columns AND _handle_as_bool ** * ** * **

                RecursiveCls = MinCountTransformer(
                    self.count_threshold,
                    ignore_float_columns=self.ignore_float_columns,
                    ignore_non_binary_integer_columns=
                        self.ignore_non_binary_integer_columns,
                    ignore_columns=NEW_IGN_COL,
                    ignore_nan=self.ignore_nan,
                    handle_as_bool=NEW_HDL_AS_BOOL_COL,
                    delete_axis_0=self.delete_axis_0,
                    max_recursions=self.max_recursions-1,
                    n_jobs=self._n_jobs
                )

                del NEW_IGN_COL, NEW_HDL_AS_BOOL_COL

                RecursiveCls.set_output(transform=None)

                # IF WAS PASSED WITH HEADER, REAPPLY TO DATA FOR RE-ENTRY
                if hasattr(self, 'feature_names_in_'):
                    X = pd.DataFrame(data=X,
                             columns=self.feature_names_in_[COLUMN_KEEP_MASK]
                    )

                if y is not None:
                    X, y = RecursiveCls.fit_transform(X, y)
                else:
                    X = RecursiveCls.fit_transform(X)

                FEATURE_NAMES = RecursiveCls.get_feature_names_out(None)

                MAP_DICT = dict((
                    zip(np.arange(RecursiveCls.n_features_in_),
                        self.get_support(True))
                ))

                # vvv tcbc update vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

                self._total_counts_by_column = \
                    _tcbc_update(
                        deepcopy(self._total_counts_by_column),
                        RecursiveCls._total_counts_by_column,
                        MAP_DICT
                )

                # ^^^ tcbc update ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                self._row_support[self._row_support] = RecursiveCls._row_support

                del RecursiveCls, MAP_DICT

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
                                 range(y.shape[1] if len(y.shape)==2 else 1)])
                if 'series' in __:
                    if y.shape[1] > 1:
                        raise ValueError(
                            f"cannot return y as Series when y is multi-class.")
                    y = y.squeeze()

            return X, y
        else:
            return X


    def fit_transform(self, X, y=None, **fit_kwargs):
        """
        Fits MinCountTransformer to X and returns a transformed version
        of X.

        Parameters
        ----------
        X : {ndarray, pandas.DataFrame, pandas.Series } of shape
            (n_samples, n_features)} - The data used to determine the
            uniques and their frequencies and to be transformed by rules
            created from those frequencies.

        y : {ndarray, pandas.DataFrame, pandas.Series} of shape
            (n_samples,) or (n_samples, n_outputs), default=None - Target
            values (None for unsupervised transformations).

        Returns
        -------
        X_tr : {ndarray, pandas.DataFrame, pandas.Series} of shape
                (n_samples_new, n_features_new)
            Transformed data.

        y_tr : {ndarray, pandas.DataFrame, pandas.Series} of shape
                (n_samples_new,) or (n_samples_new, n_outputs)
            Transformed target.
        """

        self._validate()
        self._reset()

        self._base_fit(X, y)

        _original_recursion_block = self._recursion_block
        self._recursion_block = False
        __ = self.transform(X, y)
        self._recursion_block = _original_recursion_block
        del _original_recursion_block

        return __


    def inverse_transform(self, X_tr):
        """
        Reverse the transformation operation. This operation cannot
        restore removed examples, only features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features_new) - The input
            samples.

        Returns
        -------
        -
            X_inv : ndarray of shape (n_samples, n_original_features)
                X with columns of zeros inserted where features would
                have been removed by transform.

        """

        self._must_be_fitted()

        X_tr = self._handle_X_y(X_tr, None)[0]

        # MOCK X WITH np.zeros, check_array WONT TAKE STRS
        check_array(np.zeros(X_tr.shape))

        __ = self.get_support(False)

        if X_tr.shape[1] != sum(__):
            raise ValueError(f"X has a different shape than during fitting.")

        X_inv = np.zeros((X_tr.shape[0], self.n_features_in_), dtype=object)
        X_inv[:, __] = X_tr

        del __

        return X_inv


    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None - Input
            features. If input_features is None, then feature_names_in_
            is used as feature names in. If feature_names_in_ is not
            defined, then the following input feature names are generated:
                ["x0", "x1", ..., "x(n_features_in_ - 1)"].
            If input_features is an array-like, then input_features must
            match feature_names_in_ if feature_names_in_ is defined.

        Returns
        -------
        -
            feature_names_out : ndarray of str objects - Transformed
                feature names.

        """

        self._must_be_fitted()

        COLUMN_MASK = self.get_support(indices=False)

        err_msg = f"input_features must be a list-type of strings or None"
        if input_features is None:
            if hasattr(self, 'feature_names_in_'):
                return self.feature_names_in_[COLUMN_MASK]
            else:
                return np.array(
                    [f"x{i}" for i in range(self.n_features_in_)]
                )[COLUMN_MASK]
        else:
            if isinstance(input_features, (str, dict)):
                raise TypeError(err_msg)

            try:
                input_features = np.array(list(input_features))
            except:
                raise TypeError(err_msg)

            if False in ['str' in str(type(__)).lower() for __ in input_features]:
                raise TypeError(err_msg)
            elif len(np.array(input_features).ravel()) != self.n_features_in_:
                raise ValueError(f"number of passed input_features does not "
                     f"match number of features seen during (partial_)fit().")
            elif hasattr(self, 'feature_names_in_') and \
                not np.array_equiv(input_features, self.feature_names_in_):
                    raise ValueError(f"passed input_features does not match "
                                f"feature names seen during (partial_)fit().")
            else:
                return np.array(input_features).ravel()[COLUMN_MASK]


    def get_metadata_routing(self):
        """Get metadata routing of this object - Not implemented."""
        __ = type(self).__name__
        raise NotImplementedError(f"get_metadata_routing is not available in {__}")


    def get_params(self, deep=True):
        """Get parameters for this transformer.

        Parameters
        ----------
        deep : bool, default=True - Ignored.

        Returns
        -------
        params : dict - Parameter names mapped to their values.
        """

        params = {
                    'count_threshold': self.count_threshold,
                    'ignore_float_columns': self.ignore_float_columns,
                    'ignore_non_binary_integer_columns':
                        self.ignore_non_binary_integer_columns,
                    'ignore_columns': self.ignore_columns,
                    'ignore_nan': self.ignore_nan,
                    'delete_axis_0': self.delete_axis_0,
                    'max_recursions': self.max_recursions,
                    'n_jobs': self.n_jobs
        }

        return params


    def get_row_support(self, indices:bool=False):
        """Get a mask, or integer index, of the rows selected.

        Parameters
        ----------
        indices : bool, default=False - If True, the return value will
            be an array of integers, rather than a boolean mask.

        Returns
        -------
        support : ndarray - A slicer that selects the retained rows from
            the X most recently seen by transform. If indices is False,
            this is a boolean array of shape (n_input_features, ) in
            which an element is True if its corresponding row is selected
            for retention. If indices is True, this is an integer array
            of shape (n_output_features, ) whose values are indices into
            the input feature vector.
        """

        self._must_be_fitted()

        if not hasattr(self, '_row_support'):
            raise AttributeError(f"get_row_support() can only be accessed after "
                                 f"some data has been transformed")

        if indices is False:
            return self._row_support
        elif indices is True:
            return np.arange(len(self._row_support))[self._row_support]


    def get_support(self, indices:bool=False):
        """Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        indices : bool, default=False - If True, the return value will
            be an array of integers rather than a boolean mask.

        Returns
        -------
        support : ndarray - An index that selects the retained features
            from a feature vector. If indices is False, this is a boolean
            array of shape (n_input_features,) in which an element is
            True if its corresponding feature is selected for retention.
            If indices is True, this is an integer array of shape
            (n_output_features, ) whose values are indices into the input
            feature vector.
        """

        self._must_be_fitted()

        if callable(self._ignore_columns) or callable(self._handle_as_bool):
            raise ValueError(f"if ignore_columns or handle_as_bool is callable, "
                f"get_support() is only available after a transform is done.")

        COLUMNS = np.array(
            ['DELETE COLUMN' not in v for k, v in self._make_instructions().items()]
        )

        if indices is False:
            return np.array(COLUMNS)
        elif indices is True:
            return np.arange(self.n_features_in_)[COLUMNS].astype(np.uint32)


    def set_output(self, transform=None):
        """
        Set the output container when "transform" and "fit_transform"
        are called.

        Parameters
        ----------
        transform : {“default”, "numpy_array", “pandas_dataframe”,
            "pandas_series"},
            default = None - Configure output of transform and fit_transform.
            "default": Default output format of a transformer (numpy array)
            "numpy_array": np.ndarray output
            "pandas_dataframe": DataFrame output
            "pandas_series": Series output
            None: Transform configuration is unchanged

        Returns
        -------
        self : this instance - MinCountTransformer instance.

        """

        from ._validation._val_transform import _val_transform

        self._output_transform = _val_transform(transform)

        return self


    def set_params(self, **params):
        """Set the parameters of this transformer.

        Valid parameter keys can be listed with get_params(). Note that
        you can directly set the parameters of MinCountTransformer.

        Parameters
        ----------
        params : dict - MinCountTransformer parameters.

        Returns
        -------
        self : MinCountTransformer - This instance.
        """

        ALLOWED = ['count_threshold', 'ignore_float_columns',
                   'ignore_non_binary_integer_columns', 'ignore_columns',
                    'ignore_nan', 'handle_as_bool', 'delete_axis_0',
                   'reject_unseen_values', 'max_recursions', 'n_jobs']

        for _kwarg in params:
            if _kwarg not in ALLOWED:
                raise ValueError(
                    f"unknown param '{_kwarg}' passed to set_params()")

        del ALLOWED

        # MAKE SOME PARAMETERS UNCHANGEABLE ONCE AN INSTANCE IS FITTED
        if self._check_is_fitted() and self._max_recursions > 1:
            # IF CHANGING PARAMS WHEN max_recursions WAS >1, RESET THE
            # INSTANCE, BLOWING AWAY INTERNAL STATES ASSOCIATED WITH PRIOR
            # FITS, WITH EXCEPTION FOR n_jobs & reject_unseen_values
            # (r_u_v IS IRRELEVANT WHEN >1 RCRS BECAUSE ONLY fit_transform())
            _PARAMS = \
                [_ for _ in params if _ not in ('n_jobs','reject_unseen_values')]
            if len(_PARAMS) > 0:
                self._reset()
            del _PARAMS


        if 'count_threshold' in params: self.count_threshold = \
            params['count_threshold']
        if 'ignore_float_columns' in params: self.ignore_float_columns = \
            params['ignore_float_columns']
        if 'ignore_non_binary_integer_columns' in params:
            self.ignore_non_binary_integer_columns = \
                params['ignore_non_binary_integer_columns']
        if 'ignore_columns' in params: self.ignore_columns = \
            params['ignore_columns']
        if 'ignore_nan' in params: self.ignore_nan = params['ignore_nan']
        if 'handle_as_bool' in params: self.handle_as_bool = \
            params['handle_as_bool']
        if 'delete_axis_0' in params:
            self.delete_axis_0 = params['delete_axis_0']
        if 'reject_unseen_values' in params: self.reject_unseen_values = \
            params['reject_unseen_values']
        if 'max_recursions' in params: self.max_recursions = \
            params['max_recursions']
        if 'n_jobs' in params: self.n_jobs = params['n_jobs']

        self._validate()

        return self


    def _make_instructions(self, _threshold=None):

        self._must_be_fitted()  # must be before _make_instructions()

        return _make_instructions(
            self._count_threshold,
            self._ignore_float_columns,
            self._ignore_non_binary_integer_columns,
            self._ignore_columns,
            self._ignore_nan,
            self._handle_as_bool,
            self._delete_axis_0,
            self._original_dtypes,
            self.n_features_in_,
            self._total_counts_by_column,
            _threshold=_threshold
        )



    def test_threshold(
            self,
            threshold:int=None,
            clean_printout:bool=True
        ) -> None:

        """
        Display instructions generated for the current fitted state,
        subject to the passed threshold and the current settings of other
        parameters. The printout will indicate what rows / columns will
        be deleted, and if all columns or all rows will be deleted.

        If the instance has multiple recursions, the results are displayed
        as a single set of instructions, as if to perform the cumulative
        effect of the recursions in a single step.


        Parameters
        ----------
        threshold : int, default=None - count_threshold value to test.

        clean_printout: bool, default=True - Truncate printout to fit on
        screen

        Return
        ------
        None

        """

        self._must_be_fitted()
        if callable(self._ignore_columns) or callable(self._handle_as_bool):
            raise ValueError(f"if ignore_columns or handle_as_bool is "
                f"callable, get_support() is only available after a "
                f"transform is done.")

        _test_threshold(self, _threshold=threshold, _clean_printout=clean_printout)



    def _handle_X_y(self, X, y):

        """
        Validate dimensions of X and y and standardize the containers
        for processing.

        Parameters
        ----------
        X : {ndarray, pandas.DataFrame, pandas.Series, dask.array,
                dask.DataFrame, dask.Series} - data object

        y : {ndarray, pandas.DataFrame, pandas.Series, dask.array,
                dask.DataFrame, dask.Series} - target object


        Returns
        ----------
        X : ndarray - The given data as ndarray.
        y : ndarray - The given target as ndarray.
        _columns : ndarray - Feature names extracted from X.

        """

        # 24_03_03_09_54_00 THE INTENT IS TO RUN EVERYTHING AS NP ARRAY
        # TO STANDARDIZE INDEXING. IF DFS ARE PASSED, COLUMNS CAN
        # OPTIONALLY BE PULLED OFF AND RETAINED.


        # 24_06_17 When dask_ml Incremental and ParallelPostFit (versions
        # 2024.4.4 and 2023.5.27 at least) are passed y = None, they are
        # putting y = ('', order[i]) into the dask graph for y and sending
        # that junk as the value of y to the wrapped partial fit method.
        # All use cases where y=None are like this, it will always happen
        # and there is no way around it. To get around this, diagnose if
        # MCT is wrapped with Incr and/or PPF, then look to see if y is
        # of the form tuple(str, int). If both are the case, override y
        # with y = None.

        _is_garbage_y_from_dask_ml = False
        if isinstance(y, tuple) and isinstance(y[0], str) and isinstance(y[1], int):
            _is_garbage_y_from_dask_ml = True

        if self._using_dask_ml_wrapper and _is_garbage_y_from_dask_ml:
            y = None

        del _is_garbage_y_from_dask_ml
        # END accommodate dask_ml junk y ** * ** * ** * ** * ** * ** * *


        _columns = None

        # out is (X, y, _x_original_obj_type, _y_original_obj_type, columns)
        out = _handle_X_y(
            X,
            y,
            type(self).__name__,
            self._x_original_obj_type,
            self._y_original_obj_type
        )

        self._x_original_obj_type = out[2]
        self._y_original_obj_type = out[3]

        _dask_objects = ['', '', '']
        _non_dask_objects = ['', '', '']

        del _dask_objects, _non_dask_objects

        return (out[0], out[1], out[4])


    def _validate(self):
        """
        Validate MinCountTransformer arg and kwargs.

        """

        self._wrapped_by_dask_ml()

        _n_features_in = None
        _feature_names_in = None
        _original_dtypes = None
        _mct_has_been_fit = self._check_is_fitted()
        if _mct_has_been_fit:
            _n_features_in = self.n_features_in_
            if hasattr(self, 'feature_names_in_'):
                _feature_names_in = self.feature_names_in_
            _original_dtypes = self._original_dtypes


        self._count_threshold, self._ignore_float_columns, \
        self._ignore_non_binary_integer_columns, self._ignore_nan, \
        self._delete_axis_0, self._ignore_columns, self._handle_as_bool, \
        self._reject_unseen_values, self._max_recursions, self._n_jobs = \
            _mct_validation(
                self.count_threshold,
                self.ignore_float_columns,
                self.ignore_non_binary_integer_columns,
                self.ignore_nan,
                self.delete_axis_0,
                self.ignore_columns,
                self.handle_as_bool,
                self.reject_unseen_values,
                self.max_recursions,
                self.n_jobs,
                _mct_has_been_fit=_mct_has_been_fit,
                _n_features_in=_n_features_in,
                _feature_names_in=_feature_names_in,
                _original_dtypes=_original_dtypes
        )

        # extra count_threshold val
        if hasattr(self, '_n_rows_in') and self._count_threshold >= self._n_rows_in:
            raise ValueError(f"count_threshold is >= the number of rows, "
                             f"every column not ignored would be deleted.")



    def _check_is_fitted(self):
        """Check to see if the instance has been fitted."""
        return hasattr(self, '_total_counts_by_column')


    def _must_be_fitted(self):
        """Allows access only if the instance has been fitted."""
        if not self._check_is_fitted():
            raise NotFittedError(f"This instance has not been fitted yet. "
                         f"Fit data with partial_fit() or fit() first.")
        else:
            return True


    def _recursion_check(self):
        """Raise exception if attempting to use recursion with partial_fit,
        fit, and transform. Only allow fit_transform.
        """

        if self._recursion_block:
            raise ValueError(f"partial_fit(), fit(), and transform() are "
            f"not available if max_recursions > 1. fit_transform() only.")


    def _wrapped_by_dask_ml(self) -> None:
        """
        Determine if the MCT instance is wrapped by dask_ml Incremental
        or ParallelPostFit by checking the stack and looking for the
        '_partial' method used by those modules. Wrapping with these
        modules imposes limitations on passing a value to y.
        Pizza finish.

        """


        import inspect

        self._using_dask_ml_wrapper = False
        for frame_info in inspect.stack():
            _module = inspect.getmodule(frame_info.frame)
            if _module:
                if _module.__name__ == 'dask_ml._partial':
                    self._using_dask_ml_wrapper = True
                    break
        del _module


    def score(self):
        """
        Dummy method to spoof dask Incremental and ParallelPostFit
        wrappers. Verified must be here for dask wrappers.
        Pizza verify this!
        """

        pass






































