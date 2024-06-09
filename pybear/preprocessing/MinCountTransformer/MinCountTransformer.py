# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from copy import deepcopy
from typing import Union, TypeAlias
from ._type_aliases import OriginalDtypesDtype, TotalCountsByColumnType
import numpy as np
import pandas as pd
import joblib
from preprocessing.docs.mincounttransformer_docs import mincounttransformer_docs
from sklearn.exceptions import NotFittedError
from sklearn.base import check_array

from ._shared._make_instructions import _make_instructions




class MinCountTransformer:

    __doc__ = mincounttransformer_docs.__doc__

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
                 ignore_columns:[list[Union[int, str]], callable, None]=None,
                 ignore_nan:bool=True,
                 handle_as_bool:[list[Union[str, int]], callable, None]=None,
                 delete_axis_0:bool=False,
                 reject_unseen_values:bool=False,
                 max_recursions:int=1,
                 n_jobs:[int, None]=None
                 ):

        # pizza make sure the init typing is correct

        __doc__ = mincounttransformer_docs.__doc__

        #pizza
        #self.__init__.__doc__ =

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


    def _base_fit(self, X, y=None):
        """Shared uniques and counts collection process for partial_fit() &
        fit()."""

        X, y, _columns = self._handle_X_y(X, y, return_columns=True)

        # GET _X_rows, _X_columns, n_features_in_, feature_names_in_, _n_rows_in_

        if _columns is None and hasattr(self, 'feature_names_in_'):
            # THIS INDICATES THAT DATA WAS FIT WITH A DF AT SOME POINT BUT
            # CURRENTLY PASSED DATA IS ARRAY
            pass
        elif _columns is None and not hasattr(self, 'feature_names_in_'):
            pass
        elif _columns is not None and hasattr(self, 'feature_names_in_'):
            self._validate_feature_names(_columns, self.feature_names_in_)
        elif _columns is not None and not hasattr(self, 'feature_names_in_'):
            self.feature_names_in_ = _columns


        _X_rows, _X_columns = X.shape

        # IF PREVIOUSLY FITTED, THEN self.n_features_in_ EXISTS
        if hasattr(self, 'n_features_in_') and _X_columns != self.n_features_in_:
            raise ValueError( f"X has {_X_columns} columns, previously seen data "
                              f"had {self.n_features_in_} columns")
        else: # IF NOT PREVIOUSLY FITTED
            self.n_features_in_ = _X_columns

        try:
            # WAS PREVIOUSLY FITTED, THEN self._n_rows_in EXISTS
            self._n_rows_in += _X_rows
        except:
            self._n_rows_in = _X_rows

        del _X_columns
        # END GET _X_rows, _X_columns, n_features_in_, feature_names_in_, _n_rows_in_


        # GET TYPES, UNQS, & CTS FOR ACTIVE COLUMNS ** ** ** ** ** ** ** ** **
        @joblib.wrap_non_picklable_objects
        def _dtype_unqs_cts_processing(_column_of_X, col_idx, ignore_float_columns,
                                       ignore_non_binary_integer_columns):

            UNQ_CT_DICT: TypeAlias = dict[Union[float, str], int]

            # 24_03_23_11_28_00 SOMETIMES np.nan IS SHOWING UP MULTIPLE TIMES IN
            # UNIQUES. TROUBLESHOOTING HAS SHOWN THAT THE CONDITION THAT CAUSES
            # THIS IS WHEN dtype(_column_of_X) == object. CONVERT TO np.float64
            # IF POSSIBLE, OTHERWISE GET UNIQUES AS STR

            _column_orig_dtype = _column_of_X.dtype
            try:
                UNQ_CT_DICT = dict((zip(*np.unique(_column_of_X.astype(np.float64),
                                            return_counts=True))))
            except:
                UNQ_CT_DICT = dict((zip(*np.unique(_column_of_X.astype(str),
                                            return_counts=True))))

            UNQ_CT_DICT = dict((
                                zip(
                                    np.fromiter(UNQ_CT_DICT.keys(),
                                                    dtype=_column_orig_dtype),
                                    UNQ_CT_DICT.values()
                                )
            ))

            UNIQUES = np.fromiter(UNQ_CT_DICT.keys(), dtype=_column_orig_dtype)
            del _column_orig_dtype

            try:
                # 24_03_10_15_14_00 IF np.nan IS IN, IT EXISTS AS A str('nan')
                # IN INT AND STR COLUMNS
                UNIQUES = UNIQUES.astype(np.float64)
                # float64 RAISES ValueError ON STR DTYPES, IN THAT CASE MUST BE STR
                UNIQUES_NO_NAN = UNIQUES[np.logical_not(np.isnan(UNIQUES))]
                if np.allclose(UNIQUES_NO_NAN,
                               UNIQUES_NO_NAN.astype(np.int32), atol=1e-6):
                    if len(UNIQUES_NO_NAN) > 2 and \
                        ignore_non_binary_integer_columns:
                        UNQ_CT_DICT = {}
                    return 'int', UNQ_CT_DICT
                else:
                    if ignore_float_columns:
                        UNQ_CT_DICT = {}
                    return 'float', UNQ_CT_DICT
            except ValueError:
                # float64 RAISES ValueError ON STR DTYPES, IN THAT CASE MUST BE STR
                try:
                    UNIQUES.astype(str)
                    return 'obj', UNQ_CT_DICT
                except:
                    raise TypeError(f"Unknown datatype '{UNIQUES.dtype}' in "
                                    f"column index {col_idx}")
            except:
                raise Exception(f"Removing nans from column index {col_idx} "
                                f"excepted for reason other than ValueError")


        ARGS_PKG = tuple((self._ignore_float_columns,
                          self._ignore_non_binary_integer_columns))
        # DONT HARD-CODE backend, ALLOW A CONTEXT MANAGER TO SET
        DTYPE_UNQS_CTS_TUPLES = joblib.Parallel(return_as='list', n_jobs=self._n_jobs)(
            joblib.delayed(_dtype_unqs_cts_processing)(
                X[:,_idx], _idx, *ARGS_PKG) for _idx in range(self.n_features_in_))

        del _dtype_unqs_cts_processing, ARGS_PKG


        _col_dtypes = np.empty(self.n_features_in_, dtype='<U8')
        # DOING THIS for LOOP 2X TO KEEP DTYPE CHECK SEPARATE AND PRIOR TO MODIFYING
        # self._total_counts_by_column PREVENTS INVALID DATA FROM INVALIDATING
        # ANY VALID UNQS/CT IN THE INSTANCE'S self._total_counts_by_column
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
        self._ignore_columns = self._val_ign_cols_handle_as_bool(
                                        self._ignore_columns, 'ignore_columns')
        self._handle_as_bool = self._val_ign_cols_handle_as_bool(
                                        self._handle_as_bool, 'handle_as_bool')

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

        # END GET TYPES, UNQS, & CTS FOR ACTIVE COLUMNS ** ** ** ** ** ** ** **

        return self


    def fit(self, X, y=None):
        """Determine the uniques and their frequencies to be used for later
         thresholding.

        Parameters
        ----------
        X : {array-like, # PIZZA sparse matrix} of shape (n_samples, n_features)
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
        self._reset()

        self._recursion_check()

        return self._base_fit(X, y)


    def partial_fit(self, X, y=None):
        """Online accrual of uniques and counts for later thresholding.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

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

        return self._base_fit(X, y)


    def transform(self, X, y=None):
        """
        Reduce X by the thresholding rules found during fit.

        Parameters
        ----------
        X : {ndarray, pandas.DataFrame, pandas.Series # PIZZA sparse matrix} of
            shape (n_samples, n_features)}
            The data that is to be reduced according to the thresholding rules
            found during :term: fit.

        y : {ndarray, pandas.DataFrame, pandas.Series} of shape (n_samples,) or
            (n_samples, n_outputs),
            default=None - Target values (None for unsupervised transformations).

        Returns
        -------
        X_tr : {ndarray, pandas.DataFrame, pandas.Series PIZZA sparse matrix} of
                shape (n_samples_new, n_features_new)
            Transformed array.
        y_tr : {ndarray, pandas.DataFrame, pandas.Series} of shape
                (n_samples_new,) or (n_samples_new, n_outputs)
            Transformed target.
        """
        self._must_be_fitted()

        self._recursion_check()

        X, y, _columns = self._handle_X_y(X, y, return_columns=True)
        # X & y ARE NOW np.array

        if _columns is None and hasattr(self, 'feature_names_in_'):
            # THIS INDICATES THAT DATA WAS FIT WITH A DF AT SOME POINT BUT
            # CURRENTLY PASSED DATA IS ARRAY
            pass
        elif _columns is None and not hasattr(self, 'feature_names_in_'):
            pass
        elif _columns is not None and hasattr(self, 'feature_names_in_'):
            self._validate_feature_names(_columns, self.feature_names_in_)
        elif _columns is not None and not hasattr(self, 'feature_names_in_'):
            # THIS INDICATES DATA WAS FIT WITH ARRAY BUT CURRENTLY PASSED IS DF
            pass

        if len(X.shape)==1:
            _X_columns = 1
        elif len(X.shape)==2:
            _X_columns = X.shape[1]
        if _X_columns != self.n_features_in_:
            raise ValueError(f"X has {_X_columns} columns, previously fit data "
                             f"had {self.n_features_in_} columns")

        del _X_columns


        # VALIDATE _ignore_columns & handle_as_bool; CONVERT TO IDXs ** ** ** **
        # self._val_ign_cols_handle_as_bool INSIDE OF self._validate() SKIPPED
        # THE col_idx VALIDATE/CONVERT PART WHEN self.n_features_in_ DIDNT EXIST
        # (I.E., UP UNTIL THE START OF FIRST FIT) BUT ON FIRST PASS THRU HERE
        # (AND EACH THEREAFTER) n_features_in_ (AND POSSIBLY feature_names_in_)
        # NOW EXISTS SO PERFORM THE VALIDATE/CONVERT IDXS. 24_03_18_16_42_00
        # MUST VALIDATE ON ALL PASSES NOW BECAUSE _ignore_columns AND/OR
        # _handle_as_bool CAN BE CALLABLE BASED ON X AND X IS NEW EVERY TIME SO
        # AND THE CALLABLES MUST BE RECOMPUTED AND RE-VALIDATED BECAUSE THEY
        # MAY (UNDESIRABLY) GENERATE DIFFERENT IDXS. _ignore_columns MUST BE
        # BEFORE _make_instructions

        if callable(self._ignore_columns):
            self._ignore_columns = self._ignore_columns(X)
        self._ignore_columns = self._val_ign_cols_handle_as_bool(
                                        self._ignore_columns, 'ignore_columns'
        )
        # _handle_as_bool MUST ALSO BE HERE OR WILL NOT CATCH  obj COLUMN
        if callable(self._handle_as_bool):
            self._handle_as_bool = self._handle_as_bool(X)
        self._handle_as_bool = self._val_ign_cols_handle_as_bool(
                                    self._handle_as_bool, 'handle_as_bool'
        )
        # END VALIDATE _ignore_columns & handle_as_bool; CONVERT TO IDXs ** **





        # BUILD ROW_KEEP & COLUMN_KEEP MASKS ** ** ** ** ** ** ** ** ** ** **

        _delete_instr = self._make_instructions()

        self._validate_delete_instr(_delete_instr)

        def np_unique_handler(X):
            orig_dtype = X.dtype
            try:
                return np.unique(X.astype(np.float64)).astype(orig_dtype)
            except:
                return np.unique(X.astype(str)).astype(orig_dtype)

        def nan_getter(X):
            try:
                return np.isnan(X.astype(np.float64))
            except:
                return (np.char.lower(X.astype(str)) == 'nan').astype(np.uint8)


        _delete_columns_mask = np.zeros(X.shape[1], dtype=np.uint32)
        _delete_rows_mask = np.zeros(X.shape[0], dtype=np.uint32)

        delete_all_msg = \
            lambda x: f"this threshold and recursion depth will delete all {x}"

        for col_idx, _instr in _delete_instr.items():
            if 'INACTIVE' in _instr:
                continue

            # VERIFY VALUES IN TRFM DATA WERE SEEN DURING FIT
            if self._reject_unseen_values:
                TEST_MASK = np.zeros(X.shape[0], dtype=np.uint32)
                for unq in self._total_counts_by_column[col_idx]:
                    if str(unq).lower()=='nan':
                        TEST_MASK += nan_getter(X[:, col_idx])
                    else:
                        TEST_MASK += (X[:, col_idx] == unq).astype(np.uint8)

                if sum(TEST_MASK) != X.shape[0]:
                    UNSEEN_UNQS = \
                        np_unique_handler(X[np.logical_not(TEST_MASK), col_idx])
                    if len(UNSEEN_UNQS) > 10:
                        UNSEEN_UNQS = f"{UNSEEN_UNQS[:10]} others"
                    raise ValueError(f"Transform data has values not seen "
                                     f"during fit --- \n"
                                     f"column index = {col_idx}\n"
                                     f"unseen values = {UNSEEN_UNQS}")
                del TEST_MASK
            # END VERIFY VALUES IN TRFM DATA WERE SEEN DURING FIT

            if 'DELETE COLUMN' in _instr:
                _delete_columns_mask[col_idx] += 1
                _instr = _instr[:-1]
            if _instr == []:
                continue
            for unq in _instr:
                if str(unq).lower() == 'nan':
                    _delete_rows_mask += nan_getter(X[:, col_idx])
                else:
                    _delete_rows_mask += (X[:, col_idx] == unq)

        del np_unique_handler, nan_getter

        ROW_KEEP_MASK = np.logical_not(_delete_rows_mask)
        del _delete_rows_mask
        COLUMN_KEEP_MASK = np.logical_not(_delete_columns_mask)
        del _delete_columns_mask

        if True not in ROW_KEEP_MASK:
            raise ValueError(delete_all_msg('rows'))

        if True not in COLUMN_KEEP_MASK:
            raise ValueError(delete_all_msg('columns'))

        del delete_all_msg

        # END BUILD ROW_KEEP & COLUMN_KEEP MASKS ** ** ** ** ** ** ** ** ** **

        self._row_support = ROW_KEEP_MASK.copy()

        FEATURE_NAMES = self.get_feature_names_out()

        if False not in ROW_KEEP_MASK and False not in COLUMN_KEEP_MASK:
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
                                np.fromiter(TCBC[old_col_idx].keys(), dtype='<U100')
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
                                    __ = dict((zip(list(map(str, TCBC[old_col_idx].keys())),
                                                   list(TCBC[old_col_idx].values()))
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
        __ = self._output_transform
        __ = __ or self._x_original_obj_type
        if True in [j in __ for j in ['dataframe', 'series']]:
            X = pd.DataFrame(X, columns=FEATURE_NAMES)
            if 'series' in __:
                if self.n_features_in_ > 1:
                    raise ValueError(
                        f"cannot return as Series when n_features_in_ is > 0.")
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


    def fit_transform(self, X, y=None):
        """
        Fits MinCountTransformer to X and returns a transformed version of X.

        Parameters
        ----------
        X : {ndarray, pandas.DataFrame, pandas.Series # PIZZA sparse matrix} of
            shape (n_samples, n_features)}
            The data used to determine the uniques and their frequencies and to
            be transformed by rules created from those frequencies.

        y : {ndarray, pandas.DataFrame, pandas.Series} of shape (n_samples,) or
            (n_samples, n_outputs), default=None -
            Target values (None for unsupervised transformations).

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
        Reverse the transformation operation. This operation cannot restore
        removed examples, only features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features_new) - The input samples.

        Returns
        -------
        X_inv : ndarray of shape (n_samples, n_original_features)
            X with columns of zeros inserted where features would have been
            removed by transform.
        """

        self._must_be_fitted()

        X_tr = self._handle_X_y(X_tr, None, return_columns=True)[0]

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
        input_features : array-like of str or None, default=None - Input features.
                If input_features is None, then feature_names_in_ is used as
                feature names in. If feature_names_in_ is not defined, then the
                following input feature names are generated:
                    ["x0", "x1", ..., "x(n_features_in_ - 1)"].
                If input_features is an array-like, then input_features must
                match feature_names_in_ if feature_names_in_ is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects - Transformed feature names.
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
                raise ValueError(f"number of passed input_features does not match "
                             f"number of features seen during (partial_)fit().")
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
        indices : bool, default=False - If True, the return value will be an
                                array of integers, rather than a boolean mask.

        Returns
        -------
        support : ndarray - A slicer that selects the retained rows from the X
                    most recently seen by transform. If indices is False, this
                    is a boolean array of shape (n_input_features, ) in which
                    an element is True if its corresponding row is selected for
                    retention. If indices is True, this is an integer array of
                    shape (n_output_features, ) whose values are indices into
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
        indices : bool, default=False - If True, the return value will be an
                                array of integers rather than a boolean mask.

        Returns
        -------
        support : ndarray - An index that selects the retained features from a
                    feature vector. If indices is False, this is a boolean
                    array of shape (n_input_features,) in which an element is
                    True if its corresponding feature is selected for retention.
                    If indices is True, this is an integer array of shape
                    (n_output_features, ) whose values are indices into the
                    input feature vector.
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
        """Set the output container when "transform" and "fit_transform" are
        called.

        Parameters
        ----------
        transform : {“default”, "numpy_array", “pandas_dataframe”, "pandas_series"},
                    default = None - Configure output of transform and fit_transform.
                    "default": Default output format of a transformer (numpy array)
                    "numpy_array": np.ndarray output
                    "pandas_dataframe": DataFrame output
                    "pandas_series": Series output
                    None: Transform configuration is unchanged

        Returns
        -------
        self : estimator instance - Estimator instance.

        """

        from ._validation._val_transform import _val_transform

        self._output_transform = _val_transform(transform)

        return self


    def set_params(self, **params):
        """Set the parameters of this transformer.

        Valid parameter keys can be listed with get_params(). Note that you can
        directly set the parameters of MinCountTransformer.

        Parameters
        ----------
        params : dict - Estimator parameters.

        Returns
        -------
        self : MinCountTransformer - This estimator.
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


    def _validate_delete_instr(self, _delete_instr):
        """Hidden method. Validate that output of _make_instructions conforms
        to expected format.

        Parameters
        ----------
        _delete_instr : dict
        """

        if not isinstance(_delete_instr, dict):
            raise TypeError(f"_delete_instr must be a dictionary")
        if len(_delete_instr) != self.n_features_in_:
            raise ValueError(
                f"_delete_instr must have an entry for each column in X")

        for col_idx, _instr in _delete_instr.items():
            if 'INACTIVE' in _instr and len(_instr) > 1:
                raise ValueError(f"'INACTIVE' IN len(_delete_instr[{col_idx}]) > 1")
            if 'DELETE COLUMN' in _instr and _instr[-1] != 'DELETE COLUMN':
                raise ValueError(f"'DELETE COLUMN' IS NOT IN THE -1 POSITION "
                                 f"OF _delete_instr[{col_idx}]")
            if len([_ for _ in _instr if _ == 'DELETE COLUMN']) > 1:
                raise ValueError(f"'DELETE COLUMN' IS IN _delete_instr[{col_idx}] "
                                 f"MORE THAN ONCE")


    def test_threshold(self, threshold:int=None, clean_printout=True):
        """Display instructions generated for the current fitted state, subject
        to the passed threshold and the current settings of other parameters.
        The printout will indicate what rows / columns will be deleted, and if
        all columns or all rows will be deleted.

        Parameters
        ----------
        threshold : int, default=None - count_threshold value to test.

        clean_printout: bool, default=True - Truncate printout to fit on screen.
        """

        self._must_be_fitted()
        if callable(self._ignore_columns) or callable(self._handle_as_bool):
            raise ValueError(f"if ignore_columns or handle_as_bool is callable, "
                f"get_support() is only available after a transform is done.")


        if threshold is None:
            threshold = self._count_threshold
            # OK FOR 1 OR MORE RECURSIONS
        else:
            if self._max_recursions == 1:
                if int(threshold) != threshold:
                    raise ValueError(f"threshold must be an integer >= 2")
                if not threshold >= 2:
                    raise ValueError(f"threshold must be an integer >= 2")
            elif self._max_recursions > 1:
                if threshold != self._count_threshold:
                    raise ValueError(f"can only test the original threshold "
                                     f"when max_recursions > 1")

        _delete_instr = self._make_instructions(_threshold=threshold)

        self._validate_delete_instr(_delete_instr)

        print(f'\nThreshold = {threshold}')

        if hasattr(self, 'feature_names_in_'):
            _pad = min(25, max(map(len, self.feature_names_in_)))
        else:
            _pad = len(str(f"Column {self.n_features_in_}"))

        _all_rows_deleted = False
        ALL_COLUMNS_DELETED = []
        _ardm = f"\nAll rows will be deleted." # all_rows_deleted_msg
        for col_idx, _instr in _delete_instr.items():
            _ardd = False   # all_rows_deleted_dummy
            if hasattr(self, 'feature_names_in_'):
                _column_name = self.feature_names_in_[col_idx]
            else: _column_name = f"Column {col_idx+1}"

            if len(_instr)==0:
                print(f"{_column_name[:_pad]}".ljust(_pad+5), "No operations.")
                continue

            if _instr[0] == 'INACTIVE':
                print(f"{_column_name[:_pad]}".ljust(_pad+5), "Ignored")
                continue

            _delete_rows, _delete_col = "", ""
            if 'DELETE COLUMN' in _instr:
                _delete_col = f"Delete column."
                # IF IS BIN-INT & NOT DELETE ROWS, ONLY ENTRY WOULD BE "Delete column."
                _instr = _instr[:-1]
                ALL_COLUMNS_DELETED.append(True)
            else:
                ALL_COLUMNS_DELETED.append(False)

            if len(_instr) == len(self._total_counts_by_column[col_idx]):
                _ardd = True
                _all_rows_deleted = True

            if len(_instr):
                _delete_rows = "Delete rows associated with "
                ctr = 0
                for _value in _instr:
                    try:
                        _value = np.float64(str(_value)[:7])
                        _delete_rows += f"{_value}"
                    except:
                        _delete_rows += str(_value)
                    ctr += 1
                    _max_print_len = (80-_pad-5 - _ardd*(len(_ardm) - 20))
                    if clean_printout and len(_delete_rows) >= _max_print_len and \
                            len(_instr[ctr:]) > 0:
                        _delete_rows += f"... + {len(_instr[ctr:])} other(s). "
                        break
                    if len(_instr[ctr:]) > 0:
                        _delete_rows += ", "
                    else:
                        _delete_rows += ". "

                if _all_rows_deleted:
                    _delete_rows += _ardm

                del ctr, _value

            print(f"{_column_name[:_pad]}".ljust(_pad+5), _delete_rows + _delete_col)

        if False not in ALL_COLUMNS_DELETED:
            print(f'\nAll columns will be deleted.')
        if _all_rows_deleted:
            print(f'{_ardm}')

        del threshold, _delete_instr, _all_rows_deleted, _ardm, col_idx, _instr
        del _column_name, _delete_rows, _delete_col, _pad, _ardd, ALL_COLUMNS_DELETED


    def _check_is_fitted(self):
        """Hidden method. Check to see if the instance has been fitted."""
        return hasattr(self, '_total_counts_by_column')


    def _must_be_fitted(self):
        """Hidden method. Allows access only if the instance has been fitted."""
        if not self._check_is_fitted():
            raise NotFittedError(f"This instance has not been fitted yet. "
                                 f"Fit data with partial_fit() or fit() first.")
        else:
            return True


    def _recursion_check(self):
        if self._recursion_block:
            raise ValueError(f"partial_fit(), fit(), and transform() are not "
                     f"available if max_recursions > 1. fit_transform() only.")


    def _handle_X_y(self, X, y, return_columns=False):
        """Hidden method. Validate dimensions of X and y and standardize to a
        the containers for processing.

        Parameters
        ----------
        X : {ndarray, pandas.DataFrame, pandas.Series} - data object

        y : {ndarray, pandas.DataFrame, pandas.Series} - target object

        return_columns : bool, default=False - If X is given as a pandas dataframe,
            retain the columns attribute of the dataframe and return.

        Returns
        ----------
        X : ndarray - The given data as ndarray.
        y : ndarray - The given target as ndarray.
        _columns : ndarray - Feature names extracted from X.
        """

        # 24_03_03_09_54_00 THE INTENT IS TO RUN EVERYTHING AS NP ARRAY TO
        # STANDARDIZE INDEXING. IF DFS ARE PASSED, COLUMNS CAN OPTIONALLY BE
        # PULLED OFF AND RETAINED.

        _columns = None

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        # IF X IS Series, MAKE DataFrame
        try:
            X = X.to_frame()
            # self._original_obj_type ONLY MATTERS WHEN _handle_X_y IS CALLED
            # THROUGH transform() (OR fit_transform())
            self._x_original_obj_type = self._x_original_obj_type or 'pandas_series'
        except: pass

        # X COULD BE np OR pdDF
        try:
            _columns = np.array(X.columns)
        except:
            pass
        # IF pdDF CONVERT TO np
        try:
            X = X.to_numpy()
            self._x_original_obj_type = \
                self._x_original_obj_type or 'pandas_dataframe'
        except: pass

        # self._x_original_obj_type ONLY MATTERS WHEN _handle_X_y IS CALLED
        # THROUGH transform() (OR fit_transform())
        self._x_original_obj_type = self._x_original_obj_type or 'numpy_array'

        if isinstance(X, np.recarray):
            __ = type(self).__name__
            raise TypeError(f"{__} cannot take numpy recarrays. "
                f"Pass X as a numpy.ndarray, pandas dataframe, or pandas series.")

        if isinstance(X, np.ma.core.MaskedArray):
            __ = type(self).__name__
            raise TypeError(f"{__} cannot take numpy masked arrays. "
                f"Pass X as a numpy.ndarray, pandas dataframe, or pandas series.")

        if not isinstance(X, (type(None), str, dict)):
            try:
                list(X[:10])
                _dtype = str(np.array(X).dtype).lower()
                if True in [_ in _dtype for _ in ['int', 'float']]:
                    X = np.array(X, dtype=_dtype)
                else:
                    X = np.array(X, dtype=object)
                del _dtype
            except:
                pass

        if not isinstance(X, np.ndarray):
            raise TypeError(f"X is an invalid data-type {type(X)}")

        if len(X.shape)==1:
            X = X.reshape((-1,1))

        # X MUST BE np ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        #  ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if y is not None:
            # IF y IS Series, MAKE DataFrame
            try:
                y = y.to_frame()
                # self._y_original_obj_type ONLY MATTERS WHEN _handle_X_y IS
                # CALLED THROUGH transform() (OR fit_transform())
                self._y_original_obj_type = \
                    self._y_original_obj_type or 'pandas_series'
            except: pass

            # X COULD BE np OR pdDF
            # IF pdDF, CONVERT TO np
            try:
                y = y.to_numpy()
                self._y_original_obj_type = \
                    self._y_original_obj_type or 'pandas_dataframe'
            except: pass

            # self._y_original_obj_type ONLY MATTERS WHEN _handle_X_y IS CALLED
            # THROUGH transform() (OR fit_transform())
            self._y_original_obj_type = self._y_original_obj_type or 'numpy_array'

            if isinstance(y, np.recarray):
                __ = type(self).__name__
                raise TypeError(f"{__} cannot take numpy recarrays. "
                                f"Pass y as a numpy.ndarray.")

            if isinstance(y, np.ma.core.MaskedArray):
                __ = type(self).__name__
                raise TypeError(
                    f"{__} cannot take numpy masked arrays. Pass y as a "
                    f"numpy.ndarray, pandas dataframe, or pandas series.")

            try:
                y = np.array(y)
            except:
                pass

            if not isinstance(y, np.ndarray):
                raise TypeError(f"y is an unknown data-type {type(y)}")
            # y MUST BE np OR ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

            # WORKS IF len(y.shape) IS 1 or 2
            _X_rows, _y_rows = X.shape[0], y.shape[0]
            if _X_rows != _y_rows:
                raise ValueError(f"the number of rows in y ({_y_rows}) does "
                             f"not match the number of rows in X ({_X_rows})")

        if return_columns:
            return X, y, _columns
        elif not return_columns:
            return X, y


    @staticmethod
    def _validate_feature_names(_columns, _feature_names_in):
        # see _val_feature_names

        # pizza move import statement
        from ._validation._val_feature_names import _val_feature_names

        _val_feature_names(_columns, _feature_names_in)




    def _val_ign_cols_handle_as_bool(self, kwarg_value, _name):

        """Hidden method. Validate containers and content for ignore_columns and
        handle as bool.

        Parameters
        ----------
        kwarg_value : array-like, callable, None - The current ignore_columns
                        or handle_as_bool object.

        _name : str - Name of the object.

        Returns
        ----------
        kwarg_value : array-like, callable, None - Standardized container.

        """

        while True:  # ESCAPE VALIDATION IF kwarg_value IS CALLABLE
            err_msg = (f"{_name} must be i) None; ii) a list-type that contains "
                       f"all integers or all column names as strings, "
                       f"cannot mix types; iii) a callable that returns ii)")
            if kwarg_value is None:
                kwarg_value = np.array([], dtype=np.uint32)
            elif isinstance(kwarg_value, (str, bool, dict)):
                raise TypeError(err_msg)
            elif callable(kwarg_value):
                break
            else:
                try:
                    kwarg_value = np.array(list(set(kwarg_value)), dtype=object)
                except:
                    raise TypeError(err_msg) from None

            _dtypes = []
            for idx, value in enumerate(kwarg_value):
                if isinstance(value, str):
                    _dtypes.append('str')
                    continue

                try:
                    float(value)
                    if int(value) != value:
                        raise ValueError(err_msg)
                    kwarg_value[idx] = int(value)
                    _dtypes.append('int')
                except:
                    raise ValueError(err_msg)

            if len(_dtypes) != len(kwarg_value):
                raise Exception(f"Error building _dtypes, len != len({_name})")

            if len(np.unique(_dtypes)) not in [0, 1]:
                raise ValueError(err_msg)

            del err_msg, _dtypes

            if hasattr(self, 'n_features_in_'):
                for idx, _column in enumerate(kwarg_value):
                    if 'int' in str(type(_column)).lower():
                        if _column >= self.n_features_in_:
                            raise ValueError(f"{_name} index {_column} is out "
                                 f"of bounds for data with {self.n_features_in_} "
                                 f"columns")
                    elif isinstance(_column, str):
                        if hasattr(self, 'feature_names_in_'):
                            if _column not in self.feature_names_in_:
                                raise ValueError(f'{_name} entry, column '
                                    f'"{_column}", is not in column names seen '
                                                 f'during original fit')
                            # CONVERT COLUMN NAMES TO COLUMN INDEX
                            kwarg_value[idx] = \
                                np.arange(self.n_features_in_)[
                                    self.feature_names_in_==_column][0]
                        elif not hasattr(self, 'feature_names_in_'):
                            raise ValueError(f"{_name} can only be passed as "
                                             f"column names when the data is "
                                             f"passed with column names")
                    else:
                        raise TypeError(f"invalid dtype {type(_column)} in {_name}")

                kwarg_value = kwarg_value.astype(np.uint32)

                if _name=='handle_as_bool' and kwarg_value is not None:
                    if 'obj' in self._original_dtypes[kwarg_value]:
                        IDXS = map(str, kwarg_value[self._original_dtypes[
                                                    kwarg_value] == 'obj'])
                        raise ValueError(f"cannot use handle_as_bool on "
                                         f"str/object columns --- column "
                                         f"index(es) == {', '.join(IDXS)}")
            break

        return kwarg_value


    def _validate(self):
        """
        Validate MinCountTransformer arg and kwargs.

        """

        # pizza move imports when finished
        from _validation._mct_validation import _mct_validation

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
                self.n_jobs
        )


        # PIZZA THINK ABOUT PLACEMENT OF THIS RELATIVE TO _mct_validation
        # extra count_threshold val
        if hasattr(self, '_n_rows_in') and _count_threshold >= self._n_rows_in:
            raise ValueError(f"count_threshold is >= the number of rows, "
                             f"every column not ignored would be deleted.")


    def score(self):
        """Dummy method to spoof dask Incremental and ParallelPostFit wrappers.
        Verified must be here for dask wrappers.
        """

        pass






































