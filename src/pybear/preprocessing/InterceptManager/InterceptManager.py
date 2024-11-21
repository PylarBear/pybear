# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


# pizza dont forget to clean up these imports!!
from typing import Iterable, Literal, Optional
from typing_extensions import Union, Self
from ._type_aliases import (
    KeepType, DataType
)
from numbers import Real, Integral

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.exceptions import NotFittedError

from ._validation._validation import _validation

from ._partial_fit._find_constants import _find_constants
from ._partial_fit._make_instructions import _make_instructions
from ._partial_fit._set_attributes import _set_attributes
from ._inverse_transform._inverse_transform import _inverse_transform
from ._transform._transform import _transform


class InterceptManager(BaseEstimator, TransformerMixin):

    """
    pizza say something


    Parameters
    ----------
    keep:
        Optional[Union[Literal['first', 'last', 'random'], dict[str,any], None]], int, str, callable]
        default='last' - pizza!
    equal_nan:
        Optional[bool], default=True - pizza!
    rtol:
        real number, default = 1e-5 - The relative difference tolerance
            for equality. See numpy.allclose.
    atol:
        real number, default = 1e-8 - The absolute tolerance parameter
            for equality. See numpy.allclose.
    n_jobs:
        # pizza finalize this based on benchmarking
        Optional[Integral], default=-1 - The number of joblib Parallel
        jobs to use when scanning the data for columns of constants. The
        default is to use processes, but can be overridden externally
        using a joblib parallel_config context manager. The default
        number of jobs is -1 (all processors). To get maximum speed
        benefit, pybear recommends using the default setting.


    Attributes
    ----------
    n_features_in_:
        int - number of features in the fitted data before deduplication.

    feature_names_in_:
        NDArray[str] - The names of the features as seen during fitting.
        Only accessible if X is passed to :methods: partial_fit or fit
        as a pandas dataframe that has a header.

    constant_columns_:
        dict[int, any] - pizza!

    kept_columns_:
        dict[int, any] - pizza!

    removed_columns_:
        dict[int, any] - pizza!

    column_mask_:
        NDArray[Union[bool]


    See Also
    --------
    numpy.ndarray
    pandas.core.frame.DataFrame
    scipy.sparse
    numpy.isclose
    numpy.unique


    """


    _parameter_constraints: dict = {
        "keep": [StrOptions({"first", "last", "random", "none"}), dict, Integral, str, callable],
        "equal_nan": ["boolean"],
        "rtol": [Real],
        "atol": [Real],
        "n_jobs": [Integral, None]
    }


    def __init__(
        self,
        *,
        keep: KeepType='last',
        equal_nan: Optional[bool]=True,
        rtol: Optional[Real]=1e-5,
        atol: Optional[Real]=1e-8,
        n_jobs: Optional[Integral]=-1  # pizza benchmark what is the best setting
    ):

        self.keep = keep
        self.equal_nan = equal_nan
        self.rtol = rtol
        self.atol = atol
        self.n_jobs = n_jobs



    def _reset(self):

        """
        Reset internal data-dependent state of the transformer.
        __init__ parameters are not changed.

        """

        if hasattr(self, 'constant_columns_'):
            del self.constant_columns_
            del self.kept_columns_
            del self.removed_columns_
            del self.column_mask_


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
        # and/or adds columns, must build a one-off.

        # when there is a {'Intercept': 1} in :param: keep, need to make sure
        # that that column is accounted for here! and the dropped columns are
        # also accounted for!


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

            # adjust if appending a new intercept column
            if len(input_features) != self.n_features_in_:
                raise ValueError("input_features should have length equal")

            if hasattr(self, 'feature_names_in_'):

                if not np.array_equal(input_features, self.feature_names_in_):
                    raise ValueError(
                        f"input_features is not equal to feature_names_in_"
                    )

            if isinstance(self.keep, dict):
                input_features = np.hstack((input_features, list(self.keep.keys())[0]))

            out = np.array(input_features, dtype=object)[self.column_mask_]

            return out

        elif hasattr(self, 'feature_names_in_'):  # and input_features is None
            if isinstance(self.keep, dict):
                input_features = np.hstack((self.feature_names_in_, list(self.keep.keys())[0]))
                out = input_features[self.column_mask_].astype(object)
            else:
                out = self.feature_names_in_[self.column_mask_].astype(object)

            return out

        else:  # and input_features is None
            try:
                input_features = \
                    np.array([f"x{i}" for i in range(self.n_features_in_)])
                if isinstance(self.keep, dict):
                    input_features = np.hstack((input_features, list(self.keep.keys())[0]))
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


    # def get_params - inherited from BaseEstimator
    # if ever needed, hard code that can be substituted for the
    # BaseEstimator get/set_params can be found in GSTCV_Mixin


    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(
        self,
        X:DataType,
        y:any=None
    ) -> Self:

        """
        Perform incremental fitting on one or more data sets. Determine
        the constant columns in the given data, subject to the criteria
        defined in :params: rtol, atol, and equal_nan.


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - Data to remove duplicate columns from.
        y:
            {vector-like of shape (n_samples,) or None}, default = None -
            ignored. The target for the data.


        Return
        ------
        -
            self: the fitted InterceptManager instance.




        """


        # validation of X must be done here, not in a separate module
        # BaseEstimator has _validate_data method, which when called
        # exposes n_features_in_ and feature_names_in_.
        X = self._validate_data(
            X=X,
            reset=not hasattr(self, "constant_columns_"),
            cast_to_ndarray=False,
            # vvv become **check_params, and get fed to check_array()
            accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
            dtype=None,
            force_all_finite="allow-nan",
            ensure_2d=True,
            ensure_min_features=1,
            order='F'
        )

        # reset – Whether to reset the n_features_in_ attribute. If False,
        # the input will be checked for consistency with data provided when
        # reset was last True.
        # It is recommended to call reset=True in fit and in the first call
        # to partial_fit. All other methods that validate X should set
        # reset=False.
        #
        # cast_to_ndarray – Cast X and y to ndarray with checks in
        # check_params. If False, X and y are unchanged and only
        # feature_names_in_ and n_features_in_ are checked.

        # ^^^^^^ pizza be sure to review _validate_data! ^^^^^^^

        _validation(
            X,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            self.keep,
            self.equal_nan,
            self.rtol,
            self.atol,
            self.n_jobs
        )


        # dictionary of column indices and respective constant values
        self.constant_columns_:dict[int, any] = \
            _find_constants(
                X,
                self.constant_columns_ if hasattr(self, 'constant_columns_') else {},
                self.equal_nan,
                self.rtol,
                self.atol,
                self.n_jobs
            )


        if callable(self.keep):
            _keep = self.keep(X)
            if _keep not in self.constant_columns_:
                raise ValueError(
                    f"'keep' callable has returned an integer column index ({_keep}) "
                    f"that is not a column of constants. \nconstant columns: "
                    f"{self.constant_columns_}"
                )
        else:
            _keep = self.keep


        _instructions = _make_instructions(
            _keep,
            self.constant_columns_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            X.shape
        )

        self.kept_columns_, self.removed_columns_, self.column_mask_ = \
            _set_attributes(
                self.constant_columns_,
                _instructions,
                self.n_features_in_
            )

        return self


    def fit(
        self,
        X:DataType,
        y:any=None
    ) -> Self:

        """
        Perform a single fitting on a data set. Determine the constant
        columns in the given data, subject to the criteria defined
        in :params: rtol, atol, and equal_nan.


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - Data to remove duplicate columns from.
        y:
            {vector-like of shape (n_samples,) or None}, default = None -
            ignored. The target for the data.


        Return
        ------
        -
            self: the fitted InterceptManager instance.



        """

        self._reset()
        return self.partial_fit(X, y=y)


        return self.partial_fit(X)


    def inverse_transform(
        self,
        X: DataType,
        *,
        copy: bool = None
    ) -> DataType:

        """
        Revert transformed data back to its original state. This operation
        cannot restore any nan-like values that may have been in the
        original untransformed data.


        Parameters
        ----------
        X :
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features - n_features_removed) - A transformed data set.
        copy:
            Union[bool, None], default=None - Whether to make a copy of
            X before the inverse transform.

        Return
        ------
        -
            X_tr : {array-like, scipy sparse matrix} of shape (n_samples,
                n_features) - Transformed data reverted to its original
                untransformed state.



        """

        # pizza what if set_params is changed?
        # think on this
        # if we cant come up with a way to validate/ensure that the data
        # being inverted back matches with the current state of the params,
        # then will have to put some disclaimers in the docs.

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
            dtype=None,
            force_all_finite="allow-nan",
            ensure_2d=True,
            copy=copy or False,
            order='F'
        )

        # if _keep is a dict, a column of constants was stacked to the right
        # side of the data. check that the passed data matches against _keep,
        # and remove the column
        if isinstance(self.keep, dict):
            _unqs = np.unique(X[:, -1])
            if len(_unqs) == 1 and _unqs[0] == self.keep[list(self.keep.keys())[0]]:
                X = np.delete(X, -1, axis=1)
            else:
                raise ValueError(
                    f":param: 'keep' is a dictionary but the last column of the "
                    f"data to be inverse transformed does not match."
                )


        out = _inverse_transform(
            X,
            self.removed_columns_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None
        )

        if isinstance(out, np.ndarray):
            out = np.ascontiguousarray(out)


        return out


    def score(self, X, y=None):
        """
        Dummy method to spoof dask Incremental and ParallelPostFit
        wrappers. Verified must be here for dask wrappers.
        """

        pass


    # def set_params(self):
        # pizza! dont forget! once the instance is fitted, cannot change equal_nan, rtol, and atol!
        # ... or maybe u can.... its just that new fits will be fitted subject to
        # different rules than prior fits


    # def set_output(self) - inherited from TransformerMixin


    def transform(
        self,
        X: DataType,
        copy: bool=None
    ) -> DataType:

        """
        Manage the constant columns from X. Apply the criteria given
        by :param: keep to the sets of constant columns found during fit.


        Parameters
        ----------
        X:
            {array-like, scipy sparse matrix} of shape (n_samples,
            n_features) - The data.
        copy:
            Union[bool, None], default=None - Whether to make a copy of
            X before the transform.


        Return
        ------
        -
            X: {array-like, scipy sparse matrix} of shape (n_samples,
                n_features - n_removed_features) - The deduplicated data.




        """

        check_is_fitted(self)

        if not isinstance(copy, (bool, type(None))):
            raise TypeError(f"'copy' must be boolean or None")

        X = self._validate_data(
            X=X,
            reset=False,
            cast_to_ndarray=False,
            copy=copy or False,
            accept_sparse=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
            dtype=None,
            force_all_finite="allow-nan",
            ensure_2d=True,
            ensure_min_features=1,
            order ='F'
        )

        _validation(
            X,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            self.keep,
            self.equal_nan,
            self.rtol,
            self.atol,
            self.n_jobs
        )

        # this needs to be redone every transform in case 'keep' was
        # changed via set params after fit
        _instructions = _make_instructions(
            self.keep,
            self.constant_columns_,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            X.shape
        )

        self.kept_columns_, self.removed_columns_, self.column_mask_ = \
            _set_attributes(
                self.constant_columns_,
                _instructions,
                self.n_features_in_
            )


        out = _transform(X, _instructions)

        if isinstance(out, np.ndarray):
            out = np.ascontiguousarray(out)

        return out















