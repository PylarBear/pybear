# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable
from typing_extensions import (
    TypeAlias,
    Union
)
import numpy.typing as npt
from ..__type_aliases import (
    NumpyTypes,
    PandasTypes,
    PolarsTypes,
    ScipySparseTypes
)

from .._get_feature_names_out import get_feature_names_out
from .._check_n_features import check_n_features
from .._check_feature_names import check_feature_names
from .._check_is_fitted import check_is_fitted

XContainer: TypeAlias = \
    Union[NumpyTypes, PandasTypes, PolarsTypes, ScipySparseTypes]



class FeatureMixin:


    def _check_n_features(
        self,
        X: XContainer,
        reset: bool
    ) -> int:
        """Set the `n_features_in_` attribute, or check against it.

        pybear recommends calling reset=True in :meth:`fit` and in the
        first call to :meth:`partial_fit`. All other methods that
        validate `X` should set 'reset=False'.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features) or (n_samples,)
            The input data; with a 'shape' attribute.
        reset : bool
            If True, the `n_features_in_` attribute is set to 'X.shape[1]'

            If False:
                if `n_features_in_` exists check it is equal to 'X.shape[1]'

                if `n_features_in_` does *not* exist the check is skipped

        Returns
        -------
        n_features : int
            The number of features in `X`.

        Notes
        -----

        **Type Aliases**

        NumpyTypes:
            numpy.ndarray

        PandasTypes:
            Union[pandas.core.series.Series, pandas.core.frame.DataFrame]

        PolarsTypes:
            Union[polars.series.Series, polars.dataframe.DataFrame]

        ScipySparseTypes:
            Union[
                ss.csc_matrix, ss.csc_array, ss.csr_matrix, ss.csr_array,
                ss.coo_matrix, ss.coo_array, ss.dia_matrix, ss.dia_array,
                ss.lil_matrix, ss.lil_array, ss.dok_matrix, ss.dok_array,
                ss.bsr_matrix, ss.bsr_array
            ]

        XContainer:
            Union[NumpyTypes, PandasTypes, PolarsTypes, ScipySparseTypes]

        """

        n_features_in_ = check_n_features(
            X,
            getattr(self, 'n_features_in_', None),
            reset
        )

        self.n_features_in_ = n_features_in_

        return n_features_in_


    def _check_feature_names(
        self,
        X: XContainer,
        reset: bool
    ) -> npt.NDArray[object]:
        """Set or check the `feature_names_in_` attribute.

        pybear recommends setting 'reset=True' in :meth:`fit` and in the
        first call to :meth:`partial_fit`. All other methods that validate
        `X` should set 'reset=False'.

        If reset is True:
            Get the feature names from `X` and return. If `X` does not
            have valid string feature names, return None.
            `feature_names_in_` does not matter.

        If reset is False:
            When `feature_names_in_` exists and the checks of this module
            are satisfied then `feature_names_in_` is always returned.
            If `feature_names_in_` does not exist and the checks of this
            module are satisfied then None is always returned regardless
            of any header that the current `X` may have.

            If `feature_names_in_` exists (a header was seen on first fit):
                if `X` has a (valid) header:
                    Validate that the feature names of `X` (if any) have
                    the exact names and order as those seen during fit.
                    If they are equal, return the feature names; if they
                    are not equal, raise ValueError.

                if `X` does not have a (valid) header:
                    Warn and return `feature_names_in_`.

            If `feature_names_in_` does not exist (a header was not
                seen on the first fit):

                if `X` does not have a (valid) header: return None

                if `X` has a (valid) header:  Warn and return None.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features) or (n_samples, ).
            The data from which to extract feature names. `X` will
            provide feature names if it is a dataframe constructed with
            a valid header of strings. Some objects that are known to
            yield feature names are pandas dataframes, dask dataframes,
            and polars dataframes. If `X` does not have a valid header
            then None is returned. Objects that are known to not yield
            feature names are numpy arrays, dask arrays, and scipy sparse
            matrices/arrays.                  .
        feature_names_in_ : NDArray[object] of shape (n_features, )
            The feature names seen on the first fit, if an object with a
            valid header was passed on the first fit. None if feature
            names were not seen on the first fit.
        reset : bool
            Whether to reset the `feature_names_in_` attribute. If False,
            the feature names of `X` will be checked for consistency with
            feature names of data provided when reset was last True.

        Returns
        -------
        feature_names_in_ : Union[NDArray[object], None]
            The validated feature names if feature names were seen the
            last time reset was True. None if the estimator/transformer
            did not see valid feature names at the first fit.

        Notes
        -----

        **Type Aliases**

        NumpyTypes:
            numpy.ndarray

        PandasTypes:
            Union[pandas.core.series.Series, pandas.core.frame.DataFrame]

        PolarsTypes:
            Union[polars.series.Series, polars.dataframe.DataFrame]

        ScipySparseTypes:
            Union[
                ss.csc_matrix, ss.csc_array, ss.csr_matrix, ss.csr_array,
                ss.coo_matrix, ss.coo_array, ss.dia_matrix, ss.dia_array,
                ss.lil_matrix, ss.lil_array, ss.dok_matrix, ss.dok_array,
                ss.bsr_matrix, ss.bsr_array
            ]

        XContainer:
            Union[NumpyTypes, PandasTypes, PolarsTypes, ScipySparseTypes]

        """

        feature_names_in = check_feature_names(
            X,
            getattr(self, 'feature_names_in_', None),
            reset
        )

        # if hasattr(self, 'feature_names_in_') and check_feature_names()
        # returns None when reset is True, then that means that the new
        # object passed to fit() does not have a header and need to
        # delete the feature_names_in_ attribute from self
        if feature_names_in is None:
            if reset and hasattr(self, 'feature_names_in_'):
                delattr(self, "feature_names_in_")
        elif feature_names_in is not None:
            self.feature_names_in_ = feature_names_in

        return feature_names_in


    def get_feature_names_out(
        self,
        input_features: Union[Iterable[str], None]=None
    ) -> npt.NDArray[object]:
        """Return the feature name vector for the transformed output.

        - If `input_features` is None, then :attr:`feature_names_in_` is
          used as feature names in. If :attr:`feature_names_in_` is not
          defined, then the following input feature names are generated:
          '["x0", "x1", ..., "x(:attr:`n_features_in_` - 1)"]'.
        - If :attr:`input_features` is an array-like, then :attr:`input_features`
          must match :attr:`feature_names_in_` if :attr:`feature_names_in_`
          is defined.

        This mixin provides the get_feature_names_out() method of the
        pybear API to pybear transformers. The `get_feature_names_out`
        method returns the features names that correspond to the output
        of :meth:`transform`. This particular mixin can only be used
        for transformers that do not alter the feature axis, that is,
        the feature name output is one-to-one with the feature name
        input. If the transformer does alter the feature axis of the
        data, then a dedicated `get_feature_names_out` method will need
        to be used in place of this.

        Parameters
        ----------
        input_features : Union[Iterable[str], None]
            Input features.

        Returns
        -------
        feature_names_out : npt.NDArray[object]
            The feature names for the transformed output.

        """

        check_is_fitted(self)

        return get_feature_names_out(
            input_features,
            getattr(self, 'feature_names_in_', None),
            self.n_features_in_
        )




