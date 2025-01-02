# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from typing import Iterable
import numpy.typing as npt

from .._get_feature_names_out import get_feature_names_out
from .._check_n_features_in import check_n_features_in





class FeatureMixin:


    def _check_n_features(
        self,
        X,
        reset
    ) -> int:


        """
        Set the 'n_features_in_' attribute, or check against it.

        pybear recommends calling reset=True in 'fit' and in the first call
        to 'partial_fit'. All other methods that validate 'X' should set
        'reset=False'.


        Parameters
        ----------
        X:
            array-like with a 'shape' attribute of shape (n_samples,
            n_features) - The input data.
        n_features_in_:
            Union[int, None] - the number of features in the
            data. If this attribute exists, it is integer. If it does not
            exist, it is None.
        reset:
            bool -
            If True, the 'n_features_in_' attribute is set to 'X.shape[1]'
            If False:
                if n_features_in_ exists check it is equal to 'X.shape[1]'
                if n_features_in_ does *not* exist the check is skipped


        Return
        ------
        -
            n_features: int - the number of features in X.

        """

        return check_n_features_in(
            X,
            self.n_features_in_,
            reset
        )


    # # pizza
    # def _check_feature_names_in(
    #     self
    #
    # ):
    #
    #     """
    #
    #
    #     """
    #
    #     return check_feature_names_in(
    #         X,
    #         self.n_features_in_,
    #         reset
    #     )



    def get_feature_names_out(
        self,
        input_features: Iterable[str]
    ) -> npt.NDArray[object]:

        """
        This mixin provides the get_feature_names_out() method of the pybear
        API to pybear transformers.
        The get_feature_names_out() method returns the features names that
        correspond to the output of :method: transform. This can only be
        used for transformers that do not alter the feature axis, that is,
        the feature name output is one-to-one with the feature name input.

        """


        return get_feature_names_out(
            input_features,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            self.n_features_in_
        )

















