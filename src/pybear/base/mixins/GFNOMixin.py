# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from typing import Iterable
import numpy.typing as npt

from .._get_feature_names_out import get_feature_names_out






class GFNOMixin:

    """
    This mixin provides the get_feature_names_out() method of the pybear
    API to pybear transformers.
    The get_feature_names_out() method returns the features names that
    correspond to the output of :method: transform. This can only be
    used for transformers that do not alter the feature axis, that is,
    the feature name output is one-to-one with the feature name input.

    """


    def get_feature_names_out(
        self,
        input_features: Iterable[str]
    ) -> npt.NDArray[object]:


        return get_feature_names_out(
            input_features,
            self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None,
            self.n_features_in_
        )































