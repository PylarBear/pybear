# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest
import numpy as np
from pybear.feature_extraction.text._stop_words import _stop_words


class TestStopWords:


    def test_returns_numpy_array(self):

        out = _stop_words()

        assert isinstance(out, np.ndarray)

















