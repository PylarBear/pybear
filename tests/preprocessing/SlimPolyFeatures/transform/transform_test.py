# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from typing_extensions import Union

import numbers

import numpy as np
import pandas as pd
import scipy.sparse as ss
from joblib import Parallel, delayed, wrap_non_picklable_objects

from pybear.preprocessing.SlimPolyFeatures._transform import _transform

import pytest



pytest.skip(reason=f"not started, not finished", allow_module_level=True)



# pizza finish this


class TransformTest:


    def test_transform(self):


        # def _transform(
        #     X: ss.csc_array,
        #     _combos: list[tuple[int, ...]],
        #     dropped_poly_duplicates_: dict[tuple[int, ...], tuple[int, ...]],
        #     poly_constants_: dict[tuple[int, ...], any],
        #     _n_jobs: Union[numbers.Integral, None]
        # ) -> ss.csc_array:

        """
        Pizza. Build the polynomial expansion for X as a scipy sparse csc array.
        Index tuples in :param: _combos that are not in :param: dropped_poly_duplicates_
        are omitted from the expansion.
    
    
        Parameters
        ----------
        X:
            {scipy sparse csc_array} of shape (n_samples,
            n_features) - The data to be expanded.
        _combos:
            list[tuple[int, ...]] -
        dropped_poly_duplicates_:
            dict[tuple[int, ...], tuple[int, ...]] -
    
    
        Return
        ------
        -
            POLY: scipy sparse csc array of shape (n_samples, n_kept_polynomial_features) -
            The polynomial expansion.
    
        """


        # validation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        assert isinstance(X, ss.csc_array)
        assert isinstance(_combos, list)
        for _tuple in _combos:
            assert isinstance(_tuple, tuple)
            assert all(map(isinstance, _tuple, (int for _ in _tuple)))
        for k, v in dropped_poly_duplicates_.items():
            assert isinstance(k, tuple)
            assert all(map(isinstance, k, (int for _ in k)))
            assert isinstance(v, tuple)
            assert all(map(isinstance, v, (int for _ in v)))
        assert isinstance(poly_constants_, dict)
        assert all(map(isinstance, poly_constants_, (tuple for _ in poly_constants_)))
        assert isinstance(_n_jobs, (numbers.Integral, type(None)))
        assert _n_jobs >= -1 and _n_jobs != 0
        # END validation - - - - - - - - - - - - - - - - - - - - - - - - - - - -


        ACTIVE_COMBOS = []
        for _combo in _combos:

            if _combo in dropped_poly_duplicates_:
                continue

            if _combo in poly_constants_:
                continue

            ACTIVE_COMBOS.append(_combo)


        @wrap_non_picklable_objects
        def _poly_stacker(_columns):
            return ss.csc_array(_columns.prod(1))


        # pizza, do a benchmark on this, is it faster to just do a for loop with all this serialization?
        joblib_kwargs = {'prefer': 'processes', 'return_as': 'list', 'n_jobs': _n_jobs}
        out = Parallel(**joblib_kwargs)(
            delayed(
                _poly_stacker(_columns_getter(X, combo))
            ) for combo in ACTIVE_COMBOS
        )


        POLY = ss.hstack(out)


        return POLY









