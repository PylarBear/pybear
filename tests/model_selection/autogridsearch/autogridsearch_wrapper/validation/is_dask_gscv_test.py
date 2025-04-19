# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.model_selection.autogridsearch._autogridsearch_wrapper._validation. \
    _is_dask_gscv import _is_dask_gscv as val_dask_gscv

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    GridSearchCV as sk_GridSearchCV,
    RandomizedSearchCV as sk_RandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV
)

from dask_ml.model_selection import (
    GridSearchCV as dask_GridSearchCV,
    RandomizedSearchCV as dask_RandomizedSearchCV,
    IncrementalSearchCV,
    HyperbandSearchCV,
    SuccessiveHalvingSearchCV,
    InverseDecaySearchCV
)

from pybear.model_selection import (
    GSTCV,
    GSTCVDask
)



class TestIsDaskGSCV:


    @pytest.mark.parametrize('junk_gscv',
        (0, 1, 'junk', [0,1], None)
    )
    def test_raises_on_non_module(self, junk_gscv):

        with pytest.raises(AttributeError):
            val_dask_gscv(junk_gscv)


    def test_false_for_sklearn_gscvs(self):

        for sk_gscv_parent in (sk_GridSearchCV, sk_RandomizedSearchCV,
            HalvingGridSearchCV, HalvingRandomSearchCV, GSTCV, GSTCVDask
        ):

            assert not val_dask_gscv(sk_gscv_parent)


    def test_true_for_dask_gscvs(self):

        for dask_gscv_parent in (
            dask_GridSearchCV, dask_RandomizedSearchCV, IncrementalSearchCV,
            HyperbandSearchCV, SuccessiveHalvingSearchCV, InverseDecaySearchCV
        ):

            assert val_dask_gscv(dask_gscv_parent)





