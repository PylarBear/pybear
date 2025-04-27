# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ..._type_aliases import ClassifierProtocol

from ....GSTCV._GSTCVDask._validation._dask_estimator import _val_dask_estimator
from ....GSTCV._GSTCVDask._validation._cache_cv import _val_cache_cv
from ....GSTCV._GSTCVDask._validation._iid import _val_iid



def _validation(
    _estimator: ClassifierProtocol,
    _iid: bool,
    _cache_cv: bool
) -> None:

    """
    Centralized hub for GSTCVDask parameter validation. See the
    submodules for more information.


    Parameters
    ----------
    _estimator:
        ClassifierProtocol
    _iid:
        bool
    _cache_cv:
        bool


    Returns
    -------
    -
        None

    """


    _val_dask_estimator(_estimator)

    _val_iid(_iid)

    _val_cache_cv(_cache_cv)




