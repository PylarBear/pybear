# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import inspect
from copy import deepcopy

from .._validation._is_dask_gscv import _is_dask_gscv as val_is_dask_gscv

from dask_ml.model_selection import (
    GridSearchCV as dask_GridSearchCV,
    RandomizedSearchCV as dask_RandomizedSearchCV
)





def _val_parent_gscv_kwargs(
        _GSCV_parent,
        _parent_gscv_kwargs: dict[str, any]
    ) -> None:

    """

    1) verify parent kwargs passed via parent_gscv_kwargs are valid for
        the parent grid search class
    2) block dask_ml refit==False - dask GridSearchCV does not expose
        best_params_ whenever refit==False
    3) block dask_ml when multiple scorers are used and refit is callable
    4) block all parents when multiple scorers are used and refit=False"


    Parameters
    ----------
    _GSCV_parent:
        gridsearch class - not instantiated - the parent gridsearch class
        passed to agsw.
    _parent_gscv_kwargs:
        dict[str, any] - the kwargs passed to the agsw instance for the
        parent grid search.


    Return
    ------
    -
        _parent_gscv_kwargs:
            dict[str, any] - validated _parent_gscv_kwargs

    """


    _is_dask = val_is_dask_gscv(_GSCV_parent)

    ALLOWED_KWARGS = inspect.signature(_GSCV_parent).parameters.keys()

    for _kwarg, _value in deepcopy(_parent_gscv_kwargs).items():

        if _kwarg not in ALLOWED_KWARGS:
            raise ValueError(f"invalid kwarg '{_kwarg}' for parent "
                 f"GridSearch class{' dask' if _is_dask else ''} "
                 f"'{_GSCV_parent.__name__}'")

        # there is an anomaly in sklearn and dask_ml GridSearchCV that
        # 'scoring' passed as a list sets the GSCV attr 'multimetric_'
        # to True, even if there is only one string inside the list. The
        # problem it that when multimetric_ is True and refit is False,
        # they block exposing best_params_. This does not happen when
        # 'scoring' is a str. To get around this, if 'scoring' is a
        # list-like and the length is 1, take out what is in it.
        if _kwarg == 'scoring':
            try:
                iter(_value)
                if isinstance(_value, str):
                    raise
                if len(_value) > 1:
                    raise
                _parent_gscv_kwargs[_kwarg] = _value[0]
            except:
                pass


    del ALLOWED_KWARGS


    _scorer = _parent_gscv_kwargs.get('scoring', 'dummy')
    _refit = _parent_gscv_kwargs.get('refit', 'dummy')

    base_err_msg = (f"autogridsearch_wrapper requires that the parent "
        f"grid search class always exposes the best_params_ attribute.")
    err_msg1 = base_err_msg + (f"\ndask_ml GridSearchCV does not expose "
        f"best_params_ under any circumstance when refit=False.")
    err_msg2 = (f"\ndask_ml GridSearchCV does not expose best_params_ "
        f"when multiple scorers are used and refit is a callable")
    if _is_dask:

        # these are the dask_ml grid search modules
        # from dask_ml.model_selection import (
        #     GridSearchCV,
        #     RandomizedSearchCV,
        #     IncrementalSearchCV,
        #     HyperbandSearchCV,
        #     SuccessiveHalvingSearchCV,
        #     InverseDecaySearchCV
        # )

        # block access to situations where dask does not expose best_params_.
        if  _GSCV_parent in (dask_GridSearchCV, dask_RandomizedSearchCV):
            # only GridSearchCV and RandomizedSearchCV have a 'refit' method
            if _refit is False:
                raise AttributeError(err_msg1)
            if callable(_refit) and \
                not isinstance(_scorer, str) \
                and len(_scorer) > 1:
                raise AttributeError(err_msg2)

    err_msg3 = base_err_msg + (
        f"\nNo grid search parent exposes the best_params_ attribute "
        f"when multiple scorers are used and refit=False"
    )
    if not isinstance(_scorer, (str, type(None))) and not callable(_scorer):
        if len(_scorer) > 1 and _refit is False:
            raise AttributeError(err_msg3)

    del _is_dask, base_err_msg, err_msg1, err_msg2, err_msg3


    return _parent_gscv_kwargs























