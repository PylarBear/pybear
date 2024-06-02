# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import inspect

# verify parent GSCV kwargs passed to AutoGridSearch via parent_gscv_kwargs
# are valid

# verify 'refit', for a dask GSCV, if passed, is True


def _val_parent_gscv_kwargs(
        _estimator,
        _GSCV_parent,
        _parent_gscv_kwargs: dict[str, any]
    ) -> None:


    _is_dask = 'DASK' in str(type(_estimator)).upper()


    ALLOWED_KWARGS = inspect.signature(_GSCV_parent).parameters.keys()

    for _kwarg in _parent_gscv_kwargs:

        if _kwarg not in ALLOWED_KWARGS:
            raise ValueError(f"invalid kwarg '{_kwarg}' for parent "
                 f"GridSearch class{' dask' if _is_dask else ''} "
                 f"'{_GSCV_parent.__name__}'")

    del ALLOWED_KWARGS

    err_msg = (f"With a dask estimator, refit must be True to "
               f"expose best_params_. \nDefault refit=True for dask GSCV, "
               f"so refit need not be passed to it via **parent_gscv_kwargs, "
               f"\nbut if refit is passed, refit must be True.")
    if _is_dask:
        if _parent_gscv_kwargs.get('refit', True) is False:
            raise AttributeError(err_msg)

    del _is_dask


























