# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Any
from .._type_aliases import SKKFoldType

import numpy as np




def _estimator_fit_params_helper(
    data_len: int,
    fit_params: dict[str, Any],
    KFOLD: SKKFoldType
) -> dict[int, dict[str, Any]]:

    """
    This module customizes the estimator's fit params for each pass of
    cv, to be passed at fit time for the respective fold. This is being
    done via a dictionary keyed by fold index, whose values are
    dictionaries holding the respective fit params for that fold. In
    particular, this is designed to perform splitting on any fit param
    whose length matches the number of examples in the data, so that the
    contents of that fit param are matched correctly to the train fold of
    data concurrently being passed to fit. Other params that are not split
    are simply replicated into each dictionary inside the helper.


    Parameters
    ----------
    data_len:
        int - the number of examples in the full data set.
    fit_params:
        dict[str, Any] - all the fit params passed to GSTCV fit for
        the estimator.
    KFOLD:
        SKKFoldType - The KFold indices that were used to create the
        train / test splits of data.


    Return
    ------
    -
        _fit_params_helper: dict[int, dict[str, Any]] - a dictionary of
        customized fit params for each pass of cv.

    """

    try:
        float(data_len)
        if isinstance(data_len, bool):
            raise
        if not int(data_len) == data_len:
            raise
        data_len = int(data_len)
        if not data_len > 0:
            raise
    except:
        raise TypeError(f"'data_len' must be an integer greater than 0")

    assert isinstance(fit_params, dict)
    assert all(map(isinstance, list(fit_params), (str for _ in fit_params)))

    assert isinstance(KFOLD, list), f"{type(KFOLD)=}"
    assert all(map(isinstance, KFOLD, (tuple for _ in KFOLD)))


    _fit_params_helper = {}

    for f_idx, (train_idxs, test_idxs) in enumerate(KFOLD):

        _fit_params_helper[f_idx] = {}

        for _fit_param_key, _fit_param_value in fit_params.items():

            try:
                iter(_fit_param_value)
                if isinstance(_fit_param_value, (dict, str)):
                    raise
                np.array(list(_fit_param_value))
            except:
                _fit_params_helper[f_idx][_fit_param_key] = _fit_param_value
                continue

            # only get here if try did not except

            __ = np.array(list(_fit_param_value)).ravel()

            if len(__) == data_len:
                _fit_params_helper[f_idx][_fit_param_key] = __[train_idxs]
            else:
                _fit_params_helper[f_idx][_fit_param_key] = _fit_param_value
            del __


    return _fit_params_helper





