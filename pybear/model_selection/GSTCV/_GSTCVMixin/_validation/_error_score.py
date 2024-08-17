# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Union, Literal




def _validate_error_score(
        _error_score: Union[Literal['raise'], float, int]
    ) -> Union[Literal['raise'], float, int]:

    """
    sklearn & dask 24_07_07_16_21_00, both are identical except sklearn
    default = np.nan, dask default = 'raise'

    Value to assign to the score if an error occurs in estimator fitting.
    If set to ‘raise’, the error is raised. If a numeric value is given,
    a warning is raised and the error score value is inserted into the
    subsequent calculations in place of the missing value. This parameter
    does not affect the refit step, which will always raise the error.

    Parameters
    ----------
    _error_score: Union[int, float, Literal['raise']] - 

    Returns
    -------
    -
        _error_score: Union[int, float, Literal['raise']] -

    """


    err_msg = (f"kwarg 'error_score' must be 1) literal string 'raise', "
               f"or 2) any number including np.nan")

    if isinstance(_error_score, str):
        _error_score = _error_score.lower()
        if _error_score != 'raise':
            raise ValueError(err_msg)
    else:
        try:
            float(_error_score)
            if isinstance(_error_score, bool):
                raise Exception
        except:
            raise TypeError(err_msg)

    del err_msg

    return _error_score





