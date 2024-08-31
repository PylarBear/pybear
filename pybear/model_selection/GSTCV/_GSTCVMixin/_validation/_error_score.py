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

    Validate that error_score is a numeric value or literal 'raise'.

    Parameters
    ----------
    _error_score:
        Union[int, float, Literal['raise']] -
        Score to assign if an error occurs in estimator fitting.

    Returns
    -------
    -
        _error_score: Union[int, float, Literal['raise']] - the validated
            error_score

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





