# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _core_bool_val(
    _param: str,
    _value: bool
) -> None:

    """
    Supports several _validation modules that validate boolean values.


    Parameters
    ----------
    _param:
        str - the name of the parameter being validated

    _value:
        the value for _param that was passed to the MCT instance


    Return
    ------
    -
        None


    """


    if not isinstance(_param, str):
        raise AssertionError(f"design error: '_param' must be a string")


    if not isinstance(_value, bool):
        raise TypeError(f"'{_param}' must be a boolean")


    return









