# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _core_bool_val(_attr: str, _value: bool) -> bool:

    """
    Supports several _validation modules

    """


    if not isinstance(_value, bool):
        raise TypeError(f"{_attr} must be a bool")

    return _value









