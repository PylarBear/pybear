# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing_extensions import Union


def _val_transform(_transform: Union[str, None]) -> Union[str, None]:

    """
    Validate 'transform' kwarg to set_output() method is in allowed values.

    """


    ALLOWED_TRANSFORMS = [
        "default",
        "numpy_array",
        "pandas_dataframe",
        "pandas_series",
        None
    ]

    err_msg = (f"transform must be in {', '.join(map(str, ALLOWED_TRANSFORMS))}. "
               f"passed {_transform}.")

    if not isinstance(_transform, (str, type(None))):
        raise TypeError(err_msg(_transform))

    if _transform is None:
        pass
    else:
        _transform = _transform.lower()

    if _transform not in ALLOWED_TRANSFORMS:
        raise ValueError(err_msg)

    del err_msg

    return _transform









