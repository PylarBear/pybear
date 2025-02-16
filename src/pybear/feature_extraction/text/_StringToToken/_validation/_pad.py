# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union


def _val_pad(
    _pad: Union[str, None],
    _using_dask_ml_wrapper: bool
) -> None:


    """
    Validate 'pad'. Must be a string or None.


    Parameters
    ----------
    _pad:
        Union[str, None] - String used to fill ragged area.
    _using_dask_ml_wrapper:
        bool - Whether the StringToToken instance is wrapped by at least
        one dask_ml wrapper.


    Return
    ------
    -
        None


    """


    if not isinstance(_pad, (str, type(None))):
        raise TypeError(f"'pad' must be a string or None")


    if _using_dask_ml_wrapper:
        # 'pad' must be a string
        if not isinstance(_pad, str):
            raise ValueError(
                f"When using dask_ml wrappers, 'pad' must be provided "
                f"as a string. '' is recommended for compatibility "
                f"with other pybear modules."
            )







