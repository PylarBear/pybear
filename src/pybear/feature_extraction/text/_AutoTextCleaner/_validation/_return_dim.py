# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import ReturnDimType



def _val_return_dim(
    _return_dim: ReturnDimType
) -> None:

    """
    Validate 'return_dim', must be None or literal '1D' or '2D'.


    Parameters
    ----------
    _return_dim:
        ReturnDimType - the dimensionality of the container to return
        the cleaned data in, regardless of the input dimension. If None,
        return the output with the same dimensionality as given.


    Returns
    -------
    -
        None


    """

    # return_dim:Optional[Union[Literal['1D', '2D'], None]] = None,


    if _return_dim is None:
        return


    err_msg = "'return_dim' must be None or literal '1D' or '2D'."


    if _return_dim not in ['1D', '2D']:
        raise ValueError(err_msg)





