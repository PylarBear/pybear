# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#





def _check_X_constants_dupls(
    _X_constants: dict[int, any],
    _X_dupls: list[list[int]]
) -> None:

    """
    Raise exception if InterceptManager found constants in X or if
    ColumnDeduplicateTransformer found duplicates in X.


    Parameters
    ----------
    _X_constants:
        dict[int, any] - the constants found in X by InterceptManager.
    _X_dupls:
        list[list[int]]] - the sets of duplicates found in X by
        ColumnDeduplicateTransformer.


    Return
    ------
    -
        None



    """



    assert isinstance(_X_constants, dict)
    if len(_X_constants):
        assert all(map(isinstance, _X_constants, (int for _ in _X_constants)))

    assert isinstance(_X_dupls, list)
    if len(_X_dupls):
        for _list in _X_dupls:
            assert isinstance(_list, list)
            assert all(map(isinstance, _list, (int for _ in _list)))


    constants_err = (
        f"X has columns of constants: {_X_constants}"
        f"\nPlease use pybear InterceptManager to remove all constant "
        f"columns from X before using SlimPolyFeatures."
    )

    dupl_err = (
        f"X has duplicate columns: {_X_dupls}"
        f"\nPlease use pybear ColumnDeduplicateTransformer to remove all "
        f"duplicate columns from X before using SlimPolyFeatures."
    )


    if len(_X_constants) and len(_X_dupls):
        raise ValueError(f"\n{constants_err}" + f"\n{dupl_err}")

    elif len(_X_constants):
        raise ValueError(constants_err)

    elif len(_X_dupls):
        raise ValueError(dupl_err)
    else:
        # no constants, no duplicates in X, all good
        pass

















