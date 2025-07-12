# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import (
    TypeAlias,
    Union
)
import numpy.typing as npt

import pandas as pd
import polars as pl



Python1DTypes: TypeAlias = Union[list[str], tuple[str], set[str]]

Numpy1DTypes: TypeAlias = npt.NDArray[str]

Pandas1DTypes: TypeAlias = pd.Series

Polars1DTypes: TypeAlias = pl.Series

Dim1Types: TypeAlias = Union[
    Python1DTypes, Numpy1DTypes, Pandas1DTypes, Polars1DTypes
]


Python2DTypes: TypeAlias = Sequence[Sequence[str]]

Numpy2DTypes: TypeAlias = npt.NDArray[str]

Pandas2DTypes: TypeAlias = pd.DataFrame

Polars2DTypes: TypeAlias = pl.DataFrame

Dim2Types: TypeAlias = Union[
    Python2DTypes, Numpy2DTypes, Pandas2DTypes, Polars2DTypes
]



def _map_X_to_list(
    _X: Union[Dim1Types, Dim2Types]
) -> Union[list[str], list[list[str]]]:
    """
    Convert the given 1D or (possibly ragged) 2D container of strings
    into list[str] for 1D or list[list[str]] for 2D.

    Parameters
    ----------
    _X : Union[Dim1Types, Dim2Types]
        The 1D or (possibly ragged) 2D data container to be converted to
        list[str] or list[list[str]].

    Returns
    -------
    _X : Union[list[str], list[list[str]]]
        The data container mapped to list[str] for 1D or list[list[str]]
        for 2D containers.

    Notes
    -----

    **Type Aliases**

    Python1DTypes:
        Union[list[str], tuple[str], set[str]]

    Numpy1DTypes:
        numpy.ndarray[str]

    Pandas1DTypes:
        pandas.core.series.Series

    Polars1DTypes:
        polars.series.Series

    Dim1Types:
        Union[Python1DTypes, Numpy1DTypes, Pandas1DTypes, Polars1DTypes]

    Python2DTypes:
        Sequence[Sequence[str]]

    Numpy2DTypes:
        numpy.ndarray[str]

    Pandas2DTypes:
        pandas.core.frame.DataFrame

    Polars2DTypes:
        polars.dataframe.DataFrame

    Dim2Types:
        Union[Python2DTypes, Numpy2DTypes, Pandas2DTypes, Polars2DTypes]

    """


    if all(map(isinstance, _X, (str for _ in _X))):
        _X = list(_X)
    else:
        if isinstance(_X, pd.DataFrame):
            _X = list(map(list, _X.values))
        elif isinstance(_X, pl.DataFrame):
            _X = list(map(list, _X.rows()))
        else:
            _X = list(map(list, _X))


    return _X






