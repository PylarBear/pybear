# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Literal


from ._atol import _val_atol
from ._conflict import _val_conflict
from ._degree import _val_degree
from ._do_not_drop import _val_do_not_drop
from ._drop_constants import _val_drop_constants
from ._drop_duplicates import _val_drop_duplicates
from ._equal_nan import _val_equal_nan
from ._include_bias import _val_include_bias
from ._interaction_only import _val_interaction_only
from ._keep import _val_keep
from ._min_degree import _val_min_degree
from ._n_jobs import _val_n_jobs
from ._order import _val_order
from ._output_sparse import _val_output_sparse
from ._rtol import _val_rtol
from ._X import _val_X



from .._type_aliases import (
    ColumnsType,
    ConflictType,
    DataType,
    DoNotDropType,
    KeepType
)
from typing_extensions import Union


def _validation(
    _X:DataType,
    _columns: ColumnsType,
    _degree: int,
    _min_degree: int,
    _drop_duplicates: bool,
    _keep: KeepType,
    _do_not_drop: DoNotDropType,
    _conflict: ConflictType,
    _interaction_only: bool,
    _include_bias: bool,
    _drop_constants: bool,
    _output_sparse: bool,
    _order: Literal['C', 'F'],
    _rtol: float,
    _atol: float,
    _equal_nan: bool,
    _n_jobs: Union[int, None]
) -> None:

    """
    Centralized hub for performing parameter validation.
    See the individual modules for more information.


    Parameters
    ----------
    _X:
        {array-like, scipy sparse matrix} of shape (n_samples, n_features)
    _columns:
        Union[Iterable[str], None]
    _degree:
        int
    _min_degree:
        int
    _drop_duplicates:
        bool
    _keep:
        Literal['first', 'last', 'random'],
    _do_not_drop:
        Union[Iterable[str], Iterable[int], None],
    _conflict:
        Literal['raise', 'ignore'],
    _interaction_only:
        bool
    _include_bias:
        bool
    _drop_constants:
        bool
    _output_sparse:
        bool
    _order:
        Literal['C', 'F'],
    _rtol:
        float
    _atol:
        float
    _equal_nan:
        bool
    _n_jobs:
        Union[int, None]


    Return
    ------
    -
        None


    """

    _val_atol(_atol)

    _val_conflict(_conflict)

    _val_keep(_keep)

    _val_X(_X)

    _val_do_not_drop(_do_not_drop, _X, _columns)

    _val_equal_nan(_equal_nan)

    _val_n_jobs(_n_jobs)

    _val_rtol(_rtol)

    _val_degree(_degree)

    _val_min_degree(_min_degree)

    _val_drop_duplicates(_drop_duplicates)

    _val_interaction_only(_interaction_only)

    _val_include_bias(_include_bias)

    _val_drop_constants(_drop_constants)

    _val_output_sparse(_output_sparse)

    _val_order(_order)
















