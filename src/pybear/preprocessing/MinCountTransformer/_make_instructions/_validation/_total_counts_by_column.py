# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ..._type_aliases import TotalCountsByColumnType
import numpy as np



def _val_total_counts_by_column(
    _total_counts_by_column: TotalCountsByColumnType
) -> None:

    """
    Validate _total_counts_by_column is dict, outer keys are integer >= 0,
    values are dict with data values as keys and counts (integers >= 0)
    as values.

    """


    err_msg = (f"'_total_counts_by_column' must be a dictionary of dictionaries, \n"
               f"with positive integer keys and dictionaries as values.")

    __ = _total_counts_by_column

    # must be dict
    if not isinstance(__, dict):
        raise TypeError(err_msg)

    # outer keys must be zero-indexed contiguous integer
    try:
        map(float, __)
        if any([isinstance(_, bool) for _ in __]):
            raise Exception
        if not all([int(_)==_ for _ in __]):
            raise Exception
    except:
        raise TypeError(err_msg)

    if not np.array_equiv(list(range(len(__))), list(__.keys())):
        raise ValueError(err_msg)

    del err_msg

    err_msg = (f"_total_counts_by_column values must be dictionaries keyed with "
               f"data values and have integer values (counts)")

    # inner objects must be dict
    if not all(map(isinstance, __.values(), (dict for i in __.values()))):
        raise TypeError(err_msg)

    del err_msg

    err_msg = (f"_total_counts_by_column inner dictionaries must be keyed "
               f"with data values (DataType)")

    # inner key (DataType) must be non-iterable
    for _outer_key in __:
        for _inner_key in __[_outer_key]:
            try:
                iter(list(_inner_key))
                if not isinstance(_inner_key, str):
                    raise UnicodeError
            except UnicodeError:
                raise TypeError(err_msg)
            except:
                pass

    del err_msg
    try:
        del _outer_key, _inner_key
    except:
        pass


    # inner values must be int
    _inner_values = list(map(dict.values, __.values()))

    err_msg = (f"_total_counts_by_column inner dictionaries' counts "
               f"must be integers")

    if any(map(
            lambda x: any(map(isinstance, x, (bool for _ in x))),
            _inner_values
    )):
        raise TypeError(err_msg)
    if not all(map(
            lambda x: all(map(isinstance, x, (int for _ in x))),
            _inner_values
    )):
        raise TypeError(err_msg)

    # inner values must be >= 0
    if not all([v >= 0 for COLUMN_VALUES in _inner_values for v in COLUMN_VALUES]):
        raise ValueError(f"all unique value counts must be >= 0")

    del _inner_values


































