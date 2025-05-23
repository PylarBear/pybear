# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._partial_fit._parallel_column_comparer import _parallel_column_comparer



def _chunk_comparer(_chunk, _COLUMN, _rtol, _atol, _equal_nan):

    _poly_dupls = []
    for _c_idx in range(_chunk.shape[1]):

        _poly_dupls.append(
            _parallel_column_comparer(_chunk[:, _c_idx], _COLUMN, _rtol, _atol, _equal_nan)
        )


    return _poly_dupls





