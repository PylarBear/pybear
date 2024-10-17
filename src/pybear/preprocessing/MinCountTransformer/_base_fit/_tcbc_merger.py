# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from copy import deepcopy
import numpy as np
import pandas as pd



def _tcbc_merger(
    _DTYPE_UNQS_CTS_TUPLES: list[tuple[str, dict[any, int]]],
    _tcbc: dict[int, dict[any, int]]
):

    """
    Combine the results in the unq_cts dictionary from the current
    {partial_}fit with the unqs/cts results from previous fits that are
    in the total_counts_by_column dictionary. If the column does not
    exist in totat_counts_by_column, add the entire unq_cts dict in that
    slot. For columns that already exist, if a unique is not in that
    column of total_counts_by_column, add the unique and its count. If
    the unique already exists in that column, add the current count to
    the old count.

    This module was originally created to ease diagnosis and fixing of
    problems with nans getting multiple entries in a single column of
    total_counts_by_column when combining new unq/cts results from
    _DTYPE_UNQS_CTS_TUPLES into the existing column in
    total_counts_by_column.


    Parameters
    ----------
    _DTYPE_UNQS_CTS_TUPLES:
        list[tuple[str, dict[any, int]]] - a list of tuples, where each
        tuple holds (dtype, unq_ct_dict) for each column in the current
        {partial_}fit.
    _tcbc:
        dict[int, dict[any, int]] - total_counts_by_column dictionary,
        outer keys are the column index of the data, values are dicts
        with keys that are the uniques in that column, and the values are
        the frequency.


    Return
    ------
    -
        _tcbc:
            dict[int, dict[any, int]] - total_counts_by_column dictionary
            updated with the uniques and counts in _DTYPE_UNQS_CTS_TUPLES.

    """

    # this is important because of back-talk
    __tcbc = deepcopy(_tcbc)

    for col_idx, (_dtype, UNQ_CT_DICT) in enumerate(_DTYPE_UNQS_CTS_TUPLES):

        if col_idx not in __tcbc:
            __tcbc[col_idx] = UNQ_CT_DICT
        else:



            # reconstruct tcbc[col_idx] in a copy
            # remove any nans from unqs, but get the count
            _tcbc_nan_symbol = None
            _tcbc_nan_ct = 0
            _tcbc_col_dict = {}
            for k,v in __tcbc[col_idx].items():
                if str(k) in ['nan', 'NAN', 'NaN', '<NA>']:
                    _tcbc_nan_symbol = k
                    _tcbc_nan_ct = v
                else:
                    _tcbc_col_dict[k] = v

            # reconstruct UNQ_CT_DICT in a copy
            # remove any nans from unqs, but get the count
            _ucd_nan_symbol = None
            _ucd_nan_ct = 0
            _ucd_col_dict = {}
            for k,v in UNQ_CT_DICT.items():
                if str(k) in ['nan', 'NAN', 'NaN', '<NA>']:
                    _ucd_nan_symbol = k
                    _ucd_nan_ct = v
                else:
                    _ucd_col_dict[k] = v

            # merge the two reconstructed dicts w/o nans
            for k in (_tcbc_col_dict | _ucd_col_dict).keys():
                if k in _tcbc_col_dict and k in _ucd_col_dict:
                    _tcbc_col_dict[k] += _ucd_col_dict[k]
                elif k not in _tcbc_col_dict:   # only in _ucd_col_dict
                    _tcbc_col_dict[k] = _ucd_col_dict[k]
                # elif k in _tcbc_col_dict and k not in _ucd_col_dict:
                #     tcbc count does not change

            # merge the nans
            if _tcbc_nan_symbol and _ucd_nan_symbol:
                _tcbc_col_dict[_tcbc_nan_symbol] = (_tcbc_nan_ct + _ucd_nan_ct)
            elif _tcbc_nan_symbol and not _ucd_nan_symbol:
                _tcbc_col_dict[_tcbc_nan_symbol] = _tcbc_nan_ct
            elif not _tcbc_nan_symbol and _ucd_nan_symbol:
                _tcbc_col_dict[_ucd_nan_symbol] = _ucd_nan_ct
            else:
                # no nans in either
                pass

            __tcbc[col_idx] = _tcbc_col_dict

    return __tcbc















