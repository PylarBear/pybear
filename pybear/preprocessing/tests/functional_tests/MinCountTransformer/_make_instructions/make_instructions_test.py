# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest


from typing import Union
from copy import deepcopy
import numpy as np

from preprocessing.MinCountTransformer._type_aliases import (
    TotalCountsByColumnType,
    InstructionsType,
    OriginalDtypesDtype,
)

def _make_instructions(
        _count_threshold: int,
        _ignore_float_columns: bool,
        _ignore_non_binary_integer_columns: bool,
        _ignore_columns: [list[Union[int, str]], callable, None],
        _ignore_nan: bool,
        _handle_as_bool: [list[Union[int, str]], callable, None],
        _delete_axis_0: bool,
        _original_dtypes: OriginalDtypesDtype,
        _n_features_in: int,
        _total_counts_by_column: TotalCountsByColumnType,
        _threshold: int = None
    ) -> InstructionsType:

    """
    Convert compiled uniques and frequencies into instructions for
    transforming data based on given parameters.

    Parameters
    ----------
    _count_threshold:
        int
    _ignore_float_columns:
        bool
    _ignore_non_binary_integer_columns:
        bool
    _ignore_columns:
        [list[Union[int, str]], callable, None],
    _ignore_nan:
        bool
    _handle_as_bool:
        [list[Union[int, str]], callable, None]
    _delete_axis_0:
        bool
    _original_dtypes:
        OriginalDtypesDtype
    _n_features_in:
        int
    _total_counts_by_column:
        TotalCountsByColumnType

    _threshold:
        int

    Returns
    ----------
    -
        _delete_instr: dict[pizza, pizza]



    """
    # _make_instructions()
    #   CALLED BY get_support(), test_threshold(), AND transform()
    # get_support()
    #   CALLED BY get_feature_names(), inverse_transform() AND transform()


    _threshold = _threshold or _count_threshold

    # pizza _make_instructions_validation

    _delete_instr = {}
    for col_idx in range(_n_features_in):
        _delete_instr[col_idx] = []
        if col_idx in _ignore_columns:
            _delete_instr[col_idx].append('INACTIVE')
        elif _total_counts_by_column[col_idx] == {}:
            _delete_instr[col_idx].append('INACTIVE')
        elif (_ignore_float_columns and
              _original_dtypes[col_idx] == 'float'):
            _delete_instr[col_idx].append('INACTIVE')
        elif (_ignore_non_binary_integer_columns and
              _original_dtypes[col_idx] == 'int'):
            UNQS = np.fromiter(_total_counts_by_column[col_idx].keys(),
                               dtype=np.float64
                               )
            if len(UNQS[np.logical_not(np.isnan(UNQS))]) >= 3:
                _delete_instr[col_idx].append('INACTIVE')
            del UNQS

    for col_idx, COLUMN_UNQ_CT_DICT in _total_counts_by_column.items():
        # _total_counts_by_column GIVES A DICT OF UNQ & CTS FOR COLUMN;
        # IF _ignore_nan, nans & THEIR CTS ARE TAKEN OUT BELOW. IF nan IS
        # IN, THIS COMPLICATES ASSESSMENT OF COLUMN HAS 1 VALUE, IS BINARY, ETC.

        if _delete_instr[col_idx] == ['INACTIVE']:
            continue

        COLUMN_UNQ_CT_DICT = deepcopy(COLUMN_UNQ_CT_DICT)

        # vvv MANAGE nan ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # GET OUT nan INFORMATION IF nan IS IN
        if _ignore_nan:
            _nan_key = False
            _nan_ct = False
        else:
            _nan_dict = {k: v for k, v in COLUMN_UNQ_CT_DICT.items() if str(k).lower() == 'nan'}
            if len(_nan_dict) == 0:
                _nan_key = False
                _nan_ct = False
            else:
                _nan_key = list(_nan_dict.keys())[0]
                _nan_ct = list(_nan_dict.values())[0]

            del _nan_dict

        # TEMPORARILY REMOVE nan FROM COLUMN_UNQ_CT_DICT, WHETHER IGNORING OR NOT
        COLUMN_UNQ_CT_DICT = {unq: ct for unq, ct in COLUMN_UNQ_CT_DICT.items()
                              if str(unq).lower() != 'nan'}
        # ^^^ END MANAGE nan ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        if len(COLUMN_UNQ_CT_DICT) == 1:  # SAME VALUE IN THE WHOLE COLUMN
            if not _nan_ct:  # EITHER IGNORING NANS OR NONE IN FEATURE
                # DONT DELETE IMPACTED ROWS NO MATTER WHAT THE DTYPE WAS
                # OR WHAT kwargs WERE GIVEN, WOULD DELETE ALL ROWS
                _delete_instr[col_idx].append('DELETE COLUMN')
            else:  # HAS NANS AND NOT IGNORING
                # STILL JUST DELETE THE COLUMN
                _delete_instr[col_idx].append('DELETE COLUMN')

        elif len(COLUMN_UNQ_CT_DICT) == 2:  # BINARY, ANY DTYPE
            # handle_as_bool DOESNT MATTER HERE
            _dtype = _original_dtypes[col_idx]
            if not _nan_ct:  # EITHER IGNORING NANS OR NONE IN FEATURE
                _ctr = 0
                for unq, ct in COLUMN_UNQ_CT_DICT.items():
                    if ct < _threshold:
                        _ctr += 1
                        if _dtype != 'int' or _delete_axis_0:
                            _delete_instr[col_idx].append(unq)
                if _ctr > 0:
                    _delete_instr[col_idx].append('DELETE COLUMN')
                del _ctr, unq, ct
            else:  # HAS NANS AND NOT IGNORING
                if _dtype != 'int' or _delete_axis_0:
                    COLUMN_UNQ_CT_DICT[_nan_key] = _nan_ct
                    non_nan_ctr = 0
                    for unq, ct in COLUMN_UNQ_CT_DICT.items():
                        if ct < _threshold:
                            if str(unq).lower() != 'nan':
                                non_nan_ctr += 1
                            _delete_instr[col_idx].append(unq)
                    if non_nan_ctr > 0:
                        _delete_instr[col_idx].append('DELETE COLUMN')
                    del non_nan_ctr, unq, ct
                else:  # elif _dtype == 'int' and not delete_axis_0
                    # nan IS NOT PUT BACK IN
                    if min(list(COLUMN_UNQ_CT_DICT.values())) < _threshold:
                        _delete_instr[col_idx].append('DELETE COLUMN')
                    # WHAT TO DO IF VALID INT VALUES' FREQS ARE >= THRESH
                    # BUT nan FREQ < THRESH AND NOT delete_axis_0?
                    # AS OF 24_03_23_19_10_00 LEAVING nans IN DESPITE
                    # BREAKING THRESHOLD

            del _dtype

        else:  # 3+ UNIQUES NOT INCLUDING nan
            # CANT BE A BINARY INTEGER
            # IF HANDLING AS BOOL, ONLY NEED TO KNOW WHAT IS NON-ZERO AND
            # IF ROWS WILL BE DELETED OR KEPT
            if not _nan_ct:  # EITHER IGNORING NANS OR NONE IN FEATURE
                if _handle_as_bool is not None and col_idx in _handle_as_bool:

                    if _original_dtypes[col_idx] == 'obj':
                        raise ValueError(f"handle_as_bool on str column")

                    UNQS = np.fromiter(COLUMN_UNQ_CT_DICT.keys(),
                                       dtype=np.float64
                                       )
                    CTS = np.fromiter(COLUMN_UNQ_CT_DICT.values(),
                                      dtype=np.float64
                                      )
                    NON_ZERO_MASK = UNQS.astype(bool)
                    NON_ZERO_UNQS = UNQS[NON_ZERO_MASK]
                    total_zeros = CTS[np.logical_not(NON_ZERO_MASK)].sum()
                    total_non_zeros = CTS[NON_ZERO_MASK].sum()
                    del NON_ZERO_MASK
                    if _delete_axis_0:
                        if total_zeros < _threshold:
                            _delete_instr[col_idx].append(0)
                        if total_non_zeros < _threshold:
                            for k in NON_ZERO_UNQS:
                                _delete_instr[col_idx].append(k)
                            del k

                    if (total_zeros < _threshold) or \
                            (total_non_zeros < _threshold):
                        _delete_instr[col_idx].append('DELETE COLUMN')

                    del UNQS, CTS, NON_ZERO_UNQS, total_zeros, total_non_zeros
                else:  # EVERYTHING EXCEPT _handle_as_bool
                    # _delete_axis_0 NO LONGER APPLIES,
                    # MUST DELETE ALONG AXIS 0
                    # IF ONLY 1 UNQ LEFT, DELETE COLUMN,
                    # IF ALL UNQS DELETED THIS SHOULD PUT ALL False IN
                    # ROW MASK AND CAUSE EXCEPT DURING transform()
                    for unq, ct in COLUMN_UNQ_CT_DICT.items():
                        if ct < _threshold:
                            _delete_instr[col_idx].append(unq)
                    if len(_delete_instr[col_idx]) == len(
                            COLUMN_UNQ_CT_DICT) - 1:
                        _delete_instr[col_idx].append('DELETE COLUMN')

                    del unq, ct
            else:  # HAS NANS AND NOT IGNORING
                if _handle_as_bool is not None and col_idx in _handle_as_bool:

                    if _original_dtypes[col_idx] == 'obj':
                        raise ValueError(f"handle_as_bool on str column")

                    # bool(np.nan) GIVES True, DONT USE IT!
                    # LEAVE nan OUT TO DETERMINE KEEP/DELETE COLUMN
                    # REMEMBER THAT nan IS ALREADY OUT OF COLUMN_UNQ_CT_DICT
                    # AND STORED SEPARATELY, USE _nan_key & _nan_ct
                    UNQS = np.fromiter(COLUMN_UNQ_CT_DICT.keys(), dtype=np.float64)
                    CTS = np.fromiter(COLUMN_UNQ_CT_DICT.values(), dtype=np.float64)
                    NON_ZERO_MASK = UNQS.astype(bool)
                    NON_ZERO_UNQS = UNQS[NON_ZERO_MASK]
                    total_non_zeros = CTS[NON_ZERO_MASK].sum()
                    total_zeros = CTS[np.logical_not(NON_ZERO_MASK)].sum()
                    del NON_ZERO_MASK

                    if _delete_axis_0:
                        if total_zeros < _threshold:
                            _delete_instr[col_idx].append(0)
                        if total_non_zeros < _threshold:
                            for k in NON_ZERO_UNQS:
                                _delete_instr[col_idx].append(k)
                                del k
                        if _nan_ct < _threshold:
                            _delete_instr[col_idx].append(_nan_key)

                    # IF _nan_ct < _threshold but not delete_axis_0
                    # AND NOT DELETE COLUMN THEY WOULD BE KEPT DESPITE
                    # BREAKING THRESHOLD
                    if total_zeros < _threshold or total_non_zeros < _threshold:
                        _delete_instr[col_idx].append('DELETE COLUMN')

                    del UNQS, CTS, NON_ZERO_UNQS, total_zeros, total_non_zeros
                else:
                    # _delete_axis_0 NO LONGER APPLIES,
                    # MUST DELETE ALONG AXIS 0
                    # IF ONLY 1 UNQ LEFT, DELETE COLUMN,
                    # IF ALL UNQS DELETED THIS SHOULD PUT ALL False IN
                    # ROW MASK AND CAUSE EXCEPT DURING transform()
                    COLUMN_UNQ_CT_DICT[_nan_key] = _nan_ct
                    for unq, ct in COLUMN_UNQ_CT_DICT.items():
                        if ct < _threshold:
                            _delete_instr[col_idx].append(unq)
                    if len(_delete_instr[col_idx]) == len(COLUMN_UNQ_CT_DICT) - 1:
                        _delete_instr[col_idx].append('DELETE COLUMN')
                    del unq, ct

    del _threshold
    try:
        del col_idx, COLUMN_UNQ_CT_DICT, _nan_key, _nan_ct
    except:
        pass

    return _delete_instr


































