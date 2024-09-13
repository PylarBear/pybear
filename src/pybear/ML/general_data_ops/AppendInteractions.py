import sys, inspect
import numpy as np, pandas as pd
import sparse_dict as sd
import MemSizes as ms
from debug import get_module_name as gmn
from data_validation import validate_user_input as vui
from general_list_ops import list_select as ls
from MLObjects import MLObject as mlo








class AppendInteractions:

    '''ONLY MODIFIES DATA OBJECT --- USE MLAppendInteractions TO MODIFY DATA AND SUPPORT OBJECT'''

    def __init__(self, DATA, data_given_orientation, bypass_validation=None):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        mem_usage = sys.getsizeof(self.DATA_OBJECT_WIP) / 1024 ** 2
        print(f'\nCURRENT DATA OBJECT IS {["SPARSE DICT" if self.is_dict() else "ARRAY"][0]} OF SIZE {mem_usage} MB.')

        # USER SELECT COLUMNS FOR INTERACTIONS
        INTX_COLUMNS = ls.list_custom_select(self.DATA_OBJECT_HEADER_WIP[0], 'idx')

        print(f'\nGENERATING ESTIMATES OF DATA OBJECT SIZE WITH INTERACTIONS.....')

        # ******************************* FOR ARRAYS & DICTS **************************************************
        if self.is_dict():
            sd_start_rows = sd.inner_len(self.DATA_OBJECT_WIP)
            sd_start_cols = sd.outer_len(self.DATA_OBJECT_WIP)
            _sparsity = sd.sparsity(self.DATA_OBJECT_WIP)
        elif self.is_list():
            sd_start_rows = len(self.DATA_OBJECT_WIP[0])
            sd_start_cols = len(self.DATA_OBJECT_WIP)
            _sparsity = sd.list_sparsity(self.DATA_OBJECT_WIP)

        l_start_rows = int(sd_start_rows)
        l_start_cols = int(sd_start_cols)
        sd_end_rows = int(sd_start_rows)
        sd_end_cols = int(sd_start_cols + len(INTX_COLUMNS) * (len(INTX_COLUMNS) - 1) / 2)
        l_end_rows = int(l_start_rows)
        l_end_cols = int(l_start_cols + len(INTX_COLUMNS) * (len(INTX_COLUMNS) - 1) / 2)

        l_elems = int(l_end_rows * l_end_cols)
        l_start_mem = mem_usage
        l_end_mem = l_start_mem

        # WORST CASE ESTIMATE OF INTX COLUMNS IS ALL DENSE DICTS
        # COPOUT, ASSUME FINAL DATA OBJECT SPARSITY = ORIGINAL OBJECT SPARITY
        sd_elems = int(sd_end_rows * sd_end_cols * (100 - _sparsity) / 100)
        sd_start_mem = mem_usage
        sd_end_mem = sd_start_mem

        np_float = ms.MemSizes('np_float').mb()
        np_int = ms.MemSizes('np_int').mb()
        sd_float = ms.MemSizes('sd_float').mb()
        sd_int = ms.MemSizes('sd_int').mb()
        for col_idx1 in INTX_COLUMNS[:-1]:
            for col_idx2 in INTX_COLUMNS[col_idx1 + 1:]:
                if 'FLOAT' in self.WORKING_SUPOBJS[0][self.msod_mdtype_idx][col_idx1, col_idx2]:
                    l_end_mem += np_float * l_end_rows
                    sd_end_mem += sd_float * sd_end_rows * (100 - _sparsity) / 100
                else:  # IF NEITHER COLUMN IS FLOAT, THEN IT'S EITHER INT OR BIN, SO TYPE IS INT
                    l_end_mem += np_int * l_end_rows
                    sd_end_mem += sd_int * sd_end_rows * (100 - _sparsity) / 100

        # ******************************* END FOR ARRAYS & DICTS **************************************************

        _print = lambda _start_, _rows_, _end_cols_, _elems_, _mem_: print(
            f'{_start_}'.ljust(65) + f' rows={_rows_:,}'.rjust(13) + \
            f' cols={_end_cols_:,}'.rjust(13) + f' elems={_elems_:,}'.rjust(20) + f' mem={int(_mem_):,} MB'.rjust(15))
        _print(f'\nAPPROXIMATE SIZE OF DATA WITH INTERACTIONS AS ARRAYS: ', l_end_rows, l_end_cols, l_elems, l_end_mem)
        _print(f'APPROXIMATE SIZE OF DATA WITH INTERACTIONS AS SPARSE DICTS: ', sd_end_rows, sd_end_cols, sd_elems,
               sd_end_mem)

        __ = vui.validate_user_str(
            f'Proceed(p), Convert to {["ARRAYS" if self.is_dict() else "SPARSE DICTS"][0]}(m), Abort interactions(a)? > ',
            'PMA')
        if __ == 'A':
            self.user_manual_or_std = 'BYPASS'
            continue
        elif __ == 'M':
            if self.is_dict():
                self.DATA_OBJECT_WIP = sd.unzip_to_ndarray(self.DATA_OBJECT_WIP)[0]
            elif self.is_list():
                self.DATA_OBJECT_WIP = sd.zip_list(self.DATA_OBJECT_WIP)

        apply_min_cutoff = vui.validate_user_str(
            f'\nApply min cutoff filter to interactions during expansion? (y/n) > ', 'YN')
        if apply_min_cutoff == 'Y':
            while True:
                intx_min_cutoff = vui.validate_user_int(f'\nEnter min cutoff applied to all > ',
                                                        min=0, max=len(self.DATA_OBJECT_WIP[0]))
                if vui.validate_user_str(f'User entered {intx_min_cutoff}, accept? (y/n) > ', 'YN') == 'Y':
                    break
        else:
            intx_min_cutoff = 0

        print(f'\nGenerating interaction columns...\n')

        for intx_idx in range(len(INTX_COLUMNS) - 1):
            for intx_idx2 in range(intx_idx + 1, len(INTX_COLUMNS)):

                if self.is_list():
                    _dtype1 = self.MODIFIED_DATATYPES[0][INTX_COLUMNS[intx_idx]]
                    _dtype2 = self.MODIFIED_DATATYPES[0][INTX_COLUMNS[intx_idx2]]
                    if _dtype1 == 'FLOAT' or _dtype2 == 'FLOAT':
                        _dtype = np.float64
                    elif _dtype1 == 'INT' or _dtype2 == 'INT':
                        _dtype = np.int32
                    else:  # _dtype1 == 'BIN' and _dtype2 == 'BIN':
                        _dtype = np.int8
                    INTX = np.multiply(self.DATA_OBJECT_WIP[INTX_COLUMNS[intx_idx]].astype(_dtype),
                                       self.DATA_OBJECT_WIP[INTX_COLUMNS[intx_idx2]].astype(_dtype), dtype=_dtype)

                elif self.is_dict():
                    INTX = sd.matrix_multiply({0: self.DATA_OBJECT_WIP[INTX_COLUMNS[intx_idx]]},
                                              {0: self.DATA_OBJECT_WIP[INTX_COLUMNS[intx_idx2]]})
                    INTX[0] = INTX.pop(list(INTX.keys())[0])  # ENSURE KEY FOR INTX IS 0

                # 4-14-22 IF ALL VALUES EQUAL, SKIP (COULD BE DUMMIES FROM THE SAME FEATURE GOING TO ZEROS)
                if self.is_list() and np.min(INTX) == np.max(INTX):
                    continue
                elif self.is_dict() and sd.min_(INTX) == sd.max_(INTX):
                    continue

                # APPLY MIN CUTOFF
                if apply_min_cutoff == 'Y':
                    # 9-23-22 THIS IS CATCHING FLOATxFLOAT AND INTxFLOAT
                    if self.is_list() and np.sum(np.int8(INTX != 0)) < intx_min_cutoff: continue
                    if self.is_dict() and len(
                        INTX[0]) - 1 < intx_min_cutoff: continue  # -1 TO REMOVE PROBABLE PLACEHOLD 0
                intx_header = self.DATA_OBJECT_HEADER_WIP[0][INTX_COLUMNS[intx_idx]] + f'_x_' + \
                              self.DATA_OBJECT_HEADER_WIP[0][INTX_COLUMNS[intx_idx2]]

                # 4-17-22 DONT APPEND A COLUMN THAT IS EQUAL TO ANOTHER COLUMN ALREADY IN

                duplicate = False
                for test_col_idx in range(len(self.DATA_OBJECT_WIP)):
                    if self.is_list() and np.array_equiv(INTX, self.DATA_OBJECT_WIP[test_col_idx]) or \
                            self.is_dict() and sd.core_sparse_equiv(INTX, {0: self.DATA_OBJECT_WIP[test_col_idx]}):
                        duplicate = True
                        print(
                            f'Not adding {intx_header} due to equality with {self.DATA_OBJECT_HEADER_WIP[0][test_col_idx]}')
                        break

                if duplicate: continue

                # IF GET THRU THE OBSTACLE COURSE, APPEND INTX COLUMN TO DATA & HEADER

                self.DATA_OBJECT_HEADER_WIP = np.insert(
                    self.DATA_OBJECT_HEADER_WIP, len(self.DATA_OBJECT_HEADER_WIP[0]), intx_header, axis=1)

                if self.is_list():
                    self.DATA_OBJECT_WIP = np.vstack((self.DATA_OBJECT_WIP, INTX))
                elif self.is_dict():
                    self.DATA_OBJECT_WIP = self.DATA_OBJECT_WIP | {len(self.DATA_OBJECT_WIP): INTX[0]}

                del INTX

                VTYPES = [self.VALIDATED_DATATYPES[0][intx_idx], self.VALIDATED_DATATYPES[0][intx_idx2]]
                if 'FLOAT' in VTYPES:
                    NEW_VTYPE = 'FLOAT'
                elif 'INT' in VTYPES:
                    NEW_VTYPE = 'INT'
                else:
                    NEW_VTYPE = 'STR'
                self.VALIDATED_DATATYPES[0] = np.insert(self.VALIDATED_DATATYPES[0], len(self.VALIDATED_DATATYPES[0]),
                                                        NEW_VTYPE, axis=0)

                MTYPES = [self.MODIFIED_DATATYPES[0][intx_idx], self.MODIFIED_DATATYPES[0][intx_idx2]]
                if 'FLOAT' in MTYPES:
                    NEW_MTYPE = 'FLOAT'
                elif 'INT' in MTYPES:
                    NEW_MTYPE = 'INT'
                else:
                    NEW_MTYPE = 'BIN'
                self.MODIFIED_DATATYPES[0] = np.insert(self.MODIFIED_DATATYPES[0], len(self.MODIFIED_DATATYPES[0]),
                                                       NEW_MTYPE, axis=0)
                self.FILTERING[0] = np.insert(self.FILTERING[0], len(self.FILTERING[0]), '', axis=0)
                self.MIN_CUTOFFS[0] = np.insert(self.MIN_CUTOFFS[0], len(self.MIN_CUTOFFS[0]), intx_min_cutoff, axis=0)
                self.USE_OTHER[0] = np.insert(self.USE_OTHER[0], len(self.USE_OTHER[0]), 'N', axis=0)
                self.START_LAG = np.insert(self.START_LAG, len(self.START_LAG), '', axis=0)
                self.END_LAG = np.insert(self.END_LAG, len(self.END_LAG), '', axis=0)
                self.SCALING = np.insert(self.SCALING, len(self.SCALING), '', axis=0)

        try:
            del VTYPES, MTYPES, NEW_VTYPE, NEW_MTYPE
        except:
            pass

        if vui.validate_user_str(f'Dump DATA to file? (y/n) > ', 'YN') == 'Y':
            basepath = bps.base_path_select()
            filename = fe.filename_wo_extension()

            if self.is_list():
                DF = pd.DataFrame(data=self.DATA_OBJECT_WIP.transpose(), columns=self.DATA_OBJECT_HEADER_WIP[0])
            elif self.is_dict():
                DF = pd.DataFrame(data=sd.sparse_transpose(self.DATA_OBJECT_WIP),
                                  columns=self.DATA_OBJECT_HEADER_WIP[0]).fillna(0)
            pd.DataFrame.to_excel(
                DF,
                excel_writer=basepath + filename + '.xlsx',
                float_format='%.5f',
                startrow=1,
                startcol=1,
                merge_cells=False
            )

        try:
            self.CONTEXT.append(self.interaction_verbage())
            print(f'\n\n\n\n\n\n *** BEAR WHEN THIS STOPS EXCEPTING TAKE THIS OUT *** \n\n\n\n\n\n')
        except:
            self.CONTEXT = self.CONTEXT.tolist()
            self.CONTEXT.append(self.interaction_verbage())

        print(f'\nInteractions complete.')

        if isinstance(self.DATA_OBJECT_WIP, (list, set, tuple, np.ndarray)):
            _cols = len(self.DATA_OBJECT_WIP)
            _rows = len(self.DATA_OBJECT_WIP[0])
        elif isinstance(self.DATA_OBJECT_WIP, dict):
            _cols, _rows = sd.shape_(self.DATA_OBJECT_WIP)

        print(f'\n *** WORKING DATA HAS {_cols} COLUMNS and {_rows} ROWS ***\n')


















