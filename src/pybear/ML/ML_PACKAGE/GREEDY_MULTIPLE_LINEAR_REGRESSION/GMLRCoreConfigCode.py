from data_validation import validate_user_input as vui
from general_data_ops import get_shape as gs


class GMLRCoreConfigCode:
    def __init__(self, gmlr_config, DATA, data_run_orientation, gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method,
                 gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, gmlr_batch_size, gmlr_type, gmlr_score_method,
                 gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg, intcpt_col_idx):

        self.gmlr_config = gmlr_config
        self.DATA = DATA
        self.data_run_orientation = data_run_orientation
        self.conv_kill = gmlr_conv_kill
        self.pct_change = gmlr_pct_change
        self.conv_end_method = gmlr_conv_end_method
        self.rglztn_type = gmlr_rglztn_type
        self.rglztn_fctr = gmlr_rglztn_fctr
        self.batch_method = gmlr_batch_method
        self.batch_size = gmlr_batch_size
        self.gmlr_type = gmlr_type
        self.score_method = gmlr_score_method
        self.float_only = gmlr_float_only
        self.max_columns = gmlr_max_columns
        self.bypass_agg = gmlr_bypass_agg
        self.intcpt_col_idx = intcpt_col_idx


    def return_fxn(self):
        return self.conv_kill, self.pct_change, self.conv_end_method,  self.rglztn_type, self.rglztn_fctr, self.batch_method, \
                self.batch_size, self.gmlr_type, self.score_method, self.float_only, self.max_columns, self.bypass_agg


    def config(self):

        data_rows, data_cols = gs.get_shape('DATA', self.DATA, self.data_run_orientation)

        ####################################################################################################################
        ###################################################################################################################
        if self.gmlr_config in 'BE':    # 'batch method (b)'
            self.batch_method = vui.validate_user_str(f'Select batch method (b)atch (m)inibatch ({data_rows} examples) > ', 'BM')
            if self.batch_method=='B': self.batch_size = data_rows
        ################################################################################################################
        #################################################################################################################

        ####################################################################################################################
        ###################################################################################################################
        if self.gmlr_config in 'CE':    # 'batch size (c)'
            if self.batch_method == 'B':
                print(f'\n*** CANNOT SET batch size WHEN BATCH METHOD IS BATCH(b) ***\n')
            elif self.batch_method == 'M':
                print(f'\nEnter batch size as integer for absolute batch size, enter as decimal < 1 for percentage of full batch ')
                self.batch_size = vui.validate_user_int(
                    f'Select batch size (DATA has {int(data_rows)} examples) > ', min=1, max=int(data_rows))
                # self.batch_size IS str AT THIS POINT
                if float(self.batch_size) == int(self.batch_size): self.batch_size = int(self.batch_size)
                else: self.batch_size = float(self.batch_size)   # IS TRANSLATED FROM % TO # ROWS IN GMLRCoreRunCode
        ################################################################################################################
        #################################################################################################################

        ################################################################################################################
        #################################################################################################################
        if self.gmlr_config in 'DE':  # select type (lazy / forward / backward) (d)
            OPTION_DICT = {'L': 'lazy', 'F': 'forward', 'B': 'backward'}
            option_txt = ", ".join([f'{v}({k.lower()})' for k,v in OPTION_DICT.items()])
            self.gmlr_type = vui.validate_user_str(f'\nSelect GMLR type {option_txt} ({data_cols} columns) > ',
                                                   ''.join(OPTION_DICT.keys()))
            del OPTION_DICT, option_txt
        ####################################################################################################################
        ###################################################################################################################

        ################################################################################################################
        #################################################################################################################
        if self.gmlr_config in 'FE':  # select score method (F, RSQ, RSQ-adj, r) (f)
            # LAZY CAN ONLY USE RSQ; FORWARD CAN USE RSQ, RSQ ADJ, OR F; BACKWARD CAN ONLY USE RSQ ADJ OR F
                if self.gmlr_type=='L':
                    print(f'\n*** LAZY GMLR SCORE METHOD CAN ONLY BE RSQ ***\n')
                    self.score_method = 'Q'
                else:
                    OPTION_DICT = {'F': 'F-score', 'Q': 'RSQ', 'A': 'adj RSQ', 'R': 'r'}
                    # CURRENTLY UNABLE TO CALCULATE r AND ADJ RSQ FOR NO-INTERCEPT STYLE REGRESSION
                    if not self.intcpt_col_idx is None:
                        if self.gmlr_type=='F': allowed = ['Q','A','F']
                        elif self.gmlr_type=='B': allowed = ['A','F']
                    elif self.intcpt_col_idx is None:
                        if self.gmlr_type=='F': allowed = ['Q','F']
                        elif self.gmlr_type=='B': allowed = ['F']
                    option_txt = ", ".join([f'{v}({k.lower()})' for k,v in OPTION_DICT.items() if k in allowed])
                    self.score_method = vui.validate_user_str(f'\nSelect score optimization method {option_txt} > ', ''.join(allowed))
                    del OPTION_DICT, allowed, option_txt
        ####################################################################################################################
        ###################################################################################################################

        ####################################################################################################################
        ####################################################################################################################
        # CANNOT BYPASS AGGLOMERATIVE GMLR IF self.gmlr_type IS FORWARD OR BACKWARD BECAUSE ARE AGGLOMERATIVE
        if self.gmlr_config in 'GE':  # bypass agglomerative MLR(g)
            if self.gmlr_type == 'L':
                self.bypass_agg = {'Y': True, 'N': False}[vui.validate_user_str(f'\nBypass agglomerative MLR? (y/n) > ', 'YN')]
            elif self.gmlr_type in ['F', 'B']:
                self.bypass_agg = False
        ####################################################################################################################
        ####################################################################################################################

        ####################################################################################################################
        ####################################################################################################################

        # adjust convergence kill(h) --- THIS ONLY APPLIES TO LAZY W FULL AGG, FORWARD, & BACKWARD, BUT NOT LAZY W/O AGG
        if self.gmlr_config in 'HE':
            if not (self.gmlr_type=='L' and self.bypass_agg is True):
                print(f'\nCurrent convergence end method: {self.conv_end_method}')
                self.conv_end_method = dict({'K':'KILL', 'P':'PROMPT', 'N':None})[
                    vui.validate_user_str(f'Enter new convergence end method - kill(k), prompt(p), none(n) > ', 'KPN')]

                if self.conv_end_method is None:
                    self.conv_kill = float('inf')
                elif not self.conv_end_method is None:
                    print(f'\nCurrent iterations with no improvement to kill: {self.conv_kill}')
                    self.conv_kill = vui.validate_user_int('Enter iterations until kill > ', min=1, max=1e12)

                if self.pct_change is None:
                    self.pct_change = float('inf')
                elif not self.pct_change is None:
                    print(f'\nCurrent min % change to avert kill: {self.pct_change}')
                    self.pct_change = vui.validate_user_float(f'Enter min % change required to avert kill > ', min=0, max=100)
            else:
                self.conv_end_method = None
                self.conv_kill = float('inf')
                self.pct_change = float('inf')

        ####################################################################################################################
        ####################################################################################################################

        ####################################################################################################################
        ###################################################################################################################
        if self.gmlr_config in 'JKE':    # 'regularization type (j)'
            if self.gmlr_config in 'JE':
                print(f'\n*** RIDGE IS CURRENTLY THE ONLY REGULARIZATION AVAILABLE FOR GMLR ***\n')
                self.rglztn_type = dict({'R':'RIDGE', 'N':None})[
                vui.validate_user_str(f'Select reqularization type RIDGE(r), None(n) > ', 'RN')]
                if self.rglztn_type is None: self.rglztn_fctr = 0
            if self.gmlr_config in 'KE':  # 'regularization factor (k)'
                if self.rglztn_type is None:
                    self.rglztn_fctr = 0
                else:
                    self.rglztn_fctr = vui.validate_user_float(f'Enter regularization factor > ', min=0, max=1e12)
                    self.rglztn_type = 'RIDGE'
        ################################################################################################################
        #################################################################################################################

        ####################################################################################################################
        ####################################################################################################################
        if self.gmlr_config in 'LE':  # toggle float_only(l)
            self.float_only = {'Y': True, 'N': False}[vui.validate_user_str(f'\nUse FLOAT columns only? (y/n) > ', 'YN')]
        ####################################################################################################################
        ####################################################################################################################

        ####################################################################################################################
        ####################################################################################################################
        if self.gmlr_config in 'ME':  # max columns(m)
            self.max_columns = vui.validate_user_int(f'\nEnter max columns (of {data_cols}) > ', min=1 + [1 if self.intcpt_col_idx is not None else 0][0])
        ####################################################################################################################
        ####################################################################################################################


        del data_rows, data_cols

        return self.return_fxn()






















if __name__ == '__main__':

    # TEST MODULE  --- MODULE AND TEST VERIFIED GOOD 5/24/2023

    import numpy as np

    _rows = 6
    _cols = 4

    standard_config = 'BYPASS'
    gmlr_config = 'BYPASS'
    DATA = np.arange(_rows * _cols, dtype=np.int32).reshape((_cols, _rows))
    data_run_orientation = 'COLUMN'
    intcpt_col_idx = None

    def seed_values():
        gmlr_conv_kill = None
        gmlr_pct_change = float('inf')
        gmlr_conv_end_method = 'KILL'
        gmlr_rglztn_type = 'RIDGE'
        gmlr_rglztn_fctr = 100
        gmlr_batch_method = 'B'
        gmlr_batch_size = int(1e12)
        gmlr_type = 'Q'
        gmlr_score_method = 'R'
        gmlr_float_only = True
        gmlr_max_columns = 10
        gmlr_bypass_agg = False


        return gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
                gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg




    gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
        gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg = \
        seed_values()


    # TEST THAT BATCH SIZE IS ONLY AVAILABLE IF BATCH METHOD IS MINIBATCH #################################################
    __ = input(f'\n*** TEST THAT BATCH SIZE IS ONLY AVAILABLE IF BATCH METHOD IS MINIBATCH ***\n')
    for itr, gmlr_config in enumerate(('BE','CE', 'BE', 'CE')):
        if itr == 0: __ = input(f'\nSelect FULL BATCH at prompt --- hit enter ')
        elif itr == 1: __ = input(f'\nShould not allow ability to change batch size --- hit enter ')
        elif itr == 2: __ = input(f'\nSelect MINIBATCH at prompt --- hit enter ')
        elif itr == 3: __ = input(f'\nShould allow ability to change batch size --- hit enter ')
        gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
        gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg = \
            GMLRCoreConfigCode(gmlr_config, DATA, data_run_orientation, gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method,
                           gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, gmlr_batch_size, gmlr_type,
                           gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg, intcpt_col_idx).config()

        if itr in [1, 3]:
            __ = input(f'batch_method = {gmlr_batch_method}, batch_size = {gmlr_batch_size} '
                   f'(should be {_rows if itr==1 else gmlr_batch_size}) --- hit enter')

            if not gmlr_batch_size == [_rows if itr==1 else gmlr_batch_size][0]:
                raise Exception(f'EXPECTED gmlr_batch_size ({_rows if itr==1 else gmlr_batch_size}) DOES NOT EQUAL ACTUAL ({gmlr_batch_size})')


    # END TEST THAT BATCH SIZE IS ONLY AVAILABLE IF BATCH METHOD IS NOT NONE #################################################

    gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
        gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg = \
        seed_values()

    # TEST THAT LAZY AUTO-SELECTS RSQ, FORWARD ALLOWS RSQ, ADJRSQ, OR F, AND BACKWARD ALLOWS ADJRSQ OR F ###############
    __ = input(f'\n*** TEST THAT LAZY AUTO-SELECTS RSQ, FORWARD ALLOWS RSQ, ADJRSQ, OR F, AND BACKWARD ALLOWS ADJRSQ OR F ***\n')
    for itr, gmlr_config in enumerate(('DE', 'FE', 'DE', 'FE', 'DE', 'FE')):
        if itr == 0: __ = input(f'\nSelect LAZY at prompt --- hit enter ')
        elif itr == 1: __ = input(f'\nShould autoselect RSQ --- hit enter ')
        elif itr == 2: __ = input(f'\nSelect FORWARD at prompt --- hit enter ')
        elif itr == 3: __ = input(f'\nShould allow select of RSQ, ADJ RSQ, or F --- hit enter ')
        elif itr == 4: __ = input(f'\nSelect BACKWARD at prompt --- hit enter ')
        elif itr == 5: __ = input(f'\nShould allow select of ADJ RSQ or F --- hit enter ')
        gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
        gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg = \
            GMLRCoreConfigCode(gmlr_config, DATA, data_run_orientation, gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method,
                           gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, gmlr_batch_size, gmlr_type,
                           gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg, intcpt_col_idx).config()

        if itr in [1, 3, 5]:
            __ = input(f'\ngmlr_score_method = {gmlr_score_method} --- hit Enter')
        if itr == 1 and gmlr_score_method != 'Q':
            raise Exception(f"EXPECTED gmlr_score_method for LAZY (Q) DOES NOT EQUAL ACTUAL ({gmlr_score_method}) ")
    # END TEST THAT LAZY AUTO-SELECTS RSQ, FORWARD ALLOWS RSQ, ADJRSQ, OR F, AND BACKWARD ALLOWS ADJRSQ OR F ###############

    gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
        gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg = \
        seed_values()


    # TEST THAT AGGLOMERATIVE CAN ONLY BE BYPASSED FOR LAZY ##################################################################
    __ = input(f'\n*** TEST THAT AGGLOMERATIVE CAN ONLY BE BYPASSED FOR LAZY ***')
    for itr, gmlr_config in enumerate(('DE','GE','DE','GE','DE','GE')):
        if itr == 0: __ = input(f'\nSelect LAZY at prompt -- hit enter')
        elif itr == 1: __ = input(f'\nShould prompt for AGG --- hit enter ')
        elif itr == 2: __ = input(f'\nSelect FORWARD at prompt -- hit enter')
        elif itr == 3: __ = input(f'\nShould not prompt for AGG and is set to False --- hit enter ')
        elif itr == 4: __ = input(f'\nSelect BACKWARD at prompt -- hit enter')
        elif itr == 5: __ = input(f'\nShould not prompt for AGG and is set to False --- hit enter ')
        gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
        gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg = \
            GMLRCoreConfigCode(gmlr_config, DATA, data_run_orientation, gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method,
                           gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, gmlr_batch_size, gmlr_type,
                           gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg, intcpt_col_idx).config()

        if itr in [1, 3, 5]:
            __ = input(f'\nbypass_agg = {gmlr_bypass_agg} --- hit enter')
            if itr in [3,5] and gmlr_bypass_agg is not False:
                raise Exception(f'gmlr_bypass_agg SHOULD BE False FOR FORWARD AND BACKWARDS')
    # END TEST THAT AGGLOMERATIVE CAN ONLY BE BYPASSED FOR LAZY ##################################################################

    gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
        gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg = \
        seed_values()

    # TEST THAT CONV KILL ONLY APPLIES TO LAZY W FULL AGG, FORWARD, & BACKWARD, BUT NOT LAZY ONLY ############################
    __ = input(f'\n*** TEST THAT CONV KILL ONLY APPLIES TO LAZY W FULL AGG, FORWARD, & BACKWARD, BUT NOT LAZY ONLY ***')
    for itr, gmlr_config in enumerate(('DE','GE','HE', 'DE','HE', 'DE','HE', 'DE','GE','HE')):
        if itr == 0: __ = input(f'\nSelect LAZY at prompt --- hit enter ')
        elif itr == 1: __ = input(f'\nSelect "N" ( DO NOT BYPASS AGG ) at prompt --- hit enter ')
        elif itr == 2: __ = input(f'\nShould prompt for conv kill et al   --- hit enter ')
        elif itr == 3: __ = input(f'\nSelect FORWARD at prompt --- hit enter ')
        elif itr == 4: __ = input(f'\nShould prompt for conv kill et al   --- hit enter ')
        elif itr == 5: __ = input(f'\nSelect BACKWARD at prompt --- hit enter ')
        elif itr == 6: __ = input(f'\nShould prompt for conv kill et al   --- hit enter ')
        elif itr == 7: __ = input(f'\nSelect LAZY at prompt --- hit enter ')
        elif itr == 8: __ = input(f'\nSelect "Y" ( BYPASS AGG ) at prompt --- hit enter ')
        elif itr == 9: __ = input(f'\nShould not prompt for any conv kill   --- hit enter ')
        gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
        gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg = \
            GMLRCoreConfigCode(gmlr_config, DATA, data_run_orientation, gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method,
                           gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, gmlr_batch_size, gmlr_type,
                           gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg, intcpt_col_idx).config()

        if itr in [2, 4, 6, 9]:
            __ = input(f'\ngmlr_conv_kill = {gmlr_conv_kill}, gmlr_pct_change = {gmlr_pct_change}, '
                       f'gmlr_conv_end_method = {gmlr_conv_end_method} --- hit enter')
    # END TEST THAT CONV KILL ONLY APPLIES TO LAZY W FULL AGG, FORWARD, & BACKWARD, BUT NOT LAZY ONLY ############################

    gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
        gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg = \
        seed_values()

    # TEST THAT REGULARIZATION CAN BE SET IF rglztn_type IS NOT None, AND DEFAULTS TO ZERO IF None ##############################
    __ = input(f'\n*** TEST THAT REGULARIZATION CAN BE SET IF rglztn_type IS NOT None, AND DEFAULTS TO ZERO IF None ***\n')
    for itr, gmlr_config in enumerate(('J','K','J','K')):
        if itr == 0: __ = input(f'\nSelect RIDGE at prompt --- hit enter ')
        elif itr == 1: __ = input(f'\nShould prompt for regularization factor --- hit enter ')
        elif itr == 2: __ = input(f'\n Select None at prompt --- hit enter ')
        elif itr == 3: __ = input(f'\nShould not prompt and auto-set regularization factor to zero --- hit enter ')

        gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
        gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg = \
            GMLRCoreConfigCode(gmlr_config, DATA, data_run_orientation, gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method,
                           gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, gmlr_batch_size, gmlr_type,
                           gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg, intcpt_col_idx).config()

        if itr in [1,3]:
            __ = input(f'\ngmlr_rglztn_type = {gmlr_rglztn_type}, gmlr_rglztn_fctr = {gmlr_rglztn_fctr}')

    # END TEST THAT REGULARIZATION CAN BE SET IF rglztn_type IS NOT None, AND DEFAULTS TO ZERO IF None ##############################

    gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
        gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg = \
        seed_values()






































