import numpy as n, pandas as pd
from copy import deepcopy
from data_validation import validate_user_input as vui
from ML_PACKAGE import MLConfigRunTemplate as mlcrt
from ML_PACKAGE.SVD import SVDConfig as sic, SVDRun as sir
from general_list_ops import list_select as ls
from ML_PACKAGE.GENERIC_PRINT import DictMenuPrint as dmp


# INHERITS #############################################################################################################
# dataclass_mlo_loader()             Loads an instance of MLObject for DATA.
# intercept_manager()                Locate columns of constants in DATA & handle. As of 11/15/22 only for MLR, MI, and GMLR.
# insert_intercept()                 Insert a column of ones in the 0 index of DATA.
# delete_column()                    Delete a column from DATA and respective holder objects.
# run_module_input_tuple()           tuple of base params that pass into run_module for all ML packages
# base_post_run_options_module()     holds post run options applied to all ML packages
# return_fxn_base()                  values returned from all ML packages
# configrun()                        runs config_module() & run_module()

# OVERWRITES ###########################################################################################################
# config_module()                    gets configuration source, returns configuration parameters for particular ML package
# run_module()                       returns run module for particular ML package
# return_fxn()                       returns user-specified output, in addition to return_fxn_base()
# sub_post_run_cmds()                package-specific options available to modify WORKING_DATA after run
# sub_post_run_options_module()      holds post run options unique to particular ML package

# UNIQUE ###############################################################################################################
# float_remove_verbage()             verbage put into CONTEXT when floats are removed from DATA
# float_reappend_verbage()           verbage put into CONTEXT when floats are re-appended to DATA

#CALLED BY ML
class SVDConfigRun(mlcrt.MLConfigRunTemplate):
    def __init__(self, standard_config, svd_config, SUPER_RAW_NUMPY_LIST, SUPER_WORKING_NUMPY_LIST, WORKING_VALIDATED_DATATYPES,
                 WORKING_MODIFIED_DATATYPES, WORKING_FILTERING, WORKING_MIN_CUTOFFS, WORKING_USE_OTHER, WORKING_CONTEXT,
                 WORKING_KEEP, WORKING_SCALING, split_method, LABEL_RULES, number_of_labels, event_value, negative_value,
                 svd_max_columns):

        # DUMMIES TO SATISIFY __init__ OF MLConfigRunTemplate, OTHERWISE NOT NEEDED IN SVD
        conv_kill, pct_change, conv_end_method, rglztn_type, rglztn_fctr = 1000, 0.1, 'K', 'dum rglztn type', 0

        super().__init__(standard_config, svd_config, SUPER_RAW_NUMPY_LIST, SUPER_WORKING_NUMPY_LIST, WORKING_VALIDATED_DATATYPES,
                         WORKING_MODIFIED_DATATYPES, WORKING_FILTERING, WORKING_MIN_CUTOFFS, WORKING_USE_OTHER,
                         WORKING_CONTEXT, WORKING_KEEP, WORKING_SCALING, split_method, LABEL_RULES, number_of_labels, event_value,
                         negative_value, conv_kill, pct_change, conv_end_method, rglztn_type, rglztn_fctr, __name__)


        self.max_columns = svd_max_columns



        self.WINNING_COLUMNS = []

        self.TRAIN_RESULTS = pd.DataFrame({})

        self.tr_score_idx = ''     # TR = TRAIN_RESULTS; COLUMN IDX IS DETERMINED AFTER GENERATION OF self.TRAIN_RESULTS

        self.context_text_holder = ''    # CREATE A STRING THAT RECORDS ALL OPERATIONS HERE, THEN IS APPENDED TO CONTEXT
        self.CONTEXT_UPDATE_HOLDER = []     # CREATE THIS TO HOLD CHOPS, THEN PUT INTO CONTEXT AFTER context_text_holder

        # KEEP THIS STUFF IN CASE EVER DO "FLOAT ONLY" OR "INT OR BIN ONLY" SVD
        # 4-30-22 sub_post_run_options_module() DYNAMIC MENU CALLS self.WORKING_MODIFIED_DATATYPES_BACKUP --- DONT DELETE
        self.SUPER_WORKING_NUMPY_LIST_BACKUP = [_.copy() if isinstance(_, n.ndarray) else deepcopy(_) for _ in SUPER_WORKING_NUMPY_LIST]
        self.WORKING_VALIDATED_DATATYPES_BACKUP = deepcopy(WORKING_VALIDATED_DATATYPES)
        self.WORKING_MODIFIED_DATATYPES_BACKUP = deepcopy(WORKING_MODIFIED_DATATYPES)
        self.WORKING_FILTERING_BACKUP = deepcopy(WORKING_FILTERING)
        self.WORKING_MIN_CUTOFFS_BACKUP = deepcopy(WORKING_MIN_CUTOFFS)
        self.WORKING_USE_OTHER_BACKUP = deepcopy(WORKING_USE_OTHER)
        self.WORKING_SCALING_BACKUP = deepcopy(WORKING_SCALING)


    # INHERITS #############################################################################################################
    # intercept_finder()                 Not used here. Locate an intercept column in DATA and handle any anamolous columns of constants.
    #                                    As of 11/15/22 only for MLR, MI, and GMLR.
    # run_module_input_tuple()           tuple of base params that pass into run_module for all ML packages
    # base_post_run_options_module()     holds post run options applied to all ML packages
    # return_fxn_base()                  values returned from all ML packages
    # configrun()                        runs config_module() & run_module()


    # OVERWRITES #######################################################################################################
    def config_module(self):
        # config_module()                    gets configuration source, returns configuration parameters for particular ML package
        self.max_columns = \
            sic.SVDConfig(self.standard_config, self.sub_config, self.SUPER_WORKING_NUMPY_LIST[0],
                 self.SUPER_WORKING_NUMPY_LIST[1], self.rglztn_fctr, self.conv_kill, self.pct_change,
                 self.conv_end_method, self.max_columns).config()


    def run_module(self):
        # run_module()                       returns run module for particular ML package

        while True:   # TO ALLOW ABORT
            '''
            if self.int_or_bin_only:
                
                __ = vui.validate_user_str(f'\nSVD is currently set for INT OR BIN ONLY.  Proceed(p), change(c), abort(a)? > ', 'PCA')
                if __ == 'P': pass
                elif __ == 'C':
                    self.float_only = {'T':True,'F':False}[vui.validate_user_str(f'Set SVD int or bin only: True(t) False(f) > ', 'TF')]
                elif __ == 'A': break

                if True not in [_ in self.WORKING_MODIFIED_DATATYPES[0] for _ in ['FLOAT']]:
                    print(f'\n*** THERE ARE NO FLOAT COLUMNS IN DATA ***')
                else:
                    print(f'\nDeleting FLOAT columns from DATA...')
                    start_len = len(self.WORKING_MODIFIED_DATATYPES[0])
                    for col_idx in range(len(self.WORKING_MODIFIED_DATATYPES[0])-1, -1, -1):
                        if self.WORKING_MODIFIED_DATATYPES[0][col_idx] in ['FLOAT']:
                            self.delete_columns(0, col_idx)

                    self.context_text_holder += ' ' + self.float_remove_verbage()
                    print(f'\n*** DELETE OF FLOAT COLUMNS COMPLETE (DELETED {start_len-len(self.WORKING_MODIFIED_DATATYPES[0])} COLUMNS OF {start_len}) ***')
                    '''
            # TRAIN_RESULTS COLUMNS HAVE A HEADER IN [0] POSN
            # TRAIN_RESULTS[0][0] = 'COLUMN'
            # TRAIN_RESULTS[1][0] = 'SVD SCORE'
            # TRAIN_RESULTS[2][0] = 'COEFFS'
            # TRAIN_RESULTS[3][0] = 'p VALUE'
            # TRAIN_RESULTS[4][0] = 'CUM R'
            # TRAIN_RESULTS[5][0] = 'CUM RSQ'
            # TRAIN_RESULTS[6][0] = 'CUM ADJ RSQ'
            # TRAIN_RESULTS[7][0] = 'CUM F'


            self.TRAIN_SWNL, self.DEV_SWNL, self.TEST_SWNL, self.PERTURBER_RESULTS, self.WINNING_COLUMNS, self.TRAIN_RESULTS = \
                sir.SVDRun(*self.run_module_input_tuple(), self.max_columns, self.TRAIN_RESULTS).run()

            self.tr_score_idx = [_ for _ in range(len(self.TRAIN_RESULTS)) if self.TRAIN_RESULTS[_][0] == 'SVD SCORE']

            self.context_text_holder += f' Ran SVD.'
            break


    def return_fxn(self):
        # return_fxn()                       returns user-specified output, in addition to return_fxn_base()
        return self.return_fxn_base()


    def sub_post_run_cmds(self):
    # package-specific options available to modify WORKING_DATA after run
        return {'k':'operations for SVD WINNING COLUMNS'}


    '''
    def float_remove_verbage(self):
    # verbage put into CONTEXT when floats are removed from DATA
        return f'Removed FLOAT columns from DATA prior to SVD.'


    def float_reappend_verbage(self):
    # verbage put into CONTEXT when floats are re-appended to DATA
        return f'Restored FLOAT columns to DATA after SVD.'
    '''

    def sub_post_run_options_module(self):
    # holds post run options unique to particular ML package
        if self.post_configrun_select == 'K':
            if self.WINNING_COLUMNS == []:
                print(f'\nMUTUAL INFORMATION HAS NOT BEEN RUN, WINNING COLUMNS LIST IS EMPTY.  NO COLUMNS DELETED.\n')
            elif len(self.SUPER_WORKING_NUMPY_LIST[0]) <= 1:
                print(f'\nWORKING DATA HAS {len(self.SUPER_WORKING_NUMPY_LIST[0])} COLUMNS, CANNOT DELETE.\n')
            else:

                while True:

                    POST_SVD_MENU = {'w': 'keep only WINNING COLUMNS from SVD',
                                     's': 'keep only WINNING COLUMNS above user-entered score',
                                     'x': 'select only top x from WINNING COLUMNS',
                                     'a': 'accept and continue'
                                     # 'f': 'append withheld FLOATS to WINNING COLUMNS'
                    }

                    # 4-20-22 IF bin_or_int_only WAS FALSE OR FLOAT ALREADY RE-APPENDED TO DATA, DONT ALLOW OPTION TO
                    # APPEND WITHHELD FLOATS

                    disallowed = ''

                    '''
                    if self.int_or_bin_only is False or \
                        self.float_reappend_verbage() in self.WORKING_CONTEXT or \
                        'FLOAT' not in self.WORKING_MODIFIED_DATATYPES_BACKUP[0]:

                        disallowed += 'f'
                    '''

                    post_svd_select = dmp.DictMenuPrint(POST_SVD_MENU, disp_len=140, disallowed=disallowed).select()

                    __ = vui.validate_user_str(f'\nUser selected {POST_SVD_MENU[post_svd_select.lower()]} '
                          f'\n--- Accept(y) Restart(n) Abort(a) > ', 'YNA')
                    if __ == 'Y': pass
                    elif __ == 'N': continue
                    elif __ == 'A': break

                    if post_svd_select == 'S':
                        # 'keep only WINNING COLUMNS above user-entered score(s)'
                        score_cutoff = vui.validate_user_float(f'Enter value of minimum score (delete all columns less than) > ', min=1e-99)
                        # TRAIN_RESULTS IS COMING IN FROM run_module() AS [] = COLUMNS, ROWS HOLDING INFO ABT A COLUMN IN DATA
                        if score_cutoff > n.max(self.TRAIN_RESULTS[self.tr_score_idx][1:]):
                            print(f'\nAll columns will be deleted.\n')
                        elif score_cutoff < n.min(self.TRAIN_RESULTS[self.tr_score_idx][1:]):
                            print(f'\nNo columns will be deleted.\n')
                        else:
                            chop_count = len([_ for _ in self.TRAIN_RESULTS[self.tr_score_idx][1:] if _ < score_cutoff])
                            print(f'\n{chop_count} columns of {len(self.TRAIN_RESULTS[self.tr_score_idx][1:])} will be deleted.\n')

                        __ = vui.validate_user_str(f'Accept(a) Restart(r) Abort(b) > ', 'ARB')

                        if __ == 'A':
                            # CHOP WINNING COLUMNS BASED ON score_cutoff... POSN IN WINNING_COLUMNS MATCHES ROW POSN IN TRAIN_RESULTS NOT INCLUDING HEADER
                            for row_idx in range(len(self.TRAIN_RESULTS[self.tr_score_idx]), 0, -1):
                                if self.TRAIN_RESULTS[self.tr_score_idx][row_idx] < score_cutoff:
                                    self.WINNING_COLUMNS.pop(row_idx - 1)  # POP CORRESPONDING COLUMN FROM WC (TR HAS HEADER, WC DOESNT)
                                    for col_idx in range(len(self.TRAIN_RESULTS)):  # POP CORRESPONDING ROW FROM ALL COLUMNS IN TR
                                        self.TRAIN_RESULTS[col_idx].pop(row_idx)

                            self.context_text_holder += f' Keeping only {len(self.WINNING_COLUMNS)} columns where score > {score_cutoff}.'
                            print(f'\n *** SCORE CHOP COMPLETE ***\n')

                        elif __ == 'R': continue
                        elif __ == 'B': break

                    if post_svd_select == 'X':
                        # 'select only top x from WINNING COLUMNS(x)'

                        top_winners = vui.validate_user_int(f'Enter number of top winners to keep > ', min=1, max=len(self.WINNING_COLUMNS))

                        __ = vui.validate_user_str(f'User entered {top_winners}, Accept(a) Restart(r) Abort(b) > ', 'ARB')
                        if __ == 'A':
                            self.context_text_holder += f' Keeping only top {top_winners} of {len(self.WINNING_COLUMNS)} winning columns.'

                            self.WINNING_COLUMNS = self.WINNING_COLUMNS[:top_winners]
                            self.TRAIN_RESULTS = [COLUMN[:top_winners + 1] for COLUMN in self.TRAIN_RESULTS]  # +1 BECAUSE OF HEADER

                            print(f'\n *** TOP {top_winners} CHOP COMPLETE ***\n')

                        elif __ == 'R': continue
                        elif __ == 'B': break

                    # THIS IS POST-RUN SO WINNING_COLUMNS IS COMING IN WITH THE RESULTS OF THE RUN.  THE STEPS ABOVE SCREEN
                    # TRAIN_RESULTS AND WINNING_COLUMNS BASED ON USER PICK "TOP X" OR "SCORE >", THEN APPLY THE SCREEN
                    # ONLY TO TRAIN RESULTS AND WINNING_COLUMNS.  THE SCREENED WINNING_COLUMNS THEN GO INTO THE 'SWX' STEP
                    # BELOW TO CHOP SWNL TO THE FINAL (MAYBE SCREENED) WINNING_COLUMNS, IF USER SELECTED TO DO SO

                    if post_svd_select in 'SWX':
                        # 'keep only WINNING COLUMNS from SVD(w)'

                        for col_idx in range(len(self.SUPER_WORKING_NUMPY_LIST[0])):
                            if col_idx not in self.WINNING_COLUMNS:
                                # 4-17-22, CANT DELETE COLUMNS IN KEEP (see delete_columns()) SO DECIDING TO SIMPLY NOTE IN "CONTEXT"
                                # THE COLUMNS THAT FAILED SVD

                                self.CONTEXT_UPDATE_HOLDER.append(f'Deleted DATA - {self.SUPER_WORKING_NUMPY_LIST[1][0][col_idx]} '
                                                            f'for failing singular value decomposition.')

                        self.SUPER_WORKING_NUMPY_LIST[0] = self.SUPER_WORKING_NUMPY_LIST[0][self.WINNING_COLUMNS]

                        self.context_text_holder += f' Retained top {len(self.WINNING_COLUMNS)} columns and removed non-winning and deselected columns.'

                        print(f'\n*** SVD NON-WINNING COLUMN CHOP COMPLETE. ***')

                        self.WINNING_COLUMNS = []  # RESET BACK TO EMPTY, DONT ALLOW COLUMN CHOPS AGAIN WITH SAME WINNING_COLUMNS
                        print(f'\nDELETE OF NON-WINNING COLUMNS FROM DATA SUCCESSFUL.  WINNING COLUMNS HAVE BEEN RESET TO NONE.\n')

                    if post_svd_select == 'F':
                        # 'append withheld FLOATS to WINNING COLUMNS(f)'
                        # ACCESS TO THIS COMMAND IN MENU IS CONTROLLED BY CONDITIONALS AT MENU GENERATION TIME

                        for col_idx in range(len(self.WORKING_MODIFIED_DATATYPES_BACKUP[0])):
                            # LOOK IN WORKING_MODIFIED_DATATYPES_BACKUP TO SEE WHAT THE ORIGINAL FLOAT COLUMNS WERE
                            if self.WORKING_MODIFIED_DATATYPES_BACKUP[0][col_idx] in ['FLOAT']:
                                self.SUPER_WORKING_NUMPY_LIST[0] = [*self.SUPER_WORKING_NUMPY_LIST[0], self.SUPER_WORKING_NUMPY_LIST_BACKUP[0][col_idx]]
                                self.SUPER_WORKING_NUMPY_LIST[1] = [[*self.SUPER_WORKING_NUMPY_LIST[1][0], self.SUPER_WORKING_NUMPY_LIST_BACKUP[1][0][col_idx]]]
                                self.WORKING_VALIDATED_DATATYPES[0] = [*self.WORKING_VALIDATED_DATATYPES[0], self.WORKING_VALIDATED_DATATYPES_BACKUP[0][col_idx]]
                                self.WORKING_MODIFIED_DATATYPES[0] = [*self.WORKING_MODIFIED_DATATYPES[0], self.WORKING_MODIFIED_DATATYPES_BACKUP[0][col_idx]]
                                self.WORKING_FILTERING[0] = [*self.WORKING_FILTERING[0], self.WORKING_FILTERING_BACKUP[0][col_idx]]
                                self.WORKING_MIN_CUTOFFS[0] = [*self.WORKING_MIN_CUTOFFS[0], self.WORKING_MIN_CUTOFFS_BACKUP[0][col_idx]]
                                self.WORKING_USE_OTHER[0] = [*self.WORKING_USE_OTHER[0], self.WORKING_USE_OTHER_BACKUP[0][col_idx]]
                                self.WORKING_SCALING = [*self.WORKING_SCALING, self.WORKING_SCALING_BACKUP[col_idx]]

                        # self.context_text_holder += " " + self.float_reappend_verbage()

                        # print(f'\n*** FLOAT COLUMNS HAVE BEEN SUCCESSFULLY RE-INTRODUCED INTO WORKING DATA ***\n')

                    if post_svd_select == 'A':
                        # 'accept and continue(a)'

                        self.WORKING_CONTEXT.append(self.context_text_holder.strip())
                        [self.WORKING_CONTEXT.append(_) for _ in self.CONTEXT_UPDATE_HOLDER]

                        del self.WORKING_MODIFIED_DATATYPES_BACKUP, self.WORKING_FILTERING_BACKUP, self.WORKING_MIN_CUTOFFS_BACKUP,\
                            self.WORKING_USE_OTHER_BACKUP, self.WORKING_SCALING_BACKUP, self.CONTEXT_UPDATE_HOLDER

                        return self.return_fxn()











if __name__ == '__main__':
    pass
















