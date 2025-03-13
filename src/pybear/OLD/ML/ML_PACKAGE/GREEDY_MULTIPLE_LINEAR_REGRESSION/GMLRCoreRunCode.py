import sys, time
import numpy as np, pandas as pd
import sparse_dict as sd
from copy import deepcopy
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv, validate_user_input as vui
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from general_list_ops import list_select as ls
from general_data_ops import get_shape as gs, new_np_random_choice as nnrc
from MLObjects import MLRowColumnOperations as mlrco
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo
from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION import LazyGMLR as lgmlr, LazyAggGMLR as lagmlr, ForwardGMLR as fgmlr, \
    BackwardGMLR as bgmlr
from read_write_file.generate_full_filename import base_path_select as bps, filename_enter as fe



class GMLRCoreRunCode:

    def __init__(self, DATA, DATA_HEADER, TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST, data_run_orientation,
                 target_run_orientation, gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type,
                 gmlr_rglztn_fctr, gmlr_batch_method, gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_max_columns,
                 intcpt_col_idx, gmlr_bypass_agg, bypass_validation):

        this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False], this_module, fxn)

        if not bypass_validation:
            data_run_orientation = akv.arg_kwarg_validater(data_run_orientation, 'data_run_orientation',
                                                                  ['ROW', 'COLUMN'], this_module, fxn)
            target_run_orientation = akv.arg_kwarg_validater(target_run_orientation, 'target_run_orientation',
                                                                  ['ROW', 'COLUMN'], this_module, fxn)

        DATA_HEADER = ldv.list_dict_validater(DATA_HEADER, 'DATA_HEADER')[1]


        ########################################################################################################################
        # SET UP BATCH OR MINIBATCH ############################################################################################

        if gmlr_batch_method == 'B':
            BATCH_DATA = DATA
            BATCH_TARGET = TARGET
            BATCH_TARGET_TRANSPOSE = TARGET_TRANSPOSE
            BATCH_TARGET_AS_LIST = TARGET_AS_LIST


        elif gmlr_batch_method == 'M':
            # KEEP "BATCH_SIZE" NUMBER OF EXAMPLES BY RANDOMLY GENERATED MASK
            # IF batch_size IS >1, USE THIS AS BATCH SIZE, IF batch_size IS <= 1 USE AS PERCENT OF DATA

            data_rows = gs.get_shape('DATA', DATA, data_run_orientation)[0]

            if gmlr_batch_size < 1: _len = int(np.ceil(gmlr_batch_size * data_rows))
            elif gmlr_batch_size >= 1: _len = int(min(gmlr_batch_size, data_rows))
            BATCH_MASK = nnrc.new_np_random_choice(range(data_rows), (1, int(_len)), replace=False).reshape((1, -1))[0]

            BATCH_DATA = mlrco.MLRowColumnOperations(DATA, data_run_orientation, name='DATA',
                bypass_validation=bypass_validation).return_rows(BATCH_MASK, return_orientation=data_run_orientation,
                                                                      return_format='AS_GIVEN')

            BATCH_TARGET = mlrco.MLRowColumnOperations(TARGET, target_run_orientation, name='TARGET',
                bypass_validation=bypass_validation).return_rows(BATCH_MASK, return_orientation=target_run_orientation,
                                                                      return_format='ARRAY')

            # REBUILT BY mloo BELOW
            BATCH_TARGET_TRANSPOSE = None
            BATCH_TARGET_AS_LIST = None

            del _len, BATCH_MASK

        else: raise Exception(f'*** ILLEGAL gmlr_batch_method "{gmlr_batch_method}". Must be "B" or "M".')

        # END SET UP BATCH OR MINIBATCH ########################################################################################
        ########################################################################################################################

        #####################################################################################################################
        # ORIENT TARGET & DATA ###############################################################################################

        # NOTES 5/14/23 --- IF MINI-BATCH WAS DONE, DATA & TARGET ARE COMING IN HERE IN "run" FORMAT & ORIENTATION,
        # AND ANY GIVEN TARGET_TRANSPOSE, ET AL KWARGS THAT WERE GIVEN ARE SET TO None AND NEED TO BE REBUILT. BUT IF IS
        # FULL BATCH, ROW SELECTION WAS BYPASSED AND DATA & TARGET ARE STILL IN GIVEN FORMAT & ORIENTATION
        # (WHICH AS OF 5/14/23 SHOULD NOW BE RUN FORMAT & ORIENTATION) AND ANY PASSED TRANSPOSE ETAL
        # KWARGS ARE STILL INTACT. ANY TRANSPOSE ETAL THAT WERE NOT PASSED GET BUILT BY mloo HERE.

        # BUILD BATCH_TARGET_TRANSPOSE & BATCH_TARGET_AS_LIST, THESE WONT CHANGE

        OrienterClass = mloo.MLObjectOrienter(
                                                DATA=BATCH_DATA,
                                                data_given_orientation=data_run_orientation,
                                                data_return_orientation=data_run_orientation,
                                                data_return_format='AS_GIVEN',

                                                target_is_multiclass=False,
                                                TARGET=BATCH_TARGET,
                                                target_given_orientation=target_run_orientation,
                                                target_return_orientation=target_run_orientation,
                                                target_return_format='ARRAY',

                                                TARGET_TRANSPOSE=BATCH_TARGET_TRANSPOSE,
                                                target_transpose_given_orientation=target_run_orientation,
                                                target_transpose_return_orientation=target_run_orientation,
                                                target_transpose_return_format='ARRAY',

                                                TARGET_AS_LIST=BATCH_TARGET_AS_LIST,
                                                target_as_list_given_orientation=target_run_orientation,
                                                target_as_list_return_orientation=target_run_orientation,

                                                RETURN_OBJECTS=['DATA', 'TARGET', 'TARGET_TRANSPOSE', 'TARGET_AS_LIST'],

                                                # CATCHES A MULTICLASS TARGET, CHECKS TRANSPOSES
                                                bypass_validation=bypass_validation,

                                                calling_module=this_module,
                                                calling_fxn=fxn
        )

        del DATA, BATCH_DATA, TARGET, BATCH_TARGET, TARGET_TRANSPOSE, BATCH_TARGET_TRANSPOSE, TARGET_AS_LIST, \
            BATCH_TARGET_AS_LIST, gmlr_batch_method, gmlr_batch_size

        data_run_orientation = OrienterClass.data_return_orientation
        target_run_orientation = OrienterClass.target_return_orientation
        BATCH_DATA = OrienterClass.DATA
        BATCH_TARGET = OrienterClass.TARGET
        BATCH_TARGET_TRANSPOSE = OrienterClass.TARGET_TRANSPOSE
        BATCH_TARGET_AS_LIST = OrienterClass.TARGET_AS_LIST

        del OrienterClass
        # END ORIENT TARGET & DATA #############################################################################################
        ########################################################################################################################

        batch_data_rows, batch_data_cols = gs.get_shape('BATCH_DATA', BATCH_DATA, data_run_orientation)
        batch_target_rows, batch_target_cols = gs.get_shape('BATCH_TARGET', BATCH_TARGET, target_run_orientation)

        if batch_data_cols == 0: raise Exception(f'BATCH DATA IN GMLRCoreRunCode HAS NO COLUMNS')

























        ########################################################################################################################
        # (GMLR VERIFY DIMENSIONS FOR MATMUL) (MI BUILD HOLDERS FOR ALL POSSIBLE Y_OCCUR, np.sum(Y_OCCUR), AND y_freq) ############

        # 9/16/22 --- IMPLEMENTED ABILITY TO CHOOSE UNSAFE sd.core_matmul TO SPEED UP, SO DO ONE CHECK HERE TO PROVE BROADCAST

        print(f'\n    Verifying dimensions for matmul.')

        if not batch_data_rows == batch_target_rows:
            raise Exception(f'{this_module}.{fxn}() TARGET ROWS != DATA ROWS.  FATAL ERROR.  TERMINATE.')

        # END (GMLR VERIFY DIMENSIONS FOR MATMUL) (MI BUILD HOLDERS FOR ALL POSSIBLE Y_OCCUR, np.sum(Y_OCCUR), AND y_freq) ############
        ########################################################################################################################


        # CREATE A HOLDER TO TRACK THE COLUMNS FOR X_TRIAL
        AVAILABLE_COLUMNS = np.arange(0, batch_data_cols, dtype=np.int32)

        if not intcpt_col_idx is None:  # IF DATA HAS INTERCEPT, MOVE THAT COLUMN TO FIRST POSITION
            # AVAILABLE COLUMNS IS OVERWROTE BY SELECTIONS OF LAZY, OTHERWISE IF FULL GMLR IT JUST PASSES THRU AS IS
            AVAILABLE_COLUMNS = np.insert(AVAILABLE_COLUMNS[AVAILABLE_COLUMNS != intcpt_col_idx], 0, intcpt_col_idx, axis=0)

        gmlr_rglztn_type = gmlr_rglztn_type.upper() if isinstance(gmlr_rglztn_type, str) else gmlr_rglztn_type

        if gmlr_rglztn_type in [None, 'NONE']: gmlr_rglztn_fctr = 0

        # *******************************************************************************************************************
        # *******************************************************************************************************************
        # RUN GREEDY GMLR ***************************************************************************************************

        ## ########################################################################################################################
        ## RUN LAZY MLR ########################################################################################################
        if gmlr_type == 'L':    # LAZY AGGLOMERATION --- top {max_columns} individual scores selected and piled together

            ## ########################################################################################################################
            ## RUN CORE LAZY MLR ###############################################################################################
            print(f'\n    RUNNING LAZY GREEDY MULTIPLE LINEAR REGRESSION...\n')

            print(f'\nCalculating individual "{["no intercept" if intcpt_col_idx is None else "intercept"][0]}" style '
                  f'lazy MLRegression results and building results table... \n')


            self.WINNING_COLUMNS, self.TRAIN_RESULTS, self.COEFFS = \
                lgmlr.LazyGMLR(BATCH_DATA,
                               DATA_HEADER,
                               data_run_orientation,
                               BATCH_TARGET if target_run_orientation=='ROW' else BATCH_TARGET_TRANSPOSE,
                               'ROW',
                               AVAILABLE_COLUMNS=AVAILABLE_COLUMNS,
                               max_columns=gmlr_max_columns,
                               intcpt_col_idx=intcpt_col_idx,
                               rglztn_fctr=gmlr_rglztn_fctr,
                               TRAIN_RESULTS=None,
                               TARGET_TRANSPOSE=BATCH_TARGET_TRANSPOSE if target_run_orientation=='ROW' else BATCH_TARGET,
                               TARGET_AS_LIST=BATCH_TARGET_AS_LIST if target_run_orientation=='ROW' else BATCH_TARGET_AS_LIST.transpose(),
                               data_run_orientation='ROW',
                               target_run_orientation='COLUMN',
                               bypass_validation=bypass_validation
            ).run()

            print(f'Lazy GMLR score calculations complete.')

            # END SORT & SELECT ######################################################################################################
            # ########################################################################################################################


            # ########################################################################################################################
            # RUN FULL AGGLOMERATIVE REGRESSION ON WINNING COLUMNS IF NOT BYPASSED ###################################################

            if gmlr_bypass_agg is False:

                print(f'\nCalculating agglomerative "{["no intercept" if intcpt_col_idx is None else "intercept"][0]}" style '
                        f'mlregression results for lazy winners and building results table... \n')

                self.WINNING_COLUMNS, self.TRAIN_RESULTS, self.COEFFS = \
                    lagmlr.LazyAggGMLR(BATCH_DATA,
                                        DATA_HEADER,
                                        data_run_orientation,
                                        BATCH_TARGET,
                                        target_run_orientation,
                                        AVAILABLE_COLUMNS=self.WINNING_COLUMNS, # WINNING_COLUMNS (NOT AV_COLS) BECAUSE WINNING_COLUMN IS RETURNED FROM Lazy
                                        max_columns=gmlr_max_columns,
                                        intcpt_col_idx=intcpt_col_idx,
                                        rglztn_fctr=gmlr_rglztn_fctr,
                                        score_method=gmlr_score_method,
                                        TRAIN_RESULTS=self.TRAIN_RESULTS,
                                        TARGET_TRANSPOSE=BATCH_TARGET_TRANSPOSE,
                                        TARGET_AS_LIST=BATCH_TARGET_AS_LIST,
                                        data_run_orientation=data_run_orientation,
                                        target_run_orientation=target_run_orientation,
                                        conv_kill=gmlr_conv_kill,
                                        pct_change=gmlr_pct_change,
                                        conv_end_method=gmlr_conv_end_method,
                                        bypass_validation=bypass_validation
                ).run()

            # END RUN FULL AGGLOMERATIVE REGRESSION ON WINNING COLUMNS IF NOT BYPASSED ###############################################
            # ########################################################################################################################

        # END RUN LAZY MLR ########################################################################################################
        # ########################################################################################################################

        # ########################################################################################################################
        # RUN FORWARD AGGLOMERATIVE MLR ####################################################################################################

        elif gmlr_type == 'F':
            print(f'\ncalculating agglomerative "{["no intercept" if intcpt_col_idx is None else "intercept"][0]}" style '
                  f'mlregression results and building results table... \n')

            self.WINNING_COLUMNS, self.TRAIN_RESULTS, self.COEFFS = \
                    fgmlr.ForwardGMLR(
                                        BATCH_DATA,
                                        DATA_HEADER,
                                        data_run_orientation,
                                        BATCH_TARGET,
                                        target_run_orientation,
                                        AVAILABLE_COLUMNS=AVAILABLE_COLUMNS,
                                        max_columns=gmlr_max_columns,
                                        intcpt_col_idx=intcpt_col_idx,
                                        rglztn_fctr=gmlr_rglztn_fctr,
                                        score_method=gmlr_score_method,
                                        TRAIN_RESULTS=None,
                                        TARGET_TRANSPOSE=BATCH_TARGET_TRANSPOSE,
                                        TARGET_AS_LIST=BATCH_TARGET_AS_LIST,
                                        data_run_orientation='ROW',
                                        target_run_orientation='ROW',
                                        conv_kill=gmlr_conv_kill,
                                        pct_change=gmlr_pct_change,
                                        conv_end_method=gmlr_conv_end_method,
                                        bypass_validation=bypass_validation
            ).run()


        # END FORWARD AGGLOMERATIVE REGRESSION #############################################################################
        #################################################################################################################



        #################################################################################################################
        # BACKWARD AGGLOMERATIVE REGRESSION #############################################################################
        elif gmlr_type=='B':

            print(f'\ncalculating backward search "{["no intercept" if intcpt_col_idx is None else "intercept"][0]}" style '
                  f'mlregression results and building results table... \n')

            self.WINNING_COLUMNS, self.TRAIN_RESULTS, self.COEFFS = \
                bgmlr.BackwardGMLR(
                                    BATCH_DATA,
                                    DATA_HEADER,
                                    data_run_orientation,
                                    BATCH_TARGET,
                                    target_run_orientation,
                                    AVAILABLE_COLUMNS=AVAILABLE_COLUMNS,
                                    max_columns=gmlr_max_columns,
                                    intcpt_col_idx=intcpt_col_idx,
                                    rglztn_fctr=gmlr_rglztn_fctr,
                                    score_method=gmlr_score_method,
                                    TRAIN_RESULTS=None,
                                    TARGET_TRANSPOSE=BATCH_TARGET_TRANSPOSE,
                                    TARGET_AS_LIST=BATCH_TARGET_AS_LIST,
                                    data_run_orientation='ROW',
                                    target_run_orientation='ROW',
                                    conv_kill=gmlr_conv_kill,
                                    pct_change=gmlr_pct_change,
                                    conv_end_method=gmlr_conv_end_method,
                                    bypass_validation=bypass_validation
            ).run()

        # END BACKWARD AGGLOMERATIVE REGRESSION #############################################################################
        #################################################################################################################


        '''
        BEAR 6/9/23
        RELICS OF nan HANDLING (ERRORS FROM MLRegression). THIS CAUSES PROMPTS DURING K-FOLD.
        
        #################################################################################################################
        # MAKE, CLEAN LAST_GOOD_TRAIN_RESULTS ########################################################################

        # FIGURE OUT IF THIS NaN HANDLING STUFF NEEDS TO GO INTO
        # ForwardGMLR ONLY, OR DOES IT NEED TO BE MADE TO APPLY TO ALL OF THEM.  IF ONLY SHOULD GO IN Forward, FIGURE
        # OUT HOW TO WORK THAT INTO ForwardGMLR().

        # self.TRAIN_RESULTS = self.TRAIN_RESULTS.iloc[self.WINNING_COLUMNS, :]

        # if error_winrank == float('inf'):   # IF MLRegression DID NOT ERROR, JUST USE COPY OF FULL RESULTS
        #     LAST_GOOD_TRAIN_RESULTS = self.TRAIN_RESULTS.copy()   # TRAIN_RESULTS IS DF
        # else:  # elif error_winrank != float('inf')   MLRegression ERRORED SOMEWHERE
        #     LAST_GOOD_TRAIN_RESULTS = self.TRAIN_RESULTS.iloc[:error_winrank, :]

        # END MAKE, CLEAN LAST_GOOD_TRAIN_RESULTS ####################################################################
        #################################################################################################################

        # del DATA_WIP, P_VALUES, R_LIST, R2_LIST, R2_ADJ_LIST, F_LIST, COEFFS_BACKUP, \
        #     P_VALUES_BACKUP, R_LIST_BACKUP, R2_LIST_BACKUP, R2_ADJ_LIST_BACKUP, F_LIST_BACKUP,
        #     error_winrank

        # LET USER DISPLAY OUTPUTS AND SELECT WHAT IS FINAL_RESULTS
        # SET MENU OPTIONS FOR DISPLAY / SELECT

        COMMANDS = {
            'f': 'display full results',
            # 'g': 'display only good results',
            'j': 'dump full results to excel',
            # 'k': 'dump only good results to excel',
            'a': 'select full results as final results and exit',
            # 'b': 'select good results as final results and exit',
        }


        while True:
            print()
            user_disp_return_select = dmp.DictMenuPrint(COMMANDS, disp_len=140).select(f'Select menu option')

            if user_disp_return_select == 'F':                        #'display full results(f)'
                print()
                print(f'\nFULL WINNING COLUMNS FOR GREEDY MLR:')
                print(self.TRAIN_RESULTS)
                print()

            # elif user_disp_return_select == 'G':                      #'display only good results(g)'
            #     print()
            #     print(f'\nONLY GOOD WINNING COLUMNS FOR GREEDY MLR:')
            #     print(LAST_GOOD_TRAIN_RESULTS)
            #     print()
            elif user_disp_return_select == 'J':                      #'dump full results to excel(j)'
                tred.train_results_excel_dump(self.TRAIN_RESULTS, 'TRAIN_RESULTS')

            # elif user_disp_return_select == 'K':                      #'dump only good results to excel(k)'
            #     tred.train_results_excel_dump(LAST_GOOD_TRAIN_RESULTS, 'LAST_GOOD_TRAIN_RESULTS')

            elif user_disp_return_select == 'A':                      #'select full results as final results(a)'
                if vui.validate_user_str(f'\nUser selected FULL RESULTS as final results. Accept? (y/n) > ', 'YN') == 'Y':
                    # self.TRAIN_RESULTS, self.WINNING_COLUMNS, self.COEFFS REMAIN AS BUILT
                    break

            # elif user_disp_return_select == 'B':                      #'select good results as final results(b)'
            #     if vui.validate_user_str(f'\nUser selected only GOOD RESULTS as final results. Accept? (y/n) > ', 'YN') == 'Y':
            #         self.TRAIN_RESULTS = LAST_GOOD_TRAIN_RESULTS
            #         self.WINNING_COLUMNS = self.WINNING_COLUMNS[:len(LAST_GOOD_TRAIN_RESULTS)]
            #         self.COEFFS = self.COEFFS[:len(LAST_GOOD_TRAIN_RESULTS)]
            #         break

        
        del COMMANDS, user_disp_return_select  # LAST_GOOD_TRAIN_RESULTS
        '''

        # END RUN ML & DISPLAY RESULTS ###############################################################################################
        # ########################################################################################################################
        # ########################################################################################################################


    def return_fxn(self):
        return self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS


    def run(self):
        return self.return_fxn()























if __name__ == '__main__':

    from general_sound import winlinsound as wls
    from MLObjects.TestObjectCreators import test_header as th


    print(f'CREATE OBJECTS')

    run_as = ls.list_single_select(['random_integer', 'binary', 'random_sparse_dict', 'mi_scratch', 'beer_reviews'],
                                   'Select dataset', 'value')[0]

    test_type = ls.list_single_select(['loop', 'one-off'], 'Select tests type', 'value')[0]

    bypass_validation = ls.list_single_select([True, False], 'Select bypass_validation', 'value')[0]

    rows = 10000
    columns = 20


    if run_as == 'random_integer':
        data_given_orientation = 'COLUMN'
        target_given_orientation = 'COLUMN'
        DATA = np.random.randint(0, 10, (columns,rows) if data_given_orientation=='COLUMN' else (rows, columns))
        DATA_HEADER = th.test_header(columns)
        TARGET = np.random.randint(0, 2, (1, rows) if target_given_orientation=='COLUMN' else (rows, 1))
        TARGET_HEADER = [['TARGET']]

    elif run_as == 'binary':
        data_given_orientation = 'COLUMN'
        target_given_orientation = 'COLUMN'
        DATA = np.random.randint(0, 2, (columns, rows) if data_given_orientation=='COLUMN' else (rows, columns))
        DATA_HEADER = th.test_header(columns)
        TARGET = np.random.randint(0, 2, (1, rows) if target_given_orientation=='COLUMN' else (rows, 1))
        TARGET_HEADER = [['TARGET']]

    elif run_as == 'random_sparse_dict':
        data_given_orientation = 'COLUMN'
        target_given_orientation = 'COLUMN'
        DATA = sd.create_random((columns, rows) if data_given_orientation=='COLUMN' else (rows,columns), 90)
        DATA_HEADER = th.test_header(columns)
        TARGET = np.random.randint(0, 2, (1,rows) if target_given_orientation=='COLUMN' else (rows, 1))
        TARGET_HEADER = [['TARGET']]

    elif run_as == 'mi_scratch':
        data_given_orientation = 'ROW'
        target_given_orientation = 'COLUMN'
        basepath = bps.base_path_select()
        READ_DATA = pd.read_excel(basepath + r'MI_SCRATCH.xlsx', sheet_name='DATA')
        READ_HEADER = [['TARGET'] + [th.test_header(10)[0][_] for _ in range(10)]]
        READ_DATA.columns = READ_HEADER
        TARGET = READ_DATA['TARGET'].copy().to_numpy().reshape((1,-1) if target_given_orientation=='COLUMN' else (-1,1))
        TARGET_HEADER = [['TARGET']]
        DATA_HEADER = np.fromiter(READ_DATA.keys()[1:], dtype=str).reshape((1,-1))
        DATA = READ_DATA.drop(columns=['TARGET'], axis=0).to_numpy().reshape((1,-1) if data_given_orientation=='COLUMN' else (-1,1))

    elif run_as == 'beer_reviews':
        data_given_orientation = 'COLUMN'
        target_given_orientation = 'COLUMN'

        DATA = pd.read_csv(r'C:\Users\Bill\Documents\WORK STUFF\RESUME\1 - OTHER\SRP\beer_reviews.csv',
                           nrows=rows,
                           header=0).dropna(axis=0)

        KEEP = [3, 4, 5, 7, 8, 9, 11]
        DATA = DATA[np.fromiter(DATA.keys(), dtype='<U20')[KEEP]]

        TARGET = DATA['review_overall'].copy().to_numpy().reshape((1,-1) if target_given_orientation=='COLUMN' else (-1,1))
        TARGET_HEADER = [['TARGET']]

        DATA = DATA.drop(columns=['review_overall'])

        DATA = pd.get_dummies(DATA, columns=['beer_style'], prefix='', prefix_sep='', dtype=np.int8)
        DATA_HEADER = np.fromiter(DATA.keys(), dtype='<U50').reshape((1, -1))

        if data_given_orientation=='COLUMN': DATA = DATA.to_numpy().transpose()

        REF_VEC = np.arange(0, len(DATA) if data_given_orientation=='ROW' else len(DATA[0]), dtype=np.int32)
        REF_VEC_HEADER = [['ROW_ID']]

        '''
        # FILE DUMP ############################################################################################

        DATA.insert(0, 'TARGET', TARGET.transpose())

        base_path = bps.base_path_select()
        file_name = fe.filename_wo_extension()

        full_path = base_path + file_name + ".xlsx"
        print(f'\nSaving file to {full_path}....')

        with pd.ExcelWriter(full_path) as writer:

            DATA.style.set_properties(**{'text-align': 'center'}).to_excel(excel_writer=writer,
                                                                             sheet_name='DATA', float_format='%.4f',
                                                                             startrow=1, startcol=1, merge_cells=False,
                                                                             index=True, na_rep='NaN')
        print(f'Done.')
        quit()
        # END FILE DUMP ############################################################################################
        '''
    else: raise Exception(f'Dataset selector logic is failing')

    #######################################################################################################################################
    #######################################################################################################################################
    #######################################################################################################################################


    #######################################################################################################################################
    #######################################################################################################################################
    #######################################################################################################################################

    OrienterClass = mloo.MLObjectOrienter(
                                             DATA=DATA,
                                             data_given_orientation=data_given_orientation,
                                             data_return_orientation='AS_GIVEN',
                                             data_return_format='AS_GIVEN',

                                             DATA_TRANSPOSE=None,
                                             data_transpose_given_orientation=None,
                                             data_transpose_return_orientation='AS_GIVEN',
                                             data_transpose_return_format='AS_GIVEN',

                                             XTX=None,
                                             xtx_return_format='AS_GIVEN',

                                             XTX_INV=None,
                                             xtx_inv_return_format='AS_GIVEN',

                                             target_is_multiclass=None,
                                             TARGET=TARGET,
                                             target_given_orientation=target_given_orientation,
                                             target_return_orientation='AS_GIVEN',
                                             target_return_format='AS_GIVEN',

                                             TARGET_TRANSPOSE=None,
                                             target_transpose_given_orientation=None,
                                             target_transpose_return_orientation='AS_GIVEN',
                                             target_transpose_return_format='AS_GIVEN',

                                             TARGET_AS_LIST=None,
                                             target_as_list_given_orientation=None,
                                             target_as_list_return_orientation='AS_GIVEN',

                                             RETURN_OBJECTS=['DATA', 'DATA_TRANSPOSE', 'TARGET', 'TARGET_TRANSPOSE',
                                                             'TARGET_AS_LIST'],

                                             bypass_validation=False,
                                             calling_module='GMLRCoreRunCode',
                                             calling_fxn='tests'
    )

    DATA = OrienterClass.DATA
    DATA_TRANSPOSE = OrienterClass.DATA_TRANSPOSE
    data_run_orientation = OrienterClass.data_return_orientation
    TARGET = OrienterClass.TARGET
    TARGET_TRANSPOSE = OrienterClass.TARGET_TRANSPOSE
    TARGET_AS_LIST = OrienterClass.TARGET_AS_LIST
    target_run_orientation = OrienterClass.target_return_orientation







    if test_type == 'loop':

        ORIG_NP_DATA = DATA
        ORIG_SD_DATA = sd.zip_list_as_py_float(DATA)
        ORIG_HEADER = DATA_HEADER
    
        print(f'CREATE OBJECTS Done.')

        MASTER_GMLR_TYPE = ['L', 'F', 'B']
        MASTER_MAX_COLUMNS = [20, 200]
        MASTER_SCORE_METHOD = ['R', 'Q', 'A', 'F']
        MASTER_USE_INTERCEPT = [True, False]
        MASTER_DATA_FORMAT = ['A', 'S']

        gmlr_conv_kill = None
        gmlr_pct_change = float('inf')
        gmlr_conv_end_method = 'KILL'
        gmlr_rglztn_type = 'RIDGE'
        gmlr_rglztn_fctr = 10000
        gmlr_batch_method = 'B'
        gmlr_batch_size = float('inf')
        gmlr_bypass_agg = False

        total_trials = np.product(list(map(len, (MASTER_GMLR_TYPE, MASTER_MAX_COLUMNS, MASTER_SCORE_METHOD,
                                                 MASTER_USE_INTERCEPT, MASTER_DATA_FORMAT))))

        ctr = 0
        for gmlr_type in MASTER_GMLR_TYPE:
            for gmlr_max_columns in MASTER_MAX_COLUMNS:
                for gmlr_score_method in MASTER_SCORE_METHOD:
                    for gmlr_use_intercept in MASTER_USE_INTERCEPT:
                        for data_format in MASTER_DATA_FORMAT:
                            ctr += 1
                            if data_format == 'S': DATA = ORIG_SD_DATA
                            elif data_format == 'A': DATA = ORIG_NP_DATA
        
                            if gmlr_use_intercept is True:
                                if data_format == 'A':
                                    DATA = np.insert(ORIG_NP_DATA, len(ORIG_NP_DATA), 1, axis=0)
                                elif data_format == 'S':
                                    DATA = sd.core_insert_outer(ORIG_SD_DATA, sd.outer_len(ORIG_SD_DATA),
                                                np.fromiter((1 for _ in ORIG_NP_DATA[0]), dtype=int))
                                intcpt_col_idx = len(DATA) - 1
                                DATA_HEADER = np.hstack((ORIG_HEADER, [['INTERCEPT']]))
                            else:
                                intcpt_col_idx = None
        
                            _len_data = len(DATA)
        
                            WINNING_COLUMNS, COEFFS, TRAIN_RESULTS = \
                                GMLRCoreRunCode(DATA, DATA_HEADER, TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST, data_run_orientation,
                                    target_run_orientation, gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type,
                                    gmlr_rglztn_fctr, gmlr_batch_method, gmlr_batch_size, gmlr_type, gmlr_score_method,
                                    gmlr_max_columns, intcpt_col_idx, gmlr_bypass_agg, bypass_validation).run()
        
                            print(f'\nJust finished trial {ctr} of {total_trials}.' + \
                                f'\ngmlr_type = {gmlr_type}' + \
                                f'\ngmlr_max_columns = {gmlr_max_columns}' + \
                                f'\ngmlr_score_method = {gmlr_score_method}' + \
                                f'\ngmlr_use_intercept = {gmlr_use_intercept}' + \
                                f'\ndata_format = {data_format}' + \
                                f'\nlen(DATA) = {_len_data}' + \
                                f'\nlen(DATA[0]) = {sd.inner_len_quick(DATA) if isinstance(DATA, dict) else len(DATA[0])}' + \
                                f'\nlen(WINNING_COLUMNS) = {len(WINNING_COLUMNS)}' + \
                                f'\nlen(COEFFS) = {len(COEFFS)}' + \
                                f'\nlen(TRAIN_RESULTS) = {len(TRAIN_RESULTS)}\n')
        
                            # print(TRAIN_RESULTS)
                            #
                            # _ = input(f'\nHIT ENTER TO CONTINUE > \n')
                            # print(f'Running...')
        
                            exception_text = \
                                f'\nDISASTER during trial {ctr} of {total_trials}.' + \
                                f'\ngmlr_type = {gmlr_type}' + \
                                f'\ngmlr_max_columns = {gmlr_max_columns}' + \
                                f'\ngmlr_score_method = {gmlr_score_method}' + \
                                f'\ngmlr_use_intercept = {gmlr_use_intercept}' + \
                                f'\ndata_format = {data_format}' + \
                                f'\nlen(DATA) = {_len_data}' + \
                                f'\nlen(DATA[0]) = {sd.inner_len_quick(DATA) if isinstance(DATA, dict) else len(DATA[0])}' + \
                                f'\nlen(WINNING_COLUMNS) = {len(WINNING_COLUMNS)}' + \
                                f'\nlen(COEFFS) = {len(COEFFS)}' + \
                                f'\nlen(TRAIN_RESULTS) = {len(TRAIN_RESULTS)}\n'
        
        
                            if len(WINNING_COLUMNS) != min(_len_data, gmlr_max_columns):
                                wls.winlinsound(888, 500)
                                raise Exception(f'\nError 1' + exception_text)
                            if len(COEFFS) != min(_len_data, gmlr_max_columns):
                                wls.winlinsound(888, 500)
                                raise Exception(f'\nError 2' + exception_text)
                            if len(TRAIN_RESULTS) != min(_len_data, gmlr_max_columns):
                                wls.winlinsound(888,500)
                                raise Exception(f'\nError 3' + exception_text)
                            if gmlr_use_intercept is True and TRAIN_RESULTS['COLUMN']['NAME'][0] != 'INTERCEPT':
                                raise Exception(f'\nError 4' + exception_text)

    elif test_type=='one-off':
        ########################################################################################################################
        ########################################################################################################################
        ###### STUFF FOR ONE-OFF TEST OF CoreRun ####################################################################################

        data_format = vui.validate_user_str(f'\nRun as sparse dict(s) or array(a) > ', 'AS')
        if data_format == 'S': DATA = sd.zip_list_as_py_float(DATA)
        # INDEX = [*range(1, rows + 1), 'SCORE']

        gmlr_conv_kill = 1
        gmlr_pct_change = 0
        gmlr_conv_end_method = 'PROMPT'
        gmlr_rglztn_type = 'RIDGE'

        gmlr_rglztn_fctr = 0
        gmlr_batch_method = 'B'
        gmlr_batch_size = 2000
        gmlr_type = vui.validate_user_str(f'Run GMLR as L, F, or B > ', 'LFB')
        gmlr_score_method = {'L':'Q', 'F':'Q', 'B':'F'}[gmlr_type]
        gmlr_use_intercept = True
        gmlr_max_columns = 20
        gmlr_bypass_agg = False

        if gmlr_use_intercept is True:
            if isinstance(DATA, (np.ndarray, list, tuple)):
                DATA = np.insert(DATA, len(DATA), 1, axis=0)
            elif isinstance(DATA, dict):
                DATA = sd.append_outer(DATA, np.fromiter((1 for _ in range(sd.inner_len_quick(DATA))), dtype=int))
            intcpt_col_idx = len(DATA) - 1
            DATA_HEADER = np.hstack((DATA_HEADER, [['INTERCEPT']]))
        else:
            intcpt_col_idx = None

        WINNING_COLUMNS, COEFFS, TRAIN_RESULTS = GMLRCoreRunCode(DATA, DATA_HEADER, TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST,
                 data_run_orientation, target_run_orientation, gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method,
                 gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, gmlr_batch_size, gmlr_type, gmlr_score_method,
                 gmlr_max_columns, intcpt_col_idx, gmlr_bypass_agg, bypass_validation).run()

        for _ in range(3): wls.winlinsound(888,1000); time.sleep(1)

        ###### STUFF FOR ONE-OFF TEST OF CoreRun ####################################################################################
        ########################################################################################################################
        ########################################################################################################################


    print(400 * '*')
    print(f'PRINTING STUFF IN GUARD:\n')
    if isinstance(DATA, dict):
        DATA = sd.unzip_to_ndarray(DATA)[0]

    print(f'\nTRAIN RESULTS:')
    print(TRAIN_RESULTS)

    PREDICTED = np.matmul(DATA[WINNING_COLUMNS.astype(int)].transpose().astype(float), COEFFS.astype(float), dtype=float)

    DATA_DICT = {}

    if isinstance(DATA, (list, tuple, np.ndarray)):
        for idx in range(len(TARGET)): DATA_DICT[TARGET_HEADER[0][idx]] = deepcopy(TARGET[idx])
    elif isinstance(DATA, dict):
        for idx in range(len(TARGET)): DATA_DICT[TARGET_HEADER[0][idx]] = sd.zip_list([deepcopy(TARGET[idx])])[0]

    DATA_DICT['PREDICTED'] = PREDICTED
    for idx in WINNING_COLUMNS:
        DATA_DICT[DATA_HEADER[0][idx]] = deepcopy(DATA[idx])

    DF = pd.DataFrame(DATA_DICT).fillna(0)
    print()
    print(DF)

    print('Done.')

    from data_validation import validate_user_input as vui

    if vui.validate_user_str(f'Dump DATA to file? (y/n) > ', 'YN') == 'Y':
        base_path = bps.base_path_select()
        file_name = fe.filename_wo_extension()
        print(f'\nSaving file to {base_path + file_name + ".xlsx"}....')
        pd.DataFrame.to_excel(DF,
                              excel_writer=base_path + file_name + '.xlsx',
                              float_format='%.2f',
                              startrow=1,
                              startcol=1,
                              merge_cells=False
                              )
        print('Done.')




















