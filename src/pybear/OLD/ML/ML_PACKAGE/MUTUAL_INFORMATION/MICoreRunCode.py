import sys, itertools, time
from copy import deepcopy
import numpy as np, pandas as pd
import sparse_dict as sd
from debug import get_module_name as gmn
from general_sound import winlinsound as wls
from general_data_ops import get_shape as gs, new_np_random_choice as nnrc
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo
from MLObjects import MLObject as mlo, MLRowColumnOperations as mlrco
from ML_PACKAGE.GENERIC_PRINT import DictMenuPrint as dmp, train_results_excel_dump as tred
from ML_PACKAGE.MLREGRESSION import MLRegression as mlr
from ML_PACKAGE.MUTUAL_INFORMATION import MutualInformation as mi, MICrossEntropyObjects as miceo, build_empty_mi_train_results as bemtr


# THIS MAKES ALL DATAFRAME HEADERS AND INDEXES "UNSPARSE"
pd.set_option('display.multi_sparse', False, 'display.colheader_justify', 'center')
pd.set_option('display.max_columns', None, 'display.width', 140, 'display.max_colwidth', 35)
pd.options.display.float_format = '{:,.5f}'.format



class MICoreRunCode:

    def __init__(self, DATA, DATA_HEADER, TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST, data_run_orientation,
                 target_run_orientation, mi_batch_method, mi_batch_size, mi_max_columns, mi_bypass_agg, intcpt_col_idx,
                 Y_OCCUR_HOLDER, Y_SUM_HOLDER, Y_FREQ_HOLDER, bypass_validation):

        this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True,False,None],
                                                    this_module, fxn, return_if_none=False)

        if not bypass_validation:
            data_run_orientation = akv.arg_kwarg_validater(data_run_orientation, 'data_run_orientation',
                                                                  ['ROW', 'COLUMN'], this_module, fxn)
            target_run_orientation = akv.arg_kwarg_validater(target_run_orientation, 'target_run_orientation',
                                                                  ['ROW', 'COLUMN'], this_module, fxn)

        DATA_HEADER = ldv.list_dict_validater(DATA_HEADER, 'DATA_HEADER')[1]

        ########################################################################################################################
        # SET UP BATCH OR MINIBATCH ############################################################################################

        if mi_batch_method == 'B':
            BATCH_DATA = DATA
            BATCH_TARGET = TARGET
            BATCH_TARGET_TRANSPOSE = TARGET_TRANSPOSE
            BATCH_TARGET_AS_LIST = TARGET_AS_LIST
            BATCH_TARGET_UNIQUES = None
            BATCH_Y_OCCUR_HOLDER = Y_OCCUR_HOLDER
            BATCH_Y_SUM_HOLDER = Y_SUM_HOLDER
            BATCH_Y_FREQ_HOLDER = Y_FREQ_HOLDER


        elif mi_batch_method == 'M':
            # KEEP "BATCH_SIZE" NUMBER OF EXAMPLES BY RANDOMLY GENERATED MASK
            # IF batch_size IS >1, USE THIS AS BATCH SIZE, IF batch_size IS <= 1 USE AS PERCENT OF DATA

            data_rows = gs.get_shape('DATA', DATA, data_run_orientation)[0]

            if mi_batch_size < 1: _len = int(np.ceil(mi_batch_size * data_rows))
            elif mi_batch_size >= 1: _len = int(min(mi_batch_size, data_rows))
            BATCH_MASK = nnrc.new_np_random_choice(range(data_rows), (1, int(_len)), replace=False).reshape((1, -1))[0]

            BATCH_DATA = mlrco.MLRowColumnOperations(DATA, data_run_orientation, name='DATA',
                bypass_validation=bypass_validation).return_rows(BATCH_MASK, return_orientation=data_run_orientation,
                                                                      return_format='AS_GIVEN')

            BATCH_TARGET = mlrco.MLRowColumnOperations(TARGET, target_run_orientation, name='TARGET',
                bypass_validation=bypass_validation).return_rows(BATCH_MASK, return_orientation=target_run_orientation,
                                                                      return_format='ARRAY')

            # MUST RECREATE THESE VIA ObjectOrienter AFTER PULLING A MINIBATCH, OVERWRITING ANYTHING THAT MAY HAVE BEEN PASSED AS KWARG
            BATCH_TARGET_UNIQUES = None   # REBUILD LATER.  ONLY KEEP TARGET_UNIQUES KWARG IF RUNNING FULL BATCH
            BATCH_TARGET_TRANSPOSE = None
            BATCH_TARGET_AS_LIST = None
            BATCH_Y_OCCUR_HOLDER = None
            BATCH_Y_SUM_HOLDER = None
            BATCH_Y_FREQ_HOLDER = None

            del _len, BATCH_MASK

        # END SET UP BATCH OR MINIBATCH ########################################################################################
        ########################################################################################################################

        #####################################################################################################################
        # ORIENT TARGET & DATA ###############################################################################################

        # NOTES 5/14/23 --- IF MINI-BATCH WAS DONE, ANY GIVEN DATA_TRANSPOSE, ET AL KWARGS THAT WERE GIVEN ARE SET TO
        # None AND NEED TO BE REBUILT. BUT IF IS FULL BATCH, ROW SELECTION WAS BYPASSED AND DATA & TARGET ARE STILL IN GIVEN
        # FORMAT & ORIENTATION (WHICH AS OF 5/14/23 SHOULD NOW BE RUN FORMAT & ORIENTATION) AND ANY PASSED TRANSPOSE ETAL
        # KWARGS ARE STILL INTACT. ANY TRANSPOSE ETAL THAT WERE NOT PASSED GET BUILT BY mloo HERE.

        # BUILD BATCH_TARGET_TRANSPOSE & BATCH_TARGET_AS_LIST, THESE WONT CHANGE

        OrienterClass = mloo.MLObjectOrienter(
                                                DATA=BATCH_DATA,
                                                data_given_orientation=data_run_orientation,
                                                data_return_orientation='COLUMN',
                                                data_return_format='AS_GIVEN',

                                                target_is_multiclass=False,
                                                TARGET=BATCH_TARGET,
                                                target_given_orientation=target_run_orientation,
                                                target_return_orientation='COLUMN',
                                                target_return_format='ARRAY',

                                                TARGET_TRANSPOSE=BATCH_TARGET_TRANSPOSE,
                                                target_transpose_given_orientation=target_run_orientation,
                                                target_transpose_return_orientation='COLUMN',
                                                target_transpose_return_format='ARRAY',

                                                TARGET_AS_LIST=BATCH_TARGET_AS_LIST,
                                                target_as_list_given_orientation=target_run_orientation,
                                                target_as_list_return_orientation='COLUMN',

                                                 RETURN_OBJECTS=['DATA', 'TARGET', 'TARGET_TRANSPOSE', 'TARGET_AS_LIST'],

                                                 # CATCHES A MULTICLASS TARGET, CHECKS TRANSPOSES
                                                 bypass_validation=bypass_validation,

                                                 calling_module=this_module,
                                                 calling_fxn=fxn
        )

        del DATA, BATCH_DATA, TARGET, BATCH_TARGET, TARGET_TRANSPOSE, BATCH_TARGET_TRANSPOSE, TARGET_AS_LIST, \
            BATCH_TARGET_AS_LIST, mi_batch_method, mi_batch_size

        data_run_orientation = OrienterClass.data_return_orientation
        target_run_orientation = OrienterClass.target_return_orientation
        BATCH_DATA = OrienterClass.DATA
        BATCH_TARGET = OrienterClass.TARGET
        BATCH_TARGET_TRANSPOSE = OrienterClass.TARGET_TRANSPOSE
        BATCH_TARGET_AS_LIST = OrienterClass.TARGET_AS_LIST

        del OrienterClass
        # END ORIENT TARGET & DATA #############################################################################################
        ########################################################################################################################

        batch_data_cols = gs.get_shape('BATCH_DATA', BATCH_DATA, data_run_orientation)[1]

        if batch_data_cols == 0: raise Exception(f'DATA IN MICoreRunCode HAS NO COLUMNS')

        # CREATE A HOLDER TO TRACK THE COLUMNS TO USE AND THOSE IN X_FINAL -- DONT HAVE TO PUT INTERCEPT FIRST HERE
        AVAILABLE_COLUMNS = np.arange(0, batch_data_cols, dtype=np.int32)



        #########################################################################################################################
        # BUILD Y_OCCUR, Y_SUM, Y_FREQ ##########################################################################################

        # THESE MAY HAVE BEEN GIVEN AS KWARGS
        # VALIDATE BECAUSE MAY NOT HAVE BEEN CREATED BY MICrossEntropyObjects

        if not True in map(lambda x: x is None, (BATCH_Y_OCCUR_HOLDER, BATCH_Y_SUM_HOLDER, BATCH_Y_FREQ_HOLDER)):
            # IF ALL MI TARGET CROSS-ENTROPY TARGETS WERE PASSED AS KWARGS, BYPASS ALL OF THIS
            pass
        else:
            if BATCH_TARGET_UNIQUES is None:
                TargetUniquesClass = mlo.MLObject(BATCH_TARGET_AS_LIST,
                                                  target_run_orientation,
                                                  name='TARGET',
                                                  return_orientation='AS_GIVEN',
                                                  return_format='AS_GIVEN',
                                                  bypass_validation=bypass_validation,
                                                  calling_module=this_module,
                                                  calling_fxn=fxn)

                BATCH_TARGET_UNIQUES = TargetUniquesClass.unique(0).reshape((1, -1))
                del TargetUniquesClass


            # IF SIZE Y_OCCUR_HOLDER WOULD BE OVER 100,000,000 IF ARRAY, RETURN AS SPARSE_DICT
            y_formats = 'AS_GIVEN' if len(BATCH_TARGET_UNIQUES[0]) * gs.get_shape(
                                            'TARGET', BATCH_TARGET_AS_LIST, target_run_orientation)[0] < 1e8 else "SPARSE_DICT"

            if BATCH_Y_OCCUR_HOLDER is None:
                BATCH_Y_OCCUR_HOLDER = miceo.occurrence(BATCH_TARGET_AS_LIST, OBJECT_UNIQUES=BATCH_TARGET_UNIQUES, return_as=y_formats,
                                bypass_validation=bypass_validation, calling_module=this_module, calling_fxn=fxn)
            else: pass # BATCH_Y_OCCUR_HOLDER STAYS THE SAME

            if BATCH_Y_SUM_HOLDER is None: BATCH_Y_SUM_HOLDER = miceo.sums(BATCH_Y_OCCUR_HOLDER, return_as=y_formats,
                                                                     calling_module=this_module, calling_fxn=fxn)
            else: pass # BATCH_Y_SUM_HOLDER STAYS THE SAME

            if BATCH_Y_FREQ_HOLDER is None: BATCH_Y_FREQ_HOLDER = miceo.frequencies(BATCH_Y_SUM_HOLDER,
                                    return_as=y_formats, calling_module=this_module, calling_fxn=fxn)
            else: pass # BATCH_Y_FREQ_HOLDER STAYS THE SAME

            del y_formats

        if not bypass_validation:
            y_sum_cols = gs.get_shape('Y_SUM_HOLDER', BATCH_Y_SUM_HOLDER, 'ROW')[1]
            y_freq_cols = gs.get_shape('Y_FREQ_HOLDER', BATCH_Y_FREQ_HOLDER, 'ROW')[1]
            if y_sum_cols != y_freq_cols:
                raise Exception(f'*** Y_SUM_HOLDER ({y_sum_cols}) AND Y_FREQ_HOLDER ({y_freq_cols}) ARE NOT EQUAL SIZE ***')
            y_occ_cols = gs.get_shape('Y_OCCUR_HOLDER', BATCH_Y_OCCUR_HOLDER, 'COLUMN')[1]
            if y_occ_cols != y_sum_cols:
                raise Exception(f'*** Y_OCCUR_HOLDER ({y_occ_cols}) IS NOT SAME SIZE AS Y_SUM AND Y_FREQ ({y_sum_cols}) ***')

            del y_sum_cols, y_freq_cols, y_occ_cols

        # END BUILD Y_OCCUR, Y_SUM, Y_FREQ ######################################################################################
        #########################################################################################################################


















        # PLACEHOLDERS
        MI_SCORES = np.zeros((1,0), dtype=object)[0]
        INDIV_R = np.zeros((1,0), dtype=object)[0]
        INDIV_RSQ = np.zeros((1,0), dtype=object)[0]
        INDIV_RSQ_ADJ = np.zeros((1,0), dtype=object)[0]
        INDIV_F = np.zeros((1,0), dtype=object)[0]




        # ########################################################################################################################
        # RUN GREEDY MI ##########################################################################################################

        print(f'\nRUNNING GREEDY MI...\n')

        ## RUN CORE MI & INDIV MLR ################################################################################################


        mimlrdf = 'ARRAY'  # MI_MLR_data_format
        mimlrdo = 'ROW'    # MI_MLR_data_orientation

        for itr, col_idx in enumerate(AVAILABLE_COLUMNS, 1):

            if itr % 100 == 0:
                print(f'Running column {itr} of {len(AVAILABLE_COLUMNS)}...')

            # CALCULATE MI & LOG RESULTS #############################################################################

            # IF DATASET 0RIGINALLY HAD AN INTCPT, GET IT HERE TO USE FOR EACH COLUMNS Reg (EXCEPT INTCPT TRIAL)... AND VICE VERSA,
            # BUT DONT PUT INTCPT INTO MutualInFormation
            if intcpt_col_idx is None or col_idx == intcpt_col_idx: COLUMNS_TO_PULL = [col_idx]
            if not intcpt_col_idx is None: COLUMNS_TO_PULL = [col_idx, intcpt_col_idx]

            MI_MLR_DATA = mlrco.MLRowColumnOperations(BATCH_DATA, data_run_orientation, bypass_validation=bypass_validation,
                name='BATCH_DATA').return_columns(COLUMNS_TO_PULL, return_orientation=mimlrdo, return_format=mimlrdf)

            del COLUMNS_TO_PULL

            # WITH Y_MICEO OBJECTS PASSED BATCH_TARGET SHOULD NOT MATTER
            total_score = mi.MutualInformation(MI_MLR_DATA[:, 0], 'COLUMN', BATCH_TARGET, target_run_orientation,
                data_run_format=mimlrdf, data_run_orientation='COLUMN', TARGET_UNIQUES=BATCH_TARGET_UNIQUES,
               Y_OCCUR_HOLDER=BATCH_Y_OCCUR_HOLDER, Y_SUM_HOLDER=BATCH_Y_SUM_HOLDER,
                Y_FREQ_HOLDER=BATCH_Y_FREQ_HOLDER, bypass_validation=bypass_validation).run()

            MI_SCORES = np.insert(MI_SCORES, len(MI_SCORES), total_score, axis=0)
            # END CALCULATE MI & LOG RESULTS #############################################################################

            # CALCULATE INDIVIDUAL COLUMN REGRESSION STASTISTICS & LOG ################################################

            # RETURNED FROM MLRegression
            # XTX_determinant, COEFFS, PREDICTED, P_VALUES, r, R2, R2_adj, F
            R_, R2_, R2_ADJ_, F_ =  mlr.MLRegression(
                                        MI_MLR_DATA,            #DATA
                                        mimlrdo,                #data_given_orientation
                                        BATCH_TARGET_TRANSPOSE, #TARGET
                                        mimlrdo,                #target_given_orientation  # BECAUSE USING BATCH_TARGET_TRANSPOSE
                                        DATA_TRANSPOSE=None,
                                        XTX = None,
                                        XTX_INV = None,
                                        TARGET_TRANSPOSE=BATCH_TARGET,
                                        TARGET_AS_LIST=BATCH_TARGET_TRANSPOSE,
                                        has_intercept=not intcpt_col_idx is None,
                                        intercept_math=not intcpt_col_idx is None,
                                        regularization_factor=0,
                                        safe_matmul=not bypass_validation,
                                        bypass_validation=bypass_validation
                ).run()[4:]

            INDIV_R = np.insert(INDIV_R, len(INDIV_R), R_, axis=0)
            INDIV_RSQ = np.insert(INDIV_RSQ, len(INDIV_RSQ), R2_, axis=0)
            INDIV_RSQ_ADJ = np.insert(INDIV_RSQ_ADJ, len(INDIV_RSQ_ADJ), R2_ADJ_, axis=0)
            INDIV_F = np.insert(INDIV_F, len(INDIV_F), F_, axis=0)











            # END CALCULATE INDIVIDUAL COLUMN REGRESSION STASTISTICS & LOG ################################################

        del Y_OCCUR_HOLDER, Y_SUM_HOLDER, Y_FREQ_HOLDER, MI_MLR_DATA, R_, R2_, R2_ADJ_, F_, BATCH_TARGET_UNIQUES, mimlrdf, mimlrdo

        print(f'MI score calculations complete.')
        ## END RUN CORE MI & MLR ################################################################################################
        # ########################################################################################################################

        # ########################################################################################################################
        # SORT & SELECT #####################################################################################################
        print(f'\nProceeding to sort and select of MI winners...')

        MASTER_SORT_DESC = np.flip(np.argsort(MI_SCORES).reshape((1,-1))[0]).astype(np.int32)
        # MAKE AS LIST FOR EASE OF FINDING AND MOVING INTCPT COL IDX (np.argwhere IS FAILING TO FIND MATCHES HERE)

        # IF DATASET HAD INTERCEPT FORCE IT INTO WINNERS, MOVE IT TO FIRST IN MASTER_SORT_DESC NO MATTER WHAT ITS SCORE WAS.
        # INTERCEPT DOESNT HAVE TO BE FIRST FOR MI SINCE ALL ARE DONE INDEPENDENTLY, BUT ITS BETTER FOR THE FOLLOWING
        # AGGLOMERATIVE MLRegression IF INTERCEPT IS INCLUDED IF WAS IN DATA (AND VICE VERSA) AND IS RUN FIRST, SO THAT ALL
        # AGGLOMORATIONS SEE INTERCEPT NATURALLY WITHOUT CODE GYMNASTICS
        if not intcpt_col_idx is None:
            # IF HAD INTERCEPT, MOVE IT BACK TO FIRST IN THE LIST AFTER ARGSORT OF LAZY_SCORES. HAVE TO LOOK IN AVAILABLE_COLUMNS
            # TO FIND THE IDX OF intcpt_col_idx, THEN MOVE THAT IDX TO FIRST IN MASK
            iiis = AVAILABLE_COLUMNS[MASTER_SORT_DESC].tolist().index(intcpt_col_idx)  # iiis = intcpt_idx_in_sort
            MASTER_SORT_DESC = np.hstack((MASTER_SORT_DESC[iiis], MASTER_SORT_DESC[:iiis], MASTER_SORT_DESC[iiis+1:]))

        MASK = MASTER_SORT_DESC[:min(batch_data_cols, mi_max_columns)]

        MI_SCORES = MI_SCORES[MASK]
        self.WINNING_COLUMNS = AVAILABLE_COLUMNS[MASK]   # MUST DO THSE BEFORE AVAILABLE_COLUMNS IS CHANGED BELOW
        DATA_HEADER_WIP = DATA_HEADER[:, MASK][0]  # BUILD THIS AHEAD OF TIME, GOES IN self.TRAIN_RESULTS

        INDIV_R = INDIV_R[MASK]
        INDIV_RSQ = INDIV_RSQ[MASK]
        INDIV_RSQ_ADJ = INDIV_RSQ_ADJ[MASK]
        INDIV_F = INDIV_F[MASK]
        # AFTER COLLECTING ALL INDIVIDUAL SCORES, THE WINNERS HERE BECOME THE AVAILABLE_COLUMNS FOR FULL, BELOW
        AVAILABLE_COLUMNS = AVAILABLE_COLUMNS[MASK]

        del MASTER_SORT_DESC, MASK, DATA_HEADER, mi_max_columns

        # END SORT & SELECT #####################################################################################################
        # ########################################################################################################################

        # ########################################################################################################################
        # RUN AGGLOMERATIVE MLR ##############################################################################################################

        # MLR ON ALL WINNING MI IDXS AGGLOMERATIVELY, GET R, RSQ, RSQ-ADJ, F, BUILD RESULTS AS COLUMNS #################
        # MAKE THESE AS ROW -- MLRegression WILL ALWAYS RUN AS ROW

        # THESE OBJECTS MUST BE AVAILABLE FOR STACK OF DATA FOR FULL_TRAIN_RESULTS, IF RUNNING MLR AGG OR NOT
        error_winrank = float('inf')
        _ = np.zeros(len(AVAILABLE_COLUMNS), dtype=np.float64)
        self.COEFFS = _.copy()
        P_VALUES = _.copy()
        R_LIST = _.copy()
        R2_LIST = _.copy()
        R2_ADJ_LIST = _.copy()
        F_LIST = _.copy()
        del _

        # DONT DELETE THESE, MUST BE HERE AS PLACEHOLDER IF MLR DOESNT ERROR (THESE WOULD BE OVERWROTE IF ERRORED)
        COEFFS_BACKUP = None
        P_VALUES_BACKUP = None
        R_LIST_BACKUP = None
        R2_LIST_BACKUP = None
        R2_ADJ_LIST_BACKUP = None
        F_LIST_BACKUP = None

        if not mi_bypass_agg:
            print(f'\nCalculating "{["no intercept" if intcpt_col_idx is None else "intercept"][0]}" style MLRegression results '
                  f'for MI winners and building results table... \n')

            ml_regression_errored_last_pass = False

            amdro = 'ROW'  # AGG_MLR_DATA_run_orientation
            for win_rank, win_idx in enumerate(AVAILABLE_COLUMNS):

                INSERT_COLUMN = mlrco.MLRowColumnOperations(BATCH_DATA, data_run_orientation, name='BATCH_DATA',
                    bypass_validation=bypass_validation).return_columns([win_idx], amdro, 'AS_GIVEN')

                if win_rank != 0:
                    AGG_MLR_DATA = mlrco.MLRowColumnOperations(AGG_MLR_DATA, amdro, bypass_validation=bypass_validation,
                        name='AGG_MLR_DATA').insert_column(win_rank, INSERT_COLUMN, amdro)
                else:  # WHEN win_rank IS 0, DO THIS BECAUSE sd.insert_inner() CANT INSERT INTO AN EMPTY :(
                    AGG_MLR_DATA = deepcopy(INSERT_COLUMN) if isinstance(INSERT_COLUMN, dict) else INSERT_COLUMN.copy()

                del INSERT_COLUMN

                # 4-18-22 CALCULATE MLR RESULTS TO GIVE SOME KIND OF ASSESSMENT OF WHAT MI IS ACCOMPLISHING
                # self.COEFFS, P_VALUES ARE OVERWRITTEN EACH CYCLE, PRESENTING FINAL COEFFS AND P_VALUES FOR LAST PASS
                # WITH FULL ASSEMBLAGE OF COLUMNS; R_LIST, R2_LIST, R2_ADJ_LIST, F_LIST ARE APPENDED
                # ON EACH CYCLE AND REPORT THE STEP-WISE CHANGES DURING AGGLOMERATION
                DUM, self.COEFFS, DUM, P_VALUES, R_LIST[win_rank], R2_LIST[win_rank], R2_ADJ_LIST[win_rank], F_LIST[win_rank] = \
                    mlr.MLRegression(
                                    AGG_MLR_DATA,           # DATA
                                    amdro,                  # data_given_orientation
                                    BATCH_TARGET_TRANSPOSE, # TARGET
                                    amdro,                  # target_given_orientation  # BECAUSE USING BATCH_TARGET_TRANSPOSE
                                    DATA_TRANSPOSE=None,
                                    XTX=None,
                                    XTX_INV=None,
                                    TARGET_TRANSPOSE=BATCH_TARGET,
                                    TARGET_AS_LIST=BATCH_TARGET_TRANSPOSE,
                                    has_intercept=not intcpt_col_idx is None,
                                    intercept_math=not intcpt_col_idx is None,
                                    regularization_factor=0,
                                    safe_matmul=not bypass_validation,
                                    bypass_validation=bypass_validation
                    ).run()


                # AT ANY POINT, self.COEFFS, P_VALUES, R_LIST, R2_LIST, R2_ADJ_LIST OR F_LIST COULD GO "ERR" OR "nan"
                # LOOK AT ENTIRE COEFFS AND P_VALUES COLUMNS, SINCE THEY ARE UPDATED IN FULL ON EVERY PASS, BUT ONLY LOOK AT
                # LAST ENTRY FOR R_LIST, R2_LIST, R2_ADJ_LIST, F_LIST, SINCE THESE MIGHT RECOVER TO GOOD VALUES AGAIN
                STR_ZIPPED_UPPER_DATA =  np.char.upper(np.array(
                        list(map(list,
                             itertools.zip_longest(*(self.COEFFS, P_VALUES, [R_LIST[win_rank]], [R2_LIST[win_rank]],
                                                     [R2_ADJ_LIST[win_rank]], [F_LIST[win_rank]]), fillvalue=0)
                        )),
                    dtype=str))

                ml_regression_errored_this_pass = True in map(lambda x: x in ['NAN', 'ERR'], STR_ZIPPED_UPPER_DATA[0])

                if ml_regression_errored_this_pass:
                    if not ml_regression_errored_last_pass:    # IF ERRORED THIS PASS BUT NOT LAST, THIS IS A NEW COTOFF
                        error_winrank = win_rank
                    # elif ml_regression_errored_last_pass: pass   # IF ERRORED THIS PASS AND LAST, JUST CARRY ON
                    ml_regression_errored_last_pass = True

                elif not ml_regression_errored_this_pass:  # IF MLR DID NOT ERROR, RECORD ITS RESULTS BECAUSE IT COULD BE THE LAST GOOD PASS
                    error_winrank = float('inf') # ALLOWS THAT IF SUBSEQUENT PASSES DONT ERROR, CAN GO BACK TO RECOGNIZING THEY ARE GOOD
                    # CREATE BACKUPS THAT HOLD LAST GOOD RESULT (non-ERR, non-nan)
                    ml_regression_errored_last_pass = False
                    COEFFS_BACKUP = deepcopy(self.COEFFS[:win_rank + 1])
                    P_VALUES_BACKUP = deepcopy(P_VALUES[:win_rank + 1])
                    R_LIST_BACKUP = deepcopy(R_LIST[:win_rank + 1])
                    R2_LIST_BACKUP = deepcopy(R2_LIST[:win_rank + 1])
                    R2_ADJ_LIST_BACKUP = deepcopy(R2_ADJ_LIST[:win_rank + 1])
                    F_LIST_BACKUP = deepcopy(F_LIST[:win_rank + 1])

            print(f'\nAGGLOMERATIVE MLRegression ANALYSIS OF MI WINNERS COMPLETE.')

            if error_winrank != float('inf'):
                print(f'\n*** AGGLOMERATIVE MLRegression ANALYSIS ERRORED ON ADDITION OF COLUMN {error_winrank+1}, '
                      f'COLUMN INDEX {AVAILABLE_COLUMNS[error_winrank]}, {DATA_HEADER_WIP[error_winrank]}. ***\n')
            else:
                print(f'\n*** AGGLOMERATIVE MLRegression ANALYSIS RAN SUCCESSFULLY WITHOUT ERROR. ***\n')

            del AVAILABLE_COLUMNS, STR_ZIPPED_UPPER_DATA, BATCH_TARGET, BATCH_TARGET_TRANSPOSE, BATCH_TARGET_AS_LIST, BATCH_DATA, \
                AGG_MLR_DATA, ml_regression_errored_this_pass, ml_regression_errored_last_pass, win_rank, intcpt_col_idx, \
                win_idx, amdro

            # END RUN AGGLOMERATIVE MLR ########################################################################################################
            ####################################################################################################################

        #################################################################################################################
        # COMPILE, CLEAN FULL_TRAIN_RESULTS ##############################################################################

        print(f'Compiling and cleaning results for display...')

        FULL_TRAIN_RESULTS = bemtr.build_empty_mi_train_results(DATA_HEADER_WIP)

        # THESE ARE [[] = COLUMNS] FOR BUILD & return FROM HERE.  TRANSPOSE AND PUT INTO DF FOR ON-SCREEN DISPLAY HERE.
        TRAIN_DATA = np.vstack((MI_SCORES, INDIV_R, INDIV_RSQ,
               INDIV_RSQ_ADJ, INDIV_F, self.COEFFS, P_VALUES, R_LIST, R2_LIST, R2_ADJ_LIST, F_LIST))

        FULL_TRAIN_RESULTS.iloc[:, -11:] = TRAIN_DATA.transpose().astype(object)

        del TRAIN_DATA

        # IF BYPASSED AGGLOMERATIVE MLR, ALL THE CUMUL SCORE COLUMNS STILL CONTAIN ZEROS, OVERWRITE WITH '-'
        if mi_bypass_agg:
            FULL_TRAIN_RESULTS.iloc[:, -6:] = '-'

        # END COMPILE, CLEAN FULL_TRAIN_RESULTS #########################################################################
        #################################################################################################################

        #################################################################################################################
        # COMPILE, CLEAN LAST_GOOD_TRAIN_RESULTS ########################################################################
        if error_winrank == float('inf'):   # IF MLRegression DID NOT ERROR, JUST USE COPY OF FULL RESULTS
            LAST_GOOD_TRAIN_RESULTS = FULL_TRAIN_RESULTS.copy()   # PANDAS, .copy() OK
        else:  # elif error_winrank != float('inf')   MLRegression ERRORED SOMEWHERE
            LAST_GOOD_TRAIN_RESULTS = bemtr.build_empty_mi_train_results(DATA_HEADER_WIP)

            LAST_GOOD_DATA = np.vstack((MI_SCORES[:error_winrank],
                INDIV_R[:error_winrank], INDIV_RSQ[:error_winrank], INDIV_RSQ_ADJ[:error_winrank],
                INDIV_F[:error_winrank], COEFFS_BACKUP, P_VALUES_BACKUP, R_LIST_BACKUP, R2_LIST_BACKUP,
                R2_ADJ_LIST_BACKUP, F_LIST_BACKUP))

            LAST_GOOD_TRAIN_RESULTS.iloc[:, -11:] = LAST_GOOD_DATA.transpose().astype(np.float64)

            del LAST_GOOD_DATA

        # END COMPILE, CLEAN LAST_GOOD_TRAIN_RESULTS ####################################################################
        #################################################################################################################

        del P_VALUES, R_LIST, R2_LIST, R2_ADJ_LIST, F_LIST, COEFFS_BACKUP, P_VALUES_BACKUP, R_LIST_BACKUP, \
            R2_LIST_BACKUP, R2_ADJ_LIST_BACKUP, F_LIST_BACKUP, INDIV_R, INDIV_RSQ, INDIV_RSQ_ADJ, INDIV_F, \
            DATA_HEADER_WIP, MI_SCORES, error_winrank


        # 6/9/23  THIS OK HERE, MI DOESNT HAVE K-FOLD (OR THIS WOULD PUT HARD STOPS INSIDE IT)
        # LET USER DISPLAY OUTPUTS AND SELECT WHAT IS FINAL_RESULTS
        # SET MENU OPTIONS FOR DISPLAY / SELECT
        COMMANDS = {
                    'f': 'display full results',
                    'g': 'display only good results',
                    'j': 'dump full results to excel',
                    'k': 'dump only good results to excel',
                    'a': 'select full results as final results and exit',
                    'b': 'select good results as final results and exit'
        }


        user_disp_return_select = 'BYPASS'

        while True:
            if user_disp_return_select == 'F':                        #'display full results(f)'
                print(f'\n\nFULL WINNING COLUMNS FOR GREEDY MUTUAL INFORMATION:')
                print(FULL_TRAIN_RESULTS)
                print()
            elif user_disp_return_select == 'G':                      #'display only good results(g)'
                print(f'\n\nONLY GOOD WINNING COLUMNS FOR GREEDY MUTUAL INFORMATION:')
                print(LAST_GOOD_TRAIN_RESULTS)
                print()
            elif user_disp_return_select == 'J':                      #'dump full results to excel(j)'
                tred.train_results_excel_dump(FULL_TRAIN_RESULTS, 'FULL_TRAIN_RESULTS')

            elif user_disp_return_select == 'K':                      #'dump only good results to excel(k)'
                tred.train_results_excel_dump(LAST_GOOD_TRAIN_RESULTS, 'LAST_GOOD_TRAIN_RESULTS')

            elif user_disp_return_select == 'A':                      #'select full results as final results(a)'
                if vui.validate_user_str(f'\nUser selected FULL RESULTS as final results. Accept? (y/n) > ', 'YN') == 'Y':
                    self.TRAIN_RESULTS = FULL_TRAIN_RESULTS
                    break
            elif user_disp_return_select == 'B':                      #'select good results as final results(b)'
                if vui.validate_user_str(f'\nUser selected only GOOD RESULTS as final results. Accept? (y/n) > ', 'YN') == 'Y':
                    self.TRAIN_RESULTS = LAST_GOOD_TRAIN_RESULTS
                    self.WINNING_COLUMNS = self.WINNING_COLUMNS[:len(LAST_GOOD_TRAIN_RESULTS)]
                    self.COEFFS = self.COEFFS[:len(LAST_GOOD_TRAIN_RESULTS)]
                    break

            # PIZZA UNHASH THESE TO TEST
            # self.TRAIN_RESULTS = FULL_TRAIN_RESULTS
            # break

            user_disp_return_select = dmp.DictMenuPrint(COMMANDS, disp_len=140).select(f'\nSelect menu option')

        del FULL_TRAIN_RESULTS, LAST_GOOD_TRAIN_RESULTS, COMMANDS, user_disp_return_select
    # DATA & TARGET ARE PROCESSED AS [[] = COLUMNS] IN MutualInformation

    # END init ###############################################################################################################
    # ########################################################################################################################
    # ########################################################################################################################


    def return_fxn(self):
        return self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS  # TRAIN_RESULTS MUST BE RETURNED AS COLUMNS


    def run(self):
        return self.return_fxn()








































if __name__ == '__main__':

    # 5/10/23 MODULE & TEST CODE ARE GOOD

    # CTRL-F "PIZZA" TO HASH/UNHASH FOR TEST

    # 5/8/23 ONLY TESTING SORTED SCORES AT THIS POINT
    # CANT TEST ACT/EXP MICEO OBJECTS BECAUSE NOT YET AVAILABLE AS selfs IN MICoreRunCode, BUT PROBABLY SHOULD BE OK BECAUSE
    # MutualInformation BUILDS X & Y MICEOs VIA MICrossEntopyObjects, WHICH IS INDEPENDENTLY TESTED

    from scipy.special import logsumexp

    this_module = gmn.get_module_name(str(sys.modules[__name__]))
    fxn = 'tests'

    header_dum = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    rows = 100
    columns = 2
    categories = 5

    # # INT
    def create_int_data(columns, rows, header_dum):
        DATA = np.random.randint(0,categories,(columns,rows),dtype=int)
        DATA_HEADER = np.fromiter((header_dum[_]+'_JUNK' for _ in range(columns)), dtype='<U100').reshape((1,-1))
        TARGET = np.random.randint(0, 2, (1,rows)).reshape((1,-1))
        TARGET_HEADER = np.array(['TARGET'], dtype=str).reshape((1,-1))

        return DATA, DATA_HEADER, TARGET, TARGET_HEADER
    #
    # BIN
    def create_bin_data(columns, rows, header_dum):
        DATA = np.random.randint(0,2,(columns, rows))
        DATA_HEADER = np.fromiter((header_dum[_] for _ in range(columns)), dtype='<U100').reshape((1,-1))
        TARGET = np.random.randint(0, 2, (1,rows)).reshape((1,-1))
        TARGET_HEADER = np.array(['TARGET'], dtype=str).reshape((1,-1))

        return DATA, DATA_HEADER, TARGET, TARGET_HEADER

    # SPARSE_DICT
    def create_sd_data(columns, rows, header_dum):
        DATA = sd.create_random_py_int(0, categories, (columns, rows), 50)
        DATA_HEADER = np.fromiter((header_dum[_] for _ in range(columns)), dtype='<U100').reshape((1,-1))
        TARGET = np.random.randint(0, 2, (1,rows)).reshape((1,-1))
        TARGET_HEADER = np.array(['TARGET'], dtype=str).reshape((1,-1))

        return DATA, DATA_HEADER, TARGET, TARGET_HEADER

   # FROM FILE
    def create_data_from_file():
        from read_write_file.generate_full_filename import base_path_select as bps
        basepath = bps.base_path_select()
        READ_DATA = pd.read_excel(basepath + r'MI_SCRATCH.xlsx', sheet_name='DATA').to_numpy(dtype=int)
        READ_HEADER = [['TARGET'] + [header_dum[_] for _ in range(10)]]
        DATA = pd.DataFrame(data=READ_DATA, columns=READ_HEADER)
        TARGET = deepcopy(DATA['TARGET']).to_numpy().transpose()
        TARGET_HEADER = [['TARGET']]
        DATA_HEADER = [[_[0] for _ in DATA.keys()[1:]]]
        DATA = DATA.drop(columns=['TARGET'], axis=0).to_numpy().transpose()
        DATA = sd.zip_list(DATA)

        return DATA, DATA_HEADER, TARGET, TARGET_HEADER



    def print_setup(_dtype, max_columns, intcpt_col_idx, orientation):
        print(f'\nTYPE = {_dtype} AS {["DICT" if _dtype == "SD" else "ARRAY"][0]}')
        print(f'MAX COLUMNS = {max_columns} (of {columns*categories + 1 if intcpt_col_idx is not False else 0} '
              f'{"with" if intcpt_col_idx is not False else "without"}' + f' intercept)')
        print(f'intcpt_col_idx = {intcpt_col_idx}')
        print(f'data orientation = {data_given_orientation}')
        print(f'target orientation = {target_given_orientation}')


    def KILL(name, _format=None, ACT_OBJ=None, EXP_OBJ=None, act=None, exp=None):
        if not _format is None and not ACT_OBJ is None and not EXP_OBJ is None:
            if (_format=='ARRAY' and not np.array_equiv(ACT_OBJ, EXP_OBJ)) or (_format=='SPARSE_DICT' and not sd.core_sparse_equiv(ACT_OBJ, EXP_OBJ)):
                wls.winlinsound(444, 1000)
                print(f'\nACTUAL {name}:'); print(ACT_OBJ); print(f'\nEXPECTED {name}:'); print(EXP_OBJ)
                raise Exception(f'*** {name} ACTUAL AND EXPECTED NOT EQUAL ***')
        elif not act is None and not exp is None:
            wls.winlinsound(444, 1000)
            raise Exception(f'*** ACTUAL {name} ({act}) DOES NOT EQUAL EXPECTED ({exp}) ***')
        else:
            wls.winlinsound(444, 1000)
            raise Exception(f'*** YOU SCREWED UP THE EXCEPTION HANDLER!!!! ****')

    MASTER_BYPASS_VALIDATION = [True, False]
    MASTER_BATCH_METHOD = ['M', 'B']
    DTYPE = ['BIN', 'INT']
    DATA_FORMAT = ['ARRAY', 'SPARSE_DICT']
    TARGET_FORMAT = ['ARRAY', 'SPARSE_DICT']
    TARGET_TRANSPOSE_IS_GIVEN = [True, False]
    TARGET_AS_LIST_IS_GIVEN = [True, False]
    TARGET_UNIQUES_IS_GIVEN = [True, False]
    Y_OCCUR_IS_PASSED = [True, False]
    Y_SUM_IS_PASSED = [True, False]
    Y_FREQ_IS_PASSED = [True, False]
    MI_MAX_COLUMNS = [15, 20, 25]
    DATA_ORIENT = ['COLUMN', 'ROW']
    TARGET_ORIENT = ['COLUMN', 'ROW']
    INTCPT = [True, False]

    total_trials = np.product(list(map(len, (MASTER_BYPASS_VALIDATION, MASTER_BATCH_METHOD, DTYPE, DATA_FORMAT,
        TARGET_FORMAT, MI_MAX_COLUMNS, DATA_ORIENT, TARGET_ORIENT, TARGET_TRANSPOSE_IS_GIVEN, TARGET_AS_LIST_IS_GIVEN,
        TARGET_UNIQUES_IS_GIVEN, Y_OCCUR_IS_PASSED, Y_SUM_IS_PASSED, Y_FREQ_IS_PASSED, INTCPT))))

    ctr = 0
    for bypass_validation in MASTER_BYPASS_VALIDATION:
        for batch_method in MASTER_BATCH_METHOD:
            if batch_method == 'B': batch_size = ...
            elif batch_method == 'M': batch_size = 50
            for _dtype in DTYPE:
                for data_given_format in DATA_FORMAT:
                    for target_given_format in TARGET_FORMAT:
                        for max_columns in MI_MAX_COLUMNS:
                            for data_given_orientation in DATA_ORIENT:
                                for target_given_orientation in TARGET_ORIENT:
                                    for target_transpose_is_given in TARGET_TRANSPOSE_IS_GIVEN:
                                        for target_as_list_is_given in TARGET_AS_LIST_IS_GIVEN:
                                            for target_uniques_is_given in TARGET_UNIQUES_IS_GIVEN:
                                                for y_occur_is_passed in Y_OCCUR_IS_PASSED:
                                                    for y_sum_is_passed in Y_SUM_IS_PASSED:
                                                        for y_freq_is_passed in Y_FREQ_IS_PASSED:
                                                            for intcpt in INTCPT:

                                                                ctr += 1
                                                                print(f'\n\nRunning trial {ctr} of {total_trials}')

                                                                if _dtype == 'INT': DATA, DATA_HEADER, TARGET, TARGET_HEADER = create_int_data(columns, rows, header_dum)
                                                                elif _dtype == 'BIN': DATA, DATA_HEADER, TARGET, TARGET_HEADER = create_bin_data(columns, rows, header_dum)
                                                                elif _dtype == 'SD': DATA, DATA_HEADER, TARGET, TARGET_HEADER = create_sd_data(columns, rows, header_dum)

                                                                ####################################################################################################
                                                                if intcpt is True:
                                                                    if isinstance(DATA, (np.ndarray, list, tuple)):
                                                                        DATA = np.insert(DATA, len(DATA), 1, axis=0)
                                                                    elif isinstance(DATA, dict):
                                                                        DATA = sd.append_outer(
                                                                            DATA, np.fromiter((1 for _ in range(sd.inner_len_quick(DATA))), dtype=float)
                                                                        )

                                                                    DATA_HEADER = np.insert(DATA_HEADER, len(DATA_HEADER[0]), 'INTERCEPT', axis=1)

                                                                if intcpt is False:
                                                                    intcpt_col_idx = None
                                                                    _has_intercept = False
                                                                    _columns = columns
                                                                elif intcpt is True:
                                                                    intcpt_col_idx = columns
                                                                    _has_intercept = True
                                                                    _columns = columns + 1

                                                                ###################################################################################################
                                                                # ORIENT OBJECTS ##################################################################################

                                                                MASTER_RETURN_OBJECTS = ['DATA', 'TARGET']
                                                                if target_transpose_is_given: MASTER_RETURN_OBJECTS += ['TARGET_TRANSPOSE']
                                                                if target_as_list_is_given: MASTER_RETURN_OBJECTS += ['TARGET_AS_LIST']

                                                                ObjectClass = mloo.MLObjectOrienter(
                                                                                                 DATA=DATA,
                                                                                                 data_given_orientation='COLUMN',
                                                                                                 data_return_orientation=data_given_orientation,
                                                                                                 data_return_format=data_given_format,

                                                                                                 DATA_TRANSPOSE=None,
                                                                                                 data_transpose_given_orientation=None,
                                                                                                 data_transpose_return_orientation=data_given_orientation,
                                                                                                 data_transpose_return_format=data_given_format,

                                                                                                 XTX=None,
                                                                                                 xtx_return_format='ARRAY',

                                                                                                 XTX_INV=None,
                                                                                                 xtx_inv_return_format=None,

                                                                                                 target_is_multiclass=None,
                                                                                                 TARGET=TARGET,
                                                                                                 target_given_orientation='COLUMN',
                                                                                                 target_return_orientation=target_given_orientation,
                                                                                                 target_return_format=target_given_format,

                                                                                                 TARGET_TRANSPOSE=None,
                                                                                                 target_transpose_given_orientation=None,
                                                                                                 target_transpose_return_orientation=target_given_orientation,
                                                                                                 target_transpose_return_format=target_given_format,

                                                                                                 TARGET_AS_LIST=None,
                                                                                                 target_as_list_given_orientation=None,
                                                                                                 target_as_list_return_orientation=target_given_orientation,

                                                                                                 RETURN_OBJECTS=MASTER_RETURN_OBJECTS,

                                                                                                 bypass_validation=True,
                                                                                                 calling_module=this_module,
                                                                                                 calling_fxn=fxn)

                                                                DATA = ObjectClass.DATA
                                                                TARGET = ObjectClass.TARGET
                                                                TARGET_TRANSPOSE = ObjectClass.TARGET_TRANSPOSE
                                                                TARGET_AS_LIST = ObjectClass.TARGET_AS_LIST

                                                                if not target_uniques_is_given: TARGET_UNIQUES = None
                                                                else:
                                                                    if target_given_format == 'ARRAY': TARGET_UNIQUES = np.unique(TARGET).reshape((1,-1))
                                                                    elif target_given_format == 'SPARSE_DICT': TARGET_UNIQUES = sd.return_uniques(TARGET).reshape((1,-1))

                                                                    if np.array_equiv(TARGET_UNIQUES, TARGET_UNIQUES.astype(np.int32)):
                                                                        TARGET_UNIQUES = TARGET_UNIQUES.astype(np.int32)

                                                                # END ORIENT OBJECTS ##############################################################################
                                                                ###################################################################################################


                                                                print_setup(_dtype, max_columns, intcpt_col_idx, data_given_orientation)

                                                                reported_cols = min(_columns, max_columns)

                                                                print(f'type = {_dtype}, orient = {data_given_orientation} > OUTPUT SHOULD BE {reported_cols} COLUMNS {["WITH" if _has_intercept else "WITHOUT"][0]} INTERCEPT AT TOP')

                                                                # _ = input(f'\nPAUSED TO LOOK AT SETUP. HIT ENTER TO CONTINUE > ')




                                                                #############################################################################################################
                                                                #############################################################################################################
                                                                # BUILD TARGET MICEO OBJECTS TO CALCULATE EXP_MIS AND CONDITIONALLY PASS AS KWARGS TO MICoreRUnCode #########

                                                                YObjsClass = miceo.MICrossEntropyObjects(TARGET,
                                                                                                         UNIQUES=TARGET_UNIQUES,
                                                                                                         return_as='ARRAY',
                                                                                                         bypass_validation=False
                                                                                                         )
                                                                PASSED_Y_OCCUR = YObjsClass.OCCURRENCES
                                                                PASSED_Y_SUM = YObjsClass.SUMS
                                                                PASSED_Y_FREQ = YObjsClass.FREQ

                                                                # END BUILD TARGET MICEO OBJECTS TO CALCULATE EXP_MIS AND CONDITIONALLY PASS AS KWARGS TO MICoreRUnCode #####E
                                                                ##############################################################################################################
                                                                ##############################################################################################################


                                                                ###############################################################################################################
                                                                # CALCULATE EXPECTED MI #######################################################################################

                                                                # ONLY CALCULATE EXP_SCORES IF RUNNING MINIBATCH. NO WAY TO KNOW AHEAD OF TIME WHAT ROWS WILL BE PULLED BY MINIBATCH
                                                                if batch_method == 'B':
                                                                    EXP_SCORES = np.empty(_columns, dtype=np.float64)
                                                                    for col_idx in range(_columns):

                                                                        WIP_DATA = mlrco.MLRowColumnOperations(DATA, data_given_orientation, 'WIP_DATA', bypass_validation=True)
                                                                        WIP_DATA = WIP_DATA.return_columns([col_idx], return_format='ARRAY', return_orientation='COLUMN')

                                                                        # GENERATE THESE TO MANUALLY CALCULATE EXP_MIs (WE KNOW THIS MUST BE CORRECT, PASSES TESTS)
                                                                        XObjsClass = miceo.MICrossEntropyObjects(WIP_DATA,
                                                                                                                 UNIQUES=None,
                                                                                                                 return_as='ARRAY',
                                                                                                                 bypass_validation=False
                                                                                                                 )

                                                                        EXP_X_OCCUR = XObjsClass.OCCURRENCES
                                                                        EXP_X_SUM = XObjsClass.SUMS
                                                                        EXP_X_FREQ = XObjsClass.FREQ

                                                                        exp_score = 0
                                                                        for x_idx in range(len(EXP_X_OCCUR)):
                                                                            for y_idx in range(len(PASSED_Y_OCCUR)):
                                                                                p_x_y = np.matmul(EXP_X_OCCUR[x_idx].astype(np.float64),
                                                                                                  PASSED_Y_OCCUR[y_idx].astype(np.float64),
                                                                                                  dtype=np.float64
                                                                                                  ) / PASSED_Y_SUM[0][y_idx].astype(np.float64)

                                                                                try:
                                                                                    with np.errstate(divide='raise'):
                                                                                        exp_score += p_x_y * (np.log10(logsumexp(p_x_y)) - np.log10(
                                                                                            logsumexp(EXP_X_FREQ[0][x_idx])) - np.log10(logsumexp(PASSED_Y_FREQ[0][y_idx])))

                                                                                except:
                                                                                    if RuntimeWarning or FloatingPointError: pass

                                                                            EXP_SCORES[col_idx] = exp_score

                                                                    if _has_intercept:
                                                                        EXP_SCORES = np.hstack((EXP_SCORES[-1], list(reversed(sorted(EXP_SCORES[:-1])))))
                                                                    else:
                                                                        EXP_SCORES.sort()
                                                                        EXP_SCORES = np.flip(EXP_SCORES)

                                                                # END CALCULATE EXPECTED MI ################################################################################
                                                                ############################################################################################################

                                                                TestClass = MICoreRunCode(DATA, DATA_HEADER, TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST,
                                                                    data_given_orientation, target_given_orientation, batch_method, batch_size, max_columns,
                                                                    bypass_agg, intcpt_col_idx,
                                                                    PASSED_Y_OCCUR if y_occur_is_passed else None,
                                                                    PASSED_Y_SUM if y_sum_is_passed else None,
                                                                    PASSED_Y_FREQ if y_freq_is_passed else None,
                                                                    bypass_validation)

                                                                WINNING_COLUMNS, COEFFS, TRAIN_RESULTS = TestClass.run()

                                                                print(f'{_dtype}, {data_given_orientation} > OUTPUT SHOULD BE {reported_cols} COLUMNS {["WITH" if _has_intercept else "WITHOUT"][0]} INTERCEPT AT TOP')
                                                                print()
                                                                print(TRAIN_RESULTS)
                                                                print()

                                                                ACT_SCORES = TRAIN_RESULTS.to_numpy().transpose()[0]

                                                                # ONLY CALCULATE EXP_SCORES IF RUNNING MINIBATCH. NO WAY TO KNOW AHEAD OF TIME WHAT ROWS WILL BE PULLED BY MINIBATCH
                                                                if batch_method == 'B':
                                                                    KILL('SCORES', _format='ARRAY', ACT_OBJ=ACT_SCORES, EXP_OBJ=EXP_SCORES, act=None, exp=None)


                                                                ###################################################################################################
                                                                # TEST PASSED TARGET MICEOs ARE EQUAL TO ACTUAL INSIDE MICoreRunCode ##############################

                                                                # Y_ACT IS NOT AVALABLE AS self THRU MICoreRunCode YET :I
                                                                # ACT_Y_OCCUR = TestClass.Y_OCCUR
                                                                # ACT_Y_SUMS = TestClass.Y_SUM
                                                                # ACT_Y_FREQ = TestClass.Y_FREQ

                                                                # KILL(name, _format=target_given_format, ACT_OBJ=ACT_Y_OCCUR, EXP_OBJ=PASSED_Y_OCCUR, act=None, exp=None)
                                                                # KILL(name, _format=target_given_format, ACT_OBJ=ACT_Y_SUMS, EXP_OBJ=PASSED_Y_SUM, act=None, exp=None)
                                                                # KILL(name, _format=target_given_format, ACT_OBJ=ACT_Y_FREQ, EXP_OBJ=PASSED_Y_FREQ, act=None, exp=None)

                                                                # END TEST PASSED TARGET MICEOs ARE EQUAL TO ACTUAL INSIDE MICoreRunCode ##########################
                                                                ###################################################################################################


                                                                ###################################################################################################
                                                                # TEST EXP X MICEOs ARE EQUAL TO ACTUAL INSIDE MICoreRunCode ######################################
                                                                #  IN MICoreRunCode

                                                                # THIS WOULD HAVE TO BE DONE ITERATIVELY, THESE ARE CALCULATED FOR EACH COLUMN IN DATA.  WOULD HAVE
                                                                # TO GET EXP AND ACT FOR EACHT COLUMN AND COMPARE.
                                                                # MOOT ANYWAY BECAUSE X_ACT IS NOT AVALABLE AS self THRU MICoreRunCode YET :I

                                                                # BOGUS
                                                                # ACT_X_OCCUR = TestClass.X_OCCUR
                                                                # ACT_X_SUMS = TestClass.X_SUM
                                                                # ACT_X_FREQ = TestClass.X_FREQ

                                                                # BOGUS
                                                                # KILL(name, _format=data_given_orientation, ACT_OBJ=ACT_X_OCCUR, EXP_OBJ=EXP_Y_OCCUR, act=None, exp=None)
                                                                # KILL(name, _format=data_given_orientation, ACT_OBJ=ACT_X_SUMS, EXP_OBJ=EXP_Y_SUM, act=None, exp=None)
                                                                # KILL(name, _format=data_given_orientation, ACT_OBJ=ACT_X_FREQ, EXP_OBJ=EXP_Y_FREQ, act=None, exp=None)

                                                                # END TEST EXP X MICEOs ARE EQUAL TO ACTUAL INSIDE MICoreRunCode ##################################
                                                                ###################################################################################################



                                                                if _has_intercept:
                                                                    if TRAIN_RESULTS.index[0] != "INTERCEPT":
                                                                        wls.winlinsound(444,1000)
                                                                        print_setup(_dtype, max_columns, intcpt_col_idx, data_given_orientation)
                                                                        print(TRAIN_RESULTS)
                                                                        raise Exception(f'SUPPOSED TO HAVE INTERCEPT AT TOP OF RESULTS BUT DOESNT')

                                                                    '''
                                                                    # TEST IF INTERCEPT RUN ALONE IS ERRORING IN MLR
                                                                    # 5/5/23 --- TAKING THIS OUT BECAUSE NaNs ARE IN THE INTERCEPT RESULT & CONSTANTLY BLOWING UP
                                                                    # DONT KNOW IF ITS IMPORTANT TO NOT HAVE NaNs IN THE RESULT DOWN THE ROAD, BUT HOW COULD IT BE AVOIDED?
                                                                    for outer_df_column in ['COLUMN', 'INDIV', 'FINAL', 'CUMUL'][1:]:
                                                                        for inner_df_column in ['NAME', 'MI SCORE', 'R', 'R2', 'ADJ R2', 'F', 'COEFFS', 'p VALUE'][1:]:
                                                                            if inner_df_column not in TRAIN_RESULTS[outer_df_column]:
                                                                                continue
                                                                            _ = str(TRAIN_RESULTS[outer_df_column][inner_df_column][0]).upper()
                                                                            if _ in ['NAN', 'ERR']:
                                                                                wls.winlinsound(444, 1000)
                                                                                # raise Exception(f'ERRORED IN R, RSQ, ADJ_RSQ, OR F IN INTERCEPT ROW')
                                                                                print(f'\nERRORED IN {outer_df_column}.{inner_df_column} ({_}) IN INTERCEPT ROW')
                                                                                print_setup(_dtype, max_columns, intcpt_col_idx, data_given_orientation)
                                                                                _ = input(f'\nHIT ENTER TO CONTINUE > ')
                                                                                break
                                                                        else: continue
                                                                        break
                                                                    '''

                                                                if not _has_intercept and "INTERCEPT" in TRAIN_RESULTS.index:
                                                                    wls.winlinsound(444, 1000)
                                                                    print_setup(_dtype, max_columns, intcpt_col_idx, data_given_orientation)
                                                                    raise Exception(f'NOT SUPPOSED TO HAVE INTERCEPT IN RESULTS BUT DOES')

                                                                if len(TRAIN_RESULTS) != reported_cols:
                                                                    wls.winlinsound(444, 1000)
                                                                    print_setup(_dtype, max_columns, intcpt_col_idx, data_given_orientation)
                                                                    raise Exception(f'SUPPOSED TO HAVE {max_columns} COLUMNS BUT HAS {len(TRAIN_RESULTS)}')







                                                                # BYPASS ALL THIS
                                                                continue

                                                                _ = input(f'\nPAUSED TO LOOK AT RESULTS. HIT ENTER TO CONTINUE > ')

                                                                print(400*'*')
                                                                print(f'\nRUNNING {["WITH" if intcpt else "WITHOUT"][0]} INTERCEPT!')
                                                                print(f'\nTRAIN RESULTS:')
                                                                print(TRAIN_RESULTS)

                                                                DATA_DICT = {}

                                                                if isinstance(DATA, (list, tuple, np.ndarray)):
                                                                    for idx in range(len(TARGET)): DATA_DICT[TARGET_HEADER[0][idx]] = deepcopy(TARGET[idx])
                                                                elif isinstance(DATA, dict):
                                                                    for idx in range(len(TARGET)): DATA_DICT[TARGET_HEADER[0][idx]] = sd.zip_list([deepcopy(TARGET[idx])])[0]

                                                                for idx in WINNING_COLUMNS:
                                                                    DATA_DICT[DATA_HEADER[0][idx]] = deepcopy(DATA[idx])

                                                                DF = pd.DataFrame(DATA_DICT).fillna(0)
                                                                print()
                                                                print(DF)

                                                                if vui.validate_user_str(f'Dump DATA to file? (y/n) > ', 'YN') == 'Y':
                                                                    base_path = bps.base_path_select()
                                                                    file_name = fe.filename_wo_extension()
                                                                    print(f'\nSaving file to {base_path+file_name+".xlsx"}....')
                                                                    pd.DataFrame.to_excel(DF,
                                                                                 excel_writer=base_path + file_name + '.xlsx',
                                                                                 float_format='%.2f',
                                                                                 startrow=1,
                                                                                 startcol=1,
                                                                                 merge_cells=False
                                                                                 )
                                                                    print('Done.')


    print(f'\033[92m\n*** ALL FUNCTIONALITY & ACCURACY TESTS PASSED ***\n\033[0m')
    for _ in range(3): wls.winlinsound(888, 500); time.sleep(1)

    print(f'\nPIZZA DONT FORGET TO GO UNHASH')
    print(f'# self.TRAIN_RESULTS = FULL_TRAIN_RESULTS')
    print(f'# break')



























