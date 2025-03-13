import sys, inspect, time
import sparse_dict as sd
from data_validation import arg_kwarg_validater as akv
from debug import get_module_name as gmn
from MLObjects import MLObject as mlo
from general_data_ops import NoImprov as ni
from ML_PACKAGE.MLREGRESSION import MLRegression as mlr
from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION import ApexGMLR as agmlr
from general_data_ops import get_shape as gs



class LazyAggGMLR(agmlr.ApexGMLR):

    def __init__(self, DATA, DATA_HEADER, data_given_orientation, TARGET, target_given_orientation, AVAILABLE_COLUMNS=None,
        max_columns=None, intcpt_col_idx=None, rglztn_fctr=None, score_method=None, TRAIN_RESULTS=None, TARGET_TRANSPOSE=None,
        TARGET_AS_LIST=None, data_run_orientation='ROW', target_run_orientation='ROW', conv_kill=None, pct_change=None,
        conv_end_method=None, bypass_validation=None):

        # MUST BE BEFORE super ##########################################################################################
        this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = inspect.stack()[0][3]

        if not AVAILABLE_COLUMNS is None: AVAILABLE_COLUMNS = AVAILABLE_COLUMNS[:max_columns]

        self.score_method = akv.arg_kwarg_validater(score_method, 'score_method', ['Q', 'R', 'A', 'F', None],
                                                                            this_module, fxn, return_if_none='Q')

        # BEAR
        self.train_results_given = not TRAIN_RESULTS is None

        self.conv_kill = conv_kill
        self.pct_change = pct_change if not pct_change is None else 0
        self.conv_end_method = conv_end_method if not conv_end_method is None else 'PROMPT'
        # END MUST BE BEFORE super ######################################################################################

        super().__init__(DATA, DATA_HEADER, data_given_orientation, TARGET, target_given_orientation, AVAILABLE_COLUMNS=AVAILABLE_COLUMNS,
            max_columns=max_columns, intcpt_col_idx=intcpt_col_idx, rglztn_fctr=rglztn_fctr, TRAIN_RESULTS=TRAIN_RESULTS,
            TARGET_TRANSPOSE=TARGET_TRANSPOSE, TARGET_AS_LIST=TARGET_AS_LIST, data_run_orientation=data_run_orientation,
            target_run_orientation=target_run_orientation, bypass_validation=bypass_validation, calling_module=this_module)

        del self.conv_kill, self.pct_change, self.conv_end_method
    # END init #########################################################################################################
    ####################################################################################################################
    ####################################################################################################################


    def core_run(self, DATA, DATA_HEADER, data_run_orientation, TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST,
                 target_run_orientation, AVAILABLE_COLUMNS, max_columns, intcpt_col_idx, rglztn_fctr, bypass_validation,
                 this_module):

        fxn = inspect.stack()[0][3]

        if not self.train_results_given:
            self.TRAIN_RESULTS = self.TRAIN_RESULTS.iloc[AVAILABLE_COLUMNS, :]
        del self.train_results_given


        # BUILD A DONOR (BATCH_DATA) AND RECEIVER (LAZY_AGG_DATA_WIP) CLASSES TO ################################
        # DONATE AND AGGLOMERATE WINNING COLUMNS ################################################################

        BatchData_Donor = mlo.MLObject(
                                        DATA,
                                        data_run_orientation,
                                        name='BATCH_DATA',
                                        return_orientation='AS_GIVEN',
                                        return_format='AS_GIVEN',
                                        bypass_validation=bypass_validation,
                                        calling_module=this_module,
                                        calling_fxn=fxn
        )

        FIRST_INSERT_COLUMN = BatchData_Donor.return_columns([AVAILABLE_COLUMNS[0]],
                                                             return_orientation='AS_GIVEN', return_format='AS_GIVEN')

        # BUILD LazyAggData_Receiver FROM FIRST COLUMN TO GO INTO LAZY AGG MLR (INTERCEPT IF USED)

        LazyAggData_Receiver = mlo.MLObject(
                                            FIRST_INSERT_COLUMN,
                                            data_run_orientation,
                                            name='LAZY_AGG_DATA_WIP',
                                            return_orientation='AS_GIVEN',
                                            return_format='AS_GIVEN',
                                            bypass_validation=bypass_validation,
                                            calling_module=this_module,
                                            calling_fxn=fxn
        )

        del FIRST_INSERT_COLUMN

        # END BUILD A DONOR (BATCH_DATA) AND RECEIVER (LAZY_AGG_WIP_DATA) CLASSES TO ############################
        # DONATE AND AGGLOMERATE WINNING COLUMNS ################################################################

        # STUFF FOR NoImprov
        ni_ctr_init = 0
        ni_ctr = 0
        best_value = float('-inf')


        for lazy_agg_itr, col_idx in enumerate(AVAILABLE_COLUMNS):

            actv_hdr = DATA_HEADER[col_idx]

            if lazy_agg_itr == 0:  # ON FIRST PASS, DO NOT INSERT A COLUMN BECAUSE ReceiverClass WAS SEEDED WITH 1st COLUMN
                if not intcpt_col_idx is None:
                    # IF ON FIRST ITR AND INTCPT IS PRESENT (IF PRESENT, MUST BE FIRST), SET ALL CUMUL SCORES TO ZERO AND
                    # BYPASS MLR
                    self.TRAIN_RESULTS.loc[actv_hdr, ('CUMUL', 'F')] = 0
                    self.TRAIN_RESULTS.loc[actv_hdr, ('CUMUL', 'R')] = 0
                    self.TRAIN_RESULTS.loc[actv_hdr, ('CUMUL', 'R2')] = 0
                    self.TRAIN_RESULTS.loc[actv_hdr, ('CUMUL', 'ADJ R2')] = 0
                    ni_ctr_init += 1
                    continue
                elif intcpt_col_idx is None:
                    pass     # JUST TAKE THE COLUMN THAT IS ALREADY SEEDED IN Receiver

            else: # ON SUBSEQUENT PASSES, INSERT CORRESPONDING COLUMN AND GET NEW LAZY_AGG_DATA_WIP
                INSERT_COLUMN = BatchData_Donor.return_columns([col_idx], return_orientation='AS_GIVEN',
                                                               return_format='AS_GIVEN')

                LazyAggData_Receiver.insert_column(lazy_agg_itr, INSERT_COLUMN, data_run_orientation)

                del INSERT_COLUMN

            if data_run_orientation=='COLUMN':
                LAZY_AGG_DATA_AS_COLUMN = LazyAggData_Receiver.return_as_column()
                LAZY_AGG_DATA_AS_ROW = LazyAggData_Receiver.return_as_row()
            elif data_run_orientation=='ROW':
                LAZY_AGG_DATA_AS_COLUMN = None
                LAZY_AGG_DATA_AS_ROW = LazyAggData_Receiver.return_as_row()

            # BEAR
            # 6/4/23 - A REAL CONUNDRUM
            # WHEN PASSING SDs TO MLR BELOW, THE RESULTING TRAIN_RESULTS TABLE IS COMING OUT INCORRECT (IS FAILING AGAINST
            # REFEREE TABLE IN TEST). THESE ARE THE THINGS THAT ARE PASSING AGAINST REFEREE TABLES IN TEST: NP ARRAYS,
            # SDs WITH NO CATEGORICALS, BUT SDs WITH CATS ARE FAILING. HOWEVER, WHEN THESE ARE CONVERTED TO NP ARRAYS WITH
            # THE CODE BELOW, THEY PASS, SO IT ISNT THE DATA WITHIN. SO THE PROBLEM SEEMS TO TRACE BACK TO THE PASSING OF
            # CATEGORICAL SDs TO MLRegression. TESTS OF MLRegression() AND LazyGMLR() USING CATEGORICAL SDs DO NOT
            # INDICATE A PROBLEM. SO FOR NOW, TO GET AROUND THIS HERE, ALWAYS EXPANDING SDs TO ARRAYS FOR THE PASS TO
            # MLRegression().

            if isinstance(LAZY_AGG_DATA_AS_ROW, dict):
                LAZY_AGG_DATA_AS_ROW = sd.unzip_to_ndarray_float64(LAZY_AGG_DATA_AS_ROW)[0]
            if isinstance(LAZY_AGG_DATA_AS_COLUMN, dict):
                LAZY_AGG_DATA_AS_COLUMN = sd.unzip_to_ndarray_float64(LAZY_AGG_DATA_AS_COLUMN)[0]

            # COEFFS AND P_VALUES ARE GENERATED EVERY lazy_agg_itr BUT ARE ONLY NEEDED FOR LAST PASS W FULL WINNERS
            # RETURNED FROM MLRegression:  XTX_determinant, self.COEFFS, PREDICTED, P_VALUES, r, R2, R2_adj, F

            DUM, COEFF_HOLDER, DUM, P_VALUE_HOLDER, R_, R2_, R2_ADJ_, F_ = \
                mlr.MLRegression(LAZY_AGG_DATA_AS_ROW,
                                 'ROW',
                                 TARGET if target_run_orientation=='ROW' else TARGET_TRANSPOSE,
                                 'ROW',
                                 DATA_TRANSPOSE=LAZY_AGG_DATA_AS_COLUMN,
                                 TARGET_TRANSPOSE=TARGET_TRANSPOSE if target_run_orientation=='ROW' else TARGET,
                                 TARGET_AS_LIST=TARGET_AS_LIST if target_run_orientation=='ROW' else TARGET_AS_LIST.transpose(),
                                 XTX=None,
                                 XTX_INV=None,
                                 has_intercept=False if intcpt_col_idx is None else True,
                                 intercept_math=False if intcpt_col_idx is None else True,
                                 regularization_factor=rglztn_fctr,
                                 safe_matmul=not bypass_validation,
                                 bypass_validation=bypass_validation).run()

            # 11/14/22 KEEP SCORE SEPARATE FROM TRAIN_RESULTS TABLE TO HANDLE gmlr_conv_kill
            # 5/25/23 R IS IRRELEVANT
            if self.score_method == 'R': score = abs(R_) if isinstance(R_, (int, float)) else R_
            elif self.score_method == 'Q': score = R2_
            elif self.score_method == 'A': score = R2_ADJ_
            elif self.score_method == 'F': score = F_

            # IF MLRegression nan or error, RETURN 0 INSTEAD.
            score = score if isinstance(score, (float, int)) else 0

            if not self.conv_kill is None:
                ni_ctr, gmlr_conv_kill, best_value, abort = \
                    ni.NoImprov(score, lazy_agg_itr, ni_ctr_init, ni_ctr, self.conv_kill, best_value, self.pct_change,
                                'LAZY AGG', conv_end_method=self.conv_end_method).max()

                if abort:
                    AVAILABLE_COLUMNS = AVAILABLE_COLUMNS[:lazy_agg_itr]
                    max_columns = lazy_agg_itr
                    # THIS SHORT CIRCUIT WILL PREVENT LAST P_VALUES AND self.COEFFS FROM BEING RECORDED IF HIT CONV KILL
                    break


            self.TRAIN_RESULTS.loc[actv_hdr, ('CUMUL', 'R')] = R_
            self.TRAIN_RESULTS.loc[actv_hdr, ('CUMUL', 'R2')] = R2_
            self.TRAIN_RESULTS.loc[actv_hdr, ('CUMUL', 'ADJ R2')] = R2_ADJ_
            self.TRAIN_RESULTS.loc[actv_hdr, ('CUMUL', 'F')] = F_

            # IF DID NOT ABORT FOR CONV KILL, RETAIN MOST CURRENT SET OF P_VALUES AND COEFFS (WILL NEED THEM
            # WHEN IT DOES KILL FOR CONV, BECAUSE 2ND TO LAST ITR IS THEN THE BEST, NOT THE LAST)
            P_VALUES = P_VALUE_HOLDER
            self.COEFFS = COEFF_HOLDER


        # DONT USE AVAILABLE_COLUMNS TO CHOP OR SORT TRAIN_RESULTS HERE
        self.TRAIN_RESULTS = self.TRAIN_RESULTS.iloc[:max_columns, :]

        self.TRAIN_RESULTS.loc[:, ('FINAL', 'COEFFS')] = self.COEFFS
        self.TRAIN_RESULTS.loc[:, ('FINAL', 'p VALUE')] = P_VALUES



        self.WINNING_COLUMNS = AVAILABLE_COLUMNS

        del BatchData_Donor, LazyAggData_Receiver, R_, R2_, R2_ADJ_, F_, P_VALUES, COEFF_HOLDER, P_VALUE_HOLDER, actv_hdr







if __name__ == '__main__':

    import numpy as np, pandas as pd
    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl
    from MLObjects.SupportObjects import master_support_object_dict as msod
    from data_validation import validate_user_input as vui
    from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION import build_empty_gmlr_train_results as begtr
    from general_sound import winlinsound as wls


    # TEST LazyAgg FOR ACCURACY
















    DATA = pd.read_csv(r'C:\Users\Bill\Documents\WORK STUFF\RESUME\1 - OTHER\SRP\beer_reviews.csv',
                      nrows=200,
                      header=0).dropna(axis=0)

    DATA = DATA[DATA.keys()[[3, 4, 5, 7, 8, 9, 11]]]

    TARGET = DATA['review_overall']
    TARGET_HEADER = [['review_overall']]
    TARGET = TARGET.to_numpy().reshape((1,-1))

    DATA = DATA.drop(columns=['review_overall'])

    RAW_DATA = DATA.copy()
    RAW_DATA_HEADER = np.fromiter(RAW_DATA.keys(), dtype='<U50').reshape((1,-1))


    RAW_DATA = RAW_DATA.to_numpy()

    REFVECS = np.vstack((np.fromiter(range(len(RAW_DATA)), dtype=int).reshape((1, -1)),
                         np.random.choice(['A', 'B', 'C', 'D', 'E'], len(RAW_DATA), replace=True),
                         np.random.choice(['A', 'B', 'C', 'D', 'E'], len(RAW_DATA), replace=True),
                         np.random.choice(['A', 'B', 'C', 'D', 'E'], len(RAW_DATA), replace=True),
                         np.random.choice(['A', 'B', 'C', 'D', 'E'], len(RAW_DATA), replace=True)
    ))

    REFVECS_HEADER = ['ROW_IDX', 'JUNK1', 'JUNK2', 'JUNK3', 'JUNK4']



    # GIVEN TO LazyAggGMLR tests, NOT CreateSXNL!
    data_given_format = 'ARRAY'
    data_given_orient = 'COLUMN'
    target_given_format = 'ARRAY'
    target_given_orient = 'COLUMN'
    refvecs_given_format = 'ARRAY'
    refvecs_given_orient = 'COLUMN'


    SXNLClass = csxnl.CreateSXNL(rows=None,
                                 bypass_validation=False,
                                 data_return_format=data_given_format,
                                 data_return_orientation=data_given_orient,
                                 DATA_OBJECT=RAW_DATA,
                                 DATA_OBJECT_HEADER=RAW_DATA_HEADER,
                                 DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 data_override_sup_obj=False,
                                 data_given_orientation='ROW',
                                 data_columns=None,
                                 DATA_BUILD_FROM_MOD_DTYPES=None,
                                 DATA_NUMBER_OF_CATEGORIES=None,
                                 DATA_MIN_VALUES=None,
                                 DATA_MAX_VALUES=None,
                                 DATA_SPARSITIES=None,
                                 DATA_WORD_COUNT=None,
                                 DATA_POOL_SIZE=None,
                                 target_return_format=target_given_format,
                                 target_return_orientation=target_given_orient,
                                 TARGET_OBJECT=TARGET,
                                 TARGET_OBJECT_HEADER=TARGET_HEADER,
                                 TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 target_type='FLOAT',
                                 target_override_sup_obj=False,
                                 target_given_orientation='COLUMN',
                                 target_sparsity=None,
                                 target_build_from_mod_dtype=None,
                                 target_min_value=None,
                                 target_max_value=None,
                                 target_number_of_categories=None,
                                 refvecs_return_format=refvecs_given_format,
                                 refvecs_return_orientation=refvecs_given_orient,
                                 REFVECS_OBJECT=REFVECS,
                                 REFVECS_OBJECT_HEADER=REFVECS_HEADER,
                                 REFVECS_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 REFVECS_BUILD_FROM_MOD_DTYPES=None,
                                 refvecs_override_sup_obj=False,
                                 refvecs_given_orientation='COLUMN',
                                 refvecs_columns=None,
                                 REFVECS_NUMBER_OF_CATEGORIES=None,
                                 REFVECS_MIN_VALUES=None,
                                 REFVECS_MAX_VALUES=None,
                                 REFVECS_SPARSITIES=None,
                                 REFVECS_WORD_COUNT=None,
                                 REFVECS_POOL_SIZE=None
                                 )

    SRNL = SXNLClass.SXNL.copy()
    RAW_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS.copy()

    SXNLClass.expand_data(expand_as_sparse_dict=False, auto_drop_rightmost_column=False)
    SWNL = SXNLClass.SXNL
    WORKING_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS

    # BEAR HASH THIS TO GO BACK TO FULL COLUMNS
    # SWNL[0] = SWNL[0][:7]
    # WORKING_SUPOBJS[0] = WORKING_SUPOBJS[0][:, :7]

    data_given_orientation = SXNLClass.data_current_orientation
    target_given_orientation = SXNLClass.target_current_orientation
    refvecs_given_orientation = SXNLClass.refvecs_current_orientation

    WORKING_CONTEXT = []
    WORKING_KEEP = SRNL[1][0]


    DATA = SWNL[0]
    DATA_HEADER = WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]].reshape((1,-1))
    TARGET = SWNL[1]

    # BEAR
    # DELETE AN ARBITRARY BIN COLUMN TO GET RID OF MULTI-COLINEARTIY
    DATA = np.delete(DATA, 10, axis=0)
    DATA_HEADER = np.delete(DATA_HEADER, 10, axis=1)

    MASTER_BYPASS_VALIDATION = [True, False]
    MASTER_INTCPT_GIVEN = [True, False]
    MASTER_INTCPT_IDX_GIVEN = [True, False]
    MASTER_RGLZTN_FCTR = [0, 100]
    MASTER_AV_COLS_GIVEN = [True, False]
    MASTER_TRAIN_RESULTS_GIVEN = [True, False]
    MASTER_TARGET_HELPERS_GIVEN = [True, False]
    MASTER_DATA_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_DATA_ORIENT = ['ROW', 'COLUMN']
    MASTER_TARGET_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_TARGET_ORIENT = ['ROW', 'COLUMN']







    # BUILD WINNING_COLUMNS & TRAIN_RESULTS ANSWER KEYS #########################################################################

    for int_idx, has_intercept in enumerate(MASTER_INTCPT_GIVEN):
        if has_intercept:
            WIP_DATA = np.insert(DATA, 0, 1, axis=0)
            WIP_HEADER = np.hstack(([['INTERCEPT']] if has_intercept else [[]], DATA_HEADER.copy()))
        else:
            WIP_DATA = DATA.copy()
            WIP_HEADER = DATA_HEADER.copy()

        for rgz_idx, rglztn_fctr in enumerate(MASTER_RGLZTN_FCTR):
            WIP_SCORE_HOLDER = np.empty(0, dtype=np.float64)
            WIP_TRAIN_HOLDER = begtr.build_empty_gmlr_train_results(WIP_HEADER)

            for col_idx in range(len(WIP_DATA)):

                if col_idx == 0 and WIP_HEADER[0][col_idx] == 'INTERCEPT':   # IF INTERCEPT WOULD BE ONLY COLUMN GOING INTO MLR
                    WIP_SCORE_HOLDER = np.insert(WIP_SCORE_HOLDER, len(WIP_SCORE_HOLDER), 0, axis=0)
                    WIP_TRAIN_HOLDER.iloc[col_idx, 6:10] = (0, 0, 0, 0)
                    continue

                # returned from MLRegression
                # xtx_determinant, self.COEFFS, PREDICTED, P_VALUES, R, R2, R2_ADJ, F
                DUM, COEFFS, DUM, P_VALUES, R_, R2_, R2_ADJ_, F_ = \
                    mlr.MLRegression(
                                     WIP_DATA[:col_idx+1, :].transpose(),
                                     'ROW',
                                     TARGET.reshape((-1,1)),
                                     'ROW',
                                     DATA_TRANSPOSE=WIP_DATA[:col_idx+1, :],
                                     TARGET_TRANSPOSE=TARGET.reshape((1,-1)),
                                     TARGET_AS_LIST=TARGET.reshape((-1,1)),
                                     XTX=None,
                                     XTX_INV=None,
                                     has_intercept=has_intercept,
                                     intercept_math=has_intercept,
                                     regularization_factor=rglztn_fctr,
                                     safe_matmul=False,     # True if bypass_validation is False
                                     bypass_validation=True
                ).run()

                del DUM

                # WIP_SCORE_HOLDER = np.insert(WIP_SCORE_HOLDER, len(WIP_SCORE_HOLDER), R2_, axis=0)
                WIP_TRAIN_HOLDER.iloc[col_idx, 6:10] = (R_, R2_, R2_ADJ_, F_)
                if col_idx==len(WIP_DATA)-1:
                    WIP_TRAIN_HOLDER.iloc[:len(COEFFS), 4] = COEFFS
                    WIP_TRAIN_HOLDER.iloc[:len(P_VALUES), 5] = P_VALUES


            AVAILABLE_COLUMNS = np.fromiter(range(len(WIP_DATA)), dtype=np.int32)


            if rgz_idx==0 and not has_intercept:
                RGZ_0_INT_0_COL_KEY = AVAILABLE_COLUMNS.copy()
                RGZ_0_INT_0_TRAIN_KEY = WIP_TRAIN_HOLDER.copy()
            elif rgz_idx==0 and has_intercept:
                RGZ_0_INT_1_COL_KEY = AVAILABLE_COLUMNS.copy()
                RGZ_0_INT_1_TRAIN_KEY = WIP_TRAIN_HOLDER.copy()
            elif rgz_idx==1 and not has_intercept:
                RGZ_1_INT_0_COL_KEY = AVAILABLE_COLUMNS.copy()
                RGZ_1_INT_0_TRAIN_KEY = WIP_TRAIN_HOLDER.copy()
            elif rgz_idx==1 and has_intercept:
                RGZ_1_INT_1_COL_KEY = AVAILABLE_COLUMNS.copy()
                RGZ_1_INT_1_TRAIN_KEY = WIP_TRAIN_HOLDER.copy()

    del WIP_DATA

    # END BUILD WINNING_COLUMNS AND TRAIN_RESULTS ANSWER KEYS #####################################################################



    total_trials = np.product(list(map(len, (MASTER_BYPASS_VALIDATION, MASTER_INTCPT_GIVEN,
        MASTER_INTCPT_IDX_GIVEN, MASTER_RGLZTN_FCTR, MASTER_AV_COLS_GIVEN, MASTER_TRAIN_RESULTS_GIVEN,
        MASTER_TARGET_HELPERS_GIVEN, MASTER_DATA_FORMAT, MASTER_DATA_ORIENT, MASTER_TARGET_FORMAT, MASTER_TARGET_ORIENT))))

    ctr = 0
    for bypass_validation in MASTER_BYPASS_VALIDATION:
        for intcpt_is_given in MASTER_INTCPT_GIVEN:
            for intcpt_idx_is_given in MASTER_INTCPT_IDX_GIVEN:
                for rglztn_fctr in MASTER_RGLZTN_FCTR:
                    for av_cols_is_given in MASTER_AV_COLS_GIVEN:
                        for train_results_given in MASTER_TRAIN_RESULTS_GIVEN:
                            for target_helpers_given in MASTER_TARGET_HELPERS_GIVEN:
                                for data_format in MASTER_DATA_FORMAT:
                                    for data_orient in MASTER_DATA_ORIENT:
                                        for target_format in MASTER_TARGET_FORMAT:
                                            for target_orient in MASTER_TARGET_ORIENT:

                                                ctr += 1
                                                print(f'*'*140)
                                                print(f'\nRunning trial {ctr} of {total_trials}...')

                                                # BUILD / ORIENT / FORMAT GIVENS ################################################################
                                                print(f'\nCreating given objects...')

                                                GIVEN_DATA = DATA.copy()
                                                GIVEN_DATA_HEADER = DATA_HEADER.copy()
                                                intcpt_col_idx = None
                                                if intcpt_is_given:
                                                    GIVEN_DATA = np.insert(GIVEN_DATA, 0, 1, axis=0)
                                                    if intcpt_idx_is_given: intcpt_col_idx = 0
                                                    # elif not intcpt_idx_is_given: intcpt_col_idx STAYS None
                                                    GIVEN_DATA_HEADER = np.hstack(([['INTERCEPT']], GIVEN_DATA_HEADER))
                                                if data_orient == 'COLUMN': pass
                                                elif data_orient == 'ROW': GIVEN_DATA = GIVEN_DATA.transpose()
                                                if data_format == 'ARRAY': pass
                                                elif data_format == 'SPARSE_DICT': GIVEN_DATA = sd.zip_list_as_py_float(GIVEN_DATA)

                                                GIVEN_TARGET = TARGET.copy()
                                                GIVEN_TARGET_TRANSPOSE = None
                                                GIVEN_TARGET_AS_LIST = None
                                                if target_orient == 'COLUMN':
                                                    # GIVEN_TARGET STAYS THE SAME
                                                    if target_helpers_given:
                                                        GIVEN_TARGET_TRANSPOSE = GIVEN_TARGET.copy().transpose()
                                                        GIVEN_TARGET_AS_LIST = GIVEN_TARGET.copy()
                                                elif target_orient == 'ROW':
                                                    GIVEN_TARGET = GIVEN_TARGET.transpose()
                                                    if target_helpers_given:
                                                        GIVEN_TARGET_TRANSPOSE = GIVEN_TARGET.copy().transpose()
                                                        GIVEN_TARGET_AS_LIST = GIVEN_TARGET.copy()
                                                if target_format == 'ARRAY': pass
                                                elif target_format == 'SPARSE_DICT':
                                                    GIVEN_TARGET = sd.zip_list_as_py_float(GIVEN_TARGET)
                                                    if target_helpers_given:
                                                        GIVEN_TARGET_TRANSPOSE = sd.zip_list_as_py_float(GIVEN_TARGET_TRANSPOSE)

                                                if av_cols_is_given: AVAILABLE_COLUMNS = np.fromiter(range(len(DATA)+int(intcpt_is_given)), dtype=np.int32)
                                                else: AVAILABLE_COLUMNS = None

                                                if train_results_given:
                                                    GIVEN_TRAIN_RESULTS = begtr.build_empty_gmlr_train_results(GIVEN_DATA_HEADER)
                                                else:
                                                    GIVEN_TRAIN_RESULTS = None

                                                print(f'\nDone creating given objects...')
                                                # END BUILD / ORIENT / FORMAT GIVENS ############################################################

                                                # GET EXPECTEDS #################################################################################
                                                if intcpt_is_given:
                                                    if rglztn_fctr==MASTER_RGLZTN_FCTR[0]:
                                                        EXP_WINNING_COLUMNS = RGZ_0_INT_1_COL_KEY
                                                        EXP_TRAIN_RESULTS = RGZ_0_INT_1_TRAIN_KEY
                                                    elif rglztn_fctr==MASTER_RGLZTN_FCTR[1]:
                                                        EXP_WINNING_COLUMNS = RGZ_1_INT_1_COL_KEY
                                                        EXP_TRAIN_RESULTS = RGZ_1_INT_1_TRAIN_KEY
                                                else:
                                                    if rglztn_fctr==MASTER_RGLZTN_FCTR[0]:
                                                        EXP_WINNING_COLUMNS = RGZ_0_INT_0_COL_KEY
                                                        EXP_TRAIN_RESULTS = RGZ_0_INT_0_TRAIN_KEY
                                                    elif rglztn_fctr==MASTER_RGLZTN_FCTR[1]:
                                                        EXP_WINNING_COLUMNS = RGZ_1_INT_0_COL_KEY
                                                        EXP_TRAIN_RESULTS = RGZ_1_INT_0_TRAIN_KEY
                                                # END GET EXPECTEDS #############################################################################


                                                ACT_WINNING_COLUMNS, ACT_TRAIN_RESULTS, COEFFS = \
                                                        LazyAggGMLR(GIVEN_DATA,
                                                                     GIVEN_DATA_HEADER,
                                                                     data_orient,
                                                                     GIVEN_TARGET,
                                                                     target_orient,
                                                                     AVAILABLE_COLUMNS=AVAILABLE_COLUMNS,
                                                                     max_columns=None,
                                                                     intcpt_col_idx=intcpt_col_idx,
                                                                     rglztn_fctr=rglztn_fctr,
                                                                     score_method='Q',
                                                                     TRAIN_RESULTS=GIVEN_TRAIN_RESULTS,
                                                                     TARGET_TRANSPOSE=GIVEN_TARGET_TRANSPOSE,
                                                                     TARGET_AS_LIST=GIVEN_TARGET_AS_LIST,
                                                                     data_run_orientation='ROW', #'AS_GIVEN',
                                                                     target_run_orientation='ROW', #'AS_GIVEN',
                                                                     conv_kill=None,
                                                                     pct_change=None,
                                                                     conv_end_method=None,
                                                                     bypass_validation=bypass_validation).run()


                                                if not np.array_equiv(ACT_WINNING_COLUMNS, EXP_WINNING_COLUMNS):

                                                    print(f'\033[91m')
                                                    print(f'ACTUAL WINNING COLUMNS VIA ACT_TRAIN_RESULTS:')
                                                    print(ACT_TRAIN_RESULTS)
                                                    print()
                                                    print(f'EXPECTED WINNING COLUMNS VIA EXP_TRAIN_RESULTS:')
                                                    print(EXP_TRAIN_RESULTS)
                                                    print()
                                                    wls.winlinsound(444, 1000)
                                                    # if vui.validate_user_str(f'\nContinue(c) or kill(k) > ', 'CK') == 'K':
                                                    raise Exception(f'*** EXP/ACT WINNING COLUMNS NOT EQUAL ***')


                                                # COEFFS ARE CHECKED WItHIN TRAIN_RESULTS
                                                if intcpt_is_given and rglztn_fctr==0:
                                                    ACT_TRAIN_RESULTS.to_numpy()[:, 4:].astype(np.float64) # IF THIS EXCEPTS
                                                    PASS_TRAIN_RESULTS_TEST = np.allclose(
                                                        ACT_TRAIN_RESULTS.to_numpy()[:, 4:].astype(np.float64),
                                                        EXP_TRAIN_RESULTS.to_numpy()[:, 4:].astype(np.float64)
                                                    )
                                                elif intcpt_is_given and rglztn_fctr!=0:
                                                    PASS_TRAIN_RESULTS_TEST = np.allclose(
                                                        ACT_TRAIN_RESULTS.to_numpy()[:, [4,6,7,8,9]].astype(np.float64),
                                                        EXP_TRAIN_RESULTS.to_numpy()[:, [4,6,7,8,9]].astype(np.float64)
                                                    )
                                                elif not intcpt_is_given and rglztn_fctr==0:
                                                    PASS_TRAIN_RESULTS_TEST = np.allclose(
                                                        ACT_TRAIN_RESULTS.to_numpy()[:, [4,5,7,9]].astype(np.float64),
                                                        EXP_TRAIN_RESULTS.to_numpy()[:, [4,5,7,9]].astype(np.float64)
                                                    )
                                                elif not intcpt_is_given and rglztn_fctr!=0:
                                                    PASS_TRAIN_RESULTS_TEST = np.allclose(
                                                        ACT_TRAIN_RESULTS.to_numpy()[:, [4,7,9]].astype(np.float64),
                                                        EXP_TRAIN_RESULTS.to_numpy()[:, [4,7,9]].astype(np.float64)
                                                    )


                                                # not ACT_TRAIN_RESULTS.equals(EXP_TRAIN_RESULTS):
                                                if not PASS_TRAIN_RESULTS_TEST:
                                                    print(f'\033[91m')
                                                    print(f'ACTUAL TRAIN RESULTS:')
                                                    print(ACT_TRAIN_RESULTS)
                                                    print()
                                                    print(f'EXPECTED TRAIN RESULTS:')
                                                    print(EXP_TRAIN_RESULTS)
                                                    wls.winlinsound(444, 2000)
                                                    print(f'*** EXP/ACT TRAIN RESULTS NOT EQUAL ***')
                                                    if vui.validate_user_str(f'\nContinue(c) or kill(k) > ', 'CK') == 'K':
                                                        raise Exception(f'*** EXP/ACT TRAIN RESULTS NOT EQUAL ***')
                                                    else: print(f'\033[0m')
                                                print(f'*'*140)


    print(f'\n\033[92m*** ALL TESTS PASSED ***\033[0m')
    for _ in range(3): wls.winlinsound(888, 500); time.sleep(1)



