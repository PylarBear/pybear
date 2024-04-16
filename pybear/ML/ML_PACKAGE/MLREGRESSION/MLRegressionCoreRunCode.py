import sys

import numpy as np, pandas as pd, time
import sparse_dict as sd
from copy import deepcopy
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from debug import get_module_name as gmn
from general_data_ops import get_shape as gs, new_np_random_choice as nnrc
from ML_PACKAGE.MLREGRESSION import MLRegression as mlr, build_empty_mlr_train_results as bemtr
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo
from MLObjects import MLRowColumnOperations as mlrco


# THIS MAKES ALL DATAFRAME HEADERS AND INDEXES "UNSPARSE" AND CENTERS HEADERS
pd.set_option('display.multi_sparse', False, 'display.colheader_justify', 'center')
pd.set_option('display.max_columns', None, 'display.width', 150, 'display.max_colwidth', 35)
pd.options.display.float_format = '{:,.5f}'.format


class MLRegressionCoreRunCode:
    # DATA & TARGET MUST RUN IN MLRegression AS [[] = ROW .... GIVEN ORIENTATION ENTERED AS PARAM]
    def __init__(self, DATA, TARGET, DATA_TRANSPOSE, TARGET_TRANSPOSE, TARGET_AS_LIST, XTX, DATA_HEADER,
            data_run_orientation, target_run_orientation, rglztn_type, rglztn_fctr, batch_method, batch_size,
            intcpt_col_idx, bypass_validation):


        self.bear_time_display = lambda t0: round((time.time()-t0), 3)

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        self.bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                         self.this_module, fxn)

        data_rows = gs.get_shape('DATA', DATA, data_run_orientation)[0]

        if self.bypass_validation:
            self.DATA_HEADER = np.array(DATA_HEADER).reshape((1,-1))
        elif not self.bypass_validation:
            data_run_orientation = akv.arg_kwarg_validater(data_run_orientation.upper(), 'data_run_orientation',
                                            ['ROW', 'COLUMN'], self.this_module, fxn)
            target_run_orientation = akv.arg_kwarg_validater(target_run_orientation.upper(), 'target_run_orientation',
                                            ['ROW', 'COLUMN'], self.this_module, fxn)

            # INITIALIZE UP HERE FOR is_list & is_dict
            DATA = ldv.list_dict_validater(DATA, 'DATA')[1]
            self.DATA_HEADER = ldv.list_dict_validater(DATA_HEADER, 'DATA_HEADER')[1]

            DATA_TRANSPOSE = ldv.list_dict_validater(DATA_TRANSPOSE, 'DATA_TRANSPOSE')[1]
            TARGET = ldv.list_dict_validater(TARGET, 'TARGET')[1]
            TARGET_TRANSPOSE = ldv.list_dict_validater(TARGET_TRANSPOSE, 'TARGET_TRANSPOSE')[1]
            TARGET_AS_LIST = ldv.list_dict_validater(TARGET_AS_LIST, 'TARGET_AS_LIST')[1]
            XTX = ldv.list_dict_validater(XTX, 'XTX')[1]

            batch_method = akv.arg_kwarg_validater(batch_method, 'batch_method', ['M','B'], self.this_module, fxn)

            if batch_method == 'M':
                if batch_size >  data_rows: batch_size = data_rows
                if batch_size == 0: raise Exception(f'*** BATCH SIZE CANNOT BE ZERO ***')

        ########################################################################################################################
        # SET UP BATCH OR MINIBATCH ############################################################################################

        if batch_method == 'B':
            BATCH_DATA = DATA
            BATCH_DATA_TRANSPOSE = DATA_TRANSPOSE
            BATCH_XTX = XTX
            BATCH_TARGET = TARGET
            BATCH_TARGET_TRANSPOSE = TARGET_TRANSPOSE
            BATCH_TARGET_AS_LIST = TARGET_AS_LIST

        elif batch_method == 'M':
            # KEEP "BATCH_SIZE" NUMBER OF EXAMPLES BY RANDOMLY GENERATED MASK
            # IF batch_size IS >1, USE THIS AS BATCH SIZE, IF batch_size IS <= 1 USE AS PERCENT OF DATA
            if batch_size < 1: _len = np.ceil(batch_size * data_rows)
            elif batch_size >= 1: _len = np.ceil(batch_size)
            BATCH_MASK = nnrc.new_np_random_choice(range(data_rows), (1, int(_len)), replace=False).reshape((1, -1))[0]

            BATCH_DATA = mlrco.MLRowColumnOperations(DATA, data_run_orientation, name='DATA',
                bypass_validation=self.bypass_validation).return_rows(BATCH_MASK, return_orientation=data_run_orientation,
                                                                      return_format='AS_GIVEN')

            BATCH_TARGET = mlrco.MLRowColumnOperations(TARGET, target_run_orientation, name='TARGET',
                bypass_validation=self.bypass_validation).return_rows(BATCH_MASK, return_orientation=target_run_orientation,
                                                                      return_format='AS_GIVEN')

            # MUST RECREATE THESE VIA ObjectOrienter AFTER PULLING A MINIBATCH, OVERWRITING ANYTHING THAT MAY HAVE BEEN PASSED AS KWARG
            BATCH_DATA_TRANSPOSE = None
            BATCH_XTX = None
            BATCH_TARGET_TRANSPOSE = None
            BATCH_TARGET_AS_LIST = None

            del BATCH_MASK
        else:
            raise Exception(f'{self.this_module}.{fxn}() >>> batch_method ({batch_method}) IS FAILING, MUST BE "B" OR "M"')

        # END SET UP BATCH OR MINIBATCH ########################################################################################
        ########################################################################################################################

        ########################################################################################################################
        # ORIENT (BATCH) DATA & TARGET #########################################################################################
        print(f'\n    BEAR IN MLRegressionCoreRunCode Orienting DATA & TARGET.  Patience...'); t0 = time.time()

        # NOTES 5/4/23 --- IF MINI-BATCH WAS DONE, DATA & TARGET ARE COMING IN HERE IN "run" FORMAT & ORIENTATION AND
        # ANY GIVEN DATA_TRANSPOSE, ET AL KWARGS THAT WERE GIVEN ARE SET TO None AND NEED TO BE REBUILT. BUT IF IS FULL
        # BATCH, ROW SELECTION WAS BYPASSED AND DATA & TARGET ARE STILL IN GIVEN FORMAT & ORIENTATION (WHICH AS OF 5/11/23
        # SHOULD NOW BE RUN FORMAT & ORIENTATION) AND ANY PASSED TRANSPOSE ETAL KWARGS ARE STILL INTACT.


        OrienterClass = mloo.MLObjectOrienter(
                                                DATA=BATCH_DATA,
                                                data_given_orientation=data_run_orientation,
                                                data_return_orientation='AS_GIVEN',
                                                data_return_format='AS_GIVEN',

                                                DATA_TRANSPOSE=BATCH_DATA_TRANSPOSE,
                                                data_transpose_given_orientation=data_run_orientation if batch_method=='B' else None,
                                                data_transpose_return_orientation='AS_GIVEN',
                                                data_transpose_return_format='AS_GIVEN',

                                                XTX=BATCH_XTX,
                                                xtx_return_format='ARRAY',

                                                XTX_INV=None,
                                                xtx_inv_return_format=None,

                                                target_is_multiclass=False,
                                                TARGET=BATCH_TARGET,
                                                target_given_orientation=target_run_orientation,
                                                target_return_orientation='AS_GIVEN',
                                                target_return_format='AS_GIVEN',

                                                TARGET_TRANSPOSE=BATCH_TARGET_TRANSPOSE,
                                                target_transpose_given_orientation=target_run_orientation if batch_method=='B' else None,
                                                target_transpose_return_orientation='AS_GIVEN',
                                                target_transpose_return_format='AS_GIVEN',

                                                TARGET_AS_LIST=BATCH_TARGET_AS_LIST,
                                                target_as_list_given_orientation=target_run_orientation if batch_method=='B' else None,
                                                target_as_list_return_orientation='AS_GIVEN',

                                                RETURN_OBJECTS=['DATA','DATA_TRANSPOSE','TARGET',
                                                                'TARGET_TRANSPOSE','TARGET_AS_LIST', 'XTX'],

                                                bypass_validation=True,
                                                calling_module=self.this_module,
                                                calling_fxn=fxn
        )

        del DATA, DATA_TRANSPOSE, XTX, TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST, BATCH_DATA, BATCH_DATA_TRANSPOSE, \
            BATCH_TARGET, BATCH_TARGET_TRANSPOSE, BATCH_TARGET_AS_LIST, BATCH_XTX, batch_method, batch_size, data_rows

        self.data_run_orientation = OrienterClass.data_return_orientation
        self.target_run_orientation = OrienterClass.target_return_orientation
        self.DATA = OrienterClass.DATA
        self.DATA_TRANSPOSE = OrienterClass.DATA_TRANSPOSE
        self.TARGET = OrienterClass.TARGET
        self.TARGET_TRANSPOSE = OrienterClass.TARGET_TRANSPOSE
        self.TARGET_AS_LIST = OrienterClass.TARGET_AS_LIST
        self.XTX = OrienterClass.XTX

        self.xtx_run_format = OrienterClass.xtx_return_format
        del OrienterClass

        print(f'    BEAR IN MLRegressionCoreRunCode Orient DATA & TARGET Done.  time = {self.bear_time_display(t0)} sec')
        # END ORIENT (BATCH) DATA & TARGET #####################################################################################
        ########################################################################################################################


        ########################################################################################################################
        # VERIFY DIMENSIONS FOR MATMUL #########################################################################################

        # 9/16/22 --- IMPLEMENTED ABILITY TO CHOOSE UNSAFE sd.core_matmul TO SPEED UP, SO DO ONE CHECK HERE TO PROVE BROADCAST

        self.data_rows, data_cols = gs.get_shape('DATA', self.DATA, self.data_run_orientation)

        print(f'\n    Verifying dimensions for matmul.  Patience...'); t0 = time.time()

        target_rows = gs.get_shape('TARGET', self.TARGET, self.target_run_orientation)[0]
        if not self.data_rows == target_rows:
            raise Exception(f'MLRegressionCoreRunCode() TARGET ROWS ({target_rows}) != DATA ROWS ({self.data_rows}).')
        del target_rows

        print(f'     Verifying dimensions for matmul IN MLRegression Done.  time = {self.bear_time_display(t0)} sec')

        # END VERIFY DIMENSIONS FOR MATMUL ####################################################################################
        ########################################################################################################################


        print(f'\n    BEAR  IN MLRegression __init__ OTHER JUNK.'); t0 = time.time()

        self.intcpt_col_idx = intcpt_col_idx

        if gs.get_shape('DATA', self.DATA, self.data_run_orientation)[1] == 0:
            raise Exception(f'DATA IN MLRegressionCoreRunCode HAS NO COLUMNS')

        self.rglztn_type = rglztn_type.upper() if isinstance(rglztn_type, str) else rglztn_type

        if self.rglztn_type in [None, 'NONE']: self.rglztn_fctr = 0
        else: self.rglztn_fctr = rglztn_fctr

        # CREATE AN EMPTY DF TO BE FILLED WITH TRAIN RESULTS ###############################################################
        self.TRAIN_RESULTS = bemtr.build_empty_mlr_train_results(DATA_HEADER)
        del data_cols
        # END CREATE AN EMPTY DF TO BE FILLED WITH TRAIN RESULTS OBTAINED DURING run() #####################################

        print(f'\n    BEAR IN MLRegression OTHER JUNK __init__ Done. time = {self.bear_time_display(t0)} sec')

        # #####################################################################################################################
        # RUN MLRegression ####################################################################################################

        print(f'\n    RUNNING MULTIPLE LINEAR REGRESSION...\n')

        print(f'\nCalculating "{["no intercept" if self.intcpt_col_idx is None else "intercept"][0]}" style '
              f'MLRegression results and building results table... \n')


        # RETURNED FROM MLRegression
        # XTX_determinant, self.COEFFS, PREDICTED, P_VALUES, r, R2, R2_adj, F
        DUM, self.COEFFS, DUM, P_VALUES, R_, R2_, R2_ADJ_, F_ = \
            mlr.MLRegression(DATA=self.DATA,
                             data_given_orientation=self.data_run_orientation,
                             DATA_TRANSPOSE=self.DATA_TRANSPOSE,
                             XTX=self.XTX,
                             XTX_INV=None,
                             TARGET=self.TARGET,
                             target_given_orientation=self.target_run_orientation,
                             TARGET_TRANSPOSE=self.TARGET_TRANSPOSE,
                             TARGET_AS_LIST=self.TARGET_AS_LIST,
                             has_intercept=False if self.intcpt_col_idx is None else True,
                             intercept_math=False if self.intcpt_col_idx is None else True,
                             regularization_factor=self.rglztn_fctr,
                             safe_matmul=False,
                             bypass_validation=self.bypass_validation).run()

        del DUM

        # TRAIN_RESULTS_HEADER = [
        #     ['COLUMN', '      ',  '      ', 'OVERALL', 'OVERALL', 'OVERALL', 'OVERALL'],
        #     [' NAME ', 'p VALUE', 'COEFFS', '   R   ', '   R2  ', ' ADJ R2', '   F   ']
        #     ]

        self.TRAIN_RESULTS.loc[:, ('      ', 'p VALUE')] = P_VALUES
        self.TRAIN_RESULTS.loc[:, ('      ', 'COEFFS')] = self.COEFFS


        # ########################################################################################################################
        # SORT ###################################################################################################################
        print(f'\nProceeding to sort of MLRegression results by p value...')

        # SORT BY ASCENDING p VALUES
        MASTER_SORT_DESC = np.argsort(self.TRAIN_RESULTS.loc[:, ('      ', 'p VALUE')].to_numpy()).reshape(
            (1,-1))[0].astype(int).tolist()
        # MAKE AS LIST FOR EASE OF FINDING AND MOVING INTCPT COL IDX (np.argwhere IS FAILING TO FIND MATCHES HERE)

        # IF DATASET HAD INTERCEPT FORCE IT INTO WINNERS, MOVE IT TO FIRST IN MASTER_SORT_DESC NO MATTER WHAT ITS p VALUE WAS.
        if not self.intcpt_col_idx is None:
            MASTER_SORT_DESC.insert(0,
                MASTER_SORT_DESC.pop(MASTER_SORT_DESC.index(self.intcpt_col_idx)))

        self.TRAIN_RESULTS = self.TRAIN_RESULTS.iloc[np.array(MASTER_SORT_DESC, dtype=np.int32), :]

        self.WINNING_COLUMNS = np.array(MASTER_SORT_DESC, dtype=np.int32)

        del MASTER_SORT_DESC

        self.TRAIN_RESULTS.iloc[0, -4:] = (R_, R2_, R2_ADJ_, F_)

        del R_, R2_, R2_ADJ_, F_, P_VALUES

        # END SORT ##############################################################################################################
        # ########################################################################################################################

        # END RUN  MLRegression ####################################################################################################
        # ########################################################################################################################

        # END init ################################################################################################################
        ###########################################################################################################################
        ###########################################################################################################################


    def run(self):
        return self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS






if __name__ == '__main__':
    from data_validation import validate_user_input as vui
    from general_sound import winlinsound as wls
    from read_write_file.generate_full_filename import base_path_select as bps, filename_enter as fe

    # TEST MODULE IS IN SHAMBLES
    # 4/27/23 TEST MODULE APPEARS TO BE A LAZY COPY & PASTE JOB FROM GMLR, WITH ONLY MLRegressionCoreRunCode FXN DROPPED IN

    run_as = 'random_integer'  # random_integer, binary, random_sparse_dict, mi_scratch, beer_reviews

    rows = 100000
    columns = 20

    #######################################################################################################################################
    #######################################################################################################################################
    #######################################################################################################################################

    # BEAR 11/16/22 8:37 PM.
    # MAKE "OBJECTS CREATE" BELOW A FXN.  FIX HOW HEADER IS CREATED WHEN ORIENTATION IS "ROW" OR "COLUMN"

    test_type = vui.validate_user_str(f'RUN LOOP(l) OR ONE-OFF PASS(o) > ', 'LO')

    # CREATE OBJECTS #####################################################################################################
    print(f'CREATING OBJECTS')

    # 'mi_scratch'
    # basepath = bps.base_path_select()
    # READ_DATA = pd.read_excel(basepath + r'MI_SCRATCH.xlsx', sheet_name='DATA', dtype=np.int32)
    # # DATA = pd.DataFrame(data=READ_DATA)
    # TARGET = READ_DATA['y'].to_numpy().transpose().copy().reshape((1,-1))
    # DATA_HEADER = np.fromiter((_[0] for _ in READ_DATA.keys()[1:]), dtype='<U100').reshape((1,-1))
    # DATA = READ_DATA.drop(columns=['y'], axis=0).to_numpy().transpose()
    # del READ_DATA
    # DATA = sd.zip_list(DATA)

    'beer_reviews'
    DATA = pd.read_csv(r'C:\Users\Bill\Documents\WORK STUFF\RESUME\1 - OTHER\SRP\beer_reviews.csv',
                      nrows=100000,
                      header=0).dropna(axis=0)

    DATA = DATA[DATA.keys()[[3, 4, 5, 7, 8, 9, 11]]]

    TARGET = pd.DataFrame({'TARGET': DATA['review_overall']})

    TARGET_HEADER = np.fromiter((_ for _ in TARGET), dtype='<U50')
    TARGET_HEADER.resize(1, len(TARGET_HEADER))
    TARGET = TARGET.to_numpy()
    TARGET.resize(1, len(TARGET))

    DATA = DATA.drop(columns=['review_overall', 'beer_style'])

    # DATA = pd.get_dummies(DATA, columns=['beer_style'], prefix='', prefix_sep='' )
    DATA_HEADER = np.fromiter((_ for _ in DATA.keys()), dtype='<U50')
    DATA_HEADER.resize((1, len(DATA_HEADER)))
    DATA = DATA.to_numpy().transpose()

    REF_VEC = np.fromiter((_ for _ in range(len(DATA))), dtype=int)
    REF_VEC_HEADER = [['ROW_ID']]

    ORIG_NP_DATA = DATA
    ORIG_SD_DATA = sd.zip_list_as_py_float(DATA)
    ORIG_HEADER = DATA_HEADER

    print(f'CREATE OBJECTS Done.')
    # END CREATE OBJECTS ################################################################################################

    if test_type == 'L':
        ###############################################################################################################
        ###############################################################################################################
        ###############################################################################################################

        ctr = 0
        for rglztn_type in ['RIDGE']:
            for rglztn_factor in [0, 1000, 100000]:
                for batch_type in ['B', 'M']:
                    for batch_size in [.01, 1, 1000]:
                        for data_format in ['A', 'S']:
                            ctr += 1
                            if data_format == 'S': DATA = ORIG_SD_DATA
                            elif data_format == 'A': DATA = ORIG_NP_DATA

                            if mlr_use_intercept is True:
                                if data_format == 'A':
                                    DATA = np.insert(ORIG_NP_DATA, len(ORIG_NP_DATA), 1, axis=0)
                                elif data_format == 'S':
                                    DATA = sd.core_insert_outer(ORIG_SD_DATA, sd.outer_len(ORIG_SD_DATA),
                                                np.fromiter((1 for _ in ORIG_NP_DATA[0]), dtype=int), sd.outer_len(ORIG_SD_DATA))
                                intcpt_col_idx = len(DATA) - 1
                                DATA_HEADER = np.hstack((ORIG_HEADER, [['INTERCEPT']]))
                            else:
                                intcpt_col_idx = None

                            _len_data = len(DATA)

                            COEFFS, TRAIN_RESULTS = \
                                MLRegressionCoreRunCode(DATA, TARGET, DATA_TRANSPOSE, TARGET_TRANSPOSE, TARGET_AS_LIST,
                                    XTX, DATA_HEADER, data_run_orientation, target_run_orientation, mlr_type, mlr_method,
                                    intcpt_col_idx, bypass_validation=False).run()

                            print(f'\nJust finished trial {ctr} of {2 * 2 * 4 * 2 * 2}.' + \
                                f'\nmlr_type = {mlr_type}' + \
                                f'\nmlr_max_columns = {mlr_max_columns}' + \
                                f'\nmlr_method = {mlr_method}' + \
                                f'\nmlr_use_intercept = {mlr_use_intercept}' + \
                                f'\ndata_format = {data_format}' + \
                                f'\nlen(DATA) = {_len_data}' + \
                                f'\nlen(DATA[0]) = {sd.inner_len_quick(DATA) if isinstance(DATA, dict) else len(DATA[0])}' + \
                                f'\nlen(COEFFS) = {len(COEFFS)}' + \
                                f'\nlen(TRAIN_RESULTS) = {len(TRAIN_RESULTS)}\n')

                            # print(TRAIN_RESULTS)
                            #
                            # _ = input(f'\nHIT ENTER TO CONTINUE > \n')
                            # print(f'Running...')

                            exception_text = \
                                f'\nDISASTER during trial {ctr} of {2 * 2 * 4 * 2 * 2}.' + \
                                f'\nmlr_type = {mlr_type}' + \
                                f'\nmlr_max_columns = {mlr_max_columns}' + \
                                f'\nmlr_method = {mlr_method}' + \
                                f'\nmlr_use_intercept = {mlr_use_intercept}' + \
                                f'\ndata_format = {data_format}' + \
                                f'\nlen(DATA) = {_len_data}' + \
                                f'\nlen(DATA[0]) = {sd.inner_len_quick(DATA) if isinstance(DATA, dict) else len(DATA[0])}' + \
                                f'\nlen(COEFFS) = {len(COEFFS)}' + \
                                f'\nlen(TRAIN_RESULTS) = {len(TRAIN_RESULTS)}\n'


                            if len(COEFFS) != min(_len_data, mlr_max_columns):
                                wls.winlinsound(888, 500)
                                raise Exception(f'\nError 2' + exception_text)
                            if len(TRAIN_RESULTS) != min(_len_data, mlr_max_columns):
                                wls.winlinsound(888,500)
                                raise Exception(f'\nError 3' + exception_text)
                            if mlr_use_intercept is True and TRAIN_RESULTS['COLUMN']['NAME'][0] != 'INTERCEPT':
                                raise Exception(f'\nError 4' + exception_text)


    elif test_type == 'O':
        ################################################################################################################
        ################################################################################################################
        ###### STUFF FOR ONE-OFF TEST OF CoreRun #######################################################################

        data_format = vui.validate_user_str(f'\nRun as sparse dict(s) or array(a) > ', 'AS')
        if data_format == 'S': DATA = sd.zip_list_as_py_float(DATA)
        # INDEX = [*range(1, rows + 1), 'SCORE']

        rglztn_type = 'RIDGE'
        rglztn_fctr = 0
        batch_method = 'B'
        batch_size = 2000
        mlr_use_intercept = True

        if mlr_use_intercept is True:
            if isinstance(DATA, (np.ndarray, list, tuple)):
                DATA = np.insert(DATA, len(DATA), 1, axis=0)
            elif isinstance(DATA, dict):
                DATA = sd.append_outer(DATA, np.fromiter((1 for _ in range(sd.inner_len_quick(DATA))), dtype=int))
            intcpt_col_idx = len(DATA) - 1
            DATA_HEADER = np.hstack((DATA_HEADER, [['INTERCEPT']]))
        else:
            intcpt_col_idx = None

        data_run_orientation = 'COLUMN'
        target_run_orientation = 'COLUMN'

        COEFFS, TRAIN_RESULTS = MLRegressionCoreRunCode(DATA, TARGET, DATA_TRANSPOSE, TARGET_TRANSPOSE,
            TARGET_AS_LIST, XTX, DATA_HEADER, data_run_orientation, target_run_orientation, rglztn_type, rglztn_fctr, batch_method,
            batch_size, intcpt_col_idx, bypass_validation=False).run()


        print(400 * '*')
        if isinstance(DATA, dict):
            DATA = sd.unzip_to_ndarray(DATA)[0]


        PREDICTED = np.matmul(DATA.astype(int).transpose().astype(float), COEFFS.astype(float), dtype=float)

        DATA_DICT = {}

        if isinstance(DATA, (list, tuple, np.ndarray)):
            for idx in range(len(TARGET)): DATA_DICT[TARGET_HEADER[0][idx]] = deepcopy(TARGET[idx])
        elif isinstance(DATA, dict):
            for idx in range(len(TARGET)): DATA_DICT[TARGET_HEADER[0][idx]] = sd.zip_list([deepcopy(TARGET[idx])])[0]

        DATA_DICT['PREDICTED'] = PREDICTED
        for idx in range(len(DATA)):
            DATA_DICT[DATA_HEADER[0][idx]] = deepcopy(DATA[idx])

        DF = pd.DataFrame(DATA_DICT).fillna(0)
        print()
        print(DF)



        print('Done.')

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

        ###### STUFF FOR ONE-OFF TEST OF CoreRun #######################################################################
        ################################################################################################################
        ################################################################################################################



















