import sys, inspect
import numpy as np
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from general_data_ops import get_shape as gs
from MLObjects import ML_find_constants as mlfc
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo
from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION import build_empty_gmlr_train_results as begtr


# gmlr_score_type DOES NOT MATTER.  ADJUSTMENTS FOR RSQ_ADJ AND F WOULD BE THE SAME FOR EVERY COLUMN, SO RSQ_ADJ AND F
# WILL ALWAYS BE PROPORTIONAL TO R2.  R WOULD REQUIRE abs ADJUSTMENT, JUST TO STILL BE PROPORTIONAL TO R2.
# USING ONLY R2 FOR SORTING.




class ApexGMLR:

    # DOESNT NEED DATA_TRANSPOSE ET AL. ALL CHILDREN WILL REPEATEDLY CREATE WORKING DATASETS FROM THE GIVEN DATASET SO
    # PASSING THE HELPER OBJECTS WOULD BE POINTLESS.

    def __init__(self, DATA, DATA_HEADER, data_given_orientation, TARGET, target_given_orientation, AVAILABLE_COLUMNS=None,
                 max_columns=None, intcpt_col_idx=None, rglztn_fctr=None, TRAIN_RESULTS=None, TARGET_TRANSPOSE=None,
                 TARGET_AS_LIST=None, data_run_orientation='ROW', target_run_orientation='ROW', bypass_validation=None,
                 calling_module=None):

        this_module = gmn.get_module_name(str(sys.modules[__name__])) if calling_module is None else calling_module
        fxn = inspect.stack()[0][3]

        ################################################################################################################################
        ################################################################################################################################
        # VALIDATION & OBJECT PREP #####################################################################################################

        bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                        this_module, fxn, return_if_none=False)

        # VALIDATE THESE REGARDLESS OF bypass_validation ###################################################################
        DATA = ldv.list_dict_validater(DATA, 'DATA')[1]
        DATA_HEADER = ldv.list_dict_validater(DATA_HEADER, 'DATA_HEADER')[1][0]
        TARGET = ldv.list_dict_validater(TARGET, 'TARGET')[1]
        if not AVAILABLE_COLUMNS is None:
            AVAILABLE_COLUMNS = ldv.list_dict_validater(AVAILABLE_COLUMNS, 'AVAILABLE_COLUMNS')[1][0]

        data_given_orientation = akv.arg_kwarg_validater(data_given_orientation, 'data_given_orientation',
                                                         ['ROW','COLUMN'], this_module, fxn)
        target_given_orientation = akv.arg_kwarg_validater(target_given_orientation, 'target_given_orientation',
                                                           ['ROW','COLUMN'], this_module, fxn)
        data_run_orientation = akv.arg_kwarg_validater(data_run_orientation, 'data_run_orientation',
                                                         ['ROW','COLUMN', 'AS_GIVEN'], this_module, fxn)
        target_run_orientation = akv.arg_kwarg_validater(target_run_orientation, 'target_run_orientation',
                                                           ['ROW','COLUMN', 'AS_GIVEN'], this_module, fxn)

        data_run_orientation = data_given_orientation if data_run_orientation=='AS_GIVEN' else data_run_orientation

        target_run_orientation = target_given_orientation if target_run_orientation=='AS_GIVEN' else target_run_orientation

        # END VALIDATE THESE REGARDLESS OF bypass_validation ################################################################

        data_rows, data_cols = gs.get_shape('DATA', DATA, data_given_orientation)

        # IF NO AVAILABLE_COLUMNS, ASSUME ALL
        if AVAILABLE_COLUMNS is None: AVAILABLE_COLUMNS = np.fromiter(range(data_cols), dtype=np.int32)

        # IF INTCPT AND intcpt_col_idx IS NOT FIRST, MOVE intcpt_col_idx TO FIRST, OR VERIFY IT ALREADY IS FIRST
        # IF intcpt_col_idx IS IN DATA BUT NOT IN AVAILABLE COLUMNS, ASSUME RUNNING NO-INTERCEPT MLR WAS INTENTIONAL
        if intcpt_col_idx is not None:
            if intcpt_col_idx in AVAILABLE_COLUMNS and AVAILABLE_COLUMNS[0] != intcpt_col_idx:
                AVAILABLE_COLUMNS = np.insert(AVAILABLE_COLUMNS[AVAILABLE_COLUMNS != intcpt_col_idx], 0, intcpt_col_idx, axis=0)

        # IF NO max_columns, ASSUME ALL
        if max_columns is None: max_columns = data_cols

        # GET intcpt_col_idx IF NOT GIVEN ###########################################################################################
        if intcpt_col_idx is None:

            # FIND COLUMNS OF CONSTANTS
            CON_COLS, ZERO_COLS = mlfc.ML_find_constants(DATA, data_given_orientation)

            if len(ZERO_COLS) > 0: raise Exception(f'*** {this_module}.{fxn}() >>> {len(ZERO_COLS)} COLUMN(S) OF ZEROS IN DATA ***')

            if len(CON_COLS) > 1: raise Exception(f'*** {this_module}.{fxn}() >>> {len(CON_COLS)} COLUMNS OF CONSTANTS IN DATA ***')
            elif len(CON_COLS) == 0: intcpt_col_idx = None
            elif len(CON_COLS) == 1: intcpt_col_idx = list(CON_COLS.keys())[0]

            del CON_COLS, ZERO_COLS
        # END GET intcpt_col_idx IF NOT GIVEN ###########################################################################################

        if rglztn_fctr is None: rglztn_fctr = 0
        elif not rglztn_fctr >= 0: raise Exception(f'*** rglztn_fctr MUST BE AN INTEGER >= 0 ***')


        # VALIDATE SHAPES OF PASSED OBJECTS #############################################################################################
        if not bypass_validation:
            target_rows, target_cols = gs.get_shape('TARGET', TARGET, target_given_orientation)
            if data_rows != target_rows:
                raise Exception(f'*** DATA ROWS ({data_rows}) DOES NOT EQUAL TARGET ROWS ({target_rows}) WRT GIVEN ORIENTATIONS ***')
            if data_cols != len(DATA_HEADER):
                raise Exception(f'*** DATA HEADER COLUMNS ({len(DATA_HEADER)}) DOES NOT MATCH DATA COLUMNS ({data_cols}) WRT GIVEN ORIENTATION ***')
            if target_cols > 1: raise Exception(f'*** TARGET IS MULTI-CLASS ({target_cols}) ***')
            # if not TRAIN_RESULTS is None and data_cols != len(TRAIN_RESULTS):  # BLOWS UP IN LazyAggGMLR() BECAUSE TRAIN_RESULTS WAS CHOPPED BY LazyGMLR()
            #     raise Exception(f'*** ROWS IN TRAIN RESULTS ({len(TRAIN_RESULTS)}) DO NOT MATCH DATA COLUMNS ({data_cols}) ***')
            if len(AVAILABLE_COLUMNS) != len(np.unique(AVAILABLE_COLUMNS)):
                raise Exception(f'*** DUPLICATE COLUMN INDEX ENTRIES IN AVAILABLE COLUMNS ***')
            if not max_columns >= 1: raise Exception(f'*** max_columns MUST BE INTEGER >= 1 ***')
            if not intcpt_col_idx is None and (intcpt_col_idx >= 0 and not intcpt_col_idx < data_cols):
                raise Exception(f'*** intcpt_col_idx MUST BE INTEGER >= 0 AND <= DATA COLUMNS ({data_cols}) OR None ***')

            del data_rows, target_rows, target_cols

        self.data_cols = data_cols

        # END VALIDATE SHAPES OF PASSED OBJECTS #########################################################################################


        # CREATE / ORIENT OBJECTS
        ObjectOrienter =  mloo.MLObjectOrienter(
                                                 DATA=DATA,
                                                 data_given_orientation=data_given_orientation,
                                                 data_return_orientation=data_run_orientation,
                                                 data_return_format='AS_GIVEN',

                                                 target_is_multiclass=False,
                                                 TARGET=TARGET,
                                                 target_given_orientation=target_given_orientation,
                                                 target_return_orientation=target_run_orientation,
                                                 target_return_format='ARRAY',   # MLRegression WILL CONVERT TO ARRAY

                                                 TARGET_TRANSPOSE=TARGET_TRANSPOSE,
                                                 target_transpose_given_orientation=target_given_orientation,
                                                 target_transpose_return_orientation=target_run_orientation,
                                                 target_transpose_return_format='ARRAY',   # MLRegression WILL CONVERT TO ARRAY

                                                 TARGET_AS_LIST=TARGET_AS_LIST,
                                                 target_as_list_given_orientation=target_given_orientation,
                                                 target_as_list_return_orientation=target_run_orientation,

                                                 RETURN_OBJECTS=['DATA', 'TARGET', 'TARGET_TRANSPOSE', 'TARGET_AS_LIST'],

                                                 bypass_validation=bypass_validation,
                                                 calling_module=this_module,
                                                 calling_fxn=fxn
        )
        DATA = ObjectOrienter.DATA
        TARGET = ObjectOrienter.TARGET
        TARGET_TRANSPOSE = ObjectOrienter.TARGET_TRANSPOSE
        TARGET_AS_LIST = ObjectOrienter.TARGET_AS_LIST
        del ObjectOrienter

        # END VALIDATION & OBJECT PREP ##################################################################################################
        #################################################################################################################################
        #################################################################################################################################

        self.WINNING_COLUMNS = None
        self.COEFFS = None

        # IF TRAIN_RESULTS NOT GIVEN, BUILD EMPTY
        if TRAIN_RESULTS is None: self.TRAIN_RESULTS = begtr.build_empty_gmlr_train_results(DATA_HEADER)
        else: self.TRAIN_RESULTS = TRAIN_RESULTS

        self.METHOD_DICT = {'F': 'F-score', 'Q': 'RSQ', 'A': 'ADJ RSQ', 'R': 'r'}

        # #####################################################################################################################
        # RUN CORE GMLR #######################################################################################################

        self.core_run(DATA, DATA_HEADER, data_run_orientation, TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST,
            target_run_orientation, AVAILABLE_COLUMNS, max_columns, intcpt_col_idx, rglztn_fctr, bypass_validation,
            this_module)

        # END RUN CORE GMLR ###################################################################################################
        # #####################################################################################################################


    # END init ##########################################################################################################
    #####################################################################################################################
    #####################################################################################################################


    def run(self):
        return self.WINNING_COLUMNS, self.TRAIN_RESULTS, self.COEFFS


    def core_run(self, DATA, DATA_HEADER, data_run_orientation, TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST,
                target_run_orientation, AVAILABLE_COLUMNS, max_columns, intcpt_col_idx, rglztn_fctr, bypass_validation,
                this_module):

        # fxn = inspect.stack()[0][3]

        # return SCORES

        pass   # MUST RETURN SCORES TO init






















if __name__ == '__main__':
    pass

























