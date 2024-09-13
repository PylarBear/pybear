import sys, inspect, warnings, time
from general_sound import winlinsound as wls
import numpy as np, sparse_dict as sd
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv, validate_user_input as vui
from general_data_ops import get_shape as gs
from MLObjects.TestObjectCreators import ApexTestObjectCreate as atoc
from MLObjects.SupportObjects import master_support_object_dict as msod
from MLObjects import MLTrainDevTestSplit as mlddt



# EXPANSION MUST BE DONE BEFORE T-D-T SPLIT!

# IF HAVE VAL & TEST OBJECTS, JUST PASS THEM TO DIFFERENT CALLS OF THIS CLASS THEN MERGE ALL THEIR SRNLS

# PRE TDT SPLIT, ACCESS OBJECTS AS
# self.DATA, self.TARGET, self.REVFECS,
# self.DATA_SUPPORT_OBJECTS, self.TARGET_SUPPORT_OBJECTS, self.REFVECS_SUPPORT_OBJECTS

# AFTER TDT SPLIT, ACCESS OBJECTS AS
# self.TRAIN_SWNL, self.DEV_SWNL, self.TEST_SWNL
# self.TRAIN_DATA, self.TRAIN_TARGET, self.TRAIN_REFVEC
# self.DEV_DATA, self.DEV_TARGET, self.DEV_REFVEC
# self.TEST_DATA, self.TEST_TARGET, self.TEST_REFVEC
# self.DATA_SUPPORT_OBJECTS, self.TARGET_SUPPORT_OBJECTS, self.REFVECS_SUPPORT_OBJECTS (DONT CHANGE DURING SPLIT)

# BEAR
# AS OF LATE 3/20/23, THE PLAN IS TO GIVE METHODS TO MasterSXNLOperations AND MAKE THAT PARENT OF THIS.  THAT WAY FANCY SXNL
# OPERATIONS CAN BE DROPPED IN AS PARENT METHODS JUST ABOUT ANYWHERE.
# FIGURE OUT HOW TO HANDLE EXPAND / TRANSPOSE / CHANGE FORMAT.   THIS IS LOOKING LIKE A NIGHTMARE TO DO IN MasterSXNLOperations.
# ALL OF THESE ObjectCreator CLASSES HAVE ALL OF THESE METHODS ALREADY.



class CreateSXNL:
    def __init__(self,
                 rows=None,
                 bypass_validation=False,
                 ##################################################################################################################
                 # DATA ############################################################################################################
                 data_return_format=None,
                 data_return_orientation=None,
                 DATA_OBJECT=None,
                 DATA_OBJECT_HEADER=None,
                 DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                 data_override_sup_obj=None,
                 # CREATE FROM GIVEN ONLY ###############################################
                 data_given_orientation=None,
                 # END CREATE FROM GIVEN ONLY #############################################
                 # CREATE FROM SCRATCH_ONLY ################################
                 data_columns=None,
                 DATA_BUILD_FROM_MOD_DTYPES=['BIN', 'INT', 'FLOAT', 'STR'],
                 DATA_NUMBER_OF_CATEGORIES=10,
                 DATA_MIN_VALUES=-10,
                 DATA_MAX_VALUES=10,
                 DATA_SPARSITIES=50,
                 DATA_WORD_COUNT=20,
                 DATA_POOL_SIZE=200,
                 # END DATA ###########################################################################################################
                 ##################################################################################################################

                 #################################################################################################################
                 # TARGET #########################################################################################################
                 target_return_format=None,
                 target_return_orientation=None,
                 TARGET_OBJECT=None,
                 TARGET_OBJECT_HEADER=None,
                 TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                 target_type=None,  # MUST BE 'BINARY','FLOAT', OR 'SOFTMAX'
                 target_override_sup_obj=None,
                 target_given_orientation=None,
                 # END CORE TARGET_ARGS ########################################################
                 # FLOAT AND BINARY
                 target_sparsity=50,
                 # FLOAT ONLY
                 target_build_from_mod_dtype='FLOAT',  # COULD BE FLOAT OR INT
                 target_min_value=-10,
                 target_max_value=10,
                 # SOFTMAX ONLY
                 target_number_of_categories=5,

                # END TARGET ####################################################################################################
                #################################################################################################################

                #################################################################################################################
                # REFVECS ########################################################################################################
                refvecs_return_format=None,    # IS ALWAYS ARRAY (WAS, CHANGED THIS 4/6/23)
                refvecs_return_orientation=None,
                REFVECS_OBJECT=None,
                REFVECS_OBJECT_HEADER=None,
                REFVECS_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                REFVECS_BUILD_FROM_MOD_DTYPES=['STR', 'STR', 'STR', 'STR', 'STR', 'BIN', 'INT'],
                refvecs_override_sup_obj=None,
                refvecs_given_orientation=None,
                refvecs_columns=None,
                REFVECS_NUMBER_OF_CATEGORIES=10,
                REFVECS_MIN_VALUES=-10,
                REFVECS_MAX_VALUES=10,
                REFVECS_SPARSITIES=50,
                REFVECS_WORD_COUNT=20,
                REFVECS_POOL_SIZE=200
                # END REFVECS ########################################################################################################
                #################################################################################################################
                ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        self.bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                         self.this_module, fxn, return_if_none=False)

        # VALIDATE ROWS ARE EQUAL IF MULTIPLE OBJECTS ARE GIVEN #################################################################
        NAMES = ('DATA', 'TARGET', 'REFVECS')
        OBJECTS = (DATA_OBJECT, TARGET_OBJECT, REFVECS_OBJECT)
        ORIENTS = (data_given_orientation, target_given_orientation, refvecs_given_orientation)
        _ROWS = [gs.get_shape(name,_,__)[0] for name,_,__ in zip(NAMES, OBJECTS, ORIENTS) if not _ is None]
        del NAMES, OBJECTS, ORIENTS
        if len(_ROWS) == 0:               # IF NO OBJECTS GIVEN, self.rows RUNS THE SHOW
            # IRREGARDLESS OF bypass_validation
            if rows not in (range(int(1e9))): self._exception(f'IF NO OBJECTS ARE GIVEN, rows MUST BE ENTERED AS INTEGER GREATER THAN ZERO', fxn=fxn)
            else: self.rows = rows
        elif len(_ROWS) == 1: # IF ONE OBJECT, # ROWS IN OBJECT OVERRIDES ANYTHING ENTERED FOR rows KWARG
            self.rows = _ROWS[0]
        elif len(_ROWS) > 1:  # IF MORE THAN ONE OBJECT, VALIDATE ROWS ARE EQUAL
            if not min(_ROWS)==max(_ROWS):
                self._exception(f'PASSED OBJECTS HAVE UNEQUAL NUMBER OF ROWS W-R-T GIVEN ORIENTATIONS')
            else: self.rows = _ROWS[0]
        del _ROWS
        # END VALIDATE ROWS ARE EQUAL IF MULTIPLE OBJECTS ARE GIVEN #################################################################

        self.target_type = akv.arg_kwarg_validater(target_type, 'target_type', ['BINARY','FLOAT','SOFTMAX'],
                                                   self.this_module, fxn)

        # LET THE INDIVIDUAL Create MODULES VALIDATE THE REST OF THE ARGS/KWARGS

        self.DataClass = atoc.CreateDataObject(
                                         data_return_format,
                                         data_return_orientation,
                                         DATA_OBJECT=DATA_OBJECT,
                                         DATA_OBJECT_HEADER=DATA_OBJECT_HEADER,
                                         FULL_SUPOBJ_OR_SINGLE_MDTYPES=DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES,
                                         override_sup_obj=data_override_sup_obj,
                                         bypass_validation=self.bypass_validation,
                                         # CREATE FROM GIVEN ONLY ###############################################
                                         given_orientation=data_given_orientation,
                                         # END CREATE FROM GIVEN ONLY #############################################
                                         # CREATE FROM SCRATCH_ONLY ################################
                                         rows=self.rows,
                                         columns=data_columns,
                                         BUILD_FROM_MOD_DTYPES=DATA_BUILD_FROM_MOD_DTYPES,
                                         NUMBER_OF_CATEGORIES=DATA_NUMBER_OF_CATEGORIES,
                                         MIN_VALUES=DATA_MIN_VALUES,
                                         MAX_VALUES=DATA_MAX_VALUES,
                                         SPARSITIES=DATA_SPARSITIES,
                                         WORD_COUNT=DATA_WORD_COUNT,
                                         POOL_SIZE=DATA_POOL_SIZE
                                         # END CREATE FROM SCRATCH_ONLY #############################
                                         )

        self.DATA, self.DATA_SUPPORT_OBJECTS = self.DataClass.OBJECT, self.DataClass.SUPPORT_OBJECTS
        del DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES
        # DONT DELETE DataClass YET, MIGHT NEED IT FOR train-dev-tests SPLIT OR EXPAND

        # CORE TARGET_ARGS ########################################################
        ARGS = target_return_format, target_return_orientation
        KWARGS = {'TARGET_OBJECT': TARGET_OBJECT,
                  'TARGET_OBJECT_HEADER': TARGET_OBJECT_HEADER,
                  'FULL_SUPOBJ_OR_SINGLE_MDTYPES': TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES,
                  'override_sup_obj': target_override_sup_obj,
                  'bypass_validation': self.bypass_validation,
                  'given_orientation': target_given_orientation,
                  'rows': self.rows}
        # END CORE TARGET_ARGS ########################################################

        if self.target_type == 'BINARY':
            self.TargetClass =  atoc.CreateBinaryTarget(*ARGS, **KWARGS, _sparsity=target_sparsity)

        elif self.target_type == 'FLOAT':
            self.TargetClass = atoc.CreateFloatTarget(*ARGS,
                                                 **KWARGS,
                                                  build_from_mod_dtype=target_build_from_mod_dtype,
                                                  min_value=target_min_value,
                                                  max_value=target_max_value,
                                                  _sparsity=target_sparsity)
        elif self.target_type == 'SOFTMAX':
            self.TargetClass = atoc.CreateSoftmaxTarget(*ARGS, **KWARGS, number_of_categories=target_number_of_categories)

        self.TARGET, self.TARGET_SUPPORT_OBJECTS =  self.TargetClass.OBJECT, self.TargetClass.SUPPORT_OBJECTS
        del TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES, ARGS, KWARGS
        # DONT DELETE self.TargetClass YET, MIGHT NEED IT FOR train-dev-tests SPLIT


        self.RefVecsClass = atoc.CreateRefVecs(
                                         refvecs_return_format,
                                         refvecs_return_orientation,
                                         REFVEC_OBJECT=REFVECS_OBJECT,
                                         REFVEC_OBJECT_HEADER=REFVECS_OBJECT_HEADER,
                                         FULL_SUPOBJ_OR_SINGLE_MDTYPES=REFVECS_FULL_SUPOBJ_OR_SINGLE_MDTYPES,
                                         BUILD_FROM_MOD_DTYPES=REFVECS_BUILD_FROM_MOD_DTYPES,
                                         override_sup_obj=refvecs_override_sup_obj,
                                         bypass_validation=self.bypass_validation,
                                         given_orientation=refvecs_given_orientation,
                                         rows=self.rows,
                                         columns=refvecs_columns,
                                         NUMBER_OF_CATEGORIES=REFVECS_NUMBER_OF_CATEGORIES,
                                         MIN_VALUES=REFVECS_MIN_VALUES,
                                         MAX_VALUES=REFVECS_MAX_VALUES,
                                         SPARSITIES=REFVECS_SPARSITIES,
                                         WORD_COUNT=REFVECS_WORD_COUNT,
                                         POOL_SIZE=REFVECS_POOL_SIZE
                                         )

        self.REFVECS, self.REFVECS_SUPPORT_OBJECTS = self.RefVecsClass.OBJECT, self.RefVecsClass.SUPPORT_OBJECTS
        del REFVECS_FULL_SUPOBJ_OR_SINGLE_MDTYPES
        # DONT DELETE RefVecClass YET, MIGHT NEED IT FOR train-dev-tests SPLIT OR WANT TO EXPAND IT

        self.SXNL = list((self.DATA, self.TARGET, self.REFVECS))
        self.SXNL_SUPPORT_OBJECTS = list((self.DATA_SUPPORT_OBJECTS, self.TARGET_SUPPORT_OBJECTS, self.REFVECS_SUPPORT_OBJECTS))

        # IF NO TEXT DTYPES IN MOD_DTYPES, THEN OBJECT IS EXPANDED
        self.data_is_expanded = True not in [_ in msod.mod_text_dtypes().values() for _ in self.DATA_SUPPORT_OBJECTS[msod.QUICK_POSN_DICT()['MODIFIEDDATATYPES']]]
        self.refvecs_is_expanded = True not in [_ in msod.mod_text_dtypes().values() for _ in self.REFVECS_SUPPORT_OBJECTS[msod.QUICK_POSN_DICT()['MODIFIEDDATATYPES']]]

        self.TDT_split_is_done = False

        # IN CASE EVER DEPLOY TRANSPOSING, TO_LIST, TO_DICT, ETC.
        self.data_current_format = self.DataClass.return_format
        self.data_current_orientation = self.DataClass.return_orientation
        self.target_current_format = self.TargetClass.return_format
        self.target_current_orientation = self.TargetClass.return_orientation
        self.refvecs_current_format = self.RefVecsClass.return_format
        self.refvecs_current_orientation = self.RefVecsClass.return_orientation


        # PLACEHOLDERS
        self.TRAIN_SWNL = None
        self.DEV_SWNL = None
        self.TEST_SWNL = None

        self.TRAIN_DATA = None
        self.TRAIN_TARGET = None
        self.TRAIN_REFVECS = None
        self.DEV_DATA = None
        self.DEV_TARGET = None
        self.DEV_REFVECS = None
        self.TEST_DATA = None
        self.TEST_TARGET = None
        self.TEST_REFVECS = None


    def _exception(self, words, fxn=None):
        fxn = f'.{fxn}()' if not fxn is None else ''
        raise Exception(f'{self.this_module}{fxn} >>> {words}')


    def expand_data(self, expand_as_sparse_dict=False, auto_drop_rightmost_column=False):

        if self.data_is_expanded:
            print(f'\n*** DATA IS ALREADY EXPANDED ***\n')
        elif not self.data_is_expanded:
            self.DataClass.expand(expand_as_sparse_dict=expand_as_sparse_dict,
                                  auto_drop_rightmost_column=auto_drop_rightmost_column)

            self.DATA, self.DATA_SUPPORT_OBJECTS = self.DataClass.OBJECT, self.DataClass.SUPPORT_OBJECTS
            self.SXNL[0], self.SXNL_SUPPORT_OBJECTS[0] = self.DataClass.OBJECT, self.DataClass.SUPPORT_OBJECTS
            self.data_is_expanded = True
            # DONT DELETE DataClass YET, MIGHT NEED IT FOR train-dev-tests SPLIT


    def expand_refvecs(self, expand_as_sparse_dict=None, auto_drop_rightmost_column=None):

        if self.refvecs_is_expanded:
            print(f'\n*** REFVECS IS ALREADY EXPANDED ***\n')
        elif not self.refvecs_is_expanded:
            self.RefVecsClass.expand(expand_as_sparse_dict=expand_as_sparse_dict,
                                     auto_drop_rightmost_column=auto_drop_rightmost_column)

            self.REFVECS_OBJECT, self.REFVECS_SUPPORT_OBJECTS = self.RefVecsClass.OBJECT, self.RefVecsClass.SUPPORT_OBJECTS
            self.SXNL[2], self.SXNL_SUPPORT_OBJECTS[2] = self.RefVecsClass.OBJECT, self.RefVecsClass.SUPPORT_OBJECTS
            self.refvecs_is_expanded = True
            # DONT DELETE RefVecsClass YET, MIGHT NEED IT FOR train-dev-tests SPLIT


    def train_dev_test_split(self, MASK=None):

        while True:  # TO ALLOW break FOR NON-EXPANDED DATA OR T-D_T ALREADY DONE
            if not self.data_is_expanded:
                print(f'\n*** DATA MUST BE EXPANDED BEFORE DOING T-D-T SPLIT (CANNOT BE EXPANDED AFTER SPLIT) ***\n')
                break

            if self.TDT_split_is_done:
                print(f'\n*** TRAIN / DEV / TEST SPLIT HAS ALREADY BEEN DONE AND CANNOT BE CHANGED ***\n')
                break

            TDTSplitClass = mlddt.MLTrainDevTestSplitNEWSUPOBJS(
                                        DATA=self.DATA,
                                        TARGET=self.TARGET,
                                        REFVECS=self.REFVECS,
                                        data_given_orientation=self.DataClass.return_orientation,
                                        target_given_orientation=self.TargetClass.return_orientation,
                                        refvecs_given_orientation=self.RefVecsClass.return_orientation,
                                        bypass_validation=self.bypass_validation)

            __ = vui.validate_user_str(f'Split via MASK(m), RANDOM(r), PARTITION(p), CATEGORY(c) > ', 'MRPC')

            if __ == 'M':
                TDTSplitClass.mask(MASK=MASK)
            elif __ == 'R':
                TDTSplitClass.random(dev_count=None, dev_percent=None, test_count=None, test_percent=None)
            elif __ == 'P':
                TDTSplitClass.partition(number_of_partitions=None, dev_partition_number=None, test_partition_number=None)
            elif __ == 'C':
                TDTSplitClass.category(object_name_for_dev=None, dev_column_name_or_idx=None, DEV_SPLIT_ON_VALUES_AS_LIST=None,
                    object_name_for_test=None, test_column_name_or_idx=None, TEST_SPLIT_ON_VALUES_AS_LIST=None,
                    DATA_FULL_SUPOBJ=self.DataClass.SUPPORT_OBJECTS,
                    TARGET_FULL_SUPOBJ=self.TargetClass.SUPPORT_OBJECTS,
                    REFVECS_FULL_SUPOBJ=self.RefVecsClass.SUPPORT_OBJECTS)

            self.TRAIN_SWNL = TDTSplitClass.TRAIN
            self.DEV_SWNL = TDTSplitClass.DEV   # COULD (SHOULD) BE None IF ONLY TRAIN/TEST WAS DONE
            self.TEST_SWNL = TDTSplitClass.TEST

            # NO MATTER WHAT IS PASSED TO CreateSXNL, A DATA, TARGET, AND REFVECS OBJECT SHOULD ALWAYS BE CREATED IF NOT PROVIDED
            # SO self.TRAIN, self.DEV, self.TEST MUST ALWAYS BE TUPLES OF 3
            self.TRAIN_DATA = self.TRAIN_SWNL[0]
            self.TRAIN_TARGET = self.TRAIN_SWNL[1]
            self.TRAIN_REFVECS = self.TRAIN_SWNL[2]

            if self.DEV_SWNL is None: self.DEV_DATA, self.DEV_TARGET, self.DEV_REFVECS = None, None, None
            else: self.DEV_DATA, self.DEV_TARGET, self.DEV_REFVECS = self.DEV_SWNL[0], self.DEV_SWNL[1], self.DEV_SWNL[2]

            self.TEST_DATA = self.TEST_SWNL[0]
            self.TEST_TARGET = self.TEST_SWNL[1]
            self.TEST_REFVECS = self.TEST_SWNL[2]

            del self.DATA, self.TARGET, self.REFVECS, self.DataClass, self.TargetClass, self.RefVecsClass, TDTSplitClass

            print(f'\n*** TRAIN-DEV-TEST SPLIT COMPLETED SUCCESSFULLY ***\n')
            break


    # IN CASE EVER DECIDE TO DO THIS... LOOKS LIKE A PAIN IN THE NECK TO MAKE THIS DO TRAIN_SWNL, DEV_SWNL, TEST_SWNL
    # def data_to_row(self): pass
    # def data_to_column(self): pass
    # def data_to_array(self): pass
    # def data_to_dict(self): pass
    # def target_to_row(self): pass
    # def target_to_column(self): pass
    # def target_to_array(self): pass
    # def target_to_dict(self): pass
    # def refvecs_to_row(self): pass
    # def refvecs_to_column(self): pass
    # def refvecs_to_array(self): pass
    # def refvecs_to_dict(self): pass







