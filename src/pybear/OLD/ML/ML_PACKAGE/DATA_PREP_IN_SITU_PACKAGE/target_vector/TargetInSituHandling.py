import numpy as n
from copy import deepcopy
from data_validation import validate_user_input as vui
from ML_PACKAGE.GENERIC_PRINT import print_object_create_success as pocs
from ML_PACKAGE.DATA_PREP_IN_SITU_PACKAGE.target_vector import TargetInSituConfigRun as ticr
from MLObjects.SupportObjects import master_support_object_dict as msod


# CREATE A MODULE TO HANDLE CREATING WORKING TARGET VECTOR AT ALL THE DIFFERENT PLACES THIS COULD BE DONE INSITU
# -- AT FIRST CREATION ON FIRST ENTRY TO INSITU
# -- AFTER FIRST CREATION, SELECTED FROM THE InSitu MAIN MENU
# -- IN A ML RUN MODULE, FROM MENU AFTER RUNNING


class TargetInSituHandling:

    def __init__(self, standard_config, target_config, SUPER_RAW_NUMPY_LIST, SUPER_WORKING_NUMPY_LIST,
                 WORKING_SUPOBJS, split_method, LABEL_RULES, number_of_labels, event_value, negative_value):

        self.standard_config = standard_config
        self.target_config = target_config
        self.SUPER_RAW_NUMPY_LIST = SUPER_RAW_NUMPY_LIST
        self.SUPER_WORKING_NUMPY_LIST = SUPER_WORKING_NUMPY_LIST
        self.WORKING_SUPOBJS = WORKING_SUPOBJS
        self.split_method = split_method
        self.LABEL_RULES = LABEL_RULES
        self.number_of_labels = number_of_labels
        self.event_value = event_value
        self.negative_value = negative_value


    def return_fxn(self):
        return self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS, self.split_method, self.LABEL_RULES, self.number_of_labels, \
               self.event_value, self.negative_value


    def run(self):
        while True:

            # BUILD RAW_TARGET_SOURCE USING ROW_IDXs IN SWNL REF_VEC[0]
            # IF SWNL HAS LOST ROWS DURING RUNNING MLs DUE TO USER EDITS, AND LOOKING TO REBUILD WORKING TARGET FROM RAW,
            # MUST ONLY GET REMAINING ROWS FROM SRNL (MEANING USE REF_VEC IDX FROM WORKING TO GET CORRECT IDX FROM RAW)

            if n.array_equiv(self.SUPER_WORKING_NUMPY_LIST, [[[]]]) or \
                    n.array_equiv(self.SUPER_RAW_NUMPY_LIST[2][0], self.SUPER_WORKING_NUMPY_LIST[2][0]):
                print(f'\nBuilding RAW_TARGET_SOURCE as copy of RAW_TARGET...')
                RAW_TARGET_SOURCE = self.SUPER_RAW_NUMPY_LIST[1].copy()   # IF SWNL EMPTY MUST USE RAW TARGET AS TARGET_SOURCE
            else:
                print(f'\nBuilding RAW_TARGET_SOURCE from REF_VEC idxs...')
                KEEP_IDXS = n.zeros(len(self.SUPER_RAW_NUMPY_LIST[1][0]), dtype=bool)
                KEEP_IDXS[self.SUPER_RAW_NUMPY_LIST[2][0]] = True
                RAW_TARGET_SOURCE = self.SUPER_RAW_NUMPY_LIST[1][..., KEEP_IDXS]
                del KEEP_IDXS
            print(f'Done.')
            RAW_TARGET_SOURCE_HEADER = deepcopy(self.WORKING_SUPOBJS[1][msod.QUICK_POSN_DICT()["HEADER"]])

            TARGET = []

            TARGET_HOLDER, TARGET_SUPOBJS_HOLDER, split_method_holder, LABEL_RULES_HOLDER, number_of_labels_holder, \
            event_value_holder, negative_value_holder = \
                ticr.TargetInSituConfigRun(self.standard_config, self.target_config, RAW_TARGET_SOURCE,
                                           RAW_TARGET_SOURCE_HEADER, TARGET, self.WORKING_SUPOBJS[1]).configrun()

            pocs.print_object_create_success(self.standard_config, 'TARGET VECTOR')

            user_finally = vui.validate_user_str(f'Accept TARGET VECTOR config? yes(y) retry(r) abort(a) > ', 'YRA')
            if user_finally == 'Y':
                self.SUPER_WORKING_NUMPY_LIST[1] = TARGET_HOLDER
                self.WORKING_SUPOBJS[1] = TARGET_SUPOBJS_HOLDER

                self.split_method = split_method_holder
                self.LABEL_RULES = LABEL_RULES_HOLDER
                self.number_of_labels = number_of_labels_holder
                self.event_value = event_value_holder
                self.negative_value = negative_value_holder

                del TARGET_HOLDER, TARGET_SUPOBJS_HOLDER
                break

            elif user_finally == 'R': continue
            elif user_finally == 'A': break

        return self.return_fxn()





if __name__ == '__main__':
    from debug import IdentifyObjectAndPrint as ioap
    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl

    _rows = 100
    RAW_TARGET = n.random.choice(['O','X','Y','Z'], _rows, replace=True).reshape((1,-1)).astype('<U1')


    SXNLClass = csxnl.CreateSXNL(rows=_rows,
                                 bypass_validation=False,
                                 data_return_format='ARRAY',
                                 data_return_orientation='COLUMN',
                                 DATA_OBJECT=None,
                                 DATA_OBJECT_HEADER=None,
                                 DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 data_override_sup_obj=None,
                                 data_given_orientation=None,
                                 data_columns=5,
                                 DATA_BUILD_FROM_MOD_DTYPES=['INT', 'FLOAT', 'STR'],
                                 DATA_NUMBER_OF_CATEGORIES=10,
                                 DATA_MIN_VALUES=-10,
                                 DATA_MAX_VALUES=10,
                                 DATA_SPARSITIES=50,
                                 DATA_WORD_COUNT=20,
                                 DATA_POOL_SIZE=200,

                                 target_return_format='ARRAY',
                                 target_return_orientation='COLUMN',
                                 TARGET_OBJECT=RAW_TARGET,
                                 TARGET_OBJECT_HEADER=[['TARGET']],
                                 TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 target_type='BINARY',  # MUST BE 'BINARY','FLOAT', OR 'SOFTMAX'
                                 target_override_sup_obj=None,
                                 target_given_orientation='COLUMN',

                                 target_sparsity=None,
                                 target_build_from_mod_dtype=None,  # COULD BE FLOAT OR INT
                                 target_min_value=None,
                                 target_max_value=None,
                                 target_number_of_categories=None,

                                 refvecs_return_format='ARRAY',  # IS ALWAYS ARRAY (WAS, CHANGED THIS 4/6/23)
                                 refvecs_return_orientation='COLUMN',
                                 REFVECS_OBJECT=None,
                                 REFVECS_OBJECT_HEADER=None,
                                 REFVECS_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 REFVECS_BUILD_FROM_MOD_DTYPES=['STR', 'STR', 'STR'],
                                 refvecs_override_sup_obj=None,
                                 refvecs_given_orientation=None,
                                 refvecs_columns=3,
                                 REFVECS_NUMBER_OF_CATEGORIES=10,
                                 REFVECS_MIN_VALUES=-10,
                                 REFVECS_MAX_VALUES=10,
                                 REFVECS_SPARSITIES=50,
                                 REFVECS_WORD_COUNT=20,
                                 REFVECS_POOL_SIZE=200
    )


    SUPER_RAW_NUMPY_LIST = SXNLClass.SXNL.copy()
    SUPER_RAW_NUMPY_LIST[1] = RAW_TARGET
    RAW_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS.copy()

    SXNLClass.expand_data(expand_as_sparse_dict=False, auto_drop_rightmost_column=False)

    SUPER_WORKING_NUMPY_LIST = SXNLClass.SXNL.copy()
    WORKING_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS.copy()


    standard_config = 'AA'
    gmlr_config = 'Z'

    WORKING_CONTEXT = []
    WORKING_KEEP = deepcopy(WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()['HEADER']])
    split_method = 'NONE'
    LABEL_RULES = []
    number_of_labels = 1
    event_value = ''
    negative_value = ''
    module = __name__
    USE_COLUMNS = list(range(len(SUPER_WORKING_NUMPY_LIST[0])))

    target_config = 'Z'

    SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, split_method, LABEL_RULES, number_of_labels, event_value, negative_value = \
    TargetInSituHandling(standard_config, target_config, SUPER_RAW_NUMPY_LIST, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS,
                         split_method, LABEL_RULES, number_of_labels, event_value, negative_value).run()


    ioap.IdentifyObjectAndPrint(SUPER_WORKING_NUMPY_LIST[2], 'TARGET', __name__, 19, 1).run()







