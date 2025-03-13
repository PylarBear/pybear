import inspect, os, sys, warnings
import numpy as np, pandas as pd
from copy import deepcopy
from debug import IdentifyObjectAndPrint as ioap, get_module_name as gmn
from general_data_ops import return_uniques as ru, get_shape as gs
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import ValidateObjectType as vot, ValidateCharSeqDataType as vcsdt
from general_list_ops import list_of_lists_merger as llm, list_select as ls
from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.filter_functions import split_str_cutoff_filter as  sscf
from ML_PACKAGE.GENERIC_PRINT import print_post_run_options as ppro, obj_info as oi
from MLObjects.PrintMLObject import print_object_preview as pop
from MLObjects.SupportObjects import PrintSupportContents as psc, master_support_object_dict as msod, SupObjConverter as soc
from MLObjects.TestObjectCreators.SXNL import NewObjsToOldSXNL as notos
from read_write_file.generate_full_filename import base_path_select as bps, filename_enter as fe


# from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.class_BaseBMRVTVTMBuild import BaseBMRVTVTMBuild as bob


# LIST OF Filter FUNCTIONS ###################################################################################
# standard_config_source                (returns as fxn call - standard config file to read, to return filtered objects - someday)
# menu_datastep_commands                (list of commands for performing "data step")
# menu_filter_commands                  (list of commands for filtering)
# menu_generic_commands                 (list of general commands)
# menu_hidden_commands                  (hidden commands)
# initialize                            (generate validated data and prompt user for modified data, PreRun only)
# select_obj_and_col                    (returns idx of object and column)
# get_charseq_types                     (returns type, runs ValidateCharSeq on a single str/float/int type object)
# get_uniques                           (returns uniques from a list type object)
# undo                                  (puts everything back to the state before the last pass thru the while loop)
# modified_validated_override_template  (a template to standardize the manual editting of these objects)
# num_or_str_obj_select                 (user select obj idx from list of objs that have either num or str)
# num_or_str_col_select                 (allow selection of only num or str columns as spec. by user num filt, cat filt, use other, min cutoff)
# column_chop                           (chops a user-defined column from object and object header)
# delete_rows
# insert_other_column                   (creates "OTHER" column for a column)
# duplicate_column                      (create a duplicate of user-selected column, returns obj & col idx of created col)
# rename_column                         (rename a column)
# print_frequencies                     (print frequencies of categorical or float data)
# print_min_max_mean_median_stdev       (print statistics of numeric data)
# number_filter_options                 (holds all options for filtering by number, also becomes a str recording filtering of a column)
# number_filter_select                  (prompt user to select an option called out of number_filter_options())
# number_replace_options                (holds all options for replacing a number)
# number_replace_select                 (prompt user to select an option called out of number_replace_options())
# one_thing_left                        (sees if only one thing left in a column, for all columns in all OBJECTS)
# pending_cutoff                        (indicates if there is pending min cutoff filtering to be applied)
# return                                (return for PreRun or InSitu)
# config_run                            (exe)
#######################################################################################################################
# column_desc                           (returns object name and column name)
# validated_datatypes                   (initial read of validated types)
# validated_datatypes_override          (user override of validated types)
# modified_datatypes                    (initial config of modified types)
# modified_datatypes_update             (user update to modified types)
# print_preview                         (prints single user-defined object as df)
# print_preview_all_objects             (prints DATA, REF VECS, TARGET, & TEST as DFs)
# print_cols_and_setup_parameters       (prints column name, validated type, user type, min cutoff, "other", & filtering for all columns in all objects)
#########################################################################################################################


class PreRunFilter:
    def __init__(self,
                 standard_config,
                 user_manual_or_standard,
                 filter_method,
                 SUPER_RAW_NUMPY_LIST,
                 data_given_orientation,
                 target_given_orientation,
                 refvecs_given_orientation,
                 FULL_SUPOBJS,
                 CONTEXT,
                 KEEP,
                 # *******************************************
                 SUPER_RAW_NUMPY_LIST_BASE_BACKUP,
                 FULL_SUPOBJS_BASE_BACKUP,
                 CONTEXT_BASE_BACKUP,
                 KEEP_BASE_BACKUP,
                 # *******************************************
                 SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP,
                 FULL_SUPOBS_GLOBAL_BACKUP,
                 KEEP_GLOBAL_BACKUP,
                 bypass_validation
                 ):
                # 12-12-21 / 9-2-22 SRNL & KEEP GLOBAL_BACKUP SHOULD EXIST BY BEING CREATED JUST BEFORE CALLING PreRunConfigRun
                # NO GLOBAL BACKUPS YET FOR VDATATYPES, MDATATYPES, FILTERING, MIN_CUTOFF, USE_OTHER, CONTEXT THESE ARE ALL
                # EMPTY AT GLOBAL SNAPSHOT POINT SO INGEST DUMS FOR PreRun.
                # BASE BACKUPS ARE NECESSARY FOR InSitu, BUT NOT HERE, SO INGEST DUMS FOR PreRun.


        ################################################################################################################
        # BEAR 6/29/23 THIS COMES OUT WHEN CONVERTED OVER TO NEW SUPOBJS ###############################################
        # NON-BACKUP OBJECTS ##################################
        SXNLClass = notos.NewObjsToOldSXNL(
                                           *SUPER_RAW_NUMPY_LIST,
                                           *FULL_SUPOBJS,
                                           data_orientation=data_given_orientation,
                                           target_orientation=target_given_orientation,
                                           refvecs_orientation=refvecs_given_orientation,
                                           bypass_validation=False
        )

        SUPER_RAW_NUMPY_LIST = SXNLClass.SXNL

        VALIDATED_DATATYPES = SXNLClass.VALIDATED_DATATYPES
        MODIFIED_DATATYPES = SXNLClass.MODIFIED_DATATYPES
        FILTERING = SXNLClass.FILTERING
        MIN_CUTOFFS = SXNLClass.MIN_CUTOFFS
        USE_OTHER = SXNLClass.USE_OTHER
        START_LAG = SXNLClass.START_LAG
        END_LAG = SXNLClass.END_LAG
        SCALING = SXNLClass.SCALING
        del SXNLClass
        # END NON-BACKUP OBJECTS ###################################

        # BASE BACKUP OBJECTS ######################################
        SXNLBackupClass = notos.NewObjsToOldSXNL(
                                                    *SUPER_RAW_NUMPY_LIST_BASE_BACKUP,
                                                    *FULL_SUPOBJS_BASE_BACKUP,
                                                    data_orientation=data_given_orientation,
                                                    target_orientation=target_given_orientation,
                                                    refvecs_orientation=refvecs_given_orientation,
                                                    bypass_validation=False
        )

        SUPER_RAW_NUMPY_LIST_BASE_BACKUP = SXNLBackupClass.SXNL

        VALIDATED_DATATYPES_BASE_BACKUP = SXNLBackupClass.VALIDATED_DATATYPES
        MODIFIED_DATATYPES_BASE_BACKUP = SXNLBackupClass.MODIFIED_DATATYPES
        FILTERING_BASE_BACKUP = SXNLBackupClass.FILTERING
        MIN_CUTOFFS_BASE_BACKUP = SXNLBackupClass.MIN_CUTOFFS
        USE_OTHER_BASE_BACKUP = SXNLBackupClass.USE_OTHER
        START_LAG_BASE_BACKUP = SXNLBackupClass.START_LAG
        END_LAG_BASE_BACKUP = SXNLBackupClass.END_LAG
        SCALING_BASE_BACKUP = SXNLBackupClass.SCALING
        del SXNLBackupClass
        # END BASE BACKUP OBJECTS #########################################

        # GLOBAL BACKUP OBJECTS ###########################################
        SXNLGlobalBackupClass = notos.NewObjsToOldSXNL(
                                                        *SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP,
                                                        *FULL_SUPOBJS_GLOBAL_BACKUP,
                                                        data_orientation=data_given_orientation,
                                                        target_orientation=target_given_orientation,
                                                        refvecs_orientation=refvecs_given_orientation,
                                                        bypass_validation=False
        )

        SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP = SXNLGlobalBackupClass.SXNL

        VALIDATED_DATATYPES_GLOBAL_BACKUP = SXNLGlobalBackupClass.VALIDATED_DATATYPES
        MODIFIED_DATATYPES_GLOBAL_BACKUP = SXNLGlobalBackupClass.MODIFIED_DATATYPES
        del SXNLBackupClass
        # END GLOBAL BACKUP OBJECTS #########################################

        # END BEAR 6/29/23 THIS COMES OUT WHEN CONVERTED OVER TO NEW SUPOBJS ###########################################
        ################################################################################################################



        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        self.mode = self.mode()

        self.FXN_LIST = [self.menu_datastep_commands(), self.menu_generic_commands(),
                         self.menu_filter_commands(), self.menu_hidden_commands()]


        ####################################################################################################################
        # allowed_command_strings ######################################################################################
        # CHANGE THE COMMANDS FOR PreRun OR InSitu HERE ################################################################
        # FOR xxxSelectionsPrint ALLOWABLES, ALLOWABLES for vui.validate_user_str AND ALSO DETERMINE WHICH
        # COMMAND TEXT LINES ARE PRINTED FROM THE commands ABOVE
        if self.mode == 'INSITU':
            self.filter_str = 'FGHM0157'
            self.datastep_str = 'DEIKLNORTWVYZ2346'
            self.generic_str = 'ABCJPQSUX'
            self.hidden_str = '!@#$%^'

        if self.mode == 'PRERUN':
            self.filter_str = 'FGHM01'
            self.datastep_str = 'DEIKLNOTVYZ2346'  # 12-20-21 6:14PM 'RW' COMES OUT FOR PreRun (IN PreRun BASE & SESSION RESET = GLOBAL RESET)
            self.generic_str = 'ABCJPQSUX'
            self.hidden_str = '!@#$%^'

        self.allowed_commands_string = self.filter_str + self.datastep_str + self.generic_str + self.hidden_str

        #### JUST FOR DEBUG TAKE OUT WHEN DONE############################################################
        # CHECK THAT MENU OPTIONS AND STRS OF ALLOWABLES AGREE#############
        for str_idx, alwd_str in enumerate((self.datastep_str, self.generic_str, self.filter_str, self.hidden_str)):
            for alwd_char in alwd_str:
                # TEST IF CHAR IN ALLOWABLE STRING IS IN THE CORRESPONDING MENU OPTIONS
                if True in map(lambda x: f'({alwd_char.upper()})' in x, map(str.upper, self.FXN_LIST[str_idx])): break
                else: raise Exception(f'ALLOWABLE OPTION ({alwd_char.lower()}) NOT IN MENU OF COMMANDS')
        #### JUST FOR DEBUG TAKE OUT WHEN DONE##############################################################################
        # END allowed_command_strings ######################################################################################
        ####################################################################################################################

        self.standard_config = standard_config
        self.user_manual_or_std = user_manual_or_standard
        self.method = filter_method

        self.pending_cutoffs = 'N'

        self.max_cmd_len = np.max([len(i) for i in self.menu_filter_commands() + self.menu_datastep_commands() + \
                                  self.menu_generic_commands()])  # DONT INCLUDE HIDDEN OPTIONS

        self.max_hdr_len = lambda obj_idx: max(list(map(len, self.SUPER_RAW_NUMPY_LIST[obj_idx + 1][0])))


        self.SXNL_DICT = dict((zip(range(6),
                                        ('DATA', 'DATA_HEADER', 'TARGET', 'TARGET_HEADER', 'REFVECS', 'REFVECS_HEADER'))
                                        ))
        # BEAR THIS DICT IS FOR WHEN CONVERTED TO NEW SUPOBJS
        # self.SXNL_DICT = dict(((0,'DATA'),(1,'TARGET'),(2,'REFVECS')))
        self.SUPER_RAW_NUMPY_LIST = SUPER_RAW_NUMPY_LIST

        self.data_given_orientation = akv.arg_kwarg_validater(data_given_orientation, 'data_given_orientation',
                                              ['ROW', 'COLUMN'], self.this_module, fxn)
        self.target_given_orientation = akv.arg_kwarg_validater(target_given_orientation, 'target_given_orientation',
                                                ['ROW', 'COLUMN'], self.this_module, fxn)
        self.refvecs_given_orientation = akv.arg_kwarg_validater(refvecs_given_orientation, 'refvecs_given_orientation',
                                                ['ROW', 'COLUMN'], self.this_module, fxn)

        self.KEEP = KEEP

        self.OBJ_IDXS = list(range(0, len(self.SUPER_RAW_NUMPY_LIST) - 1, 2))
        self.HDR_IDXS = [_ + 1 for _ in self.OBJ_IDXS]  # NOT TEST MATRIX YET.... ADD 7 WHEN TEST_MATRIX IS ADDED 12-8-21

        print(f'\nCreating global backups of remaining metadata and hyperparameter objects...')

        if self.mode == 'PRERUN':  # DECLARING THESE OBJS HERE ALLOWS INGESTION OF NULLS & JUNK PLACEHOLDERS AT FUNCTION CALL TIME
            self.VALIDATED_DATATYPES = \
                np.fromiter((np.fromiter(('' for col in OBJ), dtype=object) for OBJ in self.SUPER_RAW_NUMPY_LIST), dtype=object)
            self.MODIFIED_DATATYPES = \
                np.fromiter((np.fromiter(('' for col in OBJ), dtype=object) for OBJ in self.SUPER_RAW_NUMPY_LIST), dtype=object)
            self.FILTERING = \
                np.fromiter((np.fromiter(([] for _ in __), dtype=object) for __ in self.SUPER_RAW_NUMPY_LIST), dtype=object)
            self.MIN_CUTOFFS = \
                np.fromiter((np.fromiter((0 for _ in __), dtype=object) for __ in self.SUPER_RAW_NUMPY_LIST), dtype=object)
            self.USE_OTHER = \
                np.fromiter((np.fromiter(('N' for _ in __), dtype=object) for __ in self.SUPER_RAW_NUMPY_LIST), dtype=object)
            self.START_LAG = \
                np.fromiter((np.fromiter((0 for _ in __), dtype=object) for __ in self.SUPER_RAW_NUMPY_LIST), dtype=object)
            self.END_LAG = \
                np.fromiter((np.fromiter((0 for _ in __), dtype=object) for __ in self.SUPER_RAW_NUMPY_LIST), dtype=object)
            self.SCALING = \
                np.fromiter((np.fromiter(('' for col in OBJ), dtype=object) for OBJ in self.SUPER_RAW_NUMPY_LIST), dtype=object)
            self.CONTEXT = []
            self.SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP = SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP
            self.KEEP_GLOBAL_BACKUP = KEEP_GLOBAL_BACKUP
            # self.VALIDATED_DATATYPES_GLOBAL_BACKUP = deepcopy(self.VALIDATED_DATATYPES)
            # self.MODIFIED_DATATYPES_GLOBAL_BACKUP = deepcopy(self.MODIFIED_DATATYPES)
            self.CONTEXT_GLOBAL_BACKUP = self.CONTEXT.copy()

        elif self.mode == 'INSITU':
            self.VALIDATED_DATATYPES = VALIDATED_DATATYPES
            self.MODIFIED_DATATYPES = MODIFIED_DATATYPES
            self.FILTERING = FILTERING
            self.MIN_CUTOFFS = MIN_CUTOFFS
            self.USE_OTHER = USE_OTHER
            self.START_LAG = START_LAG
            self.END_LAG = END_LAG
            self.SCALING = SCALING
            self.CONTEXT = CONTEXT
            self.SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP = SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP
            self.KEEP_GLOBAL_BACKUP = KEEP_GLOBAL_BACKUP
            self.VALIDATED_DATATYPES_GLOBAL_BACKUP = VALIDATED_DATATYPES_GLOBAL_BACKUP
            self.MODIFIED_DATATYPES_GLOBAL_BACKUP = MODIFIED_DATATYPES_GLOBAL_BACKUP
            self.CONTEXT_GLOBAL_BACKUP = []

        print(f'Done.')

        # SESSION BACKUP INITIALIZED AS INCOMING OBJECTS, BUT CAN BE OVERWROTE AS CHECKPOINT DURING FILTERING BY USER CMD
        # A SNAPSHOT OF OBJECTS B4 CHOICE TO FILTER WAS MADE--- FOR PreRun THIS IS STATE AFTER CREATION, FOR InSitu
        # THIS IS THE STATE UPON EXIT FROM LAST FILTERING SESSION (IN InSitu USER CAN ABORT A FILTERING SESSION AND EXIT
        # AND STILL RETAIN ANY PREVIOUS FILTERING THAT WAS DONE WITHOUT HAVING TO GO ALL THE WAY BACK TO BASE OR GLOBAL)

        if vui.validate_user_str(f'\nCreate SESSION_BACKUP objects? (y/n) > ', 'YN') == 'Y':
            print(f'\nCreating filtering session backup copies...')

            if self.mode == 'PRERUN':
                self.SUPER_RAW_NUMPY_LIST_SESSION_BACKUP = [_.copy() for _ in self.SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP]
                self.KEEP_SESSION_BACKUP = self.KEEP_GLOBAL_BACKUP
            elif self.mode == 'INSITU':
                self.SUPER_RAW_NUMPY_LIST_SESSION_BACKUP = [_.copy() for _ in self.SUPER_RAW_NUMPY_LIST]
                self.KEEP_SESSION_BACKUP = deepcopy(self.KEEP)

            self.VALIDATED_DATATYPES_SESSION_BACKUP = deepcopy(self.VALIDATED_DATATYPES)
            self.MODIFIED_DATATYPES_SESSION_BACKUP = deepcopy(self.MODIFIED_DATATYPES)
            self.FILTERING_SESSION_BACKUP = deepcopy(self.FILTERING)
            self.MIN_CUTOFFS_SESSION_BACKUP = deepcopy(self.MIN_CUTOFFS)
            self.USE_OTHER_SESSION_BACKUP = deepcopy(self.USE_OTHER)
            self.START_LAG_SESSION_BACKUP = deepcopy(self.START_LAG)
            self.END_LAG_SESSION_BACKUP = deepcopy(self.END_LAG)
            self.SCALING_SESSION_BACKUP = deepcopy(self.SCALING)
            self.CONTEXT_SESSION_BACKUP = deepcopy(self.CONTEXT)

            print(f'Done.')

        else:
            self.SUPER_RAW_NUMPY_LIST_SESSION_BACKUP = ''
            self.KEEP_SESSION_BACKUP = ''
            self.VALIDATED_DATATYPES_SESSION_BACKUP = ''
            self.MODIFIED_DATATYPES_SESSION_BACKUP = ''
            self.FILTERING_SESSION_BACKUP = ''
            self.MIN_CUTOFFS_SESSION_BACKUP = ''
            self.USE_OTHER_SESSION_BACKUP = ''
            self.START_LAG_SESSION_BACKUP = ''
            self.END_LAG_SESSION_BACKUP = ''
            self.SCALING_SESSION_BACKUP = ''
            self.CONTEXT_SESSION_BACKUP = ''

        # 12-21-21 BASE OBJECT RESTORE DOESNT EXIST & NOT ALLOWED FOR PreRun, SO THIS IS RUNNING FOR NOTHING
        # WHEN InSitu, BASE_BACKUPS ARE PreRun's OUTPUT (AND STATE BEING CAPTURED AT TIME OF __init__ OF InSitu)
        if self.mode == 'INSITU':
            print(f'\nCreating backups of base objects for InSitu filtering.')
            self.SUPER_RAW_NUMPY_LIST_BASE_BACKUP = np.array([_.copy() for _ in SUPER_RAW_NUMPY_LIST_BASE_BACKUP], dtype=object)
            self.KEEP_BASE_BACKUP = deepcopy(KEEP_BASE_BACKUP)
            self.VALIDATED_DATATYPES_BASE_BACKUP = deepcopy(VALIDATED_DATATYPES_BASE_BACKUP)
            self.MODIFIED_DATATYPES_BASE_BACKUP = deepcopy(MODIFIED_DATATYPES_BASE_BACKUP)
            self.FILTERING_BASE_BACKUP = deepcopy(FILTERING_BASE_BACKUP)
            self.MIN_CUTOFFS_BASE_BACKUP = deepcopy(MIN_CUTOFFS_BASE_BACKUP)
            self.USE_OTHER_BASE_BACKUP = deepcopy(USE_OTHER_BASE_BACKUP)
            self.START_LAG_BASE_BACKUP = deepcopy(START_LAG_BASE_BACKUP)
            self.END_LAG_BASE_BACKUP = deepcopy(END_LAG_BASE_BACKUP)
            self.SCALING_BASE_BACKUP = deepcopy(SCALING_BASE_BACKUP)
            self.CONTEXT_BASE_BACKUP = deepcopy(CONTEXT_BASE_BACKUP)
            print(f'Done.')

        # 12-6-21 9:07 AM
        # HAVE 3 DIFFERENT BACKUP OBJECTS "GLOBAL" IS THE FLOOR STATE -- FOR BOTH PreRun AND InSitu, THE STATE
        # OF SUPER_NUMPY ET AL AFTER FIRST CREATION OF THE SUPER NUMPY OBJECTS (NO FILTERING APPLIED) ,
        # "BASE" IS THE STATE AFTER PreRun, FOR PreRun, THIS IS JUST DUMMIES BECAUSE THEY HAVENT BEEN CREATED YET,
        # FOR InSitu, THE STATE OF SUPER NUMPY ET AL AFTER PreRun FILTERING
        # "SESSION BACKUP" IS THE STATE AT THE START OF A FILTERING SESSION, IE, A SNAPSHOT IS MADE OF THE SESSION'S
        # STARTING CONDITION, SO APPLYING THE SESSION RESET JUST PUTS OBJECTS BACK TO THEIR IMMEDIATELY PREVIOUS STATE


    def mode(self):
        return('PRERUN')


    def standard_config_source(self):
        print(f'\n*** FILTER STANDARD CONFIGS NOT AVAILABLE YET ***\n')

        # return sc.FILTER_standard_configs(self.standard_config, self.method,
        #                                   self.SUPER_RAW_NUMPY_LIST, self.CONTEXT, self.KEEP)


    #############################################################################################################
    # THESE HANDLE COMMAND TEXT FOR BOTH PreRun & InSitu, THE APPROPRIATE ITEMS PRINTED OUT FOR EACH CLASS ARE
    # HANDLED BY THE allowed_commands_str
    # USED LETTERS = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456
    # UNUSED LETTERS = 789
    def menu_datastep_commands(self):
        return ['delete column(d)',
                'delete row(z)',
                'rename column(e)',
                'round/floor/ceiling a column of floats(k)',
                'duplicate a column(l)',
                'move a column within or between objects(6)',
                'override validated datatype(n)',
                'change modified datatype(o)',
                'reinitialize val. & mod. datatypes(i)',
                'start over - session reset(r)',
                'start over - global reset(t)',
                'start over - base reset(w)',
                'replace a value(v)',
                'sort(y)',
                'create calculated field(2)',
                'reset target only to base state(3)',
                'reset target only to global state(4)'
                ]
        # RESET OPS UNDER DATASTEP TO ALLOW FOR UNDO OF A RESET (UNDO ONLY ALLOWED FOR DATASTEP & FILTER)


    def menu_filter_commands(self):
        return ['filter feature by category(f)',
                'filter numbers(g)',
                'filter by date(h)--NOT AVAILABLE',
                'filter value/string from selected columns(5)',
                'apply min cutoff filter(m)',
                'apply min cutoff to SPLIT_STR columns(7)',
                'set min cutoff(0)',
                'set "OTHERS"(1)'
                ]

    def menu_generic_commands(self):
        return ['accept config / continue / bypass(a)',
                'abandon session changes and exit filtering(b)',
                'print frequencies(c)',
                'print statistics of numerical data(j)',
                'print a preview of objects as DFs(p)',
                'print column names & data type(q)',
                'call a standard config(s)',
                'undo last operation(u)',
                'save current state as session checkpoint(x)',
                'placeholder(y)',
                'placeholder(z)'
                ]

    def menu_hidden_commands(self):
        return [
            'get object types(!)',
            'print FILTERING(@)',
            'print MIN_CUTOFFS(#)',
            'print USE_OTHER($)',
            'print KEEP(%)',
            'print CONTEXT(^)'
        ]


    def initialize(self):  # THIS IS A SEPARATE FXN FOR SIMPLICITY & EASE OF MANAGING THE DIFF BTWN PreRun & InSitu
        # FOR PreRun CONFIG, VALIDATED TYPES MUST BE GENERATED AND USER MUST SET MODIFIED TYPES, AND IF ABANDONING A PreRun
        # SESSION W/O CHANGES, THE USER STILL MUST AT LEAST SPECIFY THE MODIFIED TYPES
        # USER SETS ACTUAL TYPES FOR INT & FLOAT (EITHER CAN BE TREATED AS FLOAT OR STR)

        # REGENERATE LISTS B4 FXNS SO THEY ARE THE CORRECT len (SHOULD BE ALL '')
        # THIS APPLIES TO BOTH PreRun & InSitu
        '''8-31-2022 HASHING THIS OUT TO SEE IF IT'S ACTUALLY NEEDED, TAKING A REALLY LONG TIME TO DO deepcopy
        print(f'\nGenerating working objects from session backup copies...')
        self.VALIDATED_DATATYPES = deepcopy(self.VALIDATED_DATATYPES_SESSION_BACKUP)
        self.MODIFIED_DATATYPES = deepcopy(self.MODIFIED_DATATYPES_SESSION_BACKUP)
        self.FILTERING = deepcopy(self.FILTERING_SESSION_BACKUP)
        self.MIN_CUTOFFS = deepcopy(self.MIN_CUTOFFS_SESSION_BACKUP)
        self.USE_OTHER = deepcopy(self.USE_OTHER_SESSION_BACKUP)
        self.SUPER_RAW_NUMPY_LIST = [_.copy() for _ in self.SUPER_RAW_NUMPY_LIST_SESSION_BACKUP]
        self.KEEP = deepcopy(self.KEEP_SESSION_BACKUP)
        self.CONTEXT = deepcopy(self.CONTEXT_SESSION_BACKUP)
        print(f'Done.')
        '''

        if self.mode == 'PRERUN':

            print(f'\nUSER MUST SET DATA TYPE FOR EVERY NUMERIC COLUMN')
            self.validated_datatypes()
            self.modified_datatypes()


    def select_obj_and_col(self, obj_or_header):
        fxn = inspect.stack()[0][3]

        xx, yy = self.SUPER_RAW_NUMPY_LIST, self.SXNL_DICT

        obj_or_header = obj_or_header.upper()
        obj_or_header = akv.arg_kwarg_validater(obj_or_header, 'obj_or_header', ['OBJECT', 'HEADER'], self.this_module, fxn).upper()

        mod = 1 if obj_or_header == 'HEADER' else 0

        # IF 'OBJECT' OR 'HEADER', RETURN ACTUAL RESPECTIVE SRNL idx (n FOR OBJ, n+1 FOR HEADER)
        obj_idx = 2 * (
        ls.list_single_select([yy[_] for _ in self.OBJ_IDXS], f'SELECT {obj_or_header}', 'idx')[0]) + mod

        if len(xx[obj_idx + 1 - mod][0]) == 1:  # IF ONLY ONE COLUMN, SKIP USER SELECT, JUST SHOW VALUE
            print(f'\n0) {xx[obj_idx + 1 - mod][0][0]}')
            col_idx = 0
        else:
            col_idx = ls.list_single_select(xx[obj_idx + 1 - mod][0], 'SELECT COLUMN', 'idx')[0]

        return obj_idx, col_idx


    def get_charseq_types(self, OBJECT):
        return [vcsdt.ValidateCharSeqDataType(_).type() for _ in OBJECT]
        # print(LIST_OF_TYPES)

    def get_uniques(self, obj_idx, col_idx):  # UNIQUES OF A LIST OBJECT (LIKE A COLUMN OF DATA)
        # BEAR THIS SHOULD EVENTUALLY COME OUT AND BE REPLACED BY MLObject.get_uniques VIA ApexDataHandling
        return np.unique(self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx])


    def undo(self):
        try:
            self.SUPER_RAW_NUMPY_LIST = np.array([_.copy() for _ in self.SUPER_RAW_NUMPY_LIST_UNDO], dtype=object)
            self.KEEP = deepcopy(self.KEEP_UNDO)
            self.VALIDATED_DATATYPES = deepcopy(self.VALIDATED_DATATYPES_UNDO)
            self.MODIFIED_DATATYPES = deepcopy(self.MODIFIED_DATATYPES_UNDO)
            self.FILTERING = deepcopy(self.FILTERING_UNDO)
            self.MIN_CUTOFFS = deepcopy(self.MIN_CUTOFFS_UNDO)
            self.USE_OTHER = deepcopy(self.USE_OTHER_UNDO)
            self.START_LAG = deepcopy(self.START_LAG_UNDO)
            self.END_LAG = deepcopy(self.END_LAG_UNDO)
            self.SCALING = deepcopy(self.SCALING)
            self.CONTEXT = deepcopy(self.CONTEXT_UNDO)
            print(f'\nUNDO - {self.UNDO_DESC.upper()} - COMPLETE.\n')
        except:
            print(f"\nCAN'T UNDO, BACKUP OBJECTS HAVE NOT BEEN GENERATED YET\n")


    def modified_validated_override_template(self, name):

        name = name.upper()

        while True:

            self.print_cols_and_setup_parameters()

            print(f'SELECT OBJECT AND COLUMN TO CHANGE {name} DATATYPE:')
            obj_idx, col_idx = self.select_obj_and_col('OBJECT')

            # CREATE THIS JUST TO SIMPLIFY THE PRINT STATEMENT BELOW IT
            if name == 'MODIFIED':
                __ = self.MODIFIED_DATATYPES
            elif name == 'VALIDATED':
                __ = self.VALIDATED_DATATYPES

            print(f'\nCURRENT {name} DATATYPE FOR {self.column_desc(obj_idx, col_idx)} IS --- ' + __[obj_idx][col_idx])

            if name == 'MODIFIED':
                self.modified_datatypes_update(obj_idx, col_idx)

                self.print_cols_and_setup_parameters()

            elif name == 'VALIDATED':
                self.validated_datatypes_override(obj_idx, col_idx)
                if self.VALIDATED_DATATYPES[obj_idx][col_idx] == 'STR':
                    # IF USER SET V-TYPE TO 'STR' M-TYPE MUST ALSO BECOME 'STR'
                    self.modified_datatypes_update(obj_idx, col_idx)

                    self.print_cols_and_setup_parameters()
                    print(f'\nUSER SET VALIDATED TYPE TO "STR", MODIFIED TYPE MUST BE AND HAS ALSO BEEN SET TO "STR"')

            _ = vui.validate_user_str(
                f'\nAccept and exit datatype override(a), enter another(b), reset and retry(r), reset and exit datatype override(q) > ',
                'ABRQ')
            if _ == 'A': break
            if _ == 'B': continue
            if _ in ['R', 'Q']:
                self.undo()
                if _ == 'R': continue
                if _ == 'Q': break


    def num_or_str_obj_select(self, number_or_str):   # (12-20-21 number_or_str CAN BE 'STR', 'NUM')

        if number_or_str.upper() not in ['STR', 'NUM']: raise Exception(f'INVALID num_or_str "{number_or_str}"')

        LOOKUP_DICT = {'NUM': list(msod.val_num_dtypes().values()) , 'STR': list(msod.val_text_dtypes().values())}
        NUM_OR_STR_ALLOWABLES = LOOKUP_DICT[number_or_str]   # NEED THIS FOR 'if 'STR' in' in 2 LINES BELOW
        del LOOKUP_DICT

        # CREATE A FANCY THING TO FIND THE OBJS THAT HAVE number_or_str IN THEM (12-20-21 number_or_str CAN BE 'STR', 'NUM')
        __ = [_ for _ in self.OBJ_IDXS if True in map(lambda x: x in self.VALIDATED_DATATYPES[_], NUM_OR_STR_ALLOWABLES)]

        CURRENT_OBJS = [self.SXNL_DICT[i] for i in __]  # RETAIN THIS TO USE FOR "EXIT" STUFF
        ___ = ls.list_single_select(CURRENT_OBJS + [f'EXIT BACK TO MAIN MENU'], f'SELECT OBJECT', 'idx')[0]
        if ___ == len(CURRENT_OBJS): obj_idx = 'BREAK'
        else: obj_idx = __[___]
        del __, CURRENT_OBJS, ___
        return obj_idx


    def num_or_str_col_select(self, number_or_str):    # (12-20-21 number_or_str CAN BE 'STR', 'NUM')

        if number_or_str.upper() not in ['STR', 'NUM']: raise Exception(f'INVALID num_or_str "{number_or_str}"')

        LOOKUP_DICT = {'NUM': list(msod.val_num_dtypes().values()), 'STR': list(msod.val_text_dtypes().values())}
        NUM_OR_STR_ALLOWABLES = LOOKUP_DICT[number_or_str]
        del LOOKUP_DICT

        while True:
            self.print_cols_and_setup_parameters()
            xx, yy = self.SUPER_RAW_NUMPY_LIST, self.SXNL_DICT

            obj_idx = self.num_or_str_obj_select(number_or_str)
            if obj_idx == 'BREAK':  # IF SELECTED 'EXIT BACK' IN num_or_str_obj_select
                obj_idx, col_idx = 'BREAK', ''  # RETURN THIS TO ALSO BREAK OUT OF THE LOOP THAT'S CALLING THIS
                break

            # CREATE LIST OF COLS AVAILABLE TO GO INTO LIST SELECTOR ##########################

            hdr_ljust = self.max_hdr_len(obj_idx) + 5

            print('\nOBJECT = ' + self.SXNL_DICT[obj_idx])
            print(f'\nDATATYPES:')
            print(' ' * 5 + f'COLUMN'.ljust(hdr_ljust) + 'TYPE'.ljust(10) + 'USER TYPE'.ljust(10))

            COL_DESC = []  # CREATE OBJECTS TO HOLD DESCRIPTIONS OF IN-CLASS / DISALLOWED COLUMNS
            DISALLOWED = []  # CREATE LIST OF DISALLOWED SELECTIONS

            # CREATE A HOLDER THAT RETAINS THE col_idx / list_select RELATIONSHIP (list_select COULD BE A PARTIAL LIST OF COLS)
            COL_LIST_IDX_HOLDER = []
            for col_idx in range(len(xx[obj_idx])):
                if self.VALIDATED_DATATYPES[obj_idx][col_idx] in NUM_OR_STR_ALLOWABLES:
                    COL_DESC.append(f'{xx[obj_idx + 1][0][col_idx]}'.ljust(hdr_ljust) + \
                                    f'{self.VALIDATED_DATATYPES[obj_idx][col_idx]}'.ljust(10) + \
                                    f'{self.MODIFIED_DATATYPES[obj_idx][col_idx]}'.ljust(10) + \
                                    f'{" " * 50}'
                                    )
                    COL_LIST_IDX_HOLDER.append(col_idx)
                # else:
                #    COL_DESC.append(f'{xx[obj_idx + 1][0][col_idx]}'.ljust(hdr_ljust) + 'N/A')
                #    DISALLOWED.append(col_idx)
            del hdr_ljust
            # END CREATE LIST OF COLS AVAILABLE TO GO INTO LIST SELECTOR ##########################

            if len(COL_DESC) == 0:
                print(f'\nTHERE ARE NO COLUMNS OF TYPE {NUM_OR_STR_ALLOWABLES} IN {yy[obj_idx]}')
                print(f'ALPHA CHARACTER SEQS CAN ONLY BE DATATYPE "STR" AND CAN ONLY BE FILTERED AS "STR".')
                print(f'IF THE DATA IS NUMERIC AND YOU WANT TO FILTER IT AS A NUMBER, CHECK IF THE CURRENT DATATYPE ')
                print(f'IS "STR", THEN CHANGE THE MODIFIED DATATYPE TO FLOAT OR INT, THEN RETRY FILTERING AS A NUMBER.')
                continue

            elif len(COL_DESC) == 1:  # IF ONLY ONE COLUMN, SKIP USER SELECT, JUST SHOW VALUE
                print(f'0) {COL_DESC[0]}')
                col_idx = COL_LIST_IDX_HOLDER[0]
                break
            else:
                col_idx = COL_LIST_IDX_HOLDER[ls.list_single_select(COL_DESC, f'SELECT COLUMN', 'idx')[0]]
                break

        return obj_idx, col_idx


    def column_chop(self, obj_idx, col_idx_to_be_chopped):

        # OBJECT IDXS ARE 0,2,4,6; HEADER IDXS ARE 1,3,5,7 IN SUPER_NUMPY AND DATATYPES
        # DATATYPES HEADER IDX SLOTS ARE JUST A SINGLE '', NO NEED FOR INFO HERE, JUST A PLACEHOLDER

        if len(self.SUPER_RAW_NUMPY_LIST[obj_idx]) == 1:  # IF ONLY ONE COLUMN, DONT ALLOW DELETE
            print(f'\n{self.SXNL_DICT[obj_idx]} HAS ONLY ONE COLUMN AND CANNOT BE DELETED')
        else:
            xx, aa, bb, cc, dd, ee, ff, gg, hh = \
                self.SUPER_RAW_NUMPY_LIST, self.VALIDATED_DATATYPES, self.MODIFIED_DATATYPES, self.FILTERING, \
                self.MIN_CUTOFFS, self.USE_OTHER, self.START_LAG, self.END_LAG, self.SCALING

            # HANDLE THINGS IN FILTER B4 DELETING COLUMN NAME FROM HEADER
            if len(cc[obj_idx][col_idx_to_be_chopped]) != 0:   # IF SOMETHING IN FILTER, MOVE TO CONTEXT BEFORE DELETE.
                for _string in cc[obj_idx][col_idx_to_be_chopped]:
                    self.CONTEXT.append(f'Filtered {self.column_desc(obj_idx, col_idx_to_be_chopped)} by "{_string}", '
                                        f'then later deleted the column.')

            # CHOP COLUMNS IN THE IDX & IDX+1 (HEADER) OBJECTS, WHERE APPLICABLE
            xx[obj_idx] = np.delete(xx[obj_idx], col_idx_to_be_chopped, axis=0)
            xx[obj_idx + 1] = np.delete(xx[obj_idx + 1], col_idx_to_be_chopped, axis=1)
            aa[obj_idx] = np.delete(aa[obj_idx], col_idx_to_be_chopped, axis=0)
            bb[obj_idx] = np.delete(bb[obj_idx], col_idx_to_be_chopped, axis=0)
            # 4-3-22 np.delete ISNT WORKING RIGHT, CLEARING cc[obj_idx] ENTIRELY INSTEAD OF JUST THE ONE COLUMN, REVISTED 10/8/22
            cc[obj_idx] = np.delete(cc[obj_idx], col_idx_to_be_chopped, axis=0)
            dd[obj_idx] = np.delete(dd[obj_idx], col_idx_to_be_chopped, axis=0)
            ee[obj_idx] = np.delete(ee[obj_idx], col_idx_to_be_chopped, axis=0)
            ff[obj_idx] = np.delete(ff[obj_idx], col_idx_to_be_chopped, axis=0)
            gg[obj_idx] = np.delete(gg[obj_idx], col_idx_to_be_chopped, axis=0)
            hh[obj_idx] = np.delete(hh[obj_idx], col_idx_to_be_chopped, axis=0)
            if obj_idx == 0:  # ONLY IF DATA_NUMPY IS ALTERED IS KEEP ALTERED
                self.KEEP = np.delete(self.KEEP, col_idx_to_be_chopped, axis=0)


    # INHERITED
    # def delete_rows(self, ROW_IDXS_AS_INT_OR_LIST):


    # 12-29-2021 THIS SHOULD COME OUT OF HERE, DATA IS BEING EXPANDED SEPARATELY LATER
    def insert_other_column(self, object_idx, home_column_idx):
        xx, bb = self.SUPER_RAW_NUMPY_LIST, self.MODIFIED_DATATYPES
        # INSERT 'OTHER' COLUMNS IN THE IDX & IDX+1 (HEADER) OBJECTS
        # INSERT COLUMN OF BLANKS INTO OBJECT, RIGHT AFTER THE HOME COLUMN
        BLANKS = ['' for _ in range(len(xx[0][0]))]  # USE FIRST COLUMN OF DATA NUMPY FOR LEN
        xx[object_idx] = \
            np.insert(xx[object_idx], home_column_idx, BLANKS, axis=0)
        # INSERT "OTHER" INTO HEADER & MOD_TYPES, RIGHT AFTER HOME COLUMN
        xx[object_idx + 1] = \
            np.insert(xx[object_idx + 1], home_column_idx + 1,
                     f'{xx[object_idx + 1][0][home_column_idx]}' + ' - OTHER', axis=1)
        bb[object_idx] = np.insert(bb[object_idx], home_column_idx + 1, 'STR', axis=1)
        # OBJECT IDXS ARE 0,2,4,6 CORRESPONDING MOD_TYPES IDXS ARE 0,1,2,3


    def duplicate_column(self):
        xx, yy = self.SUPER_RAW_NUMPY_LIST, self.SXNL_DICT
        obj_idx = 2 * ls.list_single_select([yy[_] for _ in self.OBJ_IDXS],
                                            f'\nSELECT OBJECT FOR SOURCE COLUMN TO DUPLICATE', 'idx')[0]
        col_idx = ls.list_single_select(xx[obj_idx + 1][0], f'SELECT COLUMN', 'idx')[0]

        # INSERT A COL INTO EVERY OBJECT IN SRNL ET AL, DUPLICATE EVERYTHING FROM ORIGINAL
        self.SUPER_RAW_NUMPY_LIST[obj_idx] = np.insert(self.SUPER_RAW_NUMPY_LIST[obj_idx], col_idx + 1,
                                                      self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx].copy(), axis=0)

        self.SUPER_RAW_NUMPY_LIST[obj_idx + 1] = np.insert(self.SUPER_RAW_NUMPY_LIST[obj_idx + 1],
                                                          col_idx + 1, deepcopy(
                self.SUPER_RAW_NUMPY_LIST[obj_idx + 1][0][col_idx]) + ' - COPY', axis=1)

        self.VALIDATED_DATATYPES[obj_idx].insert(col_idx + 1,
                                                     deepcopy(self.VALIDATED_DATATYPES[obj_idx][col_idx]), axis=0)
        # DONT HAVE TO DO ANYTHING TO THE BLANK SPACEHOLDER IN THE HEADER POSN OF VAL_DTYPES
        self.MODIFIED_DATATYPES[obj_idx].insert(col_idx + 1, deepcopy(self.MODIFIED_DATATYPES[obj_idx][col_idx]), axis=0)
        # DONT HAVE TO DO ANYTHING TO THE BLANK SPACEHOLDER IN THE HEADER POSN OF MOD_DTYPES
        self.FILTERING[obj_idx].insert(col_idx + 1, deepcopy(self.FILTERING[obj_idx][col_idx]), axis=0)
        # DONT HAVE TO DO ANYTHING TO THE BLANK SPACEHOLDER IN THE HEADER POSN OF FILTERING
        self.MIN_CUTOFFS[obj_idx].insert(col_idx + 1, deepcopy(self.MIN_CUTOFFS[obj_idx][col_idx]), axis=0)
        # DONT HAVE TO DO ANYTHING TO THE BLANK SPACEHOLDER IN THE HEADER POSN OF MIN_CUTOFFS
        self.USE_OTHER[obj_idx].insert(col_idx + 1, deepcopy(self.USE_OTHER[obj_idx][col_idx]), axis=0)
        # DONT HAVE TO DO ANYTHING TO THE BLANK SPACEHOLDER IN THE HEADER POSN OF USE_OTHER
        self.START_LAG[obj_idx].insert(col_idx + 1, deepcopy(self.START_LAG[obj_idx][col_idx]), axis=0)
        # DONT HAVE TO DO ANYTHING TO THE BLANK SPACEHOLDER IN THE HEADER POSN OF START_LAG
        self.END_LAG[obj_idx].insert(col_idx + 1, deepcopy(self.END_LAG[obj_idx][col_idx]), axis=0)
        # DONT HAVE TO DO ANYTHING TO THE BLANK SPACEHOLDER IN THE HEADER POSN OF END_LAG
        self.SCALING[obj_idx].insert(col_idx + 1, deepcopy(self.SCALING[obj_idx][col_idx]), axis=0)
        # DONT HAVE TO DO ANYTHING TO THE BLANK SPACEHOLDER IN THE HEADER POSN OF SCALING

        if obj_idx == 0:  # ONLY IF DATA_NUMPY IS ALTERED IS KEEP ALTERED (ONLY FOR THE TIME BEING WHILE KEEP HOLDS HEADER INFO)
            self.KEEP.insert(col_idx + 1, deepcopy(self.KEEP[0][col_idx]) + ' - COPY')

        new_title = self.rename_column(obj_idx, col_idx + 1)

        # DONT CHANGE GLOBAL_BACKUP, BASE_BACKUP, SESSION_BACKUP

        print(f'DUPLICATE {self.column_desc(obj_idx, col_idx)} AS "{new_title}" SUCCESSFUL.')

        return obj_idx, col_idx + 1  # RETURN OBJ_IDX AND IDX OF NEW COLUMN


    # INHERITED
    def rename_column(self, obj_idx, col_idx):
        # BEAR THIS WILL EVENTUALLY BE INHERITED FROM ApexDataHandling
        # AND THIS CAN BE DELETED
        while True:
            new_title = input(f'\nENTER NEW COLUMN HEADER (CURRENT HEADER IS ' + \
                              f'"{self.SUPER_RAW_NUMPY_LIST[obj_idx + 1][0][col_idx]}") > ')

            __ = deepcopy(self.SUPER_RAW_NUMPY_LIST[obj_idx + 1][0])
            if new_title in np.delete(__, np.argwhere(__ == new_title)):
                print(
                    f'{new_title} IS ALREADY IN {self.SXNL_DICT[obj_idx]}. ENTER A DIFFERENT COLUMN NAME.')
                continue
            del __

            if vui.validate_user_str(f'USER ENTERED: {new_title} --- ACCEPT? (y/n) > ', 'YN') == 'Y':
                self.SUPER_RAW_NUMPY_LIST[obj_idx + 1][0][col_idx] = new_title
                if obj_idx == 0: self.KEEP[0][col_idx] = new_title  # ONLY WHILE "KEEP" HOLDS HEADER INFO
                break
        return new_title


    def print_frequencies(self, obj_idx, col_idx):
        xx = self.SUPER_RAW_NUMPY_LIST
        # UNIQUES, COUNTS = np.unique(xx[obj_idx][col_idx].astype(object), return_counts=True)
        # 12-23-21 CANT USE np.unique HERE IT CANT HANDLE STR & FLOAT MIXED TOGETHER, EVEN WHEN astype(object)

        UNIQUES, COUNTS = np.unique(xx[obj_idx][col_idx], return_counts=True)

        ARGSORT = np.flip(np.argsort(COUNTS))
        UNIQUES_SORTED = UNIQUES[ARGSORT]
        COUNTS_SORTED = COUNTS[ARGSORT]

        print(f'\nFREQUENCIES IN {self.column_desc(obj_idx, col_idx)}:')
        # GET MAX LEN OUT OF UNIQUES
        freq_max_len = max(8, np.max([len(str(_)) for _ in UNIQUES]) + 3)
        print(f'RANK'.ljust(8) + f'VALUE'.ljust(freq_max_len) + f'CT')
        [print(f'{_ + 1})'.ljust(8) + f'{str(UNIQUES_SORTED[_]).ljust(min(8, freq_max_len))}' + f'{COUNTS_SORTED[_]}')
         for _ in range(len(UNIQUES_SORTED))]


    def print_statistics(self, obj_idx, col_idx):
        xx = self.SUPER_RAW_NUMPY_LIST
        while True:
            if self.VALIDATED_DATATYPES[obj_idx][col_idx] not in ['INT', 'FLOAT', 'BIN']:
                print(f'\n{self.column_desc(obj_idx, col_idx)} IS NOT NUMERIC DATA!\n')
                break
            else:
                try:
                    zz = self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx].astype(float)
                    print(f'\n{self.column_desc(obj_idx, col_idx)}:')
                    print(
                        f'MIN = {np.min(zz)}, MAX = {np.max(zz)}, MEAN = {np.mean(zz)}, MEDIAN = {np.median(zz)}, STDEV = {np.std(zz)}')
                    break

                except:
                    print(f'UNABLE TO PROCESS DATA IN {self.column_desc(obj_idx, col_idx)} AS FLOATS!\n')
                    obj_idx = 'BREAK'
                    break

        return obj_idx


    def column_desc(self, obj_idx, col_idx):
        __ = f'{self.SXNL_DICT[obj_idx]} - {self.SUPER_RAW_NUMPY_LIST[obj_idx + 1][0][col_idx]}'
        return (__)


    def print_cols_and_setup_parameters(self):
        # prints column name, validated type, user type, min cutoff, "other", & filtering for all columns in all objects

        fxn = inspect.stack()[0][3]

        xx = self.SUPER_RAW_NUMPY_LIST

        print(f'\nDATA SETUP HYPERPARAMETERS:')

        for idx in self.OBJ_IDXS:  # idx IS obj_idx
            _ORIENT = dict((zip(self.OBJ_IDXS,(self.data_given_orientation, self.target_given_orientation, self.refvecs_given_orientation))))
            psc.PrintSupportObjects_OldObjects(xx[idx],  # SELECT OBJECT OUT OF SRNL OR SWNL AT INSTANTIATION
                                             self.SXNL_DICT[idx],
                                             orientation=_ORIENT[idx],
                                             _columns=None,
                                             HEADER=xx[idx+1],
                                             VALIDATED_DATATYPES=self.VALIDATED_DATATYPES[idx],
                                             MODIFIED_DATATYPES=self.MODIFIED_DATATYPES[idx],
                                             FILTERING=self.FILTERING[idx],
                                             MIN_CUTOFFS=self.MIN_CUTOFFS[idx],
                                             USE_OTHER=self.USE_OTHER[idx],
                                             START_LAG=self.START_LAG[idx],
                                             END_LAG=self.END_LAG[idx],
                                             SCALING=self.SCALING[idx],
                                             max_hdr_len=max(map(lambda x: self.max_hdr_len(x), self.OBJ_IDXS)),
                                             calling_module=self.this_module,
                                             calling_fxn=fxn
                                             )



    def print_preview(self, data, name, rows, columns, start_row, start_col, header=''):
        pop.print_object_preview(data, name, rows, columns, start_row, start_col,
                                 orientation=self.data_given_orientation, header='')


    def print_preview_all_objects(self, start_row=0, start_col=0):
        xx = self.SUPER_RAW_NUMPY_LIST
        yy = self.SXNL_DICT
        for idx in range(0, len(xx), 2):  # self.OBJ_IDXS:  INCLUDING TEST_MATRIX IN THIS FOR NOW
            try:
                self.print_preview(xx[idx], yy[idx], rows=20, columns=9, start_row=start_row, start_col=start_col, header=xx[idx + 1][0])
                print()
            except:
                print(f'\nBLOWUP TRYING TO PRINT DF OF {yy[idx]}')


    # THIS CREATES AN OBJECT IDENTICAL IN LAYOUT TO SUPER_RAW_NUMPY, BUT HOLDS VALIDATED DATA TYPE FOR EACH COL
    def validated_datatypes(self):  # ITERATES THRU NON-HEADER OBJECTS, DETERMINES VALIDATED DATA TYPE

        xx = self.SUPER_RAW_NUMPY_LIST

        print(f'\nReading raw datatypes...')

        for obj_idx in self.OBJ_IDXS:
            for col_idx in range(len(xx[obj_idx])):
                print(f'Reading {self.SXNL_DICT[obj_idx]} - {self.SUPER_RAW_NUMPY_LIST[obj_idx+1][0][col_idx]}')
                try:
                    # **************************************************************************************
                    # BEAR THIS IS TO GO FAST WHEN DEBUGGING
                    # UNHASH THIS
                    validated_datatype = vot.ValidateObjectType(xx[obj_idx][col_idx]).ml_package_object_type()
                    # DELETE THIS
                    # validated_datatype = 'FLOAT'
                    self.VALIDATED_DATATYPES[obj_idx][col_idx] = validated_datatype

                    if validated_datatype in ['STR', 'BIN']:
                        self.MODIFIED_DATATYPES[obj_idx][col_idx] = validated_datatype

                except:
                    zz = self.column_desc(obj_idx, col_idx)
                    print(f'ValidateObjectType GAVE EXCEPTION WHEN TRYING TO IDENTIFY ' + \
                          f'{zz}')
                    # THIS STAYS UNDER EXCEPT, THIS SHOWS WHAT DATA IS IN BAD COLUMN AND GIVES USER OPTIONS
                    print(f'\n{zz.upper()} LOOKS LIKE:')
                    self.print_preview(xx[obj_idx][col_idx], zz, rows=10, columns=1, start_row=0, start_col=0, header=f'{zz}')

                    _ = vui.validate_user_str(f'Ignore(i), Override(o), or Terminate(t)?', 'IOT')
                    if _ == 'I':
                        self.VALIDATED_DATATYPES[obj_idx][col_idx] = 'EXCEPTION'
                        continue
                    elif _ == 'O':
                        self.validated_datatypes_override(obj_idx, col_idx)
                    elif _ == 'T':
                        raise Exception(f'ValidateObjectType GAVE EXCEPTION WHEN TRYING TO IDENTIFY ' + \
                                                  f'{self.column_desc(obj_idx, col_idx)}, USER TERMINATED.')

        print(f'Done.')

        self.print_cols_and_setup_parameters()

        # PAUSE TO LET USER LOOK AT GENERATED PRINTOUT
        DUM = input('\nPAUSED TO SHOW DATA SETUP PARAMETERS.  HIT ENTER TO CONTINUE > ')


    def validated_datatypes_override(self, obj_idx, col_idx):
        # THIS IS USED IN 2 PLACES, THIS MAY EVENTUALLY END UP BEING A LIST SELECT SO KEEPING THIS SEPARATE FOR FUTURE
        while True:

            valueholder = {'S': 'STR', 'I': 'INT', 'F': 'FLOAT', 'B': 'BIN', 'L':'BOOL'}[vui.validate_user_str(
                f'ENTER NEW VALIDATED DATATYPE -- STR(s), FLOAT(f), INT(i), BIN(b) BOOL(l)> ', 'SFIBL')]

            if vui.validate_user_str(f'USER ENTERED "{valueholder}", accept? (y/n) > ', 'YN') == 'Y':
                self.VALIDATED_DATATYPES[obj_idx][col_idx] = valueholder
                break


    # THIS CREATES AN OBJECT IDENTICAL IN LAYOUT TO SUPER_RAW_NUMPY, BUT HOLDS USER-DECLARED DATA TYPE FOR EACH COL
    def modified_datatypes(self):
        # ITERATES THRU NON-HEADER OBJECTS, FINDS TYPE OF DATA, THEN ALLOWS USER TO DECLARE HOW TO DEAL
        # WITH AN INT & FLOAT, INT CAN BE A FLOAT OR STR, FLOAT CAN BE A FLOAT OR STR

        xx = self.SUPER_RAW_NUMPY_LIST
        MODIFIED_DATATYPES_RESTORE = deepcopy(self.MODIFIED_DATATYPES)

        while True:
            for idx in self.OBJ_IDXS:
                for col_idx in range(len(xx[idx])):
                    self.modified_datatypes_update(idx, col_idx)

            self.print_cols_and_setup_parameters()

            if vui.validate_user_str(f'Accept MODIFIED TYPES? (y/n) > ', 'YN') == 'Y':
                del xx, MODIFIED_DATATYPES_RESTORE
                break
            else:
                print(f'RETURNING USER DATATYPES TO PRIOR STATE AND RESTARTING')
                self.MODIFIED_DATATYPES = deepcopy(MODIFIED_DATATYPES_RESTORE)


    def modified_datatypes_update(self, obj_idx, col_idx):

        fxn = inspect.stack()[0][3]

        type = self.VALIDATED_DATATYPES[obj_idx][col_idx]
        if type in ['STR']:
            while True:
                try:
                    if os.name=='nt':  # BEAR, IF EVER RESOLVE THE LINUX tf ISSUE
                        new_type = {'S': 'STR', 'P':'SPLIT_STR', 'N':'NNLM50'}[vui.validate_user_str(
                            f'CURRENT VALIDATED DATATYPE FOR {self.column_desc(obj_idx, col_idx)} is --- '.ljust(80) + \
                            f'{type}.  Treat as STR(s), SPLIT STRING(p), NNLM50(n)? > ', 'SPN')]
                    elif os.name=='posix':
                        new_type = {'S': 'STR', 'P': 'SPLIT_STR'}[vui.validate_user_str(
                            f'CURRENT VALIDATED DATATYPE FOR {self.column_desc(obj_idx, col_idx)} is --- '.ljust(80) + \
                            f'{type}.  Treat as STR(s), SPLIT STRING(p)? > ', 'SP')]
                    self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx] = self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx].astype(str)
                    break
                except:
                    print(f'\n{self.column_desc(obj_idx, col_idx)} raised exception with {new_type}.  Try again.')
                    continue

        elif type in ['BOOL']:
            new_type = 'STR'

        elif type in ['INT', 'FLOAT', 'BIN']:
            while True:
                try:
                    new_type = {'S': 'STR', 'F': 'FLOAT', 'I': 'INT', 'B':'BIN'}[vui.validate_user_str(
                        f'CURRENT VALIDATED DATATYPE FOR {self.column_desc(obj_idx, col_idx)} is --- '.ljust(80) + \
                        f'{type}.  Treat as STR(s), FLOAT(f), INT(i), BIN(b)? > ', 'SFIB')
                    ]

                    if new_type == 'STR': self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx] = self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx].astype(str)
                    elif new_type == 'FLOAT': self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx] = self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx].astype(np.float64)
                    elif new_type == 'INT': self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx] = self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx].astype(np.int32)
                    elif new_type == 'BIN': self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx] = self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx].astype(np.int8)
                    else: raise TimeoutError
                    break
                except TimeoutError:
                    raise Exception(f'INVALID new_type "{new_type}" IN {self.this_module}.{fxn}() WHEN SETTING MOD_DTYPE FOR NUMERICAL VAL_DTYPE')
                except:
                    print(f'\n{self.column_desc(obj_idx, col_idx)} raised exception with {new_type}.  Try againp.')
                    continue

        self.MODIFIED_DATATYPES[obj_idx][col_idx] = new_type


    def number_filter_options(self, num1, num2):
        return [f'Filter out only number equal to {num1}',
                f'Keep only number equal to {num1}',
                f'Filter out numbers greater than {num1}',
                f'Filter out numbers greater than or equal to {num1}',
                f'Filter out numbers less than {num1}',
                f'Filter out numbers less than or equal to {num1}',
                f'Filter out numbers within range >{num1} and <{num2}',
                f'Filter out numbers outside of range <{num1} and >{num2}',
                f'Exit to main menu']

    def number_filter_select(self, number_filter_option):
        return vui.validate_user_float(f'{self.number_filter_options("> ", "")[number_filter_option]}')

    def number_replace_options(self, num1, num2):
        return [f'Replace one entry based on row index {num1}',
                f'Replace all instances of number equal to {num1}',
                f'Replace numbers greater than {num1}',
                f'Replace numbers greater than or equal to {num1}',
                f'Replace numbers less than {num1}',
                f'Replace numbers less than or equal to {num1}',
                f'Replace numbers within range >{num1} and <{num2}',
                f'Replace numbers outside of range <{num1} and >{num2}',
                f'Exit to main menu']

    def number_replace_select(self, number_replace_option):
        return vui.validate_user_float(
            f'{self.number_replace_options("> ", "")[number_replace_option]}')


    def one_thing_left(self):
        xx = self.SUPER_RAW_NUMPY_LIST[0]
        for col_idx in range(len(xx)-1, -1, -1):
            one_thing_left = False
            if len(np.unique(xx[col_idx])) > 1: break
            else: one_thing_left = True

            if one_thing_left:
                print(f'\n*** ONLY ONE VALUE IN {self.column_desc(0, col_idx)}, {xx[col_idx][0]} ***')
                __ = vui.validate_user_str(
                    f'Remove column and append info to "CONTEXT"(r), undo filter/replace operation(u), ignore(i) > ',
                    'RUI')
                if __ == 'R':
                    self.CONTEXT.append(f'Deleted everything in {self.column_desc(0, col_idx)} except {xx[col_idx][0]}')
                    self.column_chop(0, col_idx)
                elif __ == 'U':
                    self.undo()
                # ignore --- JUST SKIP OUT, DO NOTHING


    def print_parameter_object(self, OBJECT_TO_PRINT, name):
        print()
        oi.obj_info(OBJECT_TO_PRINT, name, __name__)
        for obj_idx in self.OBJ_IDXS:
            print()
            print(f'{self.SXNL_DICT[obj_idx]}')
            print(f'COLUMN NAME = '.ljust(20) + f'{self.SUPER_RAW_NUMPY_LIST[obj_idx + 1]}')
            print(f'{name.upper()} VALUES = '.ljust(20) + f'{OBJECT_TO_PRINT[obj_idx]}')
        ioap.IdentifyObjectAndPrint(OBJECT_TO_PRINT, str(inspect.stack()[0][3]).upper(), __name__, 10, 10, start_row=0, start_col=0).run()


    def pending_cutoff(self):
        if self.pending_cutoffs == 'Y':
            print(f'\n\n\n****** THERE ARE PENDING MIN CUTOFF FILTERS TO BE APPLIED ******\n\n\n')


    def return_fxn(self):  # FOR PreRun --- OVERWRITTEN IN InSitu

        # BEAR 6/29/23 THIS COMES OUT WHEN CONVERTED OVER TO NEW SUPOBJS ###############################################

        SupObjClass = soc.SupObjConverter(
                                          DATA_HEADER=self.SUPER_RAW_NUMPY_LIST[1],
                                          TARGET_HEADER=self.SUPER_RAW_NUMPY_LIST[3],
                                          REFVECS_HEADER=self.SUPER_RAW_NUMPY_LIST[5],
                                          VALIDATED_DATATYPES=self.VALIDATED_DATATYPES,
                                          MODIFIED_DATATYPES=self.MODIFIED_DATATYPES,
                                          FILTERING=self.FILTERING,
                                          MIN_CUTOFFS=self.MIN_CUTOFFS,
                                          USE_OTHER=self.USE_OTHER,
                                          START_LAG=self.START_LAG,
                                          END_LAG=self.END_LAG,
                                          SCALING=self.SCALING
        )

        SUPOBJS = [SupObjClass.DATA_FULL_SUPOBJ, SupObjClass.TARGET_FULL_SUPOBJ, SupObjClass.REFVECS_FULL_SUPOBJ]
        del SupObjClass

        SXNL = [OBJ for idx, OBJ in enumerate(self.SUPER_RAW_NUMPY_LIST) if idx % 2 == 0]

        # END BEAR 6/29/23 THIS COMES OUT WHEN CONVERTED OVER TO NEW SUPOBJS ###############################################

        return SXNL, SUPOBJS, self.CONTEXT, self.KEEP, self.VALIDATED_DATATYPES_GLOBAL_BACKUP, self.MODIFIED_DATATYPES_GLOBAL_BACKUP


    def config_run(self):
        while True:

            self.initialize()

            if self.mode == 'PRERUN':
                # USER MUST ENTER DATATYPES ON FIRST ENTRY TO PreRunFilter (SUPER_NUMPY IS FULL HERE) SO TAKE SNAPSHOTS OF THESE
                # DTYPE OBJS AFTER initialize() AND MAKE THEM GLOBAL & SESSION BACKUPS, SO LATER IF A GLOBAL OR SESSION RESET
                # USER CAN HAVE OPTION NOT TO REDO THIS (THIS MAY BE A PAIN TO REDO IF DATA HAS DOZENS OR HUNDREDS OF COLUMNS)
                print(f'\nGenerating global and session backups for validated and modified datatypes...')
                self.VALIDATED_DATATYPES_GLOBAL_BACKUP = deepcopy(self.VALIDATED_DATATYPES)
                self.MODIFIED_DATATYPES_GLOBAL_BACKUP = deepcopy(self.MODIFIED_DATATYPES)
                self.VALIDATED_DATATYPES_SESSION_BACKUP = self.VALIDATED_DATATYPES_GLOBAL_BACKUP
                self.MODIFIED_DATATYPES_SESSION_BACKUP = self.MODIFIED_DATATYPES_GLOBAL_BACKUP
                print(f'Done.')
                # 12-12-21 MAYBE PUT FILTERING, MIN_CUTOFFS, USE OTHER --- SOMEHOW

            while True:
                # MAKE THIS FIRST TO APPLY UNDO B4 RESETTING UNDO OBJECTS BELOW
                if self.user_manual_or_std == 'U':  # 'undo last operation(u)'
                    self.undo()

                # SET THE INITIAL STATE OF EVERY OBJECT, FOR UNDO PURPOSES
                # DOING IT DEEPCOPY WAY TO PREVENT THE INTERACTION OF ASSIGNMENTS W ORIGINAL SOURCE OBJECTS
                # 12-12-21 10:49 AM ONLY RESET UNDO AFTER AN OPERATION, NOT PRINTS, ALLOWS USER TO LOOK AT OBJECTS
                # AND THEN DECIDE TO DO AN UNDO
                if True in map(lambda x: f'({self.user_manual_or_std})'.lower() in x,
                               self.menu_datastep_commands() + self.menu_filter_commands()):
                    self.SUPER_RAW_NUMPY_LIST_UNDO = [_.copy for _ in self.SUPER_RAW_NUMPY_LIST]  # TO AVOID DEEPCOPY
                    self.KEEP_UNDO = deepcopy(self.KEEP)
                    self.MODIFIED_DATATYPES_UNDO = deepcopy(self.MODIFIED_DATATYPES)
                    self.VALIDATED_DATATYPES_UNDO = deepcopy(self.VALIDATED_DATATYPES)
                    self.FILTERING_UNDO = deepcopy(self.FILTERING)
                    self.MIN_CUTOFFS_UNDO = deepcopy(self.MIN_CUTOFFS)
                    self.START_LAG_UNDO = deepcopy(self.START_LAG)
                    self.END_LAG_UNDO = deepcopy(self.END_LAG)
                    self.SCALING_UNDO = deepcopy(self.SCALING)
                    self.USE_OTHER_UNDO = deepcopy(self.USE_OTHER)
                    self.CONTEXT_UNDO = deepcopy(self.CONTEXT)
                    if self.user_manual_or_std in self.filter_str + self.datastep_str:  # ANYTIME DATASTEP OR FILTER CMD IS USED, RECORD
                        self.UNDO_DESC = [_ for _ in llm.list_of_lists_merger(self.FXN_LIST) if
                                              f"({self.user_manual_or_std.lower()})" in _][0]
                    else:
                        self.UNDO_DESC = f'\nUNDO NOT AVAILABLE\n'

                if self.user_manual_or_std == 'S':
                    self.SUPER_RAW_NUMPY_LIST, self.KEEP = self.standard_config_source()
                    if self.method != '':  # IF THERE WAS A STANDARD BUILD USED, SKIP OUT, IF NOT STAY IN LOOP
                        self.user_manual_or_std = ''

                if self.user_manual_or_std in 'BR':
                    # 'abandon session changes and exit filtering(b)'
                    # 'start over - session reset(r)'
                    # THIS SHOULDNT BE AVAILABLE FOR PreRun
                    while True:
                        if vui.validate_user_str(f'CAUTION!  A SESSION RESET WILL RESTORE EVERY OBJECT BACK TO THE ' + \
                                                 f'STATE IT WAS IN AT THE START OF THIS FILTERING SESSION OR TO A SESSION CHECKPOINT!  ' + \
                                                 f'PROCEED? (y/n) > ', 'YN') == 'N':
                            break
                        else:
                            self.SUPER_RAW_NUMPY_LIST = np.array([_.copy() for _ in self.SUPER_RAW_NUMPY_LIST_SESSION_BACKUP], dtype=object)
                            self.KEEP = deepcopy(self.KEEP_SESSION_BACKUP)
                            self.VALIDATED_DATATYPES = deepcopy(self.VALIDATED_DATATYPES_SESSION_BACKUP)
                            self.MODIFIED_DATATYPES = deepcopy(self.MODIFIED_DATATYPES_SESSION_BACKUP)
                            self.FILTERING = deepcopy(self.FILTERING_SESSION_BACKUP)
                            self.MIN_CUTOFFS = deepcopy(self.MIN_CUTOFFS_SESSION_BACKUP)
                            self.USE_OTHER = deepcopy(self.USE_OTHER_SESSION_BACKUP)
                            self.START_LAG = deepcopy(self.START_LAG_SESSION_BACKUP)
                            self.END_LAG = deepcopy(self.END_LAG_SESSION_BACKUP)
                            self.SCALING = deepcopy(self.SCALING_SESSION_BACKUP)
                            self.CONTEXT = deepcopy(self.CONTEXT_SESSION_BACKUP)

                            print(f'\nOBJECTS HAVE BEEN RESET TO THE INITIAL STATE OF THIS FILTERING SESSION\n')

                            if self.user_manual_or_std in 'B':
                                self.user_manual_or_std = 'A'
                            elif self.user_manual_or_std in 'R':
                                self.user_manual_or_std = 'BYPASS'
                            break

                if self.user_manual_or_std in 'C':  # 'print frequencies(c)'
                    obj_idx, col_idx = self.select_obj_and_col('OBJECT')
                    self.print_frequencies(obj_idx, col_idx)

                if self.user_manual_or_std in 'D':  # 'delete a column(d)'

                    obj_idx, col_idx = self.select_obj_and_col('OBJECT')

                    self.column_chop(obj_idx, col_idx)

                    print(f'\nDELETE COMPLETE\n')

                if self.user_manual_or_std in 'E':  # 'rename column(e)'
                    while True:
                        obj_idx, col_idx = self.select_obj_and_col('OBJECT')
                        self.rename_column(obj_idx, col_idx)
                        if vui.validate_user_str(f'\nrename other column(f) or exit(e) > ', 'EF') == 'E':
                            break


                if self.user_manual_or_std in 'F':  # 'filter feature by category(f)'
                    print(f'\nFILTER A FEATURE BY CATEGORY')
                    xx = self.SUPER_RAW_NUMPY_LIST

                    CAT_FILTER_CMDS = [
                        f'Accept and exit category filter(a)',
                        f'filter feature by category(s) (f)',
                        f'only keep selected category(s) of a feature(g)',
                        f'filter out a single row of a feature(h)',
                        f'reset(r)',
                        f'abandon and exit(x) '
                    ]

                    cat_filter_str = 'AFGHRX'
                    user_cat_filter = 'BYPASS'

                    while True:
                        delete_ctr = 0
                        if user_cat_filter in 'F':  # 'filter feature by category(s) (f)'
                            while True:
                                # self.print_cols_and_setup_parameters()

                                obj_idx, col_idx = self.num_or_str_col_select('STR')
                                if obj_idx == 'BREAK': break

                                # LET USER SEE UNIQUES IN COLUMN AND GIVE ESCAPE HATCH #####################################
                                UNIQUES = sorted(self.get_uniques(obj_idx, col_idx))

                                print(f'\nCATEGORIES IN {self.column_desc(obj_idx, col_idx)}:')
                                [print(_) for _ in UNIQUES]

                                __ = vui.validate_user_str(f'\nProceed with filter(y) abort(a) select another column(k)> ', 'YAK')
                                if __ == 'Y': pass
                                elif __ == 'A': obj_idx = 'BREAK'; break
                                elif __ == 'K': continue
                                # END LET USER SEE UNIQUES IN COLUMN AND GIVE ESCAPE HATCH ##################################

                                if vui.validate_user_str(f'\nReally really really proceed with filter? (y/n) > ', 'YN') == 'N':
                                    obj_idx = 'BREAK'
                                    break

                                print(f'\nSELECT CATEGORY(S) TO FILTER OUT')
                                TO_DELETE = ls.list_custom_select(UNIQUES,'value')

                                print(f'\nUSER SELECTED TO DELETE {", ".join([str(_) for _ in TO_DELETE])} FROM {self.column_desc(obj_idx, col_idx)}')

                                if vui.validate_user_str(f'\n ... ACCEPT? (y/n) > ', 'YN') == 'Y':
                                    self.FILTERING[obj_idx][col_idx] = [*self.FILTERING[obj_idx][col_idx], *[f'Filter out ' + str(_) for _ in TO_DELETE]]
                                    del UNIQUES
                                    break

                            if obj_idx == 'BREAK': break

                            DELETE_IDX_BOOLS = np.fromiter((_ in TO_DELETE for _ in xx[obj_idx][col_idx]), dtype=bool)

                            if self.USE_OTHER[obj_idx][col_idx] == 'Y':  # IF USING "OTHERS"
                                xx[obj_idx][col_idx] = \
                                    np.where(DELETE_IDX_BOOLS, 'OTHER', xx[obj_idx][col_idx] )
                            else:
                                for all_obj_idxs in self.OBJ_IDXS:
                                    xx[all_obj_idxs] = np.delete(xx[all_obj_idxs], DELETE_IDX_BOOLS, axis=1)

                            print(f'\nFILTERING OF {self.column_desc(obj_idx, col_idx)} COMPLETE... DELETED {np.sum(DELETE_IDX_BOOLS.astype(int))} ROWS.')

                            del TO_DELETE, DELETE_IDX_BOOLS


                        elif user_cat_filter == 'G':  # 'only keep selected categories of a feature(g)'
                            while True:
                                # self.print_cols_and_setup_parameters()

                                obj_idx, col_idx = self.num_or_str_col_select('STR')
                                if obj_idx == 'BREAK': break

                                print( f'\nSELECT CATEGORY(S) TO KEEP')
                                TO_KEEP = ls.list_custom_select(sorted(self.get_uniques(obj_idx, col_idx)), 'value')
                                # BEAR 3-12-22 FOR SOME REASON get_uniques() RETURNED AN EMPTY (CRASH BELOW), CANT FIGURE OUT Y

                                print(f'\nUSER SELECTED TO KEEP FROM {self.column_desc(obj_idx, col_idx)}')
                                [print(_) for _ in TO_KEEP]
                                if vui.validate_user_str(f'\n ... ACCEPT? (y/n) > ', 'YN') == 'Y':
                                    [self.FILTERING[obj_idx][col_idx].append(f'Keep {_}') for _ in TO_KEEP]
                                    break

                            if obj_idx == 'BREAK': break
                            for row_idx in range(len(xx[obj_idx][col_idx]) - 1, -1, -1):  # DELETE EVERYTHING ELSE EXCEPT "to_keep"
                                if xx[obj_idx][col_idx][row_idx] not in TO_KEEP:
                                    if self.USE_OTHER[obj_idx][col_idx] == 'N':
                                        delete_ctr += 1
                                        self.delete_rows(row_idx)
                                    else:  # IF USING "OTHERS"
                                        xx[obj_idx][col_idx][row_idx] = 'OTHER'

                        elif user_cat_filter == 'H':  # 'filter out a single row of a feature(h)'
                            while True:
                                # self.print_cols_and_setup_parameters()

                                obj_idx, col_idx = self.num_or_str_col_select('STR')
                                if obj_idx == 'BREAK': break

                                row_idx = vui.validate_user_int(
                                                        f'Enter row index to filter out for {self.column_desc(obj_idx, col_idx)} > ',
                                                        min=0, max=len(self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx]) - 1
                                            )
                                print(f'\nUSER ENTERED ROW INDEX {row_idx}')
                                print(
                                    f'VALUE BEING REPLACED IS {self.column_desc(obj_idx, col_idx)} - {xx[obj_idx][col_idx][row_idx]}')

                                if vui.validate_user_str(
                                        f'\nUSER SELECTED {self.column_desc(obj_idx, col_idx)}, ROW INDEX = {row_idx} ... ACCEPT? (y/n) > ',
                                        'YN') == 'Y':
                                    self.FILTERING[obj_idx][col_idx].append(
                                        f'Filter out row {row_idx} ({xx[obj_idx][col_idx][row_idx]})')
                                    break

                            if obj_idx == 'BREAK': break
                            if self.USE_OTHER[obj_idx][col_idx] == 'N':
                                delete_ctr += 1
                                self.delete_rows(row_idx)
                            else:  # IF USING "OTHERS"
                                xx[obj_idx][col_idx][row_idx] = 'OTHER'

                        elif user_cat_filter == 'R':  # 'reset(r)'
                            self.undo()

                        elif user_cat_filter == 'X':  # 'abandon and exit(x) '
                            self.undo()
                            break

                        elif user_cat_filter == 'A':  # 'Accept and exit(a)'
                            break

                        self.one_thing_left()
                        ppro.SelectionsPrint(CAT_FILTER_CMDS, cat_filter_str, append_ct_limit=3)
                        user_cat_filter = vui.validate_user_str(' > ', cat_filter_str)

                if self.user_manual_or_std in 'G':  # 'filter numbers(g)'
                    xx = self.SUPER_RAW_NUMPY_LIST
                    print(f'\nFILTER A NUMERICAL COLUMN')

                    while True:
                        while True:
                            obj_idx, col_idx = self.num_or_str_col_select('NUM')
                            if obj_idx == 'BREAK': break

                            self.print_statistics(obj_idx, col_idx)
                            print()

                            filter_method = \
                            ls.list_single_select(self.number_filter_options('', ''), 'Select filtering criteria',
                                                  'idx')[0]

                            greater_than, equal_to_1, equal_to_2, less_than = float('inf'), float('inf'), float(
                                'inf'), float('-inf')
                            # THESE HOLD INFO ON WHAT GETS CHOPPED, NOT WHAT GETS KEPT!

                            # GIVE INSTRUCTIONS FOR THE ROWS TO DELETE BASED ON USER INPUT
                            if filter_method == 0:  # 'Filter out only number equal to {num1}'
                                _ = self.number_filter_select(0)
                                equal_to_1 = _
                                text_str = self.number_filter_options(_, "")[0]

                            elif filter_method == 1:  # 'Keep only number equal to {num1}'
                                _ = self.number_filter_select(1)
                                greater_than, less_than = _, _
                                text_str = self.number_filter_options(_, "")[1]

                            elif filter_method == 2:  # f'Filter out numbers greater than {num1}'
                                _ = self.number_filter_select(2)
                                greater_than = _
                                text_str = self.number_filter_options(_, "")[2]

                            elif filter_method == 3:  # 'Filter out numbers greater than or equal to {num1}'
                                _ = self.number_filter_select(3)
                                greater_than, equal_to_1 = _, _
                                text_str = self.number_filter_options(_, "")[3]

                            elif filter_method == 4:  # 'Filter out numbers less than {num1}'
                                _ = self.number_filter_select(4)
                                less_than = _
                                text_str = self.number_filter_options(_, "")[4]

                            elif filter_method == 5:  # 'Filter out numbers less than or equal to {num1}'
                                _ = self.number_filter_select(5)
                                less_than, equal_to_1 = _, _
                                text_str = self.number_filter_options(_, "")[5]

                            elif filter_method == 6:  # 'Filter out numbers within range > {num1} and < {num2}'
                                while True:
                                    greater_than = self.number_filter_select(2)
                                    less_than = self.number_filter_select(4)
                                    if greater_than < less_than:
                                        break
                                    else:
                                        print(f'\n GREATER-THAN CUTOFF MUST BE LESS THAN LESS-THAN CUTOFF')
                                text_str = self.number_filter_options(less_than, greater_than)[6]

                            elif filter_method == 7:  # 'Filter out numbers outside of range, < {num1} and > {num2}'
                                while True:
                                    less_than = self.number_filter_select(4)
                                    greater_than = self.number_filter_select(2)
                                    if greater_than > less_than:
                                        break
                                    else:
                                        print(f'\n GREATER-THAN CUTOFF MUST BE GREATER THAN LESS-THAN CUTOFF')
                                text_str = self.number_filter_options(less_than, greater_than)[7]

                            elif filter_method == 8:  # 'Exit to main menu'
                                obj_idx = 'BREAK'
                                break

                            if vui.validate_user_str(
                                    f'\nUSER SELECTED: \n{self.column_desc(obj_idx, col_idx)}' + \
                                    f'\n{text_str} \nAccept? (y/n) > ',
                                    'YN') == 'N':
                                if vui.validate_user_str(f'\nTry again(t) or abandon without filtering(a)? > ',
                                                         'AT') == 'A':
                                    break
                                else:
                                    continue

                            # xx = self.SUPER_RAW_NUMPY_LIST
                            zz = xx[obj_idx][col_idx]

                            if filter_method == 6:  # 'Filter out numbers inside of range'
                                mask = ((zz > greater_than) * (zz < less_than)).astype(bool)
                                delete_ctr = np.sum(mask.astype(int))
                                for idx in self.OBJ_IDXS:
                                    self.SUPER_RAW_NUMPY_LIST[idx]= np.delete(self.SUPER_RAW_NUMPY_LIST[idx], mask, axis=1)
                                # DONT DO "OTHERS" STUFF HERE, NOT APPLICABLE TO NUMBERS

                            else:  # any other thing than method 6 (code that usually is run)
                                mask = ((zz == equal_to_1) + \
                                        (zz == equal_to_2) + \
                                        (zz < less_than) + \
                                        (zz > greater_than)).astype(bool)
                                delete_ctr = np.sum(mask.astype(int))
                                for idx in self.OBJ_IDXS:
                                    self.SUPER_RAW_NUMPY_LIST[idx] = np.delete(self.SUPER_RAW_NUMPY_LIST[idx], mask,
                                                                              axis=1)
                                # DONT DO "OTHERS" STUFF HERE, NOT APPLICABLE TO NUMBERS

                            self.FILTERING[obj_idx][col_idx].append(text_str)

                            del zz, mask

                            print(f'\nFILTERING COMPLETE... FILTERED {delete_ctr} ROWS.\n')
                            break

                        if obj_idx == 'BREAK': break

                        self.one_thing_left()

                        if vui.validate_user_str(f'Filter again(f) or exit to main menu(e) > ', 'EF') == 'E': break

                if self.user_manual_or_std in 'H':  # 'filter by date(h)'
                    # BEAR NEEDS WORK
                    # OLD FILTER JUNK #
                    pass
                    # while True:
                    #     start_date = vui.ValidateUserDate(
                    #         'Start date (inclusive)', format='MM-DD-YYYY', user_verify='Y').return_datetime()
                    #     start_month = start_date.month
                    #     start_day = start_date.day
                    #     start_year = start_date.year
                    #
                    #     end_date = vui.ValidateUserDate('End date (inclusive)', format='MM-DD-YYYY',
                    #                                     min=start_date, user_verify='Y').return_datetime()
                    #     end_month = end_date.month
                    #     end_day = end_date.day
                    #     end_year = end_date.year
                    #
                    #     if vui.validate_user_str('Accept date filter? (y/n) > ', 'YN') == 'Y':
                    #         break
                    #
                    # if user_date == 'Y':
                    #     print('Applying date filter.....')
                    #     for app_index in range(len(APPS_DF['APP DATE']) - 1, -1, -1):
                    #         app_day = int(str(APPS_DF['APP DATE'][app_index])[8:10]) + 0
                    #         app_month = int(str(APPS_DF['APP DATE'][app_index])[5:7]) + 0
                    #         app_year = int(str(APPS_DF['APP DATE'][app_index])[0:4]) + 0
                    #
                    #         if app_year < start_year or \
                    #                 (app_year == start_year and app_month < start_month) or \
                    #                 (app_year == start_year and app_month == start_month and app_day < start_day) or \
                    #                 app_year > end_year or \
                    #                 (app_year == end_year and app_month > end_month) or \
                    #                 (app_year == end_year and app_month == end_month and app_day > end_day):
                    #             APPS_DF = APPS_DF.drop(app_index)
                    # # END OLD FILTER JUNK #

                if self.user_manual_or_std == '5':  #  'filter value/string from selected columns(5)'
                    while True:
                        delete_ctr = 0
                        OBJECTS = ls.list_custom_select([self.SXNL_DICT[_] for _ in self.OBJ_IDXS], 'idx')
                        COLUMNS = []
                        for obj_idx in OBJECTS:
                            self.print_cols_and_setup_parameters()
                            COLUMN_IDXS = ls.list_custom_select(self.SUPER_RAW_NUMPY_LIST[obj_idx+1][0], 'idx')
                            COLUMNS.append(COLUMN_IDXS)

                        while True:
                            filter_on = input(f'Enter value to filter on > ')
                            if vui.validate_user_str(f'User entered < {filter_on} > ... Accept? (y/n) > ', 'YN') == 'Y':
                                break

                        print(f'\nUSER HAS SELECTED TO FILTER < {filter_on} > FROM THE FOLLOWING OBJECTS / COLUMNS:\n')
                        for obj_idx in OBJECTS:
                            print(f'{self.SXNL_DICT[obj_idx]}')
                            [print(self.SUPER_RAW_NUMPY_LIST[obj_idx+1][0][_]) for _ in COLUMNS[obj_idx]]
                            print()

                        __ = vui.validate_user_str(f'Accept? (y/n), Abort(a) > ', 'YNA')
                        if __ == 'N': continue
                        if __ == 'Y': pass
                        if __ == 'A': break

                        for obj_idx in OBJECTS:
                            for column_idx in COLUMNS[obj_idx]:
                                for row_idx in range(len(self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx])-1, -1, -1):
                                    if self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx][row_idx] == filter_on:
                                        print(f'{filter_on} FOUND IN {self.column_desc(obj_idx, column_idx)}')
                                        delete_ctr += 1
                                        self.delete_rows(row_idx)

                        print(f'\n*** FILTERING OF < {filter_on} > COMPLETE... DELETED {delete_ctr} ROWS. ***\n')
                        break


                if self.user_manual_or_std == 'i':  # 'reinitialize val. & mod. datatypes(i)'
                    self.validated_datatypes()
                    self.modified_datatypes()

                if self.user_manual_or_std in 'J':  # 'print statistics of numeric data(j)'
                    obj_idx, col_idx = self.select_obj_and_col('OBJECT')
                    xx, yy = self.SUPER_RAW_NUMPY_LIST, self.SXNL_DICT
                    print(f'\nPRINTING STATISTICS FOR --- {yy[obj_idx]}, {xx[obj_idx + 1][0][col_idx]}:')
                    self.print_statistics(obj_idx, col_idx)

                if self.user_manual_or_std in 'K':  # 'round/floor/ceiling a column of floats(k)'

                    xx = self.SUPER_RAW_NUMPY_LIST
                    while True:
                        obj_idx, col_idx = self.num_or_str_col_select('NUM')
                        if obj_idx == 'BREAK': break

                        rfc = vui.validate_user_str(f'Round(r), floor(f), ceiling(c), abort & exit(e)? > ', 'RFCE')
                        round_digits = 0
                        if rfc == 'E': break
                        elif rfc == 'R':
                            if self.MODIFIED_DATATYPES[obj_idx][col_idx] not in 'FLOAT':
                                print(
                                    f'MODIFIED DATATYPE FOR {self.column_desc(obj_idx, col_idx)} MUST BE SET TO FLOAT BEFORE ROUNDING CAN OCCUR.')
                                break
                            else:
                                round_digits = vui.validate_user_int(f'\nENTER THE NUMBER OF DECIMALS TO ROUND TO > ')

                        if round_digits <= 0: _dtype = int
                        else: _dtype = float

                        RFC_DICT = {'R': f'ROUND TO {round_digits} ', 'F': 'ROUND TO FLOOR', 'C': 'ROUND TO CEILING'}
                        if vui.validate_user_str(
                                f'\nUSER SELECTED: \n{self.column_desc(obj_idx, col_idx)} --- {RFC_DICT[rfc]} ' + \
                                f'... Accept? (y/n) > ', 'YN') == 'N':
                            if vui.validate_user_str(f'\nTry again(t) or abandon without rounding(a)? > ', 'AT') == 'A':
                                break
                            else:
                                continue

                        try:
                            if rfc == 'R':
                                self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx] = \
                                    np.round(self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx].astype('float'), round_digits).astype(_dtype)
                            elif rfc == 'F':
                                self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx] = \
                                    np.floor(self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx].astype('float')).astype(np.int32)
                            elif rfc == 'C':
                                self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx] = \
                                    np.ceil(self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx].astype('float')).astype(np.int32)
                            print(f'\nROUNDING COMPLETE!\n')
                        except:
                            print(
                                f'HAVING DIFFICULTY WHEN TRYING TO {RFC_DICT[rfc]} {self.column_desc(obj_idx, col_idx)}')

                        if vui.validate_user_str(f'Round another(f) or exit rounding(e) > ', 'EF') == 'E':
                            break

                if self.user_manual_or_std in 'L':  # 'duplicate a column(l)'
                    self.duplicate_column()  # IGNORE RETURNED OBJECTS

                if self.user_manual_or_std in 'M':  # 'apply min cutoff filter(m)' ###############################################
                    # (filter features to include only categories that appear at least x times)
                    # 12-14-21 DONT NEED ANY OBJ/COL ROW INPUTS HERE, THIS SIMPLY APPLYING INPUTS SET IN "MIN_CUTOFF" & "OTHER"
                    xx = self.SUPER_RAW_NUMPY_LIST

                    # 12-13-21 MAY WANT TO MOVE MIN_CUTOFF AND USE_OTHER SETUP INTO HERE INSTEAD OF SEPARATE CMDS

                    print(f'\nAPPLYING MIN CUTOFF FILTER....\n')

                    while True:
                        print(f'MIN_CUTOFFS WILL BE APPLIED AS FOLLOWS:')
                        self.print_cols_and_setup_parameters()
                        if vui.validate_user_str(f'Accept? (y/n) > ', 'YN') == 'N':
                            break

                        chop_pass = 0
                        while True:
                            # ITERATE THRU ALL OBJECTS (NOT HEADERS) OVER AND OVER UNTIL ALL COLUMNS MEET RESPECTIVE CUTOFF CRITERIA
                            chop_pass += 1
                            pass_row_chops = 0
                            pass_col_chops = 0
                            pass_other_conv = 0
                            did_chop_a_row = 'N'  # IF ANY PASS HAS NO CHOPS, ITS FINISHED & BREAK
                            for obj_idx in self.OBJ_IDXS:
                                for col_idx in range(len(xx[obj_idx]) - 1, -1,
                                                     -1):  # ITERATE THRU OBJ BACKWARDS CUZ COULD BE DELETING COLUMNS
                                    _ = self.MIN_CUTOFFS[obj_idx][col_idx]

                                    if _ == 0:
                                        continue  # IF NO CUTOFF, SKIP COLUMN
                                    else:
                                        # GET THE UNIQUES & COUNT OF ALL UNIQUES, PUT UNIQUES W/ CT >=CUTOFF IN A BUCKET
                                        NP_UNIQUES, NP_COUNTS = np.unique(xx[obj_idx][col_idx], return_counts=True)
                                        ct_of_other = 0
                                        FINAL_UNIQUES_TO_KEEP = []
                                        # print()
                                        for unique_idx in range(len(NP_UNIQUES)):
                                            if NP_COUNTS[unique_idx] >= _:
                                                FINAL_UNIQUES_TO_KEEP.append(NP_UNIQUES[unique_idx])
                                            else:  # IF "OTHER" IS IN NP_UNIQUES AND GOING TO BE OMITTED FOR ct, GET ct
                                                if NP_UNIQUES[unique_idx] == 'OTHER':
                                                    ct_of_other = NP_COUNTS[unique_idx]
                                                # print(f'GOING TO DELETE {self.column_desc(obj_idx, col_idx)}'
                                                #       f' - {NP_UNIQUES[unique_idx]}')
                                        # print()

                                        # IF "USE OTHER", NOTIFY IF "OTHER" WAS ACTUALLY ORIGINALLY IN THE COLUMN, IGNORE IF ct OF "OTHER" WAS ZERO
                                        if self.USE_OTHER[obj_idx][col_idx] == 'Y' and \
                                                "OTHER" not in FINAL_UNIQUES_TO_KEEP and \
                                                ct_of_other > 0:
                                            print(
                                                f'{self.column_desc(obj_idx, col_idx)} - "OTHER" ({ct_of_other}) DOES NOT MEET MINIMUM CUTOFF CRITERIA OF {_}.')
                                            if vui.validate_user_str(f'Keep anyway(k) or proceed with delete(d) > ',
                                                                     'KD') == 'K':
                                                FINAL_UNIQUES_TO_KEEP.append('OTHER')  # IF KEEP, PUT "OTHER" BACK INTO UNIQUES

                                        # IF FINAL_UNIQUES_TO_KEEP IS EMPTY AT THIS POINT, ALL ROWS WOULD BE CHOPPED, NOTIFY USER
                                        if len(FINAL_UNIQUES_TO_KEEP) == 0:
                                            print(f'\nAPPLYING CUTOFF OF {self.MIN_CUTOFFS[obj_idx][col_idx]} TO ' + \
                                                  f'{self.column_desc(obj_idx, col_idx)} WILL RESULT IN ALL ROWS BEING DELETED!')
                                            if vui.validate_user_str(
                                                    f'\nCHANGE CUTOFF(c), BYPASS CUTOFF FOR THIS COLUMN(b) > ', 'BC') == 'B':
                                                self.MIN_CUTOFFS[obj_idx][col_idx] = 0
                                                col_idx += 1  # STAY ON SAME COLUMN AND FILL FINAL_UNIQUES_TO_KEEP AGAIN BASED ON NO CUTOFF
                                            else:  # REMEMBER --- ITERATING BACKWARDS SO += 1
                                                self.MIN_CUTOFFS[obj_idx][col_idx] = vui.validate_user_int(f'\nENTER ' + \
                                                   f'NEW CUTOFF FOR {self.column_desc(obj_idx, col_idx)} > ', min=0)

                                                col_idx -= 1  # STAY ON SAME COLUMN AND FILL FINAL_UNIQUES_TO_KEEP AGAIN BASED ON NEW CUTOFF

                                        # IF UNIQUES HOLDER HAS ONLY 1 ENTRY, COLUMN WOULD BECOME MULTICOLINIEAR
                                        elif len(FINAL_UNIQUES_TO_KEEP) == 1:
                                            print(f'\nMIN CUTOFF CHOPPING LEAVES ONLY {FINAL_UNIQUES_TO_KEEP[0]} in ' + \
                                                  f'{self.column_desc(obj_idx, col_idx)}. THIS COLUMN WILL BE REMOVED AND NOTED ' + \
                                                  f'IN "CONTEXT".')
                                            input('HIT ENTER TO CONTINUE > ')
                                            self.CONTEXT.append(self.column_desc(obj_idx, col_idx) + ' - ' + FINAL_UNIQUES_TO_KEEP[0])
                                            self.column_chop(obj_idx, col_idx)
                                            pass_col_chops += 1

                                        else:  # AT THIS POINT, FINAL_UNIQUES_TO_KEEP IS HOLDING THINGS THAT STAY IN THE OBJECT'S COLUMN,
                                            # AND ANYTHING ELSE GETS DELETED OR TURNED TO "OTHER"

                                            if self.USE_OTHER[obj_idx][col_idx] == 'Y':  # ONLY APPLIES TO CURRENT COLUMN!
                                                TO_CHANGE = np.ones((1,len(xx[obj_idx][col_idx])), dtype=bool)[0]  # START OUT SAYING ALL GO TO "OTHER"
                                                for kept in FINAL_UNIQUES_TO_KEEP: # FOR EACH THING TO KEEP, CHANGE ALL MATCHES TO False (NOT TO DELETE)
                                                    TO_CHANGE = np.where(xx[obj_idx][col_idx]==kept, False, TO_CHANGE)
                                                del NP_UNIQUES, NP_COUNTS, FINAL_UNIQUES_TO_KEEP
                                                if np.sum(TO_CHANGE) == 0:
                                                    del TO_CHANGE
                                                    continue

                                                # SET REMAINING Trues TO "OTHER"
                                                xx[obj_idx][col_idx][TO_CHANGE] = 'OTHER'
                                                pass_other_conv += np.sum(TO_CHANGE.astype(int))
                                                del TO_CHANGE

                                            else:  # elif self.USE_OTHER[obj_idx][col_idx] == 'N', THEN DELETE.... APPLIES TO ALL ROWS IN ALL OBJECTS
                                                TO_DELETE = np.ones(len(xx[obj_idx][col_idx]), dtype=bool) # SET TO DELETE ALL TO START
                                                for kept in FINAL_UNIQUES_TO_KEEP: # FOR EACH THING TO KEEP, CHANGE ALL MATCHES TO False (NOT TO DELETE)
                                                    TO_DELETE = np.where(xx[obj_idx][col_idx]==kept, False, TO_DELETE)
                                                del NP_UNIQUES, NP_COUNTS, FINAL_UNIQUES_TO_KEEP
                                                if np.sum(TO_DELETE) == 0:
                                                    del TO_DELETE
                                                    continue

                                                # MUST APPLY DELETE TO ALL OBJECTS
                                                for del_obj_idx in self.OBJ_IDXS:
                                                    xx[del_obj_idx] = np.delete(xx[del_obj_idx], TO_DELETE, axis=1)
                                                pass_row_chops += np.sum(TO_DELETE.astype(int))
                                                did_chop_a_row = 'Y'  # IF ANY PASS HAS NO CHOPS, ITS FINISHED & BREAK
                                                del TO_DELETE
                            print(
                                f'MIN CUTOFF FILTER PASS {chop_pass} SUCCESSFULLY COMPLETED {pass_row_chops} ROW CHOPS, '
                                f'{pass_col_chops} COLUMN CHOPS, AND {pass_other_conv} "OTHER" CONVERSIONS!')
                            if did_chop_a_row == 'N':
                                break

                        self.pending_cutoffs = 'N'
                        self.CONTEXT.append(f'Applied min cutoff filter(s).')
                        print(f'\nMIN CUTOFF FILTERING COMPLETED SUCCESSFULLY.')
                        break
                # END MIN CUTOFF CHOPPING ##########################################################################################

                if self.user_manual_or_std == '7':  # 'apply min cutoff to SPLIT_STR columns(7)'
                    # COLUMNS THAT ARE SPLIT_STR MUST BE VTYPE 'STR' AND MTYPE 'INT'
                    # (STR/BIN WOULD BE FROM "STR" AND STR/FLOAT WOULD BE FROM "NNLM50")
                    # SPLIT_STR COULD NEVER APPLY TO TARGET (COULD APPLY TO REFVECS BUT REFVECS DONT GET EXPANDED)

                    MASK, CONTEXT_UPDATE, self.MIN_CUTOFFS[0] = sscf.split_str_cutoff_filter(self.SUPER_RAW_NUMPY_LIST[0],
                       self.data_given_orientation, self.SUPER_RAW_NUMPY_LIST[1], self.VALIDATED_DATATYPES[0],
                       self.MODIFIED_DATATYPES[0], self.MIN_CUTOFFS[0])

                    self.SUPER_RAW_NUMPY_LIST[0] = np.delete(self.SUPER_RAW_NUMPY_LIST[0], MASK, axis=0)
                    self.SUPER_RAW_NUMPY_LIST[1] = np.delete(self.SUPER_RAW_NUMPY_LIST[1], MASK, axis=1)
                    self.VALIDATED_DATATYPES[0] = np.delete(self.VALIDATED_DATATYPES[0], MASK, axis=0)
                    self.MODIFIED_DATATYPES[0] = np.delete(self.MODIFIED_DATATYPES[0], MASK, axis=0)
                    self.FILTERING[0] = np.delete(self.FILTERING[0], MASK, axis=0)
                    self.MIN_CUTOFFS[0] = np.delete(self.MIN_CUTOFFS[0], MASK, axis=0)
                    self.USE_OTHER[0] = np.delete(self.USE_OTHER[0], MASK, axis=0)
                    self.START_LAG[0] = np.delete(self.START_LAG[0], MASK, axis=0)
                    self.END_LAG[0] = np.delete(self.END_LAG[0], MASK, axis=0)
                    self.SCALING[0] = np.delete(self.SCALING[0], MASK, axis=0)



                    self.CONTEXT = self.CONTEXT + CONTEXT_UPDATE

                    del MASK, CONTEXT_UPDATE

                    print(f'*** SUCCESSFULLY COMPLETED MIN CUTOFF CHOPS OF SPLIT_STR COLUMNS ***\n')


                if self.user_manual_or_std in 'N':  # 'override validated datatype(n)'
                    self.modified_validated_override_template('VALIDATED')

                if self.user_manual_or_std in 'O':  # 'set modified datatype(o)'
                    self.modified_validated_override_template('MODIFIED')

                if self.user_manual_or_std in 'P':  # 'print a preview of objects as DFs(p)'
                    print()
                    self.print_preview_all_objects(
                                            start_row=vui.validate_user_int(f'Enter start row of preview (zero-indexed) > ',
                                                                            min=0,
                                                                            max=len(self.SUPER_RAW_NUMPY_LIST[0][0])-1
                                                                            ),
                                            start_col=vui.validate_user_int(f'Enter start column of preview (zero-indexed) > ',
                                                                            min=0,
                                                                            max=len(self.SUPER_RAW_NUMPY_LIST[0][0])-1
                                                                            )
                    )

                if self.user_manual_or_std in 'Q':  # 'print column names & setup parameters(q)'
                    self.print_cols_and_setup_parameters()

                if self.user_manual_or_std in 'T':  # 'start over - global reset(t)'
                    if vui.validate_user_str(
                            f'\nCAUTION!  A GLOBAL RESET WILL RESTORE EVERY OBJECT BACK TO THE STATE ' + \
                            f'IT WAS IN AFTER INITIAL CREATION!  USER MAY CHOOSE TO RE-ENTER THE DATATYPES FOR ALL ' + \
                            f'NUMERICAL COLUMNS.  PROCEED? (y/n) > ', 'YN') == 'Y':
                        self.SUPER_RAW_NUMPY_LIST = np.array([_.copy() for _ in self.SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP], dtype=object)
                        self.KEEP = deepcopy(self.KEEP_GLOBAL_BACKUP)
                        self.CONTEXT = deepcopy(self.CONTEXT_GLOBAL_BACKUP)
                        self.FILTERING = [[[] for _ in __] for __ in self.SUPER_RAW_NUMPY_LIST]
                        self.MIN_CUTOFFS = [[0 for _ in __] for __ in self.SUPER_RAW_NUMPY_LIST]
                        self.USE_OTHER = [['N' for _ in __] for __ in self.SUPER_RAW_NUMPY_LIST]
                        self.START_LAG = [[0 for _ in __] for __ in self.SUPER_RAW_NUMPY_LIST]
                        self.END_LAG = [[0 for _ in __] for __ in self.SUPER_RAW_NUMPY_LIST]
                        self.SCALING = [['' for _ in __] for __ in self.SUPER_RAW_NUMPY_LIST]

                        if vui.validate_user_str(f'\nRE-ENTER USER-ENTERED DATA TYPES? (y/n) > ', 'YN') == 'Y':
                            # WHEN GLOBAL RESET FOR BOTH PreRun & InSitu, DATATYPES OBJS len MUST = ALL lens IN SUPER_NUMPY
                            # (SOME COLUMNS MAY HAVE BEEN DELETED DURING PreRun & InSitu OPERATIONS) SO REGENERATE

                            self.VALIDATED_DATATYPES = [['' for _ in __] for __ in self.SUPER_RAW_NUMPY_LIST]
                            self.MODIFIED_DATATYPES = [['' for _ in __] for __ in self.SUPER_RAW_NUMPY_LIST]

                            self.validated_datatypes()
                            self.modified_datatypes()
                            # THIS FORCES USER TO ENTER NUM DATATYPES
                        else:
                            self.VALIDATED_DATATYPES = deepcopy(self.VALIDATED_DATATYPES_GLOBAL_BACKUP)
                            self.MODIFIED_DATATYPES = deepcopy(self.MODIFIED_DATATYPES_GLOBAL_BACKUP)

                        print(f'\nOBJECTS HAVE BEEN RESET TO THE STATE AT INITIAL CREATION BEFORE ANY FILTERING\n')

                if self.user_manual_or_std in 'V':  # 'replace a value(v)'
                    aa, bb = self.VALIDATED_DATATYPES, self.MODIFIED_DATATYPES
                    xx= self.SUPER_RAW_NUMPY_LIST

                    while True:
                        obj_idx, col_idx = self.select_obj_and_col('OBJECT')

                        print(f'\n{self.column_desc(obj_idx, col_idx)} > ')
                        print(f'VALIDATED DATATYPE IS {aa[obj_idx][col_idx]}')
                        print(f'MODIFIED DATATYPE IS {bb[obj_idx][col_idx]}\n')

                        escape_text = 'Abort & exit'

                        if bb[obj_idx][col_idx] == 'STR':

                            str_replace_cmds = ['Replace all instances of a value', 'Replace one entry by row index']

                            self.print_frequencies(obj_idx, col_idx)
                            _ = str_replace_cmds + [escape_text]
                            print()
                            user_str_cmd = ls.list_single_select(_, f'Select command', 'idx')[0]

                            if user_str_cmd == len(_) - 1:
                                break

                            elif user_str_cmd == 0:  # 'Replace all instances of a value(i)'

                                old_value = ls.list_single_select(sorted(self.get_uniques(obj_idx, col_idx)) + \
                                                                  [escape_text], f'SELECT VALUE TO REPLACE', 'value')[0]
                                if old_value == escape_text: break

                                while True:
                                    new_value = input(f'ENTER NEW VALUE (OLD VALUE IS {old_value}) > ')
                                    if vui.validate_user_str(f'\nUSER ENTERED {new_value}.  Accept? (y/n) > ',
                                                             'YN') == 'Y': break

                                if vui.validate_user_str(
                                        f'USER ENTERED "{new_value}" TO REPLACE ALL INSTANCES OF "{old_value}" IN ' + \
                                        f'{self.column_desc(obj_idx, col_idx)}--- ACCEPT? (y/n) > ', 'YN') == 'N':
                                    continue

                                self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx] = np.where(
                                    self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx] == old_value,
                                    new_value,
                                    self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx]
                                )

                                self.FILTERING[obj_idx][col_idx].append(
                                                    f'Replace all instances of {old_value} with {new_value}')
                                print(f'\nSUBSTITUTION COMPLETE.')

                            elif user_str_cmd == 1:  # 'Replace one entry by row index(r)'
                                row_idx = vui.validate_user_int(
                                    f'Enter row index to replace for {self.column_desc(obj_idx, col_idx)} > ',
                                    min=0, max=len(xx[obj_idx][col_idx]) - 1)
                                print(f'\nUSER ENTERED ROW INDEX {row_idx}')
                                print(
                                    f'VALUE BEING REPLACED IS {self.column_desc(obj_idx, col_idx)} - {xx[obj_idx][col_idx][row_idx]}')
                                if vui.validate_user_str(f'PROCEED? (y/n) > ', 'YN') == 'N':
                                    break
                                else:
                                    __ = xx[obj_idx][col_idx][row_idx]
                                    while True:
                                        new_value = input(
                                            f'ENTER NEW VALUE (OLD VALUE IS {__}) > ')
                                        if vui.validate_user_str(f'USER ENTERED {new_value}.  Accept? (y/n) > ',
                                                                 'YN') == 'Y': break

                                    if vui.validate_user_str(
                                            f'USER ENTERED {new_value} TO REPLACE {__} IN ROW {row_idx} IN ' + \
                                            f'{self.column_desc(obj_idx, col_idx)}--- ACCEPT? (y/n) > ', 'YN') == 'N':
                                        continue

                                    self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx][
                                        row_idx] = new_value  # INPUT IS ALWAYS "STR"
                                    self.FILTERING[obj_idx][col_idx].append(
                                        f'Replace row index {row_idx} ({xx[obj_idx][col_idx][row_idx]}) with {new_value}')
                                    print(f'\nSUBSTITUTION COMPLETE.')

                                    break

                        elif bb[obj_idx][col_idx] in ['FLOAT', 'INT', 'BIN']:

                            print(f'\nREPLACE A VALUE IN A NUMERICAL COLUMN\n')

                            while True:
                                # SHOW STATISTICS FOR SELECTED COLUMN
                                self.print_statistics(obj_idx, col_idx)
                                if obj_idx == 'BREAK': break
                                print()

                                if vui.validate_user_str(f'Print frequencies? (y/n) > ', 'YN') == 'Y':
                                    self.print_frequencies(obj_idx, col_idx)

                                replace_method = ls.list_single_select(self.number_replace_options('', ''),
                                                                       'Select replacement criteria', 'idx')[0]

                                greater_than, equal_to_1, equal_to_2, less_than = float('inf'), float('inf'), float(
                                    'inf'), float('-inf')
                                # HOLDS INFO ON WHAT VALUES TO REPLACE!  NOT WHAT VALUES TO IGNORE!

                                # GIVE INSTRUCTIONS FOR THE VALUES TO REPLACE BASED ON USER INPUT
                                if replace_method == 0:  # 'Replace one entry based on row index'
                                    row_idx = vui.validate_user_int(f'Select row index (zero-based index) > ', min=0,
                                                                    max=len(xx[obj_idx][0])-1)
                                    num_old_value = xx[obj_idx][col_idx][row_idx]
                                    text_str = \
                                    self.number_replace_options(f'ROW = {row_idx}, VALUE = {num_old_value}', "")[0]

                                elif replace_method == 1:  # 'Replace all instances of number equal to'
                                    _ = self.number_replace_select(1)
                                    equal_to_1 = _
                                    text_str = self.number_replace_options(_, "")[1]

                                elif replace_method == 2:  # 'Replace numbers greater than'
                                    _ = self.number_replace_select(2)
                                    greater_than = _
                                    text_str = self.number_replace_options(_, "")[2]

                                elif replace_method == 3:  # 'Replace numbers greater than or equal to'
                                    _ = self.number_replace_select(3)
                                    greater_than, equal_to_1 = _, _

                                    text_str = self.number_replace_options(_, "")[3]

                                elif replace_method == 4:  # 'Replace numbers less than'
                                    _ = self.number_replace_select(4)
                                    less_than = _
                                    text_str = self.number_replace_options(_, "")[4]

                                elif replace_method == 5:  # 'Replace numbers less than or equal to'
                                    _ = self.number_replace_select(5)
                                    less_than, equal_to_1 = _, _
                                    text_str = self.number_replace_options(_, "")[5]

                                elif replace_method == 6:  # 'Replace numbers within range > and <'
                                    while True:
                                        greater_than = self.number_replace_select(2)
                                        less_than = self.number_replace_select(4)
                                        if greater_than < less_than:
                                            break
                                        else:
                                            print(f'\n GREATER-THAN CUTOFF MUST BE LESS THAN LESS-THAN CUTOFF')
                                    text_str = self.number_replace_options(greater_than, less_than)[6]

                                elif replace_method == 7:  # 'Replace numbers outside of range, < and >'
                                    while True:
                                        less_than = self.number_replace_select(4)
                                        greater_than = self.number_replace_select(2)
                                        if greater_than > less_than:
                                            break
                                        else:
                                            print(f'\n GREATER-THAN CUTOFF MUST BE GREATER THAN LESS-THAN CUTOFF')
                                    text_str = self.number_replace_options(less_than, greater_than)[7]

                                elif replace_method == 8:   # 'Exit to main menu'
                                    obj_idx = 'BREAK'
                                    break

                                if vui.validate_user_str(f'\nUSER SELECTED: \n{self.column_desc(obj_idx, col_idx)}' + \
                                        f'\n{text_str} \nAccept? (y/n) > ', 'YN') == 'N':
                                    if vui.validate_user_str(f'\nTry again(t) or abandon without filtering(a)? > ',
                                                             'AT') == 'A':
                                        break
                                    else:
                                        continue

                                while True:
                                    new_value = float(input(f'ENTER NEW VALUE > '))
                                    # GET VALIDATED DATATYPE FOR USER ENTRY
                                    nvvt = vcsdt.ValidateCharSeqDataType(new_value).type()[0].upper()  # new_value_validated_type
                                    if nvvt not in ['INT', 'FLOAT', 'BIN']:
                                        print(f'\nMUST ENTER A NUMBER\n')
                                        continue
                                    if vui.validate_user_str(f'\nUSER ENTERED {new_value}.  Accept? (y/n) > ',
                                                             'YN') == 'Y':
                                        break

                                d_type = self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx].dtype
                                if vui.validate_user_str(
                                        f'USER ENTERED --- {new_value} AS {d_type} TO {text_str.upper()} IN ' + \
                                        f'{self.column_desc(obj_idx, col_idx)}--- ACCEPT? (y/n) > ', 'YN') == 'N':
                                    break

                                if replace_method == 0:  # 'Replace one entry based on row index {num1}'
                                    self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx][row_idx] = new_value
                                    print(f'\nREPLACE COMPLETE.\n')  # JUST SO DONT GO THRU for LOOP BELOW
                                    break

                                for _ in range(len(xx[obj_idx][col_idx])):
                                    zz = xx[obj_idx][col_idx][_]
                                    if replace_method == 6:  # f'Replace numbers within range > {num1} and < {num2}'
                                        if zz > greater_than and zz < less_than:
                                            self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx][_] = new_value

                                    else:  # any other thing than method 6
                                        if zz == equal_to_1 or \
                                                zz == equal_to_2 or \
                                                zz < less_than or \
                                                zz > greater_than:
                                            self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx][_] = new_value

                                self.FILTERING[obj_idx][col_idx].append(f'{text_str} with {new_value}')

                                print(f'\nREPLACE COMPLETE.\n')

                                break

                        if obj_idx == 'BREAK': break

                        self.one_thing_left()

                        if vui.validate_user_str(f'Replace another value (f) or exit to main menu(e) > ', 'EF') == 'E': break

                if self.user_manual_or_std in 'W':  # 'start_over_base_reset(w)'
                    # should not be available for PreRun
                    # resets InSitu objs to state after PreRun filtering

                    if vui.validate_user_str(
                            f'CAUTION!  A BASE RESET WILL RESTORE EVERY OBJECT BACK TO THE STATE ' + \
                            f'IT WAS IN AFTER INITIAL FILTERING!  PROCEED? (y/n) > ', 'YN') == 'Y':
                        self.SUPER_RAW_NUMPY_LIST = np.array([_.copy() for _ in self.SUPER_RAW_NUMPY_LIST_BASE_BACKUP], dtype=object)
                        self.MODIFIED_DATATYPES = deepcopy(self.MODIFIED_DATATYPES_BASE_BACKUP)
                        self.VALIDATED_DATATYPES = deepcopy(self.VALIDATED_DATATYPES_BASE_BACKUP)
                        self.FILTERING = deepcopy(self.FILTERING_BASE_BACKUP)
                        self.MIN_CUTOFFS = deepcopy(self.MIN_CUTOFFS_BASE_BACKUP)
                        self.USE_OTHER = deepcopy(self.USE_OTHER_BASE_BACKUP)
                        self.START_LAG = deepcopy(self.START_LAG_BASE_BACKUP)
                        self.END_LAG = deepcopy(self.END_LAG_BASE_BACKUP)
                        self.SCALING = deepcopy(self.SCALING_BASE_BACKUP)
                        self.KEEP = deepcopy(self.KEEP_BASE_BACKUP)
                        self.CONTEXT = deepcopy(self.CONTEXT_BASE_BACKUP)

                        print(f'\nOBJECTS HAVE BEEN RESET TO THE INITIAL STATE OF THIS FILTERING SESSION\n')

                if self.user_manual_or_std in 'X':  # 'save current state as session checkpoint(x)'
                    self.SUPER_RAW_NUMPY_LIST_SESSION_BACKUP = np.array([_.copy() for _ in self.SUPER_RAW_NUMPY_LIST], dtype=object)
                    self.VALIDATED_DATATYPES_SESSION_BACKUP = deepcopy(self.VALIDATED_DATATYPES)
                    self.MODIFIED_DATATYPES_SESSION_BACKUP = deepcopy(self.MODIFIED_DATATYPES)
                    self.FILTERING_SESSION_BACKUP = deepcopy(self.FILTERING)
                    self.MIN_CUTOFFS_SESSION_BACKUP = deepcopy(self.MIN_CUTOFFS)
                    self.USE_OTHER_SESSION_BACKUP = deepcopy(self.USE_OTHER)
                    self.START_LAG_SESSION_BACKUP = deepcopy(self.START_LAG)
                    self.END_LAG_SESSION_BACKUP = deepcopy(self.END_LAG)
                    self.SCALING_SESSION_BACKUP = deepcopy(self.SCALING)
                    self.KEEP_SESSION_BACKUP = deepcopy(self.KEEP)
                    self.CONTEXT_SESSION_BACKUP = deepcopy(self.CONTEXT)

                    if self.mode == 'PRERUN' and 'R' not in self.datastep_str:
                        self.datastep_str += 'R'  # IF IN PreRun & USER DOES A CHECKPOINT, THEN ALLOW SESSION RESTORE
                        self.allowed_commands_string += 'R'

                    print(f'\nCURRENT STATE SAVED AS SESSION CHECKPOINT.')

                if self.user_manual_or_std in 'Y':  # 'sort(y)'
                    while True:
                        obj_idx, col_idx = self.select_obj_and_col('OBJECT')
                        print()
                        sort_type = ls.list_single_select(['ASCENDING', 'DESCENDING'], 'Select sort order', 'value')[0]

                        sort_option = vui.validate_user_str(
                            f'\nUser entered {self.column_desc(obj_idx, col_idx)}, {sort_type} ' + \
                            f'... Accept? (y/n), Exit to main menu(e) > ', 'YNE')

                        if sort_option == 'Y':
                            pass
                        elif sort_option == 'N':
                            continue
                        elif sort_option == 'E':
                            break

                        ARG_SORT_LIST_ASC = np.argsort(self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx])

                        for obj_idx2 in self.OBJ_IDXS:
                            for col_idx2 in range(len(self.SUPER_RAW_NUMPY_LIST[obj_idx2])):
                                # 12-30-21 MAKE A COPY OF COLUMN, TO PULL FROM WHILE OVERWRITING THE ACTUAL COLUMN
                                _ = self.SUPER_RAW_NUMPY_LIST[obj_idx2][col_idx2].copy()

                                for row_idx2 in range(len(_)):
                                    if sort_type == 'ASCENDING':
                                        self.SUPER_RAW_NUMPY_LIST[obj_idx2][col_idx2][row_idx2] = _[
                                            ARG_SORT_LIST_ASC[row_idx2]]
                                    elif sort_type == 'DESCENDING':
                                        self.SUPER_RAW_NUMPY_LIST[obj_idx2][col_idx2][row_idx2] = _[
                                            ARG_SORT_LIST_ASC[-1 - row_idx2]]

                        print(f'\nSORT COMPLETE.\n')

                        break

                if self.user_manual_or_std in 'Z':  # 'delete a row(z)'
                    _row = vui.validate_user_int(f'\nEnter index of row to delete (zero-indexed) > ',
                                                  min=0,
                                                  max=len(self.SUPER_RAW_NUMPY_LIST[0][0]) - 1
                    )
                    self.delete_rows(_row)
                    del _row

                if self.user_manual_or_std in '0':  # 'set min cutoff(0)',
                    # FIND OUT IF EACH OBJECT IS SET THE SAME OR UNIQUELY VALUED --- ALL ZEROES AT START, INDICATES IF THE
                    # USER HAS ENTERED SOMETHING
                    xx, yy = self.SUPER_RAW_NUMPY_LIST, self.SXNL_DICT

                    # BEAR 4/26/23 THIS KEEPS BLOWING UP FOR "CANT CONVERT STR TO FLOAT" FOR '' IN return_uniques()
                    # SINCE THIS IS JUST FOR DISPLAY, TAKING IT OUT
                    # if len(ru.return_uniques(llm.list_of_lists_merger(self.MIN_CUTOFFS), [], 'INT', suppress_print='Y')[0]) == 1:
                    #     print(f'\nALL MIN CUTOFFS ARE CURRENTLY SET TO {self.MIN_CUTOFFS[0][0]}.')
                    # else: print(f'\nMIN CUTOFFS ARE UNIQUELY VALUED.')

                    CUTOFF_CMDS = ['accept changes and exit(a)',
                                   'set all cutoffs for all objects the same(b)',
                                   'set all cutoffs for one object the same(c)',
                                   'set cutoff for a specific column(d)',
                                   'reset(e)',
                                   'abort and exit without saving changes(f)',
                                   'enter cutoffs based on "OTHER" entries(g)',
                                   'print column names & setup parameters(q)'
                                   ]

                    cutoff_cmds_str = 'ABCDEFQ'
                    # IF USER HAS ALREADY ENTERED "USE OTHERS", ALLOW USER TO GO THRU AND ENTER CUTOFFS FOR JUST THOSE
                    if True in map(lambda x: 'Y' in x, self.USE_OTHER):
                        cutoff_cmds_str = 'ABCDEFGQ'

                    user_min_cutoff = 'BYPASS'  # DONT DELETE THIS HAVE TO BYPASS ALL TO GET TO PROMPT ON FIRST PASS

                    obj_idx = 0 # 1-2-22 A DUMMY TO MAKE IT THRU THE if obj_idx == 'BREAK' PART OF THE LOOP AT START
                    while True:
                        # 12-12-21 THE ONLY POSNS THAT ACTUALLY MATTER ARE THOSE WHERE MODIFIED TYPE IS 'STR', OTHER POSNS IN
                        # MIN_CUTOFF CAN BE SET TO ANYTHING AND WILL NEVER BE READ, MIN_CUTOFF APPLIER FXN WILL ONLY
                        # LOOK AT POSNS WHERE MODIFIED TYPE IS 'STR'

                        if user_min_cutoff in 'B':  # 'set all cutoffs for all objects the same(b)',

                            while True:
                                min_cutoff = vui.validate_user_int(f'\nEnter min cutoff for all columns > ', min=0)
                                if vui.validate_user_str(f'\nUser entered {min_cutoff}, accept? (y/n) > ', 'YN') == 'Y':
                                    for obj_idx in self.OBJ_IDXS:
                                        for col_idx in range(len(xx[obj_idx])):
                                            if self.MODIFIED_DATATYPES[obj_idx][col_idx] in ['STR', 'BIN']:
                                                self.MIN_CUTOFFS[obj_idx][col_idx] = min_cutoff
                                            else:
                                                self.MIN_CUTOFFS[obj_idx][col_idx] = 0
                                    break

                        elif user_min_cutoff in 'C':  # 'set all cutoffs for one object the same(c)',

                            obj_idx = self.num_or_str_obj_select('STR')
                            if obj_idx == 'BREAK': break

                            while True:
                                min_cutoff = vui.validate_user_int(
                                    f'\nEnter min cutoff for all columns in {yy[obj_idx]} > ', min=0)
                                if vui.validate_user_str(
                                        f'\nUser entered {min_cutoff} as cutoff for all, accept? (y/n) > ',
                                        'YN') == 'Y':
                                    for col_idx in range(len(xx[obj_idx])):
                                        if self.MODIFIED_DATATYPES[obj_idx][col_idx] == 'STR':
                                            self.MIN_CUTOFFS[obj_idx][col_idx] = min_cutoff
                                        else:
                                            self.MIN_CUTOFFS[obj_idx][col_idx] = 0
                                    break

                        elif user_min_cutoff in 'D':  # 'set cutoff for a specific column(d)',

                            while True:
                                obj_idx, col_idx = self.num_or_str_col_select('STR')
                                if obj_idx == 'BREAK': break

                                min_cutoff = vui.validate_user_int(
                                    f'\nEnter min cutoff for {self.column_desc(obj_idx, col_idx)} > ', min=0)
                                if vui.validate_user_str(
                                        f'User entered {min_cutoff} as min cutoff for {self.column_desc(obj_idx, col_idx)}, ' + \
                                        f'accept? (y/n) > ', 'YN') == 'Y':
                                    self.MIN_CUTOFFS[obj_idx][col_idx] = min_cutoff
                                    break

                        elif user_min_cutoff in 'EF':  # 'reset(e)',   'abort and exit without saving changes(f)'
                            self.MIN_CUTOFFS = deepcopy(self.MIN_CUTOFFS_UNDO)
                            print(f'\nMIN CUTOFFS HAVE BEEN REVERTED BACK.')
                            if user_min_cutoff in 'F':
                                break

                        elif user_min_cutoff in 'G':  # 'enter cutoffs based on "OTHER" entries(g)'
                            for obj_idx2 in self.OBJ_IDXS:
                                for yn_idx in range(len(self.USE_OTHER[obj_idx2])):
                                    if self.USE_OTHER[obj_idx2][yn_idx] == 'Y':
                                        self.MIN_CUTOFFS[obj_idx2][yn_idx] = vui.validate_user_int(
                                            f'Enter min cutoff for {self.column_desc(obj_idx2, yn_idx)} > ', min=0)

                        elif user_min_cutoff in 'Q':  # 'print column names & setup parameters(q)'
                            self.print_cols_and_setup_parameters()

                        elif user_min_cutoff in 'A':
                            if self.MIN_CUTOFFS==self.MIN_CUTOFFS_UNDO:
                                self.pending_cutoffs = 'Y'
                            break

                        if obj_idx == 'BREAK': break

                        ppro.SelectionsPrint(CUTOFF_CMDS, cutoff_cmds_str, append_ct_limit=3)

                        user_min_cutoff = vui.validate_user_str(' > ', cutoff_cmds_str)

                if self.user_manual_or_std in '1':  # 'set "OTHERS"(1)'
                    # FIND OUT IF EACH OBJECT IS SET THE SAME OR UNIQUELY VALUED --- ALL 'N' AT START, INDICATES IF THE
                    #                     # USER HAS ENTERED SOMETHING
                    xx, yy = self.SUPER_RAW_NUMPY_LIST, self.SXNL_DICT
                    if len(ru.return_uniques(llm.list_of_lists_merger(self.USE_OTHER), [], 'STR', suppress_print='Y')[
                               0]) == 1:
                        print(f'\nALL "OTHERS" ARE CURRENTLY SET TO "{self.USE_OTHER[0][0]}".')
                    else:
                        print(f'\n"OTHERS" ARE UNIQUELY VALUED.')

                    USE_OTHER_CMDS = ['accept changes and exit(a)',
                                      'set all "OTHERS" for all objects the same(b)',
                                      'set all "OTHERS" for one object the same(c)',
                                      'set "OTHERS" for a specific column(d)',
                                      'reset(e)',
                                      'abort and exit without saving changes(f)',
                                      'enter "OTHERS" based on min cutoff entries(g)',
                                      'print column names & setup parameters(q)'
                                      ]
                    use_other_cmds_str = 'ABCDEFQ'

                    # IF USER HAS ALREADY ENTERED "MIN_CUTOFFS" ALLOW USER TO GO THRU AND ENTER CUTOFFS FOR JUST THOSE
                    if np.sum([np.sum(_) for _ in self.MIN_CUTOFFS]) > 0:
                        use_other_cmds_str = 'ABCDEFGQ'

                    user_use_other = 'BYPASS'  # DONT DELETE THIS, HAVE TO RUN THRU TO CMD PROMPT ON FIRST PASS

                    while True:
                        # 12-13-21 THE ONLY POSNS THAT ACTUALLY MATTER ARE THOSE WHERE MODIFIED TYPE IS 'STR', OTHER POSNS IN
                        # "OTHERS" CAN BE SET TO ANYTHING AND WILL NEVER BE READ, "OTHERS" APPLIER FXN WILL ONLY
                        # LOOK AT POSNS WHERE MODIFIED TYPE IS 'STR'

                        if user_use_other in 'B':  # 'set all "OTHERS" for all objects the same(b)',

                            while True:
                                use_other = vui.validate_user_str(f'Use "OTHER" columns for all columns? (y/n) > ',
                                                                  'YN')
                                if vui.validate_user_str(f'User entered "{use_other}", accept? (y/n) > ', 'YN') == 'Y':
                                    for obj_idx in self.OBJ_IDXS:
                                        for col_idx in range(len(xx[obj_idx])):
                                            if self.MODIFIED_DATATYPES[obj_idx][col_idx] == 'STR':
                                                self.USE_OTHER[obj_idx][col_idx] = use_other
                                            else:
                                                self.USE_OTHER[obj_idx][col_idx] = 'N'
                                    break

                        elif user_use_other in 'C':  # 'set all "OTHERS" for one object the same(c)',

                            obj_idx = self.num_or_str_obj_select('STR')
                            if obj_idx == 'BREAK': break

                            while True:
                                use_other = vui.validate_user_str(
                                    f'Use "OTHER" columns for all columns in {yy[obj_idx]} > ', 'YN')
                                if vui.validate_user_str(f'User entered "{use_other}", accept? (y/n) > ', 'YN') == 'Y':
                                    for col_idx in range(len(xx[obj_idx])):
                                        if self.MODIFIED_DATATYPES[obj_idx][col_idx] == 'STR':
                                            self.USE_OTHER[obj_idx][col_idx] = use_other
                                        else:
                                            self.USE_OTHER[obj_idx][col_idx] = 'N'
                                    break

                        elif user_use_other in 'D':  # 'set "OTHERS" for a specific column(d)',

                            while True:

                                obj_idx, col_idx = self.num_or_str_col_select('STR')
                                if obj_idx == 'BREAK': break

                                use_others = vui.validate_user_str(
                                    f'Use "OTHERS" for {self.column_desc(obj_idx, col_idx)} > ', 'YN')
                                if vui.validate_user_str(
                                        f'User entered "{use_others}" for {self.column_desc(obj_idx, col_idx)}, ' + \
                                        f'accept? (y/n) > ', 'YN') == 'Y':
                                    self.USE_OTHER[obj_idx][col_idx] = use_others
                                    break

                            if obj_idx == 'BREAK': break

                        elif user_use_other in 'EF':  # 'reset(e)', 'abort and exit without saving changes(f)'
                            self.USE_OTHER = deepcopy(self.USE_OTHER_UNDO)
                            print(f'"OTHERS" HAVE BEEN REVERTED BACK.')

                            if user_use_other in 'F':
                                break

                        elif user_use_other in 'G':  # 'enter "OTHERS" based on min cutoff entries(g)'
                            for obj_idx2 in self.OBJ_IDXS:
                                for yn_idx in range(len(self.MIN_CUTOFFS[obj_idx2])):
                                    if self.MIN_CUTOFFS[obj_idx2][yn_idx] > 0:
                                        self.USE_OTHER[obj_idx2][yn_idx] = vui.validate_user_str(
                                            f'Use "OTHER" for {self.column_desc(obj_idx2, yn_idx)} > ', 'YN')

                        elif user_use_other in 'Q':  # 'print column names & setup parameters(q)'
                            self.print_cols_and_setup_parameters()

                        elif user_use_other in 'A':
                            break

                        ppro.SelectionsPrint(USE_OTHER_CMDS, use_other_cmds_str, append_ct_limit=3)
                        user_use_other = vui.validate_user_str(' > ', use_other_cmds_str)

                if self.user_manual_or_std in '2':  # 'create calculated field(2)'
                    while True:
                        bb = self.MODIFIED_DATATYPES
                        # LOOK FOR NUMERIC COLUMNS ONLY IN DATA_NUMPY, TARGET, & REF_VECS
                        if True not in ['FLOAT' in _ or 'INT' in _ or 'BIN' in _ for _ in llm.list_of_lists_merger(deepcopy(bb))]:
                            print(
                                f'\nTHERE ARE NO NUMERIC COLUMNS IN DATA ARRAY, TARGET, OR REFERENCE VECTORS.  HOWEVER IT IS ')
                            print(
                                f'STILL POSSIBLE TO USE A NON-NUMERIC COLUMN TO GENERATE A CALCULATED FIELD, BUT THE FORMULA ')
                            print(f'CANNOT MAKE REFERENCE TO THE SOURCE COLUMN.')
                            if vui.validate_user_str(f'PROCEED? (y/n) > ', 'YN') == 'N':
                                break

                        obj_idx, col_idx = self.duplicate_column()  # RETURNS OBJ_IDX AND IDX OF NEW COLUMN, NOT SOURCE COL
                        if vui.validate_user_str(
                                f'\nCURRENT OBJECT TYPE IS {bb[obj_idx][col_idx]}, CHANGE? (y/n) > ', 'YN') == 'Y':
                            bb[obj_idx][col_idx] = {'S': 'STR', 'F': 'FLOAT', 'I': 'INT', 'B': 'BIN'}[vui.validate_user_str(
                                f'STR(s), FLOAT(f), INT(i), BIN(b) > ', 'SFIB')]

                        # DONT CARRY ANY FILTERING RECORDS THAT MAY HAVE COME FROM SOURCE COLUMN
                        self.FILTERING[obj_idx][col_idx] = []

                        try:
                            while True:
                                trial = input(
                                    '\nENTER FORMULA. REFER TO REFERENCE COLUMN AS x (e.g. -2*x**2 + 3*x + 3) > ').strip()
                                if vui.validate_user_str(f'User entered {trial} ... Accept? (y/n) > ', 'YN') == 'Y':
                                    break
                            # float(eval(trial.replace('x', str(_[row_idx])))) IS ROUNDING ENTERED DECIMALS SOMEHOW
                            # MAKE trial BE A FLOAT IF ITS A FLOAT
                            ed = vcsdt.ValidateCharSeqDataType(trial).type()[0]  # ed = ENTRY DATATYPE
                            if ed == 'STR':
                                trial = str(trial)
                            elif ed == 'INT':
                                trial = int(trial)
                            elif ed == 'BIN':
                                trial = int(trial)
                            elif ed == 'FLOAT':
                                trial = float(trial)

                            _ = self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx].copy()
                            for row_idx in range(len(_)):
                                if ed == 'STR':
                                    self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx][row_idx] = \
                                        float(eval(trial.replace('x', str(_[row_idx]))))
                                elif ed in ['FLOAT', 'INT', 'BIN']:  # THIS WORKS MAKING IT BE A FLOAT, LINE ABOVE ROUNDS (Y?)
                                    self.SUPER_RAW_NUMPY_LIST[obj_idx][col_idx][row_idx] = trial

                            print('\nCREATE CALCULATED FIELD SUCCESSFUL.\n')

                        except:
                            print(f'\nFORMULA IS INVALID.\n')
                            self.column_chop(obj_idx,
                                             col_idx)  # RESTORE TO ORIGINAL STATE (DONT USE UNDO, undo() GIVES A PRINTOUT)

                        if vui.validate_user_str(f'Create another calculated field (c) or exit(e) > ', 'CE') == 'E':
                            break
                        # 12-20-21 CREATE COLUMN OF ALL SAME VALUE GETS PICKED UP BELOW BY one_thing_left

                if self.user_manual_or_std in '3':   # 'reset target only to base state(3)'
                    # 1-8-22 THIS IS TO RESTORE TARGET ENTRIES TO BASE STATE (TO UNDO ANY REPLACEMENTS)
                    # CANT SIMPLY JUST REPLACE TARGET (SRNL[2]) WITH SRNL[2]_BASE_BACKUP BECAUSE FILTERING MAY HAVE
                    # HAPPENED, HAVE TO LOOK AT ROWID IN REFERENCE VECTORS TO GET THE CORRECT POSITIONS TO GRAB
                    # NO SOFTMAX EXPANSION SHOULD HAVE HAPPENED AT THIS POINT
                    self.SUPER_RAW_NUMPY_LIST[2] = np.array([[
                                        self.SUPER_RAW_NUMPY_LIST_BASE_BACKUP[2][0][_]
                                        for _ in self.SUPER_RAW_NUMPY_LIST[4][0]]], dtype=object)

                if self.user_manual_or_std in '4':   # 'reset target only to global state(c)'
                    # 1-8-22 SEE NOTES FOR "RESET TARGET TO BASE STATE"
                    self.SUPER_RAW_NUMPY_LIST[2] = np.array([[
                                        self.SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP[2][0][_]
                                        for _ in self.SUPER_RAW_NUMPY_LIST[4][0]]], dtype=object)

                if self.user_manual_or_std in '6':    # 'move a column within or between objects(6)'
                    while True:
                        home_obj_idx = 2 * ls.list_single_select([self.SXNL_DICT[_] for _ in self.OBJ_IDXS],
                                                                f'\nSelect object to move column from ', 'idx')[0]
                        home_obj_col_idx = ls.list_single_select(self.SUPER_RAW_NUMPY_LIST[home_obj_idx+1][0],
                                                                 f'\nSelect column to move ', 'idx')[0]
                        target_obj_idx = 2 * ls.list_single_select([self.SXNL_DICT[_] for _ in self.OBJ_IDXS],
                                                                f'\nSelect object to move column to ', 'idx')[0]
                        COLUMN_LIST_FOR_TARGET = self.SUPER_RAW_NUMPY_LIST[target_obj_idx + 1][0] + ['END']
                        target_obj_col_idx = ls.list_single_select(COLUMN_LIST_FOR_TARGET,
                                                                 f'\nSelect column to insert column before ', 'idx')[0]

                        print(f'\n Selected to move {self.SUPER_RAW_NUMPY_LIST[home_obj_idx+1][0][home_obj_col_idx]} from ' + \
                              f'{self.SXNL_DICT[home_obj_idx]} to {self.SXNL_DICT[target_obj_idx]}' + \
                              f'before column "{COLUMN_LIST_FOR_TARGET[target_obj_col_idx]}"...')
                        __ = vui.validate_user_str(f'Accept(a), try again(t), abort(b) > ', 'ATB')
                        if __ == 'B': break
                        elif __ == 'A': pass
                        elif __ == 'T': continue


                        # PUT COPY OF HOME COLUMN INTO MOVE TARGET OBJECT
                        self.SUPER_RAW_NUMPY_LIST[target_obj_idx] = np.insert(self.SUPER_RAW_NUMPY_LIST[target_obj_idx],
                            target_obj_col_idx, self.SUPER_RAW_NUMPY_LIST[home_obj_idx][home_obj_col_idx], axis=0)
                        # PUT COPY OF HOME COLUMN NAME INTO MOVE TARGET OBJECT HEADER
                        self.SUPER_RAW_NUMPY_LIST[target_obj_idx+1] = np.insert(self.SUPER_RAW_NUMPY_LIST[target_obj_idx+1],
                            target_obj_col_idx, self.SUPER_RAW_NUMPY_LIST[home_obj_idx+1][0][home_obj_col_idx],axis=1)
                        # PUT COPY OF HOME COLUMN INFO INTO OTHER RELATED MOVE TARGET OBJECTS
                        self.VALIDATED_DATATYPES[target_obj_idx].insert(target_obj_col_idx,
                                deepcopy(self.VALIDATED_DATATYPES[home_obj_idx][home_obj_col_idx]))

                        self.MODIFIED_DATATYPES[target_obj_idx].insert(target_obj_col_idx,
                                deepcopy(self.MODIFIED_DATATYPES[home_obj_idx][home_obj_col_idx]))

                        self.FILTERING[target_obj_idx].insert(target_obj_col_idx,
                                deepcopy(self.FILTERING[home_obj_idx][home_obj_col_idx]))

                        self.MIN_CUTOFFS[target_obj_idx].insert(target_obj_col_idx,
                                deepcopy(self.MIN_CUTOFFS[home_obj_idx][home_obj_col_idx]))

                        self.USE_OTHER[target_obj_idx].insert(target_obj_col_idx,
                                deepcopy(self.USE_OTHER[home_obj_idx][home_obj_col_idx]))

                        self.START_LAG[target_obj_idx].insert(target_obj_col_idx,
                                deepcopy(self.START_LAG[home_obj_idx][home_obj_col_idx]))

                        self.END_LAG[target_obj_idx].insert(target_obj_col_idx,
                                deepcopy(self.END_LAG[home_obj_idx][home_obj_col_idx]))

                        self.SCALING[target_obj_idx].insert(target_obj_col_idx,
                                deepcopy(self.SCALING[home_obj_idx][home_obj_col_idx]))

                        # DELETE HOME COLUMN FROM HOME OBJECT, HOME OBJECT HEADER, AND OTHER RELATED OBJECTS
                        self.column_chop(home_obj_idx, home_obj_col_idx)

                        print(f'\nCOLUMN MOVE COMPLETE.\n')
                        break

                if self.user_manual_or_std in '!':  # 'get object types(!)'
                    xx = self.SUPER_RAW_NUMPY_LIST
                    print(f'OBJECT DATATYPES ARE:')
                    [print(f'{self.SXNL_DICT[_]}'.ljust(40) + f'{xx[_].dtype}') for _ in self.OBJ_IDXS]

                if self.user_manual_or_std in '@':  # 'print FILTERING(@)'
                    self.print_parameter_object(self.FILTERING, 'FILTERING')

                if self.user_manual_or_std in '#':  # 'print MIN_CUTOFFS(#)'
                    self.print_parameter_object(self.MIN_CUTOFFS, 'MIN_CUTOFFS')

                if self.user_manual_or_std in '$':  # 'print USE_OTHER($)'
                    self.print_parameter_object(self.USE_OTHER, 'USE_OTHER')

                if self.user_manual_or_std in '%':  # 'print KEEP(%)'
                    oi.obj_info(self.KEEP, 'KEEP', __name__)   # THIS STAYS, PART OF THE PROGRAM

                if self.user_manual_or_std in '^':  # 'print CONTEXT(^)'
                    oi.obj_info(self.CONTEXT, 'CONTEXT', __name__)    # THIS STAYS, PART OF THE PROGRAM

                #### END OF CODE FOR COMMANDS, BEGIN CODE THAT EXECUTES AFTER EVERY COMMAND PROCESSED ##################
                ########################################################################################################

                ########################################################################################################
                ########################################################################################################
                #### DATA CHECKS AT THE END OF EVERY LOOP ##############################################################

                # ALL COLUMNS IN ALL OBJECTS EQUAL ROWS #################################################################
                _data_rows = gs.get_shape("DATA", self.SUPER_WORKING_NUMPY_LIST[0], self.data_given_orientation)[0]
                _target_rows = gs.get_shape("TARGET", self.SUPER_WORKING_NUMPY_LIST[1], self.target_given_orientation)[0]
                _refvecs_rows = gs.get_shape("REFVECS", self.SUPER_WORKING_NUMPY_LIST[2], self.refvecs_given_orientation)[0]
                if not _data_rows == _target_rows:
                    raise Exception(f'DATA ROWS ({_data_rows}) != TARGET ROWS ({_target_rows})')
                if not _data_rows == _refvecs_rows:
                    raise Exception(f'DATA ROWS ({_data_rows}) != REFVECS ROWS ({_refvecs_rows})')
                ########################################################################################################


                ### COMPARE # COLUMNS IN DATA/TARGET/REF OBJECTS TO COLUMNS IN RESPECTIVE SUPOBJS ######################
                # BEAR 6/29/23 LOOKS LIKE THIS CAN COME OUT WHEN CONVERTED TO NEW SUPOBJ
                NAMES = ("VALIDATED_DATATYPES", "MODIFIED_DATATYPES", "FILTERING", "MIN_CUTOFFS",
                         "USE_OTHER", "START_LAG", "END_LAG", "SCALING")
                _SUPOBJS = (self.VALIDATED_DATATYPES, self.MODIFIED_DATATYPES, self.FILTERING, self.MIN_CUTOFFS,
                            self.USE_OTHER, self.START_LAG, self.END_LAG, self.SCALING)

                for obj_idx in self.OBJ_IDXS:
                    base_length = len(self.SUPER_RAW_NUMPY_LIST[obj_idx])
                    obj_name = self.SXNL_DICT[obj_idx]

                    for so_name, _SUPOBJ in zip(NAMES, _SUPOBJS):
                        if base_length != len(_SUPOBJ[obj_idx]):
                            print(f'\033[91mINCONGRUITY IN COLUMNS BETWEEN {obj_name} AND {so_name}[{obj_idx}]\033[0m')

                    if len(self.SUPER_RAW_NUMPY_LIST[obj_idx]) != len(self.SUPER_RAW_NUMPY_LIST[obj_idx + 1][0]):
                        print(f'\033[91mLENGTH MISMATCH BETWEEN {obj_name} & ITS HEADER\033[0m')

                del NAMES, _SUPOBJS, base_length, obj_name
                ### END COMPARE # COLUMNS IN DATA/TARGET/REF OBJECTS TO COLUMNS IN RESPECTIVE SUPOBJS ##################


                self.pending_cutoff()
                self.one_thing_left()  # FIND IF A COLUMN ONLY CONTAINS ONE UNIQUE VALUE
                #### END DATA CHECKS ###################################################################################
                ########################################################################################################
                ########################################################################################################

                if self.user_manual_or_std == 'A':  # 'accept config / continue / bypass(a)'

                    # 12-22-21 USER MAY TOGGLE NUMERIC COLUMNS BACK-N-FORTH BETWEEN STR/FLOAT, BUT IF AT EXIT THE
                    # FINAL TYPE OF A COLUMN IS NOT 'STR' THEN THAT COLUMN CANT HAVE ANY NON-DEFAULT VALUES IN
                    # MIN_CUTOFFS & USE_OTHER
                    for obj_idx in self.OBJ_IDXS:
                        for col_idx in range(len(self.SUPER_RAW_NUMPY_LIST[obj_idx])):
                            if self.MODIFIED_DATATYPES[obj_idx][col_idx] != 'STR':
                                self.MIN_CUTOFFS[obj_idx][col_idx] = 0
                                self.USE_OTHER[obj_idx][col_idx] = 'N'

                    self.one_thing_left()

                    break  # BREAK OUT OF COMMAND ENTRY LOOP

                ppro.SelectionsPrint(self.menu_filter_commands(), self.filter_str, append_ct_limit=3, max_len=self.max_cmd_len)
                ppro.SelectionsPrint(self.menu_datastep_commands(), self.datastep_str, append_ct_limit=3, max_len=self.max_cmd_len)
                ppro.SelectionsPrint(self.menu_generic_commands(), self.generic_str, append_ct_limit=3, max_len=self.max_cmd_len)
                ppro.SelectionsPrint(self.menu_hidden_commands(), self.hidden_str, append_ct_limit=3, max_len=self.max_cmd_len)

                self.user_manual_or_std = vui.validate_user_str(' > ', self.allowed_commands_string)

            # SHOW RESULTS
            for idx in self.OBJ_IDXS:
                # 8/31/22 ioap IN print_preview TAKING A LONG TIME ON BIG DATA, REPLICATE ioap WITH SIMPLER CODE
                # self.print_preview(self.SUPER_RAW_NUMPY_LIST[idx], self.SXNL_DICT[idx],
                #           rows=20, columns=15, start_row=0, start_col=0, header=self.SUPER_RAW_NUMPY_LIST[idx + 1][0])

                _ = self.SXNL_DICT[idx]
                outer_len = len(self.SUPER_RAW_NUMPY_LIST[idx])
                inner_len = len(self.SUPER_RAW_NUMPY_LIST[idx][0])
                print(f'\nMODULE = {self.this_module}   FXN = {inspect.stack()[1][3]}   OBJECT = {_}')
                print(f'"{_}" IS A(N) ARRAY OF {outer_len} ARRAY(S) OF {"S, ".join(np.unique(self.MODIFIED_DATATYPES[idx]))}S')
                print(f'AS A DF, {_}[0:{min(20, inner_len)}][0:{min(10, outer_len)}] OF [:{inner_len}][:{outer_len}] LOOKS LIKE:')
                print()
                print(pd.DataFrame(self.SUPER_RAW_NUMPY_LIST[idx].transpose(), columns=self.SUPER_RAW_NUMPY_LIST[idx+1][0]).iloc[:20,:10])

            user_accept_filtering = vui.validate_user_str(f'\nAccept filtering? (y/n) or abort(a) > ', 'YNA')
            if user_accept_filtering == 'Y':

                post_dump_handling = 'A'

                while True:

                    if vui.validate_user_str(f'\nSave filtered DATA to file? (y/n) > ', 'YN') == 'N': break
                    else:
                        base_path = bps.base_path_select()
                        file_name = fe.filename_wo_extension()
                        _ext = vui.validate_user_str(f'Select file type -- csv(c) excel(e) > ', 'CE')

                        full_path = base_path + file_name + {'C':'.csv', 'E':'.xlsx'}[_ext]

                        print(f'\nWorking on it...')

                        if _ext == 'E':
                            OBJECTS = [self.SUPER_RAW_NUMPY_LIST[0],
                                       self.SUPER_RAW_NUMPY_LIST[2],
                                       self.SUPER_RAW_NUMPY_LIST[4]
                                       ]
                            HEADERS = [self.SUPER_RAW_NUMPY_LIST[1],
                                       self.SUPER_RAW_NUMPY_LIST[3],
                                       self.SUPER_RAW_NUMPY_LIST[5]
                                       ]
                            SHEET_NAMES = [f'DATA', f'TARGET', f'REF VECS', f'TEST']

                            with pd.ExcelWriter(full_path) as writer:
                                for idx in range(len(SHEET_NAMES)):

                                    DF = pd.DataFrame(OBJECTS[idx].transpose())

                                    try:
                                        DF.to_excel(excel_writer=writer,
                                                     sheet_name=SHEET_NAMES[idx],
                                                     header=True,
                                                     index=False
                                                     )
                                    except:
                                        # IF EXCEPTION, SHOW ON FILE SHEET
                                        pd.DataFrame([[f'*** ERROR WRITING {SHEET_NAMES[idx]} TO FILE ***']]).to_excel(
                                                    excel_writer=writer,
                                                    sheet_name=SHEET_NAMES[idx],
                                                    header=None,
                                                    index=False
                                                    )

                        elif _ext == 'C':
                            pd.DataFrame(
                                        np.vstack((self.SUPER_RAW_NUMPY_LIST[0],
                                                  self.SUPER_RAW_NUMPY_LIST[2],
                                                  self.SUPER_RAW_NUMPY_LIST[4])).transpose(),
                                        ).to_csv( full_path,
                                                  header=np.hstack((self.SUPER_RAW_NUMPY_LIST[1],
                                                                  self.SUPER_RAW_NUMPY_LIST[3],
                                                                  self.SUPER_RAW_NUMPY_LIST[5]))[0]
                            )

                        print(f'Done.\n')

                    post_dump_handling = vui.validate_user_str(
                        f'Accept file dump(a), try file dump again(t), go back to filtering(g), quit(q) > ', 'ATGQ')

                    if post_dump_handling == 'Q': sys.exit(f'\n*** TERMINATED BY USER ***\n')
                    elif post_dump_handling == 'T': continue
                    elif post_dump_handling in ['A', 'G']: break

                if post_dump_handling == 'G': break     # BREAK BEFORE del
                # if post_dump_handling == 'A': pass     # JUST FOR CLARITY

                del self.SUPER_RAW_NUMPY_LIST_SESSION_BACKUP, self.KEEP_SESSION_BACKUP, self.VALIDATED_DATATYPES_SESSION_BACKUP, \
                    self.MODIFIED_DATATYPES_SESSION_BACKUP, self.FILTERING_SESSION_BACKUP, self.MIN_CUTOFFS_SESSION_BACKUP, \
                    self.USE_OTHER_SESSION_BACKUP, self.START_LAG_SESSION_BACKUP, self.END_LAG_SESSION_BACKUP, \
                    self.SCALING_SESSION_BACKUP, self.CONTEXT_SESSION_BACKUP
                try:
                    del self.SUPER_RAW_NUMPY_LIST_UNDO, self.KEEP_UNDO, self.MODIFIED_DATATYPES_UNDO,self.VALIDATED_DATATYPES_UNDO, \
                        self.FILTERING_UNDO,self.MIN_CUTOFFS_UNDO, self.USE_OTHER_UNDO, self.CONTEXT_UNDO, self.START_LAG_UNDO, \
                        self.END_LAG_UNDO, self.SCALING_UNDO
                except: pass

                break # BREAK OF TOP LEVEL WHILE LOOP

            else:  # user_accept_filtering is NO or ABORT

                # 10/15/22 THIS IS BLOWING UP ON RETURN TOP OF while AND REACHING initialize() WHEN SESSION_BACKUP OBJECTS
                # WERE NOT CREATED. NOT SURE WHAT THE ORIGINAL INTENT OF "ABORT" WAS.  FOR NOW, SIMPLY GOING TO ASK TO TERMINATE,
                # OR START OVER (WHICH REPLICATES SAYING "N" AT "Accept filtering?")

                self.user_manual_or_std = 'BYPASS'

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if np.array_equiv(self.SUPER_RAW_NUMPY_LIST_SESSION_BACKUP, []) or self.SUPER_RAW_NUMPY_LIST_SESSION_BACKUP == '':
                        # IF DID NOT CREATE BACKUP OBJECTS
                        print(f'\n**** BACKUP OBJECTS WERE NOT CREATED.  CAN ONLY MAKE MORE MODIFICATIONS TO EXISTING OBJECTS. ****\n')

                        _ = vui.validate_user_str(f'Restart filtering(f), accept current filtering(a), or terminate program(t) > ', 'FAT')
                        if _ == 'F': self.user_manual_or_std = 'BYPASS'
                        elif _ == 'A': break
                        elif _ == 'T': sys.exit(f'TERMINATED BY USER.')

                    else:
                        # RESTORE OBJECTS TO ORIGINAL STATE
                        self.SUPER_RAW_NUMPY_LIST = np.array([_.copy() for _ in self.SUPER_RAW_NUMPY_LIST_SESSION_BACKUP], dtype=object)
                        self.KEEP = deepcopy(self.KEEP_SESSION_BACKUP)
                        self.VALIDATED_DATATYPES = deepcopy(self.VALIDATED_DATATYPES_SESSION_BACKUP)
                        self.MODIFIED_DATATYPES = deepcopy(self.MODIFIED_DATATYPES_SESSION_BACKUP)
                        self.FILTERING = deepcopy(self.FILTERING_SESSION_BACKUP)
                        self.MIN_CUTOFFS = deepcopy(self.MIN_CUTOFFS_SESSION_BACKUP)
                        self.USE_OTHER = deepcopy(self.USE_OTHER_SESSION_BACKUP)
                        self.START_LAG = deepcopy(self.START_LAG_SESSION_BACKUP)
                        self.END_LAG = deepcopy(self.END_LAG_SESSION_BACKUP)
                        self.SCALING = deepcopy(self.SCALING_SESSION_BACKUP)
                        self.CONTEXT = deepcopy(self.CONTEXT_SESSION_BACKUP)

                # if user_accept_filtering == 'N':
                # elif user_accept_filtering == 'A':


        return self.return_fxn()


########################################################################################################################
########################################################################################################################
########################################################################################################################

if __name__ == '__main__':

    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl
    from MLObjects.TestObjectCreators import test_header as th

    columns = 9
    rows = 10000
    _format = 'ARRAY'
    _orient = 'COLUMN'

    # ##################################################################################################################
    # GENERATE SUPER_NUMPY OBJECTS #####################################################################################

    ##### CREATE DATA##################################################################################################
    SXNLClass = csxnl.CreateSXNL(
                                    rows=rows,
                                    bypass_validation=True,
                                    data_return_format=_format,
                                    data_return_orientation=_orient,
                                    DATA_OBJECT=None,
                                    DATA_OBJECT_HEADER=th.test_header(columns),
                                    DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                    data_override_sup_obj=False,
                                    data_given_orientation=None,
                                    data_columns=columns,
                                    DATA_BUILD_FROM_MOD_DTYPES=np.random.choice(['FLOAT','INT','BIN','STR'], columns, replace=True),
                                    DATA_NUMBER_OF_CATEGORIES=10,
                                    DATA_MIN_VALUES=-10,
                                    DATA_MAX_VALUES=10,
                                    DATA_SPARSITIES=None,
                                    DATA_WORD_COUNT=None,
                                    DATA_POOL_SIZE=None,
                                    target_return_format='ARRAY',
                                    target_return_orientation=_orient,
                                    TARGET_OBJECT=None,
                                    TARGET_OBJECT_HEADER=[['STATUS']],
                                    TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                    target_type='FLOAT',  # MUST BE 'BINARY','FLOAT', OR 'SOFTMAX'
                                    target_override_sup_obj=False,
                                    target_given_orientation=None,
                                    target_sparsity=0,
                                    target_build_from_mod_dtype='FLOAT',  # COULD BE FLOAT OR INT
                                    target_min_value=0,
                                    target_max_value=20,
                                    target_number_of_categories=None,
                                    refvecs_return_format='ARRAY',
                                    refvecs_return_orientation=_orient,
                                    REFVECS_OBJECT=np.array([[*range(rows)], [*range(1,rows+1)]], dtype=np.int16),
                                    REFVECS_OBJECT_HEADER=np.array([['IDX', 'ROWID']], dtype='<U5'),
                                    REFVECS_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                    REFVECS_BUILD_FROM_MOD_DTYPES=None,
                                    refvecs_override_sup_obj=False,
                                    refvecs_given_orientation=None,
                                    refvecs_columns=None,
                                    REFVECS_NUMBER_OF_CATEGORIES=None,
                                    REFVECS_MIN_VALUES=None,
                                    REFVECS_MAX_VALUES=None,
                                    REFVECS_SPARSITIES=None,
                                    REFVECS_WORD_COUNT=None,
                                    REFVECS_POOL_SIZE=None
                )

    SUPER_RAW_NUMPY_LIST = SXNLClass.SXNL
    RAW_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS
    CONTEXT = []
    KEEP = RAW_SUPOBJS[1][0].copy()


    # GENERATE BACKUP OBJECTS
    # NO BASE_BACKUP AT THIS POINT, GOING INTO PreRun
    SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP = [_.copy() for _ in SUPER_RAW_NUMPY_LIST]
    KEEP_GLOBAL_BACKUP = KEEP.copy()

    # 12-20-21 8:42 PM CREATE DUMMY DTYPE GLOBALS FOR PreRun - NEED TO BE INGESTED BY InSitu BECAUSE GLOBAL BACKUPS
    # FOR DTYPES ARE GENERATED AFTER FIRST TIME FILLING (SINCE DTYPES MUST BE SELECTED FOR ALL COLUMNS AS COMING
    # IN AFTER ORIGINAL SRNL CREATION)
    FULL_SUPOBJS_GLOBAL_BACKUP = deepcopy(RAW_SUPOBJS)
    CONTEXT_GLOBAL_BACKUP = deepcopy(CONTEXT)

    # END GENERATE SUPER_NUMPY OBJECTS #################################################################################
    # ##################################################################################################################
    # ##################################################################################################################

    standard_config = 'BLAH'
    user_manual_or_standard = 'BYPASS'
    filter_method = ''



    # PreRun EMULATOR
    SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, CONTEXT, KEEP, VALIDATED_DATATYPES_GLOBAL_BACKUP, MODIFIED_DATATYPES_GLOBAL_BACKUP = \
        PreRunFilter(
                        standard_config,
                        user_manual_or_standard,
                        filter_method,
                        SUPER_RAW_NUMPY_LIST,
                        _orient,  # data_given_orientation
                        _orient,  # target_given_orientation
                        _orient,  # refvecs_given_orientation
                        RAW_SUPOBJS,
                        CONTEXT,
                        KEEP,
                        *range(4),              #CREATE DUMS FOR BASE_BACKUPS (ONLY IN PreRun)
                        SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP,
                        FULL_SUPOBJS_GLOBAL_BACKUP,
                        KEEP_GLOBAL_BACKUP,
                        bypass_validation=False
    ).config_run()


    SUPER_RAW_NUMPY_LIST_BASE_BACKUP = np.array([_.copy() for _ in SUPER_RAW_NUMPY_LIST], dtype=object)
    FULL_SUPOBJS_BASE_BACKUP = deepcopy(RAW_SUPOBJS)
    CONTEXT_BASE_BACKUP = deepcopy(CONTEXT)
    KEEP_BASE_BACKUP = deepcopy(KEEP)

    # InSitu EMULATOR:
    from ML_PACKAGE.DATA_PREP_IN_SITU_PACKAGE import InSituFilter as isf

    while True:
        SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, CONTEXT, KEEP = \
            isf.InSituFilter(
                                standard_config,
                                user_manual_or_standard,
                                filter_method,
                                SUPER_RAW_NUMPY_LIST,
                                RAW_SUPOBJS,
                                CONTEXT,
                                KEEP,
                                SUPER_RAW_NUMPY_LIST_BASE_BACKUP,
                                FULL_SUPOBJS_BASE_BACKUP,
                                CONTEXT_BASE_BACKUP,
                                KEEP_BASE_BACKUP,
                                SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP,
                                FULL_SUPOBJS_GLOBAL_BACKUP,
                                KEEP_GLOBAL_BACKUP,
                                bypass_validation=False
        ).config_run()

        if vui.validate_user_str(f'Filter again(a) or quit(q) > ', 'AQ') == 'Q':
            break





