import sys, inspect
import numpy as np
from copy import deepcopy
import sparse_dict as sd
from debug import get_module_name as gmn, IdentifyObjectAndPrint as ioap
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import ValidateObjectType as vot, list_dict_validater as ldv
from general_list_ops import list_select as ls
from general_data_ops import return_uniques as ru, get_shape as gs
from ML_PACKAGE.GENERIC_PRINT import print_post_run_options as ppro, obj_info as oi
from ML_PACKAGE.standard_configs import standard_configs as sc
from linear_algebra import XTX_determinant as xtxd
from MLObjects import MLObject as mlo, MLStandardizeNormalize as mlsn, ML_find_constants as mlfc
from MLObjects.SupportObjects import master_support_object_dict as msod, PrintSupportContents as psc



# standard_config_source                (returns standard config module - NOT FINISHED)
# generic_menu_commands                 (list of general commands)
# augment_menu_commands                 (list of commands for performing augmentation)
# hidden_menu_commands                  (list of hidden commands)



# is_dict                               (True if data object is dict)
# is_list                               (True if data object is list, tuple, ndarray)
# print_preview                         (print data object)
# print_cols_and_setup_parameters       (prints column name, validated type, user type, min cutoff, "other", scaling, lag start, lag end, & filtering for all columns in all objects)
# delete_column                         (delete column from DATA)
# delete_row                            (delete row from DATA)
# insert_column                         (append column to DATA)
# intercept_verbage                     (text added to CONTEXT when INTERCEPT is appended)



# interaction_verbage                   (text added to CONTEXT when INTERACTIONS are appended)
# column_drop_iterator                  (iterate thru object, drop 1 column at a time, report freq of dropped & determ for all iter)
# set_start_end_lags                    (set start or end lags for float columns in DATA)
# undo                                  (undo last non-generic operation)
# return_fxn                            (return)
# config_run                            (exe)



class PreRunDataAugment:
    def __init__(self,
                 standard_config,
                 user_manual_or_standard,
                 augment_method,
                 SWNL,
                 data_given_orientation,
                 target_given_orientation,
                 refvecs_given_orientation,
                 WORKING_SUPOBJS,
                 CONTEXT,
                 KEEP,
                 bypass_validation
         ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        self.bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False],
                                                         self.this_module, fxn)

        # THESE MUST BE ABOVE ALL self.augment_menu_commands() ##################################################################
        self.CONTEXT = CONTEXT
        # END THESE MUST BE ABOVE ALL self.augment_menu_commands() ##############################################################

        # THESE MUST BE ABOVE ALL self.generic_menu_commands() ##################################################################
        self.DATA_OBJECT_WIP = ldv.list_dict_validater(SWNL[0], "DATA")  # THIS MUST BE HERE FOR self.is_list() & self.is_dict() TO WORK

        self.allow_undo = {'Y': True, 'N': False}[
            vui.validate_user_str(f'\nEnable BACKUP objects for UNDO in DataAugment? (Disabling frees up memory) (y/n) > ', 'YN')]
        # END THESE MUST BE ABOVE ALL self.generic_menu_commands() ###############################################################






        ###############################################################################################################
        # DETERMINES xxxSelectionsPrint ALLOWABLES, ALLOWABLES for vui.validate_user_str, AND ALSO WHICH
        # COMMAND TEXT LINES ARE PRINTED FROM THE commands ABOVE

        self.MASTER_DICT = self.augment_menu_commands() | self.generic_menu_commands() | self.hidden_menu_commands()

        # USED CHARS:   ABCDEFG I  LMN PQRSTU   Y 12345   90!@#$%^&   _+
        # UNUSED CHARS:        H JK   O      VWX Z     678         *()

        self.generic_str = "".join(list(self.generic_menu_commands().keys())).upper()     # 'ABCFGMPQRTUY'
        self.augment_str = "".join(list(self.augment_menu_commands().keys())).upper()     # 'DEILNS1234590'
        self.hidden_str = "".join(list(self.hidden_menu_commands().keys())).upper()      # '!@#$%^&*_+'

        self.allowed_commands_string = self.generic_str + self.augment_str + self.hidden_str


        if not self.bypass_validation:
            if len(self.MASTER_DICT) != len(self.allowed_commands_string):
                raise ValueError(f'\n*** DUPLICATE KEY IN generic_menu_commands, augment_menu_commands,'
                                            f'AND hidden_menu_commands. ***')
        ###############################################################################################################


        self.standard_config = standard_config
        self.user_manual_or_standard = user_manual_or_standard
        self.method = augment_method

        # DONT INCLUDE HIDDEN OPTIONS
        self.max_cmd_len = max(map(len, dict((self.generic_menu_commands() | self.augment_menu_commands())).values()))

        self.data_given_orientation = akv.arg_kwarg_validater(data_given_orientation, 'data_given_orientation',
                                                             ['ROW', 'COLUMN'], self.this_module, fxn)
        self.target_given_orientation = akv.arg_kwarg_validater(target_given_orientation, 'target_given_orientation',
                                                             ['ROW', 'COLUMN'], self.this_module, fxn)
        self.refvecs_given_orientation = akv.arg_kwarg_validater(refvecs_given_orientation, 'refvecs_given_orientation',
                                                             ['ROW', 'COLUMN'], self.this_module, fxn)

        self.WORKING_SUPOBJS = [_.copy() for _ in WORKING_SUPOBJS]
        self.msod_hdr_idx = msod.QUICK_POSN_DICT()["HEADER"]
        self.msod_mdtype_idx = msod.QUICK_POSN_DICT()["MODIFIEDDATATYPES"]

        self.DATA_OBJECT_HEADER_WIP = self.WORKING_SUPOBJS[0][self.msod_hdr_idx]
        self.TARGET_OBJECT = ldv.list_dict_validater(SWNL[1], 'TARGET')
        self.TARGET_OBJECT_HEADER = self.WORKING_SUPOBJS[1][self.msod_hdr_idx]

        self.REFVECS = ldv.list_dict_validater(SWNL[2], 'REFVECS')


        self.KEEP = KEEP

        self.allow_reset = {'Y': True, 'N': False}[
            vui.validate_user_str(f'\nEnable SESSION BACKUP objects for reset in DataAugment? (Disabling frees up memory) (y/n) > ', 'YN')]
        if self.allow_reset:
            print(f'Creating DATA reset backup object for DataAugment session.  Patience...')
            self.DATA_OBJECT_WIP_BACKUP = self.DATA_OBJECT_WIP.copy() if self.is_list() else deepcopy(self.DATA_OBJECT_WIP)
            print(f'Done.')
            print(f'Creating other reset backup objects for DataAugment session...')
            self.DATA_OBJECT_HEADER_WIP_BACKUP = deepcopy(self.DATA_OBJECT_HEADER_WIP)
            self.WORKING_SUPOBJS_BACKUP = [_.copy() for _ in self.WORKING_SUPOBJS]
            self.CONTEXT_BACKUP = deepcopy(CONTEXT)
            self.KEEP_BACKUP = deepcopy(KEEP)
            print(f'Done.')
        else:
            self.DATA_OBJECT_HEADER_WIP_BACKUP, self.WORKING_SUPOBJS_BACKUP, self.CONTEXT_BACKUP, self.KEEP_BACKUP = '','','',''

        self.DATA_OBJECT_WIP_UNDO = ''
        self.DATA_OBJECT_HEADER_WIP_UNDO = ''
        self.WORKING_SUPOBJS = ''
        self.CONTEXT_UNDO = ''
        self.KEEP_UNDO = ''

        self.DATA_OBJECT_HEADER_ORIGINALS = ''

        self.header_width = 50
        self.first_col_width = 12
        self.freq_width = 10
        self.stat_width = 15

    # END init ############################################################################################################
    #######################################################################################################################
    #######################################################################################################################




    def standard_config_module(self):
        return sc.AUGMENT_standard_configs(self.standard_config, self.method,
                                           self.SUPER_RAW_NUMPY_LIST, self.CONTEXT, self.KEEP)


    def generic_menu_commands(self):    # generic_menu_commands ARE NOT REMEMBERED BY undo
        return {
                'a': 'accept / continue',
                'b': 'get size & sparsity',
                'c': 'calculate column intercorrelations',
                'f': 'float check',
                'g': f'turn undo functionality {"OFF" if self.allow_undo else "ON"}',
                'm': 'colinearity iterator on DATA MATRIX',
                'p': 'print preview data object',
                'q': 'print column names & setup info',
                'r': 'reset and start over',
                't': 'check colinearity / tests inverse on DATA',
                'u': 'undo last operation',
                'y': f'convert DATA to {"sparse dicts" if self.is_list() else "lists"}]'
        }


    def augment_menu_commands(self):   # augment_menu_commands ARE REMEMBERED BY undo
        return {
                'd': 'delete column',
                'e': 'delete row',
                'i': 'append intercept to data',
                'l': 'apply lag',
                'n': f'add interactions ({"Done" if self.interaction_verbage() in self.CONTEXT else "Not done"})',
                's': 'standardize / normalize',
                '1': 'add moving average',
                '2': 'add forward derivative',
                '3': 'add trailing derivative',
                '4': 'add centered derivative',
                '5': 'add % change from previous',
                '9': 'set lag start',
                '0': 'set lag end'
        }


    def hidden_menu_commands(self):
        return {
                '!': 'print DATA_OBJECT_HEADER',
                '@': 'print VALIDATED_DATATYPES',
                '#': 'print MODIFIED_DATATYPES',
                '_': 'show FILTERING',
                '$': 'show MIN_CUTOFFS',
                '%': 'show USE_OTHER',
                '^': 'show CONTEXT',
                '+': 'show KEEP',
                '&': 'show SCALING',
                '*': 'show LAG'
        }


    def is_dict(self):    # MUST STAY AS FXNS, CANNOT BE DONE JUST ONCE IN __init__, BECAUSE TYPE CAN CHANGE
        '''True if data object is dict.'''
        return isinstance(self.DATA_OBJECT_WIP, dict)


    def is_list(self):
        '''True if data object is list, tuple, ndarray.'''
        return isinstance(self.DATA_OBJECT_WIP, (list, tuple, np.ndarray))


    def print_preview(self, rows, columns):

        # THIS IS SCREWED UP, IOAP PRINT AS DF IS DUPLICATING DUMMY COLUMNS, BUT NOT FLOAT (?????)
        # DATA_OBJECT_WIP IS FINE WHEN PRINTED OTHER THAN AS DF
        txt = lambda row_or_column: f'Enter start {row_or_column} of preview (zero-indexed) > '

        if self.is_list():
            ioap.IdentifyObjectAndPrint(self.DATA_OBJECT_WIP, 'DATA_OBJECT AS LISTS', __name__, rows, columns,
                start_row=vui.validate_user_int(txt('row'), min=0, max=len(self.DATA_OBJECT_WIP[0]) - 1),
                start_col=vui.validate_user_int(txt('column'), min=0, max=len(self.DATA_OBJECT_WIP) - 1),
                ).run()

        elif self.is_dict():
            ioap.IdentifyObjectAndPrint(self.DATA_OBJECT_WIP, 'DATA_OBJECT AS SPARSEDICT', __name__, rows, columns,
                start_row=vui.validate_user_int(txt('row'), min=0, max=sd.inner_len(self.DATA_OBJECT_WIP) - 1),
                start_col=vui.validate_user_int(txt('column'), min=0, max=sd.outer_len(self.DATA_OBJECT_WIP) - 1),
                ).run()


    def print_cols_and_setup_parameters(self):
        #  (prints column name, validated type, user type, min cutoff, "other", scaling, lag start, lag end, & filtering for all columns in all objects)

        fxn = inspect.stack()[0][3]

        psc.PrintSupportObjects_NewObjects(
                                            self.DATA_OBJECT_WIP,
                                            'DATA',
                                            orientation=self.data_given_orientation,
                                            SUPPORT_OBJECT_HOLDER=self.DATA_SUPOBJ,
                                            max_hdr_len=max(list(map(len, self.DATA_OBJECT_HEADER_WIP[0]))),
                                            calling_module=self.this_module,
                                            calling_fxn=fxn
        )

        psc.PrintSupportObjects_NewObjects(
                                            self.TARGET_OBJECT,
                                            'TARGET',
                                            orientation=self.target_given_orientation,
                                            SUPPORT_OBJECT_HOLDER=self.TARGET_SUPOBJ,
                                            max_hdr_len=max(list(map(len, self.TARGET_SUPOBJ[msod.QUICK_POSN_DICT()["HEADER"]]))),
                                            calling_module=self.this_module,
                                            calling_fxn=fxn
        )

        '''
        6/28/23
        OLD STUFF>>>> BEAR DELETE IF ABOVE IS OK
        print(f'\nDATA SETUP HYPERPARAMETERS:')

        # GET MAX_HDR_LEN OF ALL OBJECTS, FOR SYMMETRY OF ALL OBJECTS IN PRINTOUT

        all_max_hdr_len = np.max([len(_) for _ in self.DATA_OBJECT_HEADER_WIP[0]])

        index_width = 5
        column_width = min(max(8, all_max_hdr_len + 5), 40)
        type_width = 7
        user_type_width = 7
        min_cutoff_width = 7
        use_other_width = 7
        scaling_width = 28
        start_lag_width = 7
        end_lag_width = 7
        filtering_width = 10

        OBJECTS = [self.DATA_OBJECT_WIP, self.DATA_OBJECT_HEADER_WIP[0], self.TARGET_OBJECT, self.TARGET_OBJECT_HEADER[0]]

        for idx in [0,1]:  # idx IS obj_idx
            print('OBJECT = ' + {0: 'DATA', 1: 'TARGET', 2: 'REFVECS'}[idx])

            print(" "*(index_width+column_width+type_width) + f'USER'.ljust(user_type_width) + f'MIN'.ljust(min_cutoff_width) + \
                  f'USE'.ljust(use_other_width) + " "*(scaling_width) + 'START'.ljust(start_lag_width) + 'END'.ljust(end_lag_width))

            print(' ' * index_width + f'COLUMN'.ljust(column_width) + 'TYPE'.ljust(type_width) + 'TYPE'.ljust(user_type_width) + \
                'CUTOFF'.ljust(min_cutoff_width) + 'OTHER'.ljust(use_other_width) + 'SCALING'.ljust(scaling_width) + \
                'LAG'.ljust(start_lag_width) + 'LAG'.ljust(end_lag_width) + 'FILTERING'
                )

            for y in range(len(OBJECTS[idx])):  # idx IS obj_idx     y IS ESSENTIALLY col_idx
                # CREATE A BASE TEXT STRING THAT HOLDS EVERYTHING
                try: scaling_text = [self.SCALING[y].ljust(scaling_width) if idx==0 else " "*scaling_width][0]
                except: scaling_text = " "*scaling_width
                print(f'{y}) '.ljust(index_width) + \
                               f'{OBJECTS[idx + 1][y][:column_width - 2]}'.ljust(column_width) + \
                               f'{self.VALIDATED_DATATYPES[idx][y]}'.ljust(type_width) + \
                               f'{self.MODIFIED_DATATYPES[idx][y]}'.ljust(user_type_width) + \
                               f'{self.MIN_CUTOFFS[idx][y]}'.ljust(min_cutoff_width) + \
                               f'{self.USE_OTHER[idx][y]}'.ljust(use_other_width) + \
                               f'{scaling_text}' + \
                               f'{[str(self.START_LAG[y]).ljust(start_lag_width) if idx==0 else " "*start_lag_width][0]}' + \
                               f'{[str(self.END_LAG[y]).ljust(end_lag_width) if idx==0 else " "*end_lag_width][0]}' + \
                               [f'{self.FILTERING[idx][y]}' if idx % 2 == 0 else ''][0]
                      )
            print()

        del OBJECTS
        '''

    def delete_column(self, col_idx):
        _ = self.DATA_OBJECT_WIP[col_idx]
        has_explicit_intercept = self.DATA_OBJECT_HEADER_WIP[0][col_idx] == 'INTERCEPT' and self.intercept_verbage() in self.CONTEXT

        if self.is_list():
            # IF 1 == 1 --- DELETE INTERCEPT APPEND VERBAGE FROM CONTEXT IF DELETING INTERCEPT COLUMN
            if has_explicit_intercept or np.min(_)==np.max(_):
                self.CONTEXT.pop(np.argwhere(self.CONTEXT==self.intercept_verbage()))
            self.DATA_OBJECT_WIP = np.delete(self.DATA_OBJECT_WIP, col_idx, axis=0)

        elif self.is_dict():
            self.DATA_OBJECT_WIP = sd.delete_inner_key(self.DATA_OBJECT_WIP, [col_idx])
            # IF 1 == 1 --- DELETE INTERCEPT APPEND VERBAGE FROM CONTEXT IF DELETING INTERCEPT COLUMN
            if has_explicit_intercept or sd.min_({0:_})==sd.max_({0:_}):
                self.CONTEXT.pop(np.argwhere(self.CONTEXT == self.intercept_verbage()))
            self.DATA_OBJECT_WIP = sd.delete_outer_key(self.DATA_OBJECT_WIP, [col_idx])[0]

        self.DATA_OBJECT_HEADER_WIP = np.delete(self.DATA_OBJECT_HEADER_WIP, col_idx, axis=1)
        self.WORKING_SUPOBJS[0] = np.delete(self.WORKING_SUPOBJS[0], col_idx, axis=1)


    # INHERITED
    # def delete_row(self, ROW_IDXS_AS_INT_OR_LIST)


    def insert_column(self, col_idx, OBJECT_TO_INSERT, insert_orientation, SUPOBJ_INSERT):
        # 6/28/23 NOT USED
        fxn = inspect.stack()[0][3]

        InserterClass = mlo.MLObject(
                                        self.DATA_OBJECT_WIP,
                                        self.data_given_orientation,
                                        name="DATA OBJECT WIP",
                                        return_orientation='AS_GIVEN',
                                        return_format='AS_GIVEN',
                                        bypass_validation=self.bypass_validation,
                                        calling_module=self.this_module ,
                                        calling_fxn=fxn
        )

        InserterClass.insert_column(
                                    col_idx,
                                    OBJECT_TO_INSERT,
                                    insert_orientation,
                                    HEADER_OR_FULL_SUPOBJ=self.WORKING_SUPOBJS[0],
                                    CONTEXT=self.CONTEXT,
                                    SUPOBJ_INSERT=None
        )

        self.DATA_OBJECT_HEADER_WIP = InserterClass.OBJECT
        self.WORKING_SUPOBJS[0] = InserterClass.HEADER_OR_FULL_SUPOBJ
        self.CONTEXT = InserterClass.CONTEXT

        del InserterClass


    def intercept_verbage(self):
        # intercept_verbage                 (text added to CONTEXT when INTERCEPT is appended)
        return f'Appended INTERCEPT column.'


    def interaction_verbage(self):
        # interaction_verbage               (text added to CONTEXT when INTERACTIONS are appended)
        return f'Appended Interactions.'


    def column_drop_iterator(self, OBJECT, name, HEADER, append_ones='N'):

        fxn = inspect.stack()[0][3]

        if self.is_list(): CYCLE_OBJECT = OBJECT.copy()
        elif self.is_dict(): CYCLE_OBJECT = deepcopy(OBJECT)
        NEW_HEADER = deepcopy(HEADER)
        if append_ones == 'Y':

            if self.is_list():
                CYCLE_OBJECT = np.vstack((CYCLE_OBJECT, np.ones((1, len(CYCLE_OBJECT[0])), dtype=int)))
                CYCLE_OBJECT_XTX = np.matmul(CYCLE_OBJECT, CYCLE_OBJECT.transpose())
            elif self.is_dict():
                CYCLE_OBJECT = sd.append_outer(CYCLE_OBJECT, {_:1 for _ in range(sd.inner_len(CYCLE_OBJECT))})[0]
                CYCLE_OBJECT_XTX = sd.sparse_AAT(CYCLE_OBJECT, return_as='ARRAY')

            NEW_HEADER = np.array([*NEW_HEADER, 'ONES'])

        max_len = np.max(map(len, [*NEW_HEADER, f'ORIGINAL COLUMN: {name}']))
        # PRINT HEADER FOR RESULTS OF ENTIRE OBJECT:
        print(
            f'{"DATA COLUMN".ljust(min(max(8, max_len + 2), 50))}{"FREQ".ljust(10)}{"XTX DETERM".ljust(20)}{"minv(XTX) MIN ELEM".ljust(20)}{"minv(XTX) MAX ELEM".ljust(20)}')

        # GET RESULTS FOR ENTIRE OBJECT:
        determ, min_elem, max_elem = xtxd.XTX_determinant(DATA_AS_ARRAY_OR_SPARSEDICT=CYCLE_OBJECT_XTX, name=name,
                                          module=self.this_module, fxn=fxn)
        print(f'{str(name).ljust(min(max(8, max_len + 2), 50))}' + \
              f'{str(len(CYCLE_OBJECT[0])).ljust(10)}' + \
              f'{determ:.10g}'.ljust(20) + \
              f'{min_elem:.10g}'.ljust(20) + \
              f'{max_elem:.10g}'.ljust(20)
              )

        del CYCLE_OBJECT_XTX

        # PRINT HEADER FOR COLUMN DROP CYCLES:
        print(f'\nEFFECTS OF REMOVING A COLUMN ON DETERMINANT:')
        print(
            f'{"CATEGORY".ljust(min(max(8, max_len + 2), 50))}{"FREQ".ljust(10)}{"XTX DETERM".ljust(20)}{"minv(XTX) MIN ELEM".ljust(20)}{"minv(XTX) MAX ELEM".ljust(20)}')
        # GET RESULTS FOR OBJECT W CYCLED COLUMN DROPS:
        for col_idx in range(len(CYCLE_OBJECT)):

            if self.is_list():
                POPPED_COL = CYCLE_OBJECT[col_idx].copy()
                CYCLE_OBJECT_WIP = np.delete(CYCLE_OBJECT.copy(), col_idx, axis=0)
                CYCLE_OBJECT_WIP_XTX = np.matmul(CYCLE_OBJECT_WIP, CYCLE_OBJECT_WIP.transpose())
            elif self.is_dict():
                POPPED_COL = sd.unzip_to_ndarray({0:CYCLE_OBJECT[col_idx]})[0]
                CYCLE_OBJECT_WIP = sd.delete_outer_key(deepcopy(CYCLE_OBJECT), [col_idx])[0]
                CYCLE_OBJECT_WIP_XTX = sd.sparse_AAT(CYCLE_OBJECT_WIP, return_as='ARRAY')

            determ, min_elem, max_elem = xtxd.XTX_determinant(DATA_AS_ARRAY_OR_SPARSEDICT=CYCLE_OBJECT_WIP_XTX, name=name,
                                              module=self.this_module, fxn=fxn)

            _ = vot.ValidateObjectType(POPPED_COL).ml_package_object_type()  # GET TYPE FOR COLUMN REMOVED

            if _ == ['FLOAT', 'INT']: freq_text = f'{str(len(POPPED_COL))}'.ljust(10)  # FOR FLOAT OR INT RETURN TOTAL CT
            elif _ in ['BIN']: freq_text = f'{str(np.sum(POPPED_COL))}'.ljust(10)  # FOR BIN RETURN CT OF "1"s
            else: raise TypeError(f'INVALID DATATYPE {_} IN PreRunDataAugment.column_drop_iterator().')

            print(f'{str(NEW_HEADER[col_idx]).ljust(min(max(8, max_len + 2), 50))}' + \
                  freq_text + \
                  f'{determ:.10g}'.ljust(20) + \
                  f'{min_elem:.10g}'.ljust(20) + \
                  f'{max_elem:.10g}'.ljust(20)
                  )
        del CYCLE_OBJECT_WIP, CYCLE_OBJECT_WIP_XTX, POPPED_COL


    def set_start_end_lags(self, start_or_end):
        # (set start or end lags for float columns in DATA)
        # FIND OUT IF EACH OBJECT IS SET THE SAME OR UNIQUELY VALUED --- ALL ZEROES AT START, INDICATES IF THE
        # USER HAS ENTERED SOMETHING
        start_or_end = start_or_end.upper()
        if start_or_end == 'START': LAG_WIP = self.START_LAG
        elif start_or_end == 'END': LAG_WIP = self.END_LAG
        else: raise ValueError(f'\n*** INVALID start_or_end in PreRunDataAugment, set_start_end_lags(). ***\n')

        if len(ru.return_uniques([LAG_WIP[_] for _ in range(len(LAG_WIP)) if self.MODIFIED_DATATYPES[0][_] in ['FLOAT','INT']],
                                 [], 'STR', suppress_print='Y')[0]) == 1:
            print(f'\nALL {start_or_end} LAGS ARE CURRENTLY SET TO {LAG_WIP[0]}.')
        else:
            print(f'\n{start_or_end} LAGS ARE UNIQUELY VALUED.')

        LAG_CMDS = ['accept changes and exit(a)',
                    f'set all {start_or_end} lags for DATA the same(b)',
                    f'set {start_or_end} lags for a specific column(s)(d)',
                    'reset(e)',
                    'abort and exit without saving changes(f)',
                    f'enter {start_or_end} LAGS based on {["END" if start_or_end=="START" else "START"][0]} LAG entries(g)',
                    'print column names & setup parameters(q)'
                    ]

        lag_cmds_str = 'ABDEFQ'
        # IF USER HAS ALREADY ENTERED SOME LAGS, ALLOW USER TO GO THRU AND ENTER OTHER LAGS FOR JUST THOSE
        if True in map(lambda x: x != 0, LAG_WIP): lag_cmds_str = 'ABDEFGQ'

        user_lag = 'BYPASS'  # DONT DELETE THIS HAVE TO BYPASS ALL TO GET TO PROMPT ON FIRST PASS

        while True:
            # 4-4-21 THE ONLY POSNS THAT ACTUALLY MATTER ARE THOSE WHERE MODIFIED TYPE IS 'FLOAT' OR 'INT', OTHER POSNS IN
            # LAGS CAN BE SET TO ANYTHING AND WILL NEVER BE READ, LAG APPLIER FXN WILL ONLY
            # LOOK AT POSNS WHERE MODIFIED TYPE IS 'FLOAT' OR 'INT'  (NOT 'BIN' OR 'STR')

            if user_lag in 'B':  # 'set all lags for DATA the same(b),

                while True:
                    start_lag = vui.validate_user_int(f'\nEnter {start_or_end} LAG for all columns > ', min=0, max=len(self.DATA_OBJECT_WIP[0]))
                    if vui.validate_user_str(f'\nUser entered {start_lag}, accept? (y/n) > ', 'YN') == 'Y':
                        for col_idx in range(len(self.DATA_OBJECT_WIP)):
                            if self.MODIFIED_DATATYPES[0][col_idx] in ['FLOAT', 'INT']:
                                LAG_WIP[col_idx] = int(start_lag)
                            else:
                                LAG_WIP[col_idx] = ''
                        break

            elif user_lag in 'D':  # 'set lag for specific column(s) (d)',

                while True:
                    NUM_IDXS = [_ for _ in range(len(self.DATA_OBJECT_WIP)) if self.MODIFIED_DATATYPES[0][_] in ['FLOAT', 'INT']]
                    COL_IDXS = ls.list_custom_select([self.DATA_OBJECT_HEADER_WIP[0][_] for _ in NUM_IDXS], 'idx')

                    COLUMN_NAMES = ", ".join([self.DATA_OBJECT_HEADER_WIP[0][_] for _ in [NUM_IDXS[__] for __ in COL_IDXS]])
                    lag = vui.validate_user_int(f'\nEnter {start_or_end} LAG for {COLUMN_NAMES} > ', min=0)
                    if vui.validate_user_str(
                            f'User entered {lag} as {start_or_end} LAG for {COLUMN_NAMES}, accept? (y/n) > ', 'YN') == 'Y':
                        for col_idx in [NUM_IDXS[__] for __ in COL_IDXS]:
                            LAG_WIP[col_idx] = int(lag)
                        break

            elif user_lag in 'EF':  # 'reset(e)',   'abort and exit without saving changes(f)'
                if start_or_end == 'START': LAG_WIP = self.START_LAG
                elif start_or_end == 'END': LAG_WIP = self.END_LAG
                print(f'\n{start_or_end} LAGS HAVE BEEN RESET.')
                if user_lag in 'F':
                    break

            elif user_lag in 'G':  # 'enter lags based on lags already entered(g)'
                if start_or_end == 'START': LAG_WIP2 = self.END_LAG
                elif start_or_end == 'END': LAG_WIP2 = self.START_LAG
                for lag_idx in range(len(LAG_WIP)):
                    if LAG_WIP2[lag_idx] not in [0, '']:
                        LAG_WIP[lag_idx] = int(vui.validate_user_int(
                            f'Enter {start_or_end} LAG for {self.DATA_OBJECT_HEADER_WIP[0][lag_idx]} > ', min=0))

                del LAG_WIP2

            elif user_lag in 'Q':  # 'print column names & setup parameters(q)'
                self.print_cols_and_setup_parameters()

            elif user_lag in 'A':
                if np.array_equal(self.START_LAG, self.START_LAG_UNDO) and np.array_equal(self.END_LAG, self.END_LAG_UNDO):
                    self.pending_lags = True
                break

            ppro.SelectionsPrint(LAG_CMDS, lag_cmds_str, append_ct_limit=3)

            user_lag = vui.validate_user_str(' > ', lag_cmds_str)

        return LAG_WIP


    def undo(self):
        try:
            self.DATA_OBJECT_WIP = self.DATA_OBJECT_WIP_UNDO.copy() if self.is_list() else deepcopy(self.DATA_OBJECT_WIP_UNDO)
            self.DATA_OBJECT_HEADER_WIP = deepcopy(self.DATA_OBJECT_HEADER_WIP_UNDO)
            self.WORKING_SUPOBJS = [_.copy() for _ in self.VALIDATED_DATATYPES_UNDO]
            self.CONTEXT = deepcopy(self.CONTEXT_UNDO)
            self.KEEP = deepcopy(self.KEEP_UNDO)
            print(f'\nUNDO - {self.UNDO_DESC.upper()} - COMPLETE.\n')

        except:
            if self.allow_undo is False:
                print(f"\nCAN'T UNDO, UNDO IS DISABLED.")
                if vui.validate_user_str(f"ENABLE UNDO GOING FORWARD, UNDERSTANDING THE IMPLICATIONS FOR RAM (y/n) > ", "YN") == 'Y':
                    self.allow_undo = True

            elif self.allow_undo is True:
                print(f"\nCAN'T UNDO, BACKUP OBJECTS HAVE NOT BEEN GENERATED YET.\n")


    def return_fxn(self):

        # BEAR DONT FORGET TO SEE IF HAVE TO RECOMPILE SWNL FROM DATA & TARGET
        SUPER_WORKING_NUMPY_LIST = [self.DATA, self.TARGET_OBJECT, self.REFVECS]

        return SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS, self.CONTEXT, self.KEEP


    def config_run(self):

        fxn = inspect.stack()[0][3]

        self.user_manual_or_std = 'E'
        pending_lags = False

        while True:
            while True:
                # MAKE THIS FIRST TO APPLY UNDO B4 RESETTING UNDO OBJECTS BELOW
                if self.user_manual_or_std == 'U':  # 'undo last operation(u)'
                    self.undo()

                # SET THE INITIAL STATE OF EVERY OBJECT, FOR UNDO PURPOSES
                # DOING IT DEEPCOPY WAY TO PREVENT THE INTERACTION OF ASSIGNMENTS W ORIGINAL SOURCE OBJECTS
                # 12-12-21 10:49 AM ONLY RESET UNDO AFTER AN OPERATION, NOT PRINTS, ALLOWS USER TO LOOK AT OBJECTS
                # AND THEN DECIDE TO DO AN UNDO
                if self.allow_undo and \
                    True in [f'({self.user_manual_or_std})'.lower() in _ for _ in self.augment_menu_commands()]:
                    self.DATA_OBJECT_WIP_UNDO = self.DATA_OBJECT_WIP.copy() if self.is_list() else deepcopy(self.DATA_OBJECT_WIP)

                    self.DATA_OBJECT_HEADER_WIP_UNDO = deepcopy(self.DATA_OBJECT_HEADER_WIP)
                    self.WORKING_SUPOBJS_UNDO = [_.copy() for _ in self.WORKING_SUPOBJS]
                    self.CONTEXT_UNDO = deepcopy(self.CONTEXT)
                    self.KEEP_UNDO = deepcopy(self.KEEP)

                    if self.user_manual_or_std in self.augment_str:  # ANYTIME AUGMENT CMD IS USED, RECORD
                        self.UNDO_DESC = self.MASTER_DICT[self.user_manual_or_std.lower()]
                    else: self.UNDO_DESC = f'\nUNDO NOT AVAILABLE\n'


                # GENERIC MENU COMMANDS #######################################################################################
                if self.user_manual_or_std == 'B':  # '(b)get size & sparsity'
                    type_ = 'list' if self.is_list() else 'sparse dict'
                    rows_, cols_ = gs.get_shape('DATA', self.DATA_OBJECT_WIP, self.data_given_orientation)
                    sparsity = sd.list_sparsity(self.DATA_OBJECT_WIP) if self.is_list() else sd.sparsity(self.DATA_OBJECT_WIP)

                    print(f'\nDATA is a {type_}, has {cols_} columns and {rows_} rows (not counting header) and is {sparsity}% sparse.\n')

                elif self.user_manual_or_std == 'C':  # '(c)calculate column intercorrelations'

                    print(f'\nCalculating column intercorrelations. Patience... \n')

                    RESULTS = np.empty((3,0), dtype=object) #  # COLUMN NAMES AND RSQ HOLDER
                    ERROR_HOLDER = np.empty((2,0), dtype=object)
                    __ = self.DATA_OBJECT_WIP
                    for col_idx in range(len(__)-1):
                        for col_idx2 in range(col_idx + 1, len(__)):
                            np.seterr(divide='ignore', invalid='ignore')
                            if self.is_list():
                                rsq = np.corrcoef(__[col_idx].astype(float), __[col_idx2].astype(float))[0][1] ** 2
                            elif self.is_dict():
                                rsq = sd.rsq_({0:__[col_idx]}, {0:__[col_idx2]})

                            COLUMNS_HOLDER = np.array([ self.DATA_OBJECT_HEADER_WIP[0][col_idx],
                                                      self.DATA_OBJECT_HEADER_WIP[0][col_idx2] ], dtype=object)

                            if float(rsq) >= 0 and float(rsq) <= 1:
                                RESULTS = np.insert(RESULTS,
                                                   len(RESULTS[0]),
                                                   np.insert(COLUMNS_HOLDER, 2, rsq),
                                                   axis=1
                                )
                            else:
                                RESULTS = np.insert(RESULTS,
                                                   len(RESULTS[0]),
                                                   np.insert(COLUMNS_HOLDER, 2, 0),
                                                   axis=1
                                )
                                ERROR_HOLDER = np.insert(ERROR_HOLDER,
                                                    len(ERROR_HOLDER[0]),
                                                    self.DATA_OBJECT_HEADER_WIP[0][col_idx],
                                                    axis=1
                                )

                    ARGSORT_RSQ_DESC = np.flip(np.argsort(RESULTS[2]))

                    rsq_cutoff = vui.validate_user_float(f'Report R-squared greater than > ', min=0, max=1)

                    column_width = np.min([50, np.max([10, *list(map(len, self.DATA_OBJECT_HEADER_WIP[0]))])]) + 2
                    print(f"{'COLUMN 1'.ljust(column_width)}{'COLUMN 2'.ljust(column_width)}{'RSQ'.ljust(12)}")
                    _format = lambda col_idx, row_idx, dum_width: str(RESULTS[col_idx][ARGSORT_RSQ_DESC[row_idx]]).ljust(dum_width)
                    for row_idx in range(len(ARGSORT_RSQ_DESC)):
                        if RESULTS[2][ARGSORT_RSQ_DESC[row_idx]] >= rsq_cutoff:
                            print(
                                f"{_format(0, row_idx, column_width)}{_format(1, row_idx, column_width)}{_format(2, row_idx, 12)[:10]}")
                        else:
                            break

                    if len(ERROR_HOLDER[0]) > 0:  # IF ANY ERRORS, REPORT
                        print(f'\nCOLUMNS GIVING ERRORS:')
                        print(f"{'COLUMN 1'.ljust(column_width)}{'COLUMN 2'.ljust(column_width)}")
                        for _ in range(len(ERROR_HOLDER[0])):
                            print(
                                f"{str(ERROR_HOLDER[0][_]).ljust(column_width)}{str(ERROR_HOLDER[1][_]).ljust(column_width)}")

                    del RESULTS, ERROR_HOLDER, COLUMNS_HOLDER, ARGSORT_RSQ_DESC

                elif self.user_manual_or_std in 'F':     # '(f)float check'
                    print(f'\nPerforming float conversion check in DATA and TARGET...\n')

                    EXCEPTION_HOLDER = []
                    def float_check(OBJECT, OBJECT_HEADER):
                        is_except = False
                        for col_idx in range(len(OBJECT)):
                            print(f'Checking {OBJECT_HEADER[0][col_idx]}...')
                            for row_idx in range(len(OBJECT[col_idx])):
                                # CANT USE is_list() & is_dict() HERE BECAUSE OF TARGET
                                if isinstance(OBJECT, (list, tuple, np.ndarray)): _ = OBJECT[col_idx][row_idx]
                                elif isinstance(OBJECT, dict): _ = OBJECT[col_idx].get(row_idx, 0)
                                try:
                                    __ = float(_)
                                except:
                                    EXCEPTION_HOLDER.append(f'{OBJECT_HEADER[0][col_idx]}, ' f'{_}')
                                    is_except = True

                                if is_except:
                                    print(' ' * 5 + f'Unable to convert {OBJECT_HEADER[0][col_idx]}, row index {row_idx}, '
                                              f'{_}, to a float')
                        print()

                    float_check(self.DATA_OBJECT_WIP, self.DATA_OBJECT_HEADER_WIP)
                    float_check(self.TARGET_OBJECT, self.TARGET_OBJECT_HEADER)

                    if len(EXCEPTION_HOLDER) > 0:
                        print(f'\nEXCEPTIONS TO FLOAT CONVERSION:')
                        [print(_) for _ in EXCEPTION_HOLDER]
                    else:
                        print(f'\nALL FLOAT CONVERSIONS OK!')

                    del EXCEPTION_HOLDER

                elif self.user_manual_or_std == 'G':  # (g)turn undo functionality ON/OFF
                    if self.allow_undo: self.allow_undo = False
                    elif not self.allow_undo: self.allow_undo = True
                    print(f'\n*** ALLOW UNDO FUNCTIONIALITY TURNED {["ON" if self.allow_undo else "OFF"][0]} ***\n')

                elif self.user_manual_or_std == 'M':  # '(m)multicolinearity iterator on BIG MATRIX'
                    self.column_drop_iterator(self.DATA_OBJECT_WIP, 'BIG MATRIX', self.DATA_OBJECT_HEADER_WIP[0],
                                             append_ones='N')

                elif self.user_manual_or_std in 'P':  # '(p)print_preview data object'
                    self.print_preview(20, 10)


                elif self.user_manual_or_std in 'R':  # '(r)reset and start over'

                    if self.allow_reset:
                        try:
                            if self.is_list(): self.DATA_OBJECT_WIP = self.DATA_OBJECT_WIP_BACKUP.copy()
                            elif self.is_dict(): self.DATA_OBJECT_WIP = deepcopy(self.DATA_OBJECT_WIP_BACKUP)
                            self.DATA_OBJECT_HEADER_WIP = deepcopy(self.DATA_OBJECT_HEADER_WIP_BACKUP)
                            self.WORKING_SUPOBJS = [_.copy() for _ in self.WORKING_SUPOBJS_BACKUP]
                            self.CONTEXT = deepcopy(self.CONTEXT_BACKUP)
                            self.KEEP = deepcopy(self.KEEP_BACKUP)

                            self.user_manual_or_std = 'E'  # SET TO "E" TO GET CAUGHT IMMEDIATELY AFTER RESET BY "E"
                            print(f'*** OBJECTS SUCCESSFULLY RESTORED TO INITIAL STATE ***\n')
                            continue
                        except:
                            print(f'\n*** ERROR TRYING TO RESTORE WIP OBJECTS BACK TO INITIAL STATE ***')
                            print(f'*** OBJECTS HAVE NOT BEEN RESTORED TO INITIAL STATE ***\n')
                    elif not self.allow_reset:
                        print(f'\n*** RESET FUNCTIONALITY IS DISABLED, UNABLE TO RESET. ***\n')


                elif self.user_manual_or_std == 'Q':   #  '(q)print column names & setup info'
                    self.print_cols_and_setup_parameters()


                elif self.user_manual_or_std == 'T':  # '(t)tests inverse of XTX'
                    # 11/27/22 BELIEVE DATA IS [ [] = COLUMN ] HERE, SO INSTEAD OF TRANSPOSE, JUST MAKE XTX HERE
                    if self.is_list: DUM_XTX = np.matmul(self.DATA_OBJECT_WIP, self.DATA_OBJECT_WIP.transpose())
                    elif self.is_dict: DUM_XTX = sd.sparse_AAT(self.DATA_OBJECT_WIP, return_as='ARRAY')
                    xtxd.XTX_determinant(XTX_AS_ARRAY_OR_SPARSEDICT=DUM_XTX, name='DATA_OBJECT_WIP',
                         module=self.this_module, fxn='(t)tests inverse of XTX', print_to_screen=True)
                    del DUM_XTX

                elif self.user_manual_or_std == 'Y':   # '(y)convert DATA to sparse dict / lists'
                    if self.is_dict(): self.DATA_OBJECT_WIP = sd.unzip_to_ndarray(self.DATA_OBJECT_WIP)[0]
                    elif self.is_list(): self.DATA_OBJECT_WIP = sd.zip_list(self.DATA_OBJECT_WIP)
                    print(f'\nDATA successfully converted to {"sparse dict" if self.is_dict() else "np array"}.\n')


                # END GENERIC MENU COMMANDS #######################################################################################

                # AUGMENT MENU COMMANDS ###################################################################################################

                elif self.user_manual_or_std in 'D':  # '(d)delete column'

                    while True:
                        col_idx = ls.list_single_select(self.DATA_OBJECT_HEADER_WIP[0], f'Select column from DATA', 'idx')[0]
                        # IF COLUMN TO DELETE IS IN ORIGINALS, REQUIRE REASON & APPEND TO CONTEXT
                        if self.DATA_OBJECT_HEADER_WIP[0][col_idx] in self.DATA_OBJECT_HEADER_ORIGINALS:
                            delete_text = f'Deleted {self.DATA_OBJECT_HEADER_WIP[0][col_idx]} for '
                            delete_reason = input(delete_text + '(give reason) > ')
                            while True:
                                print(f'User entered "{delete_text + delete_reason}" ... Accept? (y/n) > ', 'YN')
                                if vui.validate_user_str('> ', 'YN') == 'Y':
                                    final_str = f' and recording "{delete_text} {delete_reason}"'
                                    break
                        else:
                            final_str = f''

                        print(f'Deleting {self.DATA_OBJECT_HEADER_WIP[0][col_idx]}{final_str} ' + \
                              f'... Accept(a), abandon(b), try again(c) > ')
                        __ = vui.validate_user_str(' > ', 'ABC')
                        if __ == 'A': pass
                        elif __ == 'B': break
                        elif __ == 'C': continue

                        if self.DATA_OBJECT_HEADER_WIP[0][col_idx] in self.DATA_OBJECT_HEADER_ORIGINALS:
                            self.CONTEXT.append(f'{delete_text} + {delete_reason}')

                        self.delete_column(col_idx)
                        break

                elif self.user_manual_or_std in 'I':  # '(i)append intercept to data'
                    while True:
                        # BEAR THIS WILL PROBABLY BECOME intercept_manager
                        if len(mlfc.ML_find_constants(self.DATA_OBJECT_WIP, self.data_given_orientation)[0]) > 0:
                            print(f'\n*** INTERCEPT has already been appended to DATA! *** \n ')
                            break
                        else:
                            # BEAR, NOW THAT MLObject.insert_standard_intercept() IS HANDLING THIS AND CONTEXT UPDATE, SEE IF THIS
                            # CAN COME OUT!
                            # DONT TAKE THIS OUT YET
                            self.intercept_verbage()

                            InterceptClass = mlo.MLObject(
                                                          self.DATA_OBJECT_WIP,
                                                          self.data_given_orientation,
                                                          name="DATA",
                                                          return_orientation='AS_GIVEN',
                                                          return_format='AS_GIVEN',
                                                          bypass_validation=self.bypass_validation,
                                                          calling_module=self.this_module,
                                                          calling_fxn=fxn
                            )

                            InterceptClass.insert_standard_intercept(
                                                                     HEADER_OR_FULL_SUPOBJ=self.WORKING_SUPOBJS[0],
                                                                     CONTEXT=self.CONTEXT
                            )

                            self.DATA_OBJECT_WIP = InterceptClass.OBJECT
                            self.WORKING_SUPOBJS[0] = InterceptClass.HEADER_OR_FULL_SUPOBJ
                            self.CONTEXT = InterceptClass.CONTEXT
                            del InterceptClass

                            print(f'\nINTERCEPT COLUMN SUCCESSFULLY APPENDED TO DATA.\n')
                            break

                elif self.user_manual_or_std == 'L':     #  '(l)apply lag'

                    # 9/30/22 BEAR THIS NEEDS A LOT OF WORK

                    if len(self.START_LAG) != len(self.END_LAG):
                        print(f'\n*** START AND END LAG VECTORS HAVE DIFFERENT LENGTHS. ***\n')

                    start_greater_than_end = False
                    lag_incongruity = False
                    while True:
                        for col_idx in range(len(self.START_LAG)):
                            try:
                                if float(self.START_LAG[col_idx]) > float(self.END_LAG[col_idx]):
                                    start_greater_than_end = True
                                    break
                            except: pass

                            if type(self.START_LAG[col_idx]) != type(self.END_LAG[col_idx]):
                                lag_incongruity = True
                                break

                        if start_greater_than_end:
                            print(f'\n*** START LAG IS GREATER THAN END LAG.  FIX LAGS. ***\n')
                            break

                        if lag_incongruity:
                            print(f'\n*** THERE IS INCONGRUITY OF ENTRIES BETWEEN START LAG AND END LAG.  FIX LAGS. ***\n')
                            break

                        # 4-8-22 DONT NEED ANY OBJ/COL ROW INPUTS HERE, THIS SIMPLY APPLYING INPUTS SET IN "START_LAG" & "END_LAG"
                        # TO "DATA" OBJECT, ITS HEADER, AND ASSOCIATED HYPERPARAMETER OBJECTS

                        print(f'\nLAG WILL BE APPLIED AS FOLLOWS:')
                        self.print_cols_and_setup_parameters()
                        if vui.validate_user_str(f'Accept? (y/n) > ', 'YN') == 'N':
                            break

                        print(f'\nAPPLYING LAG....\n')

                        # UNNUMPY DATA & HEADER TO HANDLE RAGGEDNESS ########################################################################
                        if self.is_list():
                            self.DATA_OBJECT_WIP = list(map(list, self.DATA_OBJECT_WIP))
                            self.DATA_OBJECT_HEADER_WIP = list(map(list, self.DATA_OBJECT_HEADER_WIP))
                        if self.is_dict():
                            _inner_len = sd.inner_len(self.DATA_OBJECT_WIP)

                        # END UNNUMPY DATA & HEADER ########################################################################

                        for col_idx in range(len(self.DATA_OBJECT_WIP) - 1, -1, -1):  # ITERATE THRU OBJ BACKWARDS
                            sl = self.START_LAG[col_idx]
                            el = self.END_LAG[col_idx]

                            if not isinstance(sl, int):
                                continue  # IF NO LAG, SKIP COLUMN
                            else:
                                for lag_period in range(el, sl-1, -1):

                                    if self.is_list():
                                        self.DATA_OBJECT_WIP.insert(col_idx+1, deepcopy(self.DATA_OBJECT_WIP[col_idx])[lag_period:])

                                    elif self.is_dict():
                                        # GO BACKWARDS UP TO (BUT NOT INCLUDING) col_idx AND ADD 1 TO ALL OUTER_KEYS
                                        for col_idx2 in range(len(self.DATA_OBJECT_WIP) - 1, col_idx, -1):
                                            self.DATA_OBJECT_WIP[col_idx2 + 1] = self.DATA_OBJECT_WIP.pop(col_idx2)
                                        # END RE-NUMBER OF OUTER_KEYS AFTER INSERTED LAG
                                        xx = deepcopy(self.DATA_OBJECT_WIP)
                                        self.DATA_OBJECT_WIP = sd.insert_row(
                                            xx,
                                            col_idx+1,
                                            {_-lag_period:xx[col_idx][_] for _ in xx[col_idx] if _ >= lag_period} | {_inner_len-1:0}
                                        )

                                    self.DATA_OBJECT_HEADER_WIP = np.insert(
                                        self.DATA_OBJECT_HEADER_WIP, col_idx+1, self.DATA_OBJECT_HEADER_WIP[0][col_idx]+f'LAG{lag_period}', axis=1)
                                    '''
                                    # 4-13-22 THESE ARE HERE TO INSERT DUMS INTO DATA_WIP & HEAD_WIP TO ALLOW FOR USE OF
                                    # delete_column BELOW, OTHERWISE WOULD HAVE TO DO A SEPARATE np.delete FOR THE 8 OTHER
                                    # OBJECTS BELOW ###############################################################################
                                    self.DATA_OBJECT_WIP.insert(col_idx+1, deepcopy(self.DATA_OBJECT_WIP[col_idx]))
                                    self.DATA_OBJECT_HEADER_WIP.insert(col_idx+1, deepcopy(self.DATA_OBJECT_HEADER_WIP[0][col_idx]))
                                    # END DUMS FOR np.delete #########################################################################
                                    '''
                                    self.VALIDATED_DATATYPES[0] = np.insert(self.VALIDATED_DATATYPES[0], col_idx+1, deepcopy(self.VALIDATED_DATATYPES[0][col_idx]), axis=0)
                                    self.MODIFIED_DATATYPES[0] = np.insert(self.MODIFIED_DATATYPES[0], col_idx+1, deepcopy(self.MODIFIED_DATATYPES[0][col_idx]), axis=0)
                                    self.FILTERING[0] = np.insert(self.FILTERING[0], col_idx+1, deepcopy(self.FILTERING[0][col_idx]), axis=0)
                                    self.MIN_CUTOFFS[0] = np.insert(self.MIN_CUTOFFS[0], col_idx+1, deepcopy(self.MIN_CUTOFFS[0][col_idx]), axis=0)
                                    self.USE_OTHER[0] = np.insert(self.USE_OTHER[0], col_idx+1, deepcopy(self.USE_OTHER[0][col_idx]), axis=0)
                                    self.START_LAG = np.insert(self.START_LAG, col_idx + 1, lag_period, axis=0)
                                    self.END_LAG = np.insert(self.END_LAG, col_idx + 1, lag_period, axis=0)
                                    self.SCALING = np.insert(self.SCALING, col_idx+1, self.SCALING[col_idx], axis=0)

                                self.delete_column(col_idx)

                        if self.is_list():
                            # 4-13-22 CONVERT BACK TO NUMPY IF NOT SPARSE_DICTS
                            self.DATA_OBJECT_WIP = np.fromiter(map(np.ndarray, self.DATA_OBJECT_WIP), dtype=object)
                            self.DATA_OBJECT_HEADER_WIP = np.fromiter(map(np.ndarray, self.DATA_OBJECT_HEADER_WIP), dtype=str)

                        # 4-11-22 AFTER EXPANDING OUT LAG COLUMNS, ASK USER TO "SQUARE-UP" COLUMN LENGTHS OR NOT
                        squaredup = vui.validate_user_str(f'Square up DATA to shortest column length(s) or leave ragged(r) > ', 'SR')
                        if squaredup == 'S':
                            if self.is_list():
                                shortest_column = np.min([len(_) for _ in self.DATA_OBJECT_WIP])
                                for __ in range(len(self.DATA_OBJECT_WIP)):
                                    self.DATA_OBJECT_WIP[__] = self.DATA_OBJECT_WIP[__][:shortest_column]
                            elif self.is_dict():
                                new_inner_len = int(_inner_len) - int(np.max(self.END_LAG))
                                self.DATA_OBJECT_WIP = sd.resize_inner(self.DATA_OBJECT_WIP, new_inner_len)

                        print(f'\nLAG COMPLETED SUCCESSFULLY.\n')

                        empty_column = False
                        for _ in self.DATA_OBJECT_WIP:
                            if self.is_list() and _ == []:
                                empty_column = True
                            elif self.is_dict():
                                if list(self.DATA_OBJECT_WIP[_].getitems())[0] == (new_inner_len, 0):
                                    empty_column = True
                            if empty_column:
                                print(f'\n*** LAG HAS LEFT A DATA COLUMN EMPTY.  UNDO IS RECOMMENDED ***\n ')
                                input(f'Hit Enter to continue > ')
                                break

                        pending_lags = False

                        break

                elif self.user_manual_or_std == 'N':  # '(n)add interactions'
                    print(f'\n*** NOT DONE YET :( ***\n')
                    pass
                    # self.DATA_OBJECT_WIP, self.WORKING_SUPOBJS[0] = \
                    #     mlai.MLAppendInteractions(
                    #
                    #     )

                elif self.user_manual_or_std == 'S':  # '(s)standardize / normalize'

                    SNClass = mlsn.StandardizeNormalize(self.DATA_OBJECT_WIP, self.WORKING_SUPOBJS[0],
                                                    self.data_given_orientation, bypass_validation=self.bypass_validation)

                    self.DATA_OBJECT_WIP = SNClass.DATA_OBJECT

                    del SNClass


                elif self.user_manual_or_std == '1':  # '(1)add moving average'
                    print(f'\n ***MOVING AVERAGE NOT AVAILABLE YET :( ***\n')

                elif self.user_manual_or_std == '2':  # '(2)add forward derivative'
                    print(f'\n ***FORWARD DERIVATIVE NOT AVAILABLE YET :( ***\n')

                elif self.user_manual_or_std == '3':  # '(3)add trailing derivative'
                    print(f'\n ***TRAILING DERIVATIVE NOT AVAILABLE YET :( ***\n')

                elif self.user_manual_or_std == '4':  # '(4)add centered derivative'
                    print(f'\n ***CENTERED DERIVATIVE NOT AVAILABLE YET :( ***\n')

                elif self.user_manual_or_std == '5':  # '(5)add % change from previous'
                    print(f'\n ***% CHANGE FROM PREVIOUS NOT AVAILABLE YET :( ***\n')

                elif self.user_manual_or_std == '9':  # '(9)set lag start'
                    self.START_LAG = self.set_start_end_lags('start')

                elif self.user_manual_or_std == '0':  # '(0)set lag end'
                    self.END_LAG = self.set_start_end_lags('end')

                # END AUGMENT MENU COMMANDS ###################################################################################################

                # HIDDEN MENU COMMANDS #####################################################################################
                elif self.user_manual_or_std == '!':  # '(!)print DATA_OBJECT_HEADER_WIP
                    print(f'\nCOLUMN HEADERS')
                    [print(_) for _ in self.DATA_OBJECT_HEADER_WIP[0]]

                elif self.user_manual_or_std == '@':  # '(@)print VALIDATED_DATATYPES
                    print(f'\nVALIDATED DATATYPES')
                    max_len = np.max([len(_) for _ in self.DATA_OBJECT_HEADER_WIP[0]+self.TARGET_OBJECT_HEADER[0]])
                    [print(
                        f'{str(self.DATA_OBJECT_HEADER_WIP[0][_][:50]).ljust(min(max_len + 5, 50))}{self.VALIDATED_DATATYPES[0][_]}')
                     for _ in range(len(self.VALIDATED_DATATYPES[0]))]
                    print()
                    [print(
                        f'{str(self.TARGET_OBJECT_HEADER[0][_][:50]).ljust(min(max_len + 5, 50))}{self.VALIDATED_DATATYPES[2][_]}')
                     for _ in range(len(self.VALIDATED_DATATYPES[2]))]

                elif self.user_manual_or_std == '#':  # '(#)print MODIFIED_DATATYPES
                    print(f'\nMODIFIED DATATYPES')
                    max_len = np.max([len(_) for _ in self.DATA_OBJECT_HEADER_WIP[0]+self.TARGET_OBJECT_HEADER])
                    [print(
                        f'{str(self.DATA_OBJECT_HEADER_WIP[0][_][:50]).ljust(min(max_len + 5, 50))}{self.MODIFIED_DATATYPES[0][_]}')
                        for _ in range(len(self.MODIFIED_DATATYPES[0]))]
                    print()
                    [print(
                        f'{str(self.TARGET_OBJECT_HEADER[0][_][:50]).ljust(min(max_len + 5, 50))}{self.MODIFIED_DATATYPES[2][_]}')
                        for _ in range(len(self.MODIFIED_DATATYPES[2]))]

                elif self.user_manual_or_std == '$':  # '($)show MIN_CUTOFFS'
                    oi.obj_info(self.MIN_CUTOFFS, 'MIN_CUTOFFS', __name__)

                elif self.user_manual_or_std == '%':  # '(%)show USE_OTHER'
                    oi.obj_info(self.USE_OTHER, 'USE_OTHER', __name__)

                elif self.user_manual_or_std == '^':  # '(^)show CONTEXT'
                    if len(self.CONTEXT) == 0: print(f'\n*** CONTEXT IS EMPTY.*** \n')
                    else: oi.obj_info(self.CONTEXT, 'CONTEXT', __name__)

                elif self.user_manual_or_std == '+':  # '(+)show KEEP'
                    oi.obj_info(self.KEEP, 'KEEP', __name__)

                elif self.user_manual_or_std == '&':  # '(&)show SCALING'
                    oi.obj_info(self.SCALING, 'SCALING', __name__)

                elif self.user_manual_or_std == '*':  # '(*)show LAG'
                    oi.obj_info(self.START_LAG, 'LAG START', __name__)
                    oi.obj_info(self.END_LAG, 'LAG END', __name__)

                elif self.user_manual_or_std == '_':  # '(_)show FILTERING'
                    oi.obj_info(self.FILTERING, 'FILTERING', __name__)

                # END HIDDEN MENU COMMANDS #####################################################################################

                elif self.user_manual_or_std == 'A':  # 'accept / continue(a)'

                    break  # BREAK OUT OF COMMAND ENTRY LOOP

                ppro.SelectionsPrint(self.generic_menu_commands(), self.generic_str, append_ct_limit=3,
                                     max_len=self.max_cmd_len)
                ppro.SelectionsPrint(self.augment_menu_commands(), self.augment_str, append_ct_limit=3,
                                     max_len=self.max_cmd_len)
                ppro.SelectionsPrint(self.hidden_menu_commands(), self.hidden_str, append_ct_limit=3,
                                     max_len=self.max_cmd_len)

                if pending_lags: print(f'\n *** THERE ARE PENDING LAGS TO BE APPLIED *** \n')

                self.user_manual_or_std = vui.validate_user_str(' > ', self.allowed_commands_string)

            if vui.validate_user_str(f'\nAccept data augmentation? (y/n) > ', 'YN') == 'Y':
                self.DATA_OBJECT = self.DATA_OBJECT_WIP
                self.DATA_OBJECT_HEADER = np.array(self.DATA_OBJECT_HEADER_WIP, dtype=str)
                # CLEAR BACKUP AND UNDO OBJECTS
                del self.DATA_OBJECT_WIP, self.DATA_OBJECT_HEADER_WIP

                try: del self.DATA_OBJECT_WIP_BACKUP, self.DATA_OBJECT_HEADER_WIP_BACKUP, self.WORKING_SUPOBJS_BACKUP, \
                        self.CONTEXT_BACKUP, self.KEEP_BACKUP
                except: pass
                try: del self.DATA_OBJECT_WIP_UNDO, self.DATA_OBJECT_HEADER_WIP_UNDO, self.WORKING_SUPOBJS_UNDO, \
                        self.CONTEXT_UNDO, self.KEEP_UNDO
                except: pass

                break  # BREAK OF TOP LEVEL WHILE LOOP
            else:
                self.user_manual_or_std = 'E'  # 1-1-22 IF USER NOT ACCEPT, RESTART AT 'E' IS COMPULSORY

        return self.return_fxn()

# BEAR THIS MODULE WAS 1372 LINES BEFORE OVERHAUL




# BEAR TEST STARTED AT LINE 1380 BEFORE OVERHAUL
if __name__ == '__main__':

    from MLObjects.TestObjectCreators import test_header as th
    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as cs


    _rows = 300
    _cols = 4
    _orient = 'COLUMN'
    _format = 'ARRAY'
    MOD_DTYPES = np.repeat(['FLOAT','STR'], (_rows//2, _cols-(_rows//2)))

    SXNLClass = cs.CreateSXNL(
                             rows=_rows,
                             bypass_validation=True,
                             ##################################################################################################################
                             # DATA ############################################################################################################
                             data_return_format=_format,
                             data_return_orientation=_orient,
                             DATA_OBJECT=None,
                             DATA_OBJECT_HEADER=th.test_header(_cols),
                             DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES=MOD_DTYPES,
                             data_override_sup_obj=None,
                             # CREATE FROM GIVEN ONLY ###############################################
                             data_given_orientation=None,
                             # END CREATE FROM GIVEN ONLY #############################################
                             # CREATE FROM SCRATCH_ONLY ################################
                             data_columns=_cols,
                             DATA_BUILD_FROM_MOD_DTYPES=None,
                             DATA_NUMBER_OF_CATEGORIES=5,
                             DATA_MIN_VALUES=-10,
                             DATA_MAX_VALUES=10,
                             DATA_SPARSITIES=0,
                             DATA_WORD_COUNT=20,
                             DATA_POOL_SIZE=200,
                             # END DATA ###########################################################################################################
                             ##################################################################################################################

                             #################################################################################################################
                             # TARGET #########################################################################################################
                             target_return_format=_format,
                             target_return_orientation=_orient,
                             TARGET_OBJECT=None,
                             TARGET_OBJECT_HEADER=None,
                             TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                             target_type='FLOAT',  # MUST BE 'BINARY','FLOAT', OR 'SOFTMAX'
                             target_override_sup_obj=None,
                             target_given_orientation=None,
                             # END CORE TARGET_ARGS ########################################################
                             # FLOAT AND BINARY
                             target_sparsity=0,
                             # FLOAT ONLY
                             target_build_from_mod_dtype='FLOAT',  # COULD BE FLOAT OR INT
                             target_min_value=-9,
                             target_max_value=10,
                             # SOFTMAX ONLY
                             target_number_of_categories=5,

                             # END TARGET ####################################################################################################
                             #################################################################################################################

                             #################################################################################################################
                             # REFVECS ########################################################################################################
                             refvecs_return_format=_format,  # IS ALWAYS ARRAY (WAS, CHANGED THIS 4/6/23)
                             refvecs_return_orientation=_orient,
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
    )

    KEEP = deepcopy(SXNLClass.SXNL_SUPPORT_OBJECTS[0][0])

    SXNLClass.expand_data(expand_as_sparse_dict=_format=='SPARSE_DICT', auto_drop_rightmost_column=True)

    SWNL = SXNLClass.SXNL
    WORKING_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS

    CONTEXT = []

    standard_config = 'AA'
    user_manual_or_standard = 'Z'
    augment_method = 'OATMEAL'
    bypass_validation = False


    SWNL, SUPOBJS, CONTEXT, KEEP = \
        PreRunDataAugment(
                          standard_config,
                          user_manual_or_standard,
                          augment_method,
                          SWNL,
                          _orient,
                          _orient,
                          _orient,
                          CONTEXT,
                          WORKING_SUPOBJS,
                          KEEP,
                          bypass_validation
        ).config_run

    ioap.IdentifyObjectAndPrint(SWNL[0], 'DATA', __name__, 20, 10).run_print_as_df(df_columns=WORKING_SUPOBJS[0][0],
                                                                                    orientation=_orient)









