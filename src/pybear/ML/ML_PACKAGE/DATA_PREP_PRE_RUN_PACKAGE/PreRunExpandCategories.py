import numpy as np
import sys, inspect
from copy import deepcopy
import sparse_dict as sd
from debug import get_module_name as gmn, IdentifyObjectAndPrint as ioap
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv
from linear_algebra import XTX_determinant as xtxd
from general_list_ops import list_select as ls
from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE import ExpandCategoriesMLPackage as ecmp
from ML_PACKAGE.GENERIC_PRINT import DictMenuPrint as dmp
from MLObjects import MLObject as mlo, ML_find_constants as mlfc
from MLObjects.PrintMLObject import print_object_preview as pop
from MLObjects.SupportObjects import PrintSupportContents as psc, master_support_object_dict as msod



# 1-9-22 NOTES.  A WIP COPY OF DATA (DATA_OBJECT) IS MADE (self.DATA_OBJECT_WIP) & A WIP COPY IS MADE OF THE HEADER
# (self.DATA_OBJECT_HEADER_WIP).  MODIFICATIONS ARE MADE BY USER TO THESE WIP OBJECTS DURING CONFIG, AND THEN AFTER FINAL
# ACCEPTANCE OF ALL MODS THE WIP OBJECTS OVERWRITE THE MASTERS.  HOWEVER... WIP MODS ARE MADE DIRECTLY TO
# CONTEXT, VALIDATED_DATATYPES, MODIFIED_DATATYPES MASTER OBJECTS.  CHANGES MADE TO THE MASTER OBJECTS CAN BE UNDONE
# BY DOING A RESET(r)

# **************** statistics = determinant, min of inv(XTX), max of inv(XTX), r, R**2 ************************
# starting_user_manual_or_standard      (controls config_run behavior on first pass... returns 'E'  FOR ExpandCategories (compulsory column drops), 'BYPASS' FOR DataAugmentConfigRun (go to main menu)
# standard_config_source                (standard config module to read - NOT FINISHED)
# menu_generic_commands                 (list of general commands)
# expand_menu_commands                  (list of commands for performing expansion)
# is_dict                               (True if data object is dict)
# is_list                               (True if data object is list, tuple, ndarray)
# print_preview                         (print data object)
# print_cols_and_setup_parameters       (prints column name, validated type, user type, min cutoff, "other", & filtering for all columns in all objects)
# select_column                         (select column index from DATA)
# delete_column                         (delete column from DATA)
# delete_row                            (delete row from DATA)
# intercept_verbage                     (verbage associated with appending intercept to DATA)
# undo                                  (undo last non-generic operation)
# return_fxn                            (return)
# config_run                            (exe)





class PrerunExpandCategories:

    def __init__(self,
                standard_config,
                user_manual_or_standard,   # INGESTING FOR NO REASON AS OF 1-24-22
                expand_method,
                SUPER_RAW_NUMPY_LIST,
                data_given_orientation,
                target_given_orientation,
                refvecs_given_orientation,
                RAW_SUPOBJS,
                CONTEXT,
                KEEP,
                bypass_validation
                ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        ##########################################################################################################################
        # DETERMINES MENU ALLOWABLES, AND ALSO WHICH COMMAND TEXT LINES ARE PRINTED FROM THE commands()

        # USED CHARS:   ABCD FG I      PQRSTU               !@#% ^       E(dont change)
        # UNUSED CHARS:        H JKLMNO      VWXYZ1234567890    % &

        self.expand_str = ''.join(list(self.expand_menu_commands().keys()))
        self.generic_str = ''.join(list(self.generic_menu_commands().keys()))
        self.hidden_str = ''.join(list(self.hidden_menu_commands().keys()))

        self.allowed_commands_string = self.expand_str + self.generic_str + self.hidden_str

        if not len(self.expand_menu_commands() | self.generic_menu_commands() | self.hidden_menu_commands()) == \
               int(len(self.allowed_commands_string)): raise Exception(f'THERE IS A DUPLICATE KEY IN THE MENU COMMAND DICTS')
        #######################################################################################################################

        self.standard_config = standard_config
        self.user_manual_or_standard = user_manual_or_standard
        self.method = expand_method

        self.max_cmd_len = max(map(len, dict((self.generic_menu_commands() | self.hidden_menu_commands())).values()))  # DONT INCLUDE HIDDEN OPTIONS

        self.DATA_OBJECT_WIP = SUPER_RAW_NUMPY_LIST[0].copy()
        self.TARGET_OBJECT = SUPER_RAW_NUMPY_LIST[1]
        self.REFVECS_OBJECT = SUPER_RAW_NUMPY_LIST[2]
        self.DATA_OBJECT_HEADER_WIP = RAW_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]].copy().reshape((1,-1))

        self.DATA_SUPOBJ = RAW_SUPOBJS[0]
        self.TARGET_SUPOBJ = RAW_SUPOBJS[1]
        self.REFVECS_SUPOBJ = RAW_SUPOBJS[2]



        self.data_given_orientation = akv.arg_kwarg_validater(data_given_orientation, 'data_given_orientation',
                                                ['ROW', 'COLUMN'], self.this_module, fxn)

        self.target_given_orientation = akv.arg_kwarg_validater(target_given_orientation, 'target_given_orientation',
                                                ['ROW', 'COLUMN'], self.this_module, fxn)

        self.refvecs_given_orientation = akv.arg_kwarg_validater(refvecs_given_orientation, 'refvecs_given_orientation',
                                                ['ROW', 'COLUMN'], self.this_module, fxn)

        self.CONTEXT = CONTEXT
        self.KEEP = KEEP  # BEAR, HASNT EXPANDED YET MAYBE SHOULD JUST BE DATA_OBJECT_HEADER[0].copy()?


        # THINKING MUST BE ARRAY HERE AND CANT BE SPARSE DICT?  4/7/23
        self.DATA_OBJECT_WIP_BACKUP = deepcopy(SUPER_RAW_NUMPY_LIST[0]) if self.is_dict() else SUPER_RAW_NUMPY_LIST[0].copy()
        self.DATA_OBJECT_HEADER_WIP_BACKUP = RAW_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]].copy()
        self.DATA_SUPOBJ_BACKUP = RAW_SUPOBJS[0].copy()
        self.REFVECS_SUPOBJ_BACKUP = RAW_SUPOBJS[2].copy()
        self.CONTEXT_BACKUP = deepcopy(self.CONTEXT)
        self.KEEP_BACKUP = deepcopy(self.KEEP)


        self.DATA_OBJECT_WIP_UNDO = None
        self.DATA_OBJECT_HEADER_WIP_UNDO = None
        self.DATA_SUPOBJ_UNDO = None
        self.REFVECS_SUPOBJ_UNDO = None
        self.CONTEXT_UNDO = None
        self.KEEP_UNDO = None

        self.DATA_OBJECT_HEADER_ORIGINALS = None

        self.COEFFICIENTS = None
        self.P_VALUES = None
        self.r = None
        self.R2 = None
        self.R2_adj = None
        self.F = None

        self.header_width = 50
        self.first_col_width = 12
        self.type_width = 8
        self.freq_width = 8
        self.stat_width = 12

        self.calc_allow = 'Y'
        self.expand_as_sparse_dict = 'N'
        self.starting_user_manual_or_std = 'E'


    def standard_config_source(self):
        # BEAR NEEDS WORK
        from ML_PACKAGE.standard_configs import standard_configs as sc
        return sc.PrerunExpand_standard_configs(self.standard_config, self.method, self.DATA_OBJECT_WIP, self.DATA_OBJECT_HEADER_WIP,
                                    self.TARGET_OBJECT, self.TARGET_SUPOBJ[msod.QUICK_POSN_DICT()["HEADER"]], self.CONTEXT, self.KEEP)


    def generic_menu_commands(self):   # generic_menu_commands ARE NOT REMEMBERED BY undo
        return {
                'a': 'accept / continue',
                'c': 'calculate column intercorrelations',
                'f': 'DATA columns iterative drop',
                'g': 'DATA rows iterative drop',
                'p': 'print preview data object',
                'q': 'print column names & setup parameters',
                'r': 'reset and start over',
                's': 'calculate and print statistics for DATA',
                't': 'tests inverse for DATA',
                'u': 'undo last operation',
                'z': 'toggle statistics print allow'
        }


    def expand_menu_commands(self):    # expand_menu_commands ARE REMEMBERED BY undo
        return {
                'd': 'delete column',
                'b': 'delete equal columns',
                'i': 'append intercept to data'
        }


    def hidden_menu_commands(self):
        return  {
                '!': 'show DATA_OBJECT_HEADER',
                '@': 'show VALIDATED_DATATYPES',
                '#': 'show MODIFIED_DATATYPES',
                '$': 'show FILTERING',
                '^': 'show CONTEXT'
        }


    def is_dict(self):   # THESE MUST STAY AS FXNS, CANNOT BE SET ONCE IN __init__ BECAUSE CAN CHANGE
        return isinstance(self.DATA_OBJECT_WIP, dict)


    def is_list(self):
        return isinstance(self.DATA_OBJECT_WIP, (list, tuple, np.ndarray))


    def print_preview(self, rows, columns):

        # 1/1/23 BEAR REVISIT. THIS IS SCREWED UP, IOAP PRINT AS DF IS DUPLICATING DUMMY COLUMNS, BUT NOT FLOAT (?????)
        # DATA_OBJECT_WIP IS FINE WHEN PRINTED OTHER THAN AS DF
        txt = lambda row_or_column: f'Enter start {row_or_column} of preview (zero-indexed) > '

        pop.print_object_preview(self.DATA_OBJECT_WIP,
                                 f'DATA_OBJECT AS {"SPARSEDICT" if self.is_dict() else "ARRAY" if self.is_list() else "UKNNOWN TYPE"}',
                                 rows,
                                 columns,
                                 vui.validate_user_int(txt('row'),
                                                       min=0,
                                                       max=sd.inner_len(self.DATA_OBJECT_WIP) if self.is_dict() else len(self.DATA_OBJECT_WIP[0]) - 1
                                                       ),
                                 vui.validate_user_int(txt('column'),
                                                       min=0,
                                                       max=sd.outer_len(self.DATA_OBJECT_WIP) if self.is_dict() else len(self.DATA_OBJECT_WIP) - 1
                                                       ),
                                 orientation=self.data_given_orientation,
                                 header='')

        del txt

    def print_cols_and_setup_parameters(self):
        # prints column name, validated type, user type, min cutoff, use other, start lag, end lag, scaling &
        # filtering for all columns

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

    def select_column(self):
        COL_IDX = ls.list_custom_select(self.DATA_OBJECT_HEADER_WIP[0], 'idx')
        return COL_IDX


    def delete_column(self, col_idx):
        _ = self.DATA_OBJECT_WIP[col_idx]
        has_explicit_intercept = self.DATA_OBJECT_HEADER_WIP[0][col_idx] == 'INTERCEPT' and self.intercept_verbage() in self.CONTEXT

        if self.is_list():
            # IF 1 == 1 --- DELETE INTERCEPT APPEND VERBAGE FROM CONTEXT IF DELETING INTERCEPT COLUMN
            if has_explicit_intercept or np.min(_)==np.max(_):
                try: self.CONTEXT.pop(np.argwhere(self.CONTEXT==self.intercept_verbage()))
                except: pass
            self.DATA_OBJECT_WIP = np.delete(self.DATA_OBJECT_WIP, col_idx, axis=0)

        elif self.is_dict():
            # IF 1 == 1 --- DELETE INTERCEPT APPEND VERBAGE FROM CONTEXT IF DELETING INTERCEPT COLUMN
            if has_explicit_intercept or sd.min_({0:_})==sd.max_({0:_}):
                try: self.CONTEXT.pop(np.argwhere(self.CONTEXT == self.intercept_verbage()))
                except: pass
            self.DATA_OBJECT_WIP = sd.delete_outer_key(self.DATA_OBJECT_WIP, [int(col_idx)])[0]

        self.DATA_OBJECT_HEADER_WIP = np.delete(self.DATA_OBJECT_HEADER_WIP, col_idx, axis=1)

        self.DATA_SUPOBJ = np.delete(self.DATA_SUPOBJ, col_idx, axis=1)


    # INHERITED
    # def delete_row(self, ROW_IDXS_AS_INT_OR_LIST)


    def intercept_verbage(self):
        return 'Appended INTERCEPT column.'


    def undo(self):
        try:
            self.DATA_OBJECT_WIP = \
                deepcopy(self.DATA_OBJECT_WIP_UNDO) if isinstance(self.DATA_OBJECT_WIP_UNDO, dict) else self.DATA_OBJECT_WIP_UNDO.copy()
            self.DATA_OBJECT_HEADER_WIP = deepcopy(self.DATA_OBJECT_HEADER_WIP_UNDO)

            self.DATA_SUPOBJ = deepcopy(self.DATA_SUPOBJ_UNDO)
            self.REFVECS_SUPOBJ = deepcopy(self.REFVECS_SUPOBJ_UNDO)

            self.CONTEXT = deepcopy(self.CONTEXT_UNDO)
            self.KEEP = deepcopy(self.KEEP_UNDO)
            print(f'\nUNDO - {self.UNDO_DESC.upper()} - COMPLETE.\n')

        except:
            print(f"\nCAN'T UNDO, BACKUP OBJECTS HAVE NOT BEEN GENERATED YET\n")


    def return_fxn(self):

        SUPER_RAW_NUMPY_LIST = [self.DATA_OBJECT, self.TARGET_OBJECT, self.REFVECS_OBJECT]
        RAW_SUPOBJS = [self.DATA_SUPOBJ, self.TARGET_SUPOBJ, self.REFVECS_SUPOBJ]

        # SUPER_RAW_NUMPY_LIST IS NOW ACTUALLY SUPER_WORKING_NUMPY_LIST, AND SHOULD BE NAMED SUCH WHERE IT IS RETURNED
        return SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, self.CONTEXT, self.KEEP


    def config_run(self):
        self.user_manual_or_std = self.starting_user_manual_or_std

        fxn = inspect.stack()[0][3]

        while True:
            while True:
                # MAKE THIS FIRST TO APPLY UNDO B4 RESETTING UNDO OBJECTS BELOW
                if self.user_manual_or_std == 'U':  # 'undo last operation(u)'
                    self.undo()

                # SET THE INITIAL STATE OF EVERY OBJECT, FOR UNDO PURPOSES
                # 12-12-21 10:49 AM ONLY RESET UNDO AFTER AN OPERATION, NOT PRINTS, ALLOWS USER TO LOOK AT OBJECTS
                # AND THEN DECIDE TO DO AN UNDO
                if True in map(lambda x: f'({self.user_manual_or_std})'.lower() in x, self.expand_menu_commands()):
                    self.DATA_OBJECT_WIP_UNDO = \
                        deepcopy(self.DATA_OBJECT_WIP) if instance(self.DATA_OBJECT_WIP, dict) else self.DATA_OBJECT_WIP.copy()
                    self.DATA_OBJECT_HEADER_WIP_UNDO = deepcopy(self.DATA_OBJECT_HEADER_WIP)
                    self.DATA_SUPOBJ_UNDO = deepcopy(self.DATA_SUPOBJ)
                    self.REFVECS_SUPOBJ_UNDO = deepcopy(self.REFVECS_SUPOBJ)
                    self.CONTEXT_UNDO = deepcopy(self.CONTEXT)
                    self.KEEP_UNDO = deepcopy(self.KEEP)

                    if self.user_manual_or_std in self.expand_str:  # ANYTIME EXPAND CMD IS USED, RECORD
                        self.UNDO_DESC = self.expand_menu_commands()[self.user_manual_or_std]
                    else: self.UNDO_DESC = f'\nUNDO NOT AVAILABLE\n'


                if self.user_manual_or_std in 'P':  # '(p)print_preview data object'
                    self.print_preview(20, 10)


                elif self.user_manual_or_std in 'Q':  #
                    # prints column name, validated type, user type, min cutoff, "other", & filtering for all columns in all objects
                    self.print_cols_and_setup_parameters()


                elif self.user_manual_or_std in 'R':  #'(r)reset and start over'

                    self.DATA_OBJECT_WIP = \
                        deepcopy(self.DATA_OBJECT_WIP_BACKUP) if isinstance(self.DATA_OBJECT_WIP_BACKUP, dict) else self.DATA_OBJECT_WIP_BACKUP.copy()
                    self.DATA_OBJECT_HEADER_WIP = deepcopy(self.DATA_OBJECT_HEADER_WIP_BACKUP)

                    self.DATA_SUPOBJ = deepcopy(self.DATA_SUPOBJ_BACKUP)
                    self.REFVECS_SUPOBJ = deepcopy(self.REFVECS_SUPOBJ_BACKUP)

                    self.CONTEXT = deepcopy(self.CONTEXT_BACKUP)
                    self.KEEP = deepcopy(self.KEEP_BACKUP)

                    self.user_manual_or_std = 'E'  # SET TO "E" TO GET CAUGHT IMMEDIATELY AFTER RESET BY "E"
                    continue


                elif self.user_manual_or_std in 'D':  # '(d)delete column'

                    while True:
                        print(f'\nSELECT COLUMNS TO DELETE FROM DATA:')
                        COL_IDX = self.select_column()

                        for col_idx in reversed(COL_IDX):
                            # IF COLUMN TO DELETE IS IN ORIGINALS, REQUIRE REASON & APPEND TO CONTEXT
                            if self.DATA_OBJECT_HEADER_WIP[0][col_idx] in self.DATA_OBJECT_HEADER_WIP_ORIGINALS:
                                delete_text = f'Deleted {self.DATA_OBJECT_HEADER_WIP[0][col_idx]} for '
                                delete_reason = input(delete_text + '(give reason) > ')
                                while True:
                                    if vui.validate_user_str(f'User entered "{delete_text}{delete_reason}" ... Accept? (y/n) > ', 'YN') == 'Y':
                                        final_str = f' and recording "{delete_text}{delete_reason}"'
                                        break
                            else:
                                final_str = f''

                            print(f'Deleting {self.DATA_OBJECT_HEADER_WIP[0][col_idx]}{final_str}')
                            __ = vui.validate_user_str('Accept(a), skip this deletion(s), skip all remaining deletions(c) > ', 'ASC')
                            if __ == 'A': pass
                            elif __ == 'S': continue
                            elif __ == 'C': break

                            if self.DATA_OBJECT_HEADER_WIP[0][col_idx] in self.DATA_OBJECT_HEADER_WIP_ORIGINALS:
                                self.CONTEXT += [f'{delete_text}{delete_reason}']

                            self.delete_column(col_idx)

                        break


                elif self.user_manual_or_std == 'B':  # 'delete equal columns(b)  (iterate thru DATA and prompt user to delete one of two equal columns)
                    TO_DELETE_HOLDER = []
                    MATCHING_COLUMNS = []
                    while True:  # MANAGES ESCAPE IF THERE ARE NO EQUAL COLUMNS
                        auto_delete = vui.validate_user_str(f'\nAuto delete right-most column of equal pairs(y) or manual select(n) > ', 'YN')
                        for col_idx1 in range(len(self.DATA_OBJECT_WIP) - 1, 0, -1):
                            if col_idx1 in TO_DELETE_HOLDER: continue
                            for col_idx2 in range(col_idx1 - 1, -1, -1):
                                if col_idx2 in TO_DELETE_HOLDER: continue
                                if np.array_equiv(self.DATA_OBJECT_WIP[col_idx1], self.DATA_OBJECT_WIP[col_idx2]):
                                    if auto_delete == 'Y': del_idx = 0
                                    else:
                                        del_idx = ls.list_single_select([self.DATA_OBJECT_HEADER_WIP[0][_] for _ in [col_idx1, col_idx2]],
                                                                    f'Select column to delete', 'idx')[0]
                                    TO_DELETE_HOLDER.append([col_idx1 if del_idx==0 else col_idx2][0])
                                    MATCHING_COLUMNS.append([col_idx2 if del_idx==0 else col_idx1][0])
                                    if del_idx == 0: break  # SKIP ANY REMAINING CHECKS AGAINST col_idx1 SINCE ITS GETTING DELETED ANYWAY


                        if len(TO_DELETE_HOLDER) == 0:
                            print(f'\nTHERE ARE NO EQUAL COLUMNS.\n')
                            break

                        MASTER_ORDER = list(reversed(np.argsort(TO_DELETE_HOLDER).tolist()))
                        TO_DELETE_HOLDER = np.array(TO_DELETE_HOLDER)[MASTER_ORDER]
                        MATCHING_COLUMNS = np.array(MATCHING_COLUMNS)[MASTER_ORDER]
                        del MASTER_ORDER

                        __ = self.DATA_OBJECT_HEADER_WIP[0]
                        [print(f'{__[TO_DELETE_HOLDER[idx]]} (equal to {__[MATCHING_COLUMNS[idx]]})') for idx in range(len(TO_DELETE_HOLDER))]
                        del __

                        if vui.validate_user_str(f'\nAccept(y) Abort(n)? > ', 'YN') == 'Y':
                            for _ in TO_DELETE_HOLDER:
                                self.CONTEXT += [f'Deleted {self.DATA_OBJECT_HEADER_WIP[0][_]} for equality with another column']

                            [self.delete_column(idx) for idx in TO_DELETE_HOLDER]

                            print(f'\nDuplicate columns successfully deleted.\n')
                        else: break

                        break

                # COMPULSORY AT START, ONLY ACCESSIBLE AT START OR AFTER RESET (NOT ACCESSIBLE IN MENU COMMANDS)
                elif self.user_manual_or_std == 'E':  #'(e)expand all categorical features to levels'

                    # 11/29/22 BEAR TOOK THIS OUT OF ExpandCategories_expand_categories.  MAKE IT WORK.
                    # CREATE A MASTER OF ORIGINAL COLUMNS FOR REFERENCE IN DELETING / ADDING COLUMNS
                    self.DATA_OBJECT_HEADER_WIP_ORIGINALS = deepcopy(self.DATA_OBJECT_HEADER_WIP)

                    data_return_format = {'A':'ARRAY', 'P':'SPARSE_DICT'}[
                                                vui.validate_user_str(f'\nExpand as ARRAY(a) or SPARSE_DICT(p) > ', 'AP')]

                    Expander = ecmp.ExpandCategoriesMLPackage(
                                                     self.DATA_OBJECT_WIP,
                                                     self.data_given_orientation,
                                                     self.data_given_orientation,
                                                     data_return_format,
                                                     FULL_SUPPORT_OBJECT=self.DATA_SUPOBJ,
                                                     CONTEXT=self.CONTEXT,
                                                     KEEP=self.KEEP,
                                                     TARGET=self.TARGET_OBJECT,  # MUST HAVE A TARGET TO DO FULL cycler, OTHERWISE CAN ONLY GET determ!!!
                                                     target_given_orientation=self.target_given_orientation,
                                                     TARGET_TRANSPOSE=None,
                                                     target_transpose_given_orientation=None,
                                                     TARGET_AS_LIST=self.TARGET_OBJECT,
                                                     target_as_list_given_orientation=self.target_given_orientation,
                                                     target_is_multiclass=False,
                                                     address_multicolinearity='PROMPT',
                                                     auto_drop_rightmost_column=False,
                                                     multicolinearity_cycler=True,
                                                     append_ones_for_cycler=True,
                                                     prompt_to_edit_given_mod_dtypes=False,
                                                     print_notes=False,
                                                     bypass_validation=False,
                                                     prompt_user_for_accept=True,
                                                     calling_module=self.this_module,
                                                     calling_fxn=inspect.stack()[0][3])

                    self.DATA_OBJECT_WIP = Expander.DATA_OBJECT
                    self.DATA_SUPOBJ = Expander.SUPPORT_OBJECTS
                    self.DATA_OBJECT_HEADER_WIP = Expander.SUPPORT_OBJECTS[msod.QUICK_POSN_DICT()['HEADER']].reshape((1,-1))
                    self.CONTEXT += Expander.CONTEXT
                    self.KEEP = Expander.KEEP
                    del Expander


                elif self.user_manual_or_std == 'F':   # '(f)DATA columns iterative drop'

                    self.whole_data_object_stats(self.DATA_OBJECT_WIP, 'DATA MATRIX', self.DATA_OBJECT_HEADER_WIP[0],
                                                append_ones='N')
                    self.column_drop_iterator(self.DATA_OBJECT_WIP, 'DATA MATRIX', self.DATA_OBJECT_HEADER_WIP[0],
                                                append_ones='N')

                elif self.user_manual_or_std == 'G':  # '(f)DATA rows iterative drop'
                    self.row_drop_iterator(self.DATA_OBJECT_WIP, 'DATA MATRIX', self.DATA_OBJECT_HEADER_WIP[0], TARGET)


                elif self.user_manual_or_std == 'C':   #'(c)calculate column intercorrelations'

                    print(f'\nCalculating column intercorrelations. Patience...')

                    RESULTS = np.empty((3, 0), dtype=object)  # # COLUMN NAMES AND RSQ HOLDER
                    ERROR_HOLDER = np.empty((2, 0), dtype=object)
                    __ = self.DATA_OBJECT_WIP
                    for col_idx in range(len(__) - 1):
                        for col_idx2 in range(col_idx + 1, len(__)):
                            np.seterr(divide='ignore', invalid='ignore')
                            if self.is_list():
                                rsq = np.corrcoef(__[col_idx].astype(float), __[col_idx2].astype(float))[0][1] ** 2
                            elif self.is_dict():
                                rsq = sd.rsq_({0: __[col_idx]}, {0: __[col_idx2]})

                            COLUMNS_HOLDER = np.array([self.DATA_OBJECT_HEADER_WIP[0][col_idx],
                                                      self.DATA_OBJECT_HEADER_WIP[0][col_idx2]], dtype=object)

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
                    _format = lambda col_idx, row_idx, dum_width: str(
                        RESULTS[col_idx][ARGSORT_RSQ_DESC[row_idx]]).ljust(dum_width)
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


                elif self.user_manual_or_std in 'I':  #  '(i)append intercept to data'

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


                elif self.user_manual_or_std == 'S':  # '(s)calculate and print statistics for DATA'
                    self.whole_data_object_stats(self.DATA_OBJECT_WIP, 'DATA', self.DATA_OBJECT_HEADER_WIP, append_ones='N')


                elif self.user_manual_or_std == 'T':  # '(t)tests inverse of XTX'
                    # 11/27/22 PUT DATA INTO A ROW/COLUMN INDIFFERENT STATE FOR XTX_determ (BELIEVE DATA IS [[]=COLUMN HERE)
                    if isinstance(self.DATA_OBJECT_WIP, dict):
                        DUM_XTX = sd.sparse_AAT(self.DATA_OBJECT_WIP, return_as='ARRAY')
                    elif isinstance(self.DATA_OBJECT_WIP, (np.ndarray, list, tuple)):
                        DUM_XTX = np.matmul(self.DATA_OBJECT_WIP, self.DATA_OBJECT_WIP.transpose())
                    xtxd.XTX_determinant(XTX_AS_ARRAY_OR_SPARSEDICT=DUM_XTX, name='DATA', module=self.this_module,
                                         fxn='(t)tests inverse of XTX', print_to_screen=True, return_on_exception='nan')
                    del DUM_XTX

                elif self.user_manual_or_std == 'Z':  # '(z)toggle statistics print allow'
                    self.calc_allow = vui.validate_user_str(
                    f'\nALLOW STATISTICS CALCULATIONS DURING COLUMN DROP SELECTIONS? (y/n) (THIS CAN TAKE MANY MINUTES FOR LARGE DATA) > ',
                    'YN')


                elif self.user_manual_or_std == '!':   # '(!)show DATA_OBJECT_HEADER_WIP
                    print(f'\nCOLUMN HEADERS\n')
                    [print(_[:100]) for _ in self.DATA_OBJECT_HEADER_WIP[0]]


                elif self.user_manual_or_std == '@':  # '(@)show VALIDATED_DATATYPES
                    print(f'\nVALIDATED DATATYPES\n')
                    psc.PrintSingleSupportObject(self.DATA_SUPOBJ[msod.QUICK_POSN_DICT()['VALIDATEDDATATYPES']],
                                                 'VALIDATEDDATATYPES',
                                                 HEADER=self.DATA_OBJECT_HEADER_WIP,
                                                 calling_module=self.this_module,
                                                 calling_fxn=inspect.stack()[0][3])


                elif self.user_manual_or_std == '#':  # '(#)show MODIFIED_DATATYPES
                    print(f'\nMODIFIED DATATYPES\n')
                    psc.PrintSingleSupportObject(self.DATA_SUPOBJ[msod.QUICK_POSN_DICT()['MODIFIEDDATATYPES']],
                                                 'MODIFIEDDATATYPES',
                                                 HEADER=self.DATA_OBJECT_HEADER_WIP,
                                                 calling_module=self.this_module,
                                                 calling_fxn=inspect.stack()[0][3])


                elif self.user_manual_or_std == '$':  # '($)show FILTERING
                    print(f'\nFILTERING:\n')
                    max_len = max(map(len, self.DATA_OBJECT_HEADER_WIP[0]))

                    for col_idx in range(len(self.DATA_SUPOBJ[0])):
                        base = f'{str(self.DATA_OBJECT_HEADER_WIP[0][col_idx][:48]).ljust(min(max_len + 5, 50))}'
                        if len(self.DATA_SUPOBJ[msod.QUICK_POSN_DICT()['FILTERING']]) == 0:
                            base += f'[]'; print(base); continue
                        else:
                            COL_FILTERING = self.DATA_SUPOBJ[msod.QUICK_POSN_DICT()['FILTERING']][col_idx]
                            for filter_idx in range(len(COL_FILTERING)):
                                if filter_idx==0: print(base + COL_FILTERING[0])
                                else: print(f' '*min(max_len + 5, 50) + COL_FILTERING[filter_idx])
                            del COL_FILTERING
                    del max_len

                elif self.user_manual_or_std == '^':  # '(^)show CONTEXT'
                    if len(self.CONTEXT) == 0: print(f'\n*** CONTEXT IS EMPTY.*** \n')
                    else:
                        print(f'\nCONTEXT\n')
                        [print(_) for _ in self.CONTEXT]


                elif self.user_manual_or_std == 'A':   # 'accept / continue(a)'
                    break  # BREAK OUT OF COMMAND ENTRY LOOP

                # PRINT MENU
                max_len = max(map(len, dict((self.expand_menu_commands() | self.generic_menu_commands() | self.hidden_menu_commands())).values()))
                for menu in (self.expand_menu_commands(), self.generic_menu_commands(), self.hidden_menu_commands()):
                    dmp.DictMenuPrint(menu, disp_len=140, fixed_col_width=max_len+5)
                    print()
                del max_len

                self.user_manual_or_std = vui.validate_user_str(' > ', self.allowed_commands_string)


            if vui.validate_user_str(f'\nAccept categorical feature level expansion? (y/n) > ', 'YN') == 'Y':

                self.DATA_OBJECT = self.DATA_OBJECT_WIP
                self.DATA_OBJECT_HEADER = self.DATA_OBJECT_HEADER_WIP

                del self.DATA_OBJECT_WIP, self.DATA_OBJECT_HEADER_WIP, self.DATA_OBJECT_WIP_BACKUP, \
                    self.DATA_OBJECT_HEADER_WIP_BACKUP, self.DATA_SUPOBJ_BACKUP, self.REFVECS_SUPOBJ_BACKUP, \
                    self.CONTEXT_BACKUP, self.KEEP_BACKUP

                del self.DATA_OBJECT_WIP_UNDO, self.DATA_OBJECT_HEADER_WIP_UNDO, \
                    self.DATA_SUPOBJ_UNDO, self.REFVECS_SUPOBJ_UNDO, self.CONTEXT_UNDO, self.KEEP_UNDO

                break  # BREAK OF TOP LEVEL WHILE LOOP

            else:
                self.DATA_OBJECT_WIP = \
                    deepcopy(self.DATA_OBJECT_WIP_BACKUP) if isinstance(self.DATA_OBJECT_WIP_BACKUP, dict) else self.DATA_OBJECT_WIP_BACKUP.copy()
                self.DATA_OBJECT_HEADER_WIP = deepcopy(self.DATA_OBJECT_HEADER_WIP_BACKUP)
                self.DATA_SUPOBJ = deepcopy(self.DATA_SUPOBJ_BACKUP)
                self.REFVECS_SUPOBJ = deepcopy(self.REFVECS_SUPOBJ_BACKUP)
                self.CONTEXT = deepcopy(self.CONTEXT_BACKUP)
                self.KEEP = deepcopy(self.KEEP_BACKUP)

                self.user_manual_or_std = 'E'  # 1-1-22 IF USER NOT ACCEPT, RESTART AT 'E' IS COMPULSORY

        return self.return_fxn()





if __name__ == '__main__':
    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl
    from MLObjects.TestObjectCreators import test_header as th

    rows = 300
    cols = 6
    _format = 'ARRAY'
    _orient = 'COLUMN'


    TestSRNL = csxnl.CreateSXNL(
                                 rows=rows,
                                 bypass_validation=True,
                                 data_return_format=_format,
                                 data_return_orientation=_orient,
                                 DATA_OBJECT=None,
                                 DATA_OBJECT_HEADER=th.test_header(cols),
                                 DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 data_override_sup_obj=False,
                                 data_given_orientation=None,
                                 data_columns=cols,
                                 DATA_BUILD_FROM_MOD_DTYPES=['BIN', 'INT', 'FLOAT', 'STR'],
                                 DATA_NUMBER_OF_CATEGORIES=10,
                                 DATA_MIN_VALUES=-10,
                                 DATA_MAX_VALUES=10,
                                 DATA_SPARSITIES=50,
                                 DATA_WORD_COUNT=20,
                                 DATA_POOL_SIZE=200,
                                 target_return_format=_format,
                                 target_return_orientation=_orient,
                                 TARGET_OBJECT=None,
                                 TARGET_OBJECT_HEADER=[['TARGET']],
                                 TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 target_type='FLOAT',  # MUST BE 'BINARY','FLOAT', OR 'SOFTMAX'
                                 target_override_sup_obj=False,
                                 target_given_orientation=None,
                                 target_sparsity=50,
                                 target_build_from_mod_dtype='FLOAT',  # COULD BE FLOAT OR INT
                                 target_min_value=-10,
                                 target_max_value=10,
                                 target_number_of_categories=None,
                                 refvecs_return_format='ARRAY',
                                 refvecs_return_orientation='COLUMN',
                                 REFVECS_OBJECT=None,
                                 REFVECS_OBJECT_HEADER=None,
                                 REFVECS_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 REFVECS_BUILD_FROM_MOD_DTYPES=['STR'],
                                 refvecs_override_sup_obj=False,
                                 refvecs_given_orientation=None,
                                 refvecs_columns=5,
                                 REFVECS_NUMBER_OF_CATEGORIES=10,
                                 REFVECS_MIN_VALUES=-10,
                                 REFVECS_MAX_VALUES=10,
                                 REFVECS_SPARSITIES=50,
                                 REFVECS_WORD_COUNT=20,
                                 REFVECS_POOL_SIZE=200
                                 )

    SUPER_RAW_NUMPY_LIST = TestSRNL.SXNL
    RAW_SUPOBJS = TestSRNL.SXNL_SUPPORT_OBJECTS



    print(f'TEST DATA = ')
    ioap.IdentifyObjectAndPrint(SUPER_RAW_NUMPY_LIST[0], 'DATA', __name__, 20, 10).run_print_as_df(df_columns=RAW_SUPOBJS[0][0],
                                                                                                   orientation=_orient)


    CONTEXT = []
    KEEP = RAW_SUPOBJS[0][0].copy()





    standard_config = 'AA'
    user_manual_or_standard = 'Z'
    expand_method = 'CHICKEN'
    bypass_validation = False



    SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, CONTEXT, KEEP = \
    PrerunExpandCategories(
                           standard_config,
                           user_manual_or_standard,
                           expand_method,
                           SUPER_RAW_NUMPY_LIST,
                           _orient,
                           _orient,
                           _orient,
                           RAW_SUPOBJS,
                           CONTEXT,
                           KEEP,
                           bypass_validation
    ).config_run()

    ioap.IdentifyObjectAndPrint(SUPER_WORKING_NUMPY_LIST[0], 'DATA', __name__, 20, 10).run_print_as_df(df_columns=WORKING_SUPOBJS[0][0],
                                                                                                        orientation=_orient)





































































