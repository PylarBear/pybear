import inspect
import numpy as np, pandas as pd
from copy import deepcopy
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from general_list_ops import list_select as ls
from general_data_ops import get_shape as gs
from ML_PACKAGE.GENERIC_PRINT import DictMenuPrint as dmp
from ML_PACKAGE.DATA_PREP_IN_SITU_PACKAGE.target_vector import TargetInSituHandling as tish
from MLObjects import MLObject as mlo
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo
from MLObjects.SupportObjects import master_support_object_dict as msod
from ML_PACKAGE import TestPerturber as tp


# USE THIS AS TEMPLATE FOR ML PACKAGE ConfigRun FILES /

# dataclass_mlo_loader()             Loads an instance of MLObject for DATA.
# intercept_manager()                Locate columns of constants in DATA & handle. As of 11/15/22 only for MLR, MI, and GMLR.
# insert_intercept()                 Insert a column of ones in the 0 index of DATA.
# delete_columns()                    Delete a column from DATA and respective holder objects.
# run_module_input_tuple()           tuple of base params that pass into run_module for all ML packages
# config_module()                    gets configuration source, returns configuration parameters for particular ML package
# run_module()                       returns run module for particular ML package
# return_fxn_base()                  values returned from all ML packages
# return_fxn()                       returns user-specified output, in addition to return_fxn_base()
# sub_post_run_cmds()                package-specific options available to modify WORKING_DATA after run
# base_post_run_options_module()     holds post run options applied to all ML packages
# sub_post_run_options_module()      holds post run options unique to particular ML package
# configrun()                        runs config_module() & run_module()



class MLConfigRunTemplate:
    def __init__(self,
                    standard_config,
                    sub_config,
                    SUPER_RAW_NUMPY_LIST,
                    RAW_SUPOBJS,
                    SUPER_WORKING_NUMPY_LIST,
                    WORKING_SUPOBJS,
                    data_given_orientation,
                    target_given_orientation,
                    refvecs_given_orientation,
                    data_run_orientation,
                    target_run_orientation,
                    refvecs_run_orientation,
                    data_run_format,
                    target_run_format,
                    refvecs_run_format,
                    WORKING_CONTEXT,
                    WORKING_KEEP,
                    split_method,
                    LABEL_RULES,
                    number_of_labels,
                    event_value,
                    negative_value,
                    conv_kill,
                    pct_change,
                    conv_end_method,
                    rglztn_type,
                    rglztn_fctr,
                    module):

        self.this_module = module
        fxn = '__init__'

        bypass_validation = False

        self.bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                         self.this_module, fxn)

        self.standard_config = standard_config
        self.sub_config = sub_config

        self.RAW_SUPOBJS = RAW_SUPOBJS
        self.WORKING_SUPOBJS = WORKING_SUPOBJS

        self.data_given_format = ldv.list_dict_validater(SUPER_WORKING_NUMPY_LIST[0], 'DATA')[0]
        self.target_given_format = ldv.list_dict_validater(SUPER_WORKING_NUMPY_LIST[1], 'TARGET')[0]
        self.refvecs_given_format = ldv.list_dict_validater(SUPER_WORKING_NUMPY_LIST[2], 'REFVECS')[0]

        format_validator = lambda _OBJ, _name: akv.arg_kwarg_validater(_OBJ, _name, ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN'], self.this_module, fxn)
        self.data_run_format = format_validator(data_run_format, 'data_run_format')
        self.target_run_format = format_validator(target_run_format, 'target_run_format')
        self.refvecs_run_format = format_validator(refvecs_run_format, 'refvecs_run_format')
        del format_validator

        orient_validator = lambda _OBJ, _name: akv.arg_kwarg_validater(_OBJ, _name, ['ROW', 'COLUMN'], self.this_module, fxn)
        self.data_given_orientation = orient_validator(data_given_orientation, 'data_given_orientation')
        self.target_given_orientation = orient_validator(target_given_orientation, 'target_given_orientation')
        self.refvecs_given_orientation = orient_validator(refvecs_given_orientation, 'refvecs_given_orientation')
        orient_validator = lambda _OBJ, _name: akv.arg_kwarg_validater(_OBJ, _name, ['ROW', 'COLUMN', 'AS_GIVEN'], self.this_module, fxn)
        self.data_run_orientation = orient_validator(data_run_orientation, 'data_run_orientation')
        self.target_run_orientation = orient_validator(target_run_orientation, 'target_run_orientation')
        self.refvecs_run_orientation = orient_validator(refvecs_run_orientation, 'refvecs_run_orientation')
        del orient_validator

        self.raw_target_is_multiclass = gs.get_shape('TARGET', SUPER_RAW_NUMPY_LIST[1], self.target_given_orientation)[1] > 1
        self.working_target_is_multiclass = gs.get_shape('TARGET', SUPER_WORKING_NUMPY_LIST[1], self.target_given_orientation)[1] > 1

        self.SUPER_WORKING_NUMPY_LIST_BACKUP = None
        self.WORKING_SUPOBJS_BACKUP = None

        if vui.validate_user_str(f'\nCreate backups of working objects? (y/n) - Creating allows restore of any dropped columns, ' + \
                                 f'bypassing saves memory > ', 'YN') == 'Y':
            # KEEP IN CASE WANT TO RESTORE WORKING THINGS TO ORIGINAL STATE BEFORE EXITING BACK TO ML IF COLUMN CHOPS WERE DONE.
            # BEFORE mloo SO NOT NECESSARILY IN run_orientation.

            self.SUPER_WORKING_NUMPY_LIST_BACKUP = [_.copy() if isinstance(_, np.ndarray) else deepcopy(_) for _ in SUPER_WORKING_NUMPY_LIST]
            self.WORKING_SUPOBJS_BACKUP = WORKING_SUPOBJS.copy()

        print(f'\n    BEAR IN MLConfigRunTemplate Orienting RAW DATA & TARGET IN __init__.  Patience...')

        SRNLOrienterClass = mloo.MLObjectOrienter(
                                                    DATA=SUPER_RAW_NUMPY_LIST[0],
                                                    data_given_orientation=self.data_given_orientation,
                                                    data_return_orientation=self.data_run_orientation,
                                                    data_return_format='AS_GIVEN',

                                                    target_is_multiclass=self.raw_target_is_multiclass,
                                                    TARGET=SUPER_RAW_NUMPY_LIST[1],
                                                    target_given_orientation=self.target_given_orientation,
                                                    target_return_orientation=self.target_run_orientation,
                                                    target_return_format='AS_GIVEN',

                                                    RETURN_OBJECTS=['DATA','TARGET'],

                                                    bypass_validation=self.bypass_validation,
                                                    calling_module=self.this_module,
                                                    calling_fxn=fxn
        )

        SUPER_RAW_NUMPY_LIST[0] = SRNLOrienterClass.DATA
        SUPER_RAW_NUMPY_LIST[1] = SRNLOrienterClass.TARGET
        del SRNLOrienterClass

        # BECAUSE ObjectOrienter WASNT BUILT TO HANDLE REFVECS
        SUPER_RAW_NUMPY_LIST[2] = mlo.MLObject(SUPER_RAW_NUMPY_LIST[2], self.refvecs_given_orientation,
            name='REFVECS', return_orientation=self.refvecs_run_orientation, return_format='AS_GIVEN',
            bypass_validation=self.bypass_validation, calling_module=self.this_module, calling_fxn=fxn).OBJECT

        self.SUPER_RAW_NUMPY_LIST = SUPER_RAW_NUMPY_LIST
        print(f'\n    BEAR IN MLConfigRunTemplate Done Orienting RAW DATA & TARGET in __init__')




        ########################################################################################################################
        # ORIENT SWNL IN CHILDREN TO ACCOMMODATE DIFFERENT RETURN OBJECTS AND HANDLING
        ########################################################################################################################



        self.WORKING_CONTEXT = WORKING_CONTEXT
        self.WORKING_KEEP = WORKING_KEEP
        self.split_method = split_method
        self.LABEL_RULES = LABEL_RULES
        self.number_of_labels = number_of_labels
        self.event_value = event_value
        self.negative_value = negative_value
        self.conv_kill = conv_kill
        self.pct_change = pct_change
        self.conv_end_method = conv_end_method
        self.rglztn_type = rglztn_type
        self.rglztn_fctr = rglztn_fctr

        self.SUPER_WORKING_NUMPY_DICT = dict(((0, "DATA"), (1, "TARGET"), (2, "REFERENCE")))

        self.WINNING_COLUMNS = []    # FOR USE IN MLR, GMLR, & MI
        self.PERTURBER_RESULTS = pd.DataFrame({})

        self.TRAIN_SWNL = []
        self.DEV_SWNL = []
        self.TEST_SWNL = []

        # CANT USE 'k'... HOLDS "operations for ___ WINNING COLUMNS" IN MLR, GMLR, MI ConfigRuns
        self.BASE_POST_RUN_CMDS = {
                                    'a': 'accept / return to ML menu',
                                    'b': 'run train without reconfig',
                                    'c': 'run train with reconfig',
                                    'd': 'delete columns from DATA',
                                    'i': 'manage intercept',
                                    'r': 'delete row',
                                    't': 'modify target',
                                    'v': 'operations for PERTURBER RESULTS'
                                    }

        self.ALL_POST_RUN_CMDS = self.BASE_POST_RUN_CMDS | self.sub_post_run_cmds()

        # VERIFY THERE ARE NO DUPLICATE KEYS IN self.BASE_POST_RUN_CMDS & self.sub_post_run_cmds()

        if not len(self.ALL_POST_RUN_CMDS) == len(self.BASE_POST_RUN_CMDS) + len(self.sub_post_run_cmds()):
            raise Exception(f'*** THERE ARE DUPLICATE KEYS IN BASE_POST_RUN_CMDS & sub_post_run_cmds()')

        # PLACEHOLDERS
        self.DataClass = None
        self.intcpt_col_idx = None
        self.post_configrun_select = ''

    # END init ############################################################################################################
    #######################################################################################################################
    #######################################################################################################################

    def dataclass_mlo_loader(self, DATA, name=None, fxn=None):
        """Loads an instance of MLObject for DATA."""

        name = name if not name is None else self.SUPER_WORKING_NUMPY_DICT[0]
        fxn = fxn if not fxn is None else inspect.stack()[0][3]

        self.DataClass = mlo.MLObject(
                                      DATA,
                                      self.data_given_orientation,
                                      name=name,
                                      return_orientation='AS_GIVEN',
                                      return_format='AS_GIVEN',
                                      bypass_validation=self.bypass_validation,
                                      calling_module=self.this_module,
                                      calling_fxn=fxn
        )


    def intercept_manager(self):
        """Locate columns of constants in DATA & handle. As of 11/15/22 only for MLR, MI, and GMLR."""

        fxn = inspect.stack()[0][3]

        self.dataclass_mlo_loader(self.SUPER_WORKING_NUMPY_LIST[0], name=self.SUPER_WORKING_NUMPY_DICT[0], fxn=fxn)
        self.DataClass.intercept_manager(DATA_FULL_SUPOBJ_OR_HEADER=self.WORKING_SUPOBJS[0], intcpt_col_idx=self.intcpt_col_idx)

        self.SUPER_WORKING_NUMPY_LIST[0] = self.DataClass.OBJECT
        self.WORKING_SUPOBJS[0] = self.DataClass.HEADER_OR_FULL_SUPOBJ
        self.WORKING_CONTEXT = self.DataClass.CONTEXT
        self.intcpt_col_idx = self.DataClass.intcpt_col_idx



    def insert_intercept(self):
        """Insert a column of ones in the 0 index of DATA."""

        fxn = inspect.stack()[0][3]

        print(f'\nAppending intercept...')

        self.dataclass_mlo_loader(self.SUPER_WORKING_NUMPY_LIST[0], name=self.SUPER_WORKING_NUMPY_DICT[0], fxn=fxn)

        self.DataClass.insert_standard_intercept(HEADER_OR_FULL_SUPOBJ=self.WORKING_SUPOBJS[0], CONTEXT=self.WORKING_CONTEXT)

        self.SUPER_RAW_NUMPY_LIST[0] = self.DataClass.OBJECT
        self.WORKING_SUPOBJS[0] = self.DataClass.HEADER_OR_FULL_SUPOBJ
        self.WORKING_CONTEXT = self.DataClass.CONTEXT
        # UPDATE TO CONTEXT SHOULD HAVE BEEN APPENDED BY DataClass VIA insert_standard_intercept > insert_intercept > insert_column

        self.DataClass = None

        self.intcpt_col_idx = 0
        print(f'Done.')


    def delete_columns(self, COL_IDXS_AS_LIST, update_context=False):
        """Delete a column from DATA and respective holder objects. Clean dictionaries after running."""

        fxn = inspect.stack()[0][3]

        COL_IDXS_AS_LIST = np.array(COL_IDXS_AS_LIST).reshape(1,)

        if gs.get_shape(self.SUPER_WORKING_NUMPY_DICT[0], self.SUPER_WORKING_NUMPY_LIST[0], self.data_run_orientation)[1] == 1:
            # IF ONLY ONE COLUMN, DONT ALLOW DELETE
            print(f'\n*** {self.SUPER_WORKING_NUMPY_DICT[0]} OBJECT HAS ONLY ONE COLUMN AND IT CANNOT BE DELETED *** \n')
        else:
            # HANDLE THINGS IN FILTER B4 DELETING COLUMN NAME FROM HEADER ##############################################
            # IF SOMETHING IN FILTER, MOVE TO CONTEXT BEFORE DELETE.
            for col_idx in COL_IDXS_AS_LIST:
                ACTIVE_HEADER = self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()['HEADER']][col_idx]
                for _string in self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()['FILTERING']][col_idx]:
                    self.WORKING_CONTEXT.append(f'Filtered {self.SUPER_WORKING_NUMPY_DICT[0]}, {ACTIVE_HEADER} '
                                                    f'by "{_string}", then later deleted the column.', axis=0)

            try: del ACTIVE_HEADER
            except: pass
            # END HANDLE THINGS IN FILTER B4 DELETING COLUMN NAME FROM HEADER ##########################################

            self.dataclass_mlo_loader(self.SUPER_WORKING_NUMPY_LIST[0], name=self.SUPER_WORKING_NUMPY_DICT[0], fxn=fxn)

            self.DataClass.delete_columns(
                                          COL_IDXS_AS_LIST,
                                          HEADER_OR_FULL_SUPOBJ=self.WORKING_SUPOBJS[0],
                                          CONTEXT=self.WORKING_CONTEXT
                                          )

            # ON 7-2-23 BEAR WAS WORKING ON THIS AND WANTED TO DO SOMETHING WITH self.DataClass._cols BUT CANT REMEMBER
            # WHAT IT WANTED TO DO :(
            self.SUPER_WORKING_NUMPY_LIST[0] = self.DataClass.OBJECT
            self.WORKING_SUPOBJS[0] = self.DataClass.HEADER_OR_FULL_SUPOBJ
            if update_context: self.WORKING_CONTEXT = self.DataClass.CONTEXT

            self.DataClass = None

            # DATA IS ALREADY EXPANDED, but KEEP IS NOT EXPANDED DURING ExpandCategories SO CANT DELETE THINGS IN KEEP;
            # COULD DELETE FEATURES WHERE ALL CATEG COLUMNS DISAPPEARED, BUT THATS FOR ANOTHER DAY 6:59 PM 3-22-22


    def run_module_input_tuple(self):
        # tuple of base params that pass into run_module for all ML packages
        return self.standard_config, self.sub_config, self.SUPER_RAW_NUMPY_LIST, self.RAW_SUPOBJS, self.SUPER_WORKING_NUMPY_LIST, \
            self.WORKING_SUPOBJS, self.data_run_orientation, self.target_run_orientation, self.refvecs_run_orientation, \
            self.WORKING_CONTEXT, self.WORKING_KEEP, self.TRAIN_SWNL, self.DEV_SWNL, self.TEST_SWNL, self.split_method, \
            self.LABEL_RULES, self.number_of_labels, self.event_value, self.negative_value, self.conv_kill, self.pct_change, \
            self.conv_end_method, self.rglztn_type, self.rglztn_fctr, self.bypass_validation


    def config_module(self):
        # config_module()                    gets configuration source, returns configuration parameters for particular ML package
        # *VARIABLES = <<module>>
        pass


    def run_module(self):
        # run_module()                       returns run module for particular ML package
        pass
        # USE *self.run_module_input_tuple + WHATEVER SPECIFIC PARAMS FOR CHILD'S run() PARAMS
        # return <<module>>


    def return_fxn_base(self):
        # return_fxn_base()                  values returned from all ML packages
        # 1-28-22 RETURN ALL OBJECTS & INFO HOLDERS TO MAIN SCOPE IN CASE USER MADE CHANGES TO WORKING OBJECTS
        # AND WANTS TO KEEP IN THE MAIN SCOPE FOR ANALYSIS IN ANOTHER ML FUNCTION
        fxn = inspect.stack()[0][3]

        print(f'\n    BEAR IN MLConfigRunTemplate Orienting RAW DATA & TARGET FOR return.  Patience...')

        SRNLOrienterClass = mloo.MLObjectOrienter(
                                                  DATA=self.SUPER_RAW_NUMPY_LIST[0],
                                                  data_given_orientation=self.data_run_orientation,
                                                  data_return_orientation=self.data_given_orientation,
                                                  data_return_format='AS_GIVEN',

                                                  target_is_multiclass=self.raw_target_is_multiclass,
                                                  TARGET=self.SUPER_RAW_NUMPY_LIST[1],
                                                  target_given_orientation=self.target_run_orientation,
                                                  target_return_orientation=self.target_given_orientation,
                                                  target_return_format='AS_GIVEN',

                                                  RETURN_OBJECTS=['DATA','TARGET'],

                                                  bypass_validation=True,
                                                  calling_module=self.this_module,
                                                  calling_fxn=fxn
        )

        self.SUPER_RAW_NUMPY_LIST[0] = SRNLOrienterClass.DATA
        self.SUPER_RAW_NUMPY_LIST[1] = SRNLOrienterClass.TARGET
        del SRNLOrienterClass

        # BECAUSE ObjectOrienter WASNT BUILT TO HANDLE REFVECS
        self.SUPER_RAW_NUMPY_LIST[2] = mlo.MLObject(self.SUPER_RAW_NUMPY_LIST[2], self.refvecs_run_orientation,
            name='REFVECS', return_orientation=self.refvecs_given_orientation, return_format='AS_GIVEN',
            bypass_validation=self.bypass_validation, calling_module=self.this_module, calling_fxn=fxn).OBJECT

        print(f'\n    BEAR IN MLConfigRunTemplate Done Orienting RAW DATA & TARGET FOR return')

        # IF USER MADE BACKUP COPIES, PROMPT TO RESTORE THEM (UNDO ANY CHOPS MADE DURING RUN) OR DISCARD THEM.
        # BACKUPS WERE NEVER ORIENTED TO run, IF KEEPING THEM BYPASS mloo. IF KEEPING CHANGES TO WORKING, PASS THEM
        # THRU mloo BEFORE RETURN.
        if not self.SUPER_WORKING_NUMPY_LIST_BACKUP is None:
            __ = vui.validate_user_str(f'\nRestore working objects to original state(r) or keep as is(k) > ', 'RK')
        else:
            __ = 'K'

        if __ == 'R':

            self.SUPER_WORKING_NUMPY_LIST = self.SUPER_WORKING_NUMPY_LIST_BACKUP
            self.WORKING_SUPOBJS = self.WORKING_SUPOBJS_BACKUP


        elif __ == 'K':
            print(f'\n    BEAR IN MLConfigRunTemplate Orienting WORKING DATA & TARGET FOR return.  Patience...')

            SWNLOrienterClass = mloo.MLObjectOrienter(
                                                      DATA=self.SUPER_WORKING_NUMPY_LIST[0],
                                                      data_given_orientation=self.data_run_orientation,
                                                      data_return_orientation=self.data_given_orientation,
                                                      data_return_format=self.data_given_format,

                                                      target_is_multiclass=self.working_target_is_multiclass,
                                                      TARGET=self.SUPER_WORKING_NUMPY_LIST[1],
                                                      target_given_orientation=self.target_run_orientation,
                                                      target_return_orientation=self.target_given_orientation,
                                                      target_return_format=self.target_given_format,

                                                      RETURN_OBJECTS=['DATA', 'TARGET'],

                                                      bypass_validation=True,
                                                      calling_module=self.this_module,
                                                      calling_fxn=fxn
            )

            self.SUPER_WORKING_NUMPY_LIST[0] = SWNLOrienterClass.DATA
            self.SUPER_WORKING_NUMPY_LIST[1] = SWNLOrienterClass.TARGET
            del SWNLOrienterClass

            # BECAUSE ObjectOrienter WASNT BUILT TO HANDLE REFVECS
            self.SUPER_WORKING_NUMPY_LIST[2] = mlo.MLObject(self.SUPER_WORKING_NUMPY_LIST[2], self.refvecs_run_orientation,
                name='REFVECS', return_orientation=self.refvecs_given_orientation, return_format=self.refvecs_given_format,
                bypass_validation=self.bypass_validation, calling_module=self.this_module, calling_fxn=fxn).OBJECT

            print(f'\n    BEAR IN MLConfigRunTemplate Done Orienting WORKING DATA & TARGET FOR return')

        del self.SUPER_WORKING_NUMPY_LIST_BACKUP, self.WORKING_SUPOBJS_BACKUP

        return self.SUPER_RAW_NUMPY_LIST, self.RAW_SUPOBJS, self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS, \
               self.WORKING_CONTEXT, self.WORKING_KEEP, self.split_method, self.LABEL_RULES, self.number_of_labels, \
               self.event_value, self.negative_value, self.conv_kill, self.pct_change, self.conv_end_method, \
               self.rglztn_type, self.rglztn_fctr


    def return_fxn(self):
        # return_fxn()                       returns user-specified output, in addition to return_fxn_base()
        pass  # SPECIFIED IN CHILDREN, REMEMBER TO *self.return_fxn_base() IN CHILD!


    def sub_post_run_cmds(self):
        # sub_post_run_cmds()                package-specific options available to modify WORKING_DATA after run
        return {}


    def base_post_run_options_module(self):
        # base_post_run_options_module()     holds post run options applied to all ML packages

        fxn = inspect.stack()[0][3]

        if self.post_configrun_select == 'D':  # delete columns from DATA(d)
            # DELETE IS ONLY ALLOWED FOR DATA
            COL_IDXS_TO_DELETE = ls.list_custom_select(self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]], 'idx')
            self.delete_columns(COL_IDXS_TO_DELETE, update_context=True)
            if not self.intcpt_col_idx is None: self.intcpt_col_idx -= sum(np.array(COL_IDXS_TO_DELETE) < self.intcpt_col_idx)
            del COL_IDXS_TO_DELETE

        elif self.post_configrun_select == 'I':  # manage intercept(i)
            self.intercept_manager()
            self.SUPER_WORKING_NUMPY_LIST[0] = self.DataClass.OBJECT
            self.intcpt_col_idx = self.DataClass.intcpt_col_idx


        elif self.post_configrun_select == 'R':  # delete row(r)
            # BEAR 11/3/22 FINISH THIS
            print(f'\n*** POST-RUN DELETE ROW IS NOT AVAILABLE YET ***\n')

        elif self.post_configrun_select == 'T':  # modify target(t)

            target_config = 'Z'   # DONT DELETE THIS, DIDNT BRING IT IN AS A PARAM SO HAVE TO DECLARE HERE

            self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS, self.split_method, self.LABEL_RULES, self.number_of_labels, \
            self.event_value, self.negative_value = \
                tish.TargetInSituHandling(self.standard_config, target_config, self.SUPER_RAW_NUMPY_LIST,
                    self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS, self.split_method, self.LABEL_RULES,
                    self.number_of_labels, self.event_value, self.negative_value).run()

        elif self.post_configrun_select == 'V':  # operations for PERTURBER RESULTS(v)
            # DF columns LOOKS LIKE ['COL IDX', 'STATE', 'COST', '% CHANGE'],   DF index IS WORKING DATA HEADER

            POST_RUN_SUBMENU = {
                                  'p': 'keep only PERTURBED COLUMNS above certain cost',
                                  'y': 'keep only PERTURBED COLUMNS above certain %',
                                  'v': 'delete locked columns',
                                  's': 'sort PERTURBER RESULTS',
                                  'd': 'display PERTURBER RESULTS',
                                  'f': 'dump PERTURBER RESULTS to file',
                                  'a': 'accept and continue',
                              }

            # LOAD PERTURBER_RESULTS FROM Run() IN TestPertuber TO ACCESS MENU OPTIONS, BUT DONT ALLOW build, WOULD OVERWRITE RESULTS
            TestPerturberClass = tp.TestPerturber(self.TEST_SWNL,
                                                  self.WORKING_SUPOBJS[0],
                                                  self.data_run_orientation,
                                                  self.target_run_orientation,
                                                  wb=None,
                                                  bypass_validation=self.bypass_validation
            )

            TestPerturberClass.RESULTS_TABLE = self.PERTURBER_RESULTS
            TestPerturberClass.sort_results_table(sort_values="I", ascending=True)

            disallowed = ""
            if np.sum(TestPerturberClass.RESULTS_TABLE['STATE']=='LOCKED ON') == 0: disallowed += 'v'

            MASK = None

            while True:

                _cols = len(TestPerturberClass.RESULTS_TABLE)  # _cols IS ACTUAL # COLUMNS IN DATA AND PERTURBER_RESULTS.
                                                                 # PERTURBER AND DATA ARE NOT SORTED OR CHOPPED IN ***Run()

                print()
                post_run_sub_cmd = dmp.DictMenuPrint(POST_RUN_SUBMENU, disallowed=disallowed, disp_len=140).select(f'Select letter')

                if post_run_sub_cmd == 'P': # 'keep only PERTURBED COLUMNS above certain cost'
                    TestPerturberClass.sort_results_table(sort_values="C", ascending=False)
                    TestPerturberClass.display_results_table()
                    # MUST STAY IN THIS SORT UNTIL AFTER MASK IS MADE

                    while True:
                        cost_cutoff = vui.validate_user_float(f'\nEnter cost at and above which columns are kept > ')
                        if vui.validate_user_str(f'User entered {cost_cutoff}, accept? (y/n) > ', 'YN') == 'Y': break

                    if cost_cutoff <= TestPerturberClass.RESULTS_TABLE['COST'].min(): print(f'\nAll columns will be kept.\n')
                    elif cost_cutoff > TestPerturberClass.RESULTS_TABLE['COST'].max(): print(f'\nNo columns will be kept.\n')
                    else: print(f"\n{np.sum(TestPerturberClass.RESULTS_TABLE['COST'] < cost_cutoff)} columns of {_cols} will be deleted.\n")

                    __ = vui.validate_user_str(f'User entered keep columns with cost at and above {cost_cutoff}. '
                                                f'Accept(a) Abort(b) > ', 'AB')

                    if __ == 'A':
                        # CHOP PERTURBER RESULTS "COST" COLUMN BY cost_cutoff THEN USE "COL IDX" COLUMN TO MASK TRAIN/DEV/TEST DATA
                        MASK = TestPerturberClass.RESULTS_TABLE.copy()['COL IDX'][TestPerturberClass.RESULTS_TABLE['COST'] >= cost_cutoff].to_numpy().astype(np.int32)
                        MASK = MASK.reshape((1,-1))[0]
                        self.WORKING_CONTEXT += f' Kept {len(MASK)} columns of {_cols} where cost >= {cost_cutoff}.'
                        del cost_cutoff

                    elif __ == 'B':
                        TestPerturberClass.sort_results_table(sort_values="I", ascending=True)
                        continue

                elif post_run_sub_cmd == 'Y': # 'keep only PERTURBED COLUMNS above certain %'
                    TestPerturberClass.sort_results_table(sort_values="S", ascending=False)   # BY SCORES IS EQUIVALENT TO BY PERCENT
                    TestPerturberClass.display_results_table()
                    # MUST STAY IN THIS SORT UNTIL AFTER MASK IS MADE

                    while True:
                        pct_cutoff = vui.validate_user_float(f'\nEnter % change at and above which columns are kept > ')
                        if vui.validate_user_str(f'User entered {pct_cutoff}, accept? (y/n) > ', 'YN') == 'Y': break

                    if pct_cutoff <= TestPerturberClass.RESULTS_TABLE['% CHANGE'].min(): print(f'\nAll columns will be kept.\n')
                    elif pct_cutoff > TestPerturberClass.RESULTS_TABLE['% CHANGE'].max(): print(f'\nNo columns will be kept.\n')
                    else: print(f"\n{np.sum(TestPerturberClass.RESULTS_TABLE['% CHANGE'] < pct_cutoff)} columns of {_cols} will be deleted.\n")

                    __ = vui.validate_user_str(f'User entered keep columns with % change at and above {pct_cutoff}. '
                                               f'Accept(a) Abort(b) > ', 'AB')

                    if __ == 'A':
                        # CHOP PERTURBER RESULTS "% CHANGE" COLUMN BY pct_cutoff THEN USE "COL IDX" COLUMN TO MASK TRAIN/DEV/TEST DATA
                        MASK = TestPerturberClass.RESULTS_TABLE.copy()['COL IDX'][TestPerturberClass.RESULTS_TABLE['% CHANGE'] >= pct_cutoff].to_numpy().astype(np.int32)
                        MASK = MASK.reshape((1,-1))[0]
                        self.WORKING_CONTEXT += f' Kept {len(MASK)} columns of {_cols} where % change >= {pct_cutoff}.'
                        del pct_cutoff

                    elif __ == 'B':
                        TestPerturberClass.sort_results_table(sort_values="I", ascending=True)
                        continue

                elif post_run_sub_cmd == 'V': # 'delete locked columns'
                    TestPerturberClass.sort_results_table(sort_values="I", ascending=False)  # BY SCORES IS EQUIVALENT TO BY PERCENT

                    # CANT GET THIS MENU OPTION IF NO LOCKED COLUMNS, SO NO NEED TO DISPLAY SUCH A CASE
                    if np.sum(TestPerturberClass.RESULTS_TABLE['STATE']=='LOCKED ON') == _cols:
                        print(f'\nNo columns will be kept.\n')
                    else:
                        print(f"\n{np.sum(TestPerturberClass.RESULTS_TABLE['STATE']=='LOCKED ON')} columns of {_cols} will be deleted.\n")

                    __ = vui.validate_user_str(f'Accept(a) Abort(b) > ', 'AB')

                    if __ == 'A':
                        # CHOP PERTURBER RESULTS "STATE" COLUMN BY "LOCKED ON" THEN USE "COL IDX" COLUMN TO MASK TRAIN/DEV/TEST DATA
                        MASK = TestPerturberClass.RESULTS_TABLE.copy()['COL IDX'][TestPerturberClass.RESULTS_TABLE['STATE']!='LOCKED ON'].to_numpy().astype(np.int32)
                        MASK = MASK.reshape((1,-1))[0]
                        self.WORKING_CONTEXT += f' Deleted {len(MASK)} columns of {_cols} where perturbation was locked on.'

                    elif __ == 'B':
                        continue


                elif post_run_sub_cmd == 'S': # 'sort PERTURBER RESULTS'
                    TestPerturberClass.sort_results_table()
                elif post_run_sub_cmd == 'D': # 'display PERTURBER RESULTS'
                    TestPerturberClass.display_results_table()
                elif post_run_sub_cmd == 'F': # 'dump PERTURBER RESULTS to file'
                    TestPerturberClass.dump_to_file()


                if post_run_sub_cmd in 'PYV':
                    # 'p': 'keep only PERTURBED COLUMNS abpve certain cost'
                    # 'y': 'keep only PERTURBED COLUMNS above certain %'
                    # 'v': 'delete locked columns'

                    CHOP_DICT = dict((("P", "COST"), ("Y", "PERCENT CHANGE"), ("V", "'LOCKED ON'")))

                    while True:   # TO ALLOW ABORT ONLY

                        # MUST BE PUT INTO 'COL IDX' ASC SO THAT MASK IDXS (MADE FROM 'COL IDX' VALUES) MATCH UP AGAINST iloc
                        TestPerturberClass.sort_results_table(sort_values="I", ascending=True)

                        if vui.validate_user_str(f'\nMASK IS GOING TO CHOP DATA TO {len(MASK)} COLUMNS FROM {_cols} BASED ON '
                             f'{CHOP_DICT[post_run_sub_cmd]} MASK. Proceed? (y/n) > ', 'YN') == 'N':
                            break

                        # PERTURBER_RESULTS HAS COME OUT OF ***Run() WITH THE RESULTS FOR ALL COLUMNS
                        # THE P & Y STEPS ABOVE SORT PERTURBER_RESULTS BASED ON USER PICK "TOP COST" OR "TOP PERCENT" AND CREATE
                        # A MASK TO BE APPLIED TO DATA AND PERTURBER_RESULTS.

                        if (not self.intcpt_col_idx is None) and (self.intcpt_col_idx not in MASK):
                            __ = vui.validate_user_str(f'\nIntercept is not in kept columns and will be deleted. '
                                                       f'Allow? (y/n) > ', 'YN')

                            if __ == 'Y': self.intcpt_col_idx = None
                            elif __ == 'N': MASK = np.insert(MASK, 0, self.intcpt_col_idx, axis=0)

                        if self.intcpt_col_idx in MASK:
                            MASK = np.insert(MASK[MASK!=self.intcpt_col_idx], 0, self.intcpt_col_idx, axis=0)      # MOVE INTERCEPT TO FIRST

                        # APPLY MASK TO PERTURBER RESULTS AND DATA IN TRAIN/TEST (& DEV IF AVAILABLE) ###############################
                        TestPerturberClass.RESULTS_TABLE = TestPerturberClass.RESULTS_TABLE.iloc[MASK, :]
                        # RESET PERTURBER_RESULTS IDXS IN 'COL IDX'
                        TestPerturberClass.RESULTS_TABLE.loc[:, 'COL IDX'] = np.arange(0, len(MASK), dtype=np.int32)
                        self.PERTURBER_RESULTS = TestPerturberClass.RESULTS_TABLE.copy()

                        # IF MASK CURRENTLY REPRESENTS THE CURRENT STATE OF DATA, BYPASS COLUMN PULL CODE
                        if not np.array_equiv(range(len(self.WORKING_SUPOBJS[0][0])), MASK):

                            # 6-17-23, CANT DELETE COLUMNS IN KEEP (see delete_columns()) SO NOTING IN "CONTEXT"
                            # THE COLUMNS THAT FAILED PERTURBATION
                            ACTV_HDR = self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]]
                            # CAN USE _cols HERE BECAUSE NUM ROWS IN RESULTS AND COLS IN DATA IS ALWAYS EQUAL IN THIS MODULE
                            for col_idx in range(_cols):
                                if col_idx not in MASK:
                                    self.WORKING_CONTEXT.append(f'Deleted DATA - {ACTV_HDR} for failing feature perturbation.')
                            del ACTV_HDR

                            # CHOP WORKING_SUPOBJS TO MASK
                            self.WORKING_SUPOBJS[0] = self.WORKING_SUPOBJS[0][..., MASK]

                            # USE GYMNASTICS TO CHOP WORKING DATA, TRAIN DATA, DEV DATA, AND TEST DATA TO MASK
                            NAMES = ('WORKING_DATA', 'TRAIN_DATA', 'DEV_DATA', 'TEST_DATA')
                            DATA_OBJS = (self.SUPER_WORKING_NUMPY_LIST, self.TRAIN_SWNL, self.DEV_SWNL, self.TEST_SWNL)

                            for idx, (name, DATA_OBJ) in enumerate(zip(NAMES, DATA_OBJS)):

                                # PASS OBJECTS THAT ARE EMPTY, WOULD EXCEPT WHEN TRYING TO INDEX INTO IT
                                if np.array_equiv(DATA_OBJ, []): continue

                                self.dataclass_mlo_loader(DATA_OBJ[0], name=name, fxn=fxn)

                                WINNING_COLUMNS_HOLDER = \
                                    self.DataClass.return_columns(MASK, return_orientation='AS_GIVEN', return_format='AS_GIVEN')

                                self.DataClass = None

                                if idx == 0: self.SUPER_WORKING_NUMPY_LIST[0] = WINNING_COLUMNS_HOLDER
                                if idx == 1: self.TRAIN_SWNL[0] = WINNING_COLUMNS_HOLDER
                                if idx == 2: self.DEV_SWNL[0] = WINNING_COLUMNS_HOLDER
                                if idx == 3: self.TEST_SWNL[0] = WINNING_COLUMNS_HOLDER

                            del NAMES, DATA_OBJS, WINNING_COLUMNS_HOLDER

                            print(f'\n*** DATA OBJECTS AND PERTURBATION RESULTS SUCCESSFULLY CHOPPED ON {CHOP_DICT[post_run_sub_cmd]} ***\n')

                        # END APPLY MASK TO PERTURBER RESULTS AND DATA IN TRAIN/TEST (& DEV IF AVAILABLE) ###############################

                        self.WORKING_CONTEXT += f' Retained top {len(MASK)} perturbed feature columns based on {CHOP_DICT[post_run_sub_cmd]}.'

                        if not self.intcpt_col_idx is None: self.intcpt_col_idx = 0
                        # 6/17/23 MASK IS FORCING INTERCEPT TO 0 IDX

                        print(f'\n*** DELETE OF NON-WINNING COLUMNS FROM DATA SUCCESSFUL ***\n')

                        MASK = None

                        break

                    del CHOP_DICT

                if post_run_sub_cmd == 'A': # 'accept
                    break

            del TestPerturberClass, POST_RUN_SUBMENU, MASK, _cols, post_run_sub_cmd



    def sub_post_run_options_module(self):
        # sub_post_run_options_module()      holds post run options unique to particular ML package
        pass


    def configrun(self):
        # configrun()                        runs config_module() & run_module()

        while True:
            ################################################################################################################
            # CONFIG #######################################################################################################
            # CHANGE self.CONFIG_MODULE_RETURN TO WHATEVER FOR PARTICULAR ML PACKAGE
            self.config_module()
            # END CONFIG ####################################################################################################
            ################################################################################################################

            while True:

                ################################################################################################################
                # RUN ##########################################################################################################
                self.run_module()
                # END GMLR RUN #################################################################################################
                ################################################################################################################

                disallowed = ''

                while True:

                    if len(self.PERTURBER_RESULTS)==0: disallowed += 'v'
                    else: disallowed = "".join([_ for _ in disallowed if _ != 'v'])

                    self.post_configrun_select = dmp.DictMenuPrint(self.ALL_POST_RUN_CMDS,
                                                                   disallowed=disallowed,
                                                                   disp_len=140).select(f'Select letter')   # ALL ARE ALLOWED

                    ################################################################################################################
                    # POST-RUN OPTIONS #############################################################################################
                    self.base_post_run_options_module()
                    self.sub_post_run_options_module()
                    # END POST-RUN OPTIONS #########################################################################################
                    ################################################################################################################

                    # accept/return to ML menu(a)  run again without reconfig(b)  run again with reconfig(c)
                    if self.post_configrun_select in ['A','B','C']: break

                if self.post_configrun_select == 'B': continue
                elif self.post_configrun_select in ['A','C']: break

            if self.post_configrun_select == 'C': continue
            elif self.post_configrun_select == 'A': break

        return self.return_fxn()














































