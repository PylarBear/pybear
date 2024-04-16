import sys
import numpy as np
from copy import deepcopy
from data_validation import arg_kwarg_validater as akv, validate_user_input as vui
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from debug import get_module_name as gmn
from general_data_ops import get_shape as gs
from MLObjects import MLObject as mlo
from MLObjects.SupportObjects import master_support_object_dict as msod




class ImproperShapeError(Exception): pass

class DatatypeError(Exception): pass


class ApexDataHandling:

    def __init__(self, SXNL, data_given_orientation, target_given_orientation, refvecs_given_orientation, FULL_SUPOBJS,
                 CONTEXT, KEEP, calling_module=None, calling_fxn=None, bypass_validation=None):

        # NOTES
        # REMEMBER THAT .CONTEXT FOR ApexDataHandling IS DIFFERENT FROM .CONTEXT FOR Data/Target/RefVecsClass!
        # CHANGES TO Data/Target/RefVecsClass THAT INVOLVE FULL_SUPOBJS MUST EXPLICITY BE ACCESSED AND OVERWRITE ApexDataHandling.SUPOBJS!




        # PreRunDataAugment
        # standard_config, user_manual_or_standard, augment_method, SWNL, data_given_orientation, target_given_orientation,
        # refvecs_given_orientation, WORKING_SUPOBJS, CONTEXT, KEEP, bypass_validation

        # return SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS, self.CONTEXT, self.KEEP


        # PreRunExpand
        # standard_config, user_manual_or_standard, expand_method, SUPER_RAW_NUMPY_LIST, data_given_orientation,
        # target_given_orientation, refvecs_given_orientation, RAW_SUPOBJS, CONTEXT, KEEP, bypass_validation

        # return SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, self.CONTEXT, self.KEEP


        # PreRunFilter
        # standard_config, user_manual_or_standard, filter_method, SUPER_RAW_NUMPY_LIST, data_given_orientation,
        # target_given_orientation, refvecs_given_orientation, FULL_SUPOBJS, CONTEXT, KEEP,

        #                  # *******************************************
        #                  SUPER_RAW_NUMPY_LIST_BASE_BACKUP,
        #                  FULL_SUPOBJS_BASE_BACKUP,
        #                  CONTEXT_BASE_BACKUP,
        #                  KEEP_BASE_BACKUP,
        #                  # *******************************************
        #                  SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP,
        #                  FULL_SUPOBS_GLOBAL_BACKUP,
        #                  KEEP_GLOBAL_BACKUP
        #                                                                                    bypass_validation

        # SXNL, SUPOBJS, self.CONTEXT, self.KEEP, self.VALIDATED_DATATYPES_GLOBAL_BACKUP, self.MODIFIED_DATATYPES_GLOBAL_BACKUP

        self.calling_module = calling_module if not calling_module is None else gmn.get_module_name(str(sys.modules[__name__]))
        calling_fxn = calling_fxn if not calling_fxn is None else '__init__'

        self.bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                         self.calling_module, calling_fxn, return_if_none=False)

        self.SXNL_DICT = dict(((0, "DATA"), (1, "TARGET"), (2, "REFVECS")))

        # VALIDATE ALL SXNL OBJECTS IN-PLACE
        for idx, OBJ in enumerate(SXNL): SXNL[idx] = ldv.list_dict_validater(OBJ, self.SXNL_DICT[idx])[1]

        # VALIDATE given_orientations
        self.data_given_orientation = akv.arg_kwarg_validater(data_given_orientation, 'data_given_orientation',
                                              ['ROW', 'ARRAY'], self.calling_module, calling_fxn)
        self.target_given_orientation = akv.arg_kwarg_validater(target_given_orientation, 'target_given_orientation',
                                                ['ROW', 'ARRAY'], self.calling_module, calling_fxn)
        self.refvecs_given_orientation = akv.arg_kwarg_validater(refvecs_given_orientation, 'refvecs_given_orientation',
                                                 ['ROW', 'ARRAY'], self.calling_module, calling_fxn)

        # GET rows & cols
        self.data_rows, self.data_cols = gs.get_shape(self.SXNL_DICT[0], SXNL[0], self.data_given_orientation)

        _ORIENTS = (self.data_given_orientation, self.target_given_orientation, self.refvecs_given_orientation)

        # VALIDATE INDIVIDUAL SUPOBJS
        for idx, _OBJ in FULL_SUPOBJS:
            FULL_SUPOBJS[idx] = ldv.list_dict_validater(FULL_SUPOBJS[idx], f'{self.SXNL_DICT[idx]}_SUPOBJ')[1]


        if not self.bypass_validation:
            # VALIDATE SUPPORT OBJECTS FOR CORRECT ROWS FOR FULL AND FOR CORRECT COLUMNS ###############################
            full_len = len(msod.master_support_object_dict())
            for idx, _orient in enumerate(_ORIENTS):
                _shape = FULL_SUPOBJS[idx].shape
                if _shape[0] != full_len:
                    raise ImproperShapeError(f'{self.calling_module}.{calling_fxn}() >>> '
                             f'{self.SXNL_DICT[idx]}_SUPOBJ HAS INCORRECT ROWS ({len(FULL_SUPOBJS[idx])}) FOR A FULL SUPOBJ ({full_len})')
                main_object_shape = gs.get_shape(self.SXNL_DICT[idx], SXNL[idx], _ORIENTS[idx])[1]
                if _shape[1] != main_object_shape[1]:
                    raise ImproperShapeError(f'{self.calling_module}.{calling_fxn}() >>> '
                             f'{self.SXNL_DICT[idx]} SUPPORT OBJECT COLUMNS ({_shape[1]}) DOES NOT MATCH MAIN OBJECT COLUMNS ({main_object_shape[1]}).')

            del full_len, _shape, main_object_shape
            # END VALIDATE SUPPORT OBJECTS FOR CORRECT ROWS FOR FULL AND FOR CORRECT COLUMNS ###############################

        # CREATE MLObject CLASSES FOR ALL OBJECTS IN SXNL ############################

        for idx in range(len(SXNL)):

            CreatorClass = mlo.MLObject(
                                        SXNL[idx],
                                        _ORIENTS[idx],
                                        name=self.SXNL_DICT[0],
                                        return_orientation='AS_GIVEN',
                                        return_format='AS_GIVEN',
                                        bypass_validation=self.bypass_validation,
                                        calling_module=self.calling_module,
                                        calling_fxn=calling_fxn
            )

            if idx == 0: self.DataClass = deepcopy(CreatorClass)
            elif idx == 1: self.TargetClass = deepcopy(CreatorClass)
            elif idx == 2: self.RefVecsClass = deepcopy(CreatorClass)

        del _ORIENTS, CreatorClass
        # END CREATE MLObject CLASSES FOR ALL OBJECTS IN SXNL ############################

        del SXNL

        # BEAR PUT NOTES ERE
        self.DataClass.HEADER_OR_FULL_SUPOBJ = FULL_SUPOBJS[0].copy()
        self.TargetClass.HEADER_OR_FULL_SUPOBJ = FULL_SUPOBJS[1].copy()
        self.RefVecsClass.HEADER_OR_FULL_SUPOBJ = FULL_SUPOBJS[2].copy()

        self.CONTEXT = CONTEXT
        self.KEEP = KEEP

    # END __init__ ##################################################################################################################
    #################################################################################################################################
    #################################################################################################################################

    # is_dict                               (True if data object is dict)
    # is_list                               (True if data object is list, tuple, ndarray)
    # print_preview                         (print data object)
    # print_cols_and_setup_parameters       (prints column name, validated type, user type, min cutoff, "other", scaling, lag start, lag end, & filtering for all columns in all objects)
    # delete_rowS                           (delete rows from DATA, TARGET, & REFVECS)
    # delete_column                         (delete column from DATA)
    # insert_column                         (append column to DATA)
    # intercept_verbage                     (text added to CONTEXT when INTERCEPT is appended)




    def is_list(self):
        return isinstance(self.DataClass.OBJECT, (list, tuple, np.ndarray))


    def is_dict(self):
        return isinstance(self.DataClass.OBJECT, dict)


    def object_and_column_select(self):
        pass




    def arg_kwarg_validator(self, DICT_OF_INTS={}, DICT_OF_ARRAYS_OF_INTS={}, DICT_OF_BOOLS={}):
        # MUST CONVERT TO LIST FOR CONTEXT UPDATE
        except_text = lambda obj, dtype, exc_type: exc_type(f'{obj} MUST BE {dtype}')
        
        for k, v in DICT_OF_INTS.items():
            if not 'INT' in str(type(v)).upper(): raise except_text(k, 'int', DatatypeError)

        for k, v in DICT_OF_ARRAYS_OF_INTS.items():
            try: v = np.array(v).reshape((1,))
            except: raise except_text(v, 'A LIST-TYPE THAT CAN BE CONVERTED INTO ndarray', DatatypeError)
            if not 'INT' in str(v.dtype).upper(): raise except_text(k, 'A LIST-TYPE OF ints', DatatypeError)

        for k, v in DICT_OF_BOOLS.items():
            if not isinstance(v, bool): raise except_text(k, 'bool', DatatypeError)

        del except_text


    def delete_rows(self, ROW_IDXS_AS_INT_OR_LIST, update_context=False):

        if not self.bypass_validation:
            self.arg_kwarg_validator(DICT_OF_ARRAYS_OF_INTS={'ROW_IDXS_AS_INT_OR_LIST': ROW_IDXS_AS_INT_OR_LIST},
                                     DICT_OF_BOOLS={'update_context': update_context}
             )

        # MUST CONVERT TO LIST FOR CONTEXT UPDATE
        ROW_IDXS_AS_INT_OR_LIST = np.array(ROW_IDXS_AS_INT_OR_LIST).reshape((1,))

        # ARG VALIDATION HANDLED IN MLObject

        self.DataClass.delete_rows(ROW_IDXS_AS_INT_OR_LIST)
        self.TargetClass.delete_rows(ROW_IDXS_AS_INT_OR_LIST)
        self.RefVecsClass.delete_rows(ROW_IDXS_AS_INT_OR_LIST)

        # self.FULL_SUPOBJS DID NOT CHANGE, EVEN IF IT DID, IT'S EXTRACTED FROM THE CLASSES AT THE END ANYWAY
    
        if update_context:
            for row_idx in ROW_IDXS_AS_INT_OR_LIST:
                self.CONTEXT.append(f'Deleted row {row_idx}.')


    def insert_row(self, ROW_IDXS_AS_INT_OR_LIST, update_context=False):
        pass

    def delete_column(self, obj_idx=None, col_idx=None, update_context=False):
        pass






    def insert_column(self, obj_idx=None, col_idx=None, update_context=False):
        pass


    def append_intercept(self, update_context=False):

        # BEAR DOES AN intercept_finder NEED TO BE HERE, OR CAN insert_standard_intercept HANDLE IT
        # DONT THINK insert_standard_intercept CAN HANDLE IT, JUST ADDS AN INTERCEPT IF U SAY SO

        # LOOKS LIKE find(manage)_intercept(columns of constants)() IS GOING TO BECOME A METHOD OF MLObject,
        # WHICH MEANS IT COULD BE USED DIRECTLY ON DataClass, THEN WHETHER TO ACTUALLY APPEND INTERCEPT
        # OR NOT COULD WOULD BE MANAGED HERE.



        # BEAR TIRED 6/30/23 7:11 PM.
        # DONE --- TAKE INTERCEPT-FINDING CODE OUT OF MLConfigRun AND PUT IT IN ITS OWN MODULE IN general_data_ops
        # DONE --- & MAKE 1 FOR MLObjects.
        # CALL THE ML FIND INTERCEPT MODULE HERE. (MLObjects.intercept_manager or ML_find_constants)
        # DONE --- GO TO GMLR/MLR/MI AND FIX THE CALLS TO ML_find_intercept.
        # DONE, BEAR THINKS ---MAYBE THERE IS A DIFFERENCE BETWEEN "FINDING" COLUMNS OF CONSTANTS (WHICH IS WHAT
        # ML_find_constants CURRENTLY DOES) & MANAGING COLUMNS OF CONSTANTS (WHICH IS WHAT WAS BEING DONE IN MLConfigRun)
        # BEAR HAS MADE ML_find_constants() ML_manage_intercept() AND CREATED AN intercept_manager METHOD IN MLObjects.

        # BEAR HAS TO DO OTHER THINGS 7/2/23 6:20 PM.
        # THINGS TO PICK UP:
        # --- PROOF intercept_manager() IN MLObject
        # --- LOOK AT ALL THE "CONTEXTS" BETWEEN MLRowColumnOperations / MLObject / MLConfigRUn AND WHEREVER ELSE
                # LOOK FOR REDUNDANCY, NECESSITY OF KWARGS, TRY TO STREAMLINE
        # --- DEAL WITH ALL THE BEARS LIVING IN MLObject & MLRowColumnOperations
        # --- FINISH / RUN MLObject_intercept_manager__TEST
        # --- TEST MLConfigRunTemplate ET AL


        if not self.bypass_validation:
            self.arg_kwarg_validator(DICT_OF_BOOLS={'update_context': update_context})

        self.DataClass.insert_standard_intercept()

        if update_context:
            # BEAR FIGURE OUT HOW TO HANDLE THIS, PROBABLY WILL HAVE TO DO THIS MANUALLY HERE, AND IGNORE ALL THE MLObject.CONTEXTS
            self.CONTEXT.append(f'Appended intercept to data.')


    def select_obj_and_column(self):



        '''

        TAKEN FROM PreRunExpand

        def select_column(self):
            COL_IDX = ls.list_custom_select(self.DATA_OBJECT_HEADER_WIP[0], 'idx')
            return COL_IDX
        '''



        '''
        TAKEN FROM PreRunFilter

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

        '''




    def column_description(self, obj_idx, col_idx):

        if not self.bypass_validation:
            self.arg_kwarg_validator(DICT_OF_INTS={'obj_idx': obj_idx, 'col_idx': col_idx})

        msod_hdr_idx = msod.QUICK_POSN_DICT()['HEADER']

        if obj_idx == 0: ACTV_HDR = self.DataClass.HEADER_OR_FULL_SUPOBJ[msod_hdr_idx]
        elif obj_idx == 1: ACTV_HDR = self.TargetClass.HEADER_OR_FULL_SUPOBJ[msod_hdr_idx]
        elif obj_idx == 2: ACTV_HDR = self.RefVecsClass.HEADER_OR_FULL_SUPOBJ[msod_hdr_idx]

        __ = f'{self.SXNL_DICT[obj_idx]} - {ACTV_HDR[col_idx]}'
        return __

        del ACTV_HDR

        '''
        FROM PreRunFilter (THE ONLY ONE THAT HAD column_desc)

        def column_desc(self, obj_idx, col_idx):
            __ = f'{self.SXNL_DICT[obj_idx]} - {self.SUPER_RAW_NUMPY_LIST[obj_idx + 1][0][col_idx]}'
            return (__)
        '''



    def rename_column(self, obj_idx=None, col_idx=None, update_context=False):
        "Modifies obj_idx's header in-place and returns column name"

        # BEAR PUT SELECTORS IF None
        if vui.validate_user_str(f'\nSHOW COLUMN NAMES FOR {self.SXNL_DICT[obj_idx]}? (y/n) > ', 'YN') == 'Y':
            # BEAR PUT A COLUMN PRINT-OUT HERE
            pass



        if not self.bypass_validation:
            self.arg_kwarg_validator(DICT_OF_INTS={"obj_idx": obj_idx, "col_idx": col_idx},
                                    DICT_OF_BOOLS={"update_context": update_context})

        msod_hdr_idx = msod.QUICK_POSN_DICT()["HEADER"]

        if obj_idx == 0: ACTV_HDR = self.DataClass.HEADER_OR_FULL_SUPOBJ[msod_hdr_idx]
        if obj_idx == 1: ACTV_HDR = self.TargetClass.HEADER_OR_FULL_SUPOBJ[msod_hdr_idx]
        if obj_idx == 2: ACTV_HDR = self.RefVecsClass.HEADER_OR_FULL_SUPOBJ[msod_hdr_idx]
        
        old_name = ACTV_HDR[col_idx]

        while True:
            new_name = vui.user_entry(f'\nENTER NEW COLUMN HEADER (CURRENT HEADER IS "{old_name}") > ')

            if new_name in ACTV_HDR:
                __ = vui.validate_user_str(f'{new_name} IS ALREADY IN {self.SXNL_DICT[obj_idx]}. ENTER A DIFFERENT COLUMN NAME(e) ABORT(a).', 'EA')
                if __ == 'E': continue
                elif __ == 'A': del ACTV_HDR, msod_hdr_idx, new_name, __; return old_name

            __ = vui.validate_user_str(f'USER ENTERED "{new_name}" --- ACCEPT? (y/n/(a)bort) > ', 'YNA')

            if __ == 'A': del ACTV_HDR, msod_hdr_idx, new_name, __; return old_name
            elif __ == 'N': continue
            elif __ == 'Y':
                if obj_idx == 0:
                    self.DataClass.HEADER_OR_FULL_SUPOBJ[msod_hdr_idx][col_idx] = new_name
                    if old_name in self.KEEP: self.KEEP[np.argwhere(self.KEEP==old_name)] = new_name
                elif obj_idx == 1:
                    self.TargetClass.HEADER_OR_FULL_SUPOBJ[msod_hdr_idx][col_idx] = new_name
                elif obj_idx == 2:
                    self.RefVecsClass.HEADER_OR_FULL_SUPOBJ[msod_hdr_idx][col_idx] = new_name

                del ACTV_HDR, msod_hdr_idx, old_name, __

                return new_name



        '''
        FROM PreRunFilter  (THE ONLY ONE THAT HAD rename_column)

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
        '''



    def base_return_fxn(self):

        SXNL = [self.DataClass.OBJECT, self.TargetClass.OBJECT, self.RefVecsClass.OBJECT]

        FULL_SUPOBJS = [self.DataClass.HEADER_OR_FULL_SUPOBJ, self.TargetClass.HEADER_OR_FULL_SUPOBJ,
                        self.RefVecsClass.HEADER_OR_FULL_SUPOBJ]

        del self.DataClass.OBJECT, self.TargetClass.OBJECT, self.RefVecsClass.OBJECT

        return SXNL, FULL_SUPOBJS, self.CONTEXT, self.KEEP















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














