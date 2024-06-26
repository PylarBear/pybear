import sys, time
import numpy as np, sparse_dict as sd
from debug import get_module_name as gmn
from copy import deepcopy
from MLObjects.SupportObjects import master_support_object_dict as msod
from MLObjects.TestObjectCreators import ApexCreate as ac
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv

'''OBJECT, HEADER, AND SUPPORT OBJECTS ARE ATTRIBUTES OF THE CLASS. NOTHING IS RETURNED.'''

# MODULE FOR CREATING DATA, TARGET, REFVECS, OR TEST OBJECTS FROM GIVEN OBJECTS


# OUTLINE


# INHERITS #############################################################################################################
# _exception
# validate_full_supobj_or_single_mdtypes
# build_full_supobj
# get_individual_support_objects
# to_row
# to_column
# _transpose
# to_array
# to_sparse_dict
# expand

# OVERWRITTEN ##########################################################################################################
# build



class CreateFromGiven(ac.ApexCreate):    # ApexCreate FOR METHODS ONLY, DONT DO super()!
    def __init__(self,
                 OBJECT,
                 given_orientation,
                 return_format='AS_GIVEN',
                 return_orientation='AS_GIVEN',
                 name=None,
                 OBJECT_HEADER=None,
                 FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                 override_sup_obj=None,
                 bypass_validation=None
                 ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        allowed_return_format = ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN']
        allowed_return_orientation = ['ROW', 'COLUMN', 'AS_GIVEN']
        super().__init__(OBJECT_HEADER, FULL_SUPOBJ_OR_SINGLE_MDTYPES, bypass_validation, name, return_format, allowed_return_format,
                        return_orientation, allowed_return_orientation, override_sup_obj, self.this_module, fxn)
        del allowed_return_format, allowed_return_orientation

        # DECLARED self.this_module, fxn
        # super() VALIDATED self.bypass_validation, self.name, self.return_format, self.return_orientation, self.override_sup_obj,
        #       self.calling_module, self.calling_fxn, self.mdtype_idx, self.FULL_SUPOBJ_OR_SINGLE_MDTYPES
        #       BUILT self.LOOKUP_DICT


        self.given_format, self.OBJECT = ldv.list_dict_validater(OBJECT, name)   # NOT AVAILABLE AS ATTR FROM md.ModifiedDatatypes
        if self.given_format is None:
            self._exception(f'PASSED OBJECT ARG MUST BE AN ARRAY OR SPARSE DICT, CANNOT BE None.', fxn=fxn)

        # OBJECT & given_orientation MUST ALWAYS BE GIVEN
        if self.bypass_validation:
            self.given_orientation = given_orientation
        elif not self.bypass_validation:
            self.given_orientation = \
                akv.arg_kwarg_validater(given_orientation, 'given_orientation', ['ROW', 'COLUMN'], self.this_module, fxn)


        # DECLARED self.this_module, fxn
        # super() VALIDATED self.bypass_validation, self.name, self.return_format, self.return_orientation, self.override_sup_obj,
        #       self.calling_module, self.calling_fxn, self.mdtype_idx, self.FULL_SUPOBJ_OR_SINGLE_MDTYPES
        #       BUILT self.LOOKUP_DICT
        # DECLARED / VALIDATED self.given_format, self.given_orientation


        if self.given_format == 'ARRAY': self.is_list, self.is_dict = True, False
        elif self.given_format == 'SPARSE_DICT': self.is_list, self.is_dict = False, True

        #################################################################################################################
        # GET DIMENSIONS OF OBJECT #####################################################################################

        if self.is_list and self.given_orientation == 'ROW': self.columns = len(self.OBJECT[0])
        elif self.is_list and self.given_orientation == 'COLUMN': self.columns = len(self.OBJECT)
        elif self.is_dict and self.given_orientation == 'ROW': self.columns = sd.inner_len_quick(self.OBJECT)
        elif self.is_dict and self.given_orientation == 'COLUMN': self.columns = sd.outer_len(self.OBJECT)
        else: self._exception(fxn, f'LOGIC MANAGING ASSIGNMENT OF self.columns_in, self.rows_in BASED ON is_list/dict '
                                    f'AND self.given_orientation IS FAILING')
        # END GET DIMENSIONS OF OBJECT #####################################################################################
        #############################################################################################################

        self.validate_full_supobj_or_single_mdtypes()
        # self.FULL_SUPOBJ_OR_SINGLE_MDTYPES WAS CONVERTED TO FULL SUPOBJ

        self.build_full_supobj()
        # self.SUPPORT_OBJECTS POPULATED, AND WHATEVER OTHER VALIDATION TOOK PLACE (CONTENT, SIZE AGAINST OBJECT/HEADER ETC.)

        self.get_individual_support_objects()
        self.CONTEXT = []



        # DECLARED self.this_module, fxn, built self.LOOKUP_DICT, validated self.bypass_validation
        # DECLARED / VALIDATED self.name, self.given_format, self.given_orientation, self.return_format, self.return_orientation, self.override_sup_obj
        # VALIDATED SIZE OF HEADER, OBJECT, FULL_SUPOBJ_OR_SINGLE_MDTYPES WITH bfso/ModifiedDatatypes BY WAY OF ApexSupportObjectHandle
        # PROCESSED FULL_SUPOBJ_OR_SINGLE_MDTYPES INTO FULL SUPOBJ (W CONTENT & SIZE VALIDATION), SET ALL INDIVIDUAL SUPPORT OBJECTS
        # DECLARED / VALIDATED self.is_list, self.is_dict


        # self.OBJECT_HEADER CANNOT BE None, WOULD BE DUMMY FROM bfso IN build_full_supobj IF HEADER WAS NOT GIVEN
        # OVERWRITE HEADER DUMMY BUILT BY ModifiedDatatypes IF HEADER WAS NOT GIVEN
        if OBJECT_HEADER is None:    # NOT self.OBJECT_HEADER !!!!
            for col_idx in range(self.columns):
                self.OBJECT_HEADER[0][col_idx] = f'{self.name[:3]}_{self.MODIFIED_DATATYPES[col_idx][:3]}{str(col_idx+1)}'

            self.SUPPORT_OBJECTS[self.LOOKUP_DICT['HEADER']] = self.OBJECT_HEADER[0]


        # DECLARED self.this_module, fxn, built self.LOOKUP_DICT, validated self.bypass_validation
        # DECLARED / VALIDATED self.name, self.given_orientation, self.return_format, self.return_orientation, self.override_sup_obj
        # VALIDATED SIZE OF HEADER, OBJECT, FULL_SUPOBJ_OR_SINGLE_MDTYPES WITH ModifiedDatatypes BY WAY OF ApexSupportObjectHandle
        # PROCESSED FULL_SUPOBJ_OR_SINGLE_MDTYPES INTO FULL SUPOBJ (W CONTENT & SIZE VALIDATION), SET ALL INDIVIDUAL SUPPORT OBJECTS
        # DECLARED / VALIDATED self.given_format, self.is_list, self.is_dict
        # ESTABLISHED self.columns, self.rows
        # IF HEADER WAS NOT GIVEN, OVERWROTE DUMMY MADE BY ModifiedDatatypes


        # IF A STR TYPE IS NOT IN M_DTYPES, OBJECT IS EQUIVALENT OF EXPANDED OBJECT
        self.is_expanded = True not in (_ in list(msod.mod_text_dtypes().values()) for _ in self.MODIFIED_DATATYPES)


        self.current_format = self.given_format
        self.current_orientation = self.given_orientation

        # BUILD OBJECT ##################################################################################################
        self.build()

        # END __init__ ###############################################################################################################
        #############################################################################################################################
        #############################################################################################################################


    # INHERITS
    # _exception
    # validate_full_supobj_or_single_mdtypes
    # build_full_supobj
    # get_individual_support_objects
    # to_row
    # to_column
    # _transpose
    # to_array
    # to_sparse_dict
    # expand


    def build(self):
        #########################################################################################################################
        # BUILD ##################################################################################################################
        '''3/7/23 The net effect of build() is to take a given object and change its formatting based on what's in given
            (or given and overrided) MODIFIED_DATATYPES .'''

        # self.OBJECT MUST EXIST, REQUIRED arg
        # OBJECT IS STILL IN GIVEN FORMAT AND ORIENTATION


        # IF IS ARRAY AND CONTAINS STR TYPES, CANNOT BE CONVERTED TO SD
        if self.current_format=='ARRAY' and \
                (True in (_ in list(msod.mod_text_dtypes().values()) for _ in self.MODIFIED_DATATYPES)) and \
                self.return_format=='SPARSE_DICT':
            print(f'\n*** OBJECT CONTAINS STR-TYPE VALUES, CANNOT BE CONVERTED TO SPARSE_DICT, RETURNING AS ARRAY ***\n')
            self.return_format = 'ARRAY'

        # IF OBJECT IS GIVEN AS SD, DONT RISK BLOWUP BY EXPANDING
        # UNDER ANY given_format, MAKE ORIENTATION BE COLUMN TO PERFORM THESE OPERATIONS
        if self.current_orientation == 'ROW': self.to_column()

        # NOTES 3/7/23 WHEN GIVEN OBJECT IS FLOAT, BUT RETURN IS EITHER INT OR BIN,
        # FORCES -0.5 < x < 0.5 TO ZERO, THUS INCREASING THE ORIGINAL SPARSITY.

        if self.current_format=='SPARSE_DICT' and self.return_format=='SPARSE_DICT':
            # current_format DOES NOT CHANGE

            def dict_builder(_COLUMN, _dtype):
                KEYS = np.fromiter(_COLUMN.keys(), dtype=np.int32)
                if _dtype == 'BIN': VALUES = np.fromiter(_COLUMN.values(), dtype=bool).astype(np.int8)
                elif _dtype == 'INT': VALUES = np.fromiter(_COLUMN.values(), dtype=np.int32)
                elif _dtype == 'FLOAT': VALUES = np.fromiter(_COLUMN.values(), dtype=np.float64)
                return dict((zip(KEYS.tolist(), VALUES.tolist())))

            for col_idx in range(self.columns):
                if self.MODIFIED_DATATYPES[col_idx]=='BIN': self.OBJECT[int(col_idx)] = dict_builder(self.OBJECT[col_idx], 'BIN')
                elif self.MODIFIED_DATATYPES[col_idx] == 'INT': self.OBJECT[int(col_idx)] = dict_builder(self.OBJECT[col_idx], 'INT')
                elif self.MODIFIED_DATATYPES[col_idx] == 'FLOAT': self.OBJECT[int(col_idx)] = dict_builder(self.OBJECT[col_idx], 'FLOAT')

            del dict_builder

        elif self.return_format=='ARRAY':  # and self.current_format in ['ARRAY', 'SPARSE_DICT']
            self.to_array()
            for col_idx in range(self.columns):
                if self.MODIFIED_DATATYPES[col_idx] == 'BIN':
                    self.OBJECT[col_idx] = self.OBJECT[col_idx].astype(bool).astype(np.int8)
                elif self.MODIFIED_DATATYPES[col_idx] in 'INT':
                    self.OBJECT[col_idx] = self.OBJECT[col_idx].astype(np.int32)
                elif self.MODIFIED_DATATYPES[col_idx] == 'FLOAT':
                    self.OBJECT[col_idx] = self.OBJECT[col_idx].astype(np.float64)
                elif self.MODIFIED_DATATYPES[col_idx] in msod.mod_text_dtypes():
                    self.OBJECT[col_idx] = self.OBJECT[col_idx].astype(str)

        elif self.current_format=='ARRAY' and self.return_format=='SPARSE_DICT':  # MUST BE ALL NUMBERS

            DUM_SD_OBJECT = {}   # CREATE A RECEPTACLE IF THIS IS BEING RETURNED AS DICT
            for col_idx in range(self.columns-1,-1,-1):
                if self.MODIFIED_DATATYPES[col_idx] == 'BIN':
                    self.OBJECT[col_idx] = self.OBJECT[col_idx].astype(bool).astype(np.int8)
                    DUM_SD_OBJECT[int(col_idx)] = sd.zip_list_as_py_int(self.OBJECT[col_idx].reshape((1,-1)))[0]
                elif self.MODIFIED_DATATYPES[col_idx] in 'INT':
                    self.OBJECT[col_idx] = self.OBJECT[col_idx].astype(np.int32)
                    DUM_SD_OBJECT[int(col_idx)] = sd.zip_list_as_py_int(self.OBJECT[col_idx].reshape((1,-1)))[0]
                elif self.MODIFIED_DATATYPES[col_idx] == 'FLOAT':
                    self.OBJECT[col_idx] = self.OBJECT[col_idx].astype(np.float64)
                    DUM_SD_OBJECT[int(col_idx)] = sd.zip_list_as_py_float(self.OBJECT[col_idx].reshape((1,-1)))[0]

                self.OBJECT = np.delete(self.OBJECT, col_idx, axis=0)

            del self.OBJECT  # SHOULD HAVE BEEN EMPTIED DURING SD BUILD ABOVE
            self.current_format, self.is_list, self.is_dict = 'SPARSE_DICT', False, True
            # REORDER SD SINCE WAS ASSEMBLED BY ITERATING OVER OBJECT BACKWARDS ABOVE
            self.OBJECT = {int(ok):DUM_SD_OBJECT.pop(ok) for ok in sorted(list(DUM_SD_OBJECT.keys()))}

            del DUM_SD_OBJECT

        # ALREADY ORIENTATION=='COLUMN', FROM OPERATIONS ABOVE
        # CORRENT return_format SHOULD HAVE BEEN ACHIEVED DURING PROCESSING
        if self.return_orientation=='ROW': self.to_row()

        # END BUILD #############################################################################################################
        #########################################################################################################################















if __name__ == '__main__':

    # TEST MODULE

    # 3/14/23 BEAR VERIFIED CODE & TEST MODULE ARE GOOD

    from MLObjects.SupportObjects import ModifiedDatatypes as md
    from MLObjects.TestObjectCreators import CreateCategorical as cc
    '''
    # COMMENT THIS OUT TO DO ITERATIVE TESTING ############################################################################
    return_format = 'ARRAY'
    return_orientation = 'COLUMN'
    # GIVEN_OBJECT = crsn.create_random_sparse_numpy(0,10,(3,5),50,np.float64)
    GIVEN_OBJECT = cc.CreateCategorical(name='SINGLE TEST', columns=3, rows=5, return_orientation='COLUMN',
                                        NUMBER_OF_CATEGORIES_AS_LIST=[3,3,3])
    GIVEN_OBJECT = GIVEN_OBJECT.OBJECT
    OBJECT_HEADER = ['BIG','BAGEL','ROCKS']
    FULL_SUPOBJ_OR_SINGLE_MDTYPES = ['STR','STR','STR']   #['INT','INT','INT']
    override_sup_obj = False
    bypass_validation = True

    SUPPORT_OBJECTS = msod.build_empty_support_object(3)
    SUPPORT_OBJECTS[2] = FULL_SUPOBJ_OR_SINGLE_MDTYPES

    TestClass = CreateFromGiven(GIVEN_OBJECT,
                                 'COLUMN',
                                 return_format=return_format,
                                 return_orientation=return_orientation,
                                 name='DATA',
                                 OBJECT_HEADER=OBJECT_HEADER,
                                 FULL_SUPOBJ_OR_SINGLE_MDTYPES=SUPPORT_OBJECTS,
                                 override_sup_obj=override_sup_obj,
                                 bypass_validation=bypass_validation
                                 )

    OBJECT = TestClass.OBJECT
    SupportObjects = TestClass.SUPPORT_OBJECTS

    print(f'OBJECT B4 EXPAND:')
    print(pd.DataFrame(columns=TestClass.SUPPORT_OBJECTS[2],
                       data=np.array(TestClass.OBJECT) if return_orientation=='ROW' else np.array(TestClass.OBJECT).transpose(),
                       index=None))
    print()
    print(f'SUPOBJ B4 EXPAND:')
    print(pd.DataFrame(columns=TestClass.OBJECT_HEADER[0],
                       data=np.array(TestClass.SUPPORT_OBJECTS),
                       index=None))

    TestClass.expand(expand_as_sparse_dict=False, auto_drop_rightmost_column=False)

    print(f'OBJECT AFTER EXPAND:')
    print(pd.DataFrame(columns=TestClass.SUPPORT_OBJECTS[2],
                       data=np.array(TestClass.OBJECT) if return_orientation=='ROW' else np.array(TestClass.OBJECT).transpose(),
                       index=None))
    print()
    print(f'SUPOBJ AFTER EXPAND:')
    print(pd.DataFrame(columns=TestClass.OBJECT_HEADER[0],
                       data=np.array(TestClass.SUPPORT_OBJECTS),
                       index=None))

    quit()
    # END COMMENT THIS OUT TO DO ITERATIVE TESTING ########################################################################
    '''









    # EVERYTHING BELOW IS A TEST MODULE W SUPPORTING FUNCTIONS
    # TESTS FORMATS, ORIENTATIONS, MODIFIED_DTYPES, CONGRUITY OF OBJECT/SUPOBJS. DOES NOT TEST CONTENTS OF OBJ IN ANY WAY.
    # VERIFIED 3/8/23, TEST CODE AND MODULE ARE GOOD.  REMEMBER THE reverse_it FUNCTIONALITY FOR TEST LOOPS.

    from MLObjects.TestObjectCreators import test_header as th, CreateNumerical as cn, CreateCategorical as cc
    from general_sound import winlinsound as wls

    def test_exc_handle(OBJECT, reason_text):
        time.sleep(1)
        print(f'\n\033[91mEXCEPTING OBJECT:\033[0m\x1B[0m')
        print(OBJECT)
        print()
        wls.winlinsound(888, 500)
        print(f'\n\033[91mWANTS TO RAISE EXCEPTION FOR \033[0m\x1B[0m')
        print(reason_text)
        _quit = vui.validate_user_str(f'\n\033[91mquit(q) or continue(c) > \033[0m\x1B[0m', 'QC')
        if _quit == 'Q': raise Exception(f'\033[91m*** ' + reason_text + f' ***\033[0m\x1B[0m')
        elif _quit == 'C': pass


    def test_lens(OBJECT, given_orientation, return_orientation, rows, columns):
        if isinstance(OBJECT, dict):
            if sd.is_sparse_outer(OBJECT): outer_len, inner_len = sd.outer_len(OBJECT), sd.inner_len(OBJECT)
            elif sd.is_sparse_inner(OBJECT):
                raise Exception(f'\n\033[91m*** OBJECT IS A SINGLE VECTOR NOT [[ ]]  🤬 ***\033[0m\x1B[0m\n')
                # outer_len, inner_len = 1, len(OBJECT)
        elif isinstance(OBJECT, np.ndarray):
            _shape = OBJECT.shape
            if len(_shape) == 2: outer_len, inner_len = _shape[0], _shape[1]
            elif len(_shape) == 1:
                raise Exception(f'\n\033[91m*** OBJECT IS A SINGLE VECTOR NOT [[ ]]  🤬 ***\033[0m\x1B[0m\n')
                # outer_len, inner_len = 1, len(OBJECT)

        if return_orientation == 'ROW':
            if outer_len != rows:
                test_exc_handle(OBJECT, f'\033[91mouter/inner len: MISMATCH BETWEEN outer_len ({outer_len}) AND rows ({rows}) FOR given_orient={given_orientation}, return_orient={return_orientation}\033[0m\x1B[0m')
            if inner_len != columns:
                test_exc_handle(OBJECT, f'\033[91mouter/inner len: MISMATCH BETWEEN inner_len ({inner_len}) AND columns ({columns}) FOR given_orent={given_orientation}, return_orient={return_orientation}\033[0m\x1B[0m')
        elif return_orientation == 'COLUMN':
            if outer_len != columns:
                test_exc_handle(OBJECT, f'\033[91mouter/inner len: MISMATCH BETWEEN outer_len ({outer_len}) AND columns ({columns}) FOR given_orent={given_orientation}, return_orient={return_orientation}\033[0m\x1B[0m')
            if inner_len != rows:
                test_exc_handle(OBJECT, f'\033[91mouter/inner len: MISMATCH BETWEEN inner_len ({inner_len}) AND rows ({rows}) FOR given_orent={given_orientation}, return_orient={return_orientation}\033[0m\x1B[0m')

        else: raise Exception(f'\n\033[91m*** given_orientation LOGIC FOR DETERMINING CORRECT outer/inner_len IS FAILING ***\033[0m\x1B[0m\n')


    def test_format(OBJECT, return_format):
        if isinstance(OBJECT, dict): __ = 'SPARSE_DICT'
        elif isinstance(OBJECT, np.ndarray): __ = 'ARRAY'
        else: test_exc_handle(OBJECT, f'\n\033[91m*** BIG DISASTER. OBJECT IS NOT AN ARRAY OR SPARSE_DICT ***\033[0m\x1B[0m\n')

        if __ != return_format:
            test_exc_handle(OBJECT,
                f'\n\033[91m*** return_format: OBJECT FORMAT ({__}) IS NOT THE REQUIRED FORMAT ({return_format})!!!! ***\033[0m\x1B[0m\n')


    def test_min_and_max(OBJECT, min_value, max_value):
        # GET EXPECTED MIN & MAX ##########################################################################################
        exp_min, exp_max = min_value, max_value

        # GET ACTUAL MIN & MAX ##########################################################################################
        if isinstance(OBJECT, dict):
            act_min, act_max = sd.min_(OBJECT), sd.max_(OBJECT)
        elif isinstance(OBJECT, (np.ndarray)):
            act_min, act_max = np.min(OBJECT), np.max(OBJECT)
        # END GET ACTUAL MIN & MAX ##########################################################################################

        if act_min < exp_min or act_max > exp_max:
            test_exc_handle(OBJECT,
                f'\n\033[91m*** min max: EXPECTED OUTPUT W min={min_value}, max={max_value}, GOT min={act_min}, max={act_max}\033[0m\x1B[0m')


    def test_dtype(OBJECT, return_orientation, EXP_M_DTYPES):

        # GET TYPE FROM OBJECT #####################################################################################
        if sd.is_sparse_inner(OBJECT): NEW_OBJECT = {0: deepcopy(OBJECT)}
        else: NEW_OBJECT = OBJECT

        DumClass = md.ModifiedDatatypes(
                                            OBJECT=NEW_OBJECT,
                                            object_given_orientation=return_orientation,  # REMEMBER OBJECT ALREADY PASSED THRU CreateFromGiven!
                                            columns=None,
                                            OBJECT_HEADER=None,
                                            SUPPORT_OBJECT=None,
                                            VALIDATED_DATATYPES=None,
                                            prompt_to_override=False,
                                            return_support_object_as_full_array=False,
                                            bypass_validation=False,
                                            calling_module='CreateFromGiven',
                                            calling_fxn='test_module'
                                    )

        ACT_M_DTYPES = DumClass.SUPPORT_OBJECT

        del DumClass

        if not np.array_equiv(EXP_M_DTYPES, ACT_M_DTYPES):
            test_exc_handle(OBJECT, f'\n\033[91m*** dtypes: OUTPUT IS {ACT_M_DTYPES}, \nSHOULD BE {EXP_M_DTYPES} ***\033[0m\x1B[0m\n')


    def test_header(HEADER, columns):
        if len(HEADER[0]) != columns:
            test_exc_handle(HEADER, f'\n\033[91m*** HEADER LENGTH ({len(HEADER[0])}) DOES NOT EQUAL NUMBER OF COLUMNS ({columns})\033[0m\x1B[0m')


    # ############################################################################################################################
    # #############################################################################################################################
    # ITERATIVE TESTING ###################################################################################################################

    name = 'DATA'


    GIVEN_FORMAT = ['ARRAY', 'SPARSE_DICT']
    GIVEN_ORIENTATION = ['ROW', 'COLUMN']

    RETURN_FORMAT = ['ARRAY', 'SPARSE_DICT']
    RETURN_ORIENTATION = ['ROW', 'COLUMN']

    COLUMNS = [200,20,1]
    ROWS = [1,10,200]
    MIN_VALUE = [2, 0, -5]
    MAX_VALUE = [10, 10, 6]

    BYPASS_VALIDATION = [True, False]

    DATA_OBJECTS = [
                   'create_random_np_numbers',
                   'create_random_np_strings',
                   'create_random_np_hybrid',
                   'create_random_sd_numbers'
                   # None   # 3/11/2023 VERIFIED EXCEPTS FOR OBJECT==None
                   ]

    DATA_OBJECT_HEADERS = [
                           'built_during_run',
                           None
                           ]
    BIN_INT_OR_FLOAT = ['INT', 'BIN', 'FLOAT']

    total_itrs = np.product(list(map(len, (RETURN_FORMAT, RETURN_ORIENTATION, GIVEN_FORMAT, GIVEN_ORIENTATION,
                                           COLUMNS, MIN_VALUE, DATA_OBJECTS, DATA_OBJECT_HEADERS, BYPASS_VALIDATION)
                                          )))

    reverse_it = False

    ctr = 0
    for return_format in RETURN_FORMAT if not reverse_it else reversed(RETURN_FORMAT):
        for return_orientation in RETURN_ORIENTATION if not reverse_it else reversed(RETURN_ORIENTATION):
            for given_format in GIVEN_FORMAT if not reverse_it else reversed(GIVEN_FORMAT):
                for given_orientation in GIVEN_ORIENTATION if not reverse_it else reversed(GIVEN_ORIENTATION):
                    for _columns, _rows in zip(COLUMNS, ROWS):
                        for min_value, max_value in zip(MIN_VALUE, MAX_VALUE):
                            for data_obj_desc in DATA_OBJECTS if not reverse_it else reversed(DATA_OBJECTS):
                                for data_obj_hdr_desc in DATA_OBJECT_HEADERS if not reverse_it else reversed(DATA_OBJECT_HEADERS):
                                    for bypass_validation in BYPASS_VALIDATION if not reverse_it else reversed(BYPASS_VALIDATION):
                                        ctr += 1

                                        if ctr % 1 == 0:
                                            print(f'\n\nRunning tests {ctr} of {total_itrs}...')

                                        #############################################################################################
                                        # BUILD CONDITIONAL HEADER OBJECT ###########################################################
                                        if data_obj_hdr_desc is None: GIVEN_HEADER = None
                                        else: GIVEN_HEADER = th.test_header(_columns)
                                        # END BUILD CONDITIONAL HEADER OBJECT #######################################################
                                        #############################################################################################

                                        ############################################################################################
                                        # BUILD CONDITIONAL DATA OBJECTS ############################################################
                                        if data_obj_desc == 'create_random_np_numbers':
                                            DummyObject = cn.CreateNumerical(name=name, OBJECT=None,
                                                                           OBJECT_HEADER=GIVEN_HEADER,
                                                                           given_orientation=None,
                                                                           columns=_columns,
                                                                           rows=_rows, return_format=given_format,
                                                                           return_orientation=given_orientation,
                                                                           bin_int_or_float='FLOAT',
                                                                           min_value=min_value, max_value=max_value,
                                                                           _sparsity=0)

                                            GIVEN_OBJECT = DummyObject.OBJECT

                                            if GIVEN_HEADER is None:
                                                GIVEN_HEADER = DummyObject.OBJECT_HEADER
                                            #else: GIVEN_HEADER = GIVEN_HEADER

                                            EXP_MODIFIED_DATATYPES = ['FLOAT' for _ in range(_columns)]

                                        elif data_obj_desc=='create_random_np_strings':
                                            # MUST BE GIVEN (IE CREATED HERE) AND RETURNED AS ARRAY BECAUSE HAS STR TYPES
                                            given_format = 'ARRAY'
                                            return_format = 'ARRAY'

                                            DummyObject = cc.CreateCategorical(name=name, OBJECT=None,
                                                                               OBJECT_HEADER=GIVEN_HEADER,
                                                                               given_orientation=None,
                                                                               columns=_columns,
                                                                               rows=_rows,
                                                                               return_orientation=given_orientation,
                                                                               NUMBER_OF_CATEGORIES_AS_LIST=[5 for _ in range(_columns)])

                                            GIVEN_OBJECT = DummyObject.OBJECT
                                            if GIVEN_HEADER is None:
                                                GIVEN_HEADER = DummyObject.OBJECT_HEADER
                                            # else: GIVEN_HEADER = GIVEN_HEADER
                                            EXP_MODIFIED_DATATYPES = ['STR' for _ in range(_columns)]

                                        elif data_obj_desc=='create_random_np_hybrid':

                                            # MUST BE GIVEN (IE CREATED HERE) AS AND RETURNED ARRAY BECAUSE HAS STR TYPES
                                            given_format = 'ARRAY'
                                            return_format = 'ARRAY'

                                            DummyObject1 = cn.CreateNumerical(name=name, OBJECT=None,
                                                                              OBJECT_HEADER=GIVEN_HEADER,
                                                                              given_orientation=None,
                                                                              columns=_columns, rows=_rows,
                                                                              return_format=given_format,
                                                                              return_orientation=given_orientation,
                                                                              bin_int_or_float='FLOAT',
                                                                              min_value=min_value, max_value=max_value,
                                                                              _sparsity=0)

                                            DummyObject2 = cc.CreateCategorical(name=name, OBJECT=None,
                                                                                OBJECT_HEADER=GIVEN_HEADER,
                                                                                given_orientation=None,
                                                                                columns=_columns, rows=_rows,
                                                                                return_orientation=given_orientation,
                                                                                NUMBER_OF_CATEGORIES_AS_LIST=[5 for _ in range(_columns)])

                                            # OBJECT_HEADER DOESNT CHANGE
                                            _nums_idx = int(np.ceil(_columns/2))
                                            _strs_idx = int(np.floor(_columns/2))

                                            if GIVEN_HEADER is None:
                                                GIVEN_HEADER1 = DummyObject1.OBJECT_HEADER
                                                GIVEN_HEADER2 = DummyObject2.OBJECT_HEADER
                                                GIVEN_HEADER = np.hstack((GIVEN_HEADER1[..., :_nums_idx], GIVEN_HEADER2[..., :_strs_idx]))
                                                del GIVEN_HEADER1, GIVEN_HEADER2
                                            # else: GIVEN_HEADER = GIVEN_HEADER

                                            if given_orientation=='COLUMN':
                                                GIVEN_OBJECT = np.vstack((DummyObject1.OBJECT[:_nums_idx].astype(object),
                                                                         DummyObject2.OBJECT[:_strs_idx].astype(object)))
                                            elif given_orientation=='ROW':
                                                GIVEN_OBJECT = np.hstack((DummyObject1.OBJECT[..., :_nums_idx].astype(object),
                                                                         DummyObject2.OBJECT[..., :_strs_idx].astype(object)))

                                            EXP_MODIFIED_DATATYPES = np.hstack((DummyObject1.MODIFIED_DATATYPES[:_nums_idx],
                                                                                DummyObject2.MODIFIED_DATATYPES[:_strs_idx]))

                                        elif data_obj_desc=='create_random_sd_numbers':
                                            GIVEN_OBJECT = sd.create_random_py_float(min_value, max_value,
                                                                                    (_columns if given_orientation=='COLUMN' else _rows,
                                                                                     _rows if given_orientation=='COLUMN' else _columns),
                                                                                     0)

                                            # GIVEN_HEADER = GIVEN_HEADER
                                            EXP_MODIFIED_DATATYPES = ['FLOAT' for _ in range(_columns)]

                                        elif data_obj_desc is None:
                                            GIVEN_OBJECT = None
                                            # GIVEN_HEADER = GIVEN_HEADER
                                            EXP_MODIFIED_DATATYPES = None

                                        # END BUILD CONDITIONAL DATA OBJECTS ########################################################
                                        #############################################################################################

                                        obj_desc = f"\nINCOMING DATA OBJECT IS {data_obj_desc}" + \
                                                   [f"(min={min_value} max={max_value}) WITH {_rows} ROWS AND {_columns} COLUMNS ORIENTED AS {given_orientation}. "
                                                    if not data_obj_desc is None else " "][0] + F"AND HEADER IS \n{data_obj_hdr_desc}." + \
                                                   f"\nOBJECT SHOULD BE RETURNED AS A {return_format} OF {data_obj_desc} " \
                                                   f"(min={min_value} max={max_value}) WITH {_rows} ROWS AND {_columns} COLUMNS ORIENTED AS {return_orientation}."

                                        print(f'\033[92m{obj_desc}\033[0m')


                                        TestClass = CreateFromGiven(
                                                                     GIVEN_OBJECT,
                                                                     given_orientation,
                                                                     return_format=return_format,
                                                                     return_orientation=return_orientation,
                                                                     name=name,
                                                                     OBJECT_HEADER=GIVEN_HEADER,
                                                                     FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                                                     override_sup_obj=None,
                                                                     bypass_validation=bypass_validation
                                                     )

                                        ACT_OBJECT = TestClass.OBJECT
                                        ACT_HEADER = TestClass.OBJECT_HEADER
                                        del TestClass

                                        test_format(ACT_OBJECT, return_format)
                                        test_dtype(ACT_OBJECT, return_orientation, EXP_MODIFIED_DATATYPES)
                                        if data_obj_desc not in ['create_random_np_strings', 'create_random_np_hybrid', None]:
                                            test_min_and_max(ACT_OBJECT, min_value, max_value)
                                        test_lens(ACT_OBJECT, given_orientation, return_orientation, _rows, _columns)
                                        test_header(ACT_HEADER, _columns)

    wls.winlinsound(888, 500)
    time.sleep(0.5)
    wls.winlinsound(888, 500)
    time.sleep(0.5)
    wls.winlinsound(888, 500)

    print(f'\n\033[92m*** VALIDATION COMPLETED SUCCESSFULLY ***\033[0m\x1B[0m\n')

    # END ITERATIVE TESTING ###################################################################################################################
    # ############################################################################################################################
    # #############################################################################################################################




























