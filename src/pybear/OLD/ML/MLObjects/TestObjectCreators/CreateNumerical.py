import inspect, sys, time
import numpy as np, sparse_dict as sd
from debug import get_module_name as gmn
from copy import deepcopy
from general_data_ops import create_random_sparse_numpy as crsn
from MLObjects.SupportObjects import master_support_object_dict as msod, BuildFullSupportObject as bfso, FullSupportObjectSplitter as fsos
from MLObjects.TestObjectCreators import test_header as th, ExpandCategoriesTestObjects as ecto
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv

'''OBJECT, HEADER, AND SUPPORT OBJECTS ARE ATTRIBUTES OF THE CLASS. NOTHING IS RETURNED.'''

# MODULE FOR CREATING DATA, TARGET, REFVECS, OR TEST OBJECTS W RANDOM NUMERICAL DATA OR FROM GIVEN OBJECTS
# CAN ONLY INGEST AN OBJECT THAT IS ENTIRELY NUMERICAL (WHEN GIVEN) AND CAN ONLY CREATE AN OBJECT THAT IS ENTIRELY NUMERICAL

# FUNCTIONS #############################################################################################################
# _exception
# build
# to_row
# to_column
# to_array
# to_sparse_dict
# expand


# PARENT OF CreateCategorical
class CreateNumerical:    # GENERATOR OF ALL NUMERICAL OBJECTS CREATED (DAT, TAR, REF, TST)

    def __init__(self, name='DATA', OBJECT=None, OBJECT_HEADER=None, given_orientation=None, columns=None, rows=None,
                 return_format='ARRAY', return_orientation='ROW', bin_int_or_float='INT', min_value=0, max_value=9,
                 _sparsity=0):
        # THE INTENT IS THAT columns IS ONLY GIVEN IF OBJECT & HEADER ARE NOT
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self._sparsity = _sparsity
        self.is_expanded = True   # IS NUMERICAL, SO UNEXPANDED AND EXPANDED STATES ARE EQUAL
        self.LOOKUP_DICT = {k: msod.master_support_object_dict()[k]['position'] for k in msod.master_support_object_dict()}

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        self.given_orientation = \
                akv.arg_kwarg_validater(given_orientation, 'given_orientation', ['ROW', 'COLUMN', None], self.this_module, fxn)
        self.return_format = \
                akv.arg_kwarg_validater(return_format, 'return_format', ['ARRAY', 'SPARSE_DICT'], self.this_module, fxn)
        self.return_orientation = \
                akv.arg_kwarg_validater(return_orientation, 'return_orientation', ['ROW', 'COLUMN'], self.this_module, fxn)
        self.bin_int_or_float = \
                akv.arg_kwarg_validater(bin_int_or_float, 'bin_int_or_float', ['BIN', 'INT', 'FLOAT', 'CAT', None], self.this_module, fxn)
                # 2/1/23 COP OUT 'CAT' TO ALLOW CreateCategorical TO CALL THIS AS super(), SHOULD NOT BE GIVEN AS KWARG FOR CreateNumerical

        # ESTABLISHED return_format, return_orientation, bin_int_or_float ARE GOOD

        self.OBJECT_HEADER = ldv.list_dict_validater(OBJECT_HEADER, "OBJECT_HEADER")[1]

        # ESTABLISHED HAS HEADER OR NOT

        #####################################################################################################################
        #####################################################################################################################
        # PROCESS OBJECT BUILD PARAMETERS ###################################################################################

        if not OBJECT is None:
            #################################################################################################################
            # IF OBJECT IS GIVEN, GET ATTRIBUTES OF OBJECT AND OVERWRITE column AND row INFO IF GIVEN #######################

            self.given_format, self.OBJECT = ldv.list_dict_validater(OBJECT, 'OBJECT')

            if self.given_format=='SPARSE_DICT': self.is_dict, self.is_list = True, False
            elif self.given_format=='ARRAY': self.is_dict, self.is_list = False, True
            else: self.is_dict, self.is_list = False, False

            # IF OBJECT GIVEN, OVERWRITE ANYTHING USER PUT IN columns AND rows
            if not rows is None or not columns is None:
                print(f'\n*** AN OBJECT IS GIVEN AND rows AND/OR columns ARE SPECIFIED. DIMENSIONS OF GIVEN OBJECT SUPERCEDE. ***\n')

            if self.given_orientation is None:
                self._exception(fxn, 'IF OBJECT IS GIVEN, given_orientation MUST ALSO BE SPECIFIED.')
            elif self.is_list and self.given_orientation == 'ROW':
                self.columns_in, self.rows_in = len(self.OBJECT[0]), len(self.OBJECT)
            elif self.is_list and self.given_orientation == 'COLUMN':
                self.columns_in, self.rows_in = len(self.OBJECT), len(self.OBJECT[0])
            elif self.is_dict and self.given_orientation == 'ROW':
                self.columns_in, self.rows_in = sd.inner_len_quick(self.OBJECT), sd.outer_len(self.OBJECT)
            elif self.is_dict and self.given_orientation == 'COLUMN':
                self.columns_in, self.rows_in = sd.outer_len(self.OBJECT), sd.inner_len_quick(self.OBJECT)
            else:
                self._exception(fxn, f'LOGIC MANAGING ASSIGNMENT OF self.columns_in, self.rows_in BASED ON is_list/dict '
                                f'AND self.given_orientation IN __init__ IS FAILING')

            self.columns_out, self.rows_out = self.columns_in, self.rows_in

            # VALIDATE len(HEADER) == len(OBJECT) W-R-T GIVEN OBJECT ORIENTATION
            if not self.OBJECT_HEADER is None and self.columns_in != len(self.OBJECT_HEADER[0]):
                self._exception(fxn, f'GIVEN HEADER LEN ({len(self.OBJECT_HEADER[0])}) DOES NOT MATCH GIVEN '
                     f'OBJECT LEN ({len(self.OBJECT) if self.given_orientation=="COLUMN" else len(self.OBJECT[0])}) '
                     f'W-R-T GIVEN OBJECT ORIENTATION')

            # ESTABLISHED return_format, return_orientation, bin_int_or_float ARE GOOD
            # ESTABLISHED HAS HEADER OR NOT
            # ESTABLISHED OBJECT GIVEN, is_list, is_dict, columns_in, rows_in, columns_out, rows_out

            # END IF OBJECT IS GIVEN, GET ATTRIBUTES OF OBJECT AND OVERWRITE column AND row INFO IF GIVEN ###############
            #############################################################################################################

        elif OBJECT is None:
            #############################################################################################################
            # IF OBJECT IS NOT GIVEN, GET BUILD PARAMS FROM OTHER GIVEN INFO #############################################

            self.OBJECT = None

            self.is_list, self.is_dict = False, False

            ### VALIDATE USER INPUT rows ###################################################################################
            if rows is None or (columns is None and self.OBJECT_HEADER is None):
                self._exception(fxn, 'IF OBJECT IS NOT GIVEN, rows MUST ALWAYS BE GIVEN, AND (columns OR OBJECT_HEADER) MUST BE GIVEN.')

            try:
                _ = int(rows)
                if _ != rows or rows < 1: raise Exception
                self.rows_in = _
                self.rows_out = _
                del _
            except:
                self._exception(fxn, 'INVALID rows. MUST BE INTEGER GREATER THAN ZERO.')
            ### END VALIDATE USER INPUT rows ######################################################################################


            ### VALIDATE columns ##############################################################################################
            # IF HEADER IS GIVEN, USE THAT TO DETERMINE columns, IN PLACE OF WHATEVER MAY HAVE BEEN ENTERED FOR columns
            if not self.OBJECT_HEADER is None:
                _ = int(len(self.OBJECT_HEADER[0]))
                self.columns_in = _
                self.columns_out = _
                del _

            # IF OBJECT_HEADER NOT GIVEN, columns MUST BE ENTERED
            elif self.OBJECT_HEADER is None:
                try:
                    _ = int(columns)
                    if _ != columns or columns < 1: raise Exception
                    self.columns_in = _
                    self.columns_out = _
                except:
                    self._exception(fxn, 'INVALID columns. COLUMNS MUST BE ENTERED IF (OBJECT AND HEADER NOT GIVEN) AND MUST '
                                    'BE INTEGER GREATER THAN ZERO.')
            ### END VALIDATE columns ##############################################################################################

            # END IF OBJECT IS NOT GIVEN, GET BUILD PARAMS FROM OTHER GIVEN INFO ########################################
            #############################################################################################################

            # ESTABLISHED return_format, return_orientation, bin_int_or_float ARE GOOD
            # ESTABLISHED HAS HEADER OR NOT
            # ESTABLISHED OBJECT NOT GIVEN, is_list, is_dict, columns_in, rows_in, columns_out, rows_out

        # END PROCESS OBJECT BUILD PARAMETERS ###########################################################################
        #################################################################################################################
        #################################################################################################################

        # ESTABLISHED return_format, return_orientation, bin_int_or_float ARE GOOD
        # ESTABLISHED HEADER GIVEN OR NOT GIVEN
        # ESTABLISHED OBJECT GIVEN OR NOT GIVEN, is_list is_dict, columns_in, rows_in, columns_out, rows_out

        #################################################################################################################
        # BUILD HEADER IF NOT GIVEN #####################################################################################
        if self.OBJECT_HEADER is None:
            self.OBJECT_HEADER = np.fromiter(
                (f'{self.name[:3]}_{self.bin_int_or_float}{str(idx + 1)}' for idx in range(self.columns_out)), dtype='<U1000').reshape((1,-1))
        elif not self.OBJECT_HEADER is None:
            # RELIC CODE BELOW, AS OF 12/27/22 VALIDATION ABOVE TO ENSURE GIVEN HEADER IS EXACT LEN
            # CUTS IF TOO LONG OR KEEPS SAME IF CORRECT LENGTH
            self.OBJECT_HEADER = self.OBJECT_HEADER[..., :self.columns_out]
            # USED TO EXPAND W DUMS IF TOO SHORT, NOW EXCEPTS
            for idx in range(len(self.OBJECT_HEADER[0]), self.columns_out):
                self._exception(fxn, f'WANTS TO LENGTHEN HEADER IN build_object_header')
                self.OBJECT_HEADER = np.insert(self.OBJECT_HEADER, idx, f'{self.name[:3]}_{self.bin_int_or_float}{idx+1}', axis=1)
        # END BUILD HEADER IF NOT GIVEN  ################################################################################
        #################################################################################################################


        #################################################################################################################
        # BUILD OBJECT ##################################################################################################
        self.MODIFIED_DATATYPES = None   # PLACEHOLDER FOR FILL DURING build
        self.build()
        # END BUILD OBJECT ##############################################################################################
        #################################################################################################################

        # HERE JUST IN CASE EVER WANT TO GO TO OBJECT AS MLObject INSTANCE
        # ObjectClass = mlo.MLObject(self.OBJECT, given_orientation, return_orientation='AS_GIVEN', return_format='AS_GIVEN',
        #                            bypass_validation=True, calling_module=None, calling_fxn=None)

        #################################################################################################################
        # BUILD REMAINING SUPPORT OBJECTS #########################################################################################

        # BUILD REMAINING SUPPORT OBJECTS USING SupportObjectHandle.  MUST BE ABLE TO GENERATE VAL/MOD DTYPES
        # IF AN OBJECT IS PASSED AS KWARG (AS OPPOSED TO BUILDING FROM SCRATCH)
        # IN build(), self.OBJECT WAS CREATED OR LEFT AS GIVEN, AND self.MODIFIED_DATATYPES WAS EITHER CREATED (IF POSSIBLE)
        # OR SET TO None

        SupObjClass = bfso.BuildFullSupportObject(OBJECT=self.OBJECT,
                                             object_given_orientation=self.return_orientation,
                                             OBJECT_HEADER=self.OBJECT_HEADER,
                                             SUPPORT_OBJECT=None,
                                             columns=self.columns_out,
                                             quick_vdtypes=True,
                                             MODIFIED_DATATYPES=self.MODIFIED_DATATYPES,
                                             print_notes=False,
                                             prompt_to_override=False,
                                             bypass_validation=True,
                                             calling_module=self.this_module,
                                             calling_fxn=fxn)

        # ASSIGN INDIVIDUAL SUPPORT OBJECTS
        # fsos INSTANTIATES self.SUPPORT_OBJECTS ,self.OBJECT_HEADER, self.VALIDATED_DATATYPES, self.VALIDATED_DATATYPES,
        # self.FILTERING, self.MIN_CUTOFFS, self.USE_OTHER, self.START_LAG, self.END_LAG, self.SCALING
        fsos.FullSupObjSplitter.__init__(self, SupObjClass.SUPPORT_OBJECT, bypass_validation=True)
        self.CONTEXT = []
        self.KEEP = deepcopy(self.OBJECT_HEADER[0])

        del SupObjClass

        # END BUILD REMAINING SUPPORT OBJECTS #####################################################################################
        #################################################################################################################


    def _exception(self, fxn, text):
        raise Exception(f'\n*** {self.this_module}.{fxn} {text} ***\n')


    def build(self):
        #########################################################################################################################
        #BUILD ##################################################################################################################
        '''12/27/22 The net effect of build() is to take a given object and change its numbers based on bin_int_or_float
            or create a new object from scratch based on the size, sparsity, and bin_int_or_float parameters that are given.'''
        if self.OBJECT is None:  # CREATE FROM SCRATCH
            if self.return_orientation == 'COLUMN': shape_tuple = (self.columns_out, self.rows_out)
            elif self.return_orientation == 'ROW': shape_tuple = (self.rows_out, self.columns_out)

            # IF bin_int_or_float WAS NOT SPECIFIED, DEFAULTING TO 'INT'

            if self.bin_int_or_float == 'BIN':
                if self.return_format == 'ARRAY':
                    self.OBJECT = crsn.create_random_sparse_numpy(0, 2, shape_tuple, self._sparsity, np.int8)
                elif self.return_format == 'SPARSE_DICT':
                    self.OBJECT = sd.create_random_py_bin(0, 2, shape_tuple, self._sparsity)

                self.MODIFIED_DATATYPES = np.fromiter((f'BIN' for _ in range(self.columns_out)), dtype=object)

            elif self.bin_int_or_float in [None, 'INT']:
                if self.return_format == 'ARRAY':
                    self.OBJECT = crsn.create_random_sparse_numpy(self.min_value, self.max_value + 1,
                                                                            shape_tuple, self._sparsity, np.int32)
                elif self.return_format == 'SPARSE_DICT':
                    self.OBJECT = sd.create_random_py_int(self.min_value, self.max_value + 1, shape_tuple, self._sparsity)

                self.MODIFIED_DATATYPES = np.fromiter((f'INT' for _ in range(self.columns_out)), dtype=object)

            elif self.bin_int_or_float == 'FLOAT':
                if self.return_format == 'ARRAY':
                    self.OBJECT = crsn.create_random_sparse_numpy(self.min_value, self.max_value,
                                                                            shape_tuple, self._sparsity, np.float64)
                elif self.return_format == 'SPARSE_DICT':
                    self.OBJECT = sd.create_random_py_float(self.min_value, self.max_value, shape_tuple, self._sparsity)

                self.MODIFIED_DATATYPES = np.fromiter((f'FLOAT' for _ in range(self.columns_out)), dtype=object)

            if self.return_format == 'ARRAY': self.is_list, self.is_dict = True, False
            elif self.return_format == 'SPARSE_DICT': self.is_list, self.is_dict = False, True

            del shape_tuple

        elif not self.OBJECT is None:

            if self.is_list:
                if self.given_orientation != self.return_orientation:
                    self.OBJECT = self.OBJECT.transpose()

                if self.return_format == 'ARRAY':
                    if self.bin_int_or_float == 'BIN': self.OBJECT = self.OBJECT.astype(bool).astype(np.int8)
                    elif self.bin_int_or_float in 'INT': self.OBJECT = self.OBJECT.astype(np.int32)
                    elif self.bin_int_or_float == 'FLOAT': self.OBJECT = self.OBJECT.astype(np.float64)
                    elif self.bin_int_or_float is None: pass  # YIELD ORIGINAL OBJECT

                    self.is_list, self.is_dict = True, False

                elif self.return_format == 'SPARSE_DICT':
                    if self.bin_int_or_float == 'BIN': self.OBJECT = sd.zip_list_as_py_int(self.OBJECT.astype(bool))
                    elif self.bin_int_or_float == 'INT': self.OBJECT = sd.zip_list_as_py_int(self.OBJECT.astype(int))
                    elif self.bin_int_or_float in ['FLOAT', None]: self.OBJECT = sd.zip_list_as_py_float(self.OBJECT.astype(float))
                    # IF USER DOES NOT GIVE DTYPE, DEFAULTING TO FLOAT

                    self.is_list, self.is_dict = False, True

            elif self.is_dict:
                if self.return_format == 'ARRAY':

                    self.OBJECT = sd.unzip_to_ndarray_float64(self.OBJECT)[0]

                    if self.given_orientation != self.return_orientation:
                        self.OBJECT = self.OBJECT.transpose()

                    if self.bin_int_or_float == 'INT': self.OBJECT = self.OBJECT.astype(np.int32)
                    elif self.bin_int_or_float == 'BIN': self.OBJECT = self.OBJECT.astype(bool).astype(np.int8)
                    elif self.bin_int_or_float == 'FLOAT': self.OBJECT = self.OBJECT.astype(np.float64)
                    elif self.bin_int_or_float is None: pass # YIELD ORIGINAL OBJECT

                    self.is_list, self.is_dict = True, False

                    # NOTES 11/25/22 WHEN GIVEN OBJECT IS FLOAT, BUT RETURN IS EITHER INT OR BIN,
                    # SETTING THE dtypes ABOVE FORCES -0.5 < x < 0.5 TO ZERO, THUS INCREASING THE ORIGINAL SPARSITY.

                elif self.return_format == 'SPARSE_DICT':
                    if self.return_orientation != self.given_orientation:
                        self.OBJECT = sd.core_sparse_transpose(self.OBJECT)

                    for outer_key in self.OBJECT:
                        _ = self.OBJECT[outer_key]
                        _keys = np.fromiter(_.keys(), dtype=np.int32).tolist()
                        if self.bin_int_or_float == 'INT':
                            _values = np.fromiter(_.values(), dtype=np.int32).tolist()
                        elif self.bin_int_or_float == 'BIN':
                            _values = np.fromiter(_.values(), dtype=bool).astype(np.int8).tolist()
                        elif self.bin_int_or_float in ['FLOAT', None]:
                            _values = np.fromiter(_.values(), dtype=np.float64).tolist()

                        self.OBJECT[int(outer_key)] = dict((zip(_keys, _values)))

                    del _, _keys, _values

                    self.is_list, self.is_dict = False, True

            if self.bin_int_or_float == 'BIN':
                # ANY INCOMING OBJECT FORCED TO BIN MUST HAVE MOD DTYPE BIN
                self.MODIFIED_DATATYPES = np.fromiter((f'BIN' for _ in range(self.columns_out)), dtype=object)
            elif self.bin_int_or_float == 'INT':
                # ANY INCOMING OBJECT FORCED TO INT COULD STILL BE 'BIN' SO MUST GET ACTUAL MOD DTYPES
                self.MODIFIED_DATATYPES = None
            elif self.bin_int_or_float == 'FLOAT':
                # ANY INCOMING OBJECT FORCED TO FLOAT COULD STILL BE 'BIN' OR 'INT' SO MUST GET ACTUAL MOD DTYPES
                self.MODIFIED_DATATYPES = None
            elif self.bin_int_or_float is None:
                # NO CHANGE TO ORIGINALLY GIVEN OBJECT, MUST GET MOD DTYPES
                self.MODIFIED_DATATYPES = None

        # END BUILD #############################################################################################################
        #########################################################################################################################


    def to_row(self):
        if self.return_orientation == 'ROW': pass
        elif self.return_orientation == 'COLUMN':
            self.return_orientation = 'ROW'
            if self.is_list: self.OBJECT = self.OBJECT.transpose()
            if self.is_dict: self.OBJECT = sd.core_sparse_transpose(self.OBJECT)


    def to_column(self):
        if self.return_orientation == 'COLUMN': pass
        elif self.return_orientation == 'ROW':
            self.return_orientation = 'COLUMN'
            if self.is_list: self.OBJECT = self.OBJECT.transpose()
            if self.is_dict: self.OBJECT = sd.core_sparse_transpose(self.OBJECT)


    def to_array(self):
        if self.return_format == 'ARRAY': pass
        elif self.return_format == 'SPARSE_DICT':
            self.return_format = 'ARRAY'
            if self.bin_int_or_float == 'BIN': self.OBJECT = sd.unzip_to_ndarray_int8(self.OBJECT)[0]
            elif self.bin_int_or_float == 'INT': self.OBJECT = sd.unzip_to_ndarray_int32(self.OBJECT)[0]
            elif self.bin_int_or_float == 'FLOAT': self.OBJECT = sd.unzip_to_ndarray_float64(self.OBJECT)[0]

        self.is_list, self.is_dict = True, False


    def to_sparse_dict(self):
        if self.return_format == 'SPARSE_DICT': pass
        elif self.return_format == 'ARRAY':
            self.return_format = 'SPARSE_DICT'
            if self.bin_int_or_float == ['BIN','INT']: self.OBJECT = sd.zip_list_as_py_int(self.OBJECT)
            elif self.bin_int_or_float == 'FLOAT': self.OBJECT = sd.zip_list_as_py_float(self.OBJECT)

        self.is_list, self.is_dict = False, True


    def expand(self, expand_as_sparse_dict=None, auto_drop_rightmost_column=False):
        '''expand() under CreateNumerical because it is the apex parent class. Necessary for CreateCategoricalNumpy.'''

        # ALL VALIDATION OF THESE KWARGS SHOULD BE HANDLED BY Expand

        fxn = inspect.stack()[0][3]

        # SENTINEL
        # IF IS CURRENTLY A DICT, EXPANDS AS DICT, IF LIST, EXPANDS AS LIST
        # (IF IS DICT PRE-EXPANSION, MUST BE ALL NUMBERS, AND EXPANSION DOES NOTHING)
        expand_as_sparse_dict = akv.arg_kwarg_validater(expand_as_sparse_dict, 'expand_as_sparse_dict',
                        [True, False, None], self.this_module, fxn, return_if_none=self.is_dict)

        auto_drop_rightmost_column = akv.arg_kwarg_validater(auto_drop_rightmost_column, 'auto_drop_rightmost_column',
                        [True, False], self.this_module, fxn)


        ExpandedObjects = ecto.ExpandCategoriesTestObjects(
                                     self.OBJECT,
                                     self.SUPPORT_OBJECTS,
                                     CONTEXT=None,  # DONT KNOW IF IS NEEDED
                                     KEEP=None,  # DONT KNOW IF IS NEEDED
                                     data_given_orientation=self.return_orientation,
                                     data_return_orientation=self.return_orientation,
                                     data_return_format='SPARSE_DICT' if expand_as_sparse_dict else 'ARRAY',
                                     auto_drop_rightmost_column=auto_drop_rightmost_column,
                                     bypass_validation=True,
                                     calling_module=self.this_module,
                                     calling_fxn=fxn
        )

        # MUST GET EXPANDED OBJECTS OUT OF Expand BY REASSIGNING THIS CLASS'S ATTRIBUTES TO Expanded's ATTRIBUTES (NOTHING IS RETURNED FROM Expand)
        self.OBJECT = ExpandedObjects.DATA_OBJECT
        self.OBJECT_HEADER = ExpandedObjects.SUPPORT_OBJECTS[self.LOOKUP_DICT['HEADER']].reshape((1,-1))
        self.VALIDATED_DATATYPES = ExpandedObjects.SUPPORT_OBJECTS[self.LOOKUP_DICT['VALIDATEDDATATYPES']]
        self.MODIFIED_DATATYPES = ExpandedObjects.SUPPORT_OBJECTS[self.LOOKUP_DICT['MODIFIEDDATATYPES']]
        self.FILTERING = ExpandedObjects.SUPPORT_OBJECTS[self.LOOKUP_DICT['FILTERING']]
        self.MIN_CUTOFFS = ExpandedObjects.SUPPORT_OBJECTS[self.LOOKUP_DICT['MINCUTOFFS']]
        self.USE_OTHER = ExpandedObjects.SUPPORT_OBJECTS[self.LOOKUP_DICT['USEOTHER']]
        self.START_LAG = ExpandedObjects.SUPPORT_OBJECTS[self.LOOKUP_DICT['STARTLAG']]
        self.END_LAG = ExpandedObjects.SUPPORT_OBJECTS[self.LOOKUP_DICT['ENDLAG']]
        self.SCALING = ExpandedObjects.SUPPORT_OBJECTS[self.LOOKUP_DICT['SCALING']]
        self.CONTEXT += ExpandedObjects.CONTEXT_HOLDER
        self.KEEP = self.OBJECT_HEADER[0].copy()

        self.is_expanded = True
        if expand_as_sparse_dict: self.is_list, self.is_dict = False, True

        del ExpandedObjects





























if __name__ == '__main__':


    # EVERYTHING BELOW IS A TEST MODULE W SUPPORTING FUNCTIONS

    # 3/14/2023 VERIFIED MODULE AND TEST CODE IS GOOD

    from general_sound import winlinsound as wls

    def test_exc_handle(OBJECT, reason_text):
        time.sleep(1)
        print(f'\n\033[91mEXCEPTING OBJECT:\033[0m\x1B[0m')
        print(OBJECT)
        print()
        wls.winlinsound(888, 500)
        print(f'\n\033[91mWANTS TO RAISE EXCEPTION FOR \033[0m\x1B[0m')
        print(reason_text)
        print()
        print(obj_desc)
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
                test_exc_handle(OBJECT, f'\033[91mouter/inner len: MISMATCH BETWEEN inner_len ({inner_len}) AND columns ({columns}) FOR given_orient={given_orientation}, return_orient={return_orientation}\033[0m\x1B[0m')
        elif return_orientation == 'COLUMN':
            if outer_len != columns:
                test_exc_handle(OBJECT, f'\033[91mouter/inner len: MISMATCH BETWEEN outer_len ({outer_len}) AND columns ({columns}) FOR given_orient={given_orientation}, return_orient={return_orientation}\033[0m\x1B[0m')
            if inner_len != rows:
                test_exc_handle(OBJECT, f'\033[91mouter/inner len: MISMATCH BETWEEN inner_len ({inner_len}) AND rows ({rows}) FOR given_orient={given_orientation}, return_orient={return_orientation}\033[0m\x1B[0m')

        else: raise Exception(f'\n\033[91m*** given_orientation LOGIC FOR DETERMINING CORRECT outer/inner_len IS FAILING ***\033[0m\x1B[0m\n')



    def get_sparsity(OBJECT):
        if isinstance(OBJECT, dict):
            if sd.is_sparse_inner(OBJECT) and not sd.is_sparse_outer(OBJECT): __ = sd.sparsity({0: OBJECT})
            else: __ = sd.sparsity(OBJECT)
        elif isinstance(OBJECT, np.ndarray):
            __ = sd.list_sparsity(OBJECT)

        return __


    def test_sparsity(actual_sparsity, expected_sparsity):
        sw = 3   # sparsity_window
        __ = actual_sparsity
        if __ < expected_sparsity-sw or __ > expected_sparsity+sw:
            test_exc_handle(__, f'\n\033[91m*** ACTUAL SPARSITY {__} IS OUTSIDE OF expected_sparsity +/- {sw} ({expected_sparsity}) ***\033[0m\x1B[0m\n')


    def test_format(OBJECT, return_format, ctr, obj_desc):
        if isinstance(OBJECT, dict): __ = 'SPARSE_DICT'
        elif isinstance(OBJECT, np.ndarray): __ = 'ARRAY'
        else: test_exc_handle(OBJECT, f'\n\033[91m*** BIG DISASTER. OBJECT IS NOT AN ARRAY OR SPARSE_DICT ***\033[0m\x1B[0m\n')

        if __ != return_format:
            print(f'\n\033[91m*** trial {ctr}, {obj_desc} ***\033[0m\x1B[0m\n')
            test_exc_handle(OBJECT,
                f'\n\033[91m*** return_format: OBJECT FORMAT ({__}) IS NOT THE REQUIRED FORMAT ({return_format})!!!! ***\033[0m\x1B[0m\n')


    # def test_single_or_double(OBJECT, expected_single_or_double):
    #     if isinstance(OBJECT, dict):
    #         if sd.is_sparse_inner(OBJECT) and not sd.is_sparse_outer(OBJECT): __ = 'SINGLE'
    #         elif sd.is_sparse_outer(OBJECT) and not sd.is_sparse_inner(OBJECT): __ = 'DOUBLE'
    #         else: test_exc_handle(OBJECT, f'\n\033[91m*** sd.sparse_outer & inner are screwing up! ***\033[0m\x1B[0m\n')
    #     elif isinstance(OBJECT, np.ndarray):
    #         if isinstance(OBJECT[0], np.ndarray): __ = 'DOUBLE'
    #         else: __ = 'SINGLE'
    #
    #     if __ != expected_single_or_double:
    #         test_exc_handle(OBJECT, f'\n\033[91m*** single_or_double: OUTPUT IS {__}, SHOULD BE {_single_or_double_dum} ***\033[0m\x1B[0m\n')


    def test_min_and_max(OBJECT, bin_int_or_float, min_value, max_value, _sparsity):
        # GET EXPECTED MIN & MAX ##########################################################################################
        if bin_int_or_float == 'BIN': exp_min, exp_max = 0, 1
        elif bin_int_or_float == 'INT': exp_min, exp_max = min_value, max_value
        elif bin_int_or_float == 'FLOAT': exp_min, exp_max = min_value, max_value
        else: raise Exception(f'\n\033[91m*** LOGIC FOR bin_int_or_float IN test_min_and_max() IS FAILING ***\033[0m\x1B[0m\n')
        if _sparsity == 100: exp_min, exp_max = 0, 0
        elif _sparsity == 0: pass
        else: exp_min, exp_max = min(0, min_value), max(0, max_value)
        # END GET EXPECTED MIN & MAX ##########################################################################################

        # GET ACTUAL MIN & MAX ##########################################################################################
        if isinstance(OBJECT, dict):
            act_min, act_max = sd.min_(OBJECT), sd.max_(OBJECT)
        elif isinstance(OBJECT, (np.ndarray)):
            act_min, act_max = np.min(OBJECT), np.max(OBJECT)
        # END GET ACTUAL MIN & MAX ##########################################################################################

        if bin_int_or_float == 'BIN' and int(act_min) != 0 and int(act_max) != 1 and \
                exp_min != act_min and exp_max != act_max:
            test_exc_handle(OBJECT, f'\n\033[91m*** min max: EXPECTED BIN OUTPUT W min={min_value}, max={max_value}, GOT min={act_min}, max=({act_max}\033[0m\x1B[0m')
        elif bin_int_or_float == 'INT' and (act_min < exp_min or act_max > exp_max):
            test_exc_handle(OBJECT,
                f'\n\033[91m*** min max: EXPECTED INT OUTPUT W min={min_value}, max={max_value}, GOT min={act_min}, max={act_max}\033[0m\x1B[0m')
        elif bin_int_or_float == 'FLOAT' and (act_min < exp_min or act_max > exp_max):
            test_exc_handle(OBJECT,
                f'\n\033[91m*** min max: EXPECTED FLOAT OUTPUT W min={min_value}, max={max_value}, GOT min={act_min}, max={act_max}\033[0m\x1B[0m')


    def test_bin_int_or_float(OBJECT, expected_bin_int_or_float):

        # GET TYPE FROM OBJECT #####################################################################################
        _TYPES = []
        if isinstance(OBJECT, dict):
            # IF IS INNER DICT ("SINGLE") MAKE OUTER TO SIMPLY FOR THIS PROCESS
            if sd.is_sparse_outer(OBJECT): NEW_OBJECT = deepcopy(OBJECT)
            elif sd.is_sparse_inner(OBJECT): NEW_OBJECT = {0: deepcopy(OBJECT)}
            for outer_key in NEW_OBJECT:
                for inner_key in NEW_OBJECT[outer_key]:
                    _ = NEW_OBJECT[outer_key][inner_key]
                    if float(_) == float(0) or float(_) == float(1): __ = 'BIN'
                    elif 'INT' in str(type(_)).upper(): __ = 'INT'
                    elif 'FLOAT' in str(type(_)).upper(): __ = 'FLOAT'
                    else: test_exc_handle(OBJECT, f'\n\033[91m*** UNKNOWN DICT NUMBER DTYPE {_} ***\033[0m\x1B[0m\n')

                    if __ not in _TYPES: _TYPES.append(__)

                if 'BIN' in _TYPES and 'INT' in _TYPES and 'FLOAT' in _TYPES: break

            if 'FLOAT' in _TYPES: _dtype = ['FLOAT']
            elif 'INT' in _TYPES: _dtype = ['INT']
            elif 'BIN' in _TYPES: _dtype = ['BIN'] #, 'INT']

        elif isinstance(OBJECT, np.ndarray):
            _ = str(OBJECT.dtype).upper()
            if float(np.min(OBJECT)) in [0,1] and float(np.max(OBJECT)) in [float(0),float(1)]: _dtype = ['BIN']#, 'INT']
            elif 'INT' in _: _dtype = ['INT']
            elif 'FLOAT' in _: _dtype = ['FLOAT']
            else: test_exc_handle(OBJECT, f'\n\033[91m*** UNKNOWN NDARRAY DTYPE {_} ***\033[0m\x1B[0m\n')
        # END GET TYPE FROM OBJECT #####################################################################################

        if expected_bin_int_or_float not in _dtype:
            test_exc_handle(OBJECT, f'\n\033[91m*** bin_int_or_float: OUTPUT IS {_dtype}, SHOULD BE {expected_bin_int_or_float} ***\033[0m\x1B[0m\n')


    def test_header(HEADER, columns):
        if len(HEADER[0]) != columns:
            test_exc_handle(HEADER, f'\n\033[91m*** HEADER LENGTH ({len(HEADER[0])}) DOES NOT EQUAL NUMBER OF COLUMNS ({columns})\033[0m\x1B[0m')

    #############################################################################################################################
    #############################################################################################################################
    # ONE PASS TESTING ##########################################################################################################
    if 2 == 1:    # TOGGLE THIS TO ENABLE ONE PASS TESTING.... SEE BELOW FOR ITERATIVE TESTING
        name = 'DATA'
        given_orientation = 'ROW'
        _columns = 30
        _rows = 50
        return_format = 'SPARSE_DICT'
        return_orientation = 'ROW'
        bin_int_or_float = 'FLOAT'
        min_value = 1
        max_value = 9
        _sparsity = 50

        num_of_inner = _rows if given_orientation == 'ROW' else _columns if given_orientation == 'COLUMN' else quit()
        len_of_inner = _columns if given_orientation == 'ROW' else _rows if given_orientation == 'ROW' else quit()

        OBJECT = crsn.create_random_sparse_numpy(min_value,max_value,(num_of_inner,len_of_inner),_sparsity,np.int32)
        OBJECT_HEADER = th.test_header(_columns)

        print(f'\nINPUT OBJECT:')
        print(OBJECT)
        print()

        DummyClass = CreateNumerical(name=name, OBJECT=OBJECT, OBJECT_HEADER=OBJECT_HEADER,
             given_orientation=given_orientation, columns=_columns, rows=_rows, return_format=return_format,
             return_orientation=return_orientation, bin_int_or_float=bin_int_or_float, min_value=min_value,
             max_value=max_value, _sparsity=_sparsity)

        RETURN_OBJECT = DummyClass.OBJECT
        RETURN_HEADER = DummyClass.OBJECT_HEADER

        data_object_desc = 'SPARSE_DICT' if isinstance(OBJECT, dict) else 'ARRAY'
        header_desc = "not given" if OBJECT_HEADER is None else "given"

        obj_desc = \
            f"\nINCOMING DATA OBJECT IS A {_sparsity}% SPARSE {data_object_desc} AND HEADER IS \n{header_desc}" + \
            [f", (min={min_value} max={max_value}) WITH {_rows} ROWS AND {_columns} COLUMNS ORIENTED AS {given_orientation}. "
                if not data_object_desc is None else ". "][0] + \
            f"\nOBJECT SHOULD BE A {_sparsity}% SPARSE {return_format} OF {bin_int_or_float}S (min={min_value} max={max_value}) " \
              f"WITH {_rows} ROWS AND {_columns} COLUMNS) ORIENTED AS {return_orientation}"

        print(obj_desc)

        print()
        print(f'OUTPUT_OBJECT:')
        print(RETURN_OBJECT)
        print()
        print(f'OBJECT_HEADER:')
        print(RETURN_HEADER)

        test_format(RETURN_OBJECT, return_format)
        test_bin_int_or_float(RETURN_OBJECT, 'BIN' if _sparsity==100 else bin_int_or_float)
        test_min_and_max(RETURN_OBJECT, bin_int_or_float, min_value, max_value, _sparsity)
        test_lens(RETURN_OBJECT, given_orientation, return_orientation, _rows, _columns)
        test_header(RETURN_HEADER, _columns)
        test_sparsity(get_sparsity(RETURN_OBJECT), _sparsity)

        print(f'\n\033[92m*** TEST PASSED ***\033[0m\x1B[0m\n')

        quit()


    # END ONE PASS TESTING ##########################################################################################################
    #############################################################################################################################
    #############################################################################################################################

    # ############################################################################################################################
    # #############################################################################################################################
    # ITERATIVE TESTING ###################################################################################################################

    name = 'DATA'
    DATA_OBJECT = None
    DATA_OBJECT_HEADER = None
    given_orientation = None

    GIVEN_FORMAT = ['ARRAY', 'SPARSE_DICT']
    RETURN_FORMAT = ['SPARSE_DICT', 'ARRAY']
    RETURN_ORIENTATION = ['ROW', 'COLUMN']
    COLUMNS = [10,100,1]
    ROWS = [10,1,100]
    MIN_VALUE = [2, 0, -5]
    MAX_VALUE = [10, 10, 6]

    DATA_OBJECTS = [
                           'built_during_run',
                           None,
                           ]

    DATA_OBJECT_HEADERS = [
                           'built_during_run',
                           None
                           ]
    BIN_INT_OR_FLOAT = ['INT', 'BIN', 'FLOAT']
    SPARSITY = [0, 5, 50, 95, 100]
    EXPAND = [True, False]

    total_itrs = np.product(list(map(len, (COLUMNS, RETURN_FORMAT, RETURN_ORIENTATION, BIN_INT_OR_FLOAT,
                                          MIN_VALUE, DATA_OBJECTS, DATA_OBJECT_HEADERS, SPARSITY,
                                          GIVEN_FORMAT, EXPAND))))


    ctr = 0
    for given_format in GIVEN_FORMAT:
        for return_format in RETURN_FORMAT:
            for return_orientation in RETURN_ORIENTATION:
                for _columns, _rows in zip(COLUMNS, ROWS):
                    for min_value, max_value in zip(MIN_VALUE, MAX_VALUE):
                        for data_object_desc in DATA_OBJECTS:
                            for data_object_hdr_desc in DATA_OBJECT_HEADERS:
                                for bin_int_or_float in BIN_INT_OR_FLOAT:
                                    for _sparsity in SPARSITY:
                                        for expand in EXPAND:
                                            ctr += 1

                                            if ctr % 100 == 0:
                                                print(f'\n\nRunning test {ctr} of {total_itrs}...')

                                            if not data_object_desc is None:
                                                if given_format == 'SPARSE_DICT':
                                                    if bin_int_or_float in ['BIN', 'INT']:
                                                        DATA_OBJECT = sd.create_random_py_int(
                                                                    0 if bin_int_or_float == 'BIN' else min_value,
                                                                    2 if bin_int_or_float == 'BIN' else max_value,
                                                                    (_rows, _columns),
                                                                    _sparsity)
                                                    elif bin_int_or_float == 'FLOAT':
                                                        DATA_OBJECT = sd.create_random_py_float(
                                                                    0 if bin_int_or_float == 'BIN' else min_value,
                                                                    2 if bin_int_or_float == 'BIN' else max_value,
                                                                    (_rows, _columns),
                                                                    _sparsity)
                                                elif given_format == 'ARRAY':
                                                    DATA_OBJECT = crsn.create_random_sparse_numpy(
                                                                    0 if bin_int_or_float == 'BIN' else min_value,
                                                                    2 if bin_int_or_float == 'BIN' else max_value,
                                                                    (_rows, _columns),
                                                                    _sparsity,
                                                                    np.int32 if bin_int_or_float == 'INT' else np.int8 if bin_int_or_float == 'BIN' else np.float64)
                                                given_orientation = 'ROW'
                                            else:
                                                DATA_OBJECT = None
                                                given_orientation = return_orientation

                                            if not data_object_hdr_desc is None: DATA_OBJECT_HEADER = th.test_header(_columns)
                                            else: DATA_OBJECT_HEADER = None

                                            obj_desc = f"\nINCOMING DATA OBJECT IS AN {dict(((True, 'EXPANDED'), (False, 'UNEXPANDED')))[expand]} {_sparsity}% SPARSE {given_format} AND HEADER IS \n{data_object_hdr_desc}" + \
                                                       [f", (min={min_value} max={max_value}) WITH {_rows} ROWS AND {_columns} COLUMNS ORIENTED AS {given_orientation}. "
                                                        if not data_object_desc is None else ". "][0] + \
                                                       f"\nOBJECT SHOULD BE RETURNED AS A {_sparsity}% SPARSE {return_format} OF {bin_int_or_float}S " \
                                                       f"(min={min_value} max={max_value}) WITH {_rows} ROWS AND {_columns} COLUMNS ORIENTED AS {return_orientation}."

                                            DummyObject = CreateNumerical(name=name, OBJECT=DATA_OBJECT,
                                                 OBJECT_HEADER=DATA_OBJECT_HEADER, given_orientation=given_orientation, columns=_columns,
                                                 rows=_rows, return_format=return_format, return_orientation=return_orientation,
                                                 bin_int_or_float=bin_int_or_float, min_value=min_value, max_value=max_value,
                                                 _sparsity=_sparsity)

                                            if expand:
                                                DummyObject.expand()

                                            # print(obj_desc)

                                            OUTPUT_OBJECT = DummyObject.OBJECT
                                            OBJECT_HEADER = DummyObject.OBJECT_HEADER
                                            VALIDATED_DATATYPES = DummyObject.VALIDATED_DATATYPES
                                            # FUDGE TO GET AROUND ANY COLUMN COINCIDENTALLY IN 0,1 BEING CALLED "BIN" WHEN IT WAS CREATED AS INT OR FLOAT
                                            if bin_int_or_float in ['INT', 'FLOAT'] and 'BIN' in VALIDATED_DATATYPES:
                                                VALIDATED_DATATYPES = np.where(VALIDATED_DATATYPES=='BIN', bin_int_or_float, VALIDATED_DATATYPES)
                                            MODIFIED_DATATYPES = DummyObject.MODIFIED_DATATYPES
                                            if bin_int_or_float in ['INT', 'FLOAT'] and 'BIN' in MODIFIED_DATATYPES:
                                                MODIFIED_DATATYPES = np.where(MODIFIED_DATATYPES=='BIN', bin_int_or_float, MODIFIED_DATATYPES)
                                            FILTERING = DummyObject.FILTERING
                                            MIN_CUTOFFS = DummyObject.MIN_CUTOFFS
                                            USE_OTHER = DummyObject.USE_OTHER
                                            START_LAG = DummyObject.START_LAG
                                            END_LAG = DummyObject.END_LAG
                                            SCALING = DummyObject.SCALING
                                            CONTEXT = DummyObject.CONTEXT
                                            KEEP = DummyObject.KEEP

                                            test_format(OUTPUT_OBJECT, return_format, ctr, obj_desc)
                                            # test_single_or_double(OUTPUT_OBJECT, "DOUBLE")
                                            test_bin_int_or_float(OUTPUT_OBJECT, 'BIN' if _sparsity==100 else bin_int_or_float)
                                            test_min_and_max(OUTPUT_OBJECT, bin_int_or_float, min_value, max_value, _sparsity)
                                            test_lens(OUTPUT_OBJECT, given_orientation, return_orientation, _rows, _columns)
                                            test_header(OBJECT_HEADER, _columns)
                                            test_sparsity(get_sparsity(OUTPUT_OBJECT), _sparsity)

                                            # ****** TEST SUPPORT OBJECTS ********************************************************************
                                            SUPP_OBJ_NAMES = ['HEADER', 'VALIDATED_DATATYPES', 'MODIFIED_DATATYPES', 'FILTERING', 'MIN_CUTOFFS',
                                                             'USE_OTHER', 'START_LAG', 'END_LAG', 'SCALING', 'CONTEXT', 'KEEP']
                                            if not data_object_hdr_desc is None: EXP_HEADER = th.test_header(_columns)
                                            else: EXP_HEADER = \
                                                np.fromiter((f'{name.upper()[:3]}_{bin_int_or_float}{_+1}' for _ in range(_columns)), dtype='<U20').reshape((1,-1))

                                            EXP_VALIDATED_DATATYPES = np.fromiter(
                                                (bin_int_or_float for _ in range(_columns)), dtype='<U5')

                                            EXP_MODIFIED_DATATYPES = np.fromiter(
                                                (bin_int_or_float for _ in range(_columns)), dtype='<U5')
                                            EXP_FILTERING = np.fromiter(([] for _ in range(_columns)), dtype=object)
                                            EXP_MIN_CUTOFFS = np.fromiter((0 for _ in range(_columns)), dtype=np.int16)
                                            EXP_USE_OTHER = np.fromiter(('N' for _ in range(_columns)), dtype='<U1')
                                            EXP_START_LAG = np.fromiter((0 for _ in range(_columns)), dtype=np.int16)
                                            EXP_END_LAG = np.fromiter((0 for _ in range(_columns)), dtype=np.int16)
                                            EXP_SCALING = np.fromiter(('' for _ in range(_columns)), dtype='<U200')
                                            EXP_KEEP = EXP_HEADER[0]
                                            EXP_CONTEXT = np.array([])

                                            EXP_SUPP_OBJS = [EXP_HEADER, EXP_VALIDATED_DATATYPES, EXP_MODIFIED_DATATYPES, EXP_FILTERING,
                                             EXP_MIN_CUTOFFS, EXP_USE_OTHER, EXP_START_LAG, EXP_END_LAG, EXP_SCALING, EXP_CONTEXT, EXP_KEEP]

                                            ACT_SUPP_OBJS = [OBJECT_HEADER, VALIDATED_DATATYPES, MODIFIED_DATATYPES, FILTERING, MIN_CUTOFFS,
                                             USE_OTHER, START_LAG, END_LAG, SCALING, CONTEXT, KEEP]

                                            for obj_name, EXP_SUPP_OBJ, ACT_SUPP_OBJ in zip(SUPP_OBJ_NAMES, EXP_SUPP_OBJS, ACT_SUPP_OBJS):
                                                if not np.array_equiv(EXP_SUPP_OBJ, ACT_SUPP_OBJ):
                                                    print(f'\n\n\033[91mFailed on trial {ctr} of {total_itrs}.\033[0m\x1B[0m')
                                                    print(obj_desc)
                                                    print(f'EXPECTED OBJECT = ')
                                                    print(EXP_SUPP_OBJ)
                                                    print()
                                                    print(f'ACTUAL OBJECT = ')
                                                    print(ACT_SUPP_OBJ)
                                                    print()
                                                    print(f'DATA OBJECT = ')
                                                    print(OUTPUT_OBJECT)
                                                    test_exc_handle(ACT_SUPP_OBJ, f'\n\033[91mACTUAL {obj_name} DOES NOT EQUAL EXPECTED\033[0m\x1B[0m')

                                            # ****** END TEST SUPPORT OBJECTS ********************************************************************

    print(f'\n\033[92m*** VALIDATION COMPLETED SUCCESSFULLY ***\033[0m\x1B[0m\n')
    for _ in range(3): wls.winlinsound(888, 500); time.sleep(0.5)

    # END ITERATIVE TESTING ###################################################################################################################
    # ############################################################################################################################
    # #############################################################################################################################




























