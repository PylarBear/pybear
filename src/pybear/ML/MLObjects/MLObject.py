import sys, inspect, time
import numpy as np
import sparse_dict as sd
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv, validate_user_input as vui
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from general_sound import winlinsound as wls
from MLObjects import MLRowColumnOperations as mlrco, ML_manage_intercept as mlmi
from MLObjects.SupportObjects import master_support_object_dict as msod


# METHODS
# __init__()                    Parent class for handling DATA and XTX within MLReqression operations. Object is accessed as attribute of class.
# _exception()                  Exception handling template.
# object_init()                 Unique OBJECT init sequence.
# get_shape()                   Return OBJECT shape like numpy.
# _transpose()                  Transpose OBJECT attribute.  (Not allowed in MLObjectSymmetric)
# get_transpose()               Return transpose of OBJECT in its current state without changing state.   (Not allowed in MLObjectSymmetric)
# to_array()                    Convert OBJECT attribute to a array.
# to_dict()                     Convert OBJECT attribute to a sparse dict.
# intercept_manager()           Locate columns of constants in DATA & handle.
# insert_standard_intercept()   Append a column of 1s as the last column in data and support object.
# insert_intercept()            Insert intercept into data and support object.
# return_as_array()             Return OBJECT in current state as array.
# return_as_dict()              Return OBJECT in current state as sparse dict.
# return_as_column()            Return OBJECT in current state oriented as column.     (Not allowed in MLObjectSymmetric)
# return_as_row()               Return OBJECT in current state oriented as row.     (Not allowed in MLObjectSymmetric)
# return_XTX()                  Return XTX calculated from OBJECT.   (Not allowed in MLObjectSymmetric and MLTargetObject)
# return_XTX_INV()              Return XTX_INV calculated from OBJECT.    (Not allowed in MLObjectSymmetric and MLTargetObject)
# is_equiv()                    Return boolean np.array_equiv or sd.sparse_equiv of this class's OBJECT with another MLObject class's OBJECT.
# unique()                      Return uniques of one column as list-type.

# INHERITS FROM MLRowColumnOperations
# return_rows()                 Returns selected columns without modifications to OBJECT
# return_columns()              Returns selected columns without modifications to OBJECT
# delete_rows()                 Return OBJECT and modifies in-place
# delete_columns()              Returns OBJECT and modifies in-place
# insert_row()                  Returns OBJECT and modifies in-place
# insert_column()               Returns OBJECT and modifies in-place

# ATTRIBUTES SHARED BY MLObject & MLRowColumnOperations
# _cols
# _rows
# this_module
# bypass_validation
# OBJECT
# given_orientation
# given_format
# current_format
# current_orientation


# ATTRIBUTES FROM MLObject
# calling_module
# calling_fxn
# inner_len
# outer_len
# return_format
# return_orientation

# ATTRS FROM MLRowColumnOperations
# exception_
# name
# shape
# CONTEXT
# HEADER_OR_FULL_SUPOBJ
# validate_return_orientation
# validate_return_format
# validate_idxs
# validate_idx_dtypes_len
# validate_idx_dtypes_int




class MLObject(mlrco.MLRowColumnOperations):
    '''Parent class for handling DATA, XTX, and TARGET within MLRegression operations. Object is accessed as attribute of class.'''
    def __init__(self, OBJECT, given_orientation, name=None, return_orientation='AS_GIVEN', return_format='AS_GIVEN',
                 bypass_validation=False, calling_module=None, calling_fxn=None):

        self.calling_module = calling_module
        self.calling_fxn = calling_fxn

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        self.given_format, self.OBJECT = ldv.list_dict_validater(OBJECT, 'OBJECT')

        self.bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation',
                [True, False, None], self.this_module, fxn, return_if_none=False)

        if self.bypass_validation:
            self.current_format = self.given_format
            self.given_orientation = given_orientation
            self.return_orientation = self.given_orientation if return_orientation == 'AS_GIVEN' else return_orientation
            self.return_format = self.given_format if return_format == 'AS_GIVEN' else return_format

        elif not self.bypass_validation:

            if self.given_format == 'SPARSE_DICT':
                self.OBJECT = sd.dict_init(self.OBJECT)
                sd.insufficient_dict_args_1(self.OBJECT, fxn)
                self.OBJECT = sd.clean(self.OBJECT)

            self.current_format = self.given_format
            self.given_orientation = akv.arg_kwarg_validater(given_orientation, 'given_orientation',
                    ['COLUMN', 'ROW'], self.this_module, fxn)
            self.return_orientation = akv.arg_kwarg_validater(given_orientation if return_orientation=='AS_GIVEN' else return_orientation,
                    'return_orientation', ['COLUMN', 'ROW'], self.this_module, fxn)
            self.return_format = akv.arg_kwarg_validater(self.given_format if return_format=='AS_GIVEN' else return_format,
                    'return_format', ['ARRAY', 'SPARSE_DICT'], self.this_module, fxn)

            self.get_shape()   # NEEDS current_format, child TARGET object_init NEEDS THIS FIRST

            self.object_init()  # DETOUR TO ACCOMMODATE "DATA" OR "TARGET"

        self.current_orientation = self.given_orientation

        # outer_len, inner_len BY self.get_shape() AT init, THESE WILL CHANGE WITH TRANSPOSE
        self.get_shape()

        if self.given_orientation == 'COLUMN':  self._cols, self._rows = self.outer_len, self.inner_len
        elif self.given_orientation == 'ROW':  self._cols, self._rows = self.inner_len, self.outer_len

        # DECLARED AT THIS POINT: OBJECT, this_module, fxn, calling_module, calling_fxn, given_format, current_format,
        #  return_format, given_orientation, current_orientation, return_orientation, bypass_validation, outer_len,
        #  inner_len, _cols, _rows

        super().__init__(self.OBJECT, self.given_orientation, name=name, bypass_validation=self.bypass_validation)
        # BEAR, WHEN WORKING ON CLEANING UP CONTEXT STUFF, REMEMBER THAT super() HAS A self.CONTEXT = []
        # DECLARATION AND IS super()ed HERE.

        # RESTATE SINCE super() OVERWROTE IT
        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))


        if self.return_format == 'ARRAY':
            if self.current_format != 'ARRAY': self.to_array()
            if self.return_orientation != self.current_orientation: self._transpose()
        elif self.return_format == 'SPARSE_DICT':
            if self.current_format == 'ARRAY':
                if self.current_orientation != self.return_orientation:    # IF ARRAY, TRANSPOSE AS ARRAY FOR SPEED
                    self._transpose()
                self.to_dict()
            elif self.current_format == 'SPARSE_DICT':
                if self.current_orientation != self.return_orientation:
                    self._transpose()

        # PLACEHOLDERS
        self.intcpt_col_idx = None    # AS OF 7/2/23 ONLY BEING USED WITH intercept_manager

    # END __init__ #####################################################################################################
    ####################################################################################################################
    ####################################################################################################################


    def _exception(self, fxn, verbage):
        '''Exception handling template.'''
        # ASSIGN self.calling_fxn USING InstanceOfThisClass.calling_fxn = fxn IN THE CALLING FUNCTION
        _caller = f'{self.calling_module}.{self.calling_fxn}()' if (not self.calling_module is None and not self.calling_fxn
                    is None) else None
        raise Exception([f'\n*** {self.this_module}.{fxn}() ' + f'called by {_caller} >>>' if not _caller is None else '>>>'][0] +
                        f' {verbage} ***\n')

    def object_init(self):
        '''Unique OBJECT init sequence. Overwrite in child.'''
        pass


    def get_shape(self):
        '''Return OBJECT shape like numpy.'''
        if self.current_format == 'ARRAY':
            self.outer_len, self.inner_len = len(self.OBJECT), len(self.OBJECT[0])
        elif self.current_format == 'SPARSE_DICT':
            self.outer_len, self.inner_len = sd.outer_len(self.OBJECT), sd.inner_len_quick(self.OBJECT)

        return self.outer_len, self.inner_len

    def _transpose(self):
        '''Transpose OBJECT attribute.'''
        self.current_orientation = 'ROW' if self.current_orientation == 'COLUMN' else 'COLUMN'
        if self.current_format == 'ARRAY': self.OBJECT = self.OBJECT.transpose()
        elif self.current_format == 'SPARSE_DICT': self.OBJECT = sd.core_sparse_transpose(self.OBJECT)

        self.outer_len, self.inner_len = self.inner_len, self.outer_len

    def get_transpose(self):
        """Return transpose of OBJECT in its current state without changing state."""
        # DONT CHANGE current_orientation!  A COPY IS RETURNED, ATTRIBUTE IS NOT CHANGED!
        if self.current_format == 'ARRAY': return self.OBJECT.transpose()
        elif self.current_format == 'SPARSE_DICT': return sd.core_sparse_transpose(self.OBJECT)

    # BASED ON TIME TESTS BELOW, THE KILLERS ARE ZIP/UNZIP WHEN ORIENTATION IS 'ROW' AND # EXAMPLES >> # FEATURES/TARGETS.
    # MAKE IT SO IF ZIP/UNZIPPING TO/FROM SPARSE DICT WHEN [[]=MANY]=FEW, TRANSPOSE TO INTERMEDIARY, CONVERT, THEN TRANSPOSE
    # BACK. BASED ON MANY SPEED TESTS 12/8/22, THERE IS A BLURRY AREA BETWEEN ROWS = 1000*COLS AND ROWS = 5000*COLS WHERE
    # TRANSPOSE --> ZIP/UNZIP --> TRANSPOSE BECOMES BETTER THAN BRUTE FORCE ZIP/UNZIP, SO ONLY DO IF RATIO OF
    # INNER [] IS 2000 TIMES NUMBER OF OUTER []
    def to_array(self):
        '''Convert OBJECT attribute to a array.'''
        if self.current_format == 'ARRAY': pass
        elif self.current_format == 'SPARSE_DICT':
            if (self.current_orientation == 'ROW' and self._rows > 2000 * self._cols) or \
                                        (self.current_orientation == 'COLUMN' and self._cols > 2000 * self._rows):
                self.OBJECT = sd.unzip_to_ndarray_float64(sd.core_sparse_transpose(self.OBJECT))[0].transpose()

            else: self.OBJECT = sd.unzip_to_ndarray_float64(self.OBJECT)[0]

            self.current_format = 'ARRAY'
            self.return_format = 'ARRAY'

    def to_dict(self):
        '''Convert OBJECT attribute to a sparse dict.'''
        if self.current_format == 'ARRAY':
            if (self.current_orientation == 'ROW' and self._rows > 2000 * self._cols) or \
                    (self.current_orientation == 'COLUMN' and self._cols > 2000 * self._rows):
                self.OBJECT = sd.core_sparse_transpose(sd.zip_list_as_py_float(self.OBJECT.transpose()))

            else: self.OBJECT = sd.zip_list_as_py_float(self.OBJECT)

            self.current_format = 'SPARSE_DICT'
            self.return_format = 'SPARSE_DICT'

        elif self.current_format == 'SPARSE_DICT': pass



    # BEAR IS SAYING intcpt_col_idx DOESNT NEED TO BE PASSED, ALL intcpt_col_idx ARE HARD ASSIGNMENTS TO self
    def intercept_manager(self, DATA_FULL_SUPOBJ_OR_HEADER=None): #, intcpt_col_idx=None):
        """Locate columns of constants in DATA & handle."""
        # BEAR FINISH
        # BEAR GO THRU AND FIGURE OUT ALL CTRL-SHIFT-F OF "CONTEXT", SEE WHERE THERE IS REDUNDANCY

        fxn = inspect.stack()[0][3]

        # VALIDATE HEADER OR FULL SUP OBJ AND CONVERT TO HEADER #################################################################
        DATA_FULL_SUPOBJ_OR_HEADER = ldv.list_dict_validater(DATA_FULL_SUPOBJ_OR_HEADER, 'DATA_FULL_SUPOBJ_OR_HEADER')[1]

        if DATA_FULL_SUPOBJ_OR_HEADER is None: ACTV_HDR = [f'COLUMN_{_}' for _ in range(1,self._cols+1)]
        else:
            if len(DATA_FULL_SUPOBJ_OR_HEADER) == 1: ACTV_HDR = DATA_FULL_SUPOBJ_OR_HEADER[0]
            elif len(DATA_FULL_SUPOBJ_OR_HEADER) == len(msod.master_support_object_dict()):
                ACTV_HDR = DATA_FULL_SUPOBJ_OR_HEADER[msod.QUICK_POSN_DICT()["HEADER"]]
            else: raise Exception(f'DATA_FULL_SUPOBJ_OR_HEADER ROWS ({len(DATA_FULL_SUPOBJ_OR_HEADER)}) DO NOT MATCH '
                                    f'HEADER ONLY OR FULL SUPPORT OBJECT')
        # END VALIDATE HEADER OR FULL SUP OBJ AND CONVERT TO HEADER #############################################################

        # BEAR IS SAYING THIS DOESNT NEED TO BE PASSED, ALL intcpt_col_idx ARE HARD ASSIGNMENTS
        # if not intcpt_col_idx is None and intcpt_col_idx in range(self._cols): self.intcpt_col_idx = intcpt_col_idx
        # else: self._exception(fxn, f'*** INVALID intcpt_col_idx "{intcpt_col_idx}"')

        # FIND COLUMNS OF CONSTANTS
        TO_DELETE, COLUMNS_OF_CONSTANTS_AS_DICT, COLUMNS_OF_ZEROS_AS_LIST = \
            mlmi.ML_manage_intercept(self.OBJECT, self.current_orientation, ACTV_HDR)

        # APPLY USER INSTRUCTIONS FROM ML_manage_intercept TO DATA & SUPOBJ #####################################################
        if len(TO_DELETE) > 0:
            print(f'\nDeleting user-selected columns of constants...')
            for col_idx in TO_DELETE:
                self.CONTEXT.append(f'{ACTV_HDR[col_idx]} deleted by user during cleanup of columns of constants.')

            self.delete_columns(TO_DELETE, HEADER_OR_FULL_SUPOBJ=ACTV_HDR, CONTEXT=None)  # CONTEXT UPDATE IS HANDLED EXTERNALLY ABOVE
            # DONT WORRY ABOUT self.intercept_col_idx, WILL BE SET AT THE END OF THIS, OVERWRITING WHATEVER WAS IN IT

            print(f'Done.')

        if len(COLUMNS_OF_CONSTANTS_AS_DICT) == 0:
            if vui.validate_user_str(f'\nDATA does not have an intercept. Append one now? (y/n) > ', 'YN') == 'Y':
                self.insert_standard_intercept()  # CONTEXT HANDLED INSIDE OF THE CALL
                self.intcpt_col_idx = 0
            else: self.intcpt_col_idx = None

        elif len(COLUMNS_OF_CONSTANTS_AS_DICT) == 1:
            if vui.validate_user_str(f'\nDATA has an intercept. Remove it? (y/n) > ', 'YN') == 'Y':
                self.delete_columns(list(COLUMNS_OF_CONSTANTS_AS_DICT.keys()), HEADER_OR_FULL_SUPOBJ=ACTV_HDR, CONTEXT=None)
                # BEAR CHECK THIS
                self.CONTEXT.append(f'Deleted column index {list(COLUMNS_OF_CONSTANTS_AS_DICT.keys())}.')
                self.intcpt_col_idx = None
            else: self.intcpt_col_idx = list(COLUMNS_OF_CONSTANTS_AS_DICT.keys())[0]

        elif len(COLUMNS_OF_CONSTANTS_AS_DICT) > 1:
            self.intcpt_col_idx = list(COLUMNS_OF_CONSTANTS_AS_DICT.keys())[0]
            print(f'\033[91m\n*** DATA STILL HAS MORE THAN ONE COLUMN OF CONSTANTS ***\n\033[0m')

        del ACTV_HDR, COLUMNS_OF_ZEROS_AS_LIST, COLUMNS_OF_CONSTANTS_AS_DICT, TO_DELETE
        # END APPLY USER INSTRUCTIONS FROM ML_manage_intercept TO DATA & SUPOBJ ###############################################



    def insert_standard_intercept(self, HEADER_OR_FULL_SUPOBJ=None, CONTEXT=None):
        """Append a column of 1s as the last column in data and support object."""

        datatype_string = 'INT' if not HEADER_OR_FULL_SUPOBJ is None else None
        header_string = 'INTERCEPT' if not HEADER_OR_FULL_SUPOBJ is None else None

        # ALWAYS INSERTS INTO 0 IDX (GMLR, MLR, MI, TestPerturber FORCE INTERCEPT TO 0 ANYWAY)
        return self.insert_intercept(0, 1, HEADER_OR_FULL_SUPOBJ=HEADER_OR_FULL_SUPOBJ,
                     datatype_string=datatype_string, header_string=header_string, CONTEXT=CONTEXT)


    def insert_intercept(self, col_idx, COLUMN_OR_VALUE_TO_INSERT, insert_orientation=None, HEADER_OR_FULL_SUPOBJ=None,
                         SUPOBJ_INSERT_VECTOR=None, datatype_string=None, header_string=None, CONTEXT=None):
        """Insert intercept into data and support object."""

        fxn = inspect.stack()[0][3]

        # col_idx
        # HANDLED EXCLUSIVELY BY insert_column()

        # COLUMN_OR_VALUE_TO_INSERT & insert_orientation ################################################################
        # IF VALUE IS PASSED, CONVERT TO LIST (insert_column(COLUMN_TO_INSERT) MUST TAKE LIST)
            #       - EVERYTHING ELSE ABOUT THE LIST IS VALIDATED IN insert_column()
            #       - insert_orientation MUST BE PASSED... insert_column() REQUIRES IT
        # IF LIST IS PASSED
        #       - insert_orientation MUST BE PASSED (BUT NOT IF A VALUE)... insert_column() REQUIRES IT
        #       - VERIFY IS CONSTANT
        #       - EVERYTHING ELSE ABOUT THE LIST IS VALIDATED IN insert_column()
        # END COLUMN_OR_VALUE_TO_INSERT & insert_orientation ############################################################

        # HEADER_OR_FULL_SUPOBJ, SUPOBJ_INSERT_VECTOR, datatype_string, & header_string #################################
        # IF HEADER_OR_FULL_SUPOBJ IS NOT PASSED
        #       - SUPOBJ_INSERT_VECTOR, datatype_string, AND header_string CANNOT BE PASSED
        # IF HEADER_OR_FULL_SUPOBJ IS PASSED
        #       - ROWS IN HEADER_OR_FULL_SUPOBJ MUST MATCH UP TO HEADER ONLY OR A FULL SUPOBJ
        #       - (SUPOBJ_INSERT_VECTOR) OR (datatype_string) or (datatype_string & header_string) CANNOT BE None
        #       - IF SUPOBJ_INSERT_VECTOR IS PASSED
        #           - datatype_str AND header_str CANNOT BE PASSED
        #           - SUPOBJ_INSERT_VECTOR MUST BE EQUALLY SIZED WITH HEADER_OR_FULL_SUPOBJ (FULL WITH FULL, HDR WITH HDR)
        #       - IF SUPOBJ_INSERT_VECTOR IS NOT PASSED, header_string OR (header_string & datatype_string) MUST BE PASSED'
        #       - SUPOBJ_INSERT_VECTOR MUST BE CONSTRUCTED FROM datatype_string & header_string (insert_column() REQUIRES A VECTOR)
        #           - IF HEADER_OR_FULL_SUPOBJ IS HEADER ONLY, header_string MUST BE PASSED, IGNORE datatype_string
        #           - IF HEADER_OR_FULL_SUPOBJ IS FULL, header_string & datatype_string MUST BE PASSED
        #        - SUPOBJ_INSERT_VECTOR IS PASSED TO insert_column(SUPOBJ_INSERT) AND IS VALIDATED THERE

        # CONTEXT
        # HANDLED EXCLUSIVELY BY insert_column()


        # COLUMN_OR_VALUE_TO_INSERT & insert_orientation ################################################################
        # IF VALUE IS PASSED, CONVERT TO LIST (insert_column(COLUMN_TO_INSERT) MUST TAKE LIST)
        if True in map(lambda x: x in str(type(COLUMN_OR_VALUE_TO_INSERT)).upper(), ['INT', 'FLOAT']):
            COLUMN_TO_INSERT = np.full(self._rows, COLUMN_OR_VALUE_TO_INSERT, dtype=np.int32)
            #       - EVERYTHING ELSE ABOUT THE LIST IS VALIDATED IN insert_column()
            #       - insert_orientation MUST BE PASSED... insert_column() REQUIRES IT
            insert_orientation = 'COLUMN'
        # IF LIST IS PASSED
        else:
        #       - insert_orientation MUST BE PASSED (BUT NOT IF A VALUE)... insert_column() REQUIRES IT
            if insert_orientation is None: self._exception(fxn, f'IF COLUMN_OR_VALUE_TO_INSERT IS PASSED AS COLUMN, '
                                                                f'insert_orientation MUST BE PASSED')
        #       - VERIFY IS CONSTANT
            COLUMN_TO_INSERT = COLUMN_OR_VALUE_TO_INSERT
            _ = COLUMN_TO_INSERT; not_constant = False
            if isinstance(_, (list, tuple, np.ndarray)) and not np.min(_)==np.max(_): not_constant = True
            elif isinstance(_, dict) and not sd.min_(_) == sd.max_(_): not_constant = True

            if not_constant: self._exception(fxn, f'COLUMN_OR_VALUE_TO_INSERT, WHEN PASSED AS COLUMN, MUST BE ALL THE SAME NUMBER')
            del _, not_constant
        #       - EVERYTHING ELSE ABOUT THE LIST IS VALIDATED IN insert_column()
        # END COLUMN_OR_VALUE_TO_INSERT & insert_orientation ############################################################


        # HEADER_OR_FULL_SUPOBJ, SUPOBJ_INSERT_VECTOR, datatype_string, & header_string #################################
        # IF HEADER_OR_FULL_SUPOBJ IS NOT PASSED
        if HEADER_OR_FULL_SUPOBJ is None:
        #       - SUPOBJ_INSERT_VECTOR, datatype_string, AND header_string CANNOT BE PASSED
            if not (SUPOBJ_INSERT_VECTOR is None and datatype_string is None and header_string is None):
                self._exception(fxn, f'IF HEADER_OR_FULL_SUPOBJ IS NOT PASSED, SUPOBJ_INSERT_VECTOR, header_string, AND '
                                     f'datatype_string CANNOT BE PASSED')
        # IF HEADER_OR_FULL_SUPOBJ IS PASSED
        elif not HEADER_OR_FULL_SUPOBJ is None:
            if len(HEADER_OR_FULL_SUPOBJ.shape)==1:
                HEADER_OR_FULL_SUPOBJ = HEADER_OR_FULL_SUPOBJ.reshape((1,-1))
        #       - ROWS IN HEADER_OR_FULL_SUPOBJ MUST MATCH UP TO HEADER ONLY OR A FULL SUPOBJ
            if len(HEADER_OR_FULL_SUPOBJ) not in [1, len(msod.master_support_object_dict())]:
                self._exception(f'HEADER_OR_FULL_SUPOBJ SHAPE DOES NOT MATCH UP TO HEADER ONLY OR A FULL SUPOBJ')

        #       - (SUPOBJ_INSERT_VECTOR) OR (datatype_string) or (datatype_string & header_string) CANNOT BE None
            if False not in (map(lambda x: x is None, (SUPOBJ_INSERT_VECTOR, header_string, datatype_string))):
                self._exception(fxn, f'IF HEADER_OR_FULL_SUPOBJ IS PASSED, SUPOBJ_INSERT_VECTOR, header_string, AND '
                                     f'datatype_string CANNOT ALL BE NONE')
        #       - IF SUPOBJ_INSERT_VECTOR IS PASSED
            if not SUPOBJ_INSERT_VECTOR is None:
                SUPOBJ_INSERT_VECTOR = np.array(SUPOBJ_INSERT_VECTOR, dtype=object).reshape((1,-1))[0]
        #           - datatype_str AND header_str CANNOT BE PASSED
                if not (datatype_string is None and header_string is None):
                    self._exception(f'IF SUPOBJ_INSERT_VECTOR IS PASSED, datatype_string AND header_string CANNOT BE PASSED')
        #           - SUPOBJ_INSERT_VECTOR MUST BE EQUALLY SIZED WITH HEADER_OR_FULL_SUPOBJ (FULL WITH FULL, HDR WITH HDR)
                if not len(SUPOBJ_INSERT_VECTOR)==len(HEADER_OR_FULL_SUPOBJ):
                    self._exception(fxn, f'DIMENSIONS OF HEADER_OR_FULL_SUPOBJ AND SUPOBJ_INSERT_VECTOR ARE NOT EQUAL.'
                                    f'MUST MATCH HEADER ONLY TO HEADER ONLY, OR FULL TO FULL')
            elif SUPOBJ_INSERT_VECTOR is None:
        #       - IF SUPOBJ_INSERT_VECTOR IS NOT PASSED, header_string OR (header_string & datatype_string) MUST BE PASSED'
        #       - SUPOBJ_INSERT_VECTOR MUST BE CONSTRUCTED FROM datatype_string & header_string (insert_column() REQUIRES A VECTOR)
                if len(HEADER_OR_FULL_SUPOBJ)==1:
        #           - IF HEADER_OR_FULL_SUPOBJ IS HEADER ONLY, header_string MUST BE PASSED, IGNORE datatype_string
                    if header_string is None:
                        self._exception(fxn, f'IF HEADER_OR_FULL_SUPOBJ IS PASSED AND HEADER ONLY AND SUBOBJ_INSERT_VECTOR IS '
                                             f'NOT PASSED, THEN header_string MUST BE PASSED')
                    elif not isinstance(header_string, str): self._exception(fxn, f'header_string MUST BE PASSED AS str')
                    SUPOBJ_INSERT_VECTOR = [[header_string]]
                elif len(HEADER_OR_FULL_SUPOBJ)==len(msod.master_support_object_dict()):
        #           - IF HEADER_OR_FULL_SUPOBJ IS FULL, header_string & datatype_string MUST BE PASSED
                    if datatype_string is None or header_string is None:
                        self._exception(fxn, f'IF A FULL SUPOBJ IS TO BE UPDATED WITHOUT A FULL SUPOBJ_INSERT_VECTOR, BOTH '
                                             f'datatype_string AND header_string MUST BE PASSED')
                    if False in map(lambda x: isinstance(x, str), (header_string, datatype_string)):
                        self._exception(fxn, f'header_string AND datatype_string MUST BE PASSED AS STRINGS')

                    # SOIV = SUPOBJ_INSERT_VECTOR
                    SOIV = np.empty(len(msod.master_support_object_dict()), dtype=object)
                    SOIV[msod.QUICK_POSN_DICT()["HEADER"]] = header_string
                    SOIV[msod.QUICK_POSN_DICT()["VALIDATEDDATATYPES"]] = datatype_string
                    SOIV[msod.QUICK_POSN_DICT()["MODIFIEDDATATYPES"]] = datatype_string
                    SOIV[msod.QUICK_POSN_DICT()["FILTERING"]] = []
                    SOIV[msod.QUICK_POSN_DICT()["MINCUTOFFS"]] = 0
                    SOIV[msod.QUICK_POSN_DICT()["USEOTHER"]] = 'N'
                    SOIV[msod.QUICK_POSN_DICT()["STARTLAG"]] = 0
                    SOIV[msod.QUICK_POSN_DICT()["ENDLAG"]] = 0
                    SOIV[msod.QUICK_POSN_DICT()["SCALING"]] = ''

                    SUPOBJ_INSERT_VECTOR = SOIV; del SOIV

        #        - SUPOBJ_INSERT_VECTOR IS PASSED TO insert_column(SUPOBJ_INSERT) AND IS VALIDATED THERE
        # END HEADER_OR_FULL_SUPOBJ, SUPOBJ_INSERT_VECTOR, datatype_string, & header_string ##############################

        # THIS self FUNCTION IS INHERITED FROM MLRowColumnOperations; TAKES A COLUMN_TO_INSERT, NOT A VALUE
        self.insert_column(col_idx, COLUMN_TO_INSERT, insert_orientation, HEADER_OR_FULL_SUPOBJ=HEADER_OR_FULL_SUPOBJ,
                           CONTEXT=CONTEXT, SUPOBJ_INSERT=SUPOBJ_INSERT_VECTOR)

        return self.OBJECT    # self.HEADER_OR_FULL_SUPOBJ AND self.CONTEXT MUST BE ACCESSED AS ATTRS OF INSTANCE


    def return_as_array(self):
        '''Return OBJECT in current state as array.'''
        if self.current_format == 'ARRAY': return self.OBJECT
        elif self.current_format == 'SPARSE_DICT':
            if (self.current_orientation == 'ROW' and self._rows > 2000 * self._cols) or \
                    (self.current_orientation == 'COLUMN' and self._cols > 2000 * self._rows):
                return sd.unzip_to_ndarray_float64(sd.core_sparse_transpose(self.OBJECT))[0].transpose()

            else: return sd.unzip_to_ndarray_float64(self.OBJECT)[0]

    def return_as_dict(self):
        '''Return OBJECT in current state as sparse dict.'''
        if self.current_format == 'ARRAY':
            if (self.current_orientation == 'ROW' and self._rows > 2000 * self._cols) or \
                    (self.current_orientation == 'COLUMN' and self._cols > 2000 * self._rows):
                return sd.core_sparse_transpose(sd.zip_list_as_py_float(self.OBJECT.transpose()))

            else: return sd.zip_list_as_py_float(self.OBJECT)

        elif self.current_format == 'SPARSE_DICT': return self.OBJECT

    def return_as_column(self):
        """Return OBJECT in current state oriented as column."""
        if self.current_orientation == 'COLUMN': return self.OBJECT
        elif self.current_orientation == 'ROW': return self.get_transpose()

    def return_as_row(self):
        """Return OBJECT in current state oriented as row."""
        if self.current_orientation == 'COLUMN': return self.get_transpose()
        elif self.current_orientation == 'ROW': return self.OBJECT

    def return_XTX(self, return_format=None):
        """Return XTX calculated from OBJECT in the same current state as OBJECT."""
        # 12/9/22 TIME TESTS SHOW FOR NP ARRAY, WHEN EXAMPLES >= COLUMNS (m >= n) SYMMETRIC FILL METHOD IS FASTER THAN matmul
        # FOR SD, WHEN ORIENTED AS COLUMN, SYMMETRIC FILL-IN IS THE OBVIOUS ANSWER
        # WHEN NUMPY AND AS 'ROW', SYMMETRIC WITH x[:,col_idx] IS FASTEST
        # WHEN IS SD ORIENTED AS ROW, THE SUMMARY SEEMS TO BE THAT THERE IS NOT AN OUTRIGHT WINNER, BETWEEN
        # 1) transpose / symmetric fill / transpose
        # 2) core_symmetric_matmul(sd.transpose(X), X, TRANS=sd.transpose(X))
        # 3) unzip to np / symmetric fill / zip to np
        # THERE ARE SOME TRADE-OFFS DEPENDING ON SPARSITY, SIZE, & SHAPE. IT SEEMS THAT UNZIP/SYMMETRIC BUILD XTX/ZIP
        # IS MORE OFTEN FASTER AND IS ALWAYS COMPETITIVE, COMPARED W core_symmetric_matmul APPROACHES, SO USE #3

        fxn = inspect.stack()[0][3]

        # 12/9/22 DONT CHANGE THIS. THIS IS SLYLY USING A TECHNIQUE CALLED A "sentinel", AND MUST BE USED BECAUSE CLASS
        # METHOD CANNOT TAKE self.xxx AS PARAM, SO HAVE TO SAY "None" THEN USE if TO SET TO self.xxx
        return_format = akv.arg_kwarg_validater(return_format, 'return_format', ['ARRAY', 'SPARSE_DICT'],
                                                            self.this_module, fxn, return_if_none=self.current_format)

        if self.current_format == 'ARRAY':
            if return_format == 'ARRAY': XTX = np.zeros((self._cols, self._cols), dtype=np.float64)
            elif return_format == 'SPARSE_DICT': XTX = {int(_): {} for _ in range(self._cols)}

            # LOTS OF COMPLICATION HERE, MELDED CODE FOR 'ROW' AND 'COLUMN' TOGETHER
            # DONT HAVE TO USE ANY lens HERE, IS ALREADY IN self._cols AND self._rows
            for idx1 in range(self._cols):
                for idx2 in range(idx1 + 1):
                    if self.current_orientation == 'COLUMN':
                        dot = np.matmul(self.OBJECT[idx1].astype(np.float64), self.OBJECT[idx2].astype(np.float64), dtype=np.float64)
                    elif self.current_orientation == 'ROW':
                        dot = np.matmul(self.OBJECT[:,idx1].astype(np.float64), self.OBJECT[:,idx2].astype(np.float64), dtype=np.float64)
                    if dot != 0:
                        XTX[int(idx1)][int(idx2)] = dot
                        XTX[int(idx2)][int(idx1)] = dot

            # PUT IN PLACEHOLDERS IF XTX IS RETURNED AS SD
            if return_format == 'SPARSE_DICT':
                for outer_idx in XTX:
                    if self._cols - 1 not in XTX[outer_idx]:
                        XTX[int(outer_idx)][int(self._cols - 1)] = 0

        elif self.current_format == 'SPARSE_DICT':
            if self.current_orientation == 'ROW':
                XTX = sd.sparse_ATA(self.OBJECT, return_as=return_format)
            elif self.current_orientation == 'COLUMN':
                XTX = sd.sparse_AAT(self.OBJECT, return_as=return_format)

        return XTX

    def return_XTX_INV(self, return_format=None, quit_on_exception=True, return_on_exception=None):
        """Return XTX_INV calculated from OBJECT in the same current state as OBJECT."""

        fxn = inspect.stack()[0][3]

        # 12/9/22 DONT CHANGE return_format. THIS IS SLYLY USING A TECHNIQUE CALLED "sentinel", AND MUST BE USED BECAUSE CLASS
        # METHOD CANNOT TAKE self.xxx AS PARAM, SO HAVE TO SAY "None" THEN USE if TO SET TO self.xxx
        return_format = akv.arg_kwarg_validater(return_format, 'return_format', ['ARRAY', 'SPARSE_DICT'],
                                                self.this_module, fxn, return_if_none=self.current_format)

        quit_on_exception = akv.arg_kwarg_validater(quit_on_exception, 'quit_on_exception', [True, False],
                                                self.this_module, fxn, return_if_none=True)

        # DONT DO VALIDATION FOR return_on_exception WITH akv.arg_kwarg_validater, COULD BE ANYTHING

        XTX = self.return_XTX(return_format='ARRAY')   # MUST RETURN AS ARRAY TO DO np.linalg.inv

        try:
            with np.errstate(all='ignore'):
                XTX_INV = np.linalg.inv(XTX)

                if return_format == 'ARRAY': pass
                elif return_format == 'SPARSE_DICT': XTX_INV = sd.zip_list_as_py_float(XTX_INV)
        except:
            LinAlgError_txt = f'\n*** Inverting XTX has numpy.linalg.LinAlgError, singular matrix ***\n'
            else_txt = f'\n*** Cannot invert XTX for error other than numpy.linalg.LinAlgError ***\n'

            if not quit_on_exception:

                if np.linalg.LinAlgError: print(LinAlgError_txt)
                else: print(else_txt)

                print(f'\n*** Continuing and returning {return_on_exception} instead of XTX_INV ***\n')

                return return_on_exception

            elif quit_on_exception:
                print(f'\n*** Exception getting inverse for XTX in MLObject.return_XTX_INV(). USER SAID '
                                                f'TERMINATE IF THIS HAPPENS. ***\n')

                if np.linalg.LinAlgError: self._exception(f'{LinAlgError_txt}')
                else: self._exception(f'{else_txt}')

        del XTX

        return XTX_INV

    def is_equiv(self, OTHER_OBJECT, test_as='ARRAY'):
        '''Return boolean np.array_equiv or sd.sparse_equiv of this class's OBJECT with another OBJECT (not as class).'''

        test_as = akv.arg_kwarg_validater(test_as, 'test_as', ['ARRAY', 'SPARSE_DICT'], self.this_module, 'is_equiv',
                                          return_if_none='ARRAY')

        # IF BOTH ARE ARRAY OR BOTH SD, TEST AS IS FOR SPEED & MEMORY, OVERRIDING ANY KWARG PUT IN 'test_as'
        if self.current_format == 'ARRAY' and isinstance(OTHER_OBJECT, (np.ndarray,list,tuple)):
            return np.array_equiv(self.OBJECT, np.array(OTHER_OBJECT))

        elif self.current_format == 'SPARSE_DICT' and isinstance(OTHER_OBJECT, dict):
            return sd.core_sparse_equiv(self.OBJECT, OTHER_OBJECT)

        elif self.current_format == 'ARRAY' and isinstance(OTHER_OBJECT, dict):
            if test_as == 'ARRAY':
                return np.array_equiv(self.OBJECT, sd.unzip_to_ndarray_float64(OTHER_OBJECT)[0])
            elif test_as == 'SPARSE_DICT':
                return sd.core_sparse_equiv(sd.zip_list_as_py_float(self.OBJECT), OTHER_OBJECT)
        elif self.current_format == 'SPARSE_DICT' and isinstance(OTHER_OBJECT, (np.ndarray, list, tuple)):
            if test_as == 'ARRAY':
                return np.array_equiv(sd.unzip_to_ndarray_float64(self.OBJECT)[0], np.array(OTHER_OBJECT))
            elif test_as == 'SPARSE_DICT':
                return sd.core_sparse_equiv(self.OBJECT, sd.zip_list_as_py_float(np.array(OTHER_OBJECT)))


    def unique(self, col_idx):
        '''Return uniques of one column as list-type.'''
        fxn = inspect.stack()[0][3]

        if col_idx not in range(self._cols):
            self._exception(inspect.stack()[0][3], f'col_idx OUT OF RANGE, MUST BE IN [0, {self._cols}]')

        if self.current_format == 'ARRAY':
            if len(self.OBJECT.shape)==1:
                UNIQUES = np.unique(self.OBJECT)
            elif self.current_orientation == 'COLUMN': UNIQUES = np.unique(self.OBJECT[col_idx, ...])
            elif self.current_orientation == 'ROW': UNIQUES = np.unique(self.OBJECT[..., col_idx])
            else: self._exception(fxn, '*** LOGIC MANAGING ORIENTATION IS FAILING ***')
        elif self.current_format == 'SPARSE_DICT':
            if len(sd.shape_(self.OBJECT))==1: UNIQUES = sd.return_uniques(self.OBJECT)
            elif self.current_orientation == 'COLUMN': UNIQUES = sd.return_uniques({0: self.OBJECT[col_idx]})
            elif self.current_orientation == 'ROW': UNIQUES = np.unique(sd.multi_select_inner(
                                                                self.OBJECT, [col_idx], as_inner=False, as_dict=False))
            else: self._exception(fxn, '*** LOGIC MANAGING ORIENTATION IS FAILING ***')
        else: self._exception(fxn, '*** LOGIC MANAGING FORMAT IS FAILING ***')


        if np.array_equiv(UNIQUES, UNIQUES.astype(np.int32)):
            UNIQUES = UNIQUES.astype(np.int32)

        return UNIQUES





    # INHERITED FORM MLRowColumnOperations
    # return_rows(ROW_IDXS_AS_INT_OR_LIST, return_orientation=None, return_format=None)
    # return_columns(COL_IDXS_AS_INT_OR_LIST, return_orientation=None, return_format=None)
    # delete_rows(ROW_IDXS_AS_INT_OR_LIST)
    # delete_columns(COL_IDXS_AS_INT_OR_LIST, HEADER_OR_FULL_SUPOBJ=None, CONTEXT=None)
    # insert_row(row_idx, ROW_TO_INSERT, insert_orientation)
    # insert_column(col_idx, COLUMN_TO_INSERT, insert_orientation, HEADER_OR_FULL_SUPOBJ=None, CONTEXT=None, SUPOBJ_INSERT=None)




# TIME TESTS 12/8/22
# AS 1000000x1 OBJECT OF 0,1 E.G. [[0][1][1].......]
# unzip_to_ndarray_float64    average, sdev: time = 15.25 sec, 0.129
# zip_list_as_py_float        average, sdev: time = 48.56 sec, 0.149
# sparse_transpose            average, sdev: time = 1.157 sec, 0.010  (TRANPOSED FROM [[]] TO [[][]]

# AS 1x1000000 OBJECT OF 0,1 E.G. [[0,1,0.......]]
# unzip_to_ndarray_float64    average, sdev: time = 0.201 sec, 0.004
# zip_list_as_py_float        average, sdev: time = 0.333 sec, 0.005
# sparse_transpose            average, sdev: time = 2.756 sec, 0.030  (TRANSPOSED FROM [[][]] TO [[]]

# AS 100x10000 OBJECTS OF 0,1 E.G. 100x[[0,1,0,....]]
# unzip_to_ndarray_float64    average, sdev: time = 0.159 sec, 0.006
# zip_list_as_py_float        average, sdev: time = 0.289 sec, 0.009
# sparse_transpose            average, sdev: time = 0.437 sec, 0.007  (TRANSPOSED TO 100x[[0,1,0,....]])

# AS 10000x100 OBJECTS OF 0,1 E.G. 10000x[[0,1,0,....]]
# unzip_to_ndarray_float64    average, sdev: time = 0.312 sec, 0.005
# zip_list_as_py_float        average, sdev: time = 0.780 sec, 0.007
# sparse_transpose            average, sdev: time = 0.522 sec, 0.005  (TRANSPOSED TO 10000x[[],[],....])








if __name__ == '__main__':

    # TEST MODULE --- TEST CODE & FUNCTIONAL CODE VERIFIED GOOD 3/27/23
    # 6/27/23 SEE MLObjects.MLObjects__misc_test.MLObject_insert_intercept__TEST TO TEST insert_intercept()
    # AS OF 3/27/23 THIS MODULE DOES NOT TEST THE FUNCTIONALITY OF METHODS INHERITED FROM MLRowColumnOperations
    # AS OF 3/27/23 MLRowColumnOperations ARE NOT TESTED WHEN ACCESSED THRU THIS MODULE

    exp_this_module = gmn.get_module_name(str(sys.modules[__name__]))
    exp_calling_module = 'MLObject'
    exp_calling_fxn = 'guard_test'

    expected_trials = 4*2*2*3*3*2
    ctr = 0
    for outer_len, inner_len in ((2,100),(100,2),(20,30),(30,20)):      # LONG & THIN TO ACTUATE CONDITIONAL zip/unzip

        OBJECT1 = np.random.randint(-9,10,(outer_len,inner_len), dtype=np.int32)
        OBJECT2 = sd.zip_list_as_py_int(OBJECT1)

        for exp_given_format, GIVEN_OBJECT in zip(('ARRAY', 'SPARSE_DICT'), (OBJECT1, OBJECT2)):
            for exp_given_orientation in ['COLUMN', 'ROW']:
                exp_columns = inner_len if exp_given_orientation == 'ROW' else outer_len
                exp_rows = outer_len if exp_given_orientation == 'ROW' else inner_len
                for exp_return_orientation in ['COLUMN', 'ROW', 'AS_GIVEN']:
                    exp_return_orientation = exp_given_orientation if exp_return_orientation == 'AS_GIVEN' else exp_return_orientation
                    exp_current_orientation = exp_return_orientation
                    exp_shape = (inner_len, outer_len) if exp_return_orientation != exp_given_orientation else (outer_len, inner_len)
                    exp_outer_len = inner_len if exp_return_orientation != exp_given_orientation else outer_len
                    exp_inner_len = outer_len if exp_return_orientation != exp_given_orientation else inner_len

                    for exp_return_format in ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN']:
                        exp_current_format = exp_given_format if exp_return_format == 'AS_GIVEN' else exp_return_format
                        exp_return_format = exp_given_format if exp_return_format == 'AS_GIVEN' else exp_return_format
                        for exp_bypass_validation in [True, False]:
                            ctr += 1
                            print(f'\n' + f'*'*70)
                            print(f'\nRunning trial {ctr} of {expected_trials}...')

                            if exp_return_format == 'ARRAY':
                                if exp_given_format in ['ARRAY', 'AS_GIVEN']:
                                    EXP_OBJECT = GIVEN_OBJECT
                                elif exp_given_format == 'SPARSE_DICT':
                                    EXP_OBJECT = sd.unzip_to_ndarray_float64(GIVEN_OBJECT)[0]
                                if exp_return_orientation not in [exp_given_orientation, 'AS_GIVEN']:
                                    EXP_OBJECT = EXP_OBJECT.transpose()
                            elif exp_return_format == 'SPARSE_DICT':
                                if exp_given_format == 'ARRAY':
                                    EXP_OBJECT = sd.zip_list_as_py_float(GIVEN_OBJECT)
                                elif exp_given_format in ['SPARSE_DICT', 'AS_GIVEN']:
                                    EXP_OBJECT = GIVEN_OBJECT
                                if exp_return_orientation not in [exp_given_orientation, 'AS_GIVEN']:
                                    EXP_OBJECT = sd.core_sparse_transpose(EXP_OBJECT)

                            expected_output = (f'Expected output:\n'
                                    f'exp_this_module = {exp_this_module}\n',
                                    f'exp_calling_module = {exp_calling_module}\n',
                                    f'exp_calling_fxn = {exp_calling_fxn}\n',
                                    f'exp_columns = {exp_columns}\n',
                                    f'exp_rows = {exp_rows}\n',
                                    f'exp_shape = {exp_shape}\n',
                                    f'exp_outer_len = {exp_outer_len}\n',
                                    f'exp_inner_len = {exp_inner_len}\n',
                                    f'exp_return_orientation = {exp_return_orientation}\n',
                                    f'exp_return_format = {exp_return_format}\n',
                                    f'exp_given_orientation = {exp_given_orientation}\n',
                                    f'exp_given_format = {exp_given_format}\n',
                                    f'exp_current_orientation = {exp_current_orientation}\n',
                                    f'exp_current_format = {exp_current_format}\n'
                                    # f'EXP_XTX_AS_NP = \n{EXP_XTX_AS_NP}'
                                    # f'ACT_XTX_AS_SD = \n{ACT_XTX_AS_SD}'
                                    # f'EXP_XTX_INV_AS_NP = \n{EXP_XTX_INV_AS_NP}'
                                    # f'ACT_XTX_INV_AS_SD = \n{ACT_XTX_INV_AS_SD}'
                                  )

                            # print(expected_output)

                            if isinstance(EXP_OBJECT, np.ndarray):
                                if exp_return_orientation == 'COLUMN':
                                    EXP_XTX_AS_NP = np.matmul(EXP_OBJECT.astype(np.float64), EXP_OBJECT.transpose().astype(np.float64), dtype=np.float64)
                                elif exp_return_orientation == 'ROW':
                                    EXP_XTX_AS_NP = np.matmul(EXP_OBJECT.transpose().astype(np.float64), EXP_OBJECT.astype(np.float64), dtype=np.float64)

                                EXP_XTX_AS_SD = sd.zip_list_as_py_float(EXP_XTX_AS_NP)

                                if exp_columns <= exp_rows:  # ONLY TEST FOR INV IF THIS TRUE, ALWAYS WILL EXCEPT IF exp_columns > exp_rows
                                    EXP_XTX_INV_AS_NP = np.linalg.inv(EXP_XTX_AS_NP)
                                    EXP_XTX_INV_AS_SD = sd.zip_list_as_py_float(EXP_XTX_INV_AS_NP)
                                else:
                                    EXP_XTX_INV_AS_NP = None
                                    EXP_XTX_INV_AS_SD = None

                            elif isinstance(EXP_OBJECT, dict):
                                if exp_return_orientation == 'COLUMN':
                                    EXP_XTX_AS_SD = sd.sparse_AAT(EXP_OBJECT)
                                elif exp_return_orientation == 'ROW':
                                    EXP_XTX_AS_SD = sd.sparse_ATA(EXP_OBJECT)

                                EXP_XTX_AS_NP = sd.unzip_to_ndarray_float64(EXP_XTX_AS_SD)[0]

                                if exp_columns <= exp_rows:  # ONLY TEST FOR INV IF THIS TRUE, ALWAYS WILL EXCEPT IF exp_columns > exp_rows
                                    EXP_XTX_INV_AS_NP = np.linalg.inv(EXP_XTX_AS_NP)
                                    EXP_XTX_INV_AS_SD = sd.zip_list_as_py_float(EXP_XTX_INV_AS_NP)
                                else:
                                    EXP_XTX_INV_AS_NP = None
                                    EXP_XTX_INV_AS_SD = None


                            DummyObject = MLObject(GIVEN_OBJECT,
                                                     exp_given_orientation,
                                                     name='MLObject_TEST',
                                                     return_orientation=exp_return_orientation,
                                                     return_format=exp_return_format,
                                                     bypass_validation=exp_bypass_validation,
                                                     calling_module=exp_calling_module,
                                                     calling_fxn=exp_calling_fxn)

                            ACT_OBJECT = DummyObject.OBJECT
                            act_given_format = DummyObject.given_format
                            act_given_orientation = DummyObject.given_orientation
                            act_current_orientation = DummyObject.current_orientation
                            act_current_format = DummyObject.current_format
                            act_return_format = DummyObject.return_format
                            act_return_orientation = DummyObject.return_orientation
                            act_bypass_validation = DummyObject.bypass_validation
                            act_calling_module = DummyObject.calling_module
                            act_calling_fxn = DummyObject.calling_fxn
                            act_this_module = DummyObject.this_module
                            act_columns = DummyObject._cols
                            act_rows = DummyObject._rows
                            act_shape = DummyObject.get_shape()
                            act_outer_len = DummyObject.outer_len
                            act_inner_len = DummyObject.inner_len
                            ACT_XTX_AS_NP = DummyObject.return_XTX(return_format='ARRAY')
                            ACT_XTX_AS_SD = DummyObject.return_XTX(return_format='SPARSE_DICT')
                            if exp_columns <= exp_rows:  # ONLY TEST FOR INV IF THIS TRUE, ALWAYS WILL EXCEPT IF _cols > _rows
                                ACT_XTX_INV_AS_NP = DummyObject.return_XTX_INV(return_format='ARRAY')
                                ACT_XTX_INV_AS_SD = DummyObject.return_XTX_INV(return_format='SPARSE_DICT')
                            else:
                                ACT_XTX_INV_AS_NP = None
                                ACT_XTX_INV_AS_SD = None

                            DESCRIPTIONS = \
                                [
                                    'this_module',
                                    'calling_module',
                                    'calling_fxn',
                                    'OBJECT',
                                    'columns',
                                    'rows',
                                    'shape',
                                    'outer_len',
                                    'inner_len',
                                    'return_orientation',
                                    'return_format',
                                    'bypass_validation',
                                    'given_orientation',
                                    'given_format',
                                    'current_orientation',
                                    'current_format',
                                    'xtx_as_np',
                                    'xtx_as_sd',
                                    'xtx_inv_as_np',
                                    'xtx_inv_as_sd'
                            ]


                            EXP_OBJS = \
                                [
                                    exp_this_module,
                                    exp_calling_module,
                                    exp_calling_fxn,
                                    EXP_OBJECT,
                                    exp_columns,
                                    exp_rows,
                                    exp_shape,
                                    exp_outer_len,
                                    exp_inner_len,
                                    exp_return_orientation,
                                    exp_return_format,
                                    exp_bypass_validation,
                                    exp_given_orientation,
                                    exp_given_format,
                                    exp_current_orientation,
                                    exp_current_format,
                                    EXP_XTX_AS_NP,
                                    EXP_XTX_AS_SD,
                                    EXP_XTX_INV_AS_NP,
                                    EXP_XTX_INV_AS_SD
                            ]

                            ACT_OBJS = \
                                [
                                    act_this_module,
                                    act_calling_module,
                                    act_calling_fxn,
                                    ACT_OBJECT,
                                    act_columns,
                                    act_rows,
                                    act_shape,
                                    act_outer_len,
                                    act_inner_len,
                                    act_return_orientation,
                                    act_return_format,
                                    act_bypass_validation,
                                    act_given_orientation,
                                    act_given_format,
                                    act_current_orientation,
                                    act_current_format,
                                    ACT_XTX_AS_NP,
                                    ACT_XTX_AS_SD,
                                    ACT_XTX_INV_AS_NP,
                                    ACT_XTX_INV_AS_SD
                            ]


                            for description, expected_thing, actual_thing in zip(DESCRIPTIONS, EXP_OBJS, ACT_OBJS):

                                try:
                                    is_equal = np.array_equiv(expected_thing, actual_thing)
                                    # print(f'\033[91m\n*** TEST EXCEPTED ON np.array_equiv METHOD ***\033[0m\x1B[0m\n')
                                except:
                                    try: is_equal = expected_thing == actual_thing
                                    except:
                                        print(f'\n\033[91mEXP_OBJECT = \n{EXP_OBJECT}\033[0m\x1B[0m\n')
                                        print(f'\n\033[91mACT_OBJECT = \n{ACT_OBJECT}\033[0m\x1B[0m\n')
                                        raise Exception(f'\n*** TEST FAILED "==" METHOD ***\n')

                                if not is_equal:
                                    print(f'\n\033' + f'*' * 70 + f'\033[0m\x1B[0m\n')
                                    print(f'\n\033Failed on trial {ctr} of {expected_trials}.\033[0m\x1B[0m\n')
                                    print(expected_output)
                                    print(f'\n\033[91mEXP_OBJECT = \n{EXP_OBJECT}\033[0m\x1B[0m\n')
                                    print(f'\n\033[91mACT_OBJECT = \n{ACT_OBJECT}\033[0m\x1B[0m\n')
                                    raise Exception(f'\n*** {description} FAILED EQUALITY TEST, \nexpected = \n{expected_thing}\n'
                                                    f'actual = \n{actual_thing} ***\n')
                                else: pass # print(f'\033[92m     *** {description} PASSED ***\033[0m\x1B[0m')


    print(f'\n\033[92m*** TEST DONE. ALL PASSED. ***\033[0m\x1B[0m\n')
    for _ in range(3): wls.winlinsound(888, 500); time.sleep(1)














