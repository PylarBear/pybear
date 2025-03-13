import sys, inspect, time
from general_sound import winlinsound as wls
import numpy as np
import sparse_dict as sd
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from general_data_ops import get_shape as gs
from MLObjects.SupportObjects import master_support_object_dict as msod


# PERFORM INSERT, DELETE, SELECT ... FOR ROWS / COLUMNS FOR NP ARRAY & SPARSE DICTS

# RULES OF NUMPY SLICING 3/24/23
# WHEN SLICING WITH A SINGLE NUMBER, EG, [:, 0] OR [0, ...]  A SINGLE [] IS RETURNED REGARDLESS IF PULLING A [] OR SLICING THRU []s
# BUT IF SLICING WITH A LIST TYPE, EG, [:, (0,)], [..., (0,)], [(0,2),:] ETC, RETURNS BASED ON THE LAYOUT, SO WOULD GET
# [...] IF PULLING A [], BUT WOULD GET [[],[],...] IF SLICING THRU []s


# METHODS
# return_rows()       Returns selected columns without modifications to OBJECT
# return_columns()    Returns selected columns without modifications to OBJECT
# delete_rows()       Return OBJECT and modifies in-place
# delete_columns()    Returns OBJECT and modifies in-place
# insert_row()        Returns OBJECT and modifies in-place
# insert_column()     Returns OBJECT and modifies in-place


class MLRowColumnOperations:
    '''__init__ does not perform any operations, just initialize and validate.  Methods = return_rows(),
        return_columns(), delete_rows(), delete_columns(), insert_row(), insert_column().'''
    def __init__(self, OBJECT, given_orientation, name=None, bypass_validation=None):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        self.bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                         self.this_module, fxn, return_if_none=False)

        self.exception_ = lambda words, fxn=None: Exception(f"{self.this_module}{f'.{fxn}()' if not fxn is None else f''} >>> {words}")

        if self.bypass_validation:
            self.name = name if not name is None else 'OBJECT'
            self.given_orientation = given_orientation
        elif not self.bypass_validation:
            if not isinstance(name, str): raise self.exception_(f'name MUST BE A str', fxn)
            else: self.name = name
            self.given_orientation = akv.arg_kwarg_validater(given_orientation, 'given_orientation', ['ROW','COLUMN'],
                                                             self.this_module, fxn)

        self.given_format, self.OBJECT = ldv.list_dict_validater(OBJECT, self.name)

        self.current_format = self.given_format
        self.current_orientation = self.given_orientation

        self.shape = gs.get_shape(self.name, self.OBJECT, self.current_orientation)
        self._rows = self.shape[0]
        self._cols = self.shape[1]

        self.CONTEXT = []

        # PLACEHOLDERS
        self.HEADER_OR_FULL_SUPOBJ = None

        # VALIDATION LAMBDAS FOR METHODS ###########################################################
        self.validate_return_orientation = lambda return_orientation: akv.arg_kwarg_validater(return_orientation, 'return_orientation',
                ['ROW', 'COLUMN', 'AS_GIVEN', None], self.this_module, fxn, return_if_none=self.current_orientation)

        self.validate_return_format = lambda return_format: akv.arg_kwarg_validater(return_format, 'return_format',
                ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN', None], self.this_module, fxn, return_if_none=self.current_format)

        def exception_raiser(words, fxn):
            raise self.exception_(f'{words}', fxn)

        self.is_int = lambda x: 'INT' in str(type(x)).upper()

        self.validate_idxs = lambda name, IDXS, fxn: exception_raiser(f'{name}_IDXS_AS_INT_OR_LIST MUST BE AN INT OR A LIST-TYPE OF INTS', fxn) \
            if not self.is_int(IDXS) and not 'INT' in str(np.array(IDXS).dtype).upper() else ""

        self.validate_idx_dtypes_int = lambda name, IDXS, fxn: exception_raiser(f'{name}_IDXS_AS_INT_OR_LIST MUST BE INTEGERS', fxn) \
            if not 'INT' in str(np.array(IDXS).dtype).upper() else ""

        # END VALIDATION LAMBDAS FOR METHODS ###########################################################


    # END __init__ ################################################################################################################
    ###############################################################################################################################
    ###############################################################################################################################


    def return_rows(self, ROW_IDXS_AS_INT_OR_LIST, return_orientation=None, return_format=None):
        '''Returns selected columns without modifications to OBJECT.'''
        fxn = inspect.stack()[0][3]

        if self.bypass_validation:
            if return_orientation is None: return_orientation=self.current_orientation
            if return_format is None: return_format=self.current_format

        elif not self.bypass_validation:
            return_orientation = self.validate_return_orientation(return_orientation)
            return_format = self.validate_return_format(return_format)
            self.validate_idxs('ROW', ROW_IDXS_AS_INT_OR_LIST, fxn)

        return_orientation = self.given_orientation if return_orientation=='AS_GIVEN' else return_orientation
        return_format = self.given_format if return_format == 'AS_GIVEN' else return_format

        if self.is_int(ROW_IDXS_AS_INT_OR_LIST): ROW_IDXS = tuple(ROW_IDXS_AS_INT_OR_LIST,)
        else: ROW_IDXS = tuple(np.array(ROW_IDXS_AS_INT_OR_LIST).reshape((1,-1))[0].tolist())
        del ROW_IDXS_AS_INT_OR_LIST

        if not self.bypass_validation:
            self.validate_idx_dtypes_int('ROW', ROW_IDXS, fxn)
            if max(ROW_IDXS) > self._rows-1: raise self.exception_(f'INDICATED INDICES TO SELECT ROWS ARE OUT OF RANGE', fxn)


        # GET THE OBJECT SMALL FIRST, THEN LET ObjectOrienter DEAL W HEAVY return_format & return_orientation OPERATIONS
        is_list = self.current_format=='ARRAY'
        is_dict = self.current_format=='SPARSE_DICT'
        is_col = self.current_orientation=='COLUMN'
        is_row = self.current_orientation=='ROW'
        to_list = return_format=='ARRAY'
        to_dict = return_format=='SPARSE_DICT'
        to_col = return_orientation=='COLUMN'
        to_row = return_orientation=='ROW'

        wip_orientation = self.current_orientation

        if is_list:
            if is_col:
                if to_col: _ = self.OBJECT[..., ROW_IDXS]   # wip_format stays array, wip_orient stays col
                elif to_row:
                    _ = self.OBJECT[..., ROW_IDXS].transpose()
                    wip_orientation = 'ROW'   # wip_format stays array
            elif is_row:
                if to_row: _ = self.OBJECT[ROW_IDXS, ...]   # wip_format stays array, wip_orient stays row
                elif to_col:
                    _ = self.OBJECT[ROW_IDXS, ...].transpose()
                    wip_orientation = 'COLUMN'  # wip format stays array
            # LET ObjectOrienter EXTRACT THESE THINGS TO SD IF NEEDED

        elif is_dict:
            if is_col:
                _ = sd.multi_select_inner(self.OBJECT, ROW_IDXS, as_inner=return_orientation=='COLUMN',
                                          as_dict=return_format=='SPARSE_DICT')
                wip_orientation = return_orientation    # wip_format GOES TO ARRAY
                # SHOULDNT HAVE TO WORRY ABOUT FORMAT CHANGE AND TRANSPOSING, SHOULD JUST BLOW THRU ObjectOrienter
            elif is_row:
                _ = sd.multi_select_outer(self.OBJECT, ROW_IDXS)
                if to_list:
                    _ = sd.unzip_to_ndarray_float64(_)[0]
                    # wip_format GOES TO ARRAY
                    if to_col:
                        _ = _.transpose()
                        wip_orientation = 'COLUMN'

        # DONT US MLObjectOrienter HERE, CREATES CIRCULAR IMPORT WITH MLObject
        if return_orientation != wip_orientation:
            if isinstance(_, dict): _ = sd.core_sparse_transpose(_)
            elif isinstance(_, np.ndarray): _ = _.transpose()

        if isinstance(_, dict) and return_format == 'ARRAY': _ = sd.unzip_to_ndarray_float64(_)[0]
        elif isinstance(_, np.ndarray) and return_format == 'SPARSE_DICT': _ = sd.zip_list_as_py_float(_)

        # DONT CHANGE ANY current_formats, current_orients, etc. BECAUSE self.OBJECT HAS NOT CHANGED

        del is_list, is_dict, is_col, is_row, to_list, to_dict, to_col, to_row, wip_orientation

        return _


    def return_columns(self, COL_IDXS_AS_INT_OR_LIST, return_orientation=None, return_format=None):
        '''Returns selected columns without modifications to OBJECT.'''
        fxn = inspect.stack()[0][3]

        if self.bypass_validation:
            if return_orientation is None: return_orientation=self.current_orientation
            if return_format is None: return_format=self.current_format

        elif not self.bypass_validation:
            return_orientation = self.validate_return_orientation(return_orientation)
            return_format = self.validate_return_format(return_format)
            self.validate_idxs('COL', COL_IDXS_AS_INT_OR_LIST, fxn)

        return_orientation = self.given_orientation if return_orientation=='AS_GIVEN' else return_orientation
        return_format = self.given_format if return_format == 'AS_GIVEN' else return_format

        if self.is_int(COL_IDXS_AS_INT_OR_LIST): COL_IDXS = tuple(COL_IDXS_AS_INT_OR_LIST,)
        else: COL_IDXS = tuple(np.array(COL_IDXS_AS_INT_OR_LIST).reshape((1,-1))[0].tolist())
        del COL_IDXS_AS_INT_OR_LIST

        if not self.bypass_validation:
            self.validate_idx_dtypes_int('COL', COL_IDXS, fxn)
            if max(COL_IDXS) > self._cols-1: raise self.exception_(f'INDICATED INDICES TO SELECT COLUMNS ARE OUT OF RANGE', fxn)


        # GET THE OBJECT SMALL FIRST, THEN LET ObjectOrienter DEAL W HEAVY return_format & return_orientation OPERATIONS
        is_list = self.current_format=='ARRAY'
        is_dict = self.current_format=='SPARSE_DICT'
        is_col = self.current_orientation=='COLUMN'
        is_row = self.current_orientation=='ROW'
        to_list = return_format=='ARRAY'
        to_dict = return_format=='SPARSE_DICT'
        to_col = return_orientation=='COLUMN'
        to_row = return_orientation=='ROW'

        wip_orientation = self.current_orientation

        if is_list:
            if is_col:
                if to_col: _ = self.OBJECT[COL_IDXS, ...]   # wip_format stays array, wip_orient stays col
                elif to_row:
                    _ = self.OBJECT[COL_IDXS, ...].transpose()
                    wip_orientation = 'ROW'   # wip_format stays array
            elif is_row:
                if to_row: _ = self.OBJECT[..., COL_IDXS]   # wip_format stays array, wip_orient stays row
                elif to_col:
                    _ = self.OBJECT[..., COL_IDXS].transpose()
                    wip_orientation = 'COLUMN'  # wip format stays array
            # LET ObjectOrienter EXTRACT THESE THINGS TO SD IF NEEDED

        elif is_dict:
            if is_col:
                _ = sd.multi_select_outer(self.OBJECT, COL_IDXS)
                if to_list:
                    _ = sd.unzip_to_ndarray_float64(_)[0]
                    # wip_format GOES TO ARRAY
                    if to_row:
                        _ = _.transpose()
                        wip_orientation = 'ROW'
            elif is_row:
                _ = sd.multi_select_inner(self.OBJECT, COL_IDXS, as_inner=return_orientation=='ROW',
                                          as_dict=return_format=='SPARSE_DICT')
                wip_orientation = return_orientation    # wip_format GOES TO ARRAY
                # SHOULDNT HAVE TO WORRY ABOUT FORMAT CHANGE AND TRANSPOSING

        # DONT US MLObjectOrienter HERE, CREATES CIRCULAR IMPORT WITH MLObject
        if return_orientation != wip_orientation:
            if isinstance(_, dict): _ = sd.core_sparse_transpose(_)
            elif isinstance(_, np.ndarray): _ = _.transpose()

        if isinstance(_, dict) and return_format == 'ARRAY': _ = sd.unzip_to_ndarray_float64(_)[0]
        elif isinstance(_, np.ndarray) and return_format == 'SPARSE_DICT': _ = sd.zip_list_as_py_float(_)

        # DONT CHANGE ANY current_formats, current_orients, etc. BECAUSE self.OBJECT HAS NOT CHANGED

        del is_list, is_dict, is_col, is_row, to_list, to_dict, to_col, to_row, wip_orientation

        return _


    def delete_rows(self, ROW_IDXS_AS_INT_OR_LIST):
        '''Return OBJECT and modifies in-place.'''
        fxn = inspect.stack()[0][3]

        if not self.bypass_validation: self.validate_idxs('ROW', ROW_IDXS_AS_INT_OR_LIST, fxn)

        if self.is_int(ROW_IDXS_AS_INT_OR_LIST): ROW_IDXS = tuple(ROW_IDXS_AS_INT_OR_LIST,)
        else: ROW_IDXS = tuple(np.array(ROW_IDXS_AS_INT_OR_LIST).reshape((1,-1))[0].tolist())
        del ROW_IDXS_AS_INT_OR_LIST

        if not self.bypass_validation:
            self.validate_idx_dtypes_int('ROW', ROW_IDXS, fxn)
            if max(ROW_IDXS) > self._rows-1: raise self.exception_(f'INDICATED INDICES TO SELECT ROWS ARE OUT OF RANGE', fxn)

        is_list = self.current_format=='ARRAY'
        is_dict = self.current_format=='SPARSE_DICT'
        is_col = self.current_orientation=='COLUMN'
        is_row = self.current_orientation=='ROW'

        if is_list:
            if is_col: self.OBJECT =  np.delete(self.OBJECT, ROW_IDXS, axis=1)
            elif is_row: self.OBJECT = np.delete(self.OBJECT, ROW_IDXS, axis=0)

        elif is_dict:
            if is_col: self.OBJECT = sd.delete_inner_key(self.OBJECT, ROW_IDXS)[0]
            elif is_row: self.OBJECT = sd.delete_outer_key(self.OBJECT, ROW_IDXS)[0]

        self._rows -= len(ROW_IDXS)

        # DONT CHANGE ANY current_formats, current_orients, etc. BECAUSE self.OBJECT HAS NOT CHANGED IN THAT WAY

        del is_list, is_dict, is_col, is_row, ROW_IDXS

        return self.OBJECT


    def delete_columns(self, COL_IDXS_AS_INT_OR_LIST, HEADER_OR_FULL_SUPOBJ=None, CONTEXT=None):
        '''Returns OBJECT and modifies in-place.'''
        fxn = inspect.stack()[0][3]

        self.validate_idxs('COL', COL_IDXS_AS_INT_OR_LIST, fxn)

        if not self.bypass_validation: self.validate_idxs('COL', COL_IDXS_AS_INT_OR_LIST, fxn)

        if self.is_int(COL_IDXS_AS_INT_OR_LIST): COL_IDXS = tuple(COL_IDXS_AS_INT_OR_LIST, )
        else: COL_IDXS = tuple(np.array(COL_IDXS_AS_INT_OR_LIST).reshape((1, -1))[0].tolist())
        del COL_IDXS_AS_INT_OR_LIST

        if not CONTEXT is None: self.CONTEXT = ldv.list_dict_validater(CONTEXT, 'CONTEXT')[1][0].tolist()
        if not HEADER_OR_FULL_SUPOBJ is None: self.HEADER_OR_FULL_SUPOBJ = \
            ldv.list_dict_validater(HEADER_OR_FULL_SUPOBJ, 'HEADER_OR_FULL_SUPOBJ')[1]

        if not self.bypass_validation:
            self.validate_idx_dtypes_int('COL', COL_IDXS, fxn)
            if max(COL_IDXS) > self._cols - 1: raise self.exception_(f'AT LEAST ONE INDEX IS OUT OF RANGE', fxn)
            if not self.HEADER_OR_FULL_SUPOBJ is None and self.HEADER_OR_FULL_SUPOBJ.shape[1] != self._cols:
                raise self.exception_(f'NUMBER OF COLUMNS IN SUP_OBJ PASSED TO delete_columns DOES NOT EQUAL COLUMNS IN {self.name} OBJECT.', fxn)

        # GET THE OBJECT SMALL FIRST, THEN LET ObjectOrienter DEAL W HEAVY return_format & return_orientation OPERATIONS
        is_list = self.current_format == 'ARRAY'
        is_dict = self.current_format == 'SPARSE_DICT'
        is_col = self.current_orientation == 'COLUMN'
        is_row = self.current_orientation == 'ROW'

        if is_list:
            if is_col: self.OBJECT = np.delete(self.OBJECT, COL_IDXS, axis=0)
            elif is_row: self.OBJECT = np.delete(self.OBJECT, COL_IDXS, axis=1)

        elif is_dict:
            if is_col: self.OBJECT = sd.delete_outer_key(self.OBJECT, COL_IDXS)[0]
            elif is_row: self.OBJECT = sd.delete_inner_key(self.OBJECT, COL_IDXS)[0]

        # DONT CHANGE ANY current_formats, current_orients, etc. BECAUSE self.OBJECT HAS NOT CHANGED IN THAT WAY

        # THIS MUST BE BEFORE ACTUAL DELETE OR COULD EXCEPT FOR OUT OF RANGE
        if not self.CONTEXT is None and not self.HEADER_OR_FULL_SUPOBJ is None:
            for col_idx in COL_IDXS:
                self.CONTEXT.append(f'Deleted column "{self.HEADER_OR_FULL_SUPOBJ[0][col_idx]}" from {self.name}.')

        if not self.HEADER_OR_FULL_SUPOBJ is None:
            self.HEADER_OR_FULL_SUPOBJ = np.delete(self.HEADER_OR_FULL_SUPOBJ, COL_IDXS, axis=1)

        self._cols -= len(COL_IDXS)

        del is_list, is_dict, is_col, is_row, COL_IDXS

        return self.OBJECT    # self.HEADER_OR_FULL_SUPOBJ AND self.CONTEXT MUST BE ACCESSED AS ATTRS OF THE INSTANCE


    def insert_row(self, row_idx, ROWS_TO_INSERT, insert_orientation):
        '''Returns OBJECT and modifies in-place.'''
        # DONT US MLObjectOrienter HERE, CREATES CIRCULAR IMPORT WITH MLObject

        fxn = inspect.stack()[0][3]

        # REGARDLESS OF bypass_validation, NEED TO STANDARDIZE FORMAT OF ROWS_TO_INSERT
        ROWS_TO_INSERT = ldv.list_dict_validater(ROWS_TO_INSERT, 'ROWS_TO_INSERT')[1]  # GET INTO [[]] OR {0: {0:}} FORMAT

        if not self.bypass_validation:
            if not row_idx in range(self._rows+1): raise self.exception_(f'row_idx OUT OF RANGE MUST BE IN [0, {self._rows}]', fxn)

            insert_orientation = akv.arg_kwarg_validater(insert_orientation, 'insert_orientation',
                                                              ['ROW','COLUMN'], self.this_module, fxn)

        # GET ROWS_TO_INSERT INTO NP FORMAT & [[col0,col1,col2],[col0,col1,col2]] ORIENTATION
        if isinstance(ROWS_TO_INSERT, dict): ROWS_TO_INSERT = sd.unzip_to_ndarray_float64(ROWS_TO_INSERT)[0]
        if insert_orientation != 'ROW': ROWS_TO_INSERT = ROWS_TO_INSERT.transpose()
        del insert_orientation   # ROWS_TO_INSERT orientation MUST NOW BE current_orientation

        # COLUMNS IN INSERT MUST == COLUMNS IN OBJECT
        insert_shape = gs.get_shape('ROWS_TO_INSERT', ROWS_TO_INSERT, 'ROW')
        if insert_shape[1] != self._cols:
            raise self.exception_(f'INSERT OBJECT COLUMNS ({insert_shape[1]}) DOES NOT EQUAL OBJECT COLUMNS ({self._cols})', fxn)
        del insert_shape

        if self.current_format=='ARRAY':
            if self.current_orientation=='COLUMN': self.OBJECT = np.insert(self.OBJECT, row_idx, ROWS_TO_INSERT, axis=1)
            elif self.current_orientation=='ROW': self.OBJECT = np.insert(self.OBJECT, row_idx, ROWS_TO_INSERT, axis=0)

        elif self.current_format=='SPARSE_DICT':
            if self.current_orientation=='COLUMN': self.OBJECT = sd.core_insert_inner(self.OBJECT, row_idx, ROWS_TO_INSERT)
            elif self.current_orientation=='ROW': self.OBJECT = sd.core_insert_outer(self.OBJECT, row_idx, ROWS_TO_INSERT)

        # BEAR PUT CONTEXT STUFF FROM insert_column HERE. SEE BOTTOM OF insert_column.
        # MAY END UP TAKING CONTEXT KWARG OUT OF ALL FUNCTIONS AND RELYING ON "CONTEXT" ATTR

        self._rows += len(ROWS_TO_INSERT)

        # DONT CHANGE ANY current_formats, current_orients, etc. BECAUSE self.OBJECT HAS NOT CHANGED IN THAT WAY

        return self.OBJECT    # self.HEADER_OR_FULL_SUPOBJ AND self.CONTEXT MUST BE ACCESSED AS ATTRS OF INSTANCE


    def insert_column(self, col_idx, COLUMNS_TO_INSERT, insert_orientation, HEADER_OR_FULL_SUPOBJ=None, CONTEXT=None,
                      SUPOBJ_INSERT=None):
        '''Returns OBJECT and modifies in-place.'''
        # SUPOBJ_INSERT MUST BE PASSED AS SINGLE VECTOR, EITHER
        # DONT US MLObjectOrienter HERE, CREATES CIRCULAR IMPORT WITH MLObject

        fxn = inspect.stack()[0][3]

        # REGARDLESS OF bypass_validation, NEED TO STANDARDIZE FORMAT OF COLUMNS_TO_INSERT
        COLUMNS_TO_INSERT = ldv.list_dict_validater(COLUMNS_TO_INSERT, 'COLUMNS_TO_INSERT')[1]  # GET INTO [[]] OR {0: {0:}} FORMAT

        self.HEADER_OR_FULL_SUPOBJ = ldv.list_dict_validater(HEADER_OR_FULL_SUPOBJ, 'HEADER_OR_FULL_SUPOBJ')[1]
        self.CONTEXT = ldv.list_dict_validater(CONTEXT if not CONTEXT is None else [], 'CONTEXT')[1][0].tolist()

        if not SUPOBJ_INSERT is None:
            subobj_format, SUPOBJ_INSERT = ldv.list_dict_validater(SUPOBJ_INSERT, 'SUPOBJ_INSERT') # MAKE BE [[]]
            if subobj_format != 'ARRAY':
                raise self.exception_(f'SUPOBJ_INSERT, IF PASSED, MUST BE A LIST-TYPE THAT CAN BE CONVERTED TO AN NP ARRAY', fxn)

            # IF THIS EXCEPTS, IT IS BECAUSE TRYING TO SET A FULL SUPOBJ INSERT THAT HAS A [] IN IT TO A TEXT TYPE
            # IF IT DOES NOT EXCEPT, THEN IS HEADER ONLY
            try: SUPOBJ_INSERT = SUPOBJ_INSERT.astype('<U10000')
            except: SUPOBJ_INSERT = SUPOBJ_INSERT.astype(object)

        if not self.bypass_validation:
            if not col_idx in range(self._cols+1):
                raise self.exception_(f'col_idx OUT OF RANGE MUST BE IN [0, {self._cols}]', fxn)

            insert_orientation = akv.arg_kwarg_validater(insert_orientation, 'insert_orientation',
                                                              ['ROW','COLUMN'], self.this_module, fxn)

            if not self.HEADER_OR_FULL_SUPOBJ is None and self.HEADER_OR_FULL_SUPOBJ.shape[1] != self._cols:
                raise self.exception_(f'NUMBER OF COLUMNS IN SUP_OBJ ({self.HEADER_OR_FULL_SUPOBJ.shape[1]}) DOES NOT '
                                      f'EQUAL COLUMNS IN {self.name} OBJECT ({self._cols}).', fxn)

            if not SUPOBJ_INSERT is None:
                if self.HEADER_OR_FULL_SUPOBJ is None: raise self.exception_(f'IF SUPOBJ_INSERT IS PASSED TO insert_column HEADER_OR_FULL_SUPOBJ MUST ALSO BE PASSED')

            if not self.HEADER_OR_FULL_SUPOBJ is None:
                if SUPOBJ_INSERT is None: raise self.exception_(f'IF HEADER_OR_FULL_SUPOBJ IS PASSED TO insert_column SUPOBJ_INSERT MUST ALSO BE PASSED', fxn)
                if True not in map(lambda x: x in [1, len(msod.master_support_object_dict())], SUPOBJ_INSERT.shape):
                    raise self.exception_(f'NO DIMENSION OF SUPOBJ_INSERT CORRESPONDS TO A HEADER-ONLY OR FULL SUPPORT OBJECT')
                # IF map OF type TO [:, 0] OF SUPOBJ_INSERT GIVES SAME type, THEN SUPOBJ_INSERT ALREADY IN DESIRED ORIENT
                if len(np.unique(list(map(str, (map(type, (SUPOBJ_INSERT[:, 0])))))))==1: pass
                else: SUPOBJ_INSERT = SUPOBJ_INSERT.transpose()
                if SUPOBJ_INSERT.shape[1] != self.HEADER_OR_FULL_SUPOBJ.shape[0]: raise self.exception_(
                    f'NUMBER OF ROWS IN SUP_OBJ PASSED TO insert_column DOES NOT EQUAL ROWS IN HEADER_OR_FULL_SUPOBJ', fxn)
                if SUPOBJ_INSERT.shape[0] != 1: raise self.exception_(f'SUPOBJ_INSERT CAN ONLY HAVE ONE COLUMN', fxn)


        # GET COLUMNS_TO_INSERT INTO NP FORMAT & [[row0,row1,row2],[row0,row1,row2] ORIENTATION
        if isinstance(COLUMNS_TO_INSERT, dict): COLUMNS_TO_INSERT = sd.unzip_to_ndarray_float64(COLUMNS_TO_INSERT)[0]
        if insert_orientation != 'COLUMN': COLUMNS_TO_INSERT = COLUMNS_TO_INSERT.transpose()
        del insert_orientation   # COLUMNS_TO_INSERT orientation MUST NOW BE current_orientation

        # ROWS IN INSERT MUST == ROWS IN OBJECT
        insert_shape = gs.get_shape('COLUMNS_TO_INSERT', COLUMNS_TO_INSERT, 'COLUMN')
        if insert_shape[0] != self._rows:
            raise self.exception_(f'INSERT OBJECT ROWS ({insert_shape[0]}) DOES NOT EQUAL RECEIVING OBJECT ROWS ({self._rows})', fxn)

        if not SUPOBJ_INSERT is None and SUPOBJ_INSERT.shape[0] != insert_shape[1]:
            raise self.exception_(f'COLUMNS IN SUPOBJ_INSERT ({SUPOBJ_INSERT.shape[0]}) AND COLUMNS_TO_INSERT ({insert_shape[1]}) ARE NOT EQUAL', fxn)
        del insert_shape

        if self.current_format=='ARRAY':
            if self.current_orientation=='COLUMN': self.OBJECT = np.insert(self.OBJECT, col_idx, COLUMNS_TO_INSERT, axis=0)
            elif self.current_orientation=='ROW': self.OBJECT = np.insert(self.OBJECT, col_idx, COLUMNS_TO_INSERT, axis=1)
        elif self.current_format=='SPARSE_DICT':
            if self.current_orientation=='COLUMN': self.OBJECT = sd.core_insert_outer(self.OBJECT, col_idx, COLUMNS_TO_INSERT)
            elif self.current_orientation=='ROW': self.OBJECT = sd.core_insert_inner(self.OBJECT, col_idx, COLUMNS_TO_INSERT)


        if not self.HEADER_OR_FULL_SUPOBJ is None:
            # SUPOBJ_INSERT CANNOT BE None IF HEADER_OR_FULL_SUPOBJ WAS PASSED
            self.HEADER_OR_FULL_SUPOBJ = np.insert(self.HEADER_OR_FULL_SUPOBJ, col_idx, SUPOBJ_INSERT, axis=1)

        # BEAR REMEMBER WHEN THIS IS FIXED PUT IT INTO insert_row()
        if not self.CONTEXT is None:
            self.CONTEXT.append(f'Inserted {f"{SUPOBJ_INSERT[0][0]}" if not SUPOBJ_INSERT is None else "unnamed column"} '
                                f'INTO {self.name} in the {col_idx} index position.')

        self._cols += len(COLUMNS_TO_INSERT)

        # DONT CHANGE ANY current_formats, current_orients, etc. BECAUSE self.OBJECT HAS NOT CHANGED IN THAT WAY

        return self.OBJECT    # self.HEADER_OR_FULL_SUPOBJ AND self.CONTEXT MUST BE ACCESSED AS ATTRS OF INSTANCE







































if __name__ == '__main__':
    # TEST MODULE

    # VERIFIED TEST MODULE AND FUNCTIONAL MODULE ARE GOOD 6/27/23.
    # THIS MODULE DOES NOT TEST THE CONTEXT/SUPOBJ/SUPOBJ_INSERT KWARG FUNCTIONALITY FOR delete_columns AND insert_column

    # return_rows()
    # return_columns()
    # delete_rows()
    # delete_columns()
    # insert_row()
    # insert_column()

    def test_fail(test_name, oper_desc, GIVEN_OBJECT, EXP_OBJECT, ACT_OBJECT):
        print('\033[91m')
        print(f'\n*** {test_name} EPIC FAIL ***\n', )
        print(f'\033[92mOPERATION DESCRIPTION:\n', oper_desc)
        print(f'\n\033[92mGIVEN OBJECT:\n', GIVEN_OBJECT)
        print(f'\n\033[91mEXPECTED OBJECT:\n', EXP_OBJECT)
        print(f'\nACTUAL OBJECT:\n', ACT_OBJECT)
        print('\033[0m')
        wls.winlinsound(444, 500)
        raise Exception(f'*** EPIC FAIL 😂😂😂 ***')


    BASE_NP_OBJECT = np.fromiter(range(25), dtype=np.int8).reshape((5,5))
    '''  BASE LOOKS LIKE
    [[ 0  1  2  3  4]`
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]
     [20 21 22 23 24]]
    '''
    BASE_ANSWER_KEY = np.fromiter(range(25), dtype=np.int8).reshape((5, 5))
    '''  BASE LOOKS LIKE
    [[ 0  1  2  3  4]`
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]
     [20 21 22 23 24]]
    '''

    ######################################################################################################################################
    # TEST return_rows & return_columns ##################################################################################################

    name = 'RETURN_ROW_OR_COLUMN_TEST_OBJECT'
    MASTER_BYPASS_VALIDATION = [True, False]
    MASTER_GIVEN_ORIENTATION = ['ROW', 'COLUMN']
    MASTER_GIVEN_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_IDXS_TO_RETURN = [(0,4),(1,2,3),(0,2), (2,4)]
    MASTER_RETURN_FORMAT = ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN']
    MASTER_RETURN_ORIENTATION = ['ROW', 'COLUMN', 'AS_GIVEN']
    total_trials = np.product(list(map(len, (MASTER_BYPASS_VALIDATION, MASTER_GIVEN_FORMAT, MASTER_GIVEN_ORIENTATION,
                                              MASTER_IDXS_TO_RETURN, MASTER_RETURN_FORMAT, MASTER_RETURN_ORIENTATION))))

    print(f'\033[92mSTART TEST OF return_rows() AND return_columns()')

    ctr = 0
    for bypass_validation in MASTER_BYPASS_VALIDATION:
        for given_orientation in MASTER_GIVEN_ORIENTATION:
            for given_format in MASTER_GIVEN_FORMAT:
                for IDXS_TO_RETURN in MASTER_IDXS_TO_RETURN:
                    for return_orientation in MASTER_RETURN_ORIENTATION:
                        for return_format in MASTER_RETURN_FORMAT:

                            ctr += 1
                            print(f'\033[92mRunning return() trial {ctr} of {total_trials}...\033[0m')

                            oper_desc = f''

                            # GET GIVEN OBJECT READY #########################
                            # ASSUME BASE IS GIVEN AS ROW
                            GIVEN_OBJECT = BASE_NP_OBJECT.copy()
                            if given_orientation == 'COLUMN': GIVEN_OBJECT = GIVEN_OBJECT.transpose()
                            if given_format == 'SPARSE_DICT': GIVEN_OBJECT = sd.zip_list_as_py_int(GIVEN_OBJECT)
                            TestClass = MLRowColumnOperations(GIVEN_OBJECT, given_orientation, name=name, bypass_validation=bypass_validation)

                            return_orientation = given_orientation if return_orientation=='AS_GIVEN'else return_orientation
                            return_format = given_format if return_format=='AS_GIVEN' else return_format

                            # ASSUME BASE ANSWER KEYS ARE GIVEN AS ROW
                            # GET ROW_ANSWER_KEY READY #######################
                            ROW_ANSWER_KEY = BASE_ANSWER_KEY[IDXS_TO_RETURN, ...]
                            if return_orientation == 'COLUMN': ROW_ANSWER_KEY = ROW_ANSWER_KEY.transpose()
                            if return_format == 'SPARSE_DICT': ROW_ANSWER_KEY = sd.zip_list_as_py_int(ROW_ANSWER_KEY)

                            # GET COL_ANSWER_KEY READY #######################
                            COL_ANSWER_KEY = BASE_ANSWER_KEY[..., IDXS_TO_RETURN]
                            if return_orientation == 'COLUMN': COL_ANSWER_KEY = COL_ANSWER_KEY.transpose()
                            if return_format == 'SPARSE_DICT': COL_ANSWER_KEY = sd.zip_list_as_py_int(COL_ANSWER_KEY)

                            ACTUAL_RETURN_ROW = TestClass.return_rows(IDXS_TO_RETURN, return_orientation=return_orientation, return_format=return_format)
                            if return_format=='ARRAY':
                                if not np.array_equiv(ROW_ANSWER_KEY, ACTUAL_RETURN_ROW):
                                    test_fail('return_rows', oper_desc, GIVEN_OBJECT, ROW_ANSWER_KEY, ACTUAL_RETURN_ROW)
                            elif return_format=='SPARSE_DICT':
                                if not sd.core_sparse_equiv(ROW_ANSWER_KEY, ACTUAL_RETURN_ROW):
                                    test_fail('return_rows', oper_desc, GIVEN_OBJECT, ROW_ANSWER_KEY, ACTUAL_RETURN_ROW)

                            # DONT HAVE TO RESET TestClass.OBJECT BECAUSE OBJECT WAS NOT ALTERED BY .return_rows()
                            ACTUAL_RETURN_COLUMN = TestClass.return_columns(IDXS_TO_RETURN, return_orientation=return_orientation, return_format=return_format)
                            if return_format == 'ARRAY':
                                if not np.array_equiv(COL_ANSWER_KEY, ACTUAL_RETURN_COLUMN):
                                    test_fail('return_columns', oper_desc, GIVEN_OBJECT, COL_ANSWER_KEY, ACTUAL_RETURN_COLUMN)
                            elif return_format == 'SPARSE_DICT':
                                if not sd.core_sparse_equiv(COL_ANSWER_KEY, ACTUAL_RETURN_COLUMN):
                                    test_fail('return_columns', oper_desc, GIVEN_OBJECT, COL_ANSWER_KEY, ACTUAL_RETURN_COLUMN)

    print(f'\n\033[92m*** RETURN TESTS COMPLETE. return_rows() AND return_columns() PASSED ***\033[0m\n')
    # END TEST return_rows & return_columns ##############################################################################################
    ######################################################################################################################################

    print(f'\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n')

    ######################################################################################################################################
    # TEST delete_rows & delete_columns ##################################################################################################
    name = 'DELETE_ROW_OR_COLUMN_TEST_OBJECT'
    MASTER_BYPASS_VALIDATION = [False, True]
    MASTER_GIVEN_ORIENTATION = ['ROW', 'COLUMN']
    MASTER_GIVEN_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_IDXS_TO_DELETE = [(0,4),(1,2,3),(0,2),(2,4)]
    total_trials = np.product(list(map(len, (MASTER_BYPASS_VALIDATION, MASTER_GIVEN_ORIENTATION, MASTER_GIVEN_FORMAT,
                                               MASTER_IDXS_TO_DELETE))))

    print(f'\033[92mSTART TEST OF delete_rows() AND delete_columns()')

    ctr = 0
    for bypass_validation in MASTER_BYPASS_VALIDATION:
        for given_orientation in MASTER_GIVEN_ORIENTATION:
            for given_format in MASTER_GIVEN_FORMAT:
                for IDXS_TO_DELETE in MASTER_IDXS_TO_DELETE:

                    ctr += 1
                    print(f'\033[92mRunning delete() trial {ctr} of {total_trials}...\033[0m')

                    oper_desc = f''

                    # GET GIVEN OBJECT READY #########################
                    # ASSUME BASE IS GIVEN AS ROW
                    GIVEN_OBJECT = BASE_NP_OBJECT.copy()
                    if given_orientation == 'COLUMN': GIVEN_OBJECT = GIVEN_OBJECT.transpose()
                    if given_format == 'SPARSE_DICT': GIVEN_OBJECT = sd.zip_list_as_py_int(GIVEN_OBJECT)
                    TestClass = MLRowColumnOperations(GIVEN_OBJECT, given_orientation, name=name,
                                                    bypass_validation=bypass_validation)


                    # ASSUME BASE ANSWER KEYS ARE GIVEN AS ROW
                    # GET ROW_ANSWER_KEY READY #######################
                    ROW_ANSWER_KEY = np.delete(BASE_ANSWER_KEY, IDXS_TO_DELETE, axis=0)
                    if given_orientation == 'COLUMN': ROW_ANSWER_KEY = ROW_ANSWER_KEY.transpose()
                    if given_format == 'SPARSE_DICT': ROW_ANSWER_KEY = sd.zip_list_as_py_int(ROW_ANSWER_KEY)

                    # GET COL_ANSWER_KEY READY #######################
                    COL_ANSWER_KEY = np.delete(BASE_ANSWER_KEY, IDXS_TO_DELETE, axis=1)
                    if given_orientation == 'COLUMN': COL_ANSWER_KEY = COL_ANSWER_KEY.transpose()
                    if given_format == 'SPARSE_DICT': COL_ANSWER_KEY = sd.zip_list_as_py_int(COL_ANSWER_KEY)

                    ACTUAL_RETURN_ROW = TestClass.delete_rows(IDXS_TO_DELETE)

                    if given_format == 'ARRAY':
                        if not np.array_equiv(ROW_ANSWER_KEY, ACTUAL_RETURN_ROW):
                            test_fail('delete_rows', oper_desc, GIVEN_OBJECT, ROW_ANSWER_KEY, ACTUAL_RETURN_ROW)
                    elif given_format == 'SPARSE_DICT':
                        if not sd.core_sparse_equiv(ROW_ANSWER_KEY, ACTUAL_RETURN_ROW):
                            test_fail('delete_rows', oper_desc, GIVEN_OBJECT, ROW_ANSWER_KEY, ACTUAL_RETURN_ROW)

                    # RESET self.OBJECT AFTER BUTCHERY BY .delete_rows()
                    TestClass.OBJECT = GIVEN_OBJECT

                    ACTUAL_RETURN_COLUMN = TestClass.delete_columns(IDXS_TO_DELETE)

                    if given_format == 'ARRAY':
                        if not np.array_equiv(COL_ANSWER_KEY, ACTUAL_RETURN_COLUMN):
                            test_fail('delete_columns', oper_desc, GIVEN_OBJECT, COL_ANSWER_KEY, ACTUAL_RETURN_COLUMN)
                    elif given_format == 'SPARSE_DICT':
                        if not sd.core_sparse_equiv(COL_ANSWER_KEY, ACTUAL_RETURN_COLUMN):
                            test_fail('delete_columns', oper_desc, GIVEN_OBJECT, COL_ANSWER_KEY, ACTUAL_RETURN_COLUMN)

    print(f'\n\033[92m*** DELETE TESTS COMPLETE. delete_rows() AND delete_columns() PASSED ***\033[0m\n')
    # END TEST delete_rows & delete_columns ##############################################################################################
    ######################################################################################################################################

    print(f'\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n')

    ######################################################################################################################################
    # TEST insert_row & insert_column ####################################################################################################
    name = 'INSERT_ROW_OR_COLUMN_TEST_OBJECT'
    MASTER_BYPASS_VALIDATION = [True, False]
    MASTER_GIVEN_ORIENTATION = ['ROW', 'COLUMN']
    MASTER_GIVEN_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_IDXS_TO_INSERT = [0, 2, 4]
    MASTER_INSERT_SIZE = [1,2]
    MASTER_INSERT_GIVEN_FORMAT = ['ARRAY','SPARSE_DICT']
    MASTER_INSERT_GIVEN_ORIENTATION = ['ROW', 'COLUMN']
    MASTER_INSERT_GIVEN_SINGLE_DOUBLE = ['SINGLE', 'DOUBLE']
    total_trials = np.product(list(map(len, (MASTER_BYPASS_VALIDATION, MASTER_GIVEN_ORIENTATION, MASTER_GIVEN_FORMAT,
        MASTER_IDXS_TO_INSERT, MASTER_INSERT_SIZE, MASTER_INSERT_GIVEN_FORMAT, MASTER_INSERT_GIVEN_ORIENTATION,
        MASTER_INSERT_GIVEN_SINGLE_DOUBLE))))

    print(f'\033[92mSTART TEST OF insert_row() AND insert_column()')

    ctr = 0
    for bypass_validation in MASTER_BYPASS_VALIDATION:
        for given_orientation in MASTER_GIVEN_ORIENTATION:
            for given_format in MASTER_GIVEN_FORMAT:
                for idx_to_insert in MASTER_IDXS_TO_INSERT:
                    for insert_size in MASTER_INSERT_SIZE:
                        for insert_given_format in MASTER_INSERT_GIVEN_FORMAT:
                            for insert_given_orientation in MASTER_INSERT_GIVEN_ORIENTATION:
                                for insert_given_single_double in MASTER_INSERT_GIVEN_SINGLE_DOUBLE:
                                    ctr += 1
                                    print(f'\033[92mRunning insert() trial {ctr} of {total_trials}...\033[0m')

                                    oper_desc = f'\033[92mInsert a {insert_given_single_double if insert_size==1 else "DOUBLE"} {given_format} ' \
                                                f'of shape ({insert_size},5) oriented as {given_orientation} into the {idx_to_insert} index of a ' \
                                                f'5 x 5 {given_format} oriented as {given_orientation}. ' \
                                                f'bypass_validation is {bypass_validation}.\033[0m'

                                    # GET GIVEN OBJECT READY #########################
                                    # ASSUME BASE IS GIVEN AS ROW
                                    GIVEN_OBJECT = BASE_NP_OBJECT.copy()
                                    if given_orientation == 'COLUMN': GIVEN_OBJECT = GIVEN_OBJECT.transpose()
                                    if given_format == 'SPARSE_DICT': GIVEN_OBJECT = sd.zip_list_as_py_int(GIVEN_OBJECT)

                                    # BASE INSERT OBJECT IS [[x,x,x]] OR [x,x,x]
                                    if insert_size == 1:
                                        BASE_INSERT_OBJECT = np.fromiter(range(95, 100), dtype=np.int8).reshape((1, -1))
                                        ROW_INSERT_OBJECT = BASE_INSERT_OBJECT.copy()
                                        COL_INSERT_OBJECT = BASE_INSERT_OBJECT.copy()
                                    elif insert_size == 2:
                                        BASE_INSERT_OBJECT = np.fromiter(range(90, 100), dtype=np.int8).reshape((2, -1))
                                        ROW_INSERT_OBJECT = BASE_INSERT_OBJECT.copy()
                                        COL_INSERT_OBJECT = BASE_INSERT_OBJECT.copy()

                                    if insert_given_orientation=='ROW':
                                        COL_INSERT_OBJECT = COL_INSERT_OBJECT.transpose()   # CAN ONLY BE DOUBLE
                                        if insert_size==1:
                                            ROW_INSERT_OBJECT = ROW_INSERT_OBJECT.reshape((1,-1))
                                        if insert_given_format=='SPARSE_DICT':
                                            COL_INSERT_OBJECT = sd.zip_list_as_py_int(COL_INSERT_OBJECT)
                                            ROW_INSERT_OBJECT = sd.zip_list_as_py_int(ROW_INSERT_OBJECT)
                                        if insert_given_single_double=='SINGLE' and insert_size==1: ROW_INSERT_OBJECT = ROW_INSERT_OBJECT[0]
                                    elif insert_given_orientation=='COLUMN':
                                        ROW_INSERT_OBJECT = ROW_INSERT_OBJECT.transpose()    # CAN ONLY BE DOUBLE
                                        if insert_size==1:
                                            COL_INSERT_OBJECT = COL_INSERT_OBJECT.reshape((1,-1))
                                        if insert_given_format=='SPARSE_DICT':
                                            COL_INSERT_OBJECT = sd.zip_list_as_py_int(COL_INSERT_OBJECT)
                                            ROW_INSERT_OBJECT = sd.zip_list_as_py_int(ROW_INSERT_OBJECT)
                                        if insert_given_single_double=='SINGLE' and insert_size==1: COL_INSERT_OBJECT = COL_INSERT_OBJECT[0]

                                    # ASSUME BASE ANSWER KEYS ARE GIVEN AS ROW
                                    # GET ROW_ANSWER_KEY READY #######################
                                    ROW_ANSWER_KEY = np.insert(BASE_ANSWER_KEY, idx_to_insert, BASE_INSERT_OBJECT, axis=0)
                                    if given_orientation == 'COLUMN': ROW_ANSWER_KEY = ROW_ANSWER_KEY.transpose()
                                    if given_format == 'SPARSE_DICT': ROW_ANSWER_KEY = sd.zip_list_as_py_int(ROW_ANSWER_KEY)

                                    # GET COL_ANSWER_KEY READY #######################
                                    COL_ANSWER_KEY = np.insert(BASE_ANSWER_KEY, idx_to_insert, BASE_INSERT_OBJECT, axis=1)
                                    if given_orientation == 'COLUMN': COL_ANSWER_KEY = COL_ANSWER_KEY.transpose()
                                    if given_format == 'SPARSE_DICT': COL_ANSWER_KEY = sd.zip_list_as_py_int(COL_ANSWER_KEY)

                                    TestClass = MLRowColumnOperations(GIVEN_OBJECT, given_orientation, name=name,
                                                                    bypass_validation=bypass_validation)

                                    ACTUAL_RETURN_ROW = TestClass.insert_row(idx_to_insert, ROW_INSERT_OBJECT, insert_given_orientation)
                                    if given_format == 'ARRAY':
                                        if not np.array_equiv(ROW_ANSWER_KEY, ACTUAL_RETURN_ROW):
                                            test_fail('insert_row', oper_desc, GIVEN_OBJECT, ROW_ANSWER_KEY, ACTUAL_RETURN_ROW)
                                    elif given_format == 'SPARSE_DICT':
                                        if not sd.core_sparse_equiv(ROW_ANSWER_KEY, ACTUAL_RETURN_ROW):
                                            test_fail('insert_row', oper_desc, GIVEN_OBJECT, ROW_ANSWER_KEY, ACTUAL_RETURN_ROW)

                                    # RESET self.OBJECT AFTER BUTCHERY BY .insert_row()
                                    TestClass = MLRowColumnOperations(GIVEN_OBJECT, given_orientation, name=name,
                                                                      bypass_validation=bypass_validation)

                                    ACTUAL_RETURN_COLUMN = TestClass.insert_column(idx_to_insert, COL_INSERT_OBJECT, insert_given_orientation)
                                    if given_format == 'ARRAY':
                                        if not np.array_equiv(COL_ANSWER_KEY, ACTUAL_RETURN_COLUMN):
                                            test_fail('insert_column', oper_desc, GIVEN_OBJECT, COL_ANSWER_KEY, ACTUAL_RETURN_COLUMN)
                                    elif given_format == 'SPARSE_DICT':
                                        if not sd.core_sparse_equiv(COL_ANSWER_KEY, ACTUAL_RETURN_COLUMN):
                                            test_fail('insert_column', oper_desc, GIVEN_OBJECT, COL_ANSWER_KEY, ACTUAL_RETURN_COLUMN)

    print(f'\n\033[92m*** DELETE TESTS COMPLETE. insert_row() AND insert_column() PASSED ***\033[0m\n')
    # END TEST insert_row & insert_column ################################################################################################
    ######################################################################################################################################




    print(f'\n\033[92m*** TESTS COMPLETE. ALL PASSED ***\033[0m\n')
    for _ in range(3): wls.winlinsound(888, 500); time.sleep(1)































