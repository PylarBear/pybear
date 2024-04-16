import inspect, sys, time, warnings
import numpy as np, sparse_dict as sd
from debug import get_module_name as gmn
from copy import deepcopy
from general_list_ops import list_select as ls
from general_data_ops import new_np_random_choice as nnrc
from MLObjects.SupportObjects import master_support_object_dict as msod
from MLObjects.TestObjectCreators import ApexCreate as ac, CreateColumn as crco
from data_validation import validate_user_input as vui

'''OBJECT, HEADER, AND SUPPORT OBJECTS ARE ATTRIBUTES OF THE CLASS. NOTHING IS RETURNED.'''

# MODULE FOR CREATING DATA, TARGET, REFVECS, OR TEST OBJECTS W RANDOM DATA OR FROM GIVEN OBJECTS


# OUTLINE





''' BEAR FINISH
# ARG/KWARG RULES
   # ONLY OBJECT_HEADER, FULL_SUPOBJ_OR_SINGLE_MDTYPES OR columns CAN SET OBJECT COLUMNS

    return_format,                          # MUST BE ENTERED, 'ARRAY', 'SPARSE_DICT'
    return_orientation,                     # MUST BE ENTERED, 'ROW', 'COLUMN'   (CANNOT BE 'AS_GIVEN')
    rows,                                   # MUST BE ENTERED, INTEGER > 0
    name = None,                            # OPTIONAL, ENTER AS STR
    OBJECT_HEADER = None,                   # OPTIONAL, ENTER AS [[]], WILL BE USED IN DETERMINING # OF COLUMNS (MUST MATCH len SUPOBJ AND OTHER BUILD INSTRUCTION OBJECTS)
    FULL_SUPOBJ_OR_SINGLE_MDTYPES = None,   # OPTIONAL, CAN BE A SINGLE MODIFIED_DATATYPES VECTOR OR A FULL SUPPORT_OBJECT, USED TO BUILD OBJECT
    BUILD_FROM_MOD_DTYPES = None,           # OPTIONAL, AS [], VECTOR OF ALLOWED MOD_DTYPES TO BUILD FROM IF FULL_OR_SINGLE NOT GIVEN
    columns = None,                         # INTEGER > 0, THE INTENT IS THAT columns IS ONLY GIVEN IF HEADER OR SUPOBJ ARE NOT
    NUMBER_OF_CATEGORIES = None,            # AS [] OR INT, ONLY NEEDED IF STR-TYPES IN MOD_DTYPES, DEFAULTS TO default_number_of_categories IF NOT GIVEN
    MIN_VALUES = None,                      # AS [] OR INT, ONLY NEEDED IF NUM-TYPES IN MOD_DTYPES, DEFAULTS TO default_min_value IF NOT GIVEN
    MAX_VALUES = None,                      # AS [] OR INT, ONLY NEEDED IF NUM-TYPES IN MOD_DTYPES, DEFAULTS TO default_mix_value IF NOT GIVEN
    SPARSITIES = None,                      # AS [] OR INT, ONLY NEEDED IF NUM-TYPES IN MOD_DTYPES, DEFAULTS TO default_sparsity NOT GIVEN
    WORD_COUNT = None,  # FOR SPLIT_STR & NNLM50
    POOL_SIZE = None,  # FOR SPLIT_STR & NNLM50
    override_sup_obj = None,
    bypass_validation = None
'''





# INHERITS METHODS
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

# OVERWRITES
# build

# UNIQUE
# sd_strtype_handle


class CreateFromScratch(ac.ApexCreate):   #ApexCreate FOR METHODS ONLY, DONT DO super()!

    def __init__(self,
                 return_format,         # MUST ENTER THIS AND CANT BE AS_GIVEN BECAUSE NOTHING'S GIVEN
                 return_orientation,         # MUST ENTER THIS AND CANT BE AS_GIVEN BECAUSE NOTHING'S GIVEN
                 rows,                    # MUST BE ENTERED
                 name=None,
                 OBJECT_HEADER=None,   # ONLY OBJECT_HEADER, FULL_SUPOBJ_OR_SINGLE_MDTYPES OR columns CAN SET OBJECT COLUMNS
                 # CAN BE A SINGLE MODIFIED_DATATYPES VECTOR OR A FULL SUPPORT_OBJECT, USED TO BUILD OBJECT
                 FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                 BUILD_FROM_MOD_DTYPES=None, # VECTOR OF ALLOWED MOD_DTYPES TO BUILD FROM IF FULL_OR_SINGLE NOT GIVEN
                 #### MOD_DTYPES IS RANDOMLY FILLED FROM BUILD_FROM_MOD_DTYPES IF OBJECT AND MOD_DTYPES IS NOT GIVEN
                 columns=None,    # THE INTENT IS THAT columns IS ONLY GIVEN IF HEADER OR M_DTYPES ARE NOT
                 #### IGNORED IF MOD_DTYPES NOT GIVEN (FULL_SUPOBJ_OR_SINGLE_MDTYPES MUST BE GIVEN FOR THESE TO WORK)
                 NUMBER_OF_CATEGORIES=None, # ONLY NEEDED IF STR-TYPES IN MOD_DTYPES, RANDOMLY SELECTED FROM [2,10] IF NOT GIVENv
                 MIN_VALUES=None,  # ONLY NEEDED IF NUM-TYPES IN MOD_DTYPES, DEFAULTS TO default_min_value IF NOT GIVEN
                 MAX_VALUES=None,  # ONLY NEEDED IF NUM-TYPES IN MOD_DTYPES, DEFAULTS TO default_mix_value IF NOT GIVEN
                 SPARSITIES=None,  # ONLY NEEDED IF NUM-TYPES IN MOD_DTYPES, DEFAULTS TO default_sparsity NOT GIVEN
                 #### END IGNORED IF MOD_DTYPES NOT GIVEN (FULL_SUPOBJ_OR_SINGLE_MDTYPES MUST BE GIVEN FOR THESE TO WORK)
                 WORD_COUNT=None,  # FOR SPLIT_STR & NNLM50
                 POOL_SIZE=None,   # FOR SPLIT_STR & NNLM50
                 override_sup_obj=None,
                 bypass_validation=None
                 ):

        # 3/8/23 THE AIM OF init IS TO USE ALL THESE args/kwargs TO GENERATE A FULL & VALIDATED SUPOBJ, AND INSTRUCTOR OBJECTS
        # (MIN_VALUES, MAX_VALUES, NUMBER_OF_CATEGORIES, SPARSITIES, WORD_COUNT, POOL_SIZE) TO SEND TO self.build() TO
        # BUILD THE FINAL OUTPUT OBJECT

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        allowed_return_format = ['ARRAY', 'SPARSE_DICT']
        allowed_return_orientation = ['ROW', 'COLUMN']
        super().__init__(OBJECT_HEADER, FULL_SUPOBJ_OR_SINGLE_MDTYPES, bypass_validation, name, return_format, allowed_return_format,
                 return_orientation, allowed_return_orientation, override_sup_obj, self.this_module, fxn)
        del allowed_return_format, allowed_return_orientation

        # DECLARED self.this_module, fxn
        # super() VALIDATED HEADER, FULL_SUP_OBJ_OR_SINGLE_MDTYPES, self.bypass_validation, self.name, self.return_format,
        #       self.return_orientation, self.override_sup_obj, self.calling_module, self.calling_fxn, self.mdtype_idx
        #       BUILT self.LOOKUP_DICT

        ################################################################################################################################
        # SET / VALIDATED DEFAULT INSTRUCTOR VALUES ####################################################################################
        def default_value_setter(_OBJECT, _default_value):  # IF GIVEN AS A NUMBER, SET _OBJECT TO None AND default TO THE NUMBER
            # DONT KNOW self.columns YET SO CANT JUST BUILD THE OBJECT W default YET
            if isinstance(_OBJECT, (int, float)): return None, _OBJECT    # IF GIVEN AS LIST KEEP AS IS WITH HARD default VALUE
            else: return _OBJECT, _default_value

        MIN_VALUES, default_min_value =  default_value_setter(MIN_VALUES, -10)
        MAX_VALUES, default_max_value =  default_value_setter(MAX_VALUES, 10)
        SPARSITIES, default_sparsity =  default_value_setter(SPARSITIES, 0)
        NUMBER_OF_CATEGORIES, default_number_of_categories =  default_value_setter(NUMBER_OF_CATEGORIES, 10)
        WORD_COUNT, default_word_count =  default_value_setter(WORD_COUNT, 20)
        POOL_SIZE, default_pool_size =  default_value_setter(POOL_SIZE, 50)

        del default_value_setter

        if not self.bypass_validation:

            if 'INT' in str(type(default_min_value)).upper() and 'INT' in str(type(default_max_value)).upper() and \
                    default_min_value<=default_max_value: pass
            else: self._exception(f'min_value AND max_values WHEN ENTERED AS NUMBERS MUST BE INTEGERS AND min_value <= max_value.')

            if 'INT' in str(type(default_sparsity)).upper() and default_sparsity>=0 and default_sparsity<=100: pass
            else: self._exception(f'sparsity MUST BE AN INTEGER >=0 AND <=100.')

            if 'INT' in str(type(default_number_of_categories)).upper() and default_number_of_categories >= 1: pass
            else: self._exception(f'default_number_of_categories MUST BE AN INTEGER > 1.')

            if 'INT' in str(type(default_word_count)).upper() and default_word_count > 0: pass
            else: self._exception(f'word_count MUST BE AN INTEGER > 0.')

            if 'INT' in str(type(default_pool_size)).upper() and default_pool_size > 0: pass
            else: self._exception(f'pool_size MUST BE AN INTEGER > 0.')
        # END SET / VALIDATED DEFAULT INSTRUCTOR VALUES ####################################################################################
        ################################################################################################################################

        # TO SATISFY Apex build_full_supobj ARGS
        self.OBJECT = None
        self.given_format = None
        self.given_orientation = None

        # DECLARED self.this_module, fxn
        # super() VALIDATED HEADER, FULL_SUP_OBJ_OR_SINGLE_MDTYPES, self.bypass_validation, self.name, self.return_format,
        #       self.return_orientation, self.override_sup_obj, self.calling_module, self.calling_fxn, self.mdtype_idx
        #       BUILT self.LOOKUP_DICT
        # DECLARED/VALIDATED/OVERWROTE default_values
        # DECLARED/VALIDATED self.OBJECT, self.given_format, self.given_orientation


        # columns & rows ########################################################################################################
        # NEED TO KNOW COLUMNS BEFORE CAN BUILD MODIFIED_DATATYPES (IE FULL SUPOBJ) FROM BUILD_FROM AND BUILD OTHER BUILD INSTRUCTORS
        if self.bypass_validation:
            self.rows = rows
            self.columns = columns
        elif not self.bypass_validation:
            if 'INT' in str(type(rows)).upper() and rows > 0: self.rows = rows
            else: self._exception(f'rows MUST BE AN INTEGER > 0.')

        # IF HEADER AND M_DTYPES NOT GIVEN, NEED columns AS arg, OTHERWISE
        # columns KWARG IS IGNORED AND WHATEVER OBJECTS ARE GIVEN ARE USED TO GET columns
        if self.OBJECT_HEADER is None and self.FULL_SUPOBJ_OR_SINGLE_MDTYPES is None:
            if self.bypass_validation: self.columns = columns
            elif not self.bypass_validation:
                if 'INT' in str(type(rows)).upper() and rows > 0: self.columns = columns
                else: self._exception(f'NEITHER HEADER NOR MOD DTYPES ARE GIVEN SO columns MUST BE GIVEN AS AN INTEGER > 0.')
            self.validate_full_supobj_or_single_mdtypes()
        elif not self.OBJECT_HEADER is None and self.FULL_SUPOBJ_OR_SINGLE_MDTYPES is None:
            self.columns = len(self.OBJECT_HEADER[0])
            self.validate_full_supobj_or_single_mdtypes()
        elif self.OBJECT_HEADER is None and not self.FULL_SUPOBJ_OR_SINGLE_MDTYPES is None:
            self.validate_full_supobj_or_single_mdtypes()
            self.columns = len(self.SUPPORT_OBJECTS[0])
        elif not self.OBJECT_HEADER is None and not self.FULL_SUPOBJ_OR_SINGLE_MDTYPES is None:
            self.validate_full_supobj_or_single_mdtypes()
            self.columns = len(self.SUPPORT_OBJECTS[0])
            if len(self.OBJECT_HEADER[0]) != self.columns:
                self._exception(f'lens OF GIVEN HEADER AND GIVEN FULL_SUPOBJ_OR_SINGLE_MDTYPES ARE NOT EQUAL', fxn=fxn)
        # END columns & rows ########################################################################################################

        # DECLARED self.this_module, fxn
        # super() VALIDATED HEADER, FULL_SUP_OBJ_OR_SINGLE_MDTYPES, self.bypass_validation, self.name, self.return_format,
        #       self.return_orientation, self.override_sup_obj, self.calling_module, self.calling_fxn, self.mdtype_idx
        #       BUILT self.LOOKUP_DICT
        # DECLARED/VALIDATED/OVERWROTE default_values
        # DECLARED/VALIDATED self.OBJECT, self.given_format, self.given_orientation
        # VALIDATED rows, GOT / VALIDATED columns, BUILT FULL SUPOBJ AND PUT IN MOD_DTYPES IF AVAILABLE

        ### VALIDATE CONTENT OF MOD DTYPE OBJECTS ##############################################################################
        if not FULL_SUPOBJ_OR_SINGLE_MDTYPES is None:     # NOT self. !!!!   INDICATES INTENT TO BUILD FROM GIVEN MOD_DTYPE VECTOR...
            # ...(WHICH IS NOW LOADED INTO self.SUPPORT_OBJECTS IN mdtypes_idx POSN)
            # IF MOD_DTYPES GIVEN, DONT NEED BUILD_FROM_MOD_DTYPES

            if self.return_format == 'SPARSE_DICT': # CHECK IF RETURNING AS SD AND IF ALLOWED TYPES IN MOD DTYPES
                self.sd_strtype_handle(self.SUPPORT_OBJECTS[self.mdtype_idx], 'MODIFIED_DATATYPES', 'mra')

            elif self.return_format=='ARRAY' and not self.bypass_validation:
                # IF RETURNING AS ARRAY, CHECK IF MOD_DTYPE ENTRIES ARE VALID (IF VALIDATING)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ALLOWED = list(msod.mod_num_dtypes().values()) + list(msod.mod_text_dtypes().values())
                    for _dtype in self.SUPPORT_OBJECTS[self.mdtype_idx]:
                        if _dtype not in ALLOWED:
                            self._exception(f'MOD DTYPE "{_dtype}" IN FULL_SUPOBJ_OR_SINGLE_MDTYPES IS NOT VALID', fxn=fxn)
                    del ALLOWED

            # IF GET HERE, MODIFIED_DATATYPES ARE GOOD TO GO

        elif FULL_SUPOBJ_OR_SINGLE_MDTYPES is None:      # NOT self. !!!!    INDICATES INTENT TO RANDOMLY SELECT DTYPES
            # ALL WE NEED TO KNOW IS WHAT MOD_DTYPES TO RANDOMLY FILL WITH AND THE NUMBER OF COLUMNS.  IF BUILD_FROM_MOD_DTYPES WAS
            # GIVEN THEN VALIDATE IT, IF NOT GIVEN USE ALL MOD_DTYPES AS POOL (DEPENDING IS/ISNT SD, OF COURSE)

            if BUILD_FROM_MOD_DTYPES is None:
                if self.return_format=='ARRAY': BUILD_FROM_MOD_DTYPES = list(msod.mod_num_dtypes().values()) + list(msod.mod_text_dtypes().values())
                elif self.return_format=='SPARSE_DICT': BUILD_FROM_MOD_DTYPES = list(msod.mod_num_dtypes().values())
                else: self._exception(f'LOGIC MANAGING FILL OF EMPTY BUILD_FROM_MOD_DTYPES BASED ON return_format IS FAILING', fxn=fxn)
            elif not BUILD_FROM_MOD_DTYPES is None:
                BUILD_FROM_MOD_DTYPES = np.array(BUILD_FROM_MOD_DTYPES, dtype='<U10').reshape((1,-1))[0]

                # CHECK IF RETURNING AS SD AND IF ALLOWED TYPES IN MOD DTYPES
                if self.return_format=='SPARSE_DICT':
                    BUILD_FROM_MOD_DTYPES = self.sd_strtype_handle(BUILD_FROM_MOD_DTYPES, 'BUILD_FROM_MOD_DTYPES', 'arb')
                elif self.return_format=='ARRAY' and not self.bypass_validation:
                    # IF RETURNING AS ARRAY, CHECK IF MOD_DTYPE ENTRIES ARE VALID (IF VALIDATING)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        for _dtype in BUILD_FROM_MOD_DTYPES:
                            if _dtype not in list(msod.mod_num_dtypes().values()) + list(msod.mod_text_dtypes().values()):
                                self._exception(f'MOD DTYPE "{_dtype}" IN BUILD_FROM_MOD_DTYPES IS NOT VALID', fxn=fxn)

            # RANDOMLY SELECT FROM BUILD_FROM_MOD_DTYPES DIRECTLY INTO self.SUPPORT_OBJECTS[self.mdtype_idx]
            self.SUPPORT_OBJECTS[self.mdtype_idx] = nnrc.new_np_random_choice(BUILD_FROM_MOD_DTYPES,
                                                                  self.columns, replace=True).reshape((1, -1))[0]
            # IF GET HERE, MODIFIED_DATATYPES ARE GOOD TO GO

        del BUILD_FROM_MOD_DTYPES, FULL_SUPOBJ_OR_SINGLE_MDTYPES
        ### END VALIDATE CONTENT OF MOD DTYPE OBJECTS ##############################################################################

        # DECLARED self.this_module, fxn
        # super() VALIDATED HEADER, FULL_SUP_OBJ_OR_SINGLE_MDTYPES, self.bypass_validation, self.name, self.return_format,
        #       self.return_orientation, self.override_sup_obj, self.calling_module, self.calling_fxn, self.mdtype_idx
        #       BUILT self.LOOKUP_DICT
        # DECLARED/VALIDATED/OVERWROTE default_values
        # DECLARED/VALIDATED self.OBJECT, self.given_format, self.given_orientation
        # VALIDATED rows, GOT / VALIDATED columns, BUILT FULL SUPOBJ AND PUT IN MOD_DTYPES IF AVAILABLE
        # VALIDATED MOD_DTYPES IN FULL_SUP_OBJ_OR_SINGLE_MDTYPES OR BUILD_FROM_MOD_DTYPES, FILLED self.SUPPORT_OBJECTS[mdtype_idx]
        #  W RANDOM FROM BUILD_FROM IF NOT ALREADY FILLED FROM FULL_SUP_OBJ_OR.  self.SUPPORT_OBJECTS IS READY TO PASS TO bfso



        ### FILL IF EMPTY / VALIDATE IF FULL BUILD INSTRUCTOR OBJECTS #######################################################
        _DEFLT_FILLER = lambda _default_value, _dtype: np.full(self.columns, _default_value, dtype=_dtype).reshape((1,-1))[0]

        # LIST-TYPES
        if self.bypass_validation:
            # IF BUILD INSTRUCTORS WERE GIVEN AS NUMBERS, defaults WERE SET AND INSTRUCTORS SET TO None, ELSE COULD ONLY RIGHTFULLY BE LIST-TYPE
            self.NUMBER_OF_CATEGORIES = NUMBER_OF_CATEGORIES if not NUMBER_OF_CATEGORIES is None else _DEFLT_FILLER(default_number_of_categories, np.int32)
            self.MIN_VALUES = MIN_VALUES if not MIN_VALUES is None else _DEFLT_FILLER(default_min_value, np.float64)
            self.MAX_VALUES = MAX_VALUES if not MAX_VALUES is None else _DEFLT_FILLER(default_max_value, np.float64)
            self.SPARSITIES = SPARSITIES if not SPARSITIES is None else _DEFLT_FILLER(default_sparsity, np.float64)
            self.WORD_COUNT = WORD_COUNT if not WORD_COUNT is None else _DEFLT_FILLER(default_word_count, np.int32)
            self.POOL_SIZE = POOL_SIZE if not POOL_SIZE is None else _DEFLT_FILLER(default_pool_size, np.int32)

        elif not self.bypass_validation:
            # CHECK BUILD INSTRUCTOR OBJECTS FOR DTYPE, SIZE, AND CONTENT.  IF PASSED AS LISTS, CHECK SIZE CONGRUITY AND CONTENT
            # IF PASSED AS NUMBERS, default VALUES AT TOP OF init WERE SET, AND ALL THE FACILITATORS WERE SET TO None,
            # SO ALL FACILITATORS IF CORRECTLY ENTERED SHOULD BE LIST-TYPES OR None

            def validate_build_facilitator(_OBJECT, _name, _default_value, _dtypes):
                if _OBJECT is None:
                    _OBJECT = _DEFLT_FILLER(_default_value, type(_default_value))
                elif not _OBJECT is None:
                    if not isinstance(_OBJECT, (np.ndarray, list, tuple)):
                        self._exception(f'{_name} MUST BE A LIST-TYPE', fxn=fxn)
                    _OBJECT = np.array(_OBJECT).reshape((1,-1))[0]
                    if len(_OBJECT)!=self.columns:
                        self._exception(f'{_name} IS NOT CORRECT SIZE. len MUST BE {self.columns}.', fxn=fxn)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        TYPE_DICT = dict(((int,'INT'),(float,'FLOAT')))
                        for given_dtype in (str(_).upper() for _ in list(map(type, _OBJECT))):
                            if not True in (__ in given_dtype for __ in (TYPE_DICT[___] for ___ in tuple(_dtypes))):
                                self._exception(f'{_name} CONTAINS INVALID DTYPE "{given_dtype}". MUST BE {_dtypes}.', fxn=fxn)
                        del TYPE_DICT
                return _OBJECT

            # TEST FOR CONGRUENCE OF BUILD INSTRUCTION OBJECTS
            self.NUMBER_OF_CATEGORIES = validate_build_facilitator(NUMBER_OF_CATEGORIES, 'NUMBER_OF_CATEGORIES', default_number_of_categories, (int,))
            self.MIN_VALUES = validate_build_facilitator(MIN_VALUES, 'MIN_VALUES', default_min_value, (float,int))
            self.MAX_VALUES = validate_build_facilitator(MAX_VALUES, 'MAX_VALUES', default_max_value, (float,int))
            self.SPARSITIES = validate_build_facilitator(SPARSITIES, 'SPARSITIES', default_sparsity, (float,int))
            self.WORD_COUNT = validate_build_facilitator(WORD_COUNT, 'WORD_COUNT', default_word_count, (int,))
            self.POOL_SIZE = validate_build_facilitator(POOL_SIZE, 'POOL_SIZE', default_pool_size, (int,))

            if min(self.NUMBER_OF_CATEGORIES)<1:
                self._exception(f'NUMBER_OF_CATEGORIES MUST BE >= 1', fxn=fxn)
            for _min,_max in zip(self.MIN_VALUES, self.MAX_VALUES):
                if _min > _max: self._exception(f'RESPECTIVE MIN_VALUES MUST BE <= MAX_VALUES', fxn=fxn)
            if min(self.SPARSITIES)<0 or max(self.SPARSITIES)>100:
                self._exception(f'SPARSITIES MUST BE >= 0 and <=100',fxn=fxn)
            if min(self.WORD_COUNT)<1 or min(self.POOL_SIZE)<1:
                self._exception(f'WORD_COUNT AND POOL_SIZE MUST BE INTEGER > 0', fxn=fxn)

            del validate_build_facilitator

        del _DEFLT_FILLER

        ### END FILL IF EMPTY / VALIDATE IF FULL BUILD INSTRUCTOR OBJECTS #######################################################


        # END PROCESS OBJECT BUILD PARAMETERS ###########################################################################
        #################################################################################################################
        #################################################################################################################


        #################################################################################################################
        # PROCESS SUP_OBJ & BUILD REMAINING SUPPORT OBJECTS #############################################################

        # BUILD REMAINING SUPPORT OBJECTS USING SupportObjectHandle VIA bfso IN build_full_supobj.
        self.build_full_supobj()
        self.get_individual_support_objects()
        self.CONTEXT = []

        # REMEMBER THAT IF HEADER WAS None, ModifiedDatatypes MADE A DEFAULT HEADER, SO OVERWRITE THAT & KEEP HERE
        #################################################################################################################
        # BUILD HEADER IF NOT GIVEN #####################################################################################
        if OBJECT_HEADER is None:   # NOT self.OBJECT_HEADER !!!!
            for idx in range(columns):
                self.OBJECT_HEADER[0][idx] = f'{name[:3]}_{self.SUPPORT_OBJECTS[self.mdtype_idx][idx][:3]}{str(idx + 1)}'
            self.SUPPORT_OBJECTS[self.LOOKUP_DICT['HEADER']] = self.OBJECT_HEADER[0].copy()
            self.KEEP = self.OBJECT_HEADER[0].copy()
        # elif not OBJECT_HEADER is None.... THEN MUST HAVE BEEN INCORPORATED IN SUPOBJ BY bfso/ModifiedDatatypes
        # END BUILD HEADER IF NOT GIVEN  ################################################################################
        #################################################################################################################

        # END PROCESS SUP_OBJ & BUILD REMAINING SUPPORT OBJECTS #########################################################
        #################################################################################################################


        self.is_expanded = True not in (_ in list(msod.mod_text_dtypes().values()) for _ in self.MODIFIED_DATATYPES)

        self.is_list, self.is_dict = False, False


        # BUILD OBJECT ##################################################################################################
        self.build()
        # END BUILD OBJECT ##############################################################################################

        # HERE JUST IN CASE EVER WANT TO GO TO OBJECT AS MLObject INSTANCE
        # ObjectClass = mlo.MLObject(self.OBJECT, given_orientation, return_orientation='AS_GIVEN', return_format='AS_GIVEN',
        #                            self.bypass_validation=True, calling_module=None, calling_fxn=None)



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

    # OVERWRITES
    # build

    # UNIQUE
    # sd_strtype_handle




    def build(self):
        #########################################################################################################################
        # BUILD ##################################################################################################################
        '''3/9/23 The net effect of build() is to create a new object from scratch based on the mod dtypes, size, and other
        build instructions that are given.'''

        fxn = inspect.stack()[0][3]


        # OBJECTS THAT SHOULD BE (MUST BE) AVAILABLE FROM init:
        # self.FULL SUPOBJ
        # self.NUMBER_OF_CATEGORIES
        # self.MIN_VALUES
        # self.MAX_VALUES
        # self.SPARSITY
        # self.WORD_COUNT (NUMBER OF WORDS PER LINE IN EACH SPLIT_STR OR NNLM50 COLUMN)
        # self.POOL_SIZE (NUMBER OF WORDS OUT OF LEXICON TO DRAW FROM WHEN CONSTRUCTING SPLIT_STR & NNLM50 COLUMNS)


        def column_returner(col_idx):

            mdtype = self.SUPPORT_OBJECTS[self.LOOKUP_DICT['MODIFIEDDATATYPES']][col_idx]

            # 3/14/23 RETURN ORIENTATION MUST ALWAYS BE 'COLUMN' BECAUSE OBJECT IS COMPILED AS COLUMN THEN TURNED TO return_orientation
            if mdtype in ['FLOAT','INT','BIN']:
                _min = self.MIN_VALUES[col_idx]
                _max = self.MAX_VALUES[col_idx]
                _sparsity = self.SPARSITIES[col_idx]
                kwargs = {'return_orientation': 'COLUMN', 'return_as_sparse_dict': self.return_format == 'SPARSE_DICT'}
                if mdtype=='FLOAT': __ = crco.CreateFloatColumn(self.rows, _min, _max, _sparsity, **kwargs)
                elif mdtype=='INT': __ = crco.CreateIntColumn(self.rows, _min, _max, _sparsity, **kwargs)
                elif mdtype=='BIN': __ = crco.CreateBinColumn(self.rows, _sparsity, **kwargs)
                del _min,_max,_sparsity, kwargs

            elif mdtype in ['STR','SPLIT_STR','NNLM50']:    # CANNOT BE RETURNED AS SD
                _name = self.SUPPORT_OBJECTS[self.LOOKUP_DICT['HEADER']][col_idx]
                _categories = self.NUMBER_OF_CATEGORIES[col_idx]
                _word_count = self.WORD_COUNT[col_idx]
                _pool_size = self.POOL_SIZE[col_idx]
                kwargs = {'return_orientation':'COLUMN'}

                if mdtype=='STR': __ = crco.CreateStrColumn(_name, self.rows, _categories, **kwargs)
                elif mdtype=='SPLIT_STR': __ = crco.CreateSplitStrColumn(_name, self.rows, _pool_size, _word_count, **kwargs)
                elif mdtype=='NNLM50': __ = crco.CreateNNLM50Column(_name, self.rows, _pool_size, _word_count, **kwargs)

                del _name, _categories, _word_count, _pool_size, kwargs

            return __.COLUMN


        # FOR EASE, CONSTRUCT ARRAY OR SD AS COLUMN, THEN TRANSPOSE IF NEEDED
        if self.return_format=='ARRAY':
            self.OBJECT = np.empty((0, self.rows), dtype=np.int8)

            # MUST DO ALL THIS TO PREVENT <U DTYPES GOING INTO VSTACK FROM FORCING ENTIRE OBJECT (INCLUDING NUMBERS) TO A <U DTYPE

            all_nums, all_strs, num_str_mix = False, False, False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # IF NUMS ONLY, ALLOW vstack TO HANDLE FINAL DTYPE OF OBJECT
                # IF STRS ONLY, ALLOW vstack TO HANDLE FINAL <U OF OBJECT
                # BUT IF MIX OF NUMS AND STRS, FORCE VSTACK TO RETURN DTYPE "object"  (SEE NOTES IN CreateColumn TEST AREA)
                has_num_types = True in (_ in msod.mod_num_dtypes().values() for _ in self.SUPPORT_OBJECTS[self.mdtype_idx])
                has_str_types = True in (_ in msod.mod_text_dtypes().values() for _ in self.SUPPORT_OBJECTS[self.mdtype_idx])
                if has_num_types and has_str_types: num_str_mix = True
                elif has_num_types and not has_str_types: all_nums = True
                elif not has_num_types and has_str_types: all_strs = True
                else: self._exception(f'LOGIC DETERMINING IF MODIFIED_DATATYPES CONTAINS NUMS, STRS, OR MIX IS FAILING', fxn=self.calling_fxn)

                del has_num_types, has_str_types

            for col_idx in range(self.columns):
                if all_nums or all_strs: self.OBJECT = np.vstack((self.OBJECT, column_returner(col_idx)))
                elif num_str_mix: self.OBJECT = np.vstack((self.OBJECT.astype(object), column_returner(col_idx).astype(object)))

            del all_nums, all_strs, num_str_mix

        elif self.return_format=='SPARSE_DICT':
            self.OBJECT = {}
            for col_idx in range(self.columns):
                self.OBJECT[int(col_idx)] = column_returner(col_idx)[0]

        self.current_format, self.current_orientation = self.return_format, 'COLUMN'
        self.is_list, self.is_dict = self.return_format=='ARRAY', self.return_format=='SPARSE_DICT'

        if self.return_orientation=='ROW':
            self.to_row()

        # END BUILD #############################################################################################################
        #########################################################################################################################


    def sd_strtype_handle(self, OBJECT_BEING_CHECKED, name_as_str, allowed_options):
        fxn = inspect.stack()[0][3]

        FULL_MENU_DICT = dict((('a','abort'),('r','return as ARRAY'),('m','Override MODIFIED_DATATYPES'),
                               ('b','Override BUILD_FROM_MOD_DTYPES')))

        # FULL_SUP_OBJ_OR_SINGLE_MDTYPES
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in allowed_options:
                if not _ in FULL_MENU_DICT.keys():
                    self._exception(f'ILLEGAL OPTION "{_}" IN allowed_options. MUST BE IN {", ".join(list(FULL_MENU_DICT.keys()))}', fxn=fxn)

            if True in (_ in (msod.mod_text_dtypes().values()) for _ in OBJECT_BEING_CHECKED):
                print(f'\n*** {name_as_str} HAS STR-TYPE COLUMNS, CANNOT RETURN AS SPARSE_DICT ***\n')
                _ = vui.validate_user_str(", ".join([f'{v}({k})' for k,v in FULL_MENU_DICT.items() if k in allowed_options]),
                                          allowed_options)
                del FULL_MENU_DICT
                if _ == 'A': self._exception(f'USER TERMINATED', fxn=fxn)
                elif _ == 'R': self.return_format = 'ARRAY'
                elif _ == 'M': self.override_sup_obj = True
                elif _ == 'B':
                    while True:
                        OBJECT_BEING_CHECKED = ls.list_multi_select(list(msod.mod_num_dtypes().values()),
                                                f'Select new numerical mod dtypes to build from', 'value')[0]
                        if vui.validate_user_str(f'USER CHOSE NEW MOD DTYPES AS {OBJECT_BEING_CHECKED}, ACCEPT? (y/n) > ', 'YN') == 'Y': break

            return OBJECT_BEING_CHECKED





















































































if __name__ == '__main__':

    # EVERYTHING BELOW IS A TEST MODULE W SUPPORTING FUNCTIONS

    # 3/18/23 BEAR VERIFIED TEST CODE & MODULE ARE GOOD

    '''
    # COMMENT THIS OUT TO RUN ITERATIVE TESTING ########################################################################
    import pandas as pd
    return_format = 'ARRAY'
    return_orientation = 'COLUMN'
    rows = 10
    OBJECT_HEADER = ['BEAR','BURGER','IS','OPEN']
    FULL_SUPOBJ_OR_SINGLE_MDTYPES = ['BIN','INT','INT','INT'], #['STR','BIN','INT','FLOAT']
    BUILD_FROM_MOD_DTYPES = ['NNLM50', 'INT']
    override_sup_obj = False
    bypass_validation = False

    TestClass = CreateFromScratch(return_format,
                                    return_orientation,
                                    rows,
                                    name='TEST',
                                    OBJECT_HEADER=OBJECT_HEADER,
                                    FULL_SUPOBJ_OR_SINGLE_MDTYPES=None, #FULL_SUPOBJ_OR_SINGLE_MDTYPES,
                                    BUILD_FROM_MOD_DTYPES='STR', #BUILD_FROM_MOD_DTYPES, #None,
                                    columns=4,
                                    NUMBER_OF_CATEGORIES=5,
                                    MIN_VALUES=0,
                                    MAX_VALUES=10,
                                    SPARSITIES=50,
                                    WORD_COUNT=4,
                                    POOL_SIZE=10,
                                    override_sup_obj=override_sup_obj,
                                    bypass_validation=bypass_validation
                                    )

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
    # END COMMENT THIS OUT TO RUN ITERATIVE TESTING ####################################################################
    '''


    from general_sound import winlinsound as wls
    from ML_PACKAGE._data_validation import ValidateObjectType as vot


    def test_exc_handle(OBJECT, reason_text):
        time.sleep(1)
        print(f'\n\033[91mEXCEPTING OBJECT:\033[0m\x1B[0m')
        print(OBJECT)
        print()
        for _ in range(3): wls.winlinsound(888, 500); time.sleep(1)
        print(f'\n\033[91mWANTS TO RAISE EXCEPTION FOR \033[0m\x1B[0m')
        print(reason_text)
        _quit = vui.validate_user_str(f'\n\033[91mquit(q) or continue(c) > \033[0m\x1B[0m', 'QC')
        if _quit == 'Q': raise Exception(f'\033[91m*** ' + reason_text + f' ***\033[0m\x1B[0m')
        elif _quit == 'C': pass


    def test_lens(OBJECT, return_orientation, rows, columns):
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


    def test_min_and_max(OBJECT, EXP_VAL_DTYPES, MIN_VALUES, MAX_VALUES):
        # WAS SET TO [] = COLUMNS FOR TEST PURPOSES

        for idx, val_dtype in enumerate(EXP_VAL_DTYPES):
            # GET EXPECTED MIN & MAX ##########################################################################################
            if val_dtype == 'STR': continue
            elif val_dtype == 'BIN': exp_min, exp_max = 0, 1
            elif val_dtype == 'INT': exp_min, exp_max = MIN_VALUES[idx], MAX_VALUES[idx]
            elif val_dtype == 'FLOAT': exp_min, exp_max = MIN_VALUES[idx], MAX_VALUES[idx]
            else: raise Exception(f'\n\033[91m*** LOGIC FOR ASSIGNING MIN/MAX TO VAL_DTYPES IN test_min_and_max() IS FAILING ***\033[0m\x1B[0m\n')
            if _sparsity == 100: exp_min, exp_max = 0, 0
            elif _sparsity == 0: pass
            else: exp_min, exp_max = min(0, min_value), max(0, max_value)   # BECAUSE SPARSITY WILL ALWASY INTRODUCT ZEROS
            # END GET EXPECTED MIN & MAX ##########################################################################################

            # GET ACTUAL MIN & MAX ##########################################################################################
            if isinstance(OBJECT, dict):
                act_min, act_max = sd.min_({0:OBJECT[idx]}), sd.max_({0:OBJECT[idx]})
            elif isinstance(OBJECT, (np.ndarray)):
                act_min, act_max = np.min(OBJECT[idx]), np.max(OBJECT[idx])
            # END GET ACTUAL MIN & MAX ##########################################################################################

            if val_dtype == 'BIN' and int(act_min) != 0 and int(act_max) != 1 and \
                    exp_min != act_min and exp_max != act_max:
                test_exc_handle(OBJECT, f'\n\033[91m*** min max: EXPECTED BIN OUTPUT W min={min_value}, max={max_value}, GOT min={act_min}, max=({act_max}\033[0m\x1B[0m')
            elif val_dtype == 'INT' and (act_min < exp_min or act_max > exp_max):
                test_exc_handle(OBJECT,
                    f'\n\033[91m*** min max: EXPECTED INT OUTPUT W min={min_value}, max={max_value}, GOT min={act_min}, max={act_max}\033[0m\x1B[0m')
            elif val_dtype == 'FLOAT' and (act_min < exp_min or act_max > exp_max):
                test_exc_handle(OBJECT,
                    f'\n\033[91m*** min max: EXPECTED FLOAT OUTPUT W min={min_value}, max={max_value}, GOT min={act_min}, max={act_max}\033[0m\x1B[0m')


    def test_val_dtypes(OBJECT, EXP_VALIDATED_DATATYPES, return_orientation, obj_desc):
        # WAS SET TO [] = COLUMNS FOR TEST PURPOSES

        # GET TYPE FROM OBJECT #####################################################################################
        ACT_VALIDATED_DATATYPES = np.empty(len(EXP_VALIDATED_DATATYPES), dtype='<U10').reshape((1,-1))[0]

        def vdtype_getter(COLUMN):
            return vot.ValidateObjectType(COLUMN).ml_package_object_type(suppress_print='Y')

        if isinstance(OBJECT, dict):
            for idx, outer_idx in enumerate(OBJECT):
                ACT_VALIDATED_DATATYPES[idx] = vdtype_getter(np.fromiter(OBJECT[outer_idx].values(), dtype=object))

        elif isinstance(OBJECT, np.ndarray):
            # WAS CONVERTED TO []=COLUMN (IF ROW), SO ZIP THRU OBJECT GETTING VAL DTYPES WITH vot.ml_package_object_type
            for idx, COLUMN in enumerate(OBJECT):
                ACT_VALIDATED_DATATYPES[idx] = vdtype_getter(COLUMN)
            # else: test_exc_handle(OBJECT, f'\n\033[91m*** UNKNOWN NDARRAY DTYPE {_} ***\033[0m\x1B[0m\n')
        # END GET VAL DTYPES FROM OBJECT #####################################################################################

        if not np.array_equiv(EXP_VALIDATED_DATATYPES, ACT_VALIDATED_DATATYPES):
            test_exc_handle(OBJECT, f'\n\033[91m*** OUTPUT V_DTYPES ARE {ACT_VALIDATED_DATATYPES}, SHOULD BE {EXP_VALIDATED_DATATYPES} ***\033[0m\x1B[0m\n')

        del vdtype_getter

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
        char_str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        num_of_inner = _rows if given_orientation == 'ROW' else _columns if given_orientation == 'COLUMN' else quit()
        len_of_inner = _columns if given_orientation == 'ROW' else _rows if given_orientation == 'ROW' else quit()

        OBJECT = crsn.create_random_sparse_numpy(min_value,max_value,(num_of_inner,len_of_inner),_sparsity,np.int32)
        OBJECT_HEADER = np.array([f'{char_str[_//26]}{char_str[_%26]}' for _ in range(_columns)], dtype='<U2').reshape((1,-1))

        print(f'\nINPUT OBJECT:')
        print(OBJECT)
        print()

        DummyClass = cn.CreateNumerical(name=name, OBJECT=OBJECT, OBJECT_HEADER=OBJECT_HEADER,
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
        test_val_dtypes(RETURN_OBJECT, 'BIN' if _sparsity==100 else bin_int_or_float)
        test_min_and_max(RETURN_OBJECT, bin_int_or_float, min_value, max_value, _sparsity)
        test_lens(RETURN_OBJECT, return_orientation, _rows, _columns)
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

    char_str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    char_str = np.array([[_+__ for __ in char_str] for _ in char_str], dtype='<U5').reshape((1,-1))[0]

    MASTER_BYPASS_VALIDATION = [False, True]

    RETURN_FORMAT = ['ARRAY', 'SPARSE_DICT']
    RETURN_ORIENTATION = ['ROW', 'COLUMN']
    COLUMNS = [100,10,1]
    ROWS = [10,100,1000]    # MUST HAVE ENOUGH ROWS TO AVOID A COLUMN OF SUPPOSED-TO-BE-INTS ACTUALLY BEING BIN
    MIN_VALUE = [2, -10, -10]
    MAX_VALUE = [10, 10, -1]

    SUPPORT_OBJECT_DESCS = [
                            'full_support_object',
                            'mdtypes_only',
                            None
                            ]

    DATA_OBJECT_HEADERS = [
                           'built_during_run',
                           None
                           ]

    _sparsity = 50

    total_itrs = np.product(list(map(len, (MASTER_BYPASS_VALIDATION, RETURN_FORMAT, RETURN_ORIENTATION, COLUMNS,
                                           SUPPORT_OBJECT_DESCS, MIN_VALUE, DATA_OBJECT_HEADERS)
                                          )))


    ctr = 0
    for bypass_validation in MASTER_BYPASS_VALIDATION:
        for return_format in RETURN_FORMAT:
            for return_orientation in RETURN_ORIENTATION:
                for _columns, _rows in zip(COLUMNS, ROWS):
                    for support_object_desc in SUPPORT_OBJECT_DESCS:
                        for min_value, max_value in zip(MIN_VALUE, MAX_VALUE):
                            for data_object_hdr_desc in DATA_OBJECT_HEADERS:
                                ctr += 1

                                if ctr % 1 == 0:
                                    print(f'\n\nRunning test {ctr} of {total_itrs}...')

                                # SUPPORT OBJECT CAN BE None, FULL_SUPOBJ W JUST MOD_DTYPES FILLED, OR SINGLE MOD_DTYPES
                                if support_object_desc is None:
                                    GIVEN_SUPPORT_OBJECT = None
                                    if return_format=='ARRAY': BUILD_FROM_MOD_DTYPES = ['STR', 'FLOAT', 'INT', 'BIN']
                                    elif return_format=='SPARSE_DICT': BUILD_FROM_MOD_DTYPES = ['FLOAT','INT','BIN']
                                    else: self._exception(f'LOGIC MANAGING ASSINGMENT OF BUILD_FROM_MOD_DTYPES BASED ON return_format IS FAILING')
                                else:
                                    # list(msod.mod_text_dtypes().values()) + list(msod.mod_num_dtypes().values()),
                                    BUILD_FROM_MOD_DTYPES = None
                                    if return_format=='ARRAY': ALLOWED_MOD_DTYPES = ['STR','BIN','INT','FLOAT']
                                    if return_format=='SPARSE_DICT': ALLOWED_MOD_DTYPES = ['BIN', 'INT', 'FLOAT']
                                    EXP_MODIFIED_DATATYPES = nnrc.new_np_random_choice(ALLOWED_MOD_DTYPES, (1, _columns), replace=True).reshape((1, -1))[0]
                                    EXP_VALIDATED_DATATYPES = np.fromiter((msod.val_reverse_lookup()[_] for _ in EXP_MODIFIED_DATATYPES), dtype='<U10').reshape((1,-1))[0]
                                    del ALLOWED_MOD_DTYPES
                                    if support_object_desc == 'full_support_object':
                                        GIVEN_SUPPORT_OBJECT = msod.build_empty_support_object(_columns)
                                        GIVEN_SUPPORT_OBJECT[msod.master_support_object_dict()['MODIFIEDDATATYPES']['position']] = EXP_MODIFIED_DATATYPES
                                    elif support_object_desc == 'mdtypes_only':
                                        GIVEN_SUPPORT_OBJECT = EXP_MODIFIED_DATATYPES


                                if data_object_hdr_desc is None:
                                    GIVEN_DATA_OBJECT_HEADER = None
                                    # BUILD EXP_HEADER AFTER EXP_MODIFIED_DATATYPES IS FINALIZED BELOW.  SUPPORT_OBJECTS COULD BE None
                                    # WHICH WOULD CAUSE CreateFromScratch TO BUILD RANDOM MOD_DTYPES, WHICH WOULD ALTER FINAL HEADER
                                    # IF A HEADER WAS NOT GIVEN
                                else:
                                    GIVEN_DATA_OBJECT_HEADER = char_str[:_columns]
                                    EXP_HEADER = GIVEN_DATA_OBJECT_HEADER.copy()

                                DummyObject = CreateFromScratch(
                                                                 return_format,
                                                                 return_orientation,
                                                                 _rows,
                                                                 name=name,
                                                                 OBJECT_HEADER=GIVEN_DATA_OBJECT_HEADER,
                                                                 FULL_SUPOBJ_OR_SINGLE_MDTYPES=GIVEN_SUPPORT_OBJECT,
                                                                 # BUILD_FROM_MOD_DTYPES ONLY MATTERS IF SUPOBJ IS None, DONT DO GIGANTIC SPLIT_STR AND NNLM50
                                                                 BUILD_FROM_MOD_DTYPES=BUILD_FROM_MOD_DTYPES,
                                                                 columns=_columns,
                                                                 NUMBER_OF_CATEGORIES=5,
                                                                 MIN_VALUES=min_value,
                                                                 MAX_VALUES=max_value,
                                                                 SPARSITIES=_sparsity,
                                                                 WORD_COUNT=None,
                                                                 POOL_SIZE=None,
                                                                 override_sup_obj=False,
                                                                 bypass_validation=bypass_validation
                                                                 )

                                # DATA_OBJECT - DONT BUILD EXP FOR OBJECT, IS NOT GIVEN, DELETE ALL BUILD STUFF

                                # print(obj_desc)

                                ACT_OUTPUT_OBJECT = DummyObject.OBJECT
                                ACT_HEADER = DummyObject.OBJECT_HEADER
                                ACT_VALIDATED_DATATYPES = DummyObject.VALIDATED_DATATYPES
                                ACT_MODIFIED_DATATYPES = DummyObject.MODIFIED_DATATYPES
                                ACT_FILTERING = DummyObject.FILTERING
                                ACT_MIN_CUTOFFS = DummyObject.MIN_CUTOFFS
                                ACT_USE_OTHER = DummyObject.USE_OTHER
                                ACT_START_LAG = DummyObject.START_LAG
                                ACT_END_LAG = DummyObject.END_LAG
                                ACT_SCALING = DummyObject.SCALING
                                ACT_CONTEXT = DummyObject.CONTEXT
                                ACT_KEEP = DummyObject.KEEP
                                ACT_MIN_VALUES = DummyObject.MIN_VALUES
                                ACT_MAX_VALUES = DummyObject.MAX_VALUES

                                # IF GIVEN MOD_DTYPES WAS None, MOD_DTYPES WERE GENERATED RANDOMLY IN CreateFromScratch, SO HAVE TO GET
                                # THEM OUT IF IT TO KNOW WHAT THEY WERE TO COMPARE TO ACTUALS
                                if support_object_desc is None:
                                    EXP_MODIFIED_DATATYPES = DummyObject.SUPPORT_OBJECTS[msod.master_support_object_dict()['MODIFIEDDATATYPES']['position']]
                                    EXP_VALIDATED_DATATYPES = DummyObject.SUPPORT_OBJECTS[msod.master_support_object_dict()['VALIDATEDDATATYPES']['position']]
                                # else:  EXP_VALIDATED_DATATYPES & EXP_MODIFIED_DATATYPESWAS SET ABOVE WHEN BUILDING GIVENS

                                # IF HEADER WAS None, WAIT UNTIL EXP_MODIFIED_DATATYPES IS FINALIZED
                                if data_object_hdr_desc is None:
                                    EXP_HEADER = np.fromiter(
                                        (f'{name[:3]}_{EXP_MODIFIED_DATATYPES[idx][:3]}{str(idx + 1)}' for idx in
                                         range(_columns)), dtype='<U15')

                                obj_desc = f"\nDATA OBJECT IS NOT GIVEN, EXPECTED MODIFIED DATATYPES ARE {EXP_MODIFIED_DATATYPES} " \
                                           f"AND HEADER IS \n{data_object_hdr_desc}." + \
                                           f"\nOBJECT SHOULD BE RETURNED AS A {return_format} " \
                                           f"(min={min_value} max={max_value}) WITH {_rows} ROWS AND {_columns} COLUMNS ORIENTED AS {return_orientation}."


                                # TESTING ONLY FOR PASSING A CONSTANT TO MIN_VALUES AND MAX_VALUES
                                if np.sum((ACT_MIN_VALUES/min_value)) != len(ACT_MIN_VALUES):
                                    test_exc_handle(ACT_MIN_VALUES, f'ALL MIN_VALUES MUST EQUAL {min_value}')
                                if np.sum((ACT_MAX_VALUES/max_value)) != len(ACT_MAX_VALUES):
                                    test_exc_handle(ACT_MAX_VALUES, f'ALL MAX_VALUES MUST EQUAL {max_value}')

                                test_lens(ACT_OUTPUT_OBJECT, return_orientation, _rows, _columns)

                                # SET TO []=COLUMNS FOR REMAINING TESTS FOR SPEED, BUT LEAVE AN ORIGINAL FOR DISPLAY IF EXCEPTS
                                if return_orientation=='ROW':
                                    if return_format=='ARRAY': TRANSPOSED_ACT_OUTPUT_OBJECT = ACT_OUTPUT_OBJECT.transpose().astype(object)
                                    elif return_format=='SPARSE_DICT': TRANSPOSED_ACT_OUTPUT_OBJECT = sd.core_sparse_transpose(deepcopy(ACT_OUTPUT_OBJECT))
                                else:
                                    if return_format=='ARRAY': TRANSPOSED_ACT_OUTPUT_OBJECT = ACT_OUTPUT_OBJECT.copy()
                                    elif return_format=='SPARSE_DICT': TRANSPOSED_ACT_OUTPUT_OBJECT = deepcopy(ACT_OUTPUT_OBJECT)

                                test_format(TRANSPOSED_ACT_OUTPUT_OBJECT, return_format)
                                test_val_dtypes(TRANSPOSED_ACT_OUTPUT_OBJECT, EXP_VALIDATED_DATATYPES, return_orientation, obj_desc)
                                test_min_and_max(TRANSPOSED_ACT_OUTPUT_OBJECT, EXP_VALIDATED_DATATYPES, ACT_MIN_VALUES, ACT_MAX_VALUES)

                                # ****** TEST SUPPORT OBJECTS ********************************************************************
                                SUPP_OBJ_NAMES = ['HEADER', 'VALIDATED_DATATYPES', 'MODIFIED_DATATYPES', 'FILTERING', 'MIN_CUTOFFS',
                                                 'USE_OTHER', 'START_LAG', 'END_LAG', 'SCALING', 'CONTEXT', 'KEEP']

                                EXP_FILTERING = np.fromiter(([] for _ in range(_columns)), dtype=object).reshape((1, -1))
                                EXP_MIN_CUTOFFS = np.fromiter((0 for _ in range(_columns)), dtype=np.int16).reshape((1, -1))
                                EXP_USE_OTHER = np.fromiter(('N' for _ in range(_columns)), dtype='<U1').reshape((1, -1))
                                EXP_START_LAG = np.fromiter((0 for _ in range(_columns)), dtype=np.int16)
                                EXP_END_LAG = np.fromiter((0 for _ in range(_columns)), dtype=np.int16)
                                EXP_SCALING = np.fromiter(('' for _ in range(_columns)), dtype='<U200')
                                EXP_CONTEXT = np.array([])
                                if data_object_hdr_desc is None:
                                    EXP_KEEP = np.fromiter((f'{name[:3]}_{EXP_MODIFIED_DATATYPES[idx][:3]}{str(idx + 1)}' for idx in range(_columns)),
                                                                dtype='<U100').reshape((1, -1))[0]
                                else:
                                    EXP_KEEP = EXP_HEADER.copy()

                                EXP_SUPP_OBJS = [EXP_HEADER, EXP_VALIDATED_DATATYPES, EXP_MODIFIED_DATATYPES, EXP_FILTERING, EXP_MIN_CUTOFFS,
                                                 EXP_USE_OTHER, EXP_START_LAG, EXP_END_LAG, EXP_SCALING, EXP_CONTEXT, EXP_KEEP]

                                ACT_SUPP_OBJS = [ACT_HEADER, ACT_VALIDATED_DATATYPES, ACT_MODIFIED_DATATYPES, ACT_FILTERING,
                                                 ACT_MIN_CUTOFFS, ACT_USE_OTHER, ACT_START_LAG, ACT_END_LAG, ACT_SCALING, ACT_CONTEXT,
                                                 ACT_KEEP]

                                for supp_obj_name, EXP_SUPP_OBJ, ACT_SUPP_OBJ in zip(SUPP_OBJ_NAMES, EXP_SUPP_OBJS, ACT_SUPP_OBJS):
                                    if not np.array_equiv(EXP_SUPP_OBJ, ACT_SUPP_OBJ):
                                        print(f'\n\n\033[91mFailed on trial {ctr} of {total_itrs}.\033[0m\x1B[0m')
                                        print(obj_desc)
                                        print(f'ACTUAL DATA = ')
                                        print(TRANSPOSED_ACT_OUTPUT_OBJECT)
                                        print()
                                        print(f'EXPECTED {supp_obj_name} = ')
                                        print(EXP_SUPP_OBJ)
                                        print()
                                        print(f'ACTUAL {supp_obj_name} = ')
                                        print(ACT_SUPP_OBJ)
                                        test_exc_handle(ACT_SUPP_OBJ, f'\n\033[91mACTUAL {supp_obj_name} DOES NOT EQUAL EXPECTED\033[0m\x1B[0m')

                                # ****** END TEST SUPPORT OBJECTS ********************************************************************

    print(f'\n\033[92m*** VALIDATION COMPLETED SUCCESSFULLY ***\033[0m\x1B[0m\n')
    for _ in range(3): wls.winlinsound(888, 500); time.sleep(0.5)

    # END ITERATIVE TESTING ###################################################################################################################
    # ############################################################################################################################
    # #############################################################################################################################




























