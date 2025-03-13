import sys
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from MLObjects.TestObjectCreators import ApexCreate as ac, CreateFromGiven as cfg, CreateFromScratch as cfs







# PARENT OF ApexCreateTarget, CreateData, CreateRefVecs
class ApexCreateTestObjects(ac.ApexCreate):    # ApexCreate TO GET METHODS (esp. expand), DONT super()!
    def __init__(self,
                name,
                return_format,
                return_orientation,
                OBJECT=None,
                OBJECT_HEADER=None,
                FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                override_sup_obj=None,
                bypass_validation=None,
                calling_module=None,
                calling_fxn=None,
                # CREATE FROM GIVEN ONLY ###############################################
                given_orientation=None,
                # END CREATE FROM GIVEN ONLY #############################################
                # CREATE FROM SCRATCH_ONLY ################################
                rows = None,
                columns = None,
                BUILD_FROM_MOD_DTYPES = None,
                NUMBER_OF_CATEGORIES = None,
                MIN_VALUES = None,
                MAX_VALUES = None,
                SPARSITIES = None,
                WORD_COUNT = None,
                POOL_SIZE = None
                # END CREATE FROM SCRATCH_ONLY #############################
                ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        self.calling_module = self.this_module if calling_module is None else calling_module
        self.calling_fxn = '__init__' if calling_fxn is None else calling_fxn

        self.bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                         self.calling_module, self.calling_fxn, return_if_none=False)

        if isinstance(name, str): self.name = name
        else: self._exception(f'name MUST BE PASSED AS A str', fxn=self.calling_fxn)

        # MUST DO THIS TO GET self.given_format OF OBJECT
        self.given_format, self.OBJECT = ldv.list_dict_validater(OBJECT, self.name)

        # WOULD ALLOW CreateFromGiven & CreateFromScratch TO HANDLE ALL VALIDATION, EXCEPT MANANGING given_orientation,
        # return_format AND return_orientation HERE BECAUSE OF THE DIFFERENCES BETWEEN cfg & cfs
        if self.bypass_validation:
            self.OBJECT = OBJECT
            self.given_orientation = given_orientation
            self.return_format = return_format
            self.return_orientation = return_orientation

        elif not self.bypass_validation:
            # IF OBJECT IS GIVEN, ALLOW CreateFromGiven VALIDATION OF return_format, return_orientation
            if not self.OBJECT is None:
                self.given_orientation = akv.arg_kwarg_validater(given_orientation, 'return_format',
                     ['ROW','COLUMN'], self.calling_module, self.calling_fxn)  # CANNOT BE None
                self.return_format = akv.arg_kwarg_validater(return_format, 'return_format',
                     ['ARRAY','SPARSE_DICT','AS_GIVEN'], self.calling_module, self.calling_fxn)  # CANNOT BE None
                self.return_orientation = akv.arg_kwarg_validater(return_orientation, 'return_orientation',
                     ['ROW','COLUMN', 'AS_GIVEN'], self.calling_module, self.calling_fxn)
            # IF OBJECT IS NOT GIVEN, ALLOW CreateFromScratch VALIDATION OF return_format, return_orientation
            elif self.OBJECT is None:
                self.given_orientation = akv.arg_kwarg_validater(given_orientation, 'given_orientation',
                     ['ROW', 'COLUMN', None], self.calling_module, self.calling_fxn)  # MUST BE None
                self.return_format = akv.arg_kwarg_validater(return_format, 'return_format',
                     ['ARRAY', 'SPARSE_DICT'], self.calling_module, self.calling_fxn)  # CANNOT BE None
                self.return_orientation = akv.arg_kwarg_validater(return_orientation, 'return_orientation',
                     ['ROW', 'COLUMN'], self.calling_module, self.calling_fxn)

        self.return_format = self.given_format if self.return_format=='AS_GIVEN' else self.return_format
        self.return_orientation = self.given_orientation if self.return_orientation == 'AS_GIVEN' else self.return_orientation


        # LIST CreateFromGiven OR CreateFromScratch HANDLE VALIDATION OF THESE REMAINING args/kwargs
        self.OBJECT_HEADER = OBJECT_HEADER
        self.FULL_SUPOBJ_OR_SINGLE_MDTYPES = FULL_SUPOBJ_OR_SINGLE_MDTYPES
        self.override_sup_obj = override_sup_obj
        self.rows = rows
        self.columns = columns
        self.BUILD_FROM_MOD_DTYPES = BUILD_FROM_MOD_DTYPES
        self.NUMBER_OF_CATEGORIES = NUMBER_OF_CATEGORIES
        self.MIN_VALUES = MIN_VALUES
        self.MAX_VALUES = MAX_VALUES
        self.SPARSITIES = SPARSITIES
        self.WORD_COUNT = WORD_COUNT
        self.POOL_SIZE = POOL_SIZE

        if self.OBJECT is None: self.create_from_scratch()
        elif not self.OBJECT is None: self.create_from_given()

        self.is_list, self.is_dict = self.return_format=='ARRAY', self.return_format=='SPARSE_DICT'
        self.current_format = self.return_format
        self.current_orientation = self.return_orientation

        self.CONTEXT = []

    # INHERITS
    # validate_full_supobj_or_single_mdtypes
    # build_full_supobj
    # get_individual_support_objects
    # build  (just a pass in Apex)
    # to_row
    # to_column
    # _transpose
    # to_array
    # to_sparse_dict
    # expand


    # OVERWRITES FROM ApexCreate
    def _exception(self, text, fxn=None):
        raise Exception(f'{self.calling_module}.{self.calling_fxn}() >>> {text}')


    # UNIQUE
    def create_from_given(self):
        GivenClass = cfg.CreateFromGiven(self.OBJECT,
                                         self.given_orientation,
                                         return_format=self.return_format,
                                         return_orientation=self.return_orientation,
                                         name=self.name,
                                         OBJECT_HEADER=self.OBJECT_HEADER,
                                         FULL_SUPOBJ_OR_SINGLE_MDTYPES=self.FULL_SUPOBJ_OR_SINGLE_MDTYPES,
                                         override_sup_obj=self.override_sup_obj,
                                         bypass_validation=self.bypass_validation
                                         )

        self.OBJECT = GivenClass.OBJECT
        self.SUPPORT_OBJECTS = GivenClass.SUPPORT_OBJECTS


    # UNIQUE
    def create_from_scratch(self):
        ScratchClass = cfs.CreateFromScratch(self.return_format,
                                             self.return_orientation,
                                             self.rows,
                                             name=self.name,
                                             OBJECT_HEADER=self.OBJECT_HEADER,
                                             FULL_SUPOBJ_OR_SINGLE_MDTYPES=self.FULL_SUPOBJ_OR_SINGLE_MDTYPES,
                                             BUILD_FROM_MOD_DTYPES=self.BUILD_FROM_MOD_DTYPES,
                                             columns=self.columns,
                                             NUMBER_OF_CATEGORIES=self.NUMBER_OF_CATEGORIES,
                                             MIN_VALUES=self.MIN_VALUES,
                                             MAX_VALUES=self.MAX_VALUES,
                                             SPARSITIES=self.SPARSITIES,
                                             WORD_COUNT=self.WORD_COUNT,
                                             POOL_SIZE=self.POOL_SIZE,
                                             override_sup_obj=self.override_sup_obj,
                                             bypass_validation=self.bypass_validation)

        self.OBJECT = ScratchClass.OBJECT
        self.SUPPORT_OBJECTS = ScratchClass.SUPPORT_OBJECTS



class CreateDataObject(ApexCreateTestObjects):
    def __init__(self,
                return_format,
                return_orientation,
                DATA_OBJECT=None,
                DATA_OBJECT_HEADER=None,
                FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                override_sup_obj=None,
                bypass_validation=None,
                # CREATE FROM GIVEN ONLY ###############################################
                given_orientation=None,
                # END CREATE FROM GIVEN ONLY #############################################
                # CREATE FROM SCRATCH_ONLY ################################
                rows = None,
                columns = None,
                BUILD_FROM_MOD_DTYPES = ['BIN','INT','FLOAT','STR'],
                NUMBER_OF_CATEGORIES = 10,
                MIN_VALUES = -10,
                MAX_VALUES = 10,
                SPARSITIES = 50,
                WORD_COUNT = 20,
                POOL_SIZE = 200
                # END CREATE FROM SCRATCH_ONLY #############################
                ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        self.calling_module = self.this_module
        self.calling_fxn = '__init__'

        super().__init__('DATA',
                        return_format,
                        return_orientation,
                        OBJECT=DATA_OBJECT,
                        OBJECT_HEADER=DATA_OBJECT_HEADER,
                        FULL_SUPOBJ_OR_SINGLE_MDTYPES=FULL_SUPOBJ_OR_SINGLE_MDTYPES,
                        override_sup_obj=override_sup_obj,
                        bypass_validation=bypass_validation,
                        calling_module=self.calling_module,
                        calling_fxn=self.calling_fxn,
                        given_orientation=given_orientation,
                        rows=rows,
                        columns=columns,
                        BUILD_FROM_MOD_DTYPES=BUILD_FROM_MOD_DTYPES,
                        NUMBER_OF_CATEGORIES=NUMBER_OF_CATEGORIES,
                        MIN_VALUES=MIN_VALUES,
                        MAX_VALUES=MAX_VALUES,
                        SPARSITIES=SPARSITIES,
                        WORD_COUNT=WORD_COUNT,
                        POOL_SIZE=POOL_SIZE
                        )






# PARENT OF CreateBinaryTarget, CreateFloatTarget, CreateSoftmaxTarget
class ApexCreateTarget(ApexCreateTestObjects):

    def __init__(self,
                 return_format,
                 return_orientation,
                 TARGET_OBJECT=None,
                 TARGET_OBJECT_HEADER=None,
                 FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                 override_sup_obj=None,
                 bypass_validation=None,
                 calling_module=None,
                 calling_fxn=None,
                 # CREATE FROM GIVEN ONLY ###############################################
                 given_orientation=None,
                 # END CREATE FROM GIVEN ONLY #############################################
                 # CREATE FROM SCRATCH_ONLY ################################
                 rows=None,
                 build_from_mod_dtype=None,    # 'FLOAT' FOR FLOAT, 'BIN' FOR BIN, 'STR' FOR SOFTMAX
                 number_of_categories=None,     # SOFTMAX
                 min_value=None,                # FLOAT
                 max_value=None,                # FLOAT
                 _sparsity=None                 # COULD NEED FOR BINARY (OR MAYBE EVEN FLOAT)
                 ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        self.calling_module = self.this_module if calling_module is None else calling_module
        self.calling_fxn = '__init__' if calling_fxn is None else calling_fxn

        super().__init__('TARGET',
                        return_format,
                        return_orientation,
                        OBJECT=TARGET_OBJECT,
                        OBJECT_HEADER=TARGET_OBJECT_HEADER,
                        FULL_SUPOBJ_OR_SINGLE_MDTYPES=FULL_SUPOBJ_OR_SINGLE_MDTYPES,
                        override_sup_obj=override_sup_obj,
                        bypass_validation=bypass_validation,
                        calling_module=self.calling_module,
                        calling_fxn=self.calling_fxn,
                        # CREATE FROM GIVEN ONLY ###############################################
                        given_orientation=given_orientation,
                        # END CREATE FROM GIVEN ONLY #############################################
                        # CREATE FROM SCRATCH_ONLY ################################
                        rows = rows,
                        columns = 1,                    # FOR ALL TYPES OF TARGETS
                        BUILD_FROM_MOD_DTYPES = build_from_mod_dtype,
                        NUMBER_OF_CATEGORIES = number_of_categories,
                        MIN_VALUES = min_value,
                        MAX_VALUES = max_value,
                        SPARSITIES = _sparsity,
                        # WORD_COUNT = None,
                        # POOL_SIZE = None
                        # END CREATE FROM SCRATCH_ONLY #############################
                        )

        self.expand(auto_drop_rightmost_column=False)




class CreateBinaryTarget(ApexCreateTarget):

    def __init__(self,
                 return_format,
                 return_orientation,
                 TARGET_OBJECT=None,
                 TARGET_OBJECT_HEADER=None,
                 FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,   # MOD_DTYPE CAN ONLY BE BIN!
                 override_sup_obj=None,
                 bypass_validation=None,
                 # CREATE FROM GIVEN ONLY ###############################################
                 given_orientation=None,
                 # END CREATE FROM GIVEN ONLY #############################################
                 # CREATE FROM SCRATCH_ONLY ################################
                 rows=None,
                 _sparsity=50
                 ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        self.calling_module = self.this_module
        self.calling_fxn = '__init__'


        super().__init__(return_format,
                         return_orientation,
                         TARGET_OBJECT=TARGET_OBJECT,
                         TARGET_OBJECT_HEADER=TARGET_OBJECT_HEADER,
                         FULL_SUPOBJ_OR_SINGLE_MDTYPES=FULL_SUPOBJ_OR_SINGLE_MDTYPES,
                         override_sup_obj=override_sup_obj,
                         bypass_validation=bypass_validation,
                         calling_module=self.calling_module,
                         calling_fxn=self.calling_fxn,
                         # CREATE FROM GIVEN ONLY ###############################################
                         given_orientation=given_orientation,
                         # END CREATE FROM GIVEN ONLY #############################################
                         # CREATE FROM SCRATCH_ONLY ################################
                         rows=rows,
                         build_from_mod_dtype='BIN',
                         # number_of_categories=None,     # SOFTMAX
                         # min_value=None,                # FLOAT
                         # max_value=None,                # FLOAT
                         _sparsity=_sparsity                 # COULD NEED FOR BINARY (OR MAYBE EVEN FLOAT)
                         )



class CreateFloatTarget(ApexCreateTarget):

    def __init__(self,
                 return_format,
                 return_orientation,
                 TARGET_OBJECT=None,
                 TARGET_OBJECT_HEADER=None,
                 FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                 override_sup_obj=None,
                 bypass_validation=None,
                 # CREATE FROM GIVEN ONLY ###############################################
                 given_orientation=None,
                 # END CREATE FROM GIVEN ONLY #############################################
                 # CREATE FROM SCRATCH_ONLY ################################
                 rows=None,
                 build_from_mod_dtype=None,
                 min_value=None,
                 max_value=None,
                 _sparsity=0
                 ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        self.calling_module = self.this_module
        self.calling_fxn = '__init__'

        # DTYPE MUST BE FLOAT OR INT
        build_from_mod_dtype = akv.arg_kwarg_validater(build_from_mod_dtype, 'build_from_mod_dtype', ['FLOAT', 'INT', None],
                                                       self.calling_module, self.calling_fxn, return_if_none='FLOAT')

        super().__init__(return_format,
                         return_orientation,
                         TARGET_OBJECT=TARGET_OBJECT,
                         TARGET_OBJECT_HEADER=TARGET_OBJECT_HEADER,
                         FULL_SUPOBJ_OR_SINGLE_MDTYPES=FULL_SUPOBJ_OR_SINGLE_MDTYPES,
                         override_sup_obj=override_sup_obj,
                         bypass_validation=bypass_validation,
                         calling_module=self.calling_module,
                         calling_fxn=self.calling_fxn,
                         # CREATE FROM GIVEN ONLY ###############################################
                         given_orientation=given_orientation,
                         # END CREATE FROM GIVEN ONLY #############################################
                         # CREATE FROM SCRATCH_ONLY ################################
                         rows=rows,
                         build_from_mod_dtype=build_from_mod_dtype,    # 'FLOAT' OR 'INT'
                         min_value=min_value,                # FLOAT
                         max_value=max_value,                # FLOAT
                         _sparsity=_sparsity                 # COULD NEED FOR BINARY (OR MAYBE EVEN FLOAT)
                         )



class CreateSoftmaxTarget(ApexCreateTarget):

    def __init__(self,
                 return_format,
                 return_orientation,
                 TARGET_OBJECT=None,
                 TARGET_OBJECT_HEADER=None,
                 FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                 override_sup_obj=None,
                 bypass_validation=None,
                 # CREATE FROM GIVEN ONLY ###############################################
                 given_orientation=None,
                 # END CREATE FROM GIVEN ONLY #############################################
                 # CREATE FROM SCRATCH_ONLY ################################
                 rows=None,
                 number_of_categories=5,
                 ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        self.calling_module = self.this_module
        self.calling_fxn = '__init__'


        super().__init__(return_format,
                         return_orientation,
                         TARGET_OBJECT=TARGET_OBJECT,
                         TARGET_OBJECT_HEADER=TARGET_OBJECT_HEADER,
                         FULL_SUPOBJ_OR_SINGLE_MDTYPES=FULL_SUPOBJ_OR_SINGLE_MDTYPES,
                         override_sup_obj=override_sup_obj,
                         bypass_validation=bypass_validation,
                         calling_module=self.calling_module,
                         calling_fxn=self.calling_fxn,
                         # CREATE FROM GIVEN ONLY ###############################################
                         given_orientation=given_orientation,
                         # END CREATE FROM GIVEN ONLY #############################################
                         # CREATE FROM SCRATCH_ONLY ################################
                         rows=rows,
                         build_from_mod_dtype='STR',
                         number_of_categories=number_of_categories,
                         min_value=None,
                         max_value=None,
                         _sparsity=None
                         )








class CreateRefVecs(ApexCreateTestObjects):    # ApexCreate TO GET expand METHOD, DONT super()!
    def __init__(self,
                return_format,
                return_orientation,
                REFVEC_OBJECT=None,
                REFVEC_OBJECT_HEADER=None,
                FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                BUILD_FROM_MOD_DTYPES=['STR', 'STR', 'STR', 'STR', 'STR', 'BIN', 'INT'],
                override_sup_obj=None,
                bypass_validation=None,
                given_orientation=None,
                rows = None,
                columns = None,
                NUMBER_OF_CATEGORIES = 10,
                MIN_VALUES = -10,
                MAX_VALUES = 10,
                SPARSITIES = 50,
                WORD_COUNT = 20,
                POOL_SIZE = 200
                ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        self.calling_module = self.this_module
        self.calling_fxn = '__init__'

        super().__init__('REFVECS',
                         return_format,
                         return_orientation,
                         OBJECT=REFVEC_OBJECT,
                         OBJECT_HEADER=REFVEC_OBJECT_HEADER,
                         FULL_SUPOBJ_OR_SINGLE_MDTYPES=FULL_SUPOBJ_OR_SINGLE_MDTYPES,
                         override_sup_obj=override_sup_obj,
                         bypass_validation=bypass_validation,
                         calling_module=self.calling_module,
                         calling_fxn=self.calling_fxn,
                         # CREATE FROM GIVEN ONLY ###############################################
                         given_orientation=given_orientation,
                         # END CREATE FROM GIVEN ONLY #############################################
                         # CREATE FROM SCRATCH_ONLY ################################
                         rows=rows,
                         columns=columns,
                         BUILD_FROM_MOD_DTYPES=BUILD_FROM_MOD_DTYPES,
                         NUMBER_OF_CATEGORIES=NUMBER_OF_CATEGORIES,
                         MIN_VALUES=MIN_VALUES,
                         MAX_VALUES=MAX_VALUES,
                         SPARSITIES=SPARSITIES,
                         WORD_COUNT=WORD_COUNT,
                         POOL_SIZE=POOL_SIZE
                         )







































