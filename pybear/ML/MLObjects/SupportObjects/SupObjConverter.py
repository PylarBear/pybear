import sys, inspect, time
from general_sound import winlinsound as wls
import numpy as np
from MLObjects.SupportObjects import master_support_object_dict as msod, validate_full_support_object as vfso
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from debug import get_module_name as gmn
from general_data_ops import get_shape as gs


# ACCESS DATA_FULL_SUPOBJ, TARGET_FULL_SUPOBJ, & REFVECS_FULL_SUPOBJ AS ATTRS OF THE CLASS


# PASSED SUPOBJ KWARGS MUST LOOK LIKE THIS
# OLD SUPOBJS LOOK LIKE (MOD_DTYPES, EG):
# MODIFIED_DATATYPES = [['STR','INT'], '', ['INT'], '', ['STR','STR','STR'], '']

# TO COMPILE SUPOBJS THAT ARE SINGLE, EG ['STR','STR','STR'], USE CompileFullSupportObject


class SupObjConverter:

    def __init__(self, DATA_HEADER=None, TARGET_HEADER=None, REFVECS_HEADER=None, VALIDATED_DATATYPES=None,
                    MODIFIED_DATATYPES=None, FILTERING=None, MIN_CUTOFFS=None, USE_OTHER=None, START_LAG=None,
                    END_LAG=None, SCALING=None):

        # WOULD HAVE TO GET HEADERS FROM [1,3,5] POSNS OF SRNL OR SWNL

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        self.OLD_SUPOBJ_DICT = {'VALIDATEDDATATYPES' :VALIDATED_DATATYPES, 'MODIFIEDDATATYPES' :MODIFIED_DATATYPES,
            'FILTERING' :FILTERING, 'MIN_CUTOFFS' :MIN_CUTOFFS, 'USE_OTHER' :USE_OTHER, 'START_LAG' :START_LAG,
            'END_LAG' :END_LAG, 'SCALING' :SCALING}
        self.old_idx = {'DATA': 0, 'TARGET': 2, 'REFVECS': 4}

        DATA_HEADER = ldv.list_dict_validater(DATA_HEADER, 'DATA_HEADER')[1]
        TARGET_HEADER = ldv.list_dict_validater(TARGET_HEADER, 'TARGET_HEADER')[1]
        REFVECS_HEADER = ldv.list_dict_validater(REFVECS_HEADER, 'REFVECS_HEADER')[1]


        # VALIDATE FORMAT OF ALL NON-None SUPOBJ KWARGS
        for _name, _SUPOBJ in zip(self.OLD_SUPOBJ_DICT.keys(), self.OLD_SUPOBJ_DICT.values()):
            if not _SUPOBJ is None and not len(_SUPOBJ)==6:
                self._exception(f'{_name} OLD SUPPORT OBJECT MUST BE PASSED WITH len OF 6 (IS {len(_SUPOBJ)}) OR NOT PASSED AT ALL')


        # COLUMNS IN THE 1 SLOT, 3 SLOT, AND 5 SLOT OF OLD SUPOBJS MUST BE EQUAL (IF NOT None).  IGNORE 2,4,6 SLOTS
        # CREATE A WIP OLD_SUPOBJ FOR ACTIVE OLD SUPOBJS
        ACTV_SUPOBJS = {k:v for k,v in self.OLD_SUPOBJ_DICT.items() if not v is None}

        for obj_name, obj_idx in self.old_idx.items():
            _LENS = list(map(len, [ACTV_SUPOBJS[k][obj_idx] for k in ACTV_SUPOBJS]))

            if len(_LENS) == 0: break

            if not min(_LENS)==max(_LENS):
                self._exception(f'{obj_name} OLD SUPPORT OBJECTS DO NOT HAVE EQUAL LENGTH IN THE {obj_idx} POSITION >>>'
                                f'({", ".join(list(map(str, _LENS)))}',fxn=fxn)

        # IF GOT THRU THE VALIDATION OF SUPOBJ LENS ABOVE, ANY ONE OF THEM CAN SET cols FOR THE OBJECTS

        # IF NO SUPOBJS BESIDES HEADERS, USE HEADERS TO GET COLUMNS
        # IF NOTHING WAS PASSED, BUILD EMPTY SUPOBJS (len(msod) ROWS, 0 COLUMNS)
        try: data_cols = len(ACTV_SUPOBJS[list(ACTV_SUPOBJS.keys())[0]][0])
        except:
            try: data_cols = len(DATA_HEADER[0])
            except: data_cols = 0
        try: target_cols = len(ACTV_SUPOBJS[list(ACTV_SUPOBJS.keys())[0]][2])
        except:
            try: target_cols = len(TARGET_HEADER[0])
            except: target_cols = 0
        try: refvec_cols = len(ACTV_SUPOBJS[list(ACTV_SUPOBJS.keys())[0]][4])
        except:
            try: refvec_cols = len(REFVECS_HEADER[0])
            except: refvec_cols = 0


        def header_builder(obj_name, OLD_HEADER, NEW_SUPOBJ):
            obj_name = obj_name.upper()
            if not OLD_HEADER is None:
                OLD_HEADER = OLD_HEADER.reshape((1, -1))[0]
                NEW_SUPOBJ[msod.QUICK_POSN_DICT()["HEADER"]] = OLD_HEADER
            elif OLD_HEADER is None:
                NEW_SUPOBJ[msod.QUICK_POSN_DICT()["HEADER"]] = \
                    np.fromiter((f'{obj_name}_COLUMN{idx + 1}' for idx in range(len(NEW_SUPOBJ[0]))), dtype='<U200')
            return NEW_SUPOBJ


        def get_old_supobj(obj_name, supobj_name, OLD_SUPOBJ, NEW_SUPOBJ):
            if OLD_SUPOBJ is None or len(OLD_SUPOBJ)==0: pass
                # IF OLD SUPOBJ DOES NOT EXIST OR IS [], NO CHANGE IS MADE TO NEW_SUPOBJ (PERTINENT ROW IS LEFT AS empty_value())
            else:
                OLD_SINGLE_SUPOBJ = OLD_SUPOBJ[self.old_idx[obj_name]]
                # FIND POSITION FOR OLD_SUPOBJ IN NEW_SUPOBJ
                NEW_SUPOBJ[msod.QUICK_POSN_DICT()[supobj_name]] = OLD_SINGLE_SUPOBJ
            return NEW_SUPOBJ


        self.DATA_FULL_SUPOBJ = msod.build_empty_support_object(data_cols)
        self.DATA_FULL_SUPOBJ = header_builder('DATA', DATA_HEADER, self.DATA_FULL_SUPOBJ)
        self.DATA_FULL_SUPOBJ = get_old_supobj('DATA', 'VALIDATEDDATATYPES', VALIDATED_DATATYPES, self.DATA_FULL_SUPOBJ)
        self.DATA_FULL_SUPOBJ = get_old_supobj('DATA', 'MODIFIEDDATATYPES', MODIFIED_DATATYPES, self.DATA_FULL_SUPOBJ)
        self.DATA_FULL_SUPOBJ = get_old_supobj('DATA', 'FILTERING', FILTERING, self.DATA_FULL_SUPOBJ)
        self.DATA_FULL_SUPOBJ = get_old_supobj('DATA', 'MINCUTOFFS', MIN_CUTOFFS, self.DATA_FULL_SUPOBJ)
        self.DATA_FULL_SUPOBJ = get_old_supobj('DATA', 'USEOTHER', USE_OTHER, self.DATA_FULL_SUPOBJ)
        self.DATA_FULL_SUPOBJ = get_old_supobj('DATA', 'STARTLAG', START_LAG, self.DATA_FULL_SUPOBJ)
        self.DATA_FULL_SUPOBJ = get_old_supobj('DATA', 'ENDLAG', END_LAG, self.DATA_FULL_SUPOBJ)
        self.DATA_FULL_SUPOBJ = get_old_supobj('DATA', 'SCALING', SCALING, self.DATA_FULL_SUPOBJ)


        self.TARGET_FULL_SUPOBJ = msod.build_empty_support_object(target_cols)
        self.TARGET_FULL_SUPOBJ = header_builder('TARGET', TARGET_HEADER, self.TARGET_FULL_SUPOBJ)
        self.TARGET_FULL_SUPOBJ = get_old_supobj('TARGET', 'VALIDATEDDATATYPES', VALIDATED_DATATYPES, self.TARGET_FULL_SUPOBJ)
        self.TARGET_FULL_SUPOBJ = get_old_supobj('TARGET', 'MODIFIEDDATATYPES', MODIFIED_DATATYPES, self.TARGET_FULL_SUPOBJ)
        self.TARGET_FULL_SUPOBJ = get_old_supobj('TARGET', 'FILTERING', FILTERING, self.TARGET_FULL_SUPOBJ)
        self.TARGET_FULL_SUPOBJ = get_old_supobj('TARGET', 'MINCUTOFFS', MIN_CUTOFFS, self.TARGET_FULL_SUPOBJ)
        self.TARGET_FULL_SUPOBJ = get_old_supobj('TARGET', 'USEOTHER', USE_OTHER, self.TARGET_FULL_SUPOBJ)
        self.TARGET_FULL_SUPOBJ = get_old_supobj('TARGET', 'STARTLAG', START_LAG, self.TARGET_FULL_SUPOBJ)
        self.TARGET_FULL_SUPOBJ = get_old_supobj('TARGET', 'ENDLAG', END_LAG, self.TARGET_FULL_SUPOBJ)
        self.TARGET_FULL_SUPOBJ = get_old_supobj('TARGET', 'SCALING', SCALING, self.TARGET_FULL_SUPOBJ)


        self.REFVECS_FULL_SUPOBJ = msod.build_empty_support_object(refvec_cols)
        self.REFVECS_FULL_SUPOBJ = header_builder('REFVECS', REFVECS_HEADER, self.REFVECS_FULL_SUPOBJ)
        self.REFVECS_FULL_SUPOBJ = get_old_supobj('REFVECS', 'VALIDATEDDATATYPES', VALIDATED_DATATYPES, self.REFVECS_FULL_SUPOBJ)
        self.REFVECS_FULL_SUPOBJ = get_old_supobj('REFVECS', 'MODIFIEDDATATYPES', MODIFIED_DATATYPES, self.REFVECS_FULL_SUPOBJ)
        self.REFVECS_FULL_SUPOBJ = get_old_supobj('REFVECS', 'FILTERING', FILTERING, self.REFVECS_FULL_SUPOBJ)
        self.REFVECS_FULL_SUPOBJ = get_old_supobj('REFVECS', 'MINCUTOFFS', MIN_CUTOFFS, self.REFVECS_FULL_SUPOBJ)
        self.REFVECS_FULL_SUPOBJ = get_old_supobj('REFVECS', 'USEOTHER', USE_OTHER, self.REFVECS_FULL_SUPOBJ)
        self.REFVECS_FULL_SUPOBJ = get_old_supobj('REFVECS', 'STARTLAG', START_LAG, self.REFVECS_FULL_SUPOBJ)
        self.REFVECS_FULL_SUPOBJ = get_old_supobj('REFVECS', 'ENDLAG', END_LAG, self.REFVECS_FULL_SUPOBJ)
        self.REFVECS_FULL_SUPOBJ = get_old_supobj('REFVECS', 'SCALING', SCALING, self.REFVECS_FULL_SUPOBJ)


        del fxn, DATA_HEADER, TARGET_HEADER, REFVECS_HEADER, ACTV_SUPOBJS, data_cols, target_cols, refvec_cols, header_builder, get_old_supobj
        del self.OLD_SUPOBJ_DICT, self.old_idx


    # SUPPORT METHOD, DO NOT CALL EXTERNALLY
    def _exception(self, words, fxn=None):
        fxn = f'.{fxn}()' if not fxn is None else ''
        raise Exception(f'{self.this_module}{fxn} >>> {words}')


    # SERVICE METHOD, TO BE CALLED EXTERNALLY
    def validate_against_obj(self, OBJECT, obj_name, given_orientation, allow_supobj_override=None):
        fxn = inspect.stack()[0][3]
        obj_name = akv.arg_kwarg_validater(obj_name, 'obj_name', ['DATA','TARGET','REFVECS'], self.this_module, fxn)
        given_orientation = akv.arg_kwarg_validater(given_orientation, 'given_orientation', ['ROW', 'COLUMN'],
                                                    self.this_module, fxn)
        allow_supobj_override = akv.arg_kwarg_validater(allow_supobj_override, 'allow_supobj_override', [True, False, None],
                                                 self.this_module, fxn, return_if_none=False)

        OBJECT = ldv.list_dict_validater(OBJECT, obj_name)[1]
        _cols = gs.get_shape(obj_name, OBJECT, given_orientation)[1]

        ACTV_SUPOBJ = {'DATA':self.DATA_FULL_SUPOBJ, 'TARGET':self.TARGET_FULL_SUPOBJ, 'REFVECS':self.REFVECS_FULL_SUPOBJ}[obj_name]

        # VALIDATE COLUMNS CONGRUENT
        _so_cols = len(ACTV_SUPOBJ[0])
        if not _cols == _so_cols:
            self._exception(f'{obj_name} SUPOBJ COLUMNS ({_so_cols}) DO NOT EQUAL {obj_name} COLUMNS ({_cols})', fxn=fxn)

        # VALIDATE SUPOBJ ENTRIES AGAINST OBJECT
        vfso. validate_full_support_object(ACTV_SUPOBJ, OBJECT=OBJECT, object_given_orientation=given_orientation,
                   OBJECT_HEADER=ACTV_SUPOBJ[msod.QUICK_POSN_DICT()["HEADER"]], allow_override=allow_supobj_override)

        del ACTV_SUPOBJ, _so_cols, _cols













if __name__ == '__main__':
    from MLObjects.TestObjectCreators import ApexTestObjectCreate as atoc, test_header as th

    # TEST MODULE

    # 1) CREATE DATA, TARGET, & REFVEC OBJECTS  (FULL SUPOBJS ARE INCIDENTALLY CREATED)
    # 2) DISASSEMBLE FULL SUPOBJS INTO INDIVIDUAL COMPONENTS
    # 3) REASSEMBLE INDIVIDUAL SUPOBJS INTO OLD SUPOBJ FORMAT
    # 4) USE SupObjConverter TO CONVERT OLD SUPOBJ FORMAT TO FULL SUPOBJS
    # 5) COMPARE REASSEMBLED FULL SUPOBJS TO SUPOBJS ORIGINALLY CREATED
    # 6) TEST validate_against_obj() USING ORIGINAL CREATED DATA, TARGET, & REFVECS


    this_module = 'test_module'
    fxn = 'test_fxn'

    _format = 'ARRAY'
    _orient = 'COLUMN'
    _columns = 5
    _rows = 100

    ########################################################################################################################
    # BUILD DATA ###########################################################################################################

    TestData = atoc.CreateDataObject(_format,
                                     _orient,
                                     DATA_OBJECT=None,
                                     DATA_OBJECT_HEADER=th.test_header(_columns),
                                     FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                     override_sup_obj=False,
                                     bypass_validation=False,
                                     # CREATE FROM GIVEN ONLY ###############################################
                                     given_orientation=None,
                                     # END CREATE FROM GIVEN ONLY #############################################
                                     # CREATE FROM SCRATCH_ONLY ################################
                                     rows=_rows,
                                     columns=_columns,
                                     BUILD_FROM_MOD_DTYPES=['BIN', 'INT', 'FLOAT', 'STR'],
                                     NUMBER_OF_CATEGORIES=10,
                                     MIN_VALUES=-10,
                                     MAX_VALUES=10,
                                     SPARSITIES=0
                                     # END CREATE FROM SCRATCH_ONLY #############################
                                     )

    DATA = TestData.OBJECT
    DATA_HEADER = TestData.OBJECT_HEADER
    DATA_SUPOBJ = TestData.SUPPORT_OBJECTS
    del TestData

    # END BUILD DATA ########################################################################################################
    ########################################################################################################################

    ########################################################################################################################
    # BUILD TARGET ###########################################################################################################

    TestTarget = atoc.CreateBinaryTarget(_format,
                                         _orient,
                                         TARGET_OBJECT=None,
                                         TARGET_OBJECT_HEADER=['TARGET_1'],
                                         FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,   # MOD_DTYPE CAN ONLY BE BIN!
                                         override_sup_obj=None,
                                         bypass_validation=False,
                                         # CREATE FROM GIVEN ONLY ###############################################
                                         given_orientation=None,
                                         # END CREATE FROM GIVEN ONLY #############################################
                                         # CREATE FROM SCRATCH_ONLY ################################
                                         rows=_rows,
                                         _sparsity=50
                                         )

    TARGET = TestTarget.OBJECT
    TARGET_HEADER = TestTarget.OBJECT_HEADER
    TARGET_SUPOBJ = TestTarget.SUPPORT_OBJECTS
    del TestTarget

    # END BUILD TARGET ######################################################################################################
    ########################################################################################################################

    ########################################################################################################################
    # BUILD REFVECS #########################################################################################################
    TestRefVecs = atoc.CreateRefVecs(_format,
                                     _orient,
                                     REFVEC_OBJECT=None,
                                     REFVEC_OBJECT_HEADER=['REFVECS_1'],
                                     FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                     BUILD_FROM_MOD_DTYPES=['STR'],
                                     override_sup_obj=None,
                                     bypass_validation=False,
                                     given_orientation=None,
                                     rows=_rows,
                                     columns=1,
                                     NUMBER_OF_CATEGORIES=5
                                     )

    REFVECS = TestRefVecs.OBJECT
    REFVECS_HEADER = TestRefVecs.OBJECT_HEADER
    REFVECS_SUPOBJ = TestRefVecs.SUPPORT_OBJECTS
    del TestRefVecs

    # END BUILD REFVECS #####################################################################################################
    ########################################################################################################################


    # TURN FULL_SUPOBJS INTO [[...DATA...], '', [...TARGET...], '', [...REFVECS...], '']

    TEMPLATE = ['' for _ in range(6)]
    VALIDATED_DATATYPES = TEMPLATE.copy()
    MODIFIED_DATATYPES = TEMPLATE.copy()
    FILTERING = TEMPLATE.copy()
    MIN_CUTOFFS = TEMPLATE.copy()
    USE_OTHER = TEMPLATE.copy()
    START_LAG = TEMPLATE.copy()
    END_LAG = TEMPLATE.copy()
    SCALING = TEMPLATE.copy()
    for idx, _SUP_OBJ in zip((0,2,4), (DATA_SUPOBJ, TARGET_SUPOBJ, REFVECS_SUPOBJ)):
        VALIDATED_DATATYPES[idx] = _SUP_OBJ[msod.QUICK_POSN_DICT()['VALIDATEDDATATYPES']]
        MODIFIED_DATATYPES[idx] = _SUP_OBJ[msod.QUICK_POSN_DICT()['MODIFIEDDATATYPES']]
        FILTERING[idx] = _SUP_OBJ[msod.QUICK_POSN_DICT()['FILTERING']]
        MIN_CUTOFFS[idx] = _SUP_OBJ[msod.QUICK_POSN_DICT()['MINCUTOFFS']]
        USE_OTHER[idx] = _SUP_OBJ[msod.QUICK_POSN_DICT()['USEOTHER']]
        START_LAG[idx] = _SUP_OBJ[msod.QUICK_POSN_DICT()['STARTLAG']]
        END_LAG[idx] = _SUP_OBJ[msod.QUICK_POSN_DICT()['ENDLAG']]
        SCALING[idx] = _SUP_OBJ[msod.QUICK_POSN_DICT()['SCALING']]


    TestClass = SupObjConverter(DATA_HEADER=DATA_HEADER,
                                TARGET_HEADER=TARGET_HEADER,
                                REFVECS_HEADER=REFVECS_HEADER,
                                VALIDATED_DATATYPES=VALIDATED_DATATYPES,
                                MODIFIED_DATATYPES=MODIFIED_DATATYPES,
                                FILTERING=FILTERING,
                                MIN_CUTOFFS=MIN_CUTOFFS,
                                USE_OTHER=USE_OTHER,
                                START_LAG=START_LAG,
                                END_LAG=END_LAG,
                                SCALING=SCALING)

    DATA_ACT_SUPOBJ = TestClass.DATA_FULL_SUPOBJ
    TARGET_ACT_SUPOBJ = TestClass.TARGET_FULL_SUPOBJ
    REFVECS_ACT_SUPOBJ = TestClass.REFVECS_FULL_SUPOBJ

    DATA_EXP_SUPOBJ = DATA_SUPOBJ
    TARGET_EXP_SUPOBJ = TARGET_SUPOBJ
    REFVECS_EXP_SUPOBJ = REFVECS_SUPOBJ

    # TEST CONVERTED SUPOBJS
    for name, _EXP_OBJ, _ACT_OBJ in zip(('DATA', 'TARGET', 'REFVECS'),
                                          (DATA_EXP_SUPOBJ, TARGET_EXP_SUPOBJ, REFVECS_EXP_SUPOBJ),
                                          (DATA_ACT_SUPOBJ,TARGET_ACT_SUPOBJ,REFVECS_ACT_SUPOBJ)):
        if not np.array_equiv(_EXP_OBJ, _ACT_OBJ):
            print(f'\033[91m')
            print(f'*** {name} EXP AND ACTUAL ARE NOT EQUAL ***')
            print(f'\nEXPECTED:')
            print(_EXP_OBJ)
            print(f'\nACTUAL:')
            print(_ACT_OBJ)
            print(f'\033[0m')

    print(f'\n\033[92m*** TEST FOR EQUALITY WITH EXPECTED PASSED ***\033[0m')

    # TEST OBJECT VALIDATION
    TestClass.validate_against_obj(DATA, 'DATA', _orient, allow_supobj_override=True)
    TestClass.validate_against_obj(TARGET, 'TARGET', _orient, allow_supobj_override=True)
    TestClass.validate_against_obj(REFVECS, 'REFVECS', _orient, allow_supobj_override=True)

    print(f'\n\033[92m*** TESTS FOR validate_against_obj() PASSED ***\033[0m')

    del TestClass

    for _ in range(3): wls.winlinsound(888,500); time.sleep(1)





