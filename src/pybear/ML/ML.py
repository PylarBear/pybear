import sys, os, time, warnings
import pandas as pd, numpy as np
sys.tracebacklimit = None
from copy import deepcopy

from data_validation import validate_user_input as vui
from general_data_ops import get_shape as gs
from general_text import TextCleaner as tc
from read_write_file.generate_full_filename import base_path_select as bps, filename_enter as fe
import sparse_dict as sd

from MLObjects.SupportObjects import master_support_object_dict as msod, SupObjConverter as soc, NEWSupObjToOLD as nsoto
from MLObjects.PrintMLObject import SmallObjectPreview as sop
from MLObjects.TestObjectCreators.SXNL import NewObjsToOldSXNL as notos

from ML_PACKAGE.standard_configs import standard_configs as sc
from ML_PACKAGE.GENERIC_PRINT import print_object_create_success as pocs, print_post_run_options as ppro, ObjectsExcelDump as oed
import ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.class_DataTargetReferenceTestReadBuildDF as trdrb
from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE import class_BaseBMRVTVTMBuild as bob
from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE import PreRunFilter as prf

from ML_PACKAGE.DATA_PREP_IN_SITU_PACKAGE import InSituFilter as isf
from ML_PACKAGE.DATA_PREP_IN_SITU_PACKAGE import InSituExpandCategories as isec
from ML_PACKAGE.DATA_PREP_IN_SITU_PACKAGE import InSituDataAugment as isda
from ML_PACKAGE.DATA_PREP_IN_SITU_PACKAGE.target_vector import TargetInSituHandling as tish

from ML_PACKAGE.MLREGRESSION import MLRegressionConfigRun as mlrcr
from ML_PACKAGE.GDA_PACKAGE import GDA_run as gdar
from ML_PACKAGE.NN_PACKAGE import NNConfigRun as ncr
from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION import GMLRConfigRun as gmlrcr
from ML_PACKAGE.MUTUAL_INFORMATION import MIConfigRun as micr
from ML_PACKAGE.SVD import SVDConfigRun as svdcr
from ML_PACKAGE.SVM_PACKAGE import SVMConfigRun as scr




###########################################################################################################################
###########################################################################################################################
# BEAR THIS GETS DELETED WHEN Filter, Augment, GO TO NEW SUPOBJS #########################################

def old_objs_to_new(OLD_SXNL, VALIDATED_DATATYPES, MODIFIED_DATATYPES, FILTERING, MIN_CUTOFFS, USE_OTHER, START_LAG,
                       END_LAG, SCALING):
    SupObjClass = soc.SupObjConverter( DATA_HEADER=OLD_SXNL[1], TARGET_HEADER=OLD_SXNL[3], REFVECS_HEADER=OLD_SXNL[5],
        VALIDATED_DATATYPES=VALIDATED_DATATYPES, MODIFIED_DATATYPES=MODIFIED_DATATYPES, FILTERING=FILTERING,
        MIN_CUTOFFS=MIN_CUTOFFS, USE_OTHER=USE_OTHER, START_LAG=START_LAG, END_LAG=END_LAG, SCALING=SCALING)

    SUPOBJS = [SupObjClass.DATA_FULL_SUPOBJ, SupObjClass.TARGET_FULL_SUPOBJ, SupObjClass.REFVECS_FULL_SUPOBJ]
    del SupObjClass
    SXNL = [OLD_SXNL[idx] for idx in range(len(OLD_SXNL)) if idx % 2 == 0]

    return SXNL, SUPOBJS



def new_objs_to_old(NEW_SXNL, NEW_SUPOBJS, data_given_orientation, target_given_orientation, refvecs_given_orientation):

    SXNLClass = notos.NewObjsToOldSXNL(*NEW_SXNL, *NEW_SUPOBJS, data_orientation=data_given_orientation,
        target_orientation=target_given_orientation, refvecs_orientation=refvecs_given_orientation, bypass_validation=False)

    SXNL = SXNLClass.SXNL
    del SXNLClass

    OldSupObjClass = nsoto.NEWSupObjToOLD(*NEW_SUPOBJS)
    VALIDATED_DATATYPES = OldSupObjClass.VALIDATED_DATATYPES
    MODIFIED_DATATYPES = OldSupObjClass.MODIFIED_DATATYPES
    FILTERING = OldSupObjClass.FILTERING
    MIN_CUTOFFS = OldSupObjClass.MIN_CUTOFFS
    USE_OTHER = OldSupObjClass.USE_OTHER
    START_LAG = OldSupObjClass.START_LAG
    END_LAG = OldSupObjClass.END_LAG
    SCALING = OldSupObjClass.SCALING
    del OldSupObjClass

    return SXNL, VALIDATED_DATATYPES, MODIFIED_DATATYPES, FILTERING, MIN_CUTOFFS, USE_OTHER, START_LAG, END_LAG, SCALING

# END BEAR THIS GETS DELETED WHEN Filter, Augment, GO TO NEW SUPOBJS ######################################################
###########################################################################################################################
###########################################################################################################################




def ML():

    # SETUP*************************************************************************************************************
    # ******************************************************************************************************************
    # ******************************************************************************************************************

    # SELECT CONFIGURATION PACKAGE *************************************************************************************

    data_read_manual_config, raw_target_read_manual_config, rv_read_manual_config = 'Y','Y','Y'
    data_read_method, raw_target_read_method, rv_read_method = '','',''
    BBM_manual_config, base_raw_target_manual_config, base_rv_manual_config  = 'Y','Y','Y'
    BBM_build_method, base_raw_target_build_method, base_rv_build_method = '','',''

    standard_config, data_read_manual_config, raw_target_read_manual_config, rv_read_manual_config, data_read_method, \
    raw_target_read_method, rv_read_method, BBM_manual_config, base_raw_target_manual_config, base_rv_manual_config, \
    BBM_build_method, base_raw_target_build_method, base_rv_build_method = \
        sc.standard_configs(data_read_manual_config, raw_target_read_manual_config, rv_read_manual_config,
            data_read_method, raw_target_read_method, rv_read_method, BBM_manual_config, base_raw_target_manual_config,
            base_rv_manual_config, BBM_build_method, base_raw_target_build_method, base_rv_build_method)


    # IF MANUAL ENTRY JUST SKIP THIS, ALL MANUAL CONFIGS HAVE BEEN SET TO 'Y' ALREADY
    if standard_config != 'MANUAL ENTRY':
        print(f'\nCURRENT CONFIGURATION PARAMETERS ARE:')
        DUM_LIST = ['DATA READ MANUAL CONFIG', 'RAW TARGET READ MANUAL CONFIG', 'REFERENCE VECTOR READ MANUAL CONFIG',
                    'BBM MANUAL CONFIG', 'BASE RAW TARGET MANUAL CONFIG', 'BASE REFERENCE VECTOR MANUAL CONFIG']
        DUM_LIST2 = ['DATA READ CONFIG', 'RAW TARGET READ CONFIG', 'REFERENCE VECTOR READ CONFIG',
                     'BBM BUILD CONFIG', 'BASE RAW TARGET BUILD CONFIG', 'BASE REFERENCE VECTOR BUILD CONFIG']
        DUM_HOLDER = [data_read_manual_config, raw_target_read_manual_config, rv_read_manual_config,
                      BBM_manual_config, base_raw_target_manual_config, base_rv_manual_config]
        DUM_HOLDER2 = [data_read_method, raw_target_read_method, rv_read_method,
                       BBM_build_method, base_raw_target_build_method, base_rv_build_method]
        [print(f'{DUM_LIST[idx]} = {DUM_HOLDER[idx]} '.ljust(50," ") + f'{DUM_LIST2[idx]} = {DUM_HOLDER2[idx]}') for idx in range(len(DUM_LIST))]
        print('')

        if vui.validate_user_str(f'Override current configuration parameters? (y/n) > ', 'YN') == 'Y':
            for idx in range(len(DUM_LIST)):
                DUM_HOLDER[idx] = vui.validate_user_str(f'{DUM_LIST[idx]} set as manual entry? (y/n) > ', 'YN')

        data_read_manual_config, raw_target_read_manual_config, rv_read_manual_config, BBM_manual_config, \
        base_raw_target_read_manual_config, base_rv_read_manual_config = tuple(DUM_HOLDER)

        del DUM_LIST, DUM_LIST2, DUM_HOLDER, DUM_HOLDER2

        print(f'\nMake setup selections from available {standard_config.upper()} standard configurations\n')
    # ******************************************************************************************************

    # CONFIG DATA SOURCE #############################################################################################

    # DUM IS HERE TO CATCH A DUMMY OUTPUT, PARENT CLASS FOR DataBuild IS SHARED W RawTargetBuild & RefVecBuild
    # WHICH NEED 2 OUTPUTS (THEIR OBJECT + DATA_DF IF MODS WERE MADE) SO CLASS MUST CARRY 2 OUTPUTS TO ACCOMMODATE.
    # THE DF MODIFIER OUTPUT IS NOT NEEDED HERE
    DUM, DATA_DF = trdrb.DataBuild(standard_config, data_read_manual_config, data_read_method, pd.DataFrame(), 'DATA DF').build_object()


    print(f"\nDATA_DF COLUMNS, AS READ:")
    [print(f'{idx}) {list(DATA_DF.keys())[idx]}') for idx in range(len(DATA_DF.keys()))]
    print('')

    if data_read_manual_config == 'Y':
        if trdrb.DataBuild(standard_config, data_read_manual_config, data_read_method, DATA_DF, 'DATA DF').drop_columns_yn() == 'Y':
            DATA_DF = trdrb.DataBuild(standard_config, data_read_manual_config, data_read_method, DATA_DF, 'DATA DF').initial_column_drop(DATA_DF)
    pocs.print_object_create_success(standard_config, 'RAW DATA DATAFRAME')
    # END CONFIG DATA SOURCE READ #############################################################################################

    # CONFIG TARGET SOURCE READ #############################################################################################
    RAW_TARGET_DF, DATA_DF = \
        trdrb.RawTargetBuild(standard_config, raw_target_read_manual_config, raw_target_read_method, DATA_DF).build_object(DATA_DF)
    pocs.print_object_create_success(standard_config, 'RAW TARGET DATAFRAME')
    # END CONFIG TARGET SOURCE READ #########################################################################################

    # CONFIG RV SOURCE READ #############################################################################################
    if rv_read_manual_config == 'Y':
        if vui.validate_user_str(f'Any REFERENCE VECTORS? (y/n) > ', 'YN') == 'Y':
            REFERENCE_VECTORS_DF, DATA_DF = \
                trdrb.ReferenceVectorsBuild(standard_config, rv_read_manual_config, rv_read_method, DATA_DF,
                                            'REFERENCE VECTORS').build_object(DATA_DF)
        else:
            REFERENCE_VECTORS_DF = pd.DataFrame(data=np.fromiter(('DUM' for _ in range(len(DATA_DF))), dtype='<U3') ,columns=['DUM'])

    else:
        REFERENCE_VECTORS_DF, DATA_DF = \
            trdrb.ReferenceVectorsBuild(standard_config, rv_read_manual_config, rv_read_method, DATA_DF,
                                        'REFERENCE VECTORS').build_object(DATA_DF)

    pocs.print_object_create_success(standard_config, 'REFERENCE VECTOR DATAFRAME')
    # END CONFIG RV SOURCE READ #########################################################################################

    print(f'\nDATAFRAME SOURCE OBJECTS SUCCESSFULLY CREATED')

    # CONVERT SOURCE OBJECTS FROM DF TO NUMPY ############################################################################
    print(f'\nCREATING NUMPY SOURCE OBJECTS...')

    def make_numpy_object_and_header_from_DF(DF_OBJECT):
        OBJECT_NUMPY = DF_OBJECT.to_numpy(dtype=object).transpose()   # EVERYTHING CAN HAVE STR/NUM AT THIS POINT
        OBJECT_HEADER_NUMPY = DF_OBJECT.keys().to_numpy(dtype=str).reshape((1, -1))
        return OBJECT_NUMPY, OBJECT_HEADER_NUMPY

    # DATA
    DATA_NUMPY, DATA_NUMPY_HEADER = make_numpy_object_and_header_from_DF(DATA_DF)

    # RAW TARGET
    RAW_TARGET_NUMPY, RAW_TARGET_NUMPY_HEADER = make_numpy_object_and_header_from_DF(RAW_TARGET_DF)

    # REF VECS
    REFERENCE_VECTORS_NUMPY, REFERENCE_VECTORS_NUMPY_HEADER = make_numpy_object_and_header_from_DF(REFERENCE_VECTORS_DF)


    del DATA_DF, REFERENCE_VECTORS_DF, RAW_TARGET_DF

    print(f'\nNUMPY SOURCE OBJECTS SUCCESSFULLY CREATED\n')

    # END CONVERT SOURCE OBJECTS FROM DF TO NUMPY ############################################################################

    # CREATE A STANDARDIZED TUPLE FOR CONVENIENCE OF TAKING 6 VARIABLES THRU THE "BASE" BUILD FORMULAS
    SUPER_RAW_NUMPY_TUPLE = tuple((DATA_NUMPY, DATA_NUMPY_HEADER, RAW_TARGET_NUMPY, RAW_TARGET_NUMPY_HEADER,
                        REFERENCE_VECTORS_NUMPY, REFERENCE_VECTORS_NUMPY_HEADER))

    SXNL_DICT = dict(((0,'DATA'),(1,'TARGET'),(2,'REFVECS')))

    data_given_orientation = 'COLUMN'
    target_given_orientation = 'COLUMN'
    refvecs_given_orientation = 'COLUMN'

    # BEAR THIS vvv COMES OUT WHEN PreRunFilter IS FULLY CONVERTED TO NEW OBJS ##############################################################
    OLD_STYLE_SXNL_DICT = dict((zip(map(str, range(6)), ('DATA', 'DATA_HEADER', 'TARGET', 'TARGET_HEADER', 'REFVECS', 'REFVECS_HEADER'))))
    # END BEAR ^^^ THIS COMES OUT WHEN PreRunFilter IS FULLY CONVERTED TO NEW OBJS ##############################################################

    # *************************************************************************************************************
    # BUILD BASE OBJECTS ********************************************************************************************
    # THIS TAKES THE RAW NUMPY OBJECTS AND DOES ANY SPECIAL OPERATIONS (VLOOKUPS, ETC, AND
    # MANAGES ANY COINCIDENTAL ROW/COLUMN ADDONS AND/OR CHOPS)

    # THINGS TO REMEMBER: AS OF 10-31-21, SUPER_TUPLE IS NOT BEING TAKEN DOWN AS A WHOLE INTO THE BELLY
    # OF EACH OF THE 4 SEPARATE BUILD STRUCTURES, IT IS BEING PARTED OUT IN THE BaseRawXBuild CLASS TEMPLATE
    # AND ONLY EACH BUILD'S RESPECTIVE OBJECT & HEADER
    # ARE BEING TAKEN IN, THE IMPLICATIONS BEING THAT IF IN THE FUTURE THERE ARE BUILD INSTRUCTIONS THAT
    # REQUIRE MODS TO MULTIPLE OBJECTS AT ONCE, E.G. ROW CHOPS, THEN THIS WILL HAVE TO BE CHANGED TO TAKE
    # SUPER_TUPLE DOWN INTO EACH OF THEM.
    # BBM BUILD IS PRODUCING A "KEEP" OBJECT THAT HOLDS INFO ABT THE ATTRS RETAINED IN BBM (AT THIS POINT IT
    # IT IS EQUAL TO DATA_HEADER) THE CLASSES USED TO DO BBM BUILD ARE THE PARENTS FOR CLASSES THAT BUILD THE OTHER
    # 3 OBJECTS, SO THERE ARE 'DUMS' ALL OVER THE OTHER 3 CLASSES TO PLACEHOLD FOR THIS "KEEP" OBJECT -- THIS IS
    # REQUIRED ONLY TO SATISFY THE LOGIC OF THE PARENT CLASS.  THIS DOES, HOWEVER, EASILY ENABLE THE USE OF A
    # "KEEP"-LIKE OBJECT IN THE FUTURE FOR THE 3 OTHER OBJECTS

    SUPER_RAW_NUMPY_TUPLE, KEEP = \
        bob.BBMBuild(standard_config, BBM_manual_config, BBM_build_method, SUPER_RAW_NUMPY_TUPLE).build()

    pocs.print_object_create_success(standard_config, 'BASE BIG MATRIX')

    SUPER_RAW_NUMPY_TUPLE, DUM = bob.BaseRawTargetBuild(standard_config,
        base_raw_target_manual_config, base_raw_target_build_method, SUPER_RAW_NUMPY_TUPLE).build()

    pocs.print_object_create_success(standard_config, 'BASE RAW TARGET VECTOR')

    SUPER_RAW_NUMPY_TUPLE, DUM = bob.BaseRVBuild(standard_config,
        base_rv_manual_config, base_rv_build_method, SUPER_RAW_NUMPY_TUPLE).build()

    pocs.print_object_create_success(standard_config, 'BASE REFERENCE VECTORS')

    # END BUILD BASE OBJECTS ******************************************************************************************

    # NOW THAT DIPPING & DIVING OF TUPLE IN THE BUILD MODULES IS DONE, UNPACK AS A LIST
    SUPER_RAW_NUMPY_LIST = list(SUPER_RAW_NUMPY_TUPLE)

    # CREATE IDX VECTOR, TO ATTACH TO REFERENCE SO THAT FILTERING CAN BE DONE ON ANY OBJECT W THE INDEX
    # INSERT IDX VECTOR INTO 0 IDX OF REFERENCE_VECTOR_NUMPY & MATCHING PLACE IN HEADER
    SUPER_RAW_NUMPY_LIST[4] = np.insert(SUPER_RAW_NUMPY_LIST[4], 0, range(gs.get_shape('DATA', SUPER_RAW_NUMPY_LIST[0], data_given_orientation)[0]), axis=0)
    SUPER_RAW_NUMPY_LIST[5] = np.insert(SUPER_RAW_NUMPY_LIST[5], 0, 'IDX', axis=1)
    if SUPER_RAW_NUMPY_LIST[4][1][0] == 'DUM':  # IF RV WAS A DUMMY TO START, GET RID OF DUM
        SUPER_RAW_NUMPY_LIST[4] = np.delete(SUPER_RAW_NUMPY_LIST[4], 1, axis=0)
        SUPER_RAW_NUMPY_LIST[5] = np.delete(SUPER_RAW_NUMPY_LIST[5], 1, axis=1)


    # BEAR KEEP IS CURRENTLY [[]], CONVERT TO NEW FORMAT ([])
    KEEP = KEEP[0]


    # BEAR THIS vvv MUST STAY UNTIL THE ENTIRE FRONT END IS CONVERTED TO NEW SUPOBJS #######################################
    SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS = old_objs_to_new(SUPER_RAW_NUMPY_LIST, None, None, None, None, None, None, None, None)
    # MUST PUT ACTUAL VALUES INTO SUPOBJS TO AVOID BLOWUPS FROM WRONG CHAR TYPES

    SUPOBJS = ('MINCUTOFFS', 'USEOTHER', 'STARTLAG', 'ENDLAG', 'SCALING')
    DEFAULTS = (0, 'N', 0, 0, '')
    for so_idx in range(len(RAW_SUPOBJS)):
        # DONT DO FILTERING, LET PreRunFilter GET VDTYPES AND MDTYPES
        for SUPOBJ, default in zip(SUPOBJS, DEFAULTS):
            RAW_SUPOBJS[so_idx][msod.QUICK_POSN_DICT()[SUPOBJ]] = \
                list(map(lambda x: default, RAW_SUPOBJS[so_idx][msod.QUICK_POSN_DICT()[SUPOBJ]]))

    del SUPOBJS, DEFAULTS
    # END BEAR THIS ^^^ MUST STAY UNTIL THE ENTIRE FRONT END IS CONVERTED TO NEW SUPOBJS #######################################


    # 1-9-22 SET GLOBAL BACKUPS (VALIDATED_DATATYPES & MODIFIED_DATATYPES GLOBAL BACKUP SET LATER AT FIRST DECLARATION
    # DURING FIRST PASS OF PreRunFilter)
    print(f'\nCreating global backups of raw objects...')

    SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP = [_.copy() for _ in SUPER_RAW_NUMPY_LIST]
    FULL_SUPOBJS_GLOBAL_BACKUP = [_.copy() for _ in RAW_SUPOBJS]
    KEEP_GLOBAL_BACKUP = deepcopy(KEEP)
    print(f'Done.')




    ####################################################################################################################
    # FILTERING ############################################################################################################

    # 12-22-21 11:12 AM NOT SURE IF THIS STUFF IS NEEDED #######################################
    filter_method = ''
    filter_pre_run_manual_config = 'BYPASS'
    # END NOT SURE STUFF #######################################################################

    # 12-22-21
    # InSitu HAS TO INGEST V_TYPES & M_TYPES, FILTER, MIN_CUTOFF, USE_OTHER, ALL BASE & GLOBAL BACKUPS, SO HAVE TO
    # MAKE PreRun INGEST THEM ALSO
    # PreRun INGESTS SRNL & KEEP GLOBAL BACKUP, BUT NONE OF THE OTHERS EXIST YET, SO INGEST DUMS
    # InSitu INGESTS ALL GLOBAL AND BASE_BACKUP OBJS THAT CAME OUT OF PreRun,
    # THE SAME SRNL AND KEEP GLOBAL_BACKUPS ARE INGESTED BY BOTH PreRun & InSitu

    SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, CONTEXT, KEEP, VALIDATED_DATATYPES_GLOBAL_BACKUP, MODIFIED_DATATYPES_GLOBAL_BACKUP = \
        prf.PreRunFilter(
                         standard_config,
                         filter_pre_run_manual_config,
                         filter_method,
                         SUPER_RAW_NUMPY_LIST,
                         data_given_orientation,
                         target_given_orientation,
                         refvecs_given_orientation,
                         RAW_SUPOBJS,
                         [],  # DUM CONTEXT
                         KEEP,
                         *range(4),  # PLACEHOLDERS FOR BASE OBJECTS FOR PreRun (NOT CREATED YET)
                         SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP,
                         FULL_SUPOBJS_GLOBAL_BACKUP,
                         KEEP_GLOBAL_BACKUP,
                         bypass_validation=True
    ).config_run()

    # END FILTERING ####################################################################################################
    ####################################################################################################################


    ####################################################################################################################
    # TEXT CLEANER #####################################################################################################
    ALL_MOD_DTYPES = np.hstack(([RAW_SUPOBJS[_][msod.QUICK_POSN_DICT()["MODIFIEDDATATYPES"]] for _ in range(len(RAW_SUPOBJS))]))

    if True in map(lambda x: x in msod.mod_text_dtypes().values(), ALL_MOD_DTYPES) and \
        vui.validate_user_str(f'\nClean any text? (y/n) > ', 'YN')=='Y':

        while True:

            for idx, _OBJ in enumerate(SUPER_RAW_NUMPY_LIST):
                print(f"{SXNL_DICT[idx]}:")
                sop.GeneralSmallObjectPreview(_OBJ,'COLUMN',RAW_SUPOBJS[idx],'HEADER')
                print()

            while True:
                obj_idx = vui.validate_user_int(f'Select {", ".join([f"{v}({k})" for k,v in SXNL_DICT.items()])} > ', min=0, max=2)
                if vui.validate_user_str(f"User selected {SXNL_DICT[obj_idx]}, accept? (y/n) > ", 'YN')=='Y':break

            while True:
                col_idx = vui.validate_user_int(f'Enter column index > ', min=0,
                    max=gs.get_shape(SXNL_DICT[obj_idx], SUPER_RAW_NUMPY_LIST[obj_idx],
                                     {0:data_given_orientation, 1:target_given_orientation, 2:refvecs_given_orientation}[obj_idx])[1])
                if vui.validate_user_str(f'User selected {RAW_SUPOBJS[obj_idx][msod.QUICK_POSN_DICT()["HEADER"]][col_idx]}, accept? (y/n) > ', 'YN')=='Y': break

            CLEANED_TEXT = tc.TextCleaner(SUPER_RAW_NUMPY_LIST[obj_idx][col_idx],
                                           update_lexicon=True,
                                           auto_add=False,
                                           auto_delete=False
            ).menu(disallowed='D')


            if vui.validate_user_str(f'\nOverwrite original column with cleaned text? (y/n) > ', 'YN') == 'Y':
                SUPER_RAW_NUMPY_LIST[obj_idx][col_idx] = CLEANED_TEXT

            del obj_idx, col_idx, CLEANED_TEXT

            if vui.validate_user_str(f'Clean another column? (y/n) > ', 'YN') == 'N': break

    del ALL_MOD_DTYPES

    # END TEXT CLEANER ############################################################################################################
    ####################################################################################################################

    #######################################################################################################################
    # GIVE USER OPTION TO SAVE CLEANED FILE ###############################################################################
    if vui.validate_user_str(f'\nSave cleaned RAW DATA to file? (y/n) > ', 'YN') == 'Y':

        base_path = bps.base_path_select()

        file_name = fe.filename_wo_extension()

        _ext = vui.validate_user_str(f'Select file type -- csv(c) excel(e) > ', 'CE')

        full_path = base_path + file_name + {'C':'.csv', 'E':'.xlsx'}[_ext]

        print(f'\nWorking on it...')

        if _ext == 'E':
            OBJECTS = SUPER_RAW_NUMPY_LIST
            HEADERS = [RAW_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]].reshape((1,-1)),
                       RAW_SUPOBJS[1][msod.QUICK_POSN_DICT()["HEADER"]].reshape((1,-1)),
                       RAW_SUPOBJS[2][msod.QUICK_POSN_DICT()["HEADER"]].reshape((1,-1))
                       ]
            SHEET_NAMES = [f'DATA', f'TARGET', f'REFVECS']

            with pd.ExcelWriter(full_path) as writer:
                for idx in range(len(SHEET_NAMES)):

                    DF = pd.DataFrame(OBJECTS[idx].transpose())

                    try:
                        DF.to_excel(excel_writer=writer,
                                    sheet_name=SHEET_NAMES[idx],
                                    header=True,
                                    index=False
                                    )
                    except:
                        # IF EXCEPTION, SHOW ON FILE SHEET
                        pd.DataFrame([[f'*** ERROR WRITING {SHEET_NAMES[idx]} TO FILE ***']]).to_excel(
                            excel_writer=writer,
                            sheet_name=SHEET_NAMES[idx],
                            header=False,
                            index=False
                            )

        elif _ext == 'C':
            pd.DataFrame(
                np.vstack((SUPER_RAW_NUMPY_LIST)).transpose(),
            ).to_csv(full_path,
                     header=np.hstack((RAW_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]],
                                       RAW_SUPOBJS[1][msod.QUICK_POSN_DICT()["HEADER"]],
                                       RAW_SUPOBJS[2][msod.QUICK_POSN_DICT()["HEADER"]]))
                     )

        print(f'Done.\n')
    # END GIVE USER OPTION TO SAVE CLEANED FILE ###########################################################################
    #######################################################################################################################

    if vui.validate_user_str(f'\nKeep GLOBAL_BACKUP objects? (y/n) > ', 'YN') == 'N':
        # KEEP THESE OBJECTS TO PRESERVE FXN ARGUMENT INPUTS
        SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP = []
        KEEP_GLOBAL_BACKUP = []
        VALIDATED_DATATYPES_GLOBAL_BACKUP = []
        MODIFIED_DATATYPES_GLOBAL_BACKUP = []


    # 12-6-2021 7:33 AM BASE_BACKUP OBJECTS THAT WILL BE NEEDED FOR InSituFilter, BEING CREATED AFTER PreRunFilter
    if vui.validate_user_str(f'\nGenerate BASE_BACKUP objects? (y/n) > ', 'YN') == 'Y':
        print(f'Building BASE_BACKUP objects as copies of RAW OBJECTS...')
        FULL_SUPOBJS_BASE_BACKUP = deepcopy(RAW_SUPOBJS)
        CONTEXT_BASE_BACKUP = deepcopy(CONTEXT)
        KEEP_BASE_BACKUP = deepcopy(KEEP)
        SUPER_RAW_NUMPY_LIST_BASE_BACKUP = [_.copy() for _ in SUPER_RAW_NUMPY_LIST] # AVOID DEEPCOPY OF BIG NUMPYS IN SRNL
        print(f'Done.')
    else:
        FULL_SUPOBJS_BASE_BACKUP = []
        CONTEXT_BASE_BACKUP = []
        KEEP_BASE_BACKUP = []
        SUPER_RAW_NUMPY_LIST_BASE_BACKUP = []


    # CREATE SUPER_WORKING_NUMPY_LIST

    user_manual_or_standard = 'N'
    expand_method = ''   # 1-12-2022 BEAR NOT SURE WHAT FUNCTION THIS IS PLAYING IN ExpandCategories
    augment_method = ''

    #DECLARE RESULTS DISPLAY INITIAL CONDITIONS IN THIS SCOPE
    CSUTM_DF, display_items, display_rows = '', '', 10

    #DECLARE TARGET_VECTOR SETUP INITIAL CONDITIONS IN THIS SCOPE
    split_method = ''
    LABEL_RULES = []
    number_of_labels = 1
    event_value = 0
    negative_value = 0


    SUPER_WORKING_NUMPY_LIST = [[[]]]  # 1-9-22 SET INITIAL STATE OF SWNL HERE B4 LOOP.  SWNL MUST EXIST ON 1ST PASS TO
    # SATISFY CODE AT TOP OF LOOP THAT DOES len CHECK TO SEE IF len(DATA) CHANGED (THUS ON ALL PASSES AFTER FIRST PASS
    # WOULD REQUIRE ALL OBJECTS TO CHANGE)
    # ON FIRST PASS, SWNL (DATA & TARGET) IS REQUIRED TO BE PUT INTO CALCULABLE STATES
    # FOR FILTERING, WHERE SRNL IS ACTED ON, MUST INGEST SRNL & RETURN NEW SRNL & NEW SWNL (AFTER WHICH CODE REQUIRES
    # PROCESSING OF DATA & TARGET INTO CALCULABLE STATES & GIVES OPTION FOR REF_VECS).
    # 1-20-22 DONT HAVE TO SET INITIAL STATES OF ANY OTHER WORKING OBJECTS HERE, THEY ARE AUTOMATICALLY CREATED ON 1ST
    # PASS AFTER OPTIONAL FILTERING STEP

    # DECLARE MLR INITIAL CONDITIONS IN THIS SCOPE
    from ML_PACKAGE.MLREGRESSION import mlr_default_config_params as mlrdcp
    MLR_OUTPUT_VECTOR, mlr_batch_method, mlr_batch_size, mlr_rglztn_type, mlr_rglztn_fctr = \
        mlrdcp.mlr_default_config_params()

    # DECLARE GMLR INITIAL CONDITIONS IN THIS SCOPE
    from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION import gmlr_default_config_params as gdcp
    gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
    gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg, GMLR_OUTPUT_VECTOR = \
        gdcp.gmlr_default_config_params()

    # DECLARE MI INITIAL CONDITIONS IN THIS SCOPE
    from ML_PACKAGE.MUTUAL_INFORMATION import mi_default_config_params as midcp
    MI_OUTPUT_VECTOR, mi_batch_method, mi_batch_size, mi_int_or_bin_only, mi_max_columns, mi_bypass_agg = \
        midcp.mi_default_config_params()

    # DECLARE SVD INITIAL CONDITIONS IN THIS SCOPE
    from ML_PACKAGE.SVD import svd_default_config_params as sdcp
    svd_max_columns, SVD_OUTPUT_VECTOR = sdcp.svd_default_config_params()

    # DECLARE NN INITIAL CONDITIONS IN THIS SCOPE
    from ML_PACKAGE.NN_PACKAGE import nn_default_config_params as ndcp
    ARRAY_OF_NODES, NEURONS, nodes, node_seed, activation_constant, aon_base_path, aon_filename, nn_cost_fxn, \
    SELECT_LINK_FXN, LIST_OF_NN_ELEMENTS, NN_OUTPUT_VECTOR, batch_method, BATCH_SIZE, gd_method, conv_method, \
    lr_method, gd_iterations, LEARNING_RATE, momentum_weight, nn_conv_kill, nn_pct_change, nn_conv_end_method, \
    nn_rglztn_type, nn_rglztn_fctr, non_neg_coeffs, allow_summary_print, summary_print_interval, iteration = \
        ndcp.nn_default_config_params()

    # DECLARE SVM INITIAL CONDITIONS IN THIS SCOPE
    from ML_PACKAGE.SVM_PACKAGE import svm_default_config_params as svmdcp
    margin_type, C, svm_cost_fxn, kernel_fxn, constant, exponent, sigma, K, ALPHAS, b, alpha_seed, alpha_selection_alg, \
    max_passes, tol, svm_conv_kill, svm_pct_change, svm_conv_end_method, alpha_selection_alg, SMO_a2_selection_method = \
        svmdcp.svm_default_config_params()



    ################################################################################################################################
    # ML IN-SITU MENU STATIC INPUTS ################################################################################################

    # USED ABCDEFGHIJKLMNOPQRSTUVWXYZ -- 9/23/22 USED ALL LETTERS

    # DONT PUT Y OR Z IN ANY OF THESE!!!  Y AND Z ARE USED TO MANAGE DIFFENCES IN HANDLING BETWEEN 1ST AND SUBSEQUENT PASSES!
    ML_INSITU_BASE_CMDS = {
        'f': 'configure BASE filtering',
        '1': 'filter current WORKING DATA object',
        'r': 'generate WORKING DATA from BASE',
        't': 'generate WORKING TARGET from BASE',
        'u': 'generate WORKING REFERENCE VECTORS from BASE',
        'a': 'dump current base data objects to file',
        'x': 'go back to in-situ data augment',
        'q': 'quit'
    }

    #   BEAR 7:21 PM 4/10/23.... WHAT IS THE DIFFERENCE BETWEEN THESE?
    #   generate new WORKING TARGET VECTOR(j)
    #   generate WORKING TARGET from BASE(t)
    ML_INSITU_WORKING_CMDS = {
        'b': 'restore WORKING objects to as-created state',
        'p': 'print WORKING object headers',
        'i': 'append intercept to WORKING DATA',
        'h': 'delete intercept from WORKING DATA',
        'c': 'move a column within or between objects',
        'j': 'generate new WORKING TARGET VECTOR',
        'k': 'float check',
        'd': 'dump current data objects to file',
        'w': f'convert DATA to {["NUMPY" if isinstance(SUPER_WORKING_NUMPY_LIST[0], dict) else "SPARSE_DICT"][0]}'
    }

    ML_ALGORITMS = {
        'g': 'run GDA',
        'l': 'run MLRegression',
        'm': 'run greedy MLR',
        'o': 'run Mutual Information',
        'v': 'run SVD',
        'n': 'run NN',
        's': 'run SVM'
    }

    ml_insitu_base_str = "".join(list(ML_INSITU_BASE_CMDS.keys())).upper()  # 'FRTUEAXQ'
    ml_insitu_working_str = "".join(list(ML_INSITU_WORKING_CMDS.keys())).upper()  # 'BPIHCJKDW'
    ml_algorithms_str = "".join(list(ML_ALGORITMS.keys())).upper()  # 'GLMNOSV'
    # OTHER USED CHARS = 'ZY'  OBJECT SETUP Z=COMPULSORY/OPTIONAL ON FIRST PASS,
    #                                       Y=COMPULSORY/OPTIONAL AFTER FILTER ON PASS >1,
    #             (Z IS SET EXTERNAL PRIOR TO LOOP, Y DETERMINED AFTER USER SELECTS 'F')

    menu_max_len = max(map(len, dict((ML_INSITU_BASE_CMDS | ML_INSITU_WORKING_CMDS | ML_ALGORITMS)).values())) + 3  # +3 FOR (x)

    user_ML_select = 'Z'  # SET INITIAL STATE TO FORCE SETUP OF OBJECTS ON FIRST PASS, 'Z' ONLY ACCESSIBLE ON FIRST PASS

    # END ML IN-SITU MENU STATIC INPUTS ############################################################################################
    ################################################################################################################################

    print(f'\n********** ENTERING INSITU DATA HANDLING AND ANALYSIS **********')

    while True:

        user_filtering_select = 'DUM'
        user_target_vector_select = 'DUM'
        user_ref_vecs_select = 'DUM'

        ###################################################################################################################
        ###################################################################################################################
        # OBJECT SET UP ###################################################################################################

        # THESE ARE FOR REFERENCE, TO SEE IF USER HAS MADE FILTERING CHANGES THAT WOULD / WOULDNT REQUIRE CHANGES TO
        # OTHER OBJECTS

        if user_ML_select in 'Z':  # THIS IS HERE SO THAT USER HAS OPTION TO RECONFIG FILTERING ON FIRST PASS
            if vui.validate_user_str('\nModify FILTERING? (y/n) > ', 'YN') == 'Y':
                # BEAR, WHEN 'Y' WS SELECTED HERE IT EXCEPTED FOR "local variable 'WORKING_SUPOBJS' referenced before assignment"
                # AT LINE 829 IN TargetInSituHandling
                user_filtering_select = 'YES'
            else:  # ON FIRST PASS (Z), THESE OBJECTS WOULD BE CREATED IF FILTERING WAS SELECTED, BUT IF NOT FILTERING
                   # THESE OBJECTS MUST BE CREATED BEFORE PROCEEDING TO WORKING OBJECT CONFIGS
                print(f'\nBuilding WORKING objects as copies of RAW objects....')
                SUPER_WORKING_NUMPY_LIST = [_.copy() for _ in SUPER_RAW_NUMPY_LIST]  # AVOID deepcopy OF HUGE NUMPY OBJECTS
                WORKING_SUPOBJS = deepcopy(RAW_SUPOBJS)
                WORKING_CONTEXT = deepcopy(CONTEXT)
                WORKING_KEEP = deepcopy(KEEP)
                print('Done.')

        if user_ML_select == 'F':     #  'configure BASE filtering(f)'
            if vui.validate_user_str(f'\n*** CHANGING FILTERING OF ANY BASE OBJECT WILL FORCE RESET OF AND RECREATION OF '
                f'ALL WORKING OBJECTS!  PROCEED? (y/n) > ', 'YN') == 'N': user_ML_select = 'BYPASS'


        if user_ML_select in 'FRTU':
            if vui.validate_user_str(f'\n*** GENERATING ANY WORKING OBJECT FROM BASE REQUIRES RESET OF WORKING OBJECTS '
                f'\nTO AS-CREATED STATE!  THIS IS BECAUSE ANY INSITU ROW INSERT / DELETE ON WORKING OBJECTS CAUSES '
                f'\nINCONGRUITY WITH THE ORIGINAL BASE OBJECTS.  THEREFORE ALL WORKING OBJECTS MUST BE RESTORED TO STATE OF '
                f'\nCONGRUITY WITH BASE OBJECTS SO THAT NEWLY CREATED WORKING OBJECT IS ROW-HARMONIOUS WITH THE OTHER WORKING '
                f'\nOBJECTS.  PROCEED?  (y/n) > ', 'YN') == 'N': user_ML_select = 'BYPASS'
            # 1-28-22 OTHERWISE TO GET HARMONY A PSEUDO-BASE OBJECT WOULD HAVE TO BE GENERATED FROM BASE OBJECT USING
            # SOME KIND OF INDEXING OF THE WORKING OBJECT. (ROW ID INDEX WITHIN REF_VECS WOULD NO LONGER APPLY IF ANY
            # CHOPPING OF ROWS OCCURED IN FILTERING. THE ONLY OTHER OPTION WOULD TO BE CREATE A NEW ROW ID INDEX AFTER
            # FILTERING, TO USE FOR THESE RECONSTRUCTION PURPOSES.  FORGET IT!)  AFTER CONSTRUCTION OF THE NEW PSEUDO-
            # BASE OBJECT, IT COULD THEN BE PROCESSED ACCORDING TO CONFIG METHOD FOR THAT OBJECT.  FORGET IT!

            else:  # IF USER IS GOING TO REGENERATE A WORKING OBJECT FROM BASE MUST RESTORE WORKING
                # OBJECTS BACK TO AS-CREATED STATE
                SUPER_WORKING_NUMPY_LIST = [_.copy() if isinstance(_, np.ndarray) else deepcopy(_) for _ in SUPER_WORKING_NUMPY_LIST_BACKUP]
                WORKING_SUPOBJS = deepcopy(WORKING_SUPOBJS_BACKUP)
                WORKING_CONTEXT = deepcopy(WORKING_CONTEXT_BACKUP)
                WORKING_KEEP = deepcopy(WORKING_KEEP_BACKUP)


        if user_ML_select in 'F1' or user_filtering_select in 'YES':
            user_filtering_select = 'DUM'  # RESET SO IT'S NOT PICKED UP ON NEXT PASS THRU LOOP
            start_reference_srnl_rows, start_reference_srnl_cols = \
                gs.get_shape('DATA', SUPER_RAW_NUMPY_LIST[0], data_given_orientation)
            reference_swnl_rows, reference_swnl_columns = \
                gs.get_shape('DATA', SUPER_WORKING_NUMPY_LIST[0], data_given_orientation)

            while True:

                if user_ML_select == 'F' and len(SUPER_RAW_NUMPY_LIST_BASE_BACKUP)==0:
                    print(f'\n*** BASE BACKUP OBJECTS WERE NOT CREATED SO CANNOT RECONFIGURE BASE FILTERING ***')
                    break

                if user_ML_select == 'F':
                    SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, CONTEXT, KEEP = \
                                    isf.InSituFilter(
                                                        standard_config,
                                                        filter_pre_run_manual_config,
                                                        filter_method,
                                                        SUPER_RAW_NUMPY_LIST_BASE_BACKUP,
                                                        data_given_orientation,
                                                        target_given_orientation,
                                                        refvecs_given_orientation,
                                                        FULL_SUPOBJS_BASE_BACKUP,
                                                        CONTEXT_BASE_BACKUP,
                                                        KEEP_BASE_BACKUP,
                                                        SUPER_RAW_NUMPY_LIST_BASE_BACKUP,
                                                        FULL_SUPOBJS_BASE_BACKUP,
                                                        CONTEXT_BASE_BACKUP,
                                                        KEEP_BASE_BACKUP,
                                                        SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP,
                                                        FULL_SUPOBJS_GLOBAL_BACKUP,
                                                        KEEP_GLOBAL_BACKUP,
                                                        bypass_validation=True
                                    ).config_run()


                elif user_ML_select == '1':
                    SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, CONTEXT, KEEP = \
                                    isf.InSituFilter(
                                                        standard_config,
                                                        filter_pre_run_manual_config,
                                                        filter_method,
                                                        SUPER_WORKING_NUMPY_LIST,
                                                        data_given_orientation,
                                                        target_given_orientation,
                                                        refvecs_given_orientation,
                                                        WORKING_SUPOBJS,
                                                        CONTEXT,
                                                        KEEP,
                                                        SUPER_RAW_NUMPY_LIST_BASE_BACKUP,
                                                        FULL_SUPOBJS_BASE_BACKUP,
                                                        CONTEXT_BASE_BACKUP,
                                                        KEEP_BASE_BACKUP,
                                                        SUPER_RAW_NUMPY_LIST_GLOBAL_BACKUP,
                                                        FULL_SUPOBJS_GLOBAL_BACKUP,
                                                        KEEP_GLOBAL_BACKUP,
                                                        bypass_validation=True
                    ).config_run()


                # MAKE POST FILTERING DIMENSIONAL REFERENCE
                end_reference_srnl_rows, end_reference_srnl_cols = \
                    gs.get_shape('DATA', SUPER_RAW_NUMPY_LIST[0], data_given_orientation)

                # AT FIRST PASS OR ANY OTHER TIME RESET/FILTERING IS CHOSEN, ITS OK TO AUTOMATICALLY UPDATE THESE OBJECTS HERE
                # WITHOUT INCONVENIENCING USER BECAUSE ARDUOUS OBJECT SETUPS ARE NOT NECESSARY:
                # WORKING_SUPOBJS = deepcopy(RAW_SUPOBJS)   # BEAR HASHED THIS OUT 4/26/23

                if user_ML_select in 'YZ':   # IF FIRST PASS, GO STRAIGHT TO REQUIRED / OPTIONAL OBJECT BUILDS
                    # SUPER_WORKING_NUMPY_LIST, WORKING_VALIDATED_DATATYPES, WORKING_MODIFIED_DATATYPES ARE CREATED HERE,
                    # THEN OVERWRITTEN BY BIG_MATRIX, TARGET, & REF VEC CONFIG

                    # SUPER_WORKING_NUMPY_LIST_BACKUP, VALIDATED_DATATYPES_BACKUP, MODIFIED_DATATYPES_BACKUP, KEEP_BACKUP,
                    # AND CONTEXT_BACKUP ARE CREATED AFTER COMPLETION OF ALL OBJECT SETUPS

                    # BEAR HASHED SWNL & WORKING_SUPOBJS 4/26/23
                    # SUPER_WORKING_NUMPY_LIST = [_.copy() for _ in SUPER_RAW_NUMPY_LIST]
                    # WORKING_SUPOBJS = deepcopy(RAW_SUPOBJS)
                    WORKING_CONTEXT = deepcopy(CONTEXT)
                    WORKING_KEEP = deepcopy(KEEP)
                    break   # user_ML_select STAYS 'Y' or 'Z'

                elif user_ML_select == 'F':  # 1-20-22 DONT PUT elifs IN HERE!
                    # OTHERWISE, WORKING OBJECTS EXIST, INFORM USER OF THE CONSEQUENCES OF FILTERING AND GIVE OPTIONS
                    # DONT CREATE/MODIFY WORKING OBJECTS HERE, THEY ALREADY EXIST, CHANGES TO THEM WILL BE HANDLED BY
                    # COMPULSORY / OPTIONAL STEPS

                    # USER WAS IN 'F', BUT COLUMNS NOR ROWS DID NOT CHANGE, HIGHLY LIKELY USER JUST ABORTED OUT (THE IS NOT FOOL PROOF)
                    # SO INSTEAD OF REQUIRING / OFFERING OPTIONS TO MODIFY OBJECTS, JUST BYPASS BACK TO MENU
                    # if end_reference_srnl_cols == start_reference_srnl_cols and \
                    #         end_reference_srnl_rows == start_reference_srnl_rows:
                    #     user_ML_select = 'BYPASS'
                    #     break

                    # NO. COLUMNS IN DATA FILTERED, BIG_MATRIX_SETUP IS COMPULSORY, TARGET, & REFVECS ARE OPTIONAL
                    # if end_reference_srnl_cols != start_reference_srnl_cols: user_ML_select = 'Y'


                    # NO. ROWS IN DATA WERE FILTERED, BIG_MATRIX SETUP AND TARGET SETUP ARE COMPULSORY, REFVECS ARE OPTIONAL, NO OTHER REQUIRED CHANGES
                    # if end_reference_srnl_rows != start_reference_srnl_rows: user_ML_select = 'Z'

                    user_ML_select == 'Y'

                elif user_ML_select == '1': break


        # END HANDLING OF OBJECTS ON FIRST PASS, AFTER CHANGES TO FILTERING, ETC #######################################
        ################################################################################################################
        ################################################################################################################

        # 1-21-2022 SET UP TARGET FIRST SO ITS AVAILABLE FOR STATS IN DATA MATRIX SET UP ###############################
        # TARGET VECTOR SETUP ##########################################################################################
        # TARGET VECTOR B4 DATA BECAUSE COMPLETED TARGET IS NEEDED TO CALCULATE STATISTICS DURING DATA SETUP
        if user_ML_select in 'Y':  # THIS IS HERE SO THAT USER HAS OPTION TO CONFIG TARGET_VEC IF UPON FILTERING # COLUMNS CHANGED BUT NOT # ROWS
            if vui.validate_user_str('\nConfigure TARGET_VECTOR? (y/n) > ', 'YN') == 'Y':
                user_target_vector_select = 'YES'

        if user_ML_select in 'TZ' or user_target_vector_select == 'YES':  # TARGET_VECTOR MUST BE PUT INTO CALCULABLE
            user_target_vector_select = 'DUM'                       # STATE ON 1ST PASS OR MAYBE IF InSituFiltering IS DONE
            target_config = 'Z'

            SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, split_method, LABEL_RULES, number_of_labels, event_value, negative_value = \
            tish.TargetInSituHandling(standard_config, target_config, SUPER_RAW_NUMPY_LIST, SUPER_WORKING_NUMPY_LIST,
                WORKING_SUPOBJS, split_method, LABEL_RULES, number_of_labels, event_value, negative_value).run()

        # END TARGET VECTOR SETUP ##################################################################################



        if user_ML_select in 'RYZ':  # BIG MATRIX MUST BE PUT INTO CALCULABLE STATE ON 1ST PASS OR IF InSituFiltering IS DONE
            while True:

                SUPER_WORKING_NUMPY_LIST_HOLDER, WORKING_SUPOBJS_HOLDER, CONTEXT_HOLDER, KEEP_HOLDER = \
                    isec.InSituExpandCategories(
                                                 standard_config,
                                                 user_manual_or_standard,
                                                 expand_method,
                                                 [SUPER_RAW_NUMPY_LIST[0], *SUPER_WORKING_NUMPY_LIST[1:]],  # MUST INGEST PROCESSED TARGET THINGS FOR STATISTICS CALCS IN HERE
                                                 data_given_orientation,
                                                 target_given_orientation,
                                                 refvecs_given_orientation,
                                                 [RAW_SUPOBJS[0], *WORKING_SUPOBJS[1:]], # BUTCHERED SUPOBJS TO ALLOW FOR RECONSTRUCION FROM BASE, WHILE PRESERVING ANY PREVIOUS CHANGES TO OTHER OBJECTS
                                                 CONTEXT,
                                                 KEEP,
                                                 bypass_validation=True
                    ).config_run()


                SUPER_WORKING_NUMPY_LIST_HOLDER, WORKING_SUPOBJS, CONTEXT_HOLDER, KEEP_HOLDER = \
                    isda.InSituDataAugment(
                                           standard_config,
                                           user_manual_or_standard,
                                           augment_method,
                                           SUPER_WORKING_NUMPY_LIST_HOLDER,
                                           data_given_orientation,
                                           target_given_orientation,
                                           refvecs_given_orientation,
                                           WORKING_SUPOBJS_HOLDER,
                                           CONTEXT_HOLDER,
                                           KEEP_HOLDER,
                                           bypass_validation=True
                    ).config_run()

                pocs.print_object_create_success(standard_config, 'WORKING DATA')

                if vui.validate_user_str(f'Accept WORKING DATA config? (y/n) (no restarts expand & augment process from RAW) > ', 'YN') == 'Y':
                    SUPER_WORKING_NUMPY_LIST = SUPER_WORKING_NUMPY_LIST_HOLDER
                    WORKING_SUPOBJS = WORKING_SUPOBJS_HOLDER
                    WORKING_CONTEXT = CONTEXT_HOLDER
                    WORKING_KEEP = KEEP_HOLDER

                    # EMPTY HOLDER OBJECTS
                    del SUPER_WORKING_NUMPY_LIST_HOLDER, WORKING_SUPOBJS_HOLDER, CONTEXT_HOLDER, KEEP_HOLDER
                    break

        # END BASE_BIG_MATRIX MODS #################################################################################

        # BUILD REFERENCE VECTORS *******************************************************************************************
        if user_ML_select in 'Y':  # THIS IS HERE SO THAT USER HAS OPTION TO CONFIG REF VECS ON FIRST PASS
            if vui.validate_user_str('\nConfigure REFERENCE_VECTORS? (y/n) > ', 'YN') == 'Y':
                user_ref_vecs_select = 'YES'

        if user_ML_select in 'UZ' or user_ref_vecs_select in 'YES':

            SUPER_WORKING_NUMPY_LIST[2] = SUPER_RAW_NUMPY_LIST[2]
            WORKING_SUPOBJS[2] = RAW_SUPOBJS[2]

            print(f'\nREFERENCE VECTORS AND HEADER SUCCESSFULLY UPDATED\n')

            user_ref_vecs_select = 'DUM' # RESET SO IT'S NOT PICKED UP ON NEXT PASS THRU LOOP

        # END BUILD REFERENCE VECTORS **************************************************************************************


        # CREATE BACKUPS OF WORKING OBJECTS IF USER SELECTED CMD FOR EDITING OBJECTS - ALLOWS USER TO MAKE EDITS TO
        # WORKING OBJECTS ON THE FLY DURING ANALYSIS
        # BUT THEN RETURN TO ORIGINALLY CREATED STATE WITHOUT HAVING TO GO THRU ENTIRE OBJECT CREATION AGAIN

        if user_ML_select in 'ZYRTUE1':    # 'F' CANT GET HERE; CREATE BACKUPS IMMEDIATELY AFTER GENERATION FROM BASE
            SUPER_WORKING_NUMPY_LIST_BACKUP  = [deepcopy(_) if isinstance(_, dict) else _.copy() for _ in SUPER_WORKING_NUMPY_LIST]
            WORKING_SUPOBJS_BACKUP  = deepcopy(WORKING_SUPOBJS)
            WORKING_CONTEXT_BACKUP = deepcopy(WORKING_CONTEXT)
            WORKING_KEEP_BACKUP = deepcopy(WORKING_KEEP)
            # OTHER OBJECTS WERE ALREADY BACKED UP ABOVE... BEAR 4/10/23, VERIFY THIS OR FIX

        # END OBJECT SETUP ################################################################################################
        ###################################################################################################################
        ###################################################################################################################


        if user_ML_select == 'B':  #  'restore WORKING objects to as-created state(b)'
            SUPER_WORKING_NUMPY_LIST = [deepcopy(_) if isinstance(_, dict) else _.copy() for _ in SUPER_WORKING_NUMPY_LIST_BACKUP]
            WORKING_SUPOBJS = deepcopy(WORKING_SUPOBJS_BACKUP)
            WORKING_CONTEXT = deepcopy(WORKING_CONTEXT_BACKUP)
            WORKING_KEEP = deepcopy(WORKING_KEEP_BACKUP)

        if user_ML_select == 'P':    #  'print WORKING object headers(p)'
            hdr_idx = msod.QUICK_POSN_DICT()["HEADER"]
            print(f'\nDATA:')
            [print(_) for _ in WORKING_SUPOBJS[0][hdr_idx]]
            print(f'\nTARGET:')
            [print(_) for _ in WORKING_SUPOBJS[1][hdr_idx]]
            print(f'\nREFERENCE VECTORS:')
            [print(_) for _ in WORKING_SUPOBJS[2][hdr_idx]]
            print()
            del hdr_idx


        if user_ML_select == 'I':   #  'append intercept to WORKING DATA(i)'
            pass

        if user_ML_select == 'H':    #  'delete intercept from WORKING DATA(h)'
            pass

        if user_ML_select == 'C':     #   'move a column within or between objects(c)'

            print(f'\n *** WORKING ON IT AS OF 3-22-22 ***\n')
            # BEAR FINISH
            '''
            SUPER_NUMPY_DICT = dict({0: 'DATA', 1: 'TARGET', 3: 'REFERENCE'})
            
            while True:

                for set_idx in range(len(SET_LIST[1:])):
                    if len(SET_LIST[set_idx]) != len(SET_LIST[0]):
                        empty_objects = True
                        print(f"{SET_DICT[set_idx]} OBJECTS ARE NOT FULL.")
                if empty_objects and vui.validate_user_str(f'Abort? (y/n) > ', 'YN') == 'Y':
                    break
    
                set_idx = \
                ls.list_single_select([*SET_DICT.values()], f'Select set of objects within which to move a column > ',
                                      'idx')[0]
                if set_idx == 0: xx = SUPER_WORKING_NUMPY_LIST
                elif set_idx == 1: xx = TRAIN_SWNL
                elif set_idx == 2: xx = DEV_SWNL
                elif set_idx == 3: xx = TEST_SWNL
    
                home_obj_idx = 2 * \
                               ls.list_single_select([SXNL_DICT.values()], f'\nSelect object to move column from ', 'idx')[0]
                home_obj_col_idx = ls.list_single_select(xx[home_obj_idx + 1][0], f'\nSelect column to move ', 'idx')[0]
                
                target_obj_idx = 2 * ls.list_single_select([SXNL_DICT.values()],
                    f'\nSelect object to move column to > ', 'idx')[0]
                COLUMN_LIST_FOR_TARGET = [xx[target_obj_idx + 1][0], 'END']  f'\nSelect column to insert column before ', 'idx')[0]
                print(
                    f'\n Selected to move {xx[home_obj_idx + 1][0][home_obj_col_idx]} from ' + \
                    f'{str([*SXNL_DICT.values()][home_obj_idx])} to {str([*SUPER_NUMPY_DICT.values()][target_obj_idx / 2])}' + \
                    f'before column "{COLUMN_LIST_FOR_TARGET[target_obj_col_idx]}"...')
                __ = vui.validate_user_str(f'Accept(a), try again(t), abort(b) > ', 'ATB')
                if __ == 'B':
                    break
                elif __ == 'A':
                    pass
                elif __ == 'T':
                    continue
    
                # PUT COPY OF HOME COLUMN INTO TARGET OBJECT
                xx[target_obj_idx] = np.insert(xx[target_obj_idx],
                                              deepcopy(xx[home_obj_idx][home_obj_col_idx]), target_obj_col_idx, axis=0)
    
                # PUT COPY OF HOME COLUMN NAME INTO TARGET OBJECT HEADER
                xx[target_obj_idx + 1] = np.insert(xx[target_obj_idx + 1],
                                                  deepcopy(xx[home_obj_idx + 1][0][home_obj_col_idx]), target_obj_col_idx,
                                                  axis=1)
    
                # 3-22-22 DO COLUMN JIVE AND DELETE ON THESE 5 TOO
                    WORKING_VALIDATED_DATATYPES = WORKING_VALIDATED_DATATYPES
                    WORKING_MODIFIED_DATATYPES = WORKING_MODIFIED_DATATYPES
                    WORKING_FILTERING = WORKING_FILTERING
                    WORKING_MIN_CUTOFFS = WORKING_MIN_CUTOFFS
                    WORKING_USE_OTHER = WORKING_USE_OTHER
                    WORKING_SCALING = WORKING_SCALING
    
                # DELETE HOME COLUMN FROM HOME OBJECT
                xx[home_obj_idx] = np.delete(xx[home_obj_idx], home_obj_col_idx, axis=0)
                # DELETE HOME COLUMN NAME FROM HOME OBJECT HEADER
                xx[home_obj_idx + 1] = np.delete(xx[home_obj_idx + 1], home_obj_col_idx, axis=1)
    
                print(f'\nCOLUMN MOVE COMPLETE.\n')
                break'''


        if user_ML_select == 'J':    #   'generate new WORKING TARGET VECTOR(j)'
            # THIS ALLOWS USER TO CHANGE TARGET VECTOR ON THE FLY AFTER CREATION OF WORKING OBJECTS
            # THE ONLY THING THAT COULD IMPACT TARGET IS ROW CHOP (THINKING ROW INSERT CANT EVER HAPPEN -- MAYBE)
            # THEREFORE GENERATE PSEUDO-BASE_TARGET FROM ROWID COMPARISON BETWEEN WORKING AND BASE ROWID IN REF_VECS

            target_config = 'Z'
            SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, split_method, LABEL_RULES, number_of_labels, event_value, negative_value = \
            tish.TargetInSituHandling(standard_config, target_config, SUPER_RAW_NUMPY_LIST, SUPER_WORKING_NUMPY_LIST,
                WORKING_SUPOBJS, split_method, LABEL_RULES, number_of_labels, event_value, negative_value).run()


        if user_ML_select == 'K':   #   'float check(k)'

            print(f'\nPerforming float conversion check in DATA and TARGET...\n')

            EXCEPTION_HOLDER = []
            for obj_idx in range(2):     # CHECK ONLY DATA AND TARGET
                try:
                    _ = np.array(SUPER_WORKING_NUMPY_LIST[obj_idx]).astype(np.float64)
                    print(f'\033[91mSuccessfully converted {SXNL_DICT[obj_idx]} to float\033[0m')
                except:
                    print(f'\033[92mUnable to convert {SXNL_DICT[obj_idx]} to float\033[0m')

        if  user_ML_select == 'X':    # 'go back to in-situ data augment(x)'

            while True:

                SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, CONTEXT_HOLDER, KEEP_HOLDER = \
                    isda.InSituDataAugment(
                                             standard_config,
                                             user_manual_or_standard,
                                             augment_method,
                                             SUPER_WORKING_NUMPY_LIST,
                                             data_given_orientation,
                                             target_given_orientation,
                                             refvecs_given_orientation,
                                             WORKING_SUPOBJS,
                                             WORKING_CONTEXT,
                                             WORKING_KEEP,
                                             bypass_validation=True
                    ).config_run()

                pocs.print_object_create_success(standard_config, 'WORKING DATA')

                if vui.validate_user_str(f'Accept WORKING DATA config? (y/n) (no restarts augment) > ', 'YN') == 'Y':
                    SUPER_WORKING_NUMPY_LIST = SUPER_WORKING_NUMPY_LIST_HOLDER
                    WORKING_SUPOBJS = WORKING_SUPOBJS_HOLDER
                    WORKING_CONTEXT = CONTEXT_HOLDER
                    WORKING_KEEP = KEEP_HOLDER

                    del SUPER_WORKING_NUMPY_LIST_HOLDER, WORKING_SUPOBJS_HOLDER, CONTEXT_HOLDER, KEEP_HOLDER
                    break


        # MLRegression ************************************************************************************************
        if user_ML_select == 'L':   # 'run MLRegression(l)'

            while True:
                if gs.get_shape('TARGET', SUPER_WORKING_NUMPY_LIST[1], target_given_orientation)[1] > 1:
                    print(f'\n*** TARGET VECTOR IS MULTICLASS, CANT DO MULTIPLE LINEAR REGRESSIOnp. ***\n')
                    break

                mlr_config = 'None'

                SUPER_RAW_NUMPY_LIST_HOLDER, RAW_SUPOBJS_HOLDER, SUPER_WORKING_NUMPY_LIST_HOLDER, WORKING_SUPOBJS_HOLDER, \
                WORKING_CONTEXT_HOLDER, WORKING_KEEP_HOLDER, split_method_holder, LABEL_RULES_HOLDER, number_of_labels_holder, \
                event_value_holder, negative_value_holder, mlr_conv_kill, mlr_pct_change, mlr_conv_end_method, mlr_rglztn_type, \
                mlr_rglztn_fctr, mlr_batch_method, mlr_batch_size, MLR_OUTPUT_VECTOR = \
                    mlrcr.MLRegressionConfigRun(standard_config, mlr_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS,
                        SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, data_given_orientation, target_given_orientation,
                        refvecs_given_orientation, WORKING_CONTEXT, WORKING_KEEP, split_method, LABEL_RULES,
                        number_of_labels, event_value, negative_value, mlr_rglztn_type, mlr_rglztn_fctr,
                        mlr_batch_method, mlr_batch_size, MLR_OUTPUT_VECTOR).configrun()

                if vui.validate_user_str('\nAccept MLRegression results? (y/n) > ', 'YN') == 'Y':
                    break

        # END MLRegression ************************************************************************************************

        # GREEDY MULTIPLE LINEAR REGRESSION  ****************************************************************************
        if user_ML_select == 'M':   #  'run greedy MLR(m)'
            while True:
                # IF MULTICATEGORY TARGET, CANT DO MLR
                if gs.get_shape('TARGET', SUPER_WORKING_NUMPY_LIST[1], target_given_orientation)[1] > 1:
                    print(f'\n*** TARGET VECTOR IS MULTICLASS, CANT DO LINEAR REGRESSION. ***\n')
                    break

                gmlr_config = 'Z'

                SUPER_RAW_NUMPY_LIST_HOLDER, RAW_SUPOBJS_HOLDER, SUPER_WORKING_NUMPY_LIST_HOLDER, WORKING_SUPOBJS_HOLDER, \
                WORKING_CONTEXT_HOLDER, WORKING_KEEP_HOLDER, split_method_holder, LABEL_RULES_HOLDER, \
                number_of_labels_holder, event_value_holder, negative_value_holder, gmlr_batch_method, gmlr_batch_size, \
                gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg, GMLR_OUTPUT_VECTOR = \
                    gmlrcr.GMLRConfigRun(standard_config, gmlr_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
                    WORKING_SUPOBJS, data_given_orientation, target_given_orientation, refvecs_given_orientation, WORKING_CONTEXT,
                    WORKING_KEEP, split_method, LABEL_RULES, number_of_labels, event_value, negative_value, gmlr_conv_kill,
                    gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, gmlr_batch_size,
                    gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg, GMLR_OUTPUT_VECTOR).configrun()

                if vui.validate_user_str('\nAccept GREEDY MULTIPLE LINEAR REGRESSION results? (y/n) > ', 'YN') == 'Y':
                    break
        # END GREEDY MULTIPLE LINEAR REGRESSION #***********************************************************************

        # MUTUAL INFORMATION *******************************************************************************************
        if user_ML_select == 'O':    # 'run Mutual Information(o)'
            while True:
                # IF MULTICATEGORY TARGET, CANT DO MLR
                if gs.get_shape('TARGET', SUPER_WORKING_NUMPY_LIST[1], target_given_orientation)[1] > 1:
                    print(f'\n*** TARGET VECTOR IS MULTICLASS, CANT DO MUTUAL INFORMATION. ***\n')
                    break

                mi_config = 'Z'

                SUPER_RAW_NUMPY_LIST_HOLDER, RAW_SUPOBJS_HOLDER, SUPER_WORKING_NUMPY_LIST_HOLDER, WORKING_SUPOBJS_HOLDER, \
                WORKING_CONTEXT_HOLDER, WORKING_KEEP_HOLDER, split_method_holder, LABEL_RULES_HOLDER, \
                number_of_labels_holder, event_value_holder, negative_value_holder, mi_batch_method, mi_batch_size, \
                mi_int_or_bin_only, mi_max_columns, mi_bypass_agg, MI_OUTPUT_VECTOR = \
                    micr.MIConfigRun(standard_config, mi_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
                        WORKING_SUPOBJS, data_given_orientation, target_given_orientation, refvecs_given_orientation,
                        WORKING_CONTEXT, WORKING_KEEP, split_method, LABEL_RULES, number_of_labels, event_value,
                        negative_value, mi_batch_method, mi_batch_size, mi_int_or_bin_only, mi_max_columns, mi_bypass_agg,
                        MI_OUTPUT_VECTOR).configrun()

                if vui.validate_user_str('\nAccept MUTUAL INFORMATION results? (y/n) > ', 'YN') == 'Y':
                    break
        # END MUTUAL INFORMATION ***************************************************************************************

        # SVD ***********************************************************************************************************
        if user_ML_select == 'V':   #  'run SVD(v)'
            while True:

                svd_config = 'Z'

                SUPER_RAW_NUMPY_LIST_HOLDER, SUPER_WORKING_NUMPY_LIST_HOLDER, WORKING_SUPOBJS_HOLDER, \
                WORKING_CONTEXT_HOLDER, WORKING_KEEP_HOLDER, split_method_holder, LABEL_RULES_HOLDER, \
                number_of_labels_holder, event_value_holder, negative_value_holder, svd_conv_kill, svd_pct_change, \
                svd_conv_end_method, dum_rglztn_type, dum_rglztn_fctr = \
                    svdcr.SVDConfigRun(standard_config, svd_config, SUPER_RAW_NUMPY_LIST, SUPER_WORKING_NUMPY_LIST,
                    WORKING_SUPOBJS, data_given_orientation, target_given_orientation, refvecs_given_orientation,
                    WORKING_CONTEXT, WORKING_KEEP, split_method, LABEL_RULES, number_of_labels, event_value,
                    negative_value, svd_max_columns).configrun()


                if vui.validate_user_str('\nAccept SINGULAR VALUE DECOMPOSITION results? (y/n) > ', 'YN') == 'Y':
                    break
        # END SVD #*******************************************************************************************************

        # GDA **********************************************************************************************************
        if user_ML_select == 'G':    # 'run GDA(g)'
            while True:
                gdar.GDA_run()

                if vui.validate_user_str('\nAccept GDA results? (y/n) > ', 'YN') == 'Y':
                    break
        # END GDA **********************************************************************************************************

        # SVM **********************************************************************************************************
        if user_ML_select == 'S':   # 'run SVM(s)'

            svm_config = 'Z'

            while True:
                if not SUPER_WORKING_NUMPY_LIST[1].all() in [-1, 1]:
                    print(f'\n*** Cannot use SVM, target must be -1s and 1s *** \n')
                    break

                if gs.get_shape('TARGET', SUPER_WORKING_NUMPY_LIST[1], target_given_orientation)[1] != 1:
                    print(f'\n*** Cannot use SVM, current target is multiclass *** \n')
                    break


                SUPER_RAW_NUMPY_LIST_HOLDER, RAW_SUPOBJS_HOLDER, SUPER_WORKING_NUMPY_LIST_HOLDER, WORKING_SUPOBJS_HOLDER, \
                WORKING_CONTEXT_HOLDER, WORKING_KEEP_HOLDER, split_method_holder, LABEL_RULES_HOLDER, \
                number_of_labels_holder, event_value_holder, negative_value_holder, svm_conv_kill, svm_pct_change, \
                svm_conv_end_method, dum_rglztn_type, dum_rglztn_fctr, C, max_passes, tol, K, ALPHAS, b, margin_type, \
                svm_cost_fxn, kernel_fxn, constant, exponent, sigma, alpha_seed, alpha_selection_alg, SMO_a2_selection_method = \
                    scr.SVMConfigRun(standard_config, svm_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
                    WORKING_SUPOBJS, data_given_orientation, target_given_orientation, refvecs_given_orientation, WORKING_CONTEXT,
                    WORKING_KEEP, split_method, LABEL_RULES, number_of_labels, event_value, negative_value, svm_conv_kill,
                    svm_pct_change, svm_conv_end_method, C, max_passes, tol, K, ALPHAS, b, margin_type, svm_cost_fxn,
                    kernel_fxn, constant, exponent, sigma, alpha_seed, alpha_selection_alg,
                    SMO_a2_selection_method).configrun()

                if vui.validate_user_str('\nAccept SVM results? (y/n) > ', 'YN') == 'Y':
                        break
        # END SVM **********************************************************************************************************

        # NN **************************************************************************************************************
        if user_ML_select == 'N':    #  'run NN(n)'

            nn_config = 'Z'

            SUPER_RAW_NUMPY_LIST_HOLDER, RAW_SUPOBJS_HOLDER, SUPER_WORKING_NUMPY_LIST_HOLDER, WORKING_SUPOBJS_HOLDER, \
            WORKING_CONTEXT_HOLDER, WORKING_KEEP_HOLDER, split_method_holder, LABEL_RULES_HOLDER, number_of_labels_holder, \
            event_value_holder, negative_value_holder, nn_conv_kill, nn_pct_change, nn_conv_end_method, nn_rglztn_type, \
            nn_rglztn_fctr, ARRAY_OF_NODES, NEURONS, nodes, node_seed, activation_constant, aon_base_path, aon_filename, \
            nn_cost_fxn, SELECT_LINK_FXN, LIST_OF_NN_ELEMENTS, NN_OUTPUT_VECTOR, batch_method, BATCH_SIZE, gd_method, \
            conv_method, lr_method, LEARNING_RATE, momentum_weight, gd_iterations, non_neg_coeffs, allow_summary_print, \
            summary_print_interval, iteration = \
                ncr.NNConfigRun(standard_config, nn_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
                WORKING_SUPOBJS, data_given_orientation, target_given_orientation, refvecs_given_orientation, WORKING_CONTEXT,
                WORKING_KEEP, split_method, LABEL_RULES, number_of_labels, event_value, negative_value, nn_rglztn_type,
                nn_rglztn_fctr, nn_conv_kill, nn_pct_change, nn_conv_end_method, ARRAY_OF_NODES, NEURONS, nodes, node_seed,
                activation_constant, aon_base_path, aon_filename, nn_cost_fxn, SELECT_LINK_FXN, LIST_OF_NN_ELEMENTS,
                NN_OUTPUT_VECTOR, batch_method, BATCH_SIZE, gd_method, conv_method, lr_method, LEARNING_RATE, momentum_weight,
                gd_iterations, non_neg_coeffs, allow_summary_print, summary_print_interval, iteration).configrun()

        # END NN **********************************************************************************************************
        # ******************************************************************************************************************
        # ******************************************************************************************************************

        if user_ML_select == 'A':      # 'dump current base data objects to file(a)'
            oed.ObjectsExcelDump(SUPER_RAW_NUMPY_LIST, 'BASE').dump()

        if user_ML_select == 'D':      #  'dump current data objects to file(d)'
            oed.ObjectsExcelDump(SUPER_WORKING_NUMPY_LIST, 'WORKING').dump()

        if user_ML_select == 'Q':   # 'quit(q)'
            sys.exit(f'User terminated.')

        if user_ML_select == 'W':    # 'convert to sparse dict or numpy arrays(w)'
            if isinstance(SUPER_WORKING_NUMPY_LIST[0], (list, tuple, np.ndarray)):
                SUPER_WORKING_NUMPY_LIST[0] = sd.zip_list(SUPER_WORKING_NUMPY_LIST[0])
                print(f'\n*** DATA HAS BEEN CONVERTED TO SPARSE DICT ***')
            elif isinstance(SUPER_WORKING_NUMPY_LIST[0], dict):
                SUPER_WORKING_NUMPY_LIST[0] = sd.unzip_to_ndarray(SUPER_WORKING_NUMPY_LIST[0])[0]
                print(f'\n*** DATA HAS BEEN CONVERTED TO NP ARRAY ***')



        # 1-28-22 IF USER RAN AN ML ALG, CHECK IF CHANGES TO WORKING AND RAW AND ASK IF USER WANTS TO KEEP, THIS WILL
        # ALLOW FOR MODIFICATIONS MADE IN ONE ALG TO BE IMMEDIATELY USED IN ANOTHER ########################################
        ### TEST FOR CHANGE TO RAW & WORKING ######################################################################################
        if user_ML_select in ml_algorithms_str:

            RAW_CHANGE_HOLDER = []
            WORKING_CHANGE_HOLDER = []
            # WHEN SKIPPING OUT OF SVM FOR FAILING [-1,1] OR SKIPPING OUT OF MI, MLR, GMLR, OR SVM FOR MULTICLASS,
            # IS BLOWING UP FOR HOLDERS NOT CREATED, SO:
            if (user_ML_select in 'LMOS' and gs.get_shape('TARGET', SUPER_WORKING_NUMPY_LIST[1], target_given_orientation)[1] != 1) or \
                (user_ML_select == 'S' and not SUPER_WORKING_NUMPY_LIST[1].all() in [-1, 1]):
                pass

            else:
                # 11/13/22 DONT BELIEVE ANY OF THE ML PACKAGES CAN MODIFY RAW OBJECTS, BUT PUTTING FUNCTIONALITY FOR FUTURE
                for idx, _OBJ in enumerate(SUPER_RAW_NUMPY_LIST_HOLDER):
                    if isinstance(_OBJ, dict) and not sd.core_sparse_equiv(_OBJ, SUPER_RAW_NUMPY_LIST[idx]) \
                    or \
                    isinstance(_OBJ, np.ndarray) and not np.array_equiv(_OBJ, SUPER_RAW_NUMPY_LIST[idx]):
                        RAW_CHANGE_HOLDER.append(idx)

                for idx, _OBJ in enumerate(SUPER_WORKING_NUMPY_LIST_HOLDER):
                    if isinstance(_OBJ, dict) and not sd.core_sparse_equiv(_OBJ, SUPER_WORKING_NUMPY_LIST[idx]) \
                    or \
                    isinstance(_OBJ, np.ndarray) and not np.array_equiv(_OBJ, SUPER_WORKING_NUMPY_LIST[idx]):
                        WORKING_CHANGE_HOLDER.append(idx)

                if len(RAW_CHANGE_HOLDER) > 0:
                    print('\n*** SOME RAW OBJECTS HAVE CHANGED FROM AS-CREATED STATE ***\n')
                    print(f'{", ".join(list(map(lambda x: SXNL_DICT[x], RAW_CHANGE_HOLDER)))} OBJECT(S) HAS/HAVE CHANGED.\n')

                    if vui.validate_user_str(f'OVERRIDE CURRENT RAW OBJECTS WITH CHANGES? (y/n) > ', 'YN') == 'Y':
                        SUPER_RAW_NUMPY_LIST = [deepcopy(_) if isinstance(_, dict) else _.copy() for _ in SUPER_RAW_NUMPY_LIST_HOLDER]

                if len(WORKING_CHANGE_HOLDER) > 0:
                    print('\n*** SOME WORKING OBJECTS HAVE CHANGED FROM AS-CREATED STATE ***\n')
                    print(f'{", ".join(list(map(lambda x: SXNL_DICT[x], WORKING_CHANGE_HOLDER)))} OBJECT(S) HAS/HAVE CHANGED.\n')

                    if vui.validate_user_str(f'OVERRIDE CURRENT WORKING OBJECTS WITH CHANGES? (y/n) > ', 'YN') == 'Y':
                        SUPER_WORKING_NUMPY_LIST = [deepcopy(_) if isinstance(_, dict) else _.copy() for _ in SUPER_WORKING_NUMPY_LIST_HOLDER]
                        WORKING_SUPOBJS = deepcopy(WORKING_SUPOBJS_HOLDER)
                        WORKING_CONTEXT = deepcopy(WORKING_CONTEXT_HOLDER)
                        WORKING_KEEP = deepcopy(WORKING_KEEP_HOLDER)
                        split_method = split_method_holder
                        LABEL_RULES = LABEL_RULES_HOLDER
                        number_of_labels = number_of_labels_holder
                        event_value = event_value_holder
                        negative_value = negative_value_holder

                del SUPER_RAW_NUMPY_LIST_HOLDER, RAW_SUPOBJS_HOLDER, SUPER_WORKING_NUMPY_LIST_HOLDER, WORKING_SUPOBJS_HOLDER, \
                    WORKING_CONTEXT_HOLDER, WORKING_KEEP_HOLDER

            del RAW_CHANGE_HOLDER, WORKING_CHANGE_HOLDER
        ### END TEST FOR CHANGE TO RAW & WORKING ###################################################################################

        for MENU in (ML_INSITU_BASE_CMDS, ML_INSITU_WORKING_CMDS, ML_ALGORITMS):
            _CMDS = list({f'{v}({k})' for k, v in MENU.items()})
            _ALLWD = "".join(MENU.keys()).upper()
            ppro.TopLevelMenuPrint(_CMDS, _ALLWD, append_ct_limit=2, max_len=menu_max_len+2)
        del _CMDS, _ALLWD

        user_ML_select = vui.validate_user_str(' > ', ml_insitu_base_str + ml_insitu_working_str + ml_algorithms_str)


















if __name__ == '__main__':
    ML()




