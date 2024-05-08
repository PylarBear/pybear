from general_data_ops import TrainDevTestSplit as tdts
from MLObjects.SupportObjects import SupObjConverter as soc


class MLTrainDevTestSplitNEWSUPOBJS(tdts.TrainDevTestSplit):
    '''Dont call this directly, call with mask(), _random_(), partition(), or cateyory() methods only.'''
    def __init__(self, DATA=None, TARGET=None, REFVECS=None, data_given_orientation=None,
                 target_given_orientation=None, refvecs_given_orientation=None, bypass_validation=None):

        super().__init__(DATA=DATA, TARGET=TARGET, REFVECS=REFVECS, data_given_orientation=data_given_orientation,
                 target_given_orientation=target_given_orientation, refvecs_given_orientation=refvecs_given_orientation,
                 bypass_validation=bypass_validation)

    # INHERITS
    # mask()
    # _random_()
    # partition()
    # category()
    # core_run_fxn()


# BEAR DELETE THIS IF DONT NEEDED ANYMORE, 6/8/23 HAS NO USAGES
class MLTrainDevTestSplitOLDSUPOBJS(tdts.TrainDevTestSplit):
    '''Dont call this directly, call with mask(), _random_(), partition(), or cateyory() methods only.'''
    def __init__(self, DATA=None, TARGET=None, REFVECS=None, data_given_orientation=None,
                 target_given_orientation=None, refvecs_given_orientation=None, bypass_validation=None):

        super().__init__(DATA=DATA, TARGET=TARGET, REFVECS=REFVECS, data_given_orientation=data_given_orientation,
                 target_given_orientation=target_given_orientation, refvecs_given_orientation=refvecs_given_orientation,
                 bypass_validation=bypass_validation)

    # INHERITS
    # mask()
    # _random_()
    # partition()
    # core_run_fxn()

    # OVERWRITES
    def category(self, object_name_for_dev=None, dev_column_name_or_idx=None, DEV_SPLIT_ON_VALUES_AS_LIST=None,
                 object_name_for_test=None, test_column_name_or_idx=None, TEST_SPLIT_ON_VALUES_AS_LIST=None,
                 DATA_HEADER=None, TARGET_HEADER=None, REFVECS_HEADER=None, VALIDATED_DATATYPES=None, MODIFIED_DATATYPES=None,
                 FILTERING=None, MIN_CUTOFFS=None, USE_OTHER=None, START_LAG=None, END_LAG=None, SCALING=None):

        # WOULD HAVE TO GET HEADERS FROM [1,3,5] POSNS OF SRNL OR SWNL

        SupObjClass = soc.SupObjConverter(DATA_HEADER=DATA_HEADER,
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

        DATA_FULL_SUPOBJ = SupObjClass.DATA_FULL_SUPOBJ
        TARGET_FULL_SUPOBJ = SupObjClass.TARGET_FULL_SUPOBJ
        REFVECS_FULL_SUPOBJ = SupObjClass.REFVECS_FULL_SUPOBJ

        del SupObjClass

        super().category(object_name_for_dev=object_name_for_dev,
                        dev_column_name_or_idx=dev_column_name_or_idx,
                        DEV_SPLIT_ON_VALUES_AS_LIST=DEV_SPLIT_ON_VALUES_AS_LIST,
                        object_name_for_test=object_name_for_test,
                        test_column_name_or_idx=test_column_name_or_idx,
                        TEST_SPLIT_ON_VALUES_AS_LIST=TEST_SPLIT_ON_VALUES_AS_LIST,
                        DATA_FULL_SUPOBJ=DATA_FULL_SUPOBJ,
                        TARGET_FULL_SUPOBJ=TARGET_FULL_SUPOBJ,
                        REFVECS_FULL_SUPOBJ=REFVECS_FULL_SUPOBJ)


































