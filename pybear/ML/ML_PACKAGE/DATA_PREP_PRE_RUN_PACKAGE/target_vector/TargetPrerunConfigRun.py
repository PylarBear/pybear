import numpy as np
from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.target_vector import TargetPrerunConfig as tpc, TargetPrerunRun as tpr
from MLObjects.SupportObjects import master_support_object_dict as msod


class TargetPrerunConfigRun:
    def __init__(self, standard_config, target_config, RAW_TARGET_SOURCE, RAW_TARGET_SOURCE_HEADER, TARGET_VECTOR, TARGET_SUPOBJS):

        self.standard_config = standard_config
        self.target_config = target_config
        self.RAW_TARGET_SOURCE = RAW_TARGET_SOURCE
        self.RAW_TARGET_SOURCE_HEADER = RAW_TARGET_SOURCE_HEADER
        self.TARGET_VECTOR = TARGET_VECTOR
        self.TARGET_SUPOBJS = TARGET_SUPOBJS

        # PLACEHOLDERS
        self.RAW_TARGET_VECTOR = ''
        self.RAW_TARGET_VECTOR_HEADER = ''
        self.number_of_labels = ''
        self.split_method = ''
        self.LABEL_RULES = ''
        self.event_value = ''
        self.negative_value = ''
        self.KEEP_COLUMNS = ''


    def config_module(self):
        return tpc.TargetPrerunConfig(self.standard_config, self.target_config, self.RAW_TARGET_SOURCE,
                self.RAW_TARGET_SOURCE_HEADER, self.TARGET_VECTOR, self.TARGET_SUPOBJS).config()


    def run_module(self):
        return tpr.TargetPrerunRun(self.RAW_TARGET_SOURCE, self.RAW_TARGET_SOURCE_HEADER, self.RAW_TARGET_VECTOR,
            self.RAW_TARGET_VECTOR_HEADER, self.TARGET_VECTOR, self.TARGET_SUPOBJS, self.split_method, self.LABEL_RULES,
            self.number_of_labels, self.event_value, self.negative_value, self.KEEP_COLUMNS).run()


    def return_fxn(self):

        return self.TARGET_VECTOR, self.TARGET_SUPOBJS, self.split_method, self.LABEL_RULES, self.number_of_labels, \
                    self.event_value, self.negative_value


    def configrun(self):

        self.TARGET_VECTOR = []

        # CONFIG ###################################################################################################
        self.target_config, self.split_method, self.RAW_TARGET_VECTOR, self.RAW_TARGET_VECTOR_HEADER, self.TARGET_VECTOR, \
        self.TARGET_SUPOBJS, self.LABEL_RULES, self.number_of_labels, self.event_value, self.negative_value, \
        self.KEEP_COLUMNS = self.config_module()
        ############################################################################################################

        ############################################################################################################
        #TARGET VECTOR FILL#########################################################################################

        if self.split_method not in 'CUSTOM':
            self.TARGET_VECTOR, self.TARGET_SUPOBJS = self.run_module()

        # END TARGET VECTOR FILL####################################################################################
        ############################################################################################################

        return self.return_fxn()





if __name__ == '__main__':

    from MLObjects.SupportObjects import BuildFullSupportObject as bfso


    def supobj_builder(TARGET_OBJECT, TARGET_OBJECT_HEADER):
        SupObjClass = bfso.BuildFullSupportObject(
            OBJECT=FLOAT_RAW_TARGET_SOURCE,
            object_given_orientation='COLUMN',
            OBJECT_HEADER=FLOAT_RAW_TARGET_SOURCE_HEADER,
            SUPPORT_OBJECT=None,  # IF PASSED, MUST BE GIVEN AS FULL SUPOBJ
            columns=None,  # THE INTENT IS THAT THIS IS ONLY GIVEN IF OBJECT AND SUPPORT_OBJECT ARE NOT
            quick_vdtypes=False,
            MODIFIED_DATATYPES=None,
            print_notes=False,
            prompt_to_override=False,
            bypass_validation=False,
            calling_module='TargetPrerunRun',
            calling_fxn='test'
        )

        return SupObjClass.SUPPORT_OBJECT






    # CAT
    CAT_RAW_TARGET_SOURCE = [['X','O','Y','Z', 'Z','X','O','Y','P','X','O','X']]
    CAT_RAW_TARGET_SOURCE_HEADER = [['STATUS1']]

    # SOFTMAX
    # RAW_TARGET_SOURCE = [['X','O','Y','Z'],['Z','X','O','Y'],['P','X','O','X']]
    # RAW_TARGET_SOURCE_HEADER = [['STATUS1', 'STATUS2', 'STATUS3']]

    # FLOAT
    FLOAT_RAW_TARGET_SOURCE = np.array([[1,5,2,6,3,0,3,4,2,3,6,2,1,0,6]], dtype=object)
    FLOAT_RAW_TARGET_SOURCE_HEADER = [['STATUS1']]

    TARGET_VECTOR = []
    TARGET_VECTOR_HEADER = [[]]
    RAW_TARGET_VECTOR = []

    ##### DF STUFF
    # RAW_TARGET_SOURCE_HEADER = ['a'] #,'b','c','d']
    # RAW_TARGET_SOURCE = p.DataFrame(data=RAW_TARGET_SOURCE.transpose(), columns=RAW_TARGET_SOURCE_HEADER)
    ##### END DF STUFF


    #########################################################################################################################
    #########################################################################################################################
    # TEST FLOAT TARGET #####################################################################################################

    # FLOAT
    FLOAT_RAW_TARGET_SOURCE = np.array([[1,5,2,6,3,0,3,4,2,3,6,2,1,0,6]], dtype=object)
    FLOAT_RAW_TARGET_SOURCE_HEADER = [['STATUS1']]

    TARGET_SUPOBJ = supobj_builder(FLOAT_RAW_TARGET_SOURCE, FLOAT_RAW_TARGET_SOURCE_HEADER)

    EXP_SUPOBJ = TARGET_SUPOBJ.copy()

    target_config = 'Z'
    standard_config = 'AA'


    LABEL_RULES = [
        'None'
    ]

    TARGET_VECTOR, TARGET_SUPOBJS, split_method, LABEL_RULES, number_of_labels, event_value, negative_value = \
        TargetPrerunConfigRun(standard_config, target_config, FLOAT_RAW_TARGET_SOURCE, FLOAT_RAW_TARGET_SOURCE_HEADER,
                              TARGET_VECTOR, TARGET_SUPOBJ).configrun()

    if not np.array_equiv(TARGET_VECTOR, FLOAT_RAW_TARGET_SOURCE):
        raise Exception(f'\033[91m\n*** FAIL FLOAT TARGET OBJECT ***\n\033[0m')

    if not np.array_equiv(TARGET_SUPOBJ, EXP_SUPOBJ):
        raise Exception(f'\033[91m\n*** FAIL FLOAT SUPPORT OBJECTS ***\n\033[0m')

    print(f'\n\033[92m\*** FLOAT TARGET PASSED ***\033[0m\n')

    # END TEST FLOAT TARGET #################################################################################################
    #########################################################################################################################
    #########################################################################################################################


    #########################################################################################################################
    #########################################################################################################################
    # TEST SOFTMAX TARGET #####################################################################################################

    # SOFTMAX
    SOFTMAX_RAW_TARGET_SOURCE = np.array([['A','B','A','B','C','C']], dtype=object)
    SOFTMAX_RAW_TARGET_SOURCE_HEADER = [['STATUS']]

    TARGET_SUPOBJ = supobj_builder(FLOAT_RAW_TARGET_SOURCE, FLOAT_RAW_TARGET_SOURCE_HEADER)
    EXP_TARGET_SUPOBJ = np.empty((len(msod.master_support_object_dict()), 1), dtype=object)
    for _ in range(3):
        EXP_TARGET_SUPOBJ = np.hstack((EXP_TARGET_SUPOBJ, TARGET_SUPOBJ))

    target_config = 'Z'
    standard_config = 'AA'

    LABEL_RULES = [
        ['A'],
        ['B'],
        ['C']
    ]

    for RULE in LABEL_RULES:
        TARGET_VECTOR, TARGET_SUPOBJS, split_method, LABEL_RULES, number_of_labels, event_value, negative_value = \
            TargetPrerunConfigRun(standard_config, target_config, FLOAT_RAW_TARGET_SOURCE, FLOAT_RAW_TARGET_SOURCE_HEADER,
                                  TARGET_VECTOR, TARGET_SUPOBJS).configrun()

    EXP_TARGET_VECTOR = [[1,0,1,0,0,0],[0,1,0,1,0,0],[0,0,0,0,1,1]]

    if not np.array_equiv(TARGET_VECTOR, EXP_TARGET_VECTOR):
        raise Exception(f'\033[91m\n*** FAIL SOFTMAX TARGET OBJECT ***\n\033[0m')

    if not np.array_equiv(TARGET_SUPOBJ, EXP_TARGET_SUPOBJ):
        raise Exception(f'\033[91m\n*** FAIL SOFTMAX TARGET SUPPORT OBJECTS ***\n\033[0m')

    print(f'\n\033[92m\*** SOFTMAX TARGET PASSED ***\033[0m\n')

    # END TEST SOFTMAX TARGET #################################################################################################
    #########################################################################################################################
    #########################################################################################################################

    quit()
    #########################################################################################################################
    #########################################################################################################################
    # TEST CAT TARGET #####################################################################################################


    # CAT
    CAT_RAW_TARGET_SOURCE = [['A','B','C','D','E']]
    CAT_RAW_TARGET_SOURCE_HEADER = [['STATUS1']]

    TARGET_SUPOBJ = supobj_builder(CAT_RAW_TARGET_SOURCE, CAT_RAW_TARGET_SOURCE_HEADER)

    target_config = 'Z'
    standard_config = 'AA'


    LABEL_RULES = [
        [['A'],['B'],['COMPLEMENTARY']],
        [[],[],[]],
        [[],[],[]],
        [[],[],[]],
        [[],[],[]],
    ]

    EXP_TARGET = [
        [[], [], []],
        [[], [], []],
        [[], [], []],
        [[], [], []],
        [[], [], []],
    ]


    for RULE in LABEL_RULES:

        TARGET_VECTOR, TARGET_SUPOBJS, split_method, LABEL_RULES, number_of_labels, event_value, negative_value = \
              TargetPrerunConfigRun(standard_config, target_config, CAT_RAW_TARGET_SOURCE, CAT_RAW_TARGET_SOURCE_HEADER,
                                    TARGET_VECTOR, TARGET_SUPOBJS).configrun()

        if not np.array_equiv(TARGET_VECTOR, EXP_TARGET_VECTOR):
            raise Exception(f'\033[91m\n*** FAIL SOFTMAX TARGET OBJECT ***\n\033[0m')

        if not np.array_equiv(TARGET_SUPOBJ, EXP_TARGET_SUPOBJ):
            raise Exception(f'\033[91m\n*** FAIL SOFTMAX TARGET SUPPORT OBJECTS ***\n\033[0m')

    # END TEST FLOAT TARGET #################################################################################################
    #########################################################################################################################
    #########################################################################################################################




    print(f'\n\033[91m*** TESTS COMPLETED SUCCESSFULLY ***\033[0m\n')

