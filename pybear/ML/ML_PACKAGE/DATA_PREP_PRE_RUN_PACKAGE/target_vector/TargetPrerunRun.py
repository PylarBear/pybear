import time
import numpy as np
from copy import deepcopy
from MLObjects.SupportObjects import master_support_object_dict as msod


# CALLED by target_config_run
class TargetPrerunRun:

    def __init__(self, RAW_TARGET_SOURCE, RAW_TARGET_SOURCE_HEADER, RAW_TARGET_VECTOR, RAW_TARGET_VECTOR_HEADER, TARGET_VECTOR,
                 TARGET_SUPOBJS, split_method, LABEL_RULES, number_of_labels, event_value, negative_value, KEEP_COLUMNS):

        self.RAW_TARGET_SOURCE = RAW_TARGET_SOURCE
        self.RAW_TARGET_SOURCE_HEADER = np.array(RAW_TARGET_SOURCE_HEADER).reshape((1,-1))
        self.RAW_TARGET_VECTOR = RAW_TARGET_VECTOR
        self.RAW_TARGET_VECTOR_HEADER = np.array(RAW_TARGET_VECTOR_HEADER).reshape((1,-1))
        self.TARGET_VECTOR = TARGET_VECTOR
        self.TARGET_SUPOBJS = TARGET_SUPOBJS

        self.TARGET_VECTOR_HEADER = self.TARGET_SUPOBJS[msod.QUICK_POSN_DICT()["HEADER"]]
        self.MIN_CUTOFFS = self.TARGET_SUPOBJS[msod.QUICK_POSN_DICT()["MINCUTOFFS"]]
        self.USE_OTHER = self.TARGET_SUPOBJS[msod.QUICK_POSN_DICT()["USEOTHER"]]
        self.VALIDATED_DATATYPES = self.TARGET_SUPOBJS[msod.QUICK_POSN_DICT()["VALIDATEDDATATYPES"]]
        self.MODIFIED_DATATYPES = self.TARGET_SUPOBJS[msod.QUICK_POSN_DICT()["MODIFIEDDATATYPES"]]

        self.split_method = split_method
        self.LABEL_RULES = LABEL_RULES
        self.number_of_labels = number_of_labels
        self.event_value = event_value
        self.negative_value = negative_value
        self.KEEP_COLUMNS = KEEP_COLUMNS


    def return_fxn(self):

        self.TARGET_SUPOBJS[msod.QUICK_POSN_DICT()["HEADER"]] = self.TARGET_VECTOR_HEADER
        self.TARGET_SUPOBJS[msod.QUICK_POSN_DICT()["VALIDATEDDATATYPES"]] = self.VALIDATED_DATATYPES
        self.TARGET_SUPOBJS[msod.QUICK_POSN_DICT()["MODIFIEDDATATYPES"]] = self.MODIFIED_DATATYPES
        self.TARGET_SUPOBJS[msod.QUICK_POSN_DICT()["MINCUTOFFS"]] = self.MIN_CUTOFFS
        self.TARGET_SUPOBJS[msod.QUICK_POSN_DICT()["USEOTHER"]] = self.USE_OTHER

        return self.TARGET_VECTOR, self.TARGET_SUPOBJS


    def run(self):

        # 1-12-22 BUILD RAW TARGET VECTOR FROM SOURCE, USING COLUMN KEEP SELECTIONS FROM standard_config OR USER MANUAL
        # ENTRY IN TargetPrerunConfig
        if len(self.KEEP_COLUMNS) > 0:  # IF INSTRUCTIONS WERE GIVEN TO KEEP ONLY SOME TARGET_SOURCE COLUMNS
            self.RAW_TARGET_VECTOR = self.RAW_TARGET_SOURCE[self.KEEP_COLUMNS, ...]
            self.RAW_TARGET_VECTOR_HEADER = self.RAW_TARGET_VECTOR_HEADER[..., self.KEEP_COLUMNS]
        elif len(self.KEEP_COLUMNS) == 0:
            # NO INSTRUCTION TO KEEP TARGET SOURCE COLUMNS MEANS KEEPING ALL
            self.RAW_TARGET_VECTOR = self.RAW_TARGET_SOURCE.copy()
            self.RAW_TARGET_VECTOR_HEADER = self.RAW_TARGET_SOURCE_HEADER.copy()

        if self.split_method == 'None':
            print(self.RAW_TARGET_VECTOR)
            self.TARGET_VECTOR = self.RAW_TARGET_VECTOR.copy()
            # IF WAS GIVEN AS FLOAT OR INT, HEADER MUST STAY AS GIVEN
            self.TARGET_VECTOR_HEADER = self.RAW_TARGET_VECTOR_HEADER.copy()

        elif self.split_method == 'c2c':

            # FILL OUTPUT WITH ALL NEGATIVE VALUES FIRST, THEN GO THRU COLUMN BY COLUMN TO PUT IN POSITIVE VALUES BASED ON THAT COLUMNS
            # LABEL_RULES VIS-A-VIS RAW_TARGET_VECTOR
            self.TARGET_VECTOR = np.full((self.number_of_labels, len(self.RAW_TARGET_VECTOR[0])), self.negative_value)

            for label_idx in range(self.number_of_labels):      #NUMBER OF LABELS MUST EQUAL # OF OUTPUT TARGET VECTORS & EQUAL # OF LABELS
                for category in self.LABEL_RULES[label_idx]:
                    if category == 'COMPLEMENTARY':
                        # COMPL FILL IS THE SAME FOR CAT2CAT AND NUMBER... NO COMPL FILL FOR MULTI COL
                        for row_idx in self.TARGET_VECTOR:
                            if self.event_value not in self.TARGET_VECTOR[:, row_idx]:
                                self.TARGET_VECTOR[label_idx][row_idx] = self.event_value

                    elif category != 'COMPLEMENTARY':
                        self.TARGET_VECTOR[label_idx] = np.where(self.RAW_TARGET_VECTOR[0]==category,
                                                                self.event_value,
                                                                self.TARGET_VECTOR[label_idx]
                        )

            self.TARGET_VECTOR_HEADER = \
                np.fromiter((str(self.RAW_TARGET_VECTOR_HEADER[0][0]) + \
                             [" = " if self.event_value not in [0, -1] else " != "][0] + \
                             f'{self.LABEL_RULES[_]}' for _ in range(self.number_of_labels)), dtype='<U500').reshape((1, -1))

            for col_idx in range(self.number_of_labels - 1):
                self.TARGET_SUPOBJS = np.insert(self.TARGET_SUPOBJS, 0, self.TARGET_SUPOBJS[:,0], axis=1)

        elif self.split_method == 'n2c':

            self.TARGET_VECTOR = np.zeros((self.number_of_labels, len(self.RAW_TARGET_VECTOR[0])))

            for label_idx in range(self.number_of_labels):      #NUMBER OF LABELS MUST EQUAL # OF OUTPUT TARGET VECTORS
                _a = f'{self.LABEL_RULES[label_idx][0]}{self.LABEL_RULES[label_idx][1]}'
                _b = f'{self.LABEL_RULES[label_idx][2]}{self.LABEL_RULES[label_idx][3]}'
                for example_idx in range(len(self.RAW_TARGET_VECTOR[0])):  #COUNTS INSIDE RTV FOR C2C/N2C, #RTVS FOR MULTI_COL

                    if self.LABEL_RULES[label_idx]=='COMPLEMENTARY':
                        # COMPL FILL IS THE SAME FOR CAT2CAT AND NUMBER... NO COMPL FILL FOR MULTI COL
                        for row_idx in self.TARGET_VECTOR:
                            if self.event_value not in self.TARGET_VECTOR[:, row_idx]:
                                self.TARGET_VECTOR[label_idx][row_idx] = self.event_value

                    else:
                        _c = f'{self.RAW_TARGET_VECTOR[0][example_idx]}'
                        eval1 =  _c + _a

                        if len(self.LABEL_RULES[label_idx]) == 4: eval2 = f'and ' + _c + _b
                        else: eval2 = ''

                        eval1 += eval2

                        if eval(eval1): self.TARGET_VECTOR[label_idx] = np.array([*self.TARGET_VECTOR[-1], self.event_value], dtype=object)
                        else: self.TARGET_VECTOR[label_idx] = np.array([*self.TARGET_VECTOR[-1], self.negative_value], dtype=object)

            self.TARGET_VECTOR_HEADER = \
                np.fromiter((str(self.RAW_TARGET_VECTOR_HEADER[0][0]) + \
                             [" = " if self.event_value not in [0, -1] else " != "][0] + \
                             f'{self.LABEL_RULES[_]}' for _ in range(self.number_of_labels)), dtype='<U500').reshape((1, -1))

            for col_idx in range(self.number_of_labels - 1):
                self.TARGET_SUPOBJS = np.insert(self.TARGET_SUPOBJS, 0, self.TARGET_SUPOBJS[:,0], axis=1)

        elif self.split_method == 'mc':

            # FILL OUTPUT WITH ALL NEGATIVE VALUES FIRST, THEN GO THRU COLUMN BY COLUMN TO PUT IN POSITIVE VALUES BASED ON THAT COLUMNS
            # LABEL_RULES VIS-A-VIS RAW_TARGET_VECTOR
            self.TARGET_VECTOR = np.full((self.number_of_labels, len(self.RAW_TARGET_VECTOR[0])), self.negative_value)

            for label_idx in range(self.number_of_labels):      #NUMBER OF LABELS MUST EQUAL # OF OUTPUT TARGET VECTORS & EQUAL # OF LABELS
                for category in self.LABEL_RULES[label_idx]:
                    self.TARGET_VECTOR[label_idx] = np.where(self.TARGET_VECTOR[label_idx]==category,
                                                            self.event_value,
                                                            self.TARGET_VECTOR[label_idx]
                                                            )

            # IF WAS GIVEN AS SOFTMAX, HEADER MUST STAY AS GIVEN
            self.TARGET_VECTOR_HEADER = self.RAW_TARGET_VECTOR_HEADER.copy()

        if self.split_method in ['mc', 'c2c', 'n2c']: self.TARGET_VECTOR = self.TARGET_VECTOR.astype(np.int8)

        ################################################################################################################
        # CHECK THAT ALL EXAMPLES ADD RIGHT ACROSS LABELS###############################################################
        try:
            disaster = 'N'
            TARGET_DUM = np.array(self.TARGET_VECTOR).transpose()
            if len(self.TARGET_VECTOR) > 1:
                for idx in range(len(TARGET_DUM)):
                    target_sum_of_labels = 1 * self.event_value + (len(self.TARGET_VECTOR) - 1) * self.negative_value
                    if np.sum(TARGET_DUM[idx]) != target_sum_of_labels:
                        disaster = 'Y'
                        error_msg = f'\n****Multi-label: Examples do not sum correctly across labels.  Disaster.****' + \
                        f'\nBad idx is {idx}, sum of labels is {target_sum_of_labels} (should be {target_sum_of_labels})' + \
                        f'\nThis could be a result of multi-column input selections being bad from the start.\n'
                        break
            elif len(self.TARGET_VECTOR) == 1 and 'None' not in self.LABEL_RULES[0]:
                for idx in range(len(TARGET_DUM)):
                    if np.sum(TARGET_DUM[idx]) not in [self.event_value, self.negative_value]:
                        disaster = 'Y'
                        error_msg = f'\n****Single label: Event values not assigned correctly.  Disaster.****' + \
                                    f'\nBad idx is {idx}, label value is {TARGET_DUM[idx]}\n'
                        break
            if disaster == 'Y':
                print(error_msg)
            elif disaster == 'N':
                print('\nAll examples sum correctly.\n')

        except:
            print('\n****Fatal error in trying to verify / correct event / non-event sums in target vector.****')

        # END SUM CHECK#################################################################################################
        ################################################################################################################


        if self.split_method.upper() == 'NONE':
            _ = str(type(self.TARGET_VECTOR)).upper()
            _dtype = 'FLOAT' if 'FLOAT' in _ else 'INT'
            self.VALIDATED_DATATYPES = [_dtype for _ in range(self.number_of_labels)]
            self.MODIFIED_DATATYPES = [_dtype for _ in range(self.number_of_labels)]
        elif self.split_method.upper() in ['MC', 'N2C', 'C2C']:
            self.VALIDATED_DATATYPES = ['BIN' for _ in range(self.number_of_labels)]
            self.MODIFIED_DATATYPES = ['BIN' for _ in range(self.number_of_labels)]


        if len(self.MIN_CUTOFFS) > 1:  # IF CAME IN AS SOFTMAX, RETAIN OLD VALUES, ELSE REPLICATE OLD VALUE FOR NEW NO. OF COLUMNS
            self.MIN_CUTOFFS = self.MIN_CUTOFFS[2]
            self.USE_OTHER = self.USE_OTHER[2]
        elif len(self.MIN_CUTOFFS) == 1:
            self.MIN_CUTOFFS = [self.MIN_CUTOFFS[0] for _ in self.TARGET_VECTOR]
            self.USE_OTHER = [self.USE_OTHER[0] for _ in self.TARGET_VECTOR]


        return self.return_fxn()







if __name__ == '__main__':
    from MLObjects.SupportObjects import BuildFullSupportObject as bfso


    def supobj_builder(TARGET_OBJECT, TARGET_OBJECT_HEADER):
        SupObjClass = bfso.BuildFullSupportObject(
            OBJECT=TARGET_OBJECT,
            object_given_orientation='COLUMN',
            OBJECT_HEADER=TARGET_OBJECT_HEADER,
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

    split_method = 'None'
    number_of_labels = 1
    event_value = 1
    negative_value = 0
    KEEP_COLUMNS = [0]

    FLOAT_RAW_TARGET_SOURCE = np.array([[1,5,2,6,3,0,3,4,2,3,6,2,1,0,6]], dtype=object)
    FLOAT_RAW_TARGET_SOURCE_HEADER = [['STATUS1']]

    TARGET_SUPOBJ = supobj_builder(FLOAT_RAW_TARGET_SOURCE, FLOAT_RAW_TARGET_SOURCE_HEADER)

    EXP_SUPOBJ = TARGET_SUPOBJ.copy()

    LABEL_RULES = [
        'None'
    ]

    ACT_TARGET_VECTOR, ACT_TARGET_SUPOBJ = \
        TargetPrerunRun(FLOAT_RAW_TARGET_SOURCE, FLOAT_RAW_TARGET_SOURCE_HEADER, FLOAT_RAW_TARGET_SOURCE,
                        FLOAT_RAW_TARGET_SOURCE_HEADER, TARGET_VECTOR, TARGET_SUPOBJ, split_method, LABEL_RULES,
                        number_of_labels, event_value, negative_value, KEEP_COLUMNS).run()

    time.sleep(1)
    if not np.array_equiv(ACT_TARGET_VECTOR, FLOAT_RAW_TARGET_SOURCE):
        print(f'\033[91m')
        print(f'OUTPUT TARGET = ')
        print(ACT_TARGET_VECTOR)
        print()
        print(f'EXP TARGET = ')
        print(FLOAT_RAW_TARGET_SOURCE)
        raise Exception(f'\033[91m\n*** FAIL FLOAT TARGET OBJECT ***\n\033[0m')

    if not np.array_equiv(ACT_TARGET_SUPOBJ, EXP_SUPOBJ):
        print(f'\033[91m')
        print(f'OUTPUT SUPOBJ = ')
        print(ACT_TARGET_SUPOBJ)
        print()
        print(f'EXP SUPOBJ = ')
        print(EXP_SUPOBJ)
        raise Exception(f'\033[91m\n*** FAIL FLOAT TARGET SUPPORT OBJECT ***\n\033[0m')

    print(f'\n\033[92m*** FLOAT TARGET PASSED ***\033[0m\n')

    # END TEST FLOAT TARGET #################################################################################################
    #########################################################################################################################
    #########################################################################################################################


    #########################################################################################################################
    #########################################################################################################################
    # TEST C2C TARGET #####################################################################################################

    # C2C

    split_method = 'c2c'
    number_of_labels = 3
    event_value = 1
    negative_value = 0
    KEEP_COLUMNS = [0]

    C2C_RAW_TARGET_SOURCE = np.array([['A','B','A','B','C','C']], dtype=object)
    C2C_RAW_TARGET_SOURCE_HEADER = [['STATUS']]

    TARGET_SUPOBJ = supobj_builder(C2C_RAW_TARGET_SOURCE, C2C_RAW_TARGET_SOURCE_HEADER)

    EXP_TARGET_SUPOBJ = np.empty((len(msod.master_support_object_dict()), 0), dtype=object)
    HDR_ADD = [f" = ['A']", f" = ['B']", f" = ['C']"]
    hdr_idx = msod.QUICK_POSN_DICT()["HEADER"]
    for idx, _ in enumerate(HDR_ADD):
        EXP_TARGET_SUPOBJ = np.hstack((EXP_TARGET_SUPOBJ, TARGET_SUPOBJ))
        EXP_TARGET_SUPOBJ[hdr_idx][idx] = EXP_TARGET_SUPOBJ[hdr_idx][idx] + _
    del HDR_ADD, hdr_idx
    EXP_TARGET_SUPOBJ[msod.QUICK_POSN_DICT()["VALIDATEDDATATYPES"]] = ['BIN','BIN','BIN']
    EXP_TARGET_SUPOBJ[msod.QUICK_POSN_DICT()["MODIFIEDDATATYPES"]] = ['BIN', 'BIN', 'BIN']


    LABEL_RULES = [
        ['A'],
        ['B'],
        ['C']
    ]

    for RULE in LABEL_RULES:
        ACT_TARGET_VECTOR, ACT_TARGET_SUPOBJ = \
            TargetPrerunRun(C2C_RAW_TARGET_SOURCE, C2C_RAW_TARGET_SOURCE_HEADER, C2C_RAW_TARGET_SOURCE,
                            C2C_RAW_TARGET_SOURCE_HEADER, TARGET_VECTOR, TARGET_SUPOBJ, split_method, LABEL_RULES,
                            number_of_labels, event_value, negative_value, KEEP_COLUMNS).run()

    EXP_TARGET_VECTOR = [[1,0,1,0,0,0],[0,1,0,1,0,0],[0,0,0,0,1,1]]

    if not np.array_equiv(ACT_TARGET_VECTOR, EXP_TARGET_VECTOR):
        print(f'\033[91m')
        print(f'OUTPUT TARGET = ')
        print(ACT_TARGET_VECTOR)
        print()
        print(f'EXP TARGET = ')
        print(EXP_TARGET_VECTOR)
        raise Exception(f'\033[91m\n*** FAIL C2C TARGET OBJECT ***\n\033[0m')

    if not np.array_equiv(ACT_TARGET_SUPOBJ, EXP_TARGET_SUPOBJ):
        print(f'\033[91m')
        print(f'OUTPUT SUPOBJ = ')
        print(ACT_TARGET_SUPOBJ)
        print()
        print(f'EXP TARGET = ')
        print(EXP_TARGET_SUPOBJ)
        raise Exception(f'\033[91m\n*** FAIL C2C TARGET SUPPORT OBJECT ***\n\033[0m')

    print(f'\n\033[92m*** C2C TARGET PASSED ***\033[0m\n')

    # END TEST C2C TARGET #################################################################################################
    #########################################################################################################################
    #########################################################################################################################

    quit()
    #########################################################################################################################
    #########################################################################################################################
    # TEST CAT TARGET #####################################################################################################

    # BEAR FIX, HAVE TO DO N2C AND MC

    # CAT
    CAT_RAW_TARGET_SOURCE = [['A','B','C','D','E']]
    CAT_RAW_TARGET_SOURCE_HEADER = [['STATUS1']]

    TARGET_SUPOBJ = supobj_builder(CAT_RAW_TARGET_SOURCE, CAT_RAW_TARGET_SOURCE_HEADER)

    split_method = 'c2c'
    number_of_labels = 1
    event_value = 1
    negative_value = 0
    KEEP_COLUMNS = [0]


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

        TARGET_VECTOR, TARGET_SUPOBJS = \
            TargetPrerunRun(RAW_TARGET_SOURCE, RAW_TARGET_SOURCE_HEADER, RAW_TARGET_VECTOR, RAW_TARGET_VECTOR_HEADER,
                            TARGET_VECTOR, TARGET_SUPOBJ, split_method, LABEL_RULES, number_of_labels, event_value,
                            negative_value, KEEP_COLUMNS).run()

        if not np.array_equiv(TARGET_VECTOR, EXP_TARGET_VECTOR):
            raise Exception(f'\033[91m\n*** FAIL SOFTMAX TARGET OBJECT ***\n\033[0m')

        if not np.array_equiv(TARGET_SUPOBJ, EXP_TARGET_SUPOBJ):
            raise Exception(f'\033[91m\n*** FAIL SOFTMAX TARGET SUPPORT OBJECTS ***\n\033[0m')

    # END TEST FLOAT TARGET #################################################################################################
    #########################################################################################################################
    #########################################################################################################################




    print(f'\n\033[91m*** TESTS COMPLETED SUCCESSFULLY ***\033[0m\n')





















