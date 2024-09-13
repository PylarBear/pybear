from copy import deepcopy
import numpy as n
from data_validation import validate_user_input as vui
from ML_PACKAGE._data_validation import ValidateObjectType as vot, validate_modified_object_type as vmot
from debug import IdentifyObjectAndPrint as ioap
from general_data_ops import return_uniques as ru
from general_list_ops import list_select as ls
from ML_PACKAGE.standard_configs import standard_configs as sc
from ML_PACKAGE.SPLITTER_RULES import category_to_category as ctc, category_to_multi_column as ctmc, number_to_category as ntc
from ML_PACKAGE.GENERIC_PRINT import DictMenuPrint as dmp
from MLObjects.SupportObjects import master_support_object_dict as msod


# CALLED by TargetPrerunConfigRun
# 1-8-22 RAW_TARGET_SOURCE COMES IN AS ANY LIST-TYPE OF LIST-TYPES, RAW_TARGET_SOURCE_HEADER = []
class TargetPrerunConfig:

    def __init__(self, standard_config, target_config, RAW_TARGET_SOURCE, RAW_TARGET_SOURCE_HEADER, TARGET_VECTOR, TARGET_SUPOBJS):
        self.standard_config = standard_config
        self.target_config = target_config
        self.RAW_TARGET_SOURCE = RAW_TARGET_SOURCE
        self.RAW_TARGET_SOURCE_HEADER = n.array(RAW_TARGET_SOURCE_HEADER).reshape((1,-1))
        self.TARGET_VECTOR = TARGET_VECTOR
        self.TARGET_SUPOBJS = TARGET_SUPOBJS

        # PLACEHOLDER
        self.RAW_TARGET_VECTOR = ''
        self.RAW_TARGET_VECTOR_HEADER = ''
        self.TARGET_UNIQUE_VALUES = ''
        self.split_method = ''
        self.KEEP_COLUMNS = ''
        self.LABEL_RULES = []

        # BE SURE TO ADJUST allowed_keys IN config()
        self.MENU_OPTIONS = {
                            'a': 'accept target config / continue',
                            'i': 'load standard config',
                            'z': 'perform compulsory setup sequence',
                            'm': 'select target source columns',
                            's': 'change split method',
                            'c': 'change number of labels',
                            'l': 'adjust label rules',
                            'v': 'change event values',
                            'q': 'quit'
        }


    def standard_config_module(self):
        return sc.TARGET_VECTOR_standard_configs(self.standard_config, self.target_config, self.RAW_TARGET_SOURCE,
             self.RAW_TARGET_SOURCE_HEADER, self.TARGET_VECTOR, self.TARGET_SUPOBJS[msod.QUICK_POSN_DICT()["HEADER"]])


    def return_fxn(self):
        return self.target_config, \
                self.split_method, \
                self.RAW_TARGET_VECTOR, \
                self.RAW_TARGET_VECTOR_HEADER, \
                self.TARGET_VECTOR, \
                self.TARGET_SUPOBJS, \
                self.LABEL_RULES, \
                self.number_of_labels, \
                self.event_value, \
                self.negative_value, \
                self.KEEP_COLUMNS


    def config(self, allowed_keys = 'AIZMSCLVQ'):
            #CREATE A 'Z' SETTING THAT IS USED TO DISTINGUISH BETWEEN THE 1ST PASS & ALL SUBSEQNT PASSES
    
            ################################################################################################################
            # CHOOSE A STANDARD CONFIG OR MANUAL############################################################################
        while True:
            if self.target_config in 'IZ':   #load standard config(i),  perform compulsory setup sequence(z), use current LABEL RULES config(r)
                # print('\nload standard config(i),  proceed with manual config(m)\n')
                if self.target_config == 'Z':
                    if vui.validate_user_str('\nLoad TARGET VECTOR configs(i) or proceed with manual config(m)? > ', 'IM') == 'I':
                        self.target_config = 'I'
    
                if self.target_config in 'I':

                    self.RAW_TARGET_VECTOR, self.RAW_TARGET_VECTOR_HEADER, self.TARGET_VECTOR, \
                    self.TARGET_SUPOBJS[msod.QUICK_POSN_DICT()["HEADER"]], self.LABEL_RULES, self.number_of_labels, \
                    self.split_method, self.event_value, self.negative_value, self.KEEP_COLUMNS = self.standard_config_module()

                    # KEEP target_config AS 'I', BYPASSES EVERYTHING
    
            ################################################################################################################
            # IDENTIFY TARGET SOURCE THEN BUILD RAW TARGET VECTOR ###########################################################
            if self.target_config in 'MZ':    #select target source columns(m)
                # RAW_TARGET_SOURCE IS CONVERTED INTO RAW_TARGET_VECTOR
                # IF DOING MANUAL CONFIG
                while True:
                    print(f'\nReading datatypes in TARGET...')
                    data_type = vmot.validate_modified_object_type(self.RAW_TARGET_SOURCE)
                    if 'EMPTY' in data_type.upper():    # 1-8-22 IF EMPTY DATA, TERMINATE HERE
                        raise Exception(f'DATA IS EMPTY. TERMINATE.')
                    elif data_type == 'DICT' or data_type == 'CHAR_SEQ':    # 1-8-22 IF INVALID DATATYPE, TERMINATE HERE
                        raise Exception(f'RAW TARGET SOURCE object type is {data_type}, cannot use as TARGET SOURCE!')
                    else: pass
                        #print(f'Incoming RAW TARGET SOURCE is a {vot.ValidateObjectType(self.RAW_TARGET_SOURCE).validate_object_type()[0]}.')
                    print(f'Done.')
                    print(f'\nProducing display of TARGET as DF...')
                    ioap.IdentifyObjectAndPrint(self.RAW_TARGET_SOURCE, 'RAW_TARGET_SOURCE', __name__, rows=10, columns=5
                                                ).run_print_as_df(df_columns=self.RAW_TARGET_SOURCE_HEADER, orientation='column')
    
                    # IF INCOMING IS DF, CONVERT TO NP
                    if data_type == 'DATAFRAME': self.RAW_TARGET_SOURCE.to_numpy(inplace=True)
    
                    number_of_incoming_columns = len(self.RAW_TARGET_SOURCE)  # INCOMING DFs ALREADY TURNED TO LIST-OF-LISTS

                    # SELECT COLUMNS TO USE FROM INCOMING RAW TARGET SOURCE# (select new target source columns)
                    if number_of_incoming_columns == 1:
                        self.KEEP_COLUMNS = [0] # IF ONLY ONE COLUMN IN DF, MUST USE IT
                        print(f'\nINCOMING TARGET HAS ONLY ONE COLUMN, CANNOT DELETE ANY COLUMNS\n')
                        break
                    else:
                        print(f'\nRAW TARGET SOURCE has more than one column.')
                        print(f'Selecting one(o) or multiple columns (softmax only)(m) from RAW TARGET SOURCE?')
                        print(f'(Selecting one column can also later be split into multiple columns for softmax.)')
                        single_or_multi = vui.validate_user_str(' > ', 'OM')
                        SELECTOR_DUM = [f'{self.RAW_TARGET_SOURCE_HEADER[0][_]} - {str([__ for __ in self.RAW_TARGET_SOURCE[_][:5]])[:-1]} ...' for \
                                            _ in range(len(self.RAW_TARGET_SOURCE))]
                        if single_or_multi == 'O':
                            self.KEEP_COLUMNS = ls.list_single_select(SELECTOR_DUM, 'Select column', 'idx')
                        elif single_or_multi == 'M':
                            self.KEEP_COLUMNS = ls.list_multi_select(SELECTOR_DUM, 'Select columns', 'idx')
    
                    if vui.validate_user_str('Accept RAW TARGET VECTOR column setup? (y/n) > ', 'YN') == 'Y':
                        break

                # 1-14-22 ONLY IF DOING MANUAL CONFIG ARE RAW_TARGET_VECTOR & RAW_TARGET_VECTOR_HEADER CREATED, THIS
                # IS NECESSARY TO HAVE THE INFO FOR SUBSEQUENT MANUAL CONFIG STEPS, OTHERWISE IF DOING standard_config
                # KEEP_COLUMNS IS CARRIED INTO TargetPrerunRun FOR CHOPS. IT IS NECESSARY TO HAVE CHOP STEP IN PrerunRun
                # FOR WHENEVER KEEP_COLUMNS IS SPECIFIED IN standard_configs
                if len(self.KEEP_COLUMNS) > 0:  # IF INSTRUCTIONS WERE GIVEN TO KEEP SOME TARGET_SOURCE COLUMNS
                    self.RAW_TARGET_VECTOR = self.RAW_TARGET_SOURCE[self.KEEP_COLUMNS, ...]
                    self.RAW_TARGET_VECTOR_HEADER = self.RAW_TARGET_SOURCE_HEADER[..., self.KEEP_COLUMNS]
                elif len(self.KEEP_COLUMNS) == 0:
                    # NO INSTRUCTION TO KEEP TARGET SOURCE COLUMNS MEANS KEEPING ALL
                    self.RAW_TARGET_VECTOR = deepcopy(self.RAW_TARGET_SOURCE)
                    self.RAW_TARGET_VECTOR_HEADER = deepcopy(self.RAW_TARGET_SOURCE_HEADER)
    
            # END BUILD RAW TARGET VECTOR###################################################################################
            ################################################################################################################
    
            ####################################################################################################################
            # SELECT SPLIT METHOD###############################################################################################
            if self.target_config in 'MSZ':     #change split method(s)
                print(f'change split method(s)')
                ################################################################################################################
                # CALCULATE STUFF NEEDED TO DETERMINE SPLIT METHOD OPTIONS######################################################
                # FIND TYPE OF DATA
                data_type = vot.ValidateObjectType(self.RAW_TARGET_VECTOR).ml_package_object_type()
    
                # GET ALL UNIQUE VALUES IN ALL TARGET COLUMNS
                TARGET_UNIQUE_VALUES_HOLDER = []
                for column in self.RAW_TARGET_VECTOR:
                    TARGET_UNIQUE_VALUES_HOLDER += ru.return_uniques(column, [], data_type, suppress_print='Y')[0]

                self.TARGET_UNIQUE_VALUES = \
                    ru.return_uniques(TARGET_UNIQUE_VALUES_HOLDER, [], data_type, suppress_print='Y')[0]
    
                if len(self.TARGET_UNIQUE_VALUES) <= 1:
                    print(f'FATAL ERROR.  There is one or less unique entry in RAW TARGET VECTOR.  Analysis is impossible.')
                ################################################################################################################
                ################################################################################################################
    
                ################################################################################################################
                # SPLIT OPTIONS#################################################################################################
                while True:
                    if len(self.RAW_TARGET_VECTOR) > 1 and len(self.TARGET_UNIQUE_VALUES) == 2:
                        if data_type == 'STR':
                            print(f'There is more than 1 raw target vector, with 2 unique STRING Values. Only split option available is multi-column.')
                            self.split_method = 'mc'
    
                        elif data_type in ['FLOAT','INT', 'BIN']:
                            print(f'There is more than 1 raw target vector, with 2 unique numerical values.  Split options available are None(z) and multi-column(m)')
                            self.split_method = vui.validate_user_str(
                                f'Select none to keep current numbers in raw target, use multi to reassign event values > ', 'MZ')
    
                    elif len(self.RAW_TARGET_VECTOR) > 1 and len(self.TARGET_UNIQUE_VALUES) > 2:
                        if data_type == 'STR':
                            print(
                                f'There is more than 1 raw target vector, with more than 2 unique STRING Values. Only split option available is multi-column.')
                            self.split_method = 'mc'
    
                        elif data_type in ['FLOAT', 'INT']:
                            print(
                                f'There is more than 1 raw target vector, with more than 2 unique numerical values.  Only split option available is multi-column.')
                            self.split_method = 'mc'
    
                    elif len(self.RAW_TARGET_VECTOR) == 1 and data_type in ['FLOAT', 'INT']:
                        self.split_method = vui.validate_user_str(
                            f'Data type is {data_type}.  Split options available are None(z), number-to-category(n), and category-to-category(c) > ', 'CNZ')
    
                    elif len(self.RAW_TARGET_VECTOR) == 1 and data_type == 'STR':
                        print(f'Data type is STRING.  The only split option available is category-to-category.')
                        self.split_method = 'c2c'

                    elif len(self.RAW_TARGET_VECTOR) == 1 and data_type == 'BIN':
                        print(f'Data type is BIN.  The only split option available is None.')
                        self.split_method = 'None'
    
                    if self.split_method == 'M': self.split_method = 'mc'
                    elif self.split_method == 'Z': self.split_method = 'None'
                    elif self.split_method == 'C': self.split_method = 'c2c'
                    elif self.split_method == 'N': self.split_method = 'n2c'
    
                    if vui.validate_user_str(f'Current split method is {self.split_method}.  Accept split method? (y/n) > ', 'YN') == 'Y':
                        break
                # END SPLIT OPTIONS#############################################################################################
                ################################################################################################################
            # END SELECT SPLIT METHOD###########################################################################################
            ####################################################################################################################
    
    
            ####################################################################################################################
            # SELECT NUMBER OF LABELS###########################################################################################
            if self.target_config in 'CMSZ':     #change number of labels(c)
                if self.split_method != 'None': print(f'change number of labels(c)')
                while True:
                    if self.split_method == 'mc':
                        self.number_of_labels = len(self.RAW_TARGET_VECTOR)
                        print(f'Split method is multi-column with {self.number_of_labels} columns.  Number of labels = {self.number_of_labels}.')
                        break
    
                    elif self.split_method == 'c2c':
                        print(f'Split method is category-to-category with {len(self.TARGET_UNIQUE_VALUES)} unique categories.')
                        user_select = vui.validate_user_str(f'Use a label for each category(e) or select number of labels(n) > ', 'EN')
                        if user_select == 'E':
                            self.number_of_labels = len(self.TARGET_UNIQUE_VALUES)
                        elif user_select == 'N':
                            self.number_of_labels = vui.validate_user_int('Choose number of labels > ', min=1, max=len(self.TARGET_UNIQUE_VALUES))
    
                    elif self.split_method == 'n2c':
                        print(f'Split method is number-to-category with {len(self.TARGET_UNIQUE_VALUES)} unique values.')
                        self.number_of_labels =  vui.validate_user_int('Choose number of labels > ', min=1,
                                                                                    max=len(self.TARGET_UNIQUE_VALUES))
    
                    elif self.split_method == 'None':
                        print(f'Split method is None with {len(self.TARGET_UNIQUE_VALUES)} unique values.')
                        self.number_of_labels = 1
                        print(f'Number of labels must be 1.')
                        break
    
                    if vui.validate_user_str('Accept number of labels? (y/n) > ', 'YN') == 'Y':
                        break
    
            # END NUMBER OF LABELS##############################################################################################
            ####################################################################################################################
    
            ####################################################################################################################
            # LABEL RULES ######################################################################################################
    
            if self.target_config in 'CLMSZ':     #adjust label rules(l)
                if self.split_method != 'None': print(f'adjust label rules(l)')
                if self.split_method == 'c2c':
                    self.LABEL_RULES = ctc.category_to_category(self.LABEL_RULES, self.number_of_labels, self.TARGET_UNIQUE_VALUES)
                elif self.split_method == 'n2c':
                    self.LABEL_RULES = ntc.number_to_category(self.RAW_TARGET_VECTOR, self.LABEL_RULES, self.number_of_labels)
                elif self.split_method == 'mc':
                    self.LABEL_RULES = ctmc.category_to_multi_column(self.RAW_TARGET_VECTOR, self.LABEL_RULES, self.number_of_labels)
                elif self.split_method == 'None':
                    self.LABEL_RULES = [['None']]

            # END LABEL RULES ##################################################################################################
            ####################################################################################################################
    
            ####################################################################################################################
            # SELECT EVENT VALUES ##############################################################################################
    
            if self.target_config in 'CLMSVZ':    #change event values(v)
                if self.split_method != 'None': print('change event values(v)')
                while True:
                    if self.split_method == 'None':
                        print('No split being done, do not need to set event values.')
                        self.event_value = 0
                        self.negative_value = 0
                        break
    
                    self.event_value = vui.validate_user_int('Enter value for event (usually 1) > ')
                    self.negative_value = vui.validate_user_int('Enter value for negative case (usually 0, -1 for SVM) > ')
    
                    if vui.validate_user_str('Accept event/non-event values setup? (y/n) > ', 'YN') == 'Y':
                        break
    
    
            # END EVENT VALUES #################################################################################################
            ####################################################################################################################

            if self.target_config in 'A':   #'accept target config / continue(a)',
                break
    
            if self.target_config in 'Q':   #quit(q)
                raise Exception(f'USER TERMINATED.')

            ############################################################################################################
            # MENU PRINT HANDLING ######################################################################################


            self.target_config = dmp.DictMenuPrint(self.MENU_OPTIONS, disp_len=140, allowed=allowed_keys.upper()).select('')

            # END MENU PRINT HANDLING ##################################################################################
            ############################################################################################################

        return self.return_fxn()







if __name__ == '__main__':

    # SINGLE
    RAW_TARGET_SOURCE = [['X','O','Y','Z', 'Z','X','O','Y', 'P','X','O','X']]
    RAW_TARGET_SOURCE_HEADER = [['STATUS1']]

    # SOFTMAX
    # RAW_TARGET_SOURCE = [['X','O','Y','Z'],['Z','X','O','Y'],['P','X','O','X']]
    # RAW_TARGET_SOURCE_HEADER = [['STATUS1', 'STATUS2', 'STATUS3']]

    # FLOAT
    # RAW_TARGET_SOURCE = n.array([[1,5,2,6,3,0,3,4,2,3,6,2,1,0,6]], dtype=object)
    # RAW_TARGET_SOURCE_HEADER = [['STATUS1']]

    TARGET_VECTOR = []
    TARGET_VECTOR_HEADER = [[]]
    RAW_TARGET_VECTOR = []

    ##### DF STUFF
    # RAW_TARGET_SOURCE_HEADER = ['a'] #,'b','c','d']
    # RAW_TARGET_SOURCE = p.DataFrame(data=RAW_TARGET_SOURCE.transpose(), columns=RAW_TARGET_SOURCE_HEADER)
    ##### END DF STUFF

    target_config = 'Z'
    standard_config = 'AA'

    target_config, split_method, RAW_TARGET_VECTOR, RAW_TARGET_VECTOR_HEADER, TARGET_VECTOR, \
    TARGET_VECTOR_HEADER, LABEL_RULES, number_of_labels, event_value, negative_value, KEEP_COLUMNS = \
    TargetPrerunConfig(standard_config, target_config, RAW_TARGET_SOURCE, RAW_TARGET_SOURCE_HEADER, TARGET_VECTOR,
                 TARGET_VECTOR_HEADER).config()


    from ML_PACKAGE.GENERIC_PRINT import obj_info as oi
    oi.obj_info(target_config, 'target_config', __name__)
    oi.obj_info(split_method, 'split_method', __name__)
    oi.obj_info(RAW_TARGET_SOURCE, 'RAW_TARGET_SOURCE', __name__)
    oi.obj_info(RAW_TARGET_VECTOR, 'RAW_TARGET_VECTOR', __name__)
    oi.obj_info(RAW_TARGET_VECTOR_HEADER, 'RAW_TARGET_VECTOR_HEADER', __name__)
    oi.obj_info(TARGET_VECTOR, 'TARGET_VECTOR', __name__)
    oi.obj_info(TARGET_VECTOR_HEADER, 'TARGET_VECTOR_HEADER', __name__)
    oi.obj_info(LABEL_RULES, 'LABEL_RULES', __name__)
    oi.obj_info(number_of_labels, 'number_of_labels', __name__)
    oi.obj_info(event_value, 'event_value', __name__)
    oi.obj_info(negative_value, 'negative_value', __name__)
    oi.obj_info(KEEP_COLUMNS, 'KEEP_COLUMNS', __name__)























