import pandas as p, numpy as n
from data_validation import validate_user_input as vui
from general_list_ops import list_select as ls
from ML_PACKAGE.standard_configs import BaseObjectConfigTemplate as boct
from read_write_file.generate_full_filename import base_path_select as bps


# CALLED BY ML_PACKAGE.standard_configs.BASE_BIG_MATRIX_standard_configs()
class AABaseBigMatrixConfig(boct.BBMConfigTemplate):
    def __init__(self, standard_config, BBM_build_method, RAW_DATA_NUMPY_OBJECT, RAW_DATA_NUMPY_HEADER):
        super().__init__(standard_config, BBM_build_method, RAW_DATA_NUMPY_OBJECT, RAW_DATA_NUMPY_HEADER)

    # config(self) inherited
    # no_config(self) inherited
    # pack_tuple(self)

    # HAVE TO def build_methods and def run FOR EACH APPLICATION OF THIS TEMPLATE

    def build_methods(self):
        return [
            'STANDARD'
        ]

    # UNIQUE TO THIS APPLICATION OF BBMConfigTemplate ONLY
    def n_appender(self, object_name, col_idx):  # SPECIAL TO AA_RAW_DATA_NUMPY_config
        x = self.RAW_DATA_NUMPY_OBJECT
        y = self.RAW_DATA_NUMPY_HEADER
        _ = []
        for obj in self.RAW_DATA_NUMPY_OBJECT[col_idx]:
            if object_name == 'DAY': _.append(p.to_datetime(obj, format="%Y/%m/%d").day)
            elif object_name == 'MONTH': _.append(p.to_datetime(obj, format="%Y/%m/%d").month)
            elif object_name == 'YEAR': _.append(p.to_datetime(obj, format="%Y/%m/%d").year)
            elif object_name == 'WEEKDAY': _.append(p.to_datetime(obj, format="%Y/%m/%d").day_name()[0:3].upper())
            elif object_name == 'QUARTER': _.append(p.to_datetime(obj, format="%Y/%m/%d").quarter)
            elif 'NAICS' in object_name and len(object_name) == 6:
                _.append(str(self.NAICS_LOOKUP[obj])[:int(object_name[-1])])
            elif object_name == 'NAICS6_SUB': _.append(str(self.NAICS_LOOKUP[obj]))
            elif object_name == 'MARKETCAP': _.append(str(self.MARKETCAP_LOOKUP[obj]))
            else:
                raise ValueErrorf'OBJECT NAME IN AA_RAW_DATA_NUMPY_OBJECT IS NOT RECOGNIZED.')

        self.RAW_DATA_NUMPY_OBJECT = n.insert(x, len(x), _, axis=0)
        self.RAW_DATA_NUMPY_HEADER = n.insert(y, len(y[0]), object_name, axis=1)


    def run(self):
        # ***************************************SPECIFY ATTRIBUTES TO CHOOSE FROM******************************************
        if self.config() == 'NONE':
            self.OBJECT_DF = self.no_config()

        elif self.config() == 'STANDARD':
            ATTRIBUTES = [
                'COMPANY ID', 'PORTAL', 'WEEKDAY', 'DAY', 'MONTH', 'QUARTER', 'YEAR', 'TITLE', 'TYPE',
                'CITY', 'STATE', 'REGION', 'NAICS1', 'NAICS2', 'NAICS3', 'NAICS4', 'NAICS5',
                'NAICS6', 'NAICS6_SUB', 'MARKETCAP'
            ]

            TIME_ATTRIBUTES = ['WEEKDAY', 'DAY', 'MONTH', 'QUARTER', 'YEAR']

            # SELECT ATTRIBUTES THAT GO IN MODEL****************************************************************************************
            while True:
                KEEP = [[]]

                if vui.validate_user_str('\nUse standard attributes? (y/n) > ', 'YN') == 'N':
                    # 1-17-22 TRYING list_custom_select FOR ATTRIBUTES
                    KEEP[0] = ls.list_custom_select(ATTRIBUTES, 'value')
                    # OLD CODE KEEP INCASE list_select DOESNT WORK OUT
                    # [KEEP[0].append(attr) for attr in ATTRIBUTES \
                    #         if vui.validate_user_str(f'Include {attr}? (y/n) > ', 'YN') == 'Y']
                else:
                    KEEP[0] = ['PORTAL', 'WEEKDAY', 'TITLE', 'TYPE', 'STATE', 'NAICS2', 'MARKETCAP']

                print(f'KEPT ATTRIBUTES LOOKS LIKE:')
                print(KEEP)
                if vui.validate_user_str('\nAccept attribute inputs? (y/n) > ', 'YN') == 'Y':
                    break
            # END SELECT ATTRIBUTES*******************************************************************************************************
            # ***************************************************************************************************************************

            ##################################################################################################################
            ##################################################################################################################
            # READ ANY NECESSARY SHEETS IF REQUIRED BY INCLUDED ATTRS, AND BUILD DICTIONARIES #############################

            # ****************************************SET FILE PATH NAME SHEET PARAMETERS FOR LOOKUPS*************************************
            basepath = bps.base_path_select()
            filename = "APPLICATION ANALYSIS - NN.xlsx"

            NAICS_SHEET = "NAICS"
            REGION_SHEET = "REGION"
            MARKETCAP_SHEET = "MARKETCAP"

            print('Pulling AA lookup data......')

            # BUILD NAICS DICTIONARY (COMPANY ID KEY TO NAICS)###############################################################3
            self.NAICS_LOOKUP = {}

            if True in map(lambda x: 'NAICS' in x, KEEP[0]): #LOOK IF ANY NAICS IN KEEP
                NAICS_DF = p.DataFrame(
                                        p.read_excel(basepath + filename, sheet_name=NAICS_SHEET, header=0,
                                        index_col=None,
                                        usecols=[0, 1, 2],
                                        dtype=object)
                                        ).astype('str')  # CONVERT ALL COLUMNS IN NAICS_DF TO STRING

                self.NAICS_LOOKUP = dict((zip(NAICS_DF['COMPANY ID'][0:],
                                                    NAICS_DF['NAICS'] + '-' + NAICS_DF['SUB'])))
            # END BUILD NAICS DICTIONARIES ##############################################3##############################

            # BUILD REGION DICTIONARY (STATE KEYS TO REGION)############################################################
            self.REGION_LOOKUP = {}
            if 'REGION' in KEEP[0]:
                REGION_DF = p.DataFrame(p.read_excel(basepath + filename, sheet_name=REGION_SHEET, header=0,
                                                        index_col=None, usecols=[1, 2]))
                self.REGION_LOOKUP = dict((zip(REGION_DF['ABBR'], REGION_DF['REGION'])))
            # END BUILD REGION DICTIONARY ###################################################################################

            # BUILD MARKETCAP DICTIONARY (COMPANY ID KEYS TO MARKETCAP)###########################################################
            self.MARKETCAP_LOOKUP = {}
            if 'MARKETCAP' in KEEP[0]:
                MARKETCAP_DF = p.DataFrame(p.read_excel(basepath + filename, sheet_name=MARKETCAP_SHEET, header=0,
                                            index_col=None, usecols=[0, 5]))

                self.MARKETCAP_LOOKUP = dict((zip(MARKETCAP_DF['COMPANY ID'], MARKETCAP_DF['MCAP CATEGORY'])))
            # END BUILD MARKETCAP DICTIONARY ###################################################################################

            # END BUILD DICTIONARIES #######################################################################################
            ##################################################################################################################
            ##################################################################################################################
            print('Done.')

            # **********START PROCESSING DATA****************************************************************************************

            # IF INCLUDED, DO DATES THINGS******************************************************************************************
            if True in map(lambda x: x in TIME_ATTRIBUTES, KEEP[0]):  # FANCY WAY TO SEE IF ANY TIME THINGS IN KEEP
                __ = self.RAW_DATA_NUMPY_HEADER[0]
                col_idx = [_ for _ in range(len(__)) if __[_] == 'APP DATE'][0]

                for time_thing in ['DAY','MONTH','YEAR','QUARTER','WEEKDAY']:
                    if time_thing in KEEP[0]:
                        self.n_appender(time_thing, col_idx)

            # IF USING ANY OF THE NAICS*****************************************************************************************
            if len(self.NAICS_LOOKUP) > 0:  # NAICS_LOOKUP ONLY HAS THINGS IN IT IF A NAICS THING WAS PRESENT ABOVE
                __ = self.RAW_DATA_NUMPY_HEADER[0]
                col_idx = [_ for _ in range(len(__)) if __[_] == 'COMPANY ID'][0]
                for naics_idx in range(1, 7):
                    trial_naics = 'NAICS' + str(naics_idx)
                    if trial_naics in KEEP[0]:
                        self.n_appender(trial_naics,col_idx)
                if 'NAICS6_SUB' in KEEP[0]:
                    self.n_appender('NAICS6_SUB', col_idx)
            # END NAICS THINGS******************************************************************************************************

            # IF USING MARKETCAP*****************************************************************************************
            if len(self.MARKETCAP_LOOKUP) > 0:  # MARKETCAP_LOOKUP ONLY len>0 IF MARKETCAP WAS IN KEEP AT THE TIME OF BUILDING THE LOOKUP
                __ = self.RAW_DATA_NUMPY_HEADER[0]
                col_idx = [_ for _ in range(len(__)) if __[_] == 'COMPANY ID'][0]
                self.n_appender('MARKETCAP', col_idx)

            # END MARKETCAP LOOKUP******************************************************************************************************

            # IF STATE IN KEEP, CHECK TO SEE THAT ALL ENTRIES ARE VALID STATES
            if 'STATE' in KEEP[0]:
                from general_text import states as st
                _ = self.RAW_DATA_NUMPY_HEADER
                col_idx = [x for x in range(len(_[0])) if _[0][x]=='STATE'][0]

                __ = self.RAW_DATA_NUMPY_OBJECT
                # APPEND A '-' TO STATE LIST SO THAT IT'S RECOGNIZED, THEN WILL BE CHANGED TO 'UNKNOWN' LATER
                MOD_STATE_LIST = st.states_incl_dc() + ['-']
                if False in map(lambda state: state in MOD_STATE_LIST, __[col_idx]):
                    raise ValueError(f'UNRECOGNIZED ENTRY IN STATE.')

            # APPEND CITY, STATED COLUMN IF CITY IN KEEP***********************************************************************************
            if 'CITY' in KEEP[0]:
                __ = self.RAW_DATA_NUMPY_HEADER
                city_col_idx = [_ for _ in range(len(__[0])) if __[0][_] == 'CITY'][0]
                state_col_idx = [_ for _ in range(len(__[0])) if __[0][_] == 'STATE'][0]

                _ = []
                for idx in range(len(self.RAW_DATA_NUMPY_OBJECT[city_col_idx])):
                    city = self.RAW_DATA_NUMPY_OBJECT[city_col_idx][idx]
                    if str(city).upper() in '-':
                        _.append('UNKNOWN')
                    else:
                        _.append(city.upper() + ', ' + self.RAW_DATA_NUMPY_OBJECT[state_col_idx][idx])

                x = self.RAW_DATA_NUMPY_OBJECT
                self.RAW_DATA_NUMPY_OBJECT = n.insert(x, len(x), _, axis=0)
                self.RAW_DATA_NUMPY_HEADER = n.insert(__, len(__[0]), 'CITY,STATE', axis=1)

                # CHANGE 'CITY' IN KEEP TO 'CITY, STATE' SO THAT THE CORRECT COLUMN IS PULLED IN BUILDING FINAL MATRIX
                KEEP[0][KEEP[0].index('CITY')] = 'CITY,STATE'
            # **********************************************************************************************************************

            # PICK USED COLS OUT OF DATA & HEADER, IN ORDER
            MASK = n.empty((0,1), dtype=object).reshape((1,-1))[0]

            for attr in KEEP[0]:
                # 10/19/22 FOR SOME REASON n.argwhere SIMPLY IS NOT RECOGNIZING EQUALITY, BUT EQUALITY IS RECOGNIZED OUTSIDE OF IT.
                # NOTHING IS FIXING THIS. GO BACK TO FOR LOOP :(
                for header_idx in range(len(self.RAW_DATA_NUMPY_HEADER[0])):
                    if attr == self.RAW_DATA_NUMPY_HEADER[0][header_idx]:
                        MASK = n.insert(MASK, len(MASK), header_idx, axis=0)

            self.RAW_DATA_NUMPY_OBJECT = self.RAW_DATA_NUMPY_OBJECT[MASK.astype(int)]
            self.RAW_DATA_NUMPY_HEADER = n.array(KEEP, dtype=object)

            del MASK

            # CLEAN UP ANY '-' IN DATA, REPLACE W "UNKNOWN"
            for col_idx in range(len(self.RAW_DATA_NUMPY_OBJECT)):
                self.RAW_DATA_NUMPY_OBJECT[col_idx] = \
                    n.where(self.RAW_DATA_NUMPY_OBJECT[col_idx]=='-', 'UNKNOWN', self.RAW_DATA_NUMPY_OBJECT[col_idx])

            return self.RAW_DATA_NUMPY_OBJECT, self.RAW_DATA_NUMPY_HEADER, KEEP

        #END PREPARE BASE BIG MATRIX************************************************************************************
        # ******************************************************************************************************************

        # 10/27/21 STARTED WITH 841 LINES TOTAL FOR AA BBM BUILD & FILTER, CHOPS, ETC.

























