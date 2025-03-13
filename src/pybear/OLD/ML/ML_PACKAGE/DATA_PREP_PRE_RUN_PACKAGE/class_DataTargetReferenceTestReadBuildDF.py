import inspect
from data_validation import validate_user_input as vui
from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.data_read import DataReadConfigRun as drcr
from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.raw_target_read import raw_target_read_config_run as rtrcr
from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.rv_read import rv_read_config_run as rrcr


# TEMPLATE FOR DataBuild (DATA_DF), ReferenceVectorsBuild(REFERENCE_VECTORS_DF), RawTargetBuild(RAW_TARGET_DF),
#   TestDataBuild (TEST_DATA_DF)
class TargetReferenceDataReadBuild():
    def __init__(self, standard_config, object_name):
        self.standard_config = standard_config
        self.object_name = object_name
        print(f'\nCONSTRUCTING {self.object_name}')

    # PROMPT USER TO CHOOSE STANDARD CONFIG OR OVERRIDE
    def user_select_config(self):
        if self.standard_config == 'MANUAL ENTRY' or self.manual_config == 'Y':
            self.user_manual_or_std = vui.validate_user_str('Select a standard config(s) or manual entry(z) > ', 'SZ')
        elif self.manual_config == 'N':
            self.user_manual_or_std = 'S'

        # if self.standard_config != 'MANUAL ENTRY':
        #     self.user_manual_or_std = vui.validate_user_str(
        #     f'Load {self.standard_config} {self.object_name} config(s) or override with manual load(z)? > ', 'SZ')
        # else: self.user_manual_or_std = 'Z'

        return self.user_manual_or_std


    # PERFORM THE INITIAL PULL OF THE OBJECT (CURRENTLY BEING PULLED AS A DF)
    def read_object_from_file(self):

        try:
            self.DF, DATA_DF = self.config_run_file()
            # config_run_file NOT AVAILABLE IN TEMPLATE, BUILT IN EACH OF THE SUBCLASSES

        except Exception as e1:
            raise IOError(f'exception while trying to execute FILE READ --- {e1}\n')

        return self.DF, self.DATA_DF


    # ASK USER IF THEY WANT TO DROP COLUMNS FROM OBJECT
    def drop_columns_yn(self):
        self.drop_columns_yn = vui.validate_user_str(f'Drop columns from DATA DF? (y/n) > ', 'YN')
        return self.drop_columns_yn


    # SELECT COLUMN FUNCTION
    def user_select_columns(self, DF_OBJECT, verb):
        while True:
            OLD_COLUMNS = ['DONE','ALL'] + list(DF_OBJECT.keys())
            SELECTED_COLUMNS = []
            while True:
                [print(f'{idx}) {OLD_COLUMNS[idx]}') for idx in range(len(OLD_COLUMNS))]
                selected_column = vui.validate_user_int(
                    f'\nSelect columns to {verb} {self.object_name} (enter number) select 0 when done > ',min=0, max=len(OLD_COLUMNS)-1
                )
                if selected_column == 0:
                    break

                if selected_column == 1:
                    SELECTED_COLUMNS = list(DF_OBJECT.keys())
                    break

                if OLD_COLUMNS[selected_column] != '' and OLD_COLUMNS[selected_column] not in SELECTED_COLUMNS:
                    SELECTED_COLUMNS.append(OLD_COLUMNS[selected_column])
                    OLD_COLUMNS[selected_column] = ''
                    print(f'\nSELECTED TO DROP SO FAR:')
                    print(SELECTED_COLUMNS)
                    print('')

            print(f'\nFINAL SELECTIONS TO {verb.upper()} {self.object_name}:')
            print(SELECTED_COLUMNS)

            if vui.validate_user_str(f'\nAccept selections? (y/n) > ', 'YN') == 'Y':

                if 'TargetBuild' in str(inspect.stack()[2][4]) and verb=='keep for' and len(SELECTED_COLUMNS) == 0:
                    print(f'\n*** TARGET OBJECT CANNOT BE EMPTY ***\n')
                    continue
                else:
                    break

        return SELECTED_COLUMNS


    # DROP SELECTED COLUMNS FROM DF OBJECT
    def initial_column_drop(self, DF_OBJECT):
        print(f'\nDROPPING COLUMNS FROM {self.object_name}')
        return DF_OBJECT.drop(columns=self.user_select_columns(DF_OBJECT, 'drop from'))


    # DROP SELECTED COLUMNS FROM DF OBJECT
    def later_column_drop(self, DF_OBJECT):
        return DF_OBJECT.drop(columns=self.SELECTED_COLUMNS)


    def build_object(self, DATA_DF):  # THIS IS HERE FOR TARGET & RV , DataBuild OVERWRITES IT FOR ITSELF

        if self.user_manual_or_std == 'Z':
            if vui.validate_user_str(f'Read {self.object_name} from file(f) or from DATA_DF(p) > ', 'FP') == 'F':
                self.DF, DATA_DF = self.read_object_from_file()

            else:
                print(f'\nSelect column(s) from DATA DF for {self.object_name}')
                self.SELECTED_COLUMNS = self.user_select_columns(DATA_DF, 'keep for')
                self.DF = DATA_DF[self.SELECTED_COLUMNS]
                if self.drop_columns_yn() == 'Y':
                    DATA_DF = self.later_column_drop(DATA_DF)
                    # DATA_DF = DATA_DF.drop(SELECTED_COLUMNS)
        else:
            self.DF, DATA_DF = self.config_run_file()

        return self.DF, DATA_DF


class DataBuild(TargetReferenceDataReadBuild):
    def __init__(self, standard_config, data_manual_config, data_read_method, DATA_DF, object_name='DATA'):
        super().__init__(standard_config, object_name)
        self.standard_config = standard_config
        self.manual_config = data_manual_config
        self.method = data_read_method
        self.object_name = object_name
        self.user_manual_or_std = self.user_select_config()
        self.DATA_DF = DATA_DF


    def config_run_file(self):
        self.DF, self.DATA_DF = drcr.DataReadConfigRun(
            self.user_manual_or_std, self.standard_config, self.method, self.DATA_DF).final_output()
        return self.DF, self.DATA_DF


    def build_object(self):  # OVERWRITES build_object() FROM PARENT CLASS
        self.DF, self.DATA_DF = self.read_object_from_file()
        # self.DF is DUM PLACEHOLDER, HAVE TO RETURN 2 THINGS HERE TO SHARE THE SAME PARENT CLASS AS RV & RT
        return self.DF, self.DATA_DF



class RawTargetBuild(TargetReferenceDataReadBuild):
    def __init__(self, standard_config, raw_target_manual_config, raw_target_read_method, DATA_DF, object_name='RAW TARGET SOURCE'):
        super().__init__(standard_config, object_name)
        self.standard_config = standard_config
        self.manual_config = raw_target_manual_config
        self.raw_target_read_method = raw_target_read_method
        self.object_name = object_name
        self.user_manual_or_std = self.user_select_config()
        self.DATA_DF = DATA_DF

    def config_run_file(self):
        return rtrcr.RawTargetReadConfigRun(
            self.user_manual_or_std, self.standard_config, self.raw_target_read_method, self.DATA_DF).final_output()


# CALLED NY NN
class ReferenceVectorsBuild(TargetReferenceDataReadBuild):
    def __init__(self, standard_config, rv_manual_config, rv_read_method, DATA_DF, object_name='REFERENCE VECTORS'):
        super().__init__(standard_config, object_name)
        self.standard_config = standard_config
        self.manual_config = rv_manual_config
        self.method = rv_read_method
        self.object_name = object_name
        self.user_manual_or_std = self.user_select_config()
        self.DATA_DF = DATA_DF

    def config_run_file(self):
        return rrcr.RVReadConfigRun(
            self.user_manual_or_std, self.standard_config, self.method, self.DATA_DF).final_output()






