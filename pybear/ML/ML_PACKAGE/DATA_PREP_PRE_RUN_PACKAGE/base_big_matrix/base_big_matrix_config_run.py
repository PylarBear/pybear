import pandas as pd
from data_validation import validate_user_input as vui
from ML_PACKAGE.GENERIC_PRINT import print_post_run_options as ppro
from ML_PACKAGE.standard_configs import standard_configs as sc


# CALLED BY class BBMBuild (INSIDE class_BaseBMRVTVBuild)
# PARENT OF BaseRawTargetConfigRun, BaseRVConfigRun, BaseTMConfigRun
class BBMConfigRun:
    def __init__(self, standard_config, user_manual_or_standard, BBM_build_method, object_name, RAW_DATA_NUMPY_OBJECT,
                 RAW_DATA_NUMPY_HEADER):

        self.standard_config = standard_config
        self.user_manual_or_std = user_manual_or_standard
        self.method = BBM_build_method
        self.object_name = object_name
        self.NUMPY_OBJECT = RAW_DATA_NUMPY_OBJECT
        self.NUMPY_HEADER = RAW_DATA_NUMPY_HEADER
        print(f'\nCONSTRUCTING {self.object_name}')


    def standard_config_source(self):
        return sc.BASE_BIG_MATRIX_standard_configs(self.standard_config, self.method, self.NUMPY_OBJECT, self.NUMPY_HEADER)


    def print_results(self):
        _ = self.NUMPY_OBJECT
        __ = self.NUMPY_HEADER

        print(f'\n{self.object_name} pre-filter stats:')
        print(
            f'{len(_)} columns and {len(_[0])} rows not counting header')

        display_columns = 10
        print(f'\nFinal NUMPY {self.object_name} as dataframe[:{display_columns}][:20] for display only:')
        TEST_DF = pd.DataFrame(data=_.transpose(), columns=__[0])
        print(TEST_DF[[_ for _ in TEST_DF][:display_columns]].head(20))


    def menu_options(self):
        return ['accept config / continue(a)',
                'bypass(b)',
                'placeholder(f)',
                'placeholder(p)',
                'placeholder(s)',
                'reconfigure all(z)'
                ]


    def config_run(self):
        while True:
            while True:
                # CURRENT #############################################
                if self.user_manual_or_std == 'S':
                    self.NUMPY_OBJECT, self.NUMPY_HEADER, KEEP = self.standard_config_source()
                    if self.method != '':  # IF THERE WAS A STANDARD BUILD USED, SKIP OUT, IF NOT STAY IN LOOP
                        self.user_manual_or_std = 'A'
                # END CURRENT ##########################################

                if self.user_manual_or_std == 'B':
                    # self.NUMPY_OBJECT = self.NUMPY_OBJECT
                    # self.NUMPY_HEADER = self.NUMPY_HEADER
                    KEEP = self.NUMPY_HEADER   #[0][0]
                    self.user_manual_or_std = 'A'

                # if self.user_manual_or_std in 'FZ':
                #     pass
                #
                # if self.user_manual_or_std in 'PFZ':
                #     pass
                if self.user_manual_or_std in 'BPFZ':
                    print(f'MANUAL {self.object_name} CONFIG NOT AVAILABLE AT THIS TIME :(')

                # SHOW RESULTS
                self.print_results()

                if self.user_manual_or_std == 'A':
                    break

                ppro.TopLevelMenuPrint(self.menu_options(), 'ABFPSZ')

                self.user_manual_or_std = vui.validate_user_str(' > ', 'ABFPSZ')

            _ = input(f'\nPreview of {self.object_name} config and build.  Hit ENTER to coninue > ')
            break
            # if vui.validate_user_str(f'\nAccept {self.object_name} config and build? (y/n) > ', 'YN') == 'Y':
            #     break

        return self.NUMPY_OBJECT, self.NUMPY_HEADER, KEEP
