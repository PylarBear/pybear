from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.data_read import DataReadConfigRun as drcr

'''accept config / continue(a) select path/filename(f) select package(p) reconfigure all(z) 
filetype/package pair config(b) choose standard config(s)
'''

user_manual_or_std = 'Z'

# CALLED BY class_DataTargetReferenceTestReadBuildDF.RawTargetBuild
class RawTargetReadConfigRun(drcr.DataReadConfigRun):
    def __init__(self, user_manual_or_std, standard_config, raw_target_read_method, DATA_DF):
        super().__init__(user_manual_or_std, standard_config, raw_target_read_method, DATA_DF)
        self.method = raw_target_read_method

    def standard_config_source(self):
        from ML_PACKAGE.standard_configs import standard_configs as sc
        self.RAW_TARGET_DF, self.DATA_DF = \
            sc.raw_target_read_standard_configs(self.standard_config, self.method, self.DATA_DF)
        return self.RAW_TARGET_DF, self.DATA_DF

    # INHERITED
    # def config_run(self)

    def final_output(self):
        return self.config_run()