from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.data_read import DataReadConfigRun as drcr

'''accept config / continue(a) select path/filename(f) select package(p) reconfigure all(z) 
filetype/package pair config(b) choose standard config(s)
'''

user_manual_or_std = 'Z'

# CALLED BY class_DataTargetReferenceTestReadBuild.ReferenceVectorsBuild
class RVReadConfigRun(drcr.DataReadConfigRun):
    def __init__(self, user_manual_or_std, standard_config, rv_read_method, DATA_DF):
        self.user_manual_or_std = user_manual_or_std
        self.standard_config = standard_config
        self.method = rv_read_method
        self.DATA_DF = DATA_DF

    def standard_config_source(self):
        from ML_PACKAGE.standard_configs import standard_configs as sc
        self.REFERENCE_VECTORS_DF, self.DATA_DF = \
            sc.rv_read_standard_configs(self.standard_config, self.method, self.DATA_DF)
        return self.REFERENCE_VECTORS_DF, self.DATA_DF

    # INHERITED
    # def config_run(self)

    def final_output(self):
        return self.config_run()