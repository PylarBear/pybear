from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.target_vector import TargetPrerunConfig as tpc
from ML_PACKAGE.standard_configs import standard_configs as sc
from MLObjects.SupportObjects import master_support_object_dict as msod


# CALLED by TargetInSituConfigRun
class TargetInSituConfig(tpc.TargetPrerunConfig):

        # INHERITS
        # __init__()
        # config()
        # target_cmds()  # inherited for now
        # target_str()  # inherited for now
        # return_fxn()   # FOR NOW

        # OVERWRITES
        # standard_config_module()  (1-12-22 EVEN THO IT DOESNT NEED TO, JUST USING 1 std_config FOR NOW, FOR FUTURE MAYBE)

        def standard_config_module(self):
            # BEAR AS OF 1-12-22 GOING WITH 1 STANDARD CONFIG SETUP --- IN THE FUTURE MAY DO
            # TARGET_VECTOR_Prerun_standard_configs & TARGET_VECTOR_InSitu_standard_configs
            return sc.TARGET_VECTOR_standard_configs(self.standard_config, self.target_config, self.RAW_TARGET_SOURCE,
                    self.RAW_TARGET_SOURCE_HEADER, self.TARGET_VECTOR, self.TARGET_SUPOBJS[msod.QUICK_POSN_DICT()["HEADER"]])








