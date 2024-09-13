from ML_PACKAGE.standard_configs import BaseObjectConfigTemplate as boct


# CALLED BY ML_PACKAGE.standard_configs.BASE_REFERENCE_VECTORS_standard_configs
class AABaseRVConfig(boct.BaseRVConfigTemplate):
    def __init__(self, standard_config, rv_build_method, RAW_RV_NUMPY_OBJECT, RAW_RV_NUMPY_HEADER):
        super().__init__(standard_config, rv_build_method, RAW_RV_NUMPY_OBJECT, RAW_RV_NUMPY_HEADER)
        self.object_name = 'RV'

        # self.RAW_RV_NUMPY_OBJECT = SUPER_RAW_NUMPY_TUPLE[4]
        # self.RAW_RV_NUMPY_HEADER = SUPER_RAW_NUMPY_TUPLE[5]

        # HAVE TO def build_methods and def run FOR EACH APPLICATION OF THIS TEMPLATE

    # INHERITED
    # config(self)
    # no_config(self)

    def build_methods(self):
        return [
            'STANDARD'
        ]

    def run(self):
        pass
        return self.RAW_RV_NUMPY_OBJECT, self.RAW_RV_NUMPY_HEADER, 'DUM'
    # DUM HERE BECAUSE PARENT CLASS (BBM CONFIG) REQUIRES 3 SPOTS (FINAL FOR "KEEP") BUT AS OF 10-31-21 NOT
    # USING "KEEP" FOR ANY OTHER BASE BUILDS, SO JUST PLACEHOLDER

    # base_big_matrix_config_run IS PARENT FOR TARGET, RV, & TEST config_runs, THE ASSOCIATED config_run()
    # FXN IS USED STRAIGHT BY ALL 4 (NOT REDEFINED IN CHILDREN) AND REQUIRED 3 OUTPUTS, THE LAST TO HANDLE
    # THE "KEEP" OBJECT FROM BBM.


