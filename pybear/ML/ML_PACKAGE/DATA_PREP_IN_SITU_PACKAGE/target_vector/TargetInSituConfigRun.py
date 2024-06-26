from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.target_vector import TargetPrerunConfigRun as tpcr
from ML_PACKAGE.DATA_PREP_IN_SITU_PACKAGE.target_vector import TargetInSituRun as tisr, TargetInSituConfig as tisc


class TargetInSituConfigRun(tpcr.TargetPrerunConfigRun):

    # INHERITS
    # __init__()
    # configrun()
    # return_fxn()   (FOR NOW)

    # OVERWRITES
    # config_module()
    # run_module()

    def config_module(self):

        return tisc.TargetInSituConfig(self.standard_config, self.target_config, self.RAW_TARGET_SOURCE,
            self.RAW_TARGET_SOURCE_HEADER, self.TARGET_VECTOR, self.TARGET_SUPOBJS).config()


    def run_module(self):

        return tisr.TargetInSituRun(self.RAW_TARGET_SOURCE, self.RAW_TARGET_SOURCE_HEADER, self.RAW_TARGET_VECTOR,
            self.RAW_TARGET_VECTOR_HEADER, self.TARGET_VECTOR, self.TARGET_SUPOBJS, self.split_method, self.LABEL_RULES,
            self.number_of_labels, self.event_value, self.negative_value, self.KEEP_COLUMNS).run()




if __name__ == '__main__':
    from ML_PACKAGE.GENERIC_PRINT import obj_info as oi
    from MLObjects.SupportObjects import master_support_object_dict as msod, BuildFullSupportObject as bfso

    standard_config = 'AA'
    target_config = 'Z'
    RAW_TARGET_SOURCE = [
        ['X','O','O','X','Y','Z','J','O','X','Z','X','O','Y','Z','A','O','O','X']]
    RAW_TARGET_SOURCE_HEADER = [['TARGET']]
    TARGET_VECTOR = [[]]
    TARGET_VECTOR_HEADER = [[]]

    TargetClass = bfso.BuildFullSupportObject(
                                                OBJECT=RAW_TARGET_SOURCE,
                                                object_given_orientation='COLUMN',
                                                OBJECT_HEADER=RAW_TARGET_SOURCE_HEADER,
                                                SUPPORT_OBJECT=None,
                                                columns=None,
                                                quick_vdtypes=False,
                                                MODIFIED_DATATYPES=['STR'],
                                                print_notes=False,
                                                prompt_to_override=False,
                                                bypass_validation=False,
                                                calling_module='TargetInSituConfigRun',
                                                calling_fxn='tests'
    )

    TARGET_VECTOR = TargetClass.OBJECT
    TARGET_SUPPORT_OBJECTS = TargetClass.SUPPORT_OBJECT
    del TargetClass

    TARGET_VECTOR, TARGET_SUPPORT_OBJECTS, split_method, LABEL_RULES, number_of_labels, event_value, negative_value = \
        TargetInSituConfigRun(standard_config, target_config, RAW_TARGET_SOURCE, RAW_TARGET_SOURCE_HEADER, TARGET_VECTOR,
                              TARGET_SUPPORT_OBJECTS).configrun()


    oi.obj_info(TARGET_VECTOR, 'TARGET_VECTOR', __name__)
    oi.obj_info(TARGET_VECTOR_HEADER, 'TARGET_VECTOR_HEADER', __name__)
    oi.obj_info(split_method, 'split_method', __name__)
    oi.obj_info(LABEL_RULES, 'LABEL_RULES', __name__)
    oi.obj_info(number_of_labels, 'number_of_labels', __name__)
    oi.obj_info(event_value, 'event_value', __name__)
    oi.obj_info(negative_value, 'negative_value', __name__)



