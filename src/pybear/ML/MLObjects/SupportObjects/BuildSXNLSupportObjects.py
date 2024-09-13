from MLObjects.SupportObjects import BuildFullSupportObject as bfso




class BuildSXNLSupportObjects:
    '''Data, target, and refvecs must be passed.'''
    def __init__(self, SXNL, data_given_orientation, target_given_orientation, refvecs_given_orientation,
                 quick_vdtypes=False, DATA_HEADER=None, TARGET_HEADER=None, REFVECS_HEADER=None,
                 DATA_MODIFIED_DATATYPES=None, TARGET_MODIFIED_DATATYPES=None, REFVECS_MODIFIED_DATATYPES=None,
                 prompt_to_override=False, bypass_validation=False, calling_module=None, calling_fxn=None):

        self.DATA_SUPPORT_OBJECTS = bfso.BuildFullSupportObject(OBJECT=SXNL[0], object_given_orientation=data_given_orientation,
                OBJECT_HEADER=DATA_HEADER, SUPPORT_OBJECT=None, columns=None, quick_vdtypes=quick_vdtypes,
                MODIFIED_DATATYPES=DATA_MODIFIED_DATATYPES, print_notes=False, prompt_to_override=prompt_to_override,
                bypass_validation=bypass_validation, calling_module=calling_module, calling_fxn=calling_fxn
        ).SUPPORT_OBJECT

        self.TARGET_SUPPORT_OBJECTS = bfso.BuildFullSupportObject(OBJECT=SXNL[1], object_given_orientation=target_given_orientation,
                OBJECT_HEADER=TARGET_HEADER, SUPPORT_OBJECT=None, columns=None, quick_vdtypes=quick_vdtypes,
                MODIFIED_DATATYPES=TARGET_MODIFIED_DATATYPES, print_notes=False, prompt_to_override=prompt_to_override,
                bypass_validation=bypass_validation, calling_module=calling_module, calling_fxn=calling_fxn
        ).SUPPORT_OBJECT

        self.REFVECS_SUPPORT_OBJECTS = bfso.BuildFullSupportObject(OBJECT=SXNL[2], object_given_orientation=refvecs_given_orientation,
                OBJECT_HEADER=REFVECS_HEADER, SUPPORT_OBJECT=None, columns=None, quick_vdtypes=quick_vdtypes,
                MODIFIED_DATATYPES=REFVECS_MODIFIED_DATATYPES, print_notes=False, prompt_to_override=prompt_to_override,
                bypass_validation=bypass_validation, calling_module=calling_module, calling_fxn=calling_fxn
        ).SUPPORT_OBJECT


    def build(self):
        return [self.DATA_SUPPORT_OBJECTS, self.TARGET_SUPPORT_OBJECTS, self.REFVECS_SUPPORT_OBJECTS]


