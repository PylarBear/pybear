import sys, inspect, warnings
from MLObjects.SupportObjects import ApexSupportObjectHandling as asoh
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv

# BEAR THIS ISNT CALLED ANYWHERE.  3/13/23 TRIED TO USE THIS FOR VALIDATION IN FullSupportObjectSplitter (FSOS), BUT RAN INTO
# PROBLEMS WITH vfso's CALLS TO ApexSupportObjectHandling WHEN OBJECT IS None (OBJECT NOT PASSED TO FSOS, THEREFORE NOT TO VFSO EITHER)
# JUST ANOTHER THING TO WORK ON SOMEDAY

def validate_full_support_object(SUPPORT_OBJECT, OBJECT=None, object_given_orientation=None, OBJECT_HEADER=None,
                                 allow_override=False):

    # IF SUPPORT_OBJECT IS FULL, CAN PASS ANY SUPPORT NAME

    _module = gmn.get_module_name(str(sys.modules[__name__]))
    fxn = inspect.stack()[0][3]

    object_given_orientation = akv.arg_kwarg_validater(object_given_orientation, 'object_given_orientation',
                                                       ['ROW', 'COLUMN', None], _module, fxn)
    allow_override = akv.arg_kwarg_validater(allow_override, 'allow_override', [True, False], _module, fxn)

    _DumClass = asoh.ApexSupportObjectHandle(OBJECT=OBJECT,
                                    object_given_orientation=object_given_orientation,
                                    OBJECT_HEADER=OBJECT_HEADER,
                                    SUPPORT_OBJECT=SUPPORT_OBJECT,
                                    prompt_to_override=False,
                                    return_support_object_as_full_array = True,
                                    bypass_validation = False,
                                    calling_module = None,
                                    calling_fxn = None
    )


    _DumClass.validate_allowed(kill=not allow_override, fxn=fxn)

    _DumClass.validate_against_objects(kill=not allow_override, fxn=fxn)

    del _DumClass



