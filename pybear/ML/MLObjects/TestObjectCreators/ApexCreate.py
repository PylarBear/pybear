import inspect, sys
import numpy as np, sparse_dict as sd
from debug import get_module_name as gmn
from MLObjects.SupportObjects import BuildFullSupportObject as bfso, master_support_object_dict as msod, \
    diagnose_support_object_format as dsof
from MLObjects import MLObject as mlo
from MLObjects.TestObjectCreators import ExpandCategoriesTestObjects as ecto
from MLObjects.SupportObjects import FullSupportObjectSplitter as fsos
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv

'''OBJECT, HEADER, AND SUPPORT OBJECTS ARE ATTRIBUTES OF THE CLASS. NOTHING IS RETURNED.'''

# MODULE FOR CREATING DATA, TARGET, REFVECS, OR TEST OBJECTS W RANDOM DATA OR FROM GIVEN OBJECTS


# OUTLINE





# FUNCTIONS #############################################################################################################
# _exception
# validate_full_supobj_or_single_mdtypes
# build_full_supobj
# get_individual_support_objects
# build      OVERWRITTEN IN CHILDREN
# to_row
# to_column
# _transpose
# to_array
# to_sparse_dict
# expand



# PARENT OF CreateFromGiven, CreateFromScratch
class ApexCreate:
    '''Parent class for CreateFromGiven & CreateFromScratch, providing some arg/kwarg validation & methods.'''

    def __init__(self, OBJECT_HEADER, FULL_SUPOBJ_OR_SINGLE_MDTYPES, bypass_validation, name, return_format, allowed_return_format,
                 return_orientation, allowed_return_orientation, override_sup_obj, calling_module, calling_fxn):


        this_module = gmn.get_module_name(str(sys.modules[__name__]))
        self.calling_module = this_module if calling_module is None else calling_module
        self.calling_fxn = '__init__' if calling_fxn is None else calling_fxn


        # HELPER OBJECTS ############################################################################################
        self.LOOKUP_DICT = {k: msod.master_support_object_dict()[k]['position'] for k in
                            msod.master_support_object_dict()}
        # END HELPER OBJECTS ############################################################################################

        self.bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                             self.calling_module, self.calling_fxn, return_if_none=False)

        # NUMBERS AND STRINGS
        if self.bypass_validation:
            self.name = name if not name is None else 'OBJECT'
            self.OBJECT_HEADER = np.array(OBJECT_HEADER, dtype='<U200').reshape((1,-1)) if not OBJECT_HEADER is None else None
            self.return_format = return_format
            self.return_orientation = return_orientation
            self.override_sup_obj = override_sup_obj
            if FULL_SUPOBJ_OR_SINGLE_MDTYPES is None: self.FULL_SUPOBJ_OR_SINGLE_MDTYPES = None
            else: self.FULL_SUPOBJ_OR_SINGLE_MDTYPES = np.array(FULL_SUPOBJ_OR_SINGLE_MDTYPES)

        elif not self.bypass_validation:
            if not isinstance(name, str): self._exception(f'name MUST BE A STRING.', fxn=self.calling_fxn)
            else: self.name = name if not name is None else 'OBJECT'

            hdr_format, self.OBJECT_HEADER = ldv.list_dict_validater(OBJECT_HEADER, 'OBJECT_HEADER')
            if not hdr_format in ['ARRAY', None]:
                self._exception(f'HEADER MUST BE GIVEN AS LIST-TYPE OR None', fxn=self.calling_fxn); del hdr_format

            self.return_format = akv.arg_kwarg_validater(return_format, 'return_format', allowed_return_format,
                                                         self.calling_module, self.calling_fxn)
            self.return_orientation = akv.arg_kwarg_validater(return_orientation, 'return_orientation',
                                                         allowed_return_orientation, self.calling_module, self.calling_fxn)
            self.override_sup_obj = akv.arg_kwarg_validater(override_sup_obj, 'override_sup_obj',
                         [True, False, None], self.calling_module, self.calling_fxn, return_if_none=False)
            if not isinstance(FULL_SUPOBJ_OR_SINGLE_MDTYPES, (np.ndarray, list, tuple)) and not FULL_SUPOBJ_OR_SINGLE_MDTYPES is None:
                self._exception(f'FULL_SUPOBJ_OR_SINGLE_MDTYPES MUST BE PASSED AS A LIST-TYPE OR NOT PASSED AT ALL (None).',
                                fxn=self.calling_fxn)
            else: self.FULL_SUPOBJ_OR_SINGLE_MDTYPES = FULL_SUPOBJ_OR_SINGLE_MDTYPES


        self.mdtype_idx = msod.master_support_object_dict()['MODIFIEDDATATYPES']['position']  # SUPOBJ IS FULL

        # PROCESS FULL_SUPOBJ_OR_SINGLE_MDTYPES INTO A FULL SUPOBJ USING build_full_supobj FXN CALL IN CHILDREN
        # CANT DO THIS IN Apex init BECAUSE self.columns NEEDS TO BE ESTABLISHED IN THE CHILDREN FIRST

        # END __init__ ##################################################################################################
        #################################################################################################################
        #################################################################################################################


    def _exception(self, text, fxn=None):
        fxn = f'.{fxn}()' if not fxn is None else ''
        raise Exception(f'\n*** {self.calling_module}.{fxn} >>> {text} ***\n')


    def validate_full_supobj_or_single_mdtypes(self):
        # MODIFIED DTYPES CAN BE GIVEN AS SINGLE OR FULL SUPPORT OBJECT... STANDARDIZE TO FULL W WHATEVER INFO IS AVAILABLE

        fxn = inspect.stack()[0][3]

        if self.FULL_SUPOBJ_OR_SINGLE_MDTYPES is None:
            self.SUPPORT_OBJECTS = msod.build_empty_support_object(self.columns)

        elif self.FULL_SUPOBJ_OR_SINGLE_MDTYPES is not None:

            # DONT KNOW IF SUP_OBJ IS SINGLE M_DTYPE OR FULL OBJECT, NEED TO GET IT TO FULL
            _shape = np.array(self.FULL_SUPOBJ_OR_SINGLE_MDTYPES).shape
            if len(_shape) == 1:
                self.SUPPORT_OBJECTS = msod.build_empty_support_object(len(self.FULL_SUPOBJ_OR_SINGLE_MDTYPES))
                self.SUPPORT_OBJECTS[self.mdtype_idx] = self.FULL_SUPOBJ_OR_SINGLE_MDTYPES
                del self.FULL_SUPOBJ_OR_SINGLE_MDTYPES
            elif len(_shape) == 2:
                if dsof.diagnose_support_object_format(self.FULL_SUPOBJ_OR_SINGLE_MDTYPES) is True:
                    # LEAVE AS A FULL SUPOBJ
                    self.SUPPORT_OBJECTS = self.FULL_SUPOBJ_OR_SINGLE_MDTYPES
                else:
                    self.SUPPORT_OBJECTS = msod.build_empty_support_object(len(self.FULL_SUPOBJ_OR_SINGLE_MDTYPES[0]))
                    self.SUPPORT_OBJECTS[self.mdtype_idx] = self.FULL_SUPOBJ_OR_SINGLE_MDTYPES[0]
                    del self.FULL_SUPOBJ_OR_SINGLE_MDTYPES
            else:
                self._exception(f'LOGIC MANAGING DTYPE AND/OR SIZE OF FULL_SUPOBJ_OR_SINGLE_MDTYPES IS FAILING', fxn=fxn)
            del _shape

        # self.SUPPORT_OBJECTS IS IMPLICITLY RETURNED THRU self.SUPPORT_OBJECTS


    def build_full_supobj(self):

        fxn = inspect.stack()[0][3]

        # IF HEADER IS None, ALLOW bfso TO BUILD A DUMMY HEADER THEN OVERWRITE LATER.
        # ApexSupportObjectHandling HANDLES VALIDATION OF OBJECT, HEADER, & SUPOBJ FORMAT, AS WELL AS SIZE CONGRUENCY

        # BuildFullSupportObject #############################################################################################
        # 3/11/23 IF PASS ARRAY OF msod.empty_value (CURRENTLY str(None)) TO bfso MODIFIED_DATATYPES KWARG, DOESNT RECOGNIZE DTYPE
        # SO IF A VECTOR OF empty_value (AS MIGHT BE GENERATED BY msod.build_empty_support_object) IS TO BE PASSED, CHANGE IT
        # TO None, OTHERWISE ALLOW THE VECTOR TO PASS

        SupportObjectGenerator = bfso.BuildFullSupportObject(OBJECT=self.OBJECT,
                                     object_given_orientation=self.given_orientation,   # THIS ISNT AVAILABLE IN PARENT, IS MADE IN CHILDREN
                                     OBJECT_HEADER=self.OBJECT_HEADER,
                                     # IF PASSED, MUST BE GIVEN AS FULL SUPOBJ
                                     SUPPORT_OBJECT=self.SUPPORT_OBJECTS,
                                     columns=self.columns,
                                     quick_vdtypes=False,
                                     MODIFIED_DATATYPES=None if msod.is_empty_getter(supobj_idx=self.mdtype_idx, supobj_name=None,
                                     SUPOBJ=self.SUPPORT_OBJECTS, calling_module=self.calling_module, calling_fxn=self.calling_fxn) else self.SUPPORT_OBJECTS[self.mdtype_idx],
                                     print_notes=False,
                                     prompt_to_override=self.override_sup_obj,
                                     bypass_validation=self.bypass_validation,
                                     calling_module=self.calling_module,
                                     calling_fxn=fxn
        )

        self.SUPPORT_OBJECTS = SupportObjectGenerator.SUPPORT_OBJECT
        del SupportObjectGenerator


    def get_individual_support_objects(self):  # CALLED BY init IN CHILDREN (CreateFromGiven, CreateFromScratch)
        # ASSIGN INDIVIDUAL SUPPORT OBJECTS
        # fsos INSTANTIATES self.SUPPORT_OBJECTS ,self.OBJECT_HEADER, self.VALIDATED_DATATYPES, self.VALIDATED_DATATYPES,
        # self.FILTERING, self.MIN_CUTOFFS, self.USE_OTHER, self.START_LAG, self.END_LAG, self.SCALING
        fsos.FullSupObjSplitter.__init__(self, self.SUPPORT_OBJECTS, bypass_validation=self.bypass_validation)
        self.KEEP = self.OBJECT_HEADER[0].copy()
        # DONT DO self.CONTEXT HERE!  SEPARATE HANDLING IN CHILDREN


    def build(self):
        # OVERWROTE IN CHILDREN
        pass


    def to_row(self):
        fxn = inspect.stack()[0][3]

        self.OBJECT = mlo.MLObject(self.OBJECT, self.current_orientation, name=self.name, return_orientation='ROW',
                      return_format=self.current_format, bypass_validation=self.bypass_validation,
                      calling_module=self.calling_module, calling_fxn=fxn).return_as_row()

        self.current_orientation = 'ROW'


    def to_column(self):
        fxn = inspect.stack()[0][3]

        self.OBJECT = mlo.MLObject(self.OBJECT, self.current_orientation, name=self.name, return_orientation='COLUMN',
                      return_format=self.current_format, bypass_validation=self.bypass_validation,
                      calling_module=self.calling_module, calling_fxn=fxn).return_as_column()

        self.current_orientation = 'COLUMN'


    def _transpose(self):
        fxn = inspect.stack()[0][3]

        self.OBJECT = mlo.MLObject(self.OBJECT, self.current_orientation, name=self.name, return_orientation='COLUMN',
                                    return_format=self.current_format, bypass_validation=self.bypass_validation,
                                    calling_module=self.calling_module, calling_fxn=fxn).get_transpose()

        self.current_orientation = 'COLUMN' if self.current_orientation == 'ROW' else 'ROW'


    def to_array(self):
        # DONT USE MLObject HERE, TO ALLOW FOR DIFFERENT np.dtype

        if True in (_ in list(msod.mod_text_dtypes().values()) for _ in self.MODIFIED_DATATYPES):
            # IF STR-TYPES IN OBJECT, CANNOT BE SD, MUST ALREADY BE ARRAY
            pass
        else:
            # IF OBJECT ONLY CONTAINS NUMBERS
            if self.current_format == 'ARRAY': pass
            elif self.current_format == 'SPARSE_DICT':
                if not True in (_ in ['INT', 'FLOAT'] for _ in self.MODIFIED_DATATYPES):
                    self.OBJECT = sd.unzip_to_ndarray_int8(self.OBJECT)[0]
                elif 'FLOAT' not in self.MODIFIED_DATATYPES:
                    self.OBJECT = sd.unzip_to_ndarray_int32(self.OBJECT)[0]
                else: self.OBJECT = sd.unzip_to_ndarray_float64(self.OBJECT)[0]

        self.current_format, self.is_list, self.is_dict = 'ARRAY', True, False


    def to_sparse_dict(self):
        # DONT USE MLObject HERE, TO ALLOW FOR DIFFERENT SD VALUE DTYPES
        if True in (_ in list(msod.mod_text_dtypes().values()) for _ in self.MODIFIED_DATATYPES):
            # IF STR-TYPES IN OBJECT, CANNOT BE OR BECOME SD, MUST ALREADY BE ARRAY
            print(f'\n*** OBJECT CONTAINS NON-NUMERIC TYPES, CANNOT BE CONVERTED TO SPARSE DICT ***\n')
        else:
            # IF OBJECT ONLY CONTAINS NUMBERS
            if self.current_format == 'SPARSE_DICT': pass
            elif self.current_format == 'ARRAY':
                # IF NO FLOATS AND NO STR-TYPES, MUST ALL BE BIN AND INT
                if 'FLOAT' not in self.MODIFIED_DATATYPES: self.OBJECT = sd.zip_list_as_py_int(self.OBJECT)
                else: self.OBJECT = sd.zip_list_as_py_float(self.OBJECT)

            self.current_format, self.is_list, self.is_dict = 'SPARSE_DICT', False, True


    def expand(self, expand_as_sparse_dict=None, auto_drop_rightmost_column=False):
        '''expand() under CreateNumerical because it is the apex parent class. Necessary for CreateCategoricalNumpy.'''

        # ALL VALIDATION OF THESE KWARGS SHOULD BE HANDLED BY Expand

        fxn = inspect.stack()[0][3]

        # SENTINEL
        # IF IS CURRENTLY A DICT, EXPANDS AS DICT, IF LIST, EXPANDS AS LIST
        # (IF IS DICT PRE-EXPANSION, MUST BE ALL NUMBERS, AND EXPANSION DOES NOTHING)
        expand_as_sparse_dict = akv.arg_kwarg_validater(expand_as_sparse_dict, 'expand_as_sparse_dict',
                                [True, False, None], self.calling_module, fxn, return_if_none=self.is_dict)

        auto_drop_rightmost_column = akv.arg_kwarg_validater(auto_drop_rightmost_column,
                                         'auto_drop_rightmost_column', [True, False], self.calling_module, fxn)


        ExpandedObjects = ecto.ExpandCategoriesTestObjects(
                                                     self.OBJECT,
                                                     self.SUPPORT_OBJECTS,
                                                     CONTEXT=None,  # DONT KNOW IF IS NEEDED
                                                     KEEP=None,  # DONT KNOW IF IS NEEDED
                                                     data_given_orientation=self.current_orientation,
                                                     data_return_orientation=self.current_orientation,
                                                     data_return_format='SPARSE_DICT' if expand_as_sparse_dict else 'ARRAY',
                                                     auto_drop_rightmost_column=auto_drop_rightmost_column,
                                                     bypass_validation=True,
                                                     calling_module=self.calling_module,
                                                     calling_fxn=fxn
        )


        # MUST GET EXPANDED OBJECTS OUT OF Expand BY REASSIGNING THIS CLASS'S ATTRIBUTES TO Expanded's ATTRIBUTES (NOTHING IS RETURNED FROM Expand)
        self.OBJECT = ExpandedObjects.DATA_OBJECT
        self.SUPPORT_OBJECTS = ExpandedObjects.SUPPORT_OBJECTS

        self.get_individual_support_objects()
        self.KEEP = self.OBJECT_HEADER[0].copy()
        self.CONTEXT += ExpandedObjects.CONTEXT_HOLDER

        self.is_expanded = True
        self.is_list, self.is_dict = not expand_as_sparse_dict, expand_as_sparse_dict

        del ExpandedObjects





if __name__ == '__main__':
    pass
    # DONT TEST THIS MODULE.  THIS DOESNT MAKE ANYTHING.  ONLY HOLDS COMMON VALIDATION AND METHODS FOR CHILDREN.













































