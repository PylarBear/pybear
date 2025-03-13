import sys
from copy import deepcopy
import numpy as np
from debug import get_module_name as gmn
from MLObjects.SupportObjects import ValidatedDatatypes as vd, ModifiedDatatypes as md, Filtering as f, \
    MinCutoffs as mc, UseOther as uo, StartLag as sl, EndLag as el, Scaling as s
from MLObjects.SupportObjects import master_support_object_dict as msod


class BuildFullSupportObject:
    def __init__(self,
                 OBJECT=None,
                 object_given_orientation=None,
                 OBJECT_HEADER=None,
                 SUPPORT_OBJECT=None,  # IF PASSED, MUST BE GIVEN AS FULL SUPOBJ
                 columns=None,  # THE INTENT IS THAT THIS IS ONLY GIVEN IF OBJECT AND SUPPORT_OBJECT ARE NOT
                 quick_vdtypes=False,
                 MODIFIED_DATATYPES=None,
                 print_notes=False,
                 prompt_to_override=False,
                 bypass_validation=True,
                 calling_module=None,
                 calling_fxn=None):

        this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'
        calling_module = calling_module if not calling_module is None else this_module
        calling_fxn = calling_fxn if not calling_fxn is None else calling_fxn


        # OBJECT AND SUPPORT_OBJECT WILL BE UPDATED BY THE INDIV CLASSES. THE OTHER KWARGS WONT CHANGE.
        OTHER_KWARGS = {'object_given_orientation':object_given_orientation,
                         'columns':columns,
                         'OBJECT_HEADER':OBJECT_HEADER,
                         'prompt_to_override':prompt_to_override,
                         'return_support_object_as_full_array':True,
                         'bypass_validation':bypass_validation,
                         'calling_module':calling_module,
                         'calling_fxn':calling_fxn
                         }

        # NEW_OTHER_KWARGS = deepcopy(OTHER_KWARGS); del NEW_OTHER_KWARGS['OBJECT_HEADER']
        # _Header = h.Header(OBJECT=OBJECT, SUPPORT_OBJECT=SUPPORT_OBJECT if not SUPPORT_OBJECT is None else OBJECT_HEADER, **NEW_OTHER_KWARGS)
        # self.OBJECT, self.SUPPORT_OBJECT = _Header.OBJECT, _Header.SUPPORT_OBJECT

        # del _Header

        _Validated = vd.ValidatedDatatypes(OBJECT=OBJECT, SUPPORT_OBJECT=SUPPORT_OBJECT, **OTHER_KWARGS,
                       quick_vdtypes=quick_vdtypes, MODIFIED_DATATYPES=MODIFIED_DATATYPES, print_notes=print_notes)
        self.OBJECT, self.SUPPORT_OBJECT = _Validated.OBJECT, _Validated.SUPPORT_OBJECT

        del _Validated

        _Modified = md.ModifiedDatatypes(OBJECT=self.OBJECT, SUPPORT_OBJECT=self.SUPPORT_OBJECT, **OTHER_KWARGS)
        self.OBJECT, self.SUPPORT_OBJECT = _Modified.OBJECT, _Modified.SUPPORT_OBJECT

        del _Modified

        # MUST TAKE prompt_to_override OUT OF OTHER_KWARGS, Filtering & Scaling DONT TAKE IT
        NEW_OTHER_KWARGS = deepcopy(OTHER_KWARGS); del NEW_OTHER_KWARGS['prompt_to_override']
        _Filtering = f.Filtering(OBJECT=self.OBJECT, SUPPORT_OBJECT=self.SUPPORT_OBJECT, **NEW_OTHER_KWARGS)
        self.OBJECT, self.SUPPORT_OBJECT = _Filtering.OBJECT, _Filtering.SUPPORT_OBJECT

        del _Filtering

        _MinCutoff = mc.MinCutoffs(OBJECT=self.OBJECT, SUPPORT_OBJECT=self.SUPPORT_OBJECT, **OTHER_KWARGS)
        self.OBJECT, self.SUPPORT_OBJECT = _MinCutoff.OBJECT, _MinCutoff.SUPPORT_OBJECT
        # DONT DELETE!

        _UseOther = uo.UseOther(OBJECT=self.OBJECT, SUPPORT_OBJECT=self.SUPPORT_OBJECT, **OTHER_KWARGS)
        self.OBJECT, self.SUPPORT_OBJECT = _UseOther.OBJECT, _UseOther.SUPPORT_OBJECT

        del _UseOther

        _StartLag = sl.StartLag(OBJECT=self.OBJECT, SUPPORT_OBJECT=self.SUPPORT_OBJECT, **OTHER_KWARGS)
        self.OBJECT, self.SUPPORT_OBJECT = _StartLag.OBJECT, _StartLag.SUPPORT_OBJECT
        # DONT DELETE!

        _EndLag = el.EndLag(OBJECT=self.OBJECT, SUPPORT_OBJECT=self.SUPPORT_OBJECT, **OTHER_KWARGS)
        self.OBJECT, self.SUPPORT_OBJECT = _EndLag.OBJECT, _EndLag.SUPPORT_OBJECT

        del _EndLag

        _Scaling = s.Scaling(OBJECT=self.OBJECT, SUPPORT_OBJECT=self.SUPPORT_OBJECT, **NEW_OTHER_KWARGS)
        self.OBJECT, self.SUPPORT_OBJECT = _Scaling.OBJECT, _Scaling.SUPPORT_OBJECT
        # DONT DELETE!

        del NEW_OTHER_KWARGS











    # UNIQUE FXNS #################
    def apply_min_cutoff(self):
        # NEED TO HAVE MIN_CUTOFF AND USE_OTHER
        self._MinCutoff(OBJECT=self.OBJECT, SUPPORT_OBJECT=self.SUPPORT_OBJECT, **self.OTHER_KWARGS).apply()
        self.OBJECT, self.SUPPORT_OBJECT = self._MinCutoff.OBJECT, self._MinCutoff.SUPPORT_OBJECT


    def apply_lag(self):
        # NEED TO HAVE STARTLAG AND ENDLAG
        self._StartLag(OBJECT=self.OBJECT, SUPPORT_OBJECT=self.SUPPORT_OBJECT, **self.OTHER_KWARGS).apply()
        self.OBJECT, self.SUPPORT_OBJECT = self._StartLag.OBJECT, self._StartLag.SUPPORT_OBJECT


    def apply_scaling(self):
        self._Scaling(OBJECT=self.OBJECT, SUPPORT_OBJECT=self.SUPPORT_OBJECT, **self.OTHER_KWARGS).scale()
        self.OBJECT, self.SUPPORT_OBJECT = self._Scaling.OBJECT, self._Scaling.SUPPORT_OBJECT






























if __name__ == '__main__':
    from general_data_ops import create_random_sparse_numpy as crsn
    from MLObjects.TestObjectCreators import test_header as th

    _cols = 100
    _rows = 110
    _orientation = 'ROW'

    print(f'CREATING TEST OBJECT AND HEADER...')
    OBJECT = crsn.create_random_sparse_numpy(-9,10,
                 (_cols if _orientation=='COLUMN' else _rows, _rows if _orientation=='COLUMN' else _cols),
                 _sparsity=50, _dtype=np.int8)

    # OBJECT = sd.zip_list_as_py_int(OBJECT)

    HEADER = th.test_header(_cols)
    print(f'Done.\n')

    SUPPORT_OBJECT = [['AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ'],
                     ['INT', 'INT', 'INT', 'INT', 'INT', 'INT', 'INT', 'INT', 'INT', 'INT'],
                     ['INT', 'INT', 'INT', 'INT', 'INT', 'INT', 'INT', 'INT', 'INT', 'INT'],
                     [list([]), list([]), list([]), list([]), list([]), list([]), list([]), list([]), list([]), list([])],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N'],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     ['', '', '', '', '', '', '', '', '', '']]

    ALLOWED_MOD_DTYPES = [*msod.mod_text_dtypes().values(), *msod.mod_num_dtypes().values()]
    print(f"ALLOWED_MOD_DTYPES = {ALLOWED_MOD_DTYPES}")
    MODIFIED_DATATYPES = np.random.choice(ALLOWED_MOD_DTYPES, _cols, replace=True)

    _Test = BuildFullSupportObject(OBJECT=OBJECT,
                                    object_given_orientation=_orientation,
                                    OBJECT_HEADER=HEADER,
                                    SUPPORT_OBJECT=None, #SUPPORT_OBJECT,
                                    columns=_cols,
                                    quick_vdtypes=False,
                                    MODIFIED_DATATYPES=None, #MODIFIED_DATATYPES,
                                    print_notes=False,
                                    prompt_to_override=False,
                                    bypass_validation=False,
                                    calling_module=gmn.get_module_name(str(sys.modules[__name__])),
                                    calling_fxn='guard_test')





    print(f'FINAL self.SUPPORT_OBJECT[_row][:15] = ')
    [print(_[:15]) for _ in _Test.SUPPORT_OBJECT]




