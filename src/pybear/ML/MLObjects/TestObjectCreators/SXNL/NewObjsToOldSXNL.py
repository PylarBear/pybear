import sys
from MLObjects.SupportObjects import NEWSupObjToOLD as nsoto
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from debug import get_module_name as gmn
from general_data_ops import get_shape as gs


class NewObjsToOldSXNL:

    def __init__(self, DATA, TARGET, REFVECS,   # TO ALLOW *CreateSXNL.SXNL
                        DATA_FULL_SUPOBJS, TARGET_FULL_SUPOBJS, REFVECS_FULL_SUPBOJ,   # TO ALLOW *CreateSXNL.SXNL_SUPPORT_OBJECTS
                        data_orientation=None,
                        target_orientation=None,
                        refvecs_orientation=None,
                        bypass_validation=None):

        this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                    this_module, fxn, return_if_none=False)

        if not bypass_validation:

            def _exception(words):
                raise Exception(f'{this_module}.{fxn}() >>> *** {words} ***')

            data_orientation = akv.arg_kwarg_validater(data_orientation, 'data_orientation', ['ROW', 'COLUMN'], this_module, fxn)
            target_orientation = akv.arg_kwarg_validater(target_orientation, 'target_orientation', ['ROW', 'COLUMN'], this_module, fxn)
            refvecs_orientation = akv.arg_kwarg_validater(refvecs_orientation, 'refvecs_orientation', ['ROW', 'COLUMN'], this_module, fxn)

            DATA = ldv.list_dict_validater(DATA, 'DATA')[1]
            TARGET = ldv.list_dict_validater(TARGET, 'TARGET')[1]
            REFVECS = ldv.list_dict_validater(REFVECS, 'REFVECS')[1]

            data_rows, data_cols = gs.get_shape('DATA', DATA, data_orientation)
            target_rows, target_cols = gs.get_shape('TARGET', TARGET, target_orientation)
            refvecs_rows, refvecs_cols = gs.get_shape('REFVECS', REFVECS, refvecs_orientation)

            # VALIDATE DATA, TARGET, & REFVECS HAVE EQUAL ROWS
            _ = (data_rows, target_rows, refvecs_rows)
            if not min(_) == max(_): _exception(f'OBJECTS DO NOT HAVE EQUAL ROWS ({", ".join(map(str, _))})')

            # VALIDATE SUPOBJ COLUMNS MATCH OBJ COLUMNS
            NAMES = ('DATA', 'TARGET', 'REFVECS')
            _ = (data_cols, target_cols, refvecs_cols)
            __ = (len(DATA_FULL_SUPOBJS[0]), len(TARGET_FULL_SUPOBJS[0]), len(REFVECS_FULL_SUPBOJ[0]))
            for _name, o_cols, so_cols in zip(NAMES, _,__):
                if not _ == __: _exception(f'{_name} OBJECTS DO NOT HAVE EQUAL COLUMNS (OBJECT={o_cols}, SUPOBJ={so_cols})')

            del _, __, data_rows, data_cols , target_rows, target_cols , refvecs_rows, refvecs_cols, NAMES, _exception

        OldSupObjClass = nsoto.NEWSupObjToOLD(DATA_FULL_SUPOBJS, TARGET_FULL_SUPOBJS, REFVECS_FULL_SUPBOJ)

        self.SXNL = [DATA,
                     OldSupObjClass.DATA_HEADER,
                     TARGET,
                     OldSupObjClass.TARGET_HEADER,
                     REFVECS,
                     OldSupObjClass.REFVECS_HEADER
                     ]


        self.VALIDATED_DATATYPES = OldSupObjClass.VALIDATED_DATATYPES
        self.MODIFIED_DATATYPES = OldSupObjClass.MODIFIED_DATATYPES
        self.FILTERING = OldSupObjClass.FILTERING
        self.MIN_CUTOFFS = OldSupObjClass.MIN_CUTOFFS
        self.USE_OTHER = OldSupObjClass.USE_OTHER
        self.START_LAG = OldSupObjClass.START_LAG
        self.END_LAG = OldSupObjClass.END_LAG
        self.SCALING = OldSupObjClass.SCALING

        del OldSupObjClass

























