import sys
import numpy as np
import sparse_dict as sd
from debug import get_module_name as gmn
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from general_data_ops import get_shape as gs
from general_list_ops import list_select as ls
from ML_PACKAGE.GENERIC_PRINT import print_post_run_options as ppro
from MLObjects import MLObject as mlo
from MLObjects.SupportObjects import master_support_object_dict as msod
import plotly.graph_objects as go
from plotly.subplots import make_subplots







class StandardizeNormalize:

    # ACCESS THE STANDARDIZED / NORMALIZED DATA OBJECT VIA ATTRIBUTE OF AN INSTANCE

    def __init__(self, DATA_OBJECT, DATA_SUPOBJ, data_given_orientation, bypass_validation=None):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True,False, None],
                                                         self.this_module, fxn, return_if_none=False)

        self.DATA_OBJECT = ldv.list_dict_validater(DATA_OBJECT, 'DATA')[1]

        rows, cols = gs.get_shape("DATA", self.DATA_OBJECT, data_given_orientation)

        if not bypass_validation:
            DATA_SUPOBJ = ldv.list_dict_validater(DATA_SUPOBJ, 'DATA SUPOBJ')[1]

            data_given_orientation = akv.arg_kwarg_validater(data_given_orientation, 'data_given_orientation', ['ROW', 'COLUMN'],
                                                             self.this_module, fxn)

            _ = len(msod.master_support_object_dict())
            if not DATA_SUPOBJ.shape == (_, cols):
                self._exception(f'INVALID SHAPE {DATA_SUPOBJ.shape} FOR DATA SUPOBJ, MUST BE {_}, {cols}', fxn)
            del _

        msod_hdr_idx = msod.QUICK_POSN_DICT()["HEADER"]
        msod_mdtype_idx = msod.QUICK_POSN_DICT()["MODIFIEDDATATYPES"]
        msod_scaling_idx = msod.QUICK_POSN_DICT()["SCALING"]

        FLOAT_COLUMN_IDXS = np.arange(0, cols)[list(map(lambda x: x in ['FLOAT', 'INT'], DATA_SUPOBJ[msod_mdtype_idx]))]

        x = DATA_SUPOBJ[msod_hdr_idx]

        while True:  # JUST FOR ESCAPE HATCH]

            if len(FLOAT_COLUMN_IDXS) == 0:  # IF NO FLOAT/INT COLUMNS, DONT NEED TO RUN STANDARDIZE/NORMALIZE
                print(f'\n*** THERE ARE NO FLOAT/INT COLUMNS IN DATA, DO NOT NEED TO RUN STANDARDIZE/NORMALIZE ***')
                break

            if vui.validate_user_str(f'\nReally run standardize / normalize? (y/n)  This may take a long time for '
                                     f'large data sets. > ', 'YN') == 'N':
                break

            DataClass = mlo.MLObject(
                                        self.DATA_OBJECT,
                                        data_given_orientation,
                                        name="DATA",
                                        return_orientation='AS_GIVEN',
                                        return_format='AS_GIVEN',
                                        bypass_validation=bypass_validation,
                                        calling_module=self.this_module,
                                        calling_fxn=fxn
            )

            if vui.validate_user_str(f'Generate histograms? (y/n) > ', 'YN') == 'Y':
                # MAKE CHARTS ################################################################################################

                print(f'\nGenerating histograms...\n')

                CHART_NAMES = x[FLOAT_COLUMN_IDXS]
                # col_ct & row_ct ARE THE LAYOUT FOR A GRID OF CHARTS
                col_ct = 4
                row_ct = int(np.ceil(len(CHART_NAMES) / col_ct))

                fig = make_subplots(rows=row_ct, cols=col_ct, subplot_titles=CHART_NAMES)

                counter = 0
                for col_idx in FLOAT_COLUMN_IDXS:

                    __ = DataClass.return_columns([col_idx], return_format='AS_GIVEN', return_orientation='COLUMN')

                    if self.is_list(): _min, _max = np.min(__), np.max(__)
                    elif self.is_dict(): _min, _max = sd.min_(__), sd.max_(__)

                    with np.errstate(all='ignore'):
                        if DATA_SUPOBJ[msod_mdtype_idx][col_idx] == 'FLOAT': bin_size = np.floor(_max - _min) / 20
                        elif DATA_SUPOBJ[msod_mdtype_idx][col_idx] in ['INT', 'BIN']: bin_size = np.ceil((_max - _min) / 20)

                    fig.add_trace(
                                    go.Histogram(
                                                    x=__,
                                                    # histnorm='percent',
                                                    name=str(x[col_idx]),
                                                    xbins=dict(start=_min - 0.05 * (_max - _min),
                                                               end=_max + 0.05 * (_max - _min),
                                                               size=bin_size),
                                                    marker_color='#EB89B5',
                                                    opacity=1
                                    ),
                                    row=counter // col_ct + 1, col=counter % col_ct + 1
                    )

                    counter += 1

                del __, _min, _max

                fig.update_layout(height=300 * row_ct, width=1200, title_text="Histograms of Data in Float/Int Columns")

                fig.write_html("Histograms.html", auto_open=True)
                print(f'Histograms complete.\n')

                del fig
                # END MAKE CHARTS ################################################################################################

            # SELECT COLUMNS TO NORMALIZE / STANDARDIZE ##################################################################


            print(f'\nALL FLOAT AND INT COLUMNS ARE GOING TO NORMALIZED OR STANDARDIZED. COLUMNS THAT ARE NOT STANDARDIZED ' + \
                  f'WILL BE NORMALIZED, AND VICE VERSA. USER GETS TO PICK WHETHER TO MANUALLY SELECT THE COLUMNS TO NORMALIZE ' + \
                  f'OR MANUALLY SELECT THE COLUMNS TO STANDARDIZE, THEN THE COLUMNS NOT CHOSEN WILL GET THE ALTERIOR TREATMENT. ' + \
                  f'I.E., IF USER CHOOSES "NORMALIZE", THEN THE USER MUST MANUALLY INDICATE THE COLUMNS THAT GET NORMALIZED, AND ' + \
                  f'THE COLUMNS NOT SELECTED WILL AUTOMATICALLY BE STANDARDIZED.\n')
            select_type = vui.validate_user_str(f'Choose indicate normalized(n) or standardized(s) columns, or abort(x) > ', 'NSX')

            if select_type == 'N':
                NORMALIZE_COLS = ls.list_custom_select(x[FLOAT_COLUMN_IDXS], 'idx')
                STANDARDIZE_COLS = [_ for _ in FLOAT_COLUMN_IDXS if _ not in NORMALIZE_COLS]
            elif select_type == 'S':
                STANDARDIZE_COLS = ls.list_custom_select(x[FLOAT_COLUMN_IDXS], 'idx')
                NORMALIZE_COLS = [_ for _ in FLOAT_COLUMN_IDXS if _ not in STANDARDIZE_COLS]
            elif select_type == 'X':
                break
            # END SELECT COLUMNS TO NORMALIZE / STANDARDIZE ##################################################################

            print(f'\nUSER HAS SELECTED TO STANDARDIZE:')
            ppro.SelectionsPrintOnly([x[_] for _ in STANDARDIZE_COLS], append_ct_limit=3, max_len=np.max(map(len, x)))
            ppro.SelectionsPrintOnly([x[_] for _ in NORMALIZE_COLS], append_ct_limit=3, max_len=np.max(map(len, x)))
            print(f'\nUSER HAS SELECTED TO NORMALIZE:')

            # CAN SAFELY RESTART BECAUSE DATA & SUPOBJ HAVE NOT CHANGED, ONLY LISTS & CHARTS HAVE BEEN GENERATED
            _ = vui.validate_user_str(f'\nAccept(a), restart(r), abandon(x) > ', 'ARX')
            if _ == 'A': pass
            elif _ == 'R': continue
            elif _ == 'X': break


            for col_idx in range(cols):

                # EXTRACT PERTINENT COLUMN
                ACTV_COLUMN = DataClass.return_columns([col_idx], return_format='AS_GIVEN', return_orientation='COLUMN')
                # DELETE PERTINENT COLUMN (TO BE REINSERTED IN NORMALIZED / STANDARDIZED STATE LATER)
                DataClass.delete_columns([col_idx])

                if col_idx in STANDARDIZE_COLS:
                    if self.is_list():
                        mean, var = np.average(ACTV_COLUMN), np.var(ACTV_COLUMN)
                        ACTV_COLUMN = (ACTV_COLUMN - mean) / var
                    elif self.is_dict():
                        mean, var = sd.average_(ACTV_COLUMN), sd.variance_(ACTV_COLUMN)
                        ACTV_COLUMN = sd.scalar_divide(sd.scalar_subtract(ACTV_COLUMN, mean), var)

                    DATA_SUPOBJ[msod_scaling_idx][col_idx] = f'standardized by mean={mean:.6g}, var={var:.6g}'
                    del mean, var

                elif col_idx in NORMALIZE_COLS:
                    max_abs = ACTV_COLUMN / np.max(abs(ACTV_COLUMN))
                    if self.is_list(): ACTV_COLUMN = max_abs
                    elif self.is_dict(): ACTV_COLUMN = sd.scalar_divide(ACTV_COLUMN, sd.max_(sd.abs_(ACTV_COLUMN)))

                    DATA_SUPOBJ[msod_scaling_idx][col_idx] = f'scaled by max(abs())={max_abs: .6g}'
                    del max_abs

                DataClass.insert_column(col_idx, ACTV_COLUMN, 'COLUMN')

            print(f'\nSTANDARDIZE / NORMALIZE COMPLETE.\n')

            self.DATA_OBJECT = DataClass.OBJECT

            del DataClass

            break

        del rows, cols, msod_hdr_idx, msod_mdtype_idx, msod_scaling_idx

    # END init ############################################################################################################
    #######################################################################################################################
    #######################################################################################################################

    def _exception(self, words, fxn): raise Exception(f'{self.this_module}.{fxn}() >>> {words}')

    def is_list(self): return isinstance(self.DATA_OBJECT, (list, tuple, np.ndarray))

    def is_dict(self): return isinstance(self.DATA_OBJECT, dict)












if __name__ == '__main__':

    from MLObjects.TestObjectCreators import test_header as th
    from MLObjects.SupportObjects import BuildFullSupportObject as bfso

    _rows = 100
    _cols = 50
    _orient = 'COLUMN'

    DATA1 = np.random.normal(np.random.randint(-9,10), np.random.uniform(.1, 2),
                             (_rows if _orient=='ROW' else _cols, _cols if _orient=='ROW' else _rows))
    DATA2 = np.random.randint(-9, 10, (_rows if _orient=='ROW' else _cols, _cols if _orient=='ROW' else _rows))
    DATA = np.hstack((DATA1, DATA2))

    HEADER = th.test_header(_cols)

    SupObjClass = bfso.BuildFullSupportObject(
                                             OBJECT=DATA,
                                             object_given_orientation=_orient,
                                             OBJECT_HEADER=HEADER,
                                             SUPPORT_OBJECT=None,
                                             columns=_cols,
                                             quick_vdtypes=False,
                                             MODIFIED_DATATYPES=None,
                                             print_notes=False,
                                             prompt_to_override=False,
                                             bypass_validation=True,
                                             calling_module=gmn.get_module_name(str(sys.modules[__name__])),
                                             calling_fxn='tests'
    )

    SUPOBJ = SupObjClass.SUPPORT_OBJECT

    del SupObjClass


    StandardizeNormalize(DATA, SUPOBJ, _orient, bypass_validation=None)










