import sys, inspect
import numpy as n
import sparse_dict as sd
from copy import deepcopy
from debug import get_module_name as gmn
from data_validation import validate_user_input as vui
from ML_PACKAGE._data_validation import ValidateObjectType as vot
from ML_PACKAGE.MLREGRESSION import MLRegression as mlr
from linear_algebra import XTX_determinant as xtxd


# calc_statistics                       (calculate determinant, min of inv(XTX), max of inv(XTX), r, R2, R2_adj, F)
# stats_header_print                    (print standardized header for statistics printout)
# stats_print                           (template for printing stats results)
# whole_data_object_stats               (calculate stats for object being tested)
# column_drop_iterator                  (iterate thru object, drop 1 column at a time, report freq of dropped & stats for all iter)
# row_drop_iterator                     (iterate thru object, drop 1 row at a time, report stats for all iter)


this_module = gmn.get_module_name(str(sys.modules[__name__]))



def calc_statistics(self, DATA_OBJECT, name, DATA_OBJECT_HEADER, TARGET_OBJECT,
                    calc_allow='Y'):  # calculate determinant, min of inv(XTX), max of inv(XTX), r, R**2

    fxn = inspect.stack()[0][3]

    # UPDATE STATISTICS, HELD AS CLASS ATTRIBUTES

    if calc_allow == 'N':
        self.determ, self.min_elem, self.max_elem, self.COEFFICIENTS, self.P_VALUES, self.r, self.R2, \
        self.R2_adj, self.F = tuple(['' for _ in range(9)])
    elif calc_allow == 'Y':
        with n.errstate(all='ignore'):
            xtxd.XTX_determinant(DATA_AS_ARRAY_OR_SPARSEDICT=DATA_OBJECT, name=name, module=this_module, fxn=fxn,
                                 print_to_screen=False, return_on_exception='nan')
            # COULD ALSO GET THIS AS AN ATTRIBUTE OF MLRegression CLASS

            # RETURNED FROM MLRegression
            # XTX_determinant, self.COEFFS, PREDICTED, P_VALUES, r, R2, R2_adj, F
            RESULTS = mlr.MLRegression(DATA=DATA_OBJECT,
                                       DATA_TRANSPOSE=None,
                                       data_given_orientation='COLUMN',  # 4/15/23 BEAR FIX THIS ARGS/KWARGS, MLRegression CHANGED
                                       TARGET=TARGET_OBJECT,
                                       TARGET_TRANSPOSE=None,
                                       TARGET_AS_LIST=None,
                                       target_given_orientation='COLUMN',  # 4/15/23 BEAR FIX THIS ARGS/KWARGS, MLRegression CHANGED
                                       has_intercept='INTERCEPT' in DATA_OBJECT_HEADER[0],
                                       intercept_math=True,
                                       safe_matmul=True,
                                       transpose_check=False
                                       ).run()

        self.COEFFICIENTS, self.P_VALUES, self.r, self.R2, self.R2_adj, self.F = RESULTS[1], *RESULTS[3:]


def stats_header_print(self, top_line_description, first_column_header,
                       max_len):  # print standardized header for statistics printout
    print(top_line_description)
    print(
        f'{" " * min(max(self.first_col_width, max_len + 2), self.header_width)}{" " * self.type_width}{" " * self.freq_width}{"XTX".ljust(self.stat_width)}' + \
        f'{"minv(XTX)".ljust(self.stat_width)}{"minv(XTX)".ljust(self.stat_width)}')
    print(f'{first_column_header.ljust(min(max(self.first_col_width, max_len + 2), self.header_width))}' + \
          f'{"TYPE".ljust(self.type_width)}' +
          f'{"FREQ".ljust(self.freq_width)}' +
          f'{"DETERM".ljust(self.stat_width)}' +
          f'{"MIN ELEM".ljust(self.stat_width)}' +
          f'{"MAX ELEM".ljust(self.stat_width)}' +
          f'{"r".ljust(self.stat_width)}' +
          f'{"R2".ljust(self.stat_width)}' +
          f'{"R2_adj".ljust(self.stat_width)}' +
          f'{"F".ljust(self.stat_width)}'
          )


def stats_print(self, first_column_text, datatype, freq_text, max_len):  # template for printing stats results
    print(f'{first_column_text.ljust(min(max(self.first_col_width, max_len + 2), self.header_width))}' +
          f'{datatype.ljust(self.type_width)}' +
          f'{freq_text.ljust(self.freq_width)}' +
          [f'{self.determ:.5g}' if isinstance(self.determ, float) else f'{self.determ}'][0].ljust(self.stat_width) +
          [f'{self.min_elem:.5g}' if isinstance(self.min_elem, float) else f'{self.min_elem}'][0].ljust(
              self.stat_width) +
          [f'{self.max_elem:.5g}' if isinstance(self.max_elem, float) else f'{self.max_elem}'][0].ljust(
              self.stat_width) +
          [f'{self.r:.6f}' if isinstance(self.r, float) else f'{self.r}'][0].ljust(self.stat_width) +
          [f'{self.R2:.6f}' if isinstance(self.R2, float) else f'{self.R2}'][0].ljust(self.stat_width) +
          [f'{self.R2_adj:.6f}' if isinstance(self.R2_adj, float) else f'{self.R2_adj}'][0].ljust(self.stat_width) +
          [f'{self.F:.6f}' if isinstance(self.F, float) else f'{self.F}'][0].ljust(self.stat_width)
          )


def whole_data_object_stats(self, OBJECT, name, HEADER, append_ones='N'):  # calculate stats for object being tested

    WIP_OBJECT = n.array([_.copy() if isinstance(_, n.ndarray) else deepcopy(_) for _ in OBJECT], dtype=object)
    NEW_HEADER = deepcopy(HEADER)

    if append_ones == 'Y':
        if self.is_list():
            WIP_OBJECT = n.vstack((WIP_OBJECT, n.ones((1, len(WIP_OBJECT[0])), dtype=int)))
        elif self.is_dict():
            WIP_OBJECT = sd.append_outer(WIP_OBJECT, {_: 1 for _ in range(sd.inner_len(WIP_OBJECT))})[0]

        NEW_HEADER = n.array([*NEW_HEADER, 'INTERCEPT'], dtype=str)

    # PRINT HEADER FOR RESULTS OF ENTIRE OBJECT BEING TESTED (WHETHER ALL DATA OR LEVEL EXPANSION OF ONE COLUMN):
    max_len = n.max([len(_) for _ in [name, *NEW_HEADER, f'ORIGINAL COLUMN', f'CATEGORY']])
    self.stats_header_print(f'\nRESULTS FOR ENTIRE {name} OBJECT:', f'ORIGINAL COLUMN', max_len)

    # UPDATE RESULTS FOR ENTIRE OBJECT:
    self.calc_statistics(WIP_OBJECT, name, NEW_HEADER, self.TARGET_OBJECT, calc_allow=self.calc_allow)

    if len(WIP_OBJECT) == 1:
        datatype = vot.ValidateObjectType(WIP_OBJECT).ml_package_object_type()
    else:
        datatype = 'N/A'

    # PRINT STATISTICS
    if self.is_dict():
        _len = sd.inner_len(WIP_OBJECT)
    elif self.is_list():
        _len = len(WIP_OBJECT[0])

    self.stats_print(str(name), datatype, str(_len), max_len)

    del WIP_OBJECT, NEW_HEADER



# 11/27/22 BEAR CONSIDER MAKING A MLPackage AND A CoreRunCode FOR column_drop_iterator
def column_drop_iterator(self, OBJECT, name, HEADER, append_ones='N'):     # FROM PreRunExpandCategories

    # PRINT HEADER FOR COLUMN DROP CYCLES:
    max_len = n.max([len(_) for _ in [name, *HEADER, f'ORIGINAL COLUMN', f'CATEGORY']])
    self.stats_header_print(f'\nEFFECTS OF REMOVING A COLUMN:', 'CATEGORY', max_len)
    # GET RESULTS FOR OBJECT W CYCLED COLUMN DROPS:
    CYCLE_OBJECT = n.array([_.copy() if isinstance(_, n.ndarray) else deepcopy(_) for _ in OBJECT], dtype=object)
    NEW_HEADER = deepcopy(HEADER)

    if append_ones == 'Y':
        if self.is_list():
            CYCLE_OBJECT = n.vstack((CYCLE_OBJECT, n.ones((1, len(CYCLE_OBJECT[0])), dtype=int)))
        elif self.is_dict():
            CYCLE_OBJECT = sd.append_outer(CYCLE_OBJECT, {_:1 for _ in range(sd.inner_len(CYCLE_OBJECT))})[0]

        NEW_HEADER[0] = n.insert(NEW_HEADER[0], len(NEW_HEADER[0]), 'INTERCEPT')

    for col_idx in range(len(CYCLE_OBJECT)):

        if self.is_list():
            POPPED_COL = CYCLE_OBJECT[col_idx].copy()
            CYCLE_OBJECT_WIP = n.delete(CYCLE_OBJECT.copy(), col_idx, axis=0)
        elif self.is_dict():
            POPPED_COL = sd.unzip_to_ndarray({0:CYCLE_OBJECT[col_idx]})[0]
            CYCLE_OBJECT_WIP = sd.delete_outer_key(deepcopy(CYCLE_OBJECT), [col_idx])[0]

        # UPDATE STATISTICS HELD AS CLASS ATTRIBUTES
        self.calc_statistics(CYCLE_OBJECT_WIP, name, NEW_HEADER, self.TARGET_OBJECT, calc_allow=self.calc_allow)

        _ = vot.ValidateObjectType(POPPED_COL).ml_package_object_type()  # GET TYPE FOR COLUMN REMOVED

        if _ == ['FLOAT', 'INT']: freq_text = f'{str(len(POPPED_COL))}'  # FOR FLOAT OR INT RETURN TOTAL CT
        elif _ in ['BIN']: freq_text = f'{str(n.sum(POPPED_COL))}'   # FOR BIN RETURN CT OF "1"s
        else: ValueError(f'INVALID DATATYPE {_} IN PrerunExpandCategories.column_drop_iterator().')

        # PRINT STATISTICS
        self.stats_print(str(NEW_HEADER[col_idx]), _, freq_text, max_len)

    del CYCLE_OBJECT, NEW_HEADER, POPPED_COL, CYCLE_OBJECT_WIP



# 11/27/22 BEAR CONSIDER MAKING A MLPackage AND A CoreRunCode FOR row_drop_iterator
def row_drop_iterator(self, OBJECT, name, HEADER, TARGET_OBJECT):   # FROM PreRunExpandCategories
    while True:
        CYCLE_OBJECT = OBJECT.copy()
        CYCLE_TARGET = TARGET_OBJECT.copy()

        if len(CYCLE_TARGET[0]) >= 500:
            if vui.validate_user_str(f'\nTHERE ARE {len(CYCLE_TARGET)} ROWS.  PROCEED? (y/n) > ', 'YN') == 'N':
                del CYCLE_OBJECT, CYCLE_TARGET
                break

        # PRINT HEADER FOR ROW DROP CYCLES:
        max_len = n.max([len(_) for _ in [*[str(__) for __ in range(len(CYCLE_OBJECT[0]))], f'ROW INDEX']])
        self.stats_header_print(f'\nEFFECTS OF REMOVING A ROW:', 'ROW INDEX', max_len)

        # GET RESULTS FOR OBJECT W CYCLED ROW DROPS:
        for row_idx in range(len(CYCLE_OBJECT[0])):
            CYCLE_TARGET_WIP = n.delete(CYCLE_TARGET.copy(), row_idx, axis=1)
            if self.is_list():
                CYCLE_OBJECT_WIP = n.delete(CYCLE_OBJECT.copy(), row_idx, axis=1)
            elif self.is_dict():
                CYCLE_OBJECT_WIP = sd.delete_inner_key(deepcopy(CYCLE_OBJECT), [row_idx])[0]

            # UPDATE STATISTICS HELD AS CLASS ATTRIBUTES
            self.calc_statistics(CYCLE_OBJECT_WIP, name, HEADER, CYCLE_TARGET_WIP, calc_allow='Y')

            # PRINT STATISTICS
            self.stats_print(str(row_idx), 'n/a', ' ', max_len, )

            del CYCLE_OBJECT, CYCLE_TARGET, CYCLE_OBJECT_WIP, CYCLE_TARGET_WIP

        break















class ExpandCategories:
    def __init__(self, ):
        pass



    def return_fxn(self):
        return 'yada yada yada'










