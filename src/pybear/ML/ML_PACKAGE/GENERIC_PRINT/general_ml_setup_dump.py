from ML_PACKAGE.openpyxl_shorthand import openpyxl_write as ow
from debug import IdentifyObjectAndPrint as ioap
from MLObjects.SupportObjects import master_support_object_dict as msod


def general_ml_setup_dump(wb, standard_config, WORKING_DATA_SUPOBJS, WORKING_TARGET_SUPOBJS,
        WORKING_REFVECS_SUPOBJS, WORKING_CONTEXT, WORKING_KEEP, split_method, LABEL_RULES, number_of_labels,
        event_value, negative_value):


    def custom_write(sheet, row, column, value, horiz, vert, bold):
        ow.openpyxl_write(wb, sheet, row, column, value, horiz=horiz, vert=vert, bold=bold)

    wb.active.title = 'ML SETUP'

    custom_write('ML SETUP', 1, 1, 'DATA SETUP PARAMETERS', 'left', 'center', True)

    custom_write('ML SETUP', 3, 1, f'STANDARD CONFIG = {standard_config}', 'left', 'center', True)

    row_counter = 3
    row_counter += 2

    custom_write('ML SETUP', row_counter, 1, 'TARGET CONFIG', 'left', 'center', True)
    row_counter += 1
    ARGS = ('left', 'center', False)
    custom_write('ML SETUP', row_counter, 1, 'LABEL RULES:', *ARGS)
    row_counter += 1
    for LABEL in LABEL_RULES:
        custom_write('ML SETUP', row_counter, 2, ", ".join(LABEL), *ARGS)
        row_counter += 1
    custom_write('ML SETUP', row_counter, 1, f'EVENT VALUE:', *ARGS)
    # row_counter += 1
    custom_write('ML SETUP', row_counter, 2, f'{event_value}', *ARGS)
    row_counter += 1
    custom_write('ML SETUP', row_counter, 1, f'NEGATIVE VALUE:', *ARGS)
    # row_counter += 1
    custom_write('ML SETUP', row_counter, 2, f'{negative_value}', *ARGS)
    del ARGS

    row_counter += 2

    ML_SETUP_HEADER = ['OBJECT', 'COLUMN', 'TYPE', 'USER TYPE', 'MIN CUTOFF', 'USE OTHER', 'START_LAG', 'END_LAG',
                       'SCALING', 'FILTERING']
    for heading in range(len(ML_SETUP_HEADER)):
        custom_write('ML SETUP', row_counter, heading + 1, ML_SETUP_HEADER[heading], 'center', 'center', True)
    del ML_SETUP_HEADER

    row_counter += 1


    filter_exception = False
    OBJ_NAMES = ('DATA', 'TARGET', 'REFERENCE')
    SUP_OBJS = (WORKING_DATA_SUPOBJS, WORKING_TARGET_SUPOBJS, WORKING_REFVECS_SUPOBJS)
    ARGS = ('center', 'center', False)
    SO_NAMES = ('HEADER', 'VALIDATEDDATATYPES', 'MODIFIEDDATATYPES', 'MINCUTOFFS', 'USEOTHER', 'STARTLAG', 'ENDLAG', 'SCALING')
    for obj_idx in range(3):
        for col_idx in range(len(SUP_OBJS[obj_idx][0])):

            if col_idx == 0: custom_write('ML SETUP', row_counter, 1, OBJ_NAMES[obj_idx], 'center', 'center', True)

            for idx, so_name in enumerate(SO_NAMES, 2):
                custom_write('ML SETUP', row_counter, idx, SUP_OBJS[obj_idx][msod.QUICK_POSN_DICT()[so_name]][col_idx], *ARGS)

            try: custom_write('ML SETUP', row_counter, 10, ', '.join(SUP_OBJS[obj_idx][msod.QUICK_POSN_DICT()['FILTERING']][col_idx]), 'left', 'center', False)
            except: filter_exception = True

            row_counter += 1
        row_counter += 1

    if filter_exception:
        # KEEP THIS OUTSIDE THE for LOOPS, WOULD PRINT EVERY LOOP
        posn = msod.QUICK_POSN_DICT()['FILTERING']
        ARGS = (__name__, 100, 100)
        print(f'\n*** ATTEMPT TO PARSE "WORKING_FILTERING" TO SPREADSHEET CAUSED RUNTIME EXCEPTION.  DISPLAYING FOR DIAGNOSIS ')
        ioap.IdentifyObjectAndPrint(WORKING_DATA_SUPOBJS[posn], 'DATA FILTERING', *ARGS).run()
        ioap.IdentifyObjectAndPrint(WORKING_TARGET_SUPOBJS[posn], 'TARGET_FILTERING', *ARGS).run()
        ioap.IdentifyObjectAndPrint(WORKING_REFVECS_SUPOBJS[posn], 'REFVECS_FILTERING', *ARGS).run()
        del posn, ARGS
        input(f'\n*** PAUSED FOR REVIEW.  HIT ENTER TO CONTINUE. > ')

    del filter_exception, OBJ_NAMES, SUP_OBJS, SO_NAMES

    row_counter += 1

    custom_write('ML SETUP', row_counter, 1, 'CONTEXT', 'center', 'center', True)


    for LINE in WORKING_CONTEXT:
        custom_write('ML SETUP', row_counter, 2, LINE, 'left', 'center', False)
        row_counter += 1

    '''
    # 2-17-22 PLACEHOLDER FOR SOMEDAY WHEN CAN FILTER BY DATE :(
    # row_counter += 1
    # custom_write('NN SETUP', row_counter, 1, f'Filtering on date range: {START_DATE} - {END_DATE}', 'left', 'center', False)

    interaction, int_cutoff, intercept = 'Y', 'Y', 'Y'

    row_counter += 1
    custom_write('NN SETUP', row_counter, 1, f'Use Interactions: {interaction}', 'left', 'center', False)

    row_counter += 1
    custom_write('NN SETUP', row_counter, 1, f'Interaction cutoff: {int_cutoff}', 'left', 'center', False)

    row_counter += 1
    custom_write('NN SETUP', row_counter, 1, f'Intercept: {intercept}', 'left', 'center', False)
    '''

    return wb





if __name__ == '__main__':

    from openpyxl import Workbook

    wb = Workbook()

    print(type(wb))
    quit()

    standard_config = 'None'
    mlr_config = 'None'

    split_method = 'None'
    LABEL_RULES = []
    number_of_labels = 1
    event_value = ''
    negative_value = ''
    mlr_conv_kill = 1
    mlr_pct_change = 0
    mlr_conv_end_method = 'KILL'
    mlr_rglztn_type = 'RIDGE'
    mlr_rglztn_fctr = 10000
    mlr_batch_method = 'M' #'B'
    mlr_batch_size = 1000 #len(SWNL[0][0])
    MLR_OUTPUT_VECTOR = []


    general_ml_setup_dump(wb, standard_config, WORKING_DATA_SUPOBJS, WORKING_TARGET_SUPOBJS,
            WORKING_REFVECS_SUPOBJS, WORKING_CONTEXT, WORKING_KEEP, split_method, LABEL_RULES, number_of_labels,
            event_value, negative_value)



