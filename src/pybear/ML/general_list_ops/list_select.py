import numpy as n
from copy import deepcopy
import data_validation.validate_user_input as vui



def list_select_print_fxn(WIP_LIST):   # STANDARDIZES LIST PRINT FOR single, multi, custom
    max_chars = 100  # MAX CHAR PER PRINT LINE
    buffer = 3
    idx_chars = 5
    max_len = max([len(str(_)[:max_chars - idx_chars - buffer]) for _ in WIP_LIST])  # LONGEST STR IN "LIST", AFTER TRUNCATION
    num_items = min(max(1, max_chars // (idx_chars + max_len + buffer)), 7)  # NUMBER OF LIST ITEMS PER LINE, max 7 THINGS PER LINE

    print()
    print_str = ''
    for idx, item_str in enumerate(WIP_LIST, start=1):
        item_str = str(item_str)
        print_str += f'{idx})'.ljust(5) + item_str[:max_chars // num_items - idx_chars - buffer].ljust(idx_chars + max_len + buffer) + '' * buffer

        if (idx) % num_items == 0: # IF COUNTER HITS num_items, PRINT IT, RESET IT, & START OVER
            print(print_str)
            print_str = ''
    else: print(print_str)


# THIS ALLOWS USER TO SELECT A SINGLE ITEM FROM A LIST & RETURN EITHER THE INDEX OF THE ITEM OR THE ITEM ITSELF
def list_single_select(LIST, user_prompt, idx_or_value):

    idx_or_value = idx_or_value.lower()

    while True:
        list_select_print_fxn(LIST)

        selection = vui.validate_user_int(user_prompt + ' (select index) > ', min=1, max=len(LIST)) - 1

        if vui.validate_user_str(f'Selected {selection+1}: {LIST[selection]} ... accept? (y/n) > ', 'YN') == 'Y':
            if idx_or_value == 'idx': SELECTION = [selection]
            elif idx_or_value == 'value': SELECTION = [LIST[selection]]
            else: raise ValueError(f'INVALID PARAMETER FOR list_select.list_multi_select(), idx or value.')
            break

    return SELECTION


# THIS ALLOWS USER TO SELECT MULTIPLE ITEMS FROM A LIST & RETURN EITHER A LIST OF THE INDICES OF THE ITEMS
# OR A LIST OF THE ITEMS.  USER CAN SELECT SAME ITEM MORE THAN ONCE
# FOR number_to_choose, USER CAN ENTER AN INTEGER, OR '' FOR INDEFINITE SELECTIONS
def list_multi_select(LIST, user_prompt, idx_or_value, number_to_choose=float('inf')):

    idx_or_value = idx_or_value.lower()

    # DETERMINE IF number_to_choose IS VALID
    if not isinstance(number_to_choose, int) and not number_to_choose == float('inf'):
        raise ValueError(f'INVALID number_to_choose PARAMETER IN list_select.list_multi_select')

    while True:
        number = 0
        SELECTION = []
        while number < number_to_choose:
            number += 1

            # IF USER SPECIFIED A FINITE NUMBER TO CHOOSE
            if isinstance(number_to_choose, int):
                list_select_print_fxn(LIST)
                print(f'\nUSER SELECTIONS: {SELECTION}\n')

                selection = vui.validate_user_int(user_prompt+f' (select index) ({number} of {number_to_choose}) > ',
                                    min=1, max=len(LIST)) - 1

            # IF USER SPECIFIED "''", UNLIMITED NUMBER TO CHOOSE
            elif number_to_choose == float('inf'):
                list_select_print_fxn([*LIST, "DONE"])
                print(f'\nUSER SELECTIONS: {SELECTION}\n')

                selection = vui.validate_user_int(user_prompt+f' (select index) ({number} of UNLIMITED) > ',
                                    min=1, max=len(LIST)+1) - 1

            # THE ONLY WAY USER COULD SELECT idx == len(LIST) IS SELECTED "DONE" IN UNLIMITED MODE
            if selection == len(LIST): break

            # IF DIDN'T SELECT "DONE" THEN UPDATE "SELECTIONS" LIST
            if idx_or_value == 'idx': SELECTION.append(selection)
            elif idx_or_value == 'value': SELECTION.append(LIST[selection])
            else: ValueError(f'INVALID PARAMETER FOR list_select.list_multi_select(), idx or value.')

        print('\nUSER SELECTED:')
        print(SELECTION)

        if vui.validate_user_str('Accept selections? (y/n) > ', 'YN') == 'Y':
            break

    return SELECTION

# THIS ALLOWS USER TO SELECT USING CUSTOM OPTIONS FROM A LIST & RETURN EITHER A LIST OF THE INDICES OF THE ITEMS
# OR A LIST OF THE ITEMS.  USER CANNOT SELECT SAME ITEM MORE THAN ONCE.
def list_custom_select(LIST, idx_or_value):

    idx_or_value = idx_or_value.lower()
    if idx_or_value not in ['idx', 'value']:
        raise ValueError(f'\nINVALID PARAMETER ENTERED FOR idx_or_value IN list_select.list_custom_select().')

    CMDS = ['SELECT ALL', 'SELECT ALL EXCEPT', 'MULTI-SELECT', 'SINGLE-SELECT']


    def selections_print(SELECTION_IDX, SELECTION_VALUE, idx_or_value):
        if idx_or_value == 'idx': print(f'\nCurrent index selections: \n{[_+1 for _ in SELECTION_IDX]}\n')
        elif idx_or_value == 'value': print(f'\nCurrent value selections: \n{SELECTION_VALUE}\n')


    LIST = [*LIST]  # ACCOMMODATES IF LIST IS A LIST-TYPE OR A DF

    while True:
        SELECTION_IDX = []
        SELECTION_VALUE = []
        WIP_LIST = deepcopy(LIST)

        print(f'\nSELECT ITEMS FROM THE FOLLOWING LIST:')
        list_select_print_fxn(WIP_LIST)

        cmd = list_single_select(CMDS, f'Choose custom selection option', 'idx')[0]

        if cmd == 0:    # SELECT ALL
            SELECTION_IDX = [*range(len(LIST))]
            SELECTION_VALUE = [*LIST]     # * ACCOMMODATES IF LIST IS A LIST-TYPE OR A DF

        elif cmd == 1:     # SELECT ALL EXCEPT
            print('')
            while True:
                # CREATE HOLDER FOR USER-SELECTED ITEMS TO EXCLUDE
                EXCLUDED_IDX = []
                EXCLUDED_VALUE = []
                WIP_LIST = deepcopy(LIST)
                while True:
                    exclude_idx = list_single_select(WIP_LIST+['DONE'], f'\nSelect index of item to EXCLUDE', 'idx')[0]

                    if exclude_idx < len(WIP_LIST):  # IF USER CHOSE ANYTHING OTHER THAN "DONE"
                        if exclude_idx in EXCLUDED_IDX:  # IF USER CHOSE AN ITEM ALREADY SELECTED, DISALLOW
                            print(f'{LIST[exclude_idx]} IS ALREADY SELECTED FOR EXCLUSION, CHOOSE ANOTHER OR CHOOSE "DONE"')
                            continue
                        else:   # IF NOT "DONE" AND NOT ALREADY IN SELECTION_IDX
                            EXCLUDED_IDX.append(exclude_idx)
                            EXCLUDED_VALUE.append(LIST[exclude_idx])
                            WIP_LIST[exclude_idx] = ''
                    elif exclude_idx == len(WIP_LIST):  # IF USER CHOSE "DONE"
                        break

                print(f'\nUSER HAS CHOSEN ALL ITEMS EXCEPT:')
                selections_print(EXCLUDED_IDX, EXCLUDED_VALUE, 'value')
                print('')

                if vui.validate_user_str(f'Accept exclusions? (y/n) > ', 'YN') == 'Y':
                    break

            SELECTION_IDX = [_ for _ in range(len(LIST)) if _ not in EXCLUDED_IDX]
            SELECTION_VALUE = [_ for _ in LIST if _ not in EXCLUDED_VALUE]


        elif cmd == 2:    # MULTI-SELECT
            print(f'SELECTED THUS FAR: {SELECTION_VALUE}')
            while True:
                selection_idx = list_single_select(WIP_LIST+['DONE'], f'\nSelect index of item', 'idx')[0]
                if selection_idx < len(WIP_LIST):  # IF USER CHOSE ANYTHING OTHER THAN "DONE"
                    if selection_idx in SELECTION_IDX:  # IF USER CHOSE AN ITEM ALREADY SELECTED, DISALLOW
                        print(f'{LIST[selection_idx]} IS ALREADY SELECTED, CHOOSE ANOTHER OR CHOOSE "DONE"')
                        continue
                    else:   # IF NOT "DONE" AND NOT ALREADY IN SELECTION_IDX
                        SELECTION_IDX.append(selection_idx)
                        SELECTION_VALUE.append(LIST[selection_idx])
                        WIP_LIST[selection_idx] = ''

                elif selection_idx == len(WIP_LIST):  #IF USER CHOSE "DONE"
                    break

        elif cmd == 3:    # SINGLE-SELECT
            SELECTION_IDX = list_single_select(LIST, f'Select index of single item', 'idx')
            SELECTION_VALUE = [LIST[_] for _ in SELECTION_IDX]

        selections_print(SELECTION_IDX, SELECTION_VALUE, idx_or_value)

        if vui.validate_user_str(f'Accept selections? (y/n) > ', 'YN') == 'Y': break

    if idx_or_value == 'idx':
        return SELECTION_IDX
    elif idx_or_value == 'value':
        return SELECTION_VALUE






# USER SELECTS THE NAME OF A COLUMN IN THE SPECIFIED DF, AND RETURNS THE COLUMN NAME
def DF_column_single_select(DF, user_prompt):    #DF must be a pandas dataframe
    print('')
    COLUMN_LIST = [column_name for column_name in DF]

    SELECTION = list_single_select(COLUMN_LIST, f'Select DataFrame column', 'value')

    return DF[SELECTION]


# USER SELECTS THE NAMES OF COLUMNS IN THE SPECIFIED DF, AND RETURNS THE COLUMN NAMES
def DF_column_multi_select(DF, number_to_choose):    #DF must be a pandas dataframe
    print('')
    COLUMN_LIST = [column_name for column_name in DF]

    SELECTION = list_multi_select(COLUMN_LIST, f'SELECT DataFrame COLUMNS', 'value', number_to_choose=number_to_choose)

    return DF[SELECTION]


def DF_column_custom_select(DF):    #DF must be a pandas dataframe

    print('')
    COLUMN_LIST = [column_name for column_name in DF]

    SELECTION = list_custom_select(DF, 'value')

    return DF[SELECTION]




if __name__ == '__main__':
    from string import ascii_lowercase

    LIST = [ascii_lowercase[:n.random.randint(1,15)] for _ in range(100)]
    _ = list_custom_select(LIST, 'value')
    print(_)











