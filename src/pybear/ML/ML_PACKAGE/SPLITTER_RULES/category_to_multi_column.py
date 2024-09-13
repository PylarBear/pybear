from data_validation import validate_user_input as vui
from ML_PACKAGE._data_validation import validate_modified_object_type as vmot
from general_data_ops import return_uniques as ru

# CALLED BY target_config
def category_to_multi_column(RAW_TARGET_VECTOR, LABEL_RULES, number_of_labels):

    print('\nMulti-column conversion to labels\n'.upper())
    print('The n columns selected must be binary, and each example row must have 1 positive event case and n-1 negative cases.\n')

    ############################################################################################################
    # TARGET VECTOR SETUP########################################################################################

    # GET ALL UNIQUES IN ENTIRE TARGET VECTOR (TO FIND IF BINARY, ETC)
    INDIV_COLUMNS_UNIQUE_VALUES = []
    for COLUMN in RAW_TARGET_VECTOR:
        data_type = vmot.validate_modified_object_type(COLUMN)
        INDIV_COLUMNS_UNIQUE_VALUES.append(ru.return_uniques(COLUMN, [], data_type, suppress_print='Y')[0])

    print(f'TARGET VECTOR columns are comprised of:')
    [print(f'COLUMN {_})  {INDIV_COLUMNS_UNIQUE_VALUES[_]}') for _ in range(len(INDIV_COLUMNS_UNIQUE_VALUES))]

    ALL_COLUMNS_VALUES = []
    for _ in INDIV_COLUMNS_UNIQUE_VALUES: ALL_COLUMNS_VALUES += [*_]
    data_type = vmot.validate_modified_object_type(ALL_COLUMNS_VALUES)
    ALL_COLUMNS_UNIQUE_VALUES = ru.return_uniques(ALL_COLUMNS_VALUES, [], data_type, suppress_print='Y')[0]

    print(f'\nTARGET VECTORS uniques looks like:')
    print(ALL_COLUMNS_UNIQUE_VALUES)
    print()

    print(f'For each unique item, indicate if it corresponds to a positive(p) or negative(n) event.')

    #LABEL RULES SIZE MUST CONFORM TO THE RULES USED IN target_run FXN, WHERE CAT2CAT & NUM2CAT HAVE NO. LABEL RULES
    #EQUALS NUMBER OF LABELS. SO HERE SIMPLY REPLICATING THE (P) CASES INTO ALL LABEL_RULES COLUMNS

    LABEL_RULES_HOLDER = []   #CREATE A HOLDER LIST TO BE REPLICATED IN LABEL_RULES FOR ALL LABEL COLUMNS
    # LATER ON, EACH COLUMN IN RAW_TARGET_VECTOR IS PROCESSED BY ITS RESPECIVE LABEL_RULES LIST, AND MATCHES AGAINST
    # THE THINGS IN ITS LABEL_RULES LIST IS CONVERTED TO THE positive VALUE, & ALL NOT MATCHES TO negative VALUE
    while True:
        for item in ALL_COLUMNS_UNIQUE_VALUES:
            event_choice = vui.validate_user_str(f'{item} --- positive(p) or negative(n) event >  ', 'PN')
            if event_choice == 'P':
                LABEL_RULES_HOLDER.append(item)
            #IGNORE (N) ENTRIES

        LABEL_RULES.clear()
        for label in range(number_of_labels): LABEL_RULES.append(LABEL_RULES_HOLDER) # REPLICATE number_of_labels TIMES

        print(f'\nLABEL RULES looks like (indicated values are turned to positive values, all others to negative:')
        [print(f'COLUMN {_})   {LABEL_RULES[_]}') for _ in range(len(LABEL_RULES))]
        print()

        if vui.validate_user_str('Accept event selections? (y/n) > ','YN') == 'Y':
            break

    #THE LABEL_RULES OBJECT MUST BE SAME LEN AS TARGET VECTOR!!!!!
    #FOR MULTI-COLUMN, # OF RAW_TARGET_COLUMNS = # LABELS

    #END TARGET VECTOR SETUP####################################################################################
    ############################################################################################################


    return LABEL_RULES




if __name__ == '__main__':
    RAW_TARGET_VECTOR = [[1,2,3,4],[2,3,4,5],[3,4,5,6]]
    LABEL_RULES = []
    data_type = 'INT'
    number_of_labels = 3

    LABEL_RULES = category_splitter_multi_column(RAW_TARGET_VECTOR, LABEL_RULES, data_type, number_of_labels)

    print(f'LABEL RULES looks like:')
    print(LABEL_RULES)