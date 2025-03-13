import data_validation.validate_user_input as vui


#############################################################################################################
# DEFINE A FUNCTION FOR DYNAMICALLY PRINTING CATEGORIES WHILE SETTING UP CATEGORICAL SPLITTERS###############
# CALLED ONLY HERE
def category_splitter_print_fxn(PRINT_LIST, LABEL_RULES, label, number_of_labels, chars=120, fixed_rows=3):
    print(f'\nLabel {label} of {number_of_labels} \ncurrent label:')
    print(LABEL_RULES[-1], '\n')

    to_print, item_idx = '', -1
    while item_idx < len(PRINT_LIST)-1:
        item_idx += 1
        if item_idx < fixed_rows: print(f'{item_idx}) {PRINT_LIST[item_idx]}   ')
        else:
            new_entry = f'{item_idx}) {PRINT_LIST[item_idx]}   '
            #IF LEN OF CATEGORY STRING IS < CUTOFF, KEEP APPENDING
            if len(to_print) + len(new_entry) < chars: to_print += new_entry
            #IF LEN OF CATEGORY STRING IS >= CUTOFF OR HAVE REACHED LAST CATEGORY, PRINT
            elif len(to_print) + len(new_entry) >= chars:
                print(to_print)
                item_idx -= 1
                to_print = ''
    print(to_print, '\n')

#END DYNAMIC PRINT FUNCTION####################################################################################
#############################################################################################################


# CALLED BY target_config
def category_to_category(LABEL_RULES, number_of_labels, TARGET_UNIQUE_VALUES):

#HAVE TO BRING IN UNIQUE VALUES LIST, THIS SHLD HAVE BEEN DETERM WHEN FINDING DATA TYPE
#SHOULNT HAVE TO BRING IN DATA TYPE, SHOULD ALREADY BE DETERMINED THAT ITS CATEGORICAL

    #############################################################################################################
    # TARGET VECTOR SETUP########################################################################################
    while True:

        LABEL_RULES.clear()

        # 1-10-22  IF NO. LABELS == NO. UNIQUES, THEN JUST PARTITION BY EACH UNIQUE
        if number_of_labels == len(TARGET_UNIQUE_VALUES):
            for category in TARGET_UNIQUE_VALUES:
                LABEL_RULES.append([category])
            break

        label = 1    #STARTING LABEL NO.
        print('')
        if number_of_labels == 1:
            print(f'Number of labels = 1, logistic must be used.  Partition the categories into 2 groups, will specify')
            print(f'which categories are event or non-event later.')
        elif number_of_labels > 1:
            print(f'Number of labels > 1, softmax must be used.  Partition the categories that comprise each label,')
            print(f'one label at a time.  All categories must be selected.')

        ##################################################################################################
        #SETUP TO DISPLAY A DYNAMIC LIST OF CATEGORIES AVAILABLE FOR SELECTION, TO CHANGE / DISPLAY#######
        #AFTER EACH USER ACTION
        #RETAIN AN UNCHANGING COPY OF TARGET UNIQUES, THIS IS USED TO REFILL WIP_TUV AFTER FULL RESET
        TUV_BASE = sorted(TARGET_UNIQUE_VALUES.copy())

        # PUT COMMANDS AT THE TOP OF WIP_TUV SO COMMANDS CAN BE CHOSEN IN THE SAME WAY AS CATEGORIES...
        # WOULD BE MUCH MORE COMPLICATED TO PUT IN INTEGER VALIDATE & ALLOW WORD COMMANDS
        if number_of_labels == 1:
            TUV_COMMANDS = [
                            f'Accept and end label.  Must have at least 1 entry.',
                            f'Remove last entry',
                            f'Clear current label'
            ]

        if number_of_labels >= 2:
            TUV_COMMANDS = [
                f'Accept and end label.  Must have at least 1 entry.',
                f'Remove last entry',
                f'Clear current label',
                f'Clear all labels',
                f'Fill complementary set (on last label, & label must be empty)'
            ]

        WIP_TUV_BASE = TUV_BASE.copy()      #A SECONDARY CATEGORIES LIST THAT IS POPPED FROM DURING SELECTION
                                            #THAT IS USED WITH COMMANDS LIST TO RECONSTITUTE WIP_TUV WHEN
                                            #"REMOVE LAST ENTRY" OR "CLEAR CURRENT LABEL" IS USED
        WIP_TUV = TUV_COMMANDS + TUV_BASE
        WIP_TUV_BACKUP = WIP_TUV.copy()     #KEEP A STATIC COPY OF COMBINED LISTS FOR COMPLETE RESET

        # END DISPLAY DYNAMIC LIST SETUP##################################################################
        ##################################################################################################

        while label < number_of_labels + 1:  #REALLY DON'T NEED THIS FOR LOGIT, JUST TO MAKE LIKE SOFTMAX

            LABEL_RULES.append([])  # MAKES 'LIST WITHIN LIST' FORMAT CONSISTENT FOR LOGIT / SOFTMAX / FLOAT RULES

            while True:    #THIS while LOOPS FOR ENTRIES INTO CURRENT LABEL (1 LABEL CUZ ITS LOGIT)

                category_splitter_print_fxn(WIP_TUV, LABEL_RULES, label, number_of_labels, chars=120, fixed_rows=len(TUV_COMMANDS))

                while True:   #THIS while IS FOR DATA VALIDATION    (7/4/2021--- THIS LOOKS SUSPICIOUS, REDUNDANT)
                    selection = vui.validate_user_int(f'Select category # going into label {label}  > ', min=0, max=len(WIP_TUV) - 1)
                    if selection >= 0 and selection <= len(WIP_TUV)-1:
                        break
                    else:
                        print(f'Must be between 0 and {len(WIP_TUV)-1}')

                if number_of_labels >= 2:   #ONLY IF USING SOFT MAX ARE THESE 2 OPTIONS
                    if selection == 4:     #FILL COMPLEMENTARY (SOFTMAX ONLY)
                        if len(LABEL_RULES[-1]) > 0:
                            print('Cant put fill here, the label has to be empty (maybe clear label first?).')
                        elif len(LABEL_RULES) != number_of_labels:
                            print('Can only use complementary fill on last label.')
                        else:
                            LABEL_RULES[-1].append('COMPLEMENTARY')
                            WIP_TUV_BASE.clear()
                            WIP_TUV = TUV_COMMANDS + WIP_TUV_BASE
                            selection = 0

                    elif selection == 3:            #CLEAR ALL LABELS
                        label = 1
                        LABEL_RULES.clear()
                        WIP_TUV = WIP_TUV_BACKUP.copy()
                        WIP_TUV_BASE = TUV_BASE.copy()
                        break

                if selection == 2:            #CLEAR CURRENT LABEL
                    for item in LABEL_RULES[-1]:
                        WIP_TUV_BASE.append(item)

                    WIP_TUV_BASE.sort()
                    WIP_TUV = TUV_COMMANDS + WIP_TUV_BASE
                    LABEL_RULES[-1].clear()

                elif selection == 1 and len(LABEL_RULES[-1]) > 0:            #REMOVE LAST ENTRY
                    WIP_TUV_BASE.append(LABEL_RULES[-1].pop())
                    WIP_TUV_BASE.sort()
                    WIP_TUV = TUV_COMMANDS + WIP_TUV_BASE

                if selection == 0:            #ACCEPT CURRENT LABEL AS IS & GO TO NEXT
                    if number_of_labels > 1 and label == number_of_labels and len(WIP_TUV_BASE) > 0:
                        print('In softmax, all categories must be assigned to labels.')
                    elif len(LABEL_RULES[-1]) == 0:
                        print('The label is empty and cannot be.')
                    else:
                        print(f'\nLabel {label} ended.')
                        print(f'Label {label} looks like: {LABEL_RULES[-1]}')

                        if vui.validate_user_str(f'Accept label {label}? (y/n) > ','YN') == 'Y':
                            label += 1     #THIS DOESNT MATTER FOR LOGIT, ONLY SOFTMAX
                        else:
                            for item in LABEL_RULES[-1]:
                                WIP_TUV_BASE.append(item)

                            WIP_TUV_BASE.sort()
                            WIP_TUV = TUV_COMMANDS + WIP_TUV_BASE
                            LABEL_RULES.pop()
                        break

                elif selection not in range(len(TUV_COMMANDS)):
                    if number_of_labels == 1 and len(WIP_TUV_BASE) <= 1:
                        print('Cannot put all the categories into a single label for logistic.  Must end the label.')
                    elif number_of_labels > 1 and len(WIP_TUV_BASE) <= number_of_labels-label:
                        print('Cant put more in this label; need enough categories to put at least 1 in remaining labels.')
                    else:
                        LABEL_RULES[-1].append(WIP_TUV.pop(selection))
                        WIP_TUV_BASE.pop(selection-len(TUV_COMMANDS))

        print(f'\nLABEL_RULES:')
        for item_idx in range(len(LABEL_RULES)):
            print(f'Label {item_idx+1}: {LABEL_RULES[item_idx]}')

        if vui.validate_user_str('Accept LABEL_RULES config? (y/n) > ', 'YN') == 'Y':
            break
        else:
            LABEL_RULES.clear()

    #END TARGET VECTOR SETUP####################################################################################
    ############################################################################################################

    return LABEL_RULES




if __name__ == '__main__':
    LABEL_RULES = ['A','B','C']
    number_of_labels = 3
    TARGET_UNIQUE_VALUES = ['A','B','C','D','E','F','G','H']
    LABEL_RULES = category_splitter_category(LABEL_RULES, number_of_labels, TARGET_UNIQUE_VALUES)
    print(LABEL_RULES)






