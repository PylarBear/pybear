import data_validation.validate_user_input as vui
import numpy as n


# number_of_labels ------>  logit must be 1, softmax >= 2

# CALLED BY target_config
def number_to_category(RAW_TARGET_VECTOR, LABEL_RULES, number_of_labels):

    # num_list_min & num_list_max are just for reference, are not functional
    num_list_min = n.min(RAW_TARGET_VECTOR)
    num_list_max = n.max(RAW_TARGET_VECTOR)

    ############################################################################################################
    #TARGET VECTOR SETUP########################################################################################

    options = f'''
    FLOAT PARTITIONING COMMANDS:
    equal(==), greater than(>), greater than or equal(>=) less than(<), less than or equal(<=), fill complementary(c),  
    end this label and go to next(n), reset all(r), reset current label(s), reprint these options(p)

    minimum value = {num_list_min}, maximum value = {num_list_max}
    '''

    while True:

        print(options)

        LABEL_RULES.clear()

        label = 1
        while label < number_of_labels+1:
            LABEL_RULES.append([])
            for step in range(1,5):       #4 POSSIBLE PIECES OF INFO FOR EACH FILTER
                while True:
                    if step == 1 and label < number_of_labels:
                        input = vui.validate_user_mstr(f'Enter 1st rule for label {label} ' \
                                + f'(of {number_of_labels}), options are ==, >, >=, <, <=, P > ',
                                                       ['==','>','>=','<','<=','P'])
                    elif step == 1 and label == number_of_labels:
                        input = vui.validate_user_mstr(f'Enter 1st rule for label {label} ' \
                                + f'(of {number_of_labels}), options are ==, >, >=, <, <=, C, P, R > ',
                                                       ['==','>','>=','<','<=','C','P','R'])
                    elif step == 2:
                        input = vui.validate_user_float(f'Label {label} is {input} ')
                    elif step == 3:
                        input = vui.validate_user_mstr(f'Enter 2nd rule for label {label} ' \
                                + f'(of {number_of_labels}), options are ==, >, >=, <, <=, N, P, R, S > ',
                                                       ['==','>','>=','<','<=','N','P','R','S'])
                    elif step == 4 and input != 'N':   #SKIP OVER THIS IF PICKED N ON STEP 3
                        input = vui.validate_user_float(f'Label {label} is {input} ')

                    if input == 'P':
                        print(f'\n{options}\n')
                        continue
                    else:
                        break

                if step in [2,4]:
                    LABEL_RULES[-1].append(input)

                if str(input) in ['==','>','>=','<','<=']:
                    LABEL_RULES[-1].append(input)

                if str(input) == 'C':
                    LABEL_RULES[-1].append('COMPLEMENTARY')

                if str(input) in 'CNR':
                    break

                if str(input) in 'S':
                    LABEL_RULES.pop()
                    label -= 1
                    break

            label += 1
            print()

            if str(input) in 'R':
                break

        if str(input) not in 'R':
            print(LABEL_RULES)
            for idx in range(len(LABEL_RULES)):
                if LABEL_RULES[idx][0] == 'COMPLEMENTARY':
                    print(f'Label {idx+1} rules:  Fill complementary')
                if len(LABEL_RULES[idx]) > 2:
                    second_rule = f', {LABEL_RULES[idx][2]} {LABEL_RULES[idx][3]}'
                else:
                    second_rule = ''
                if len(LABEL_RULES[idx]) >= 2:
                    print(f'label {idx+1} rules:  {LABEL_RULES[idx][0]} {LABEL_RULES[idx][1]}' \
                          + f'{second_rule}')

            if vui.validate_user_str('Accept float splitter settings? (y/n) > ', 'YN') == 'Y':
                break

    #END TARGET VECTOR SETUP####################################################################################
    ############################################################################################################


    return LABEL_RULES



if __name__ == '__main__':
    RAW_TARGET_VECTOR = [[1,2,3,4,5,6,2,6,26,2,43,231,25,12,34,234,12,1,246,2,212,34,2,1,2,2,5,3,2]]
    LABEL_RULES = []
    number_of_labels = 3
    num_list_min = 0
    num_list_max = 19
    LABEL_RULES = number_to_category(RAW_TARGET_VECTOR, LABEL_RULES, number_of_labels)
    print(LABEL_RULES)
















