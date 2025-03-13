import numpy as np




# INHERITED BY ALL ##################################################################################################
# rsq()                         # Define function for rsq
# conditional_print_for_all()   # Define what to report based on what the output will be as determined by the final link function
# always_printed_for_all()      # Statistics printed for all ML packages
# OVERWRITTEN IN CHILD ##############################################################################################
# iteration_print()             # Available for use by ML packages that have iterations
# ml_specific_print()           # Print instructions for specific ML module
# print()                       # Build printout for specific ML module by piecing together


class SummaryStatisticsPrintTemplate:

    def __init__(self, OUTPUT_VECTOR, TARGET_VECTOR, link, error):
        self.OUTPUT_VECTOR = OUTPUT_VECTOR
        self.TARGET_VECTOR = TARGET_VECTOR
        self.link = link
        self.error = error


    def rsq(self):
        # Define function for rsq
        # CANT DO THE "JUST 1s IN SOFTMAX TARGET" STYLE... VECTOR OF ALL 1 GIVES ERROR IN RSQ
        return np.corrcoef(self.TARGET_VECTOR.ravel(), self.OUTPUT_VECTOR.ravel())[0][1]**2


    def conditional_print_for_all(self):
        # CONDITIONALLY PRINTED FOR ALL # ALWAYS PRINTED ###################################################################
        # Define what to report based on what the output will be as determined by the final link function
        # INHERITED BY CHILD
        if self.link == 'None':
            print(f'\nOUTPUT MAX = {np.max(self.OUTPUT_VECTOR[0]):.8f}')
            print(f'OUTPUT MIN = {np.min(self.OUTPUT_VECTOR[0]):.8f}')
            print(f'OUTPUT AVG = {np.mean(self.OUTPUT_VECTOR[0]):.8f}')

        elif self.link == 'ReLU_lower': pass
        elif self.link == 'ReLU_upper': pass
        elif self.link == 'ReLU_lower_and_upper': pass
        elif self.link == 'Tanh': pass

        elif self.link in ['Logistic','Softmax', 'SVM_Perceptron']:

            OUTPUT_EVENT_VALUES = []
            OUTPUT_NO_EVENT_VALUES = []

            for col_idx in range(len(self.TARGET_VECTOR)):
                for example_idx in range(len(self.TARGET_VECTOR[col_idx])):
                    if self.TARGET_VECTOR[col_idx][example_idx] == 1:
                        OUTPUT_EVENT_VALUES.append(self.OUTPUT_VECTOR[col_idx][example_idx])
                    else:
                        OUTPUT_NO_EVENT_VALUES.append(self.OUTPUT_VECTOR[col_idx][example_idx])

            print(f'\nOUTPUT_EVENT_MAX = {np.max(OUTPUT_EVENT_VALUES):.8f}')
            print(f'OUTPUT_EVENT_MIN = {np.min(OUTPUT_EVENT_VALUES):.8f}')
            print(f'OUTPUT_EVENT_AVG = {np.mean(OUTPUT_EVENT_VALUES):.8f}')
            print(f'OUTPUT_NO_EVENT_MAX = {np.max(OUTPUT_NO_EVENT_VALUES):.8f}')
            print(f'OUTPUT_NO_EVENT_MIN = {np.min(OUTPUT_NO_EVENT_VALUES):.8f}')
            print(f'OUTPUT_NO_EVENT_AVG = {np.mean(OUTPUT_NO_EVENT_VALUES):.8f}')
            print(f'TRUE HIT RATE = {round(100 * len(OUTPUT_EVENT_VALUES) / len(self.TARGET_VECTOR[0]), 2)}%')
            error_rate = 100 * np.sum([round(_,0)==0 for _ in OUTPUT_EVENT_VALUES] +
                                           [round(__,0)==1 for __ in OUTPUT_NO_EVENT_VALUES]) / \
                                    len(OUTPUT_EVENT_VALUES+OUTPUT_NO_EVENT_VALUES)
            print(f'ERROR RATE (0.5 CUTOFF) = {round(error_rate, 2)}%')


    def always_printed_for_all(self):
        # Statistics printed for all ML packages
        # INHERITED BY CHILD
        print(f'\nMIN, MAX, AVG TARGET_ELEMENT = {np.min(self.TARGET_VECTOR):.8f}, {np.max(self.TARGET_VECTOR):.8f}, {np.mean(self.TARGET_VECTOR):.8f}')
        print(f'MIN, MAX, AVG OUTPUT_ELEMENT = {np.min(self.OUTPUT_VECTOR):.8f}, {np.max(self.OUTPUT_VECTOR):.8f}, {np.mean(self.OUTPUT_VECTOR):.8f}')
        print(f'OUTPUT / TARGET R-SQUARED = {self.rsq()}')
        print(f'AVERAGE ERROR = {self.error / len(self.TARGET_VECTOR[0]):.8f}')


    def iteration_print(self):
        # Available for use by ML packages that have iterations
        # OVERWRITTEN IN CHILD
        pass


    def ml_specific_print(self):
        # Print instructions for specific ML module
        # OVERWRITTEN IN CHILD
        pass


    def print(self):
        # Build printout for specific ML module by piecing together
        # iteration_print()
        # conditional_print_for_all()
        # always_printed_for_all()
        # ml_specific_print()
        # OVERWRITTEN IN CHILD
        pass


from ML_PACKAGE.openpyxl_shorthand import openpyxl_write as ow


# INHERITED BY ALL ##################################################################################################
# custom_write                  # custom module for simplifying openpyxl data writing
# return_fxn()                  # Return
# rsq()                         # Define function for rsq
# conditional_dump_for_all()    # Define what to report based on what the output will be as determined by the final link function
# always_dumped_for_all()       # Statistics dumped for all ML packages
# OVERWRITTEN IN CHILD ##############################################################################################
# iteration_dump()              # Available for use by ML packages that have iterations
# ml_specific_dump()            # Dump instructions for specific ML module
# dump()                        # Build dump for specific ML module by piecing together



class SummaryStatisticsDumpTemplate:

    def __init__(self, wb, OUTPUT_VECTOR, TARGET_VECTOR, link, error, sheet_name):
        # DIFFERENT SHEET NAME FOR IF TRAIN, DEV, OR TEST, SPECIFIED INSITU
        self.wb = wb
        self.OUTPUT_VECTOR = OUTPUT_VECTOR
        self.TARGET_VECTOR = TARGET_VECTOR
        self.link = link
        self.error = error
        self.sheet_name = sheet_name

        self.wb.create_sheet(self.sheet_name)

        self.custom_write(self.sheet_name, 1, 1, self.sheet_name, 'left', 'center', True)

        self.row_counter = 1
        self.column_counter = 2


    def custom_write(self, sheet, row, column, value, horiz, vert, bold):
        # custom module for simplifying openpyxl data writing
        ow.openpyxl_write(self.wb, sheet, row, column, value, horiz=horiz, vert=vert, bold=bold)


    def return_fxn(self):
        return self.wb


    def rsq(self):
        # Define function for rsq
        # CANT DO THE "JUST 1s IN SOFTMAX TARGET" STYLE... VECTOR OF ALL 1 GIVES ERROR IN RSQ
        return np.corrcoef(np.hstack((self.TARGET_VECTOR)).astype(np.float64),
                          np.hstack((self.OUTPUT_VECTOR)).astype(np.float64)
                          )[0][1]**2


    def conditional_dump_for_all(self):
        # CONDITIONALLY DUMPED FOR ALL # ALWAYS DUMPED ###################################################################
        # Define what to report based on what the output will be as determined by the final link function
        # INHERITED BY CHILD

        self.row_counter += 2

        if self.link in ['None', 'ReLU_lower', 'ReLU_upper']:
            OUTPUT_STATS_HEADER = ['OUTPUT MAX', 'OUTPUT MIN', 'OUTPUT AVG']
            OUTPUT_STATS = [f'{np.max(self.OUTPUT_VECTOR[0]):.8f}',f'{np.min(self.OUTPUT_VECTOR[0]):.8f}',
                            f'{np.mean(self.OUTPUT_VECTOR[0]):.8f}']

            for idx in range(len(OUTPUT_STATS_HEADER)):
                self.custom_write(self.sheet_name, self.row_counter, self.column_counter, OUTPUT_STATS_HEADER[idx],
                                'left', 'center', False)
                self.custom_write(self.sheet_name, self.row_counter, self.column_counter + 4, OUTPUT_STATS[idx],
                                'left', 'center', False)
                self.row_counter += 1

        elif self.link in ['Logistic','Softmax', 'Tanh', 'ReLU_lower_and_upper', 'SVM_Perceptron']:

            OUTPUT_EVENT_VALUES = []
            OUTPUT_NO_EVENT_VALUES = []

            for col_idx in range(len(self.TARGET_VECTOR)):
                for example_idx in range(len(self.TARGET_VECTOR[col_idx])):
                    if self.TARGET_VECTOR[col_idx][example_idx] == 1:
                        OUTPUT_EVENT_VALUES.append(self.OUTPUT_VECTOR[col_idx][example_idx])
                    else:
                        OUTPUT_NO_EVENT_VALUES.append(self.OUTPUT_VECTOR[col_idx][example_idx])


            OUTPUT_STATS_HEADER = ['OUTPUT EVENT MAX', 'OUTPUT EVENT MIN', 'OUTPUT EVENT AVG', 'OUTPUT NO EVENT MAX',
                                   'OUTPUT NO EVENT MIN', 'OUTPUT NO EVENT AVG', 'ERROR RATE (0.5 CUTOFF)']
            error_rate = 100 * np.sum([round(_,0)==0 for _ in OUTPUT_EVENT_VALUES] +
                                           [round(__,0)==1 for __ in OUTPUT_NO_EVENT_VALUES]) / \
                                    len(OUTPUT_EVENT_VALUES+OUTPUT_NO_EVENT_VALUES)
            OUTPUT_STATS =  [f'{np.max(OUTPUT_EVENT_VALUES):.8f}', f'{np.min(OUTPUT_EVENT_VALUES):.8f}',
                             f'{np.mean(OUTPUT_EVENT_VALUES):.8f}', f'{np.max(OUTPUT_NO_EVENT_VALUES):.8f}',
                             f'{np.min(OUTPUT_NO_EVENT_VALUES)}', f'{np.mean(OUTPUT_NO_EVENT_VALUES):.8f}',
                             f'{round(error_rate,2)}%']

            for idx in range(len(OUTPUT_STATS_HEADER)):
                self.custom_write(self.sheet_name, self.row_counter, self.column_counter, OUTPUT_STATS_HEADER[idx],
                                                                'left', 'center', False)
                self.custom_write(self.sheet_name, self.row_counter, self.column_counter + 4, OUTPUT_STATS[idx], ''
                                                                'left', 'center', False)
                self.row_counter += 1


    def always_dumped_for_all(self):
        # Statistics dumped for all ML packages
        # INHERITED BY CHILD

        self.row_counter += 2

        self.custom_write(self.sheet_name, self.row_counter, self.column_counter, 'MIN, MAX, AVG TARGET ELEMENT', 'left', 'center', False)
        self.custom_write(self.sheet_name,
                          self.row_counter,
                          self.column_counter + 4,
                          f'{np.min(self.TARGET_VECTOR):.8f}, {np.max(self.TARGET_VECTOR):.8f}, {np.mean(self.TARGET_VECTOR):.8f}',
                          'left',
                          'center',
                          False)
        self.row_counter += 1

        self.custom_write(self.sheet_name, self.row_counter, self.column_counter, 'MIN, MAX, AVG OUTPUT ELEMENT', 'left', 'center', False)
        self.custom_write(self.sheet_name,
                          self.row_counter,
                          self.column_counter + 4,
                          f'{np.min(self.OUTPUT_VECTOR):.8f}, {np.max(self.OUTPUT_VECTOR):.8f}, {np.mean(self.OUTPUT_VECTOR):.8f}',
                          'left',
                          'center',
                          False)
        self.row_counter += 1

        self.custom_write(self.sheet_name, self.row_counter, self.column_counter, 'OUTPUT / TARGET R-SQUARED', 'left', 'center', False)
        self.custom_write(self.sheet_name, self.row_counter, self.column_counter + 4, f'{self.rsq()}', 'left', 'center', False)
        self.row_counter += 1

        self.custom_write(self.sheet_name, self.row_counter, self.column_counter, 'AVERAGE ERROR', 'left', 'center', False)
        self.custom_write(self.sheet_name, self.row_counter, self.column_counter + 4,
                          f'{self.error / len(self.TARGET_VECTOR[0]):.8f}', 'left', 'center', False)
        self.row_counter += 1


    def iteration_dump(self):
        # Available for use by ML packages that have iterations
        # OVERWRITTEN IN CHILD

        # self.row_counter += 2

        pass


    def ml_specific_dump(self):
        # Print instructions for specific ML module
        # OVERWRITTEN IN CHILD

        # self.row_counter += 2

        pass


    def dump(self):
        # Build excel dump for specific ML module by piecing together
        # OVERWRITTEN IN CHILD
        # iteration_dump()
        # conditional_dump_for_all()
        # always_dumped_for_all()
        # ml_specific_dump()
        # return self.return_fxn()
        pass


