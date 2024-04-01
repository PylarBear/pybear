import os, sys, inspect, io, time, base64
from copy import deepcopy
import pandas as pd, numpy as np
from debug import get_module_name as gmn
from read_write_file.generate_full_filename import base_path_select as bps, filename_enter as fe



# DIRECTIONS FOR EMBEDDING A CSV FILE INTO A JUPYTER NOTEBOOK OR ANY OLD PYTHON MODULE
# TO CONVERT CSV TO BASE64:
# 1) IMPORT CSVBase64RW MODULE AND INSTANTIATE CSVBase64RW LIKE EmbedderClass = CSVBase64RW.CSVBase64RW()
# 2) PASS FULL FILE READ PATH AS ARG TO read_csv_into_base64 METHOD & RUN IT (OPTIONALLY ENTER encoding AS KWARG)
# 3) USE METHODS print_base64_to_screen OR dump_base64_to_txt TO WRITE THE base64 STRING
# 4) COPY & PASTE THE STRING TO ASSIGN IT TO A VARIABLE IN THE NOTEBOOK / .py WHERE THE DATA IS TO BE USED / STORED

# TO READ BASE64 INTO A DF:
# 1) RUN IMPORT STATEMENTS IN THE NOTEBOOK / .py
# 2) RUN THE CODE THAT MAKES THE ASSIGNMENT OF A VARIABLE TO THE BASE64 STRING IN THE NOTEBOOK / .py
# 3) PASS THE ASSIGNED VARIABLE TO THE base64_to_df METHOD OF A CSVBase64 INSTANCE, a PANDAS DF OF THE base64 IS RETURNED



# read_csv_into_base64()            Convert passed csv file into base64 class attribute
# df_to_base64()                    Encode a df as a csv then encode as base64
# print_base64_to_screen()          Print base64 encoding to screen
# dump_base64_to_txt()              Dump base64 encoding to txt file
# path_and_filename_handling()      Support module. Path and filename handling
# data_uri_handling()               Support module. Handling for data_uri when is a class attribute and/or passed to methods
# base64_to_csv()                   Return base64 encoding as csv encoding
# base64_to_df()                    Return dataframe of base64 attribute
# dump_base64_as_df()               Convert base64 string attribute to df and save as csv file
# _exception()






class CSVBase64RW:
    '''Converts / read csv to / from base64 encoded text string'''

    def __init__(self):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))

        self.read_path = None
        self.data_uri = None
        self.data_uri_len = None


    def read_csv_into_base64(self, path, encoding=None):
        '''Convert passed csv file into base64 class attribute'''

        self.read_path = f'{path}'

        with open(self.read_path, 'r', encoding=encoding) as file:
            csv_data = file.read()

        # Encode the CSV data using Base64
        encoded_data = base64.b64encode(csv_data.encode()).decode()

        # Create the data URI string
        self.data_uri = f'data:text/csv;base64,{encoded_data}'

        self.data_uri_len = len(self.data_uri)

        del csv_data, encoded_data

        # By executing this code in a Jupyter Notebook cell, the data URI string will be created,
        # and you can use it as needed within the notebook to work with the embedded CSV data.

        # Note that embedding large CSV files in the notebook can increase the file size and may
        # impact performance. It's recommended to use this approach for smaller datasets or when
        # portability is a priority. For larger datasets, it may be more practical to read the CSV
        # file from the drive when needed.


    def df_to_base64(self, df):
        '''Encode a df as a csv then encode as base64.'''
        self.data_uri = base64.b64encode(df.to_csv(index=False).encode()).decode()
        self.data_uri_len = len(self.data_uri)


    def print_base64_to_screen(self):
        '''Print base64 encoding to screen'''

        # Display the data URI string in the notebook / terminal
        line_len = 10000   # 100 is line length (screen width 100 not 140 because of jupyter)
        for i in range(0, len(self.data_uri), line_len):
            print(self.data_uri[i:i+line_len])
            time.sleep(.001)

        del line_len


    def dump_base64_to_txt(self, full_path_and_filename_wo_ext=None):
        '''Dump base64 encoding to txt file'''

        full_path_and_filename_wo_ext = self.path_and_filename_handling(full_path_and_filename_wo_ext)

        with open(full_path_and_filename_wo_ext + '.txt', 'w') as f:
            f.write(str(self.data_uri))


    def path_and_filename_handling(self, full_path_and_filename_wo_ext=None):
        '''Support module. Path and filename handling'''
        if full_path_and_filename_wo_ext is None:
            full_path_and_filename_wo_ext = os.path.join(bps.base_path_select(), fe.filename_wo_extension())

        return full_path_and_filename_wo_ext


    def data_uri_handling(self, data_uri):
        '''Support module. Handling for data_uri when is a class attribute and/or passed to methods'''

        # IF ATTR EXISTS & data_uri IS PASSED, JUST SEND THE passed data_uri THRU, DONT UPDATE CLASS ATTRS
        if not data_uri is None and not self.data_uri is None: pass
        # IF ATTR EXISTS & data_uri NOT PASSED, SEND THRU self.data_uri
        elif data_uri is None and not self.data_uri is None:
            data_uri = self.data_uri
        # IF NO ATTR AND data_uri PASSED, UPDATE CLASS ATTRS W PASSED data_uri
        elif not data_uri is None and self.data_uri is None:
            self.data_uri = data_uri
            self.data_uri_len = len(self.data_uri)
        # IF NO ATTR AND data_uri NOT PASSED, EXCEPT
        elif not data_uri and not self.data_uri:
            self._exception(f'data_uri MUST BE PASSED AS A KWARG OR ALREADY BE AN ATTRIBUTE OF THE INSTANCE',
                            fxn=inspect.stack()[0][3])

        return data_uri


    def base64_to_csv(self, data_uri=None):
        '''Return base64 encoding as csv encoding'''

        data_uri = self.data_uri_handling(data_uri)

        # Extract the encoded CSV data
        encoded_data = data_uri.split(',', 1)[1]

        # Decode the base64-encoded data
        decoded_data = base64.b64decode(encoded_data) #data_uri)

        # Create a file-like object from the decoded data
        return io.StringIO(decoded_data.decode())


    def base64_to_df(self, data_uri=None, skiprows=None, header_row_zero_indexed=0, usecols=None):
        '''Return dataframe of base64 attribute'''

        # data_uri_handling() INSIDE base64_to_csv
        csv_like_object = self.base64_to_csv(data_uri)

        # Read the CSV data using the file-like object
        df = pd.read_csv(
                            csv_like_object,
                            header=header_row_zero_indexed,
                            skiprows=skiprows,
                            usecols=usecols
        )

        del csv_like_object

        return df


    def dump_base64_as_df(self, data_uri=None, skiprows=None, header_row_zero_indexed=0, usecols=None,
                            full_path_and_filename_wo_ext=None):
        '''Convert base64 string attribute to df and save as csv file'''

        data_uri = self.data_uri_handling(data_uri)

        df = self.base64_to_df(
                                data_uri=data_uri,
                                skiprows=skiprows,
                                header_row_zero_indexed=header_row_zero_indexed,
                                usecols=usecols
        )

        full_path_and_filename_wo_ext = self.path_and_filename_handling(full_path_and_filename_wo_ext)

        df.to_csv(full_path_and_filename_wo_ext + '.csv', index=False)


    def _exception(self, words, fxn=None):
        fxn = f'.{fxn}()' if not fxn is None else ''
        raise Exception(f'{self.this_module}{fxn} >>> {words}')








if __name__ == '__main__':

    # MODULE AND TEST CODE VERIFIED GOOD 7/15/23

    from data_validation import validate_user_input as vui

    # INSTANTIATE A CLASS
    Base64 = CSVBase64RW()

    def prompter(prompt): input(prompt + '. HIT ENTER > ')

    #########################################################################################################

    prompter(f'\n\nSTART CSV READ TO BASE64')

    __ = vui.validate_user_str(f'\nRead(r) a csv from a user-entered file path or skip(s) and use a pre-loaded uri > ', 'RS')
    if __ == 'R':
        filepath = os.path.join(bps.base_path_select(), fe.filename_wo_extension())
        Base64.read_csv_into_base64(filepath + f'.csv')  # print_to_screen=True
    elif __ == 'S':
        Base64.data_uri = 'data:text/csv;base64,YSxiLGMsZCxlLGYsZyxoLGkKMC44NjIxMjkwNjIsMC41NDgwNTY3MjUsMC4zNTQ5MDU2LDAuOTUwODc4NTc2LDAuMTQ4NDQ0OTg5LDAuNjU2NzY1NTE0LDAuMjQzMDkyNjMyLDAuNTUxNzI4MTg3LDAuNTEyNjU0NjkzCjAuNDU5MjUxOTk5LDAuMDE5NTk0MTgsMC43NTgxNTUxNTMsMC4xMjE2Njg4MDEsMC4zNjg3NzkxNTgsMC42NjEyNzI1NDQsMC45NDQzOTUzMjIsMC44MTI5NTEyNTYsMC4wODc3NDk1MjEKMC43ODIyODU5NjUsMC42NjUyNzU1NzMsMC43NTg4MDgxOTIsMC45ODAyNjA1MDksMC4wNjYyNTk4NjQsMC4zNTYyOTMyMjMsMC4zODUxMDgwOTEsMC4xODE3NjI2OTMsMC44ODk2MTI0ODYKMC41MTc0NDY3NzUsMC40MDE0NTk5LDAuNTcwNzI5MzM3LDAuMDIwNDU1MTgzLDAuOTM2ODI0MzIzLDAuODMxMjQ2NDM4LDAuNzE3MzkzMzk2LDAuMjczNDMwNTcyLDAuNjk3NDM3Njk1CjAuMzI0Njg2NywwLjk2OTg4NDk0NywwLjgwMTAxNTY4NCwwLjU5MDc3MDkwOSwwLjQyMTI4OTAyOCwwLjQ4NzUxMjIzMSwwLjg5MzYxNTA1MSwwLjgzMjAzMTUxOCwwLjgzMzc4MTI2MwowLjE4MjUxNjY2MywwLjg2OTY0NDQ2LDAuMDc2MTkxOTQ4LDAuMTQyNDU0MTcyLDAuNjgyMDQ0Mzg5LDAuNTc2MjcyNzc3LDAuODc2MDU1NDg0LDAuNzI1OTg5NDgsMC40MTk0OTA2OTUKMC40OTkxNjY5NjksMC44NTg2OTI2MTEsMC4xMTY4OTU2MTYsMC40NjE2OTI0NjEsMC4xNTAwNDkzODYsMC42MjIzOTkwMzYsMC44OTI4NjM4MzcsMC40NjU5Nzc3MzEsMC41OTg3ODk1OTkKMC44NzQyMDY0MDcsMC45MDA0MDE1MzQsMC45Mjg5Njg4NzUsMC45MTI0MTE4NzIsMC4xNjg2MDU4MjIsMC45MDU3Njk4NzEsMC42NDAxNDczNjksMC43OTgyMjk1MTcsMC4xMDM5NTM3MDkKMC40MzE3NzU0NjgsMC40NTU0MjAxMjUsMC4yMzgwMTkyMTQsMC45MTY2OTg1OTksMC40ODczOTc3NzksMC4zNzA5OTAwMzYsMC4wNjIwNjYzNjksMC4yMjI3NjQyNzksMC44OTI4NTA1NDEKMC4zNjc4MzY4MjIsMC4wOTYyMDE1MDcsMC4yODU4NDE3OTMsMC45NTk1NDU5MjksMC4wMjc4MzI1MzEsMC45MDM1Mjk3MSwwLjIwNjU1MjQ3OCwwLjU2ODY1NzE4LDAuMjA0MDY0NDUyCjAuMjY3NzEwMjIxLDAuMTc5OTE3NjMzLDAuMTcxNDM5NTYsMC43NjY4OTMwNzQsMC4zODMzMzU2ODcsMC43NTk5ODAzMzcsMC4wMDM2NjE0MjksMC42NDIwMDUzNjksMC4yNzkwNDg2MjUKMC44OTAzMzU2ODUsMC4yMzAyMzA2ODEsMC45ODEwMzYyMDcsMC43MTUzMTI5NCwwLjY2Mjc4MTkzMywwLjU4MTgyOTI0OSwwLjM2MDg5NzU2NCwwLjczMjI0NzgzNiwwLjU4NDcwODY2MwowLjA5Mjg5MjE1OCwwLjk3OTk4MjIxOSwwLjY4OTIzNzc3NSwwLjIzODgwOTUwMywwLjIzMDk2OTYwMiwwLjIxMDkzMTI5NiwwLjQyMTkyMzQ5MiwwLjcwNjY2MDMxNiwwLjcxMjczMjgzMwowLjMxMDY2NjMyMSwwLjc2ODE4OTY0OCwwLjIyMjM1NzI0NywwLjI4NjM0ODQyNywwLjkzNDk4NDk5NywwLjk1Mzk2NTE5OCwwLjM3NjYzODg5OSwwLjE3ODE2NDkyMiwwLjUxNTM4MzI1MgowLjM0NDkzMTcxOCwwLjg2MTM2MTg3MiwwLjg4NDQ1MDQxNywwLjUwNjY5MjM2OCwwLjkyNjAyOTY0MywwLjI2MTgxMjUxNiwwLjU2NzQxMTY0MSwwLjE3Mjk4MzIxNSwwLjk3MDU3MzE2OQowLjk1NTQxMTA0NiwwLjc5MjI4MTE5OCwwLjIwNDIyNTE5NCwwLjY1Nzc4NDAzNCwwLjk3NzM0NjI2NCwwLjMxNjU1NzEyMSwwLjgyMDgyMjI0MSwwLjA5ODU2MjkxNCwwLjM2NDExNDMxMgowLjg0MDQwOTk3MiwwLjE4ODQzMzc3OSwwLjE0MTgwNzI1NCwwLjU0Nzg3NDQyNSwwLjA5MDgxOTE3NCwwLjk1OTQ4MzYsMC42NTU4MTAwMjcsMC4wMjAwOTYxNDcsMC4zNjcwMTY0NDMKMC43MTA2MTM3ODgsMC40NjI3OTUxOTYsMC4wNDc0MDAxNTgsMC4zNjE3MTY0MTQsMC4wNTgwNzA1OTEsMC4zNTAxNjEwMzIsMC45OTczOTgzMDIsMC4xMjg1OTExMjcsMC4yMTUyODEyOTIKMC4zMjc5NTAwNDIsMC45NTQ1Mzg2MjksMC40NTU5MDU3MDEsMC44NTA4MDgxMzYsMC4yOTIwMTYzOCwwLjU3NjM4MTY3MSwwLjg5MzEyOTY5NSwwLjMxMTgzMzA1NywwLjk5MDM5OTk5MgowLjczNTc2OTgyOSwwLjg1MjgxMDQ1MiwwLjQzMTk1NDIzOCwwLjI1ODU4NzcwNSwwLjA5Nzk2Mzc1NCwwLjU2OTgwODMyNCwwLjc4Mzc1NTkyNCwwLjY4NTI5NDQxOCwwLjQ4ODQ5NTc4MQowLjcyNzQ2NDU1NCwwLjk0NTQxNDY0NSwwLjExMjY2NDA5MSwwLjg0MzM3NzAyNiwwLjAwNDU4ODc4MSwwLjQ2ODgzNjA5NywwLjM4OTQxMjQxNiwwLjg5NzQzODAwNSwwLjg3NTkyNDI2Cg=='
        Base64.data_uri_len = len(Base64.data_uri)

    base_uri = deepcopy(Base64.data_uri)

    print(f'\ndata uri len: {Base64.data_uri_len}')
    prompter(f'\n*** LEN OF URI STRING SHOULD BE PRINTED ABOVE ***')
    #########################################################################################################


    #########################################################################################################
    prompter(f'\n\nSTART DF TO BASE64')

    DUMMY_DF = pd.DataFrame(data=np.random.randint(0,10,(5,3)).astype(np.int8), columns=['A','B','C'])

    Base64.df_to_base64(DUMMY_DF)

    print(f'\ndata uri len: {Base64.data_uri_len}')
    prompter(f'\n*** LEN OF URI STRING SHOULD BE PRINTED ABOVE ***')

    del DUMMY_DF
    Base64.data_uri = base_uri
    #########################################################################################################


    #########################################################################################################
    prompter(f'\n\nSTART PRINT BASE64 TO SCREEN')

    DF = Base64.print_base64_to_screen()

    prompter(f'\n*** THERE SHOULD BE A PRINTOUT OF GIBBERISH ABOVE ***')
    #########################################################################################################


    #########################################################################################################
    prompter(f'\n\nSTART DUMP BASE64 TO TXT, PATH NOT GIVEN (SHOULD PROMPT FOR PATH AND FILENAME WHILE INSIDE dump_base64_to_txt)')

    Base64.dump_base64_to_txt()

    prompter(f'\n*** CHECK THE PATH ENTERED FOR A TXT FILE WITH GIBBERISH IN IT ***')
    #########################################################################################################


    #########################################################################################################
    prompter(f'\n\nSTART DUMP BASE64 TO TXT, PATH GIVEN (SHOULD PROMPT FOR A FILE PATH TO PASS TO dump_base64_to_txt)')

    txtpath = os.path.join(bps.base_path_select(), fe.filename_wo_extension())
    Base64.dump_base64_to_txt(full_path_and_filename_wo_ext=txtpath)

    prompter(f'\n*** CHECK THE PATH ENTERED FOR A TXT FILE WITH GIBBERISH IN IT ***')
    #########################################################################################################

    # START base64_to_df() TESTS

    #########################################################################################################
    prompter(f'\n\nSTART PRINT BASE64 AS DF, self.data_uri is not None and data_uri IS PASSED (SHOULD NOT PROMPT FOR ANYTHING)')

    data_uri = deepcopy(base_uri)
    DF = Base64.base64_to_df(data_uri=data_uri, skiprows=None, header_row_zero_indexed=0)
    print()
    print(DF)

    prompter(f'\n*** DATAFRAME SHOULD BE PRINTED TO SCREEN ABOVE ***')
    #########################################################################################################


    #########################################################################################################
    prompter(f'\n\nSTART PRINT BASE64 AS DF, self.data_uri is not None and data_uri NOT PASSED (SHOULD NOT PROMPT FOR ANYTHING)')

    DF = Base64.base64_to_df(data_uri=None, skiprows=None, header_row_zero_indexed=0)
    print()
    print(DF)

    prompter(f'\n*** DATAFRAME SHOULD BE PRINTED TO SCREEN ABOVE ***')
    #########################################################################################################


    #########################################################################################################
    prompter(f'\n\nSTART PRINT BASE64 AS DF, self.data_uri is None and data_uri IS PASSED (SHOULD NOT PROMPT FOR ANYTHING)')

    Base64.data_uri = None
    data_uri = deepcopy(base_uri)
    DF = Base64.base64_to_df(data_uri=data_uri, skiprows=None, header_row_zero_indexed=0, usecols=None)
    print()
    print(DF)

    prompter(f'\n*** DATAFRAME SHOULD BE PRINTED TO SCREEN ABOVE ***')
    #########################################################################################################


    #########################################################################################################
    prompter(f'\n\nSTART PRINT BASE64 AS DF, self.data_uri is None and data_uri NOT PASSED (SHOULD BE EXCEPTION HANDLED)')

    Base64.data_uri = None
    try: DF =  Base64.base64_to_df(data_uri=None, skiprows=None, header_row_zero_indexed=0, usecols=None)
    except: print(f'\033[92m\n *** EXCEPTION WAS RAISED AS EXPECTED *** \033[0m')
    prompter(f'\n*** THERE SHOULD BE A MESSAGE ABOVE INDICATING AN EXCEPTION WAS HANDLED ***')
    #########################################################################################################

    # START dump_base64_as_df()
    Base64.data_uri = deepcopy(base_uri)

    #########################################################################################################
    prompter(f'\n\nSTART DUMP BASE64 AS DF TO CSV, full_path_and_filename_wo_ext IS PASSED '
             f'(SHOULD PROMPT FOR A DUMP PATH TO PASS TO dump_base64_as_df)')

    dumppath = os.path.join(bps.base_path_select(), fe.filename_wo_extension())

    Base64.dump_base64_as_df(data_uri=None, skiprows=None, header_row_zero_indexed=0, usecols=None,
                                full_path_and_filename_wo_ext=dumppath)

    prompter(f'\n*** CHECK THE PATH ENTERED FOR A CSV FILE WITH DATA IN IT ***')
    #########################################################################################################

    #########################################################################################################
    prompter(f'\n\nSTART DUMP BASE64 AS DF TO CSV, full_path_and_filename_wo_ext NOT PASSED '
             f'(SHOULD PROMPT FOR A PATH TO DUMP TO WHILE IN dump_base64_as_df)')

    DF = Base64.dump_base64_as_df(data_uri=None, skiprows=None, header_row_zero_indexed=0, usecols=None,
                                  full_path_and_filename_wo_ext=None)

    prompter(f'\n*** CHECK THE PATH ENTERED FOR A CSV FILE WITH DATA IN IT ***')
    #########################################################################################################



    print(f'\n'*3 + f'\033[92m*** TESTS COMPLETE ***\033[0m')






























