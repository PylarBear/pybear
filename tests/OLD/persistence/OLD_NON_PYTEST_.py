# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import os
from copy import deepcopy
import numpy as np

from pybear.persistence.CSVBase64RW import CSVBase64RW

from pybear.data_validation import validate_user_input as vui



# INSTANTIATE A CLASS
Base64 = CSVBase64RW()


def prompter(prompt):
    input(prompt + '. HIT ENTER > ')


###############################################################################

prompter(f'\n\nSTART CSV READ TO BASE64')

__ = vui.validate_user_str(
    f'\nRead(r) a csv from a user-entered file path or skip(s) and use a pre-loaded uri > ',
    'RS')
if __ == 'R':
    filepath = os.path.join(bps.base_path_select(), fe.filename_wo_extension())
    Base64.read_csv_into_base64(filepath + f'.csv')  # print_to_screen=True
elif __ == 'S':
    Base64.data_uri = 'data:text/csv;base64,YSxiLGMsZCxlLGYsZyxoLGkKMC44NjIxMjkwNjIsMC41NDgwNTY3MjUsMC4zNTQ5MDU2LDAuOTUwODc4NTc2LDAuMTQ4NDQ0OTg5LDAuNjU2NzY1NTE0LDAuMjQzMDkyNjMyLDAuNTUxNzI4MTg3LDAuNTEyNjU0NjkzCjAuNDU5MjUxOTk5LDAuMDE5NTk0MTgsMC43NTgxNTUxNTMsMC4xMjE2Njg4MDEsMC4zNjg3NzkxNTgsMC42NjEyNzI1NDQsMC45NDQzOTUzMjIsMC44MTI5NTEyNTYsMC4wODc3NDk1MjEKMC43ODIyODU5NjUsMC42NjUyNzU1NzMsMC43NTg4MDgxOTIsMC45ODAyNjA1MDksMC4wNjYyNTk4NjQsMC4zNTYyOTMyMjMsMC4zODUxMDgwOTEsMC4xODE3NjI2OTMsMC44ODk2MTI0ODYKMC41MTc0NDY3NzUsMC40MDE0NTk5LDAuNTcwNzI5MzM3LDAuMDIwNDU1MTgzLDAuOTM2ODI0MzIzLDAuODMxMjQ2NDM4LDAuNzE3MzkzMzk2LDAuMjczNDMwNTcyLDAuNjk3NDM3Njk1CjAuMzI0Njg2NywwLjk2OTg4NDk0NywwLjgwMTAxNTY4NCwwLjU5MDc3MDkwOSwwLjQyMTI4OTAyOCwwLjQ4NzUxMjIzMSwwLjg5MzYxNTA1MSwwLjgzMjAzMTUxOCwwLjgzMzc4MTI2MwowLjE4MjUxNjY2MywwLjg2OTY0NDQ2LDAuMDc2MTkxOTQ4LDAuMTQyNDU0MTcyLDAuNjgyMDQ0Mzg5LDAuNTc2MjcyNzc3LDAuODc2MDU1NDg0LDAuNzI1OTg5NDgsMC40MTk0OTA2OTUKMC40OTkxNjY5NjksMC44NTg2OTI2MTEsMC4xMTY4OTU2MTYsMC40NjE2OTI0NjEsMC4xNTAwNDkzODYsMC42MjIzOTkwMzYsMC44OTI4NjM4MzcsMC40NjU5Nzc3MzEsMC41OTg3ODk1OTkKMC44NzQyMDY0MDcsMC45MDA0MDE1MzQsMC45Mjg5Njg4NzUsMC45MTI0MTE4NzIsMC4xNjg2MDU4MjIsMC45MDU3Njk4NzEsMC42NDAxNDczNjksMC43OTgyMjk1MTcsMC4xMDM5NTM3MDkKMC40MzE3NzU0NjgsMC40NTU0MjAxMjUsMC4yMzgwMTkyMTQsMC45MTY2OTg1OTksMC40ODczOTc3NzksMC4zNzA5OTAwMzYsMC4wNjIwNjYzNjksMC4yMjI3NjQyNzksMC44OTI4NTA1NDEKMC4zNjc4MzY4MjIsMC4wOTYyMDE1MDcsMC4yODU4NDE3OTMsMC45NTk1NDU5MjksMC4wMjc4MzI1MzEsMC45MDM1Mjk3MSwwLjIwNjU1MjQ3OCwwLjU2ODY1NzE4LDAuMjA0MDY0NDUyCjAuMjY3NzEwMjIxLDAuMTc5OTE3NjMzLDAuMTcxNDM5NTYsMC43NjY4OTMwNzQsMC4zODMzMzU2ODcsMC43NTk5ODAzMzcsMC4wMDM2NjE0MjksMC42NDIwMDUzNjksMC4yNzkwNDg2MjUKMC44OTAzMzU2ODUsMC4yMzAyMzA2ODEsMC45ODEwMzYyMDcsMC43MTUzMTI5NCwwLjY2Mjc4MTkzMywwLjU4MTgyOTI0OSwwLjM2MDg5NzU2NCwwLjczMjI0NzgzNiwwLjU4NDcwODY2MwowLjA5Mjg5MjE1OCwwLjk3OTk4MjIxOSwwLjY4OTIzNzc3NSwwLjIzODgwOTUwMywwLjIzMDk2OTYwMiwwLjIxMDkzMTI5NiwwLjQyMTkyMzQ5MiwwLjcwNjY2MDMxNiwwLjcxMjczMjgzMwowLjMxMDY2NjMyMSwwLjc2ODE4OTY0OCwwLjIyMjM1NzI0NywwLjI4NjM0ODQyNywwLjkzNDk4NDk5NywwLjk1Mzk2NTE5OCwwLjM3NjYzODg5OSwwLjE3ODE2NDkyMiwwLjUxNTM4MzI1MgowLjM0NDkzMTcxOCwwLjg2MTM2MTg3MiwwLjg4NDQ1MDQxNywwLjUwNjY5MjM2OCwwLjkyNjAyOTY0MywwLjI2MTgxMjUxNiwwLjU2NzQxMTY0MSwwLjE3Mjk4MzIxNSwwLjk3MDU3MzE2OQowLjk1NTQxMTA0NiwwLjc5MjI4MTE5OCwwLjIwNDIyNTE5NCwwLjY1Nzc4NDAzNCwwLjk3NzM0NjI2NCwwLjMxNjU1NzEyMSwwLjgyMDgyMjI0MSwwLjA5ODU2MjkxNCwwLjM2NDExNDMxMgowLjg0MDQwOTk3MiwwLjE4ODQzMzc3OSwwLjE0MTgwNzI1NCwwLjU0Nzg3NDQyNSwwLjA5MDgxOTE3NCwwLjk1OTQ4MzYsMC42NTU4MTAwMjcsMC4wMjAwOTYxNDcsMC4zNjcwMTY0NDMKMC43MTA2MTM3ODgsMC40NjI3OTUxOTYsMC4wNDc0MDAxNTgsMC4zNjE3MTY0MTQsMC4wNTgwNzA1OTEsMC4zNTAxNjEwMzIsMC45OTczOTgzMDIsMC4xMjg1OTExMjcsMC4yMTUyODEyOTIKMC4zMjc5NTAwNDIsMC45NTQ1Mzg2MjksMC40NTU5MDU3MDEsMC44NTA4MDgxMzYsMC4yOTIwMTYzOCwwLjU3NjM4MTY3MSwwLjg5MzEyOTY5NSwwLjMxMTgzMzA1NywwLjk5MDM5OTk5MgowLjczNTc2OTgyOSwwLjg1MjgxMDQ1MiwwLjQzMTk1NDIzOCwwLjI1ODU4NzcwNSwwLjA5Nzk2Mzc1NCwwLjU2OTgwODMyNCwwLjc4Mzc1NTkyNCwwLjY4NTI5NDQxOCwwLjQ4ODQ5NTc4MQowLjcyNzQ2NDU1NCwwLjk0NTQxNDY0NSwwLjExMjY2NDA5MSwwLjg0MzM3NzAyNiwwLjAwNDU4ODc4MSwwLjQ2ODgzNjA5NywwLjM4OTQxMjQxNiwwLjg5NzQzODAwNSwwLjg3NTkyNDI2Cg=='
    Base64.data_uri_len = len(Base64.data_uri)

base_uri = deepcopy(Base64.data_uri)

print(f'\ndata uri len: {Base64.data_uri_len}')
prompter(f'\n*** LEN OF URI STRING SHOULD BE PRINTED ABOVE ***')
###############################################################################


###############################################################################
prompter(f'\n\nSTART DF TO BASE64')

DUMMY_DF = pd.DataFrame(data=np.random.randint(0, 10, (5, 3)).astype(np.int8),
                        columns=['A', 'B', 'C'])

Base64.df_to_base64(DUMMY_DF)

print(f'\ndata uri len: {Base64.data_uri_len}')
prompter(f'\n*** LEN OF URI STRING SHOULD BE PRINTED ABOVE ***')

del DUMMY_DF
Base64.data_uri = base_uri
###############################################################################


###############################################################################
prompter(f'\n\nSTART PRINT BASE64 TO SCREEN')

DF = Base64.print_base64_to_screen()

prompter(f'\n*** THERE SHOULD BE A PRINTOUT OF GIBBERISH ABOVE ***')
###############################################################################


###############################################################################
prompter(
    f'\n\nSTART DUMP BASE64 TO TXT, PATH NOT GIVEN (SHOULD PROMPT FOR PATH AND FILENAME WHILE INSIDE dump_base64_to_txt)')

Base64.dump_base64_to_txt()

prompter(
    f'\n*** CHECK THE PATH ENTERED FOR A TXT FILE WITH GIBBERISH IN IT ***')
###############################################################################


###############################################################################
prompter(
    f'\n\nSTART DUMP BASE64 TO TXT, PATH GIVEN (SHOULD PROMPT FOR A FILE PATH TO PASS TO dump_base64_to_txt)')

txtpath = os.path.join(bps.base_path_select(), fe.filename_wo_extension())
Base64.dump_base64_to_txt(full_path_and_filename_wo_ext=txtpath)

prompter(
    f'\n*** CHECK THE PATH ENTERED FOR A TXT FILE WITH GIBBERISH IN IT ***')
###############################################################################

# START base64_to_df() TESTS

###############################################################################
prompter(
    f'\n\nSTART PRINT BASE64 AS DF, self.data_uri is not None and data_uri IS PASSED (SHOULD NOT PROMPT FOR ANYTHING)')

data_uri = deepcopy(base_uri)
DF = Base64.base64_to_df(data_uri=data_uri, skiprows=None,
                         header_row_zero_indexed=0)
print()
print(DF)

prompter(f'\n*** DATAFRAME SHOULD BE PRINTED TO SCREEN ABOVE ***')
###############################################################################


###############################################################################
prompter(
    f'\n\nSTART PRINT BASE64 AS DF, self.data_uri is not None and data_uri NOT PASSED (SHOULD NOT PROMPT FOR ANYTHING)')

DF = Base64.base64_to_df(data_uri=None, skiprows=None,
                         header_row_zero_indexed=0)
print()
print(DF)

prompter(f'\n*** DATAFRAME SHOULD BE PRINTED TO SCREEN ABOVE ***')
###############################################################################


###############################################################################
prompter(
    f'\n\nSTART PRINT BASE64 AS DF, self.data_uri is None and data_uri IS PASSED (SHOULD NOT PROMPT FOR ANYTHING)')

Base64.data_uri = None
data_uri = deepcopy(base_uri)
DF = Base64.base64_to_df(data_uri=data_uri, skiprows=None,
                         header_row_zero_indexed=0, usecols=None)
print()
print(DF)

prompter(f'\n*** DATAFRAME SHOULD BE PRINTED TO SCREEN ABOVE ***')
###############################################################################


###############################################################################
prompter(
    f'\n\nSTART PRINT BASE64 AS DF, self.data_uri is None and data_uri NOT PASSED (SHOULD BE EXCEPTION HANDLED)')

Base64.data_uri = None
try:
    DF = Base64.base64_to_df(data_uri=None, skiprows=None,
                             header_row_zero_indexed=0, usecols=None)
except:
    print(f'\033[92m\n *** EXCEPTION WAS RAISED AS EXPECTED *** \033[0m')
prompter(
    f'\n*** THERE SHOULD BE A MESSAGE ABOVE INDICATING AN EXCEPTION WAS HANDLED ***')
###############################################################################

# START dump_base64_as_df()
Base64.data_uri = deepcopy(base_uri)

###############################################################################
prompter(
    f'\n\nSTART DUMP BASE64 AS DF TO CSV, full_path_and_filename_wo_ext IS PASSED '
    f'(SHOULD PROMPT FOR A DUMP PATH TO PASS TO dump_base64_as_df)')

dumppath = os.path.join(bps.base_path_select(), fe.filename_wo_extension())

Base64.dump_base64_as_df(data_uri=None, skiprows=None,
                         header_row_zero_indexed=0, usecols=None,
                         full_path_and_filename_wo_ext=dumppath)

prompter(f'\n*** CHECK THE PATH ENTERED FOR A CSV FILE WITH DATA IN IT ***')
###############################################################################

###############################################################################
prompter(
    f'\n\nSTART DUMP BASE64 AS DF TO CSV, full_path_and_filename_wo_ext NOT PASSED '
    f'(SHOULD PROMPT FOR A PATH TO DUMP TO WHILE IN dump_base64_as_df)')

DF = Base64.dump_base64_as_df(data_uri=None, skiprows=None,
                              header_row_zero_indexed=0, usecols=None,
                              full_path_and_filename_wo_ext=None)

prompter(f'\n*** CHECK THE PATH ENTERED FOR A CSV FILE WITH DATA IN IT ***')
###############################################################################


print(f'\n' * 3 + f'\033[92m*** TESTS COMPLETE ***\033[0m')



