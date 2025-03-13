# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import os, io, base64


# MAKE BASE64 EMBED OF TEXT, EITHER FROM A PYTHON VARIABLE OR FILE


### FROM PYTHON VARIABLE ******************************************************

# ENCODE

def string_encoder(instruction_string_as_python_variable):
    """
    DUMPS A base64 ENCODING OF A PYTHON STRING TO THE GIVEN PATH AND FILENAME.
    RETRIEVE THE base64 STRING FROM THAT FILE AND PASTE INTO A STRING VARIABLE
    DECLARATION INSIDE PYTHON SCRIPT.
    """

    filename = 'base64_text_dump.txt'

    if os.name == 'nt':
        desktop_path = r'c:\users\bill\desktop'
    elif os.name == 'posix':
        desktop_path = r'/home/bear/Desktop'

    path_and_file_name = os.path.join(desktop_path, filename)

    # WRITE THE CONTENT OF THE PYTHON VARIABLE AS BINARY TO A FILE
    with open(path_and_file_name, 'w') as f:
        f.write(instruction_string_as_python_variable)

    # READ IN THE BINARY CONTENT INTO io.BytesIO
    with open(path_and_file_name, 'rb') as f:
        IO_HOLDER = io.BytesIO(f.read())

    # EXTRACT THE BINARY CONTENT FROM io.BytesIO, ENCODE, AND SAVE ENCODING TO FILE
    with open(path_and_file_name, 'w') as f:
        f.write(base64.b64encode(IO_HOLDER.getvalue()).decode())

    del instruction_string_as_python_variable, IO_HOLDER, desktop_path


# DECODE

# base64_embed = ''

# base64.b64decode(base64_embed).decode()


#### DEMO *********************************************************************

example_string = """
THIS
IS
ONLY
A
TEST
OF
EMBEDDING
TEXT
FROM
A
PYTHON
VARIABLE
"""

string_encoder(example_string)
del example_string

base64_embed = 'ClRISVMKSVMKT05MWQpBClRFU1QKT0YKRU1CRURESU5HClRFWFQKRlJPTQpBClBZVEhPTgpWQVJJQUJMRQo='

# VERIFY THE ENCODING
print(base64.b64decode(base64_embed).decode())
del base64_embed


# ONCE THE base64 ENCODING IS RETRIEVED, DELETE THE FILE
# os.remove(path_and_file_name)

#### END DEMO *****************************************************************

### END FROM PYTHON VARIABLE **************************************************


### FROM TEXT FILE ************************************************************

# ENCODE

def text_file_embedder(path_of_file_to_encode):
    """
    DUMPS A base64 ENCODING OF A TEXT FILE TO THE GIVEN PATH AND FILENAME.
    RETRIEVE THE base64 STRING FROM THAT FILE AND PASTE INTO A STRING VARIABLE
    DECLARATION INSIDE PYTHON SCRIPT.
    """

    filename = 'base64_text_dump.txt'

    if os.name == 'nt':
        desktop_path = r'c:\users\bill\desktop'
    elif os.name == 'posix':
        desktop_path = r'/home/bear/Desktop'

    path_and_file_name = os.path.join(desktop_path, filename)

    # READ IN THE BINARY CONTENT INTO io.BytesIO
    with open(path_of_file_to_encode, 'rb') as f:
        IO_HOLDER = io.BytesIO(f.read())

    # EXTRACT THE BINARY CONTENT FROM io.BytesIO, ENCODE, AND SAVE ENCODING TO FILE
    with open(path_and_file_name, 'w') as f:
        f.write(base64.b64encode(IO_HOLDER.getvalue()).decode())

    del path_of_file_to_encode, IO_HOLDER, desktop_path


# DECODE
# base64.b64decode(base64_embed).decode()


#### DEMO *********************************************************************

# CREATE AN EXAMPLE FILE

example_string = """
THIS
IS
ONLY
A
TEST
OF
EMBEDDING
TEXT
FROM
A
PYTHON
VARIABLE
"""

if os.name == 'nt':
    filepath = r'c:\users\bill\desktop\base64_demo_file.txt'
elif os.name == 'posix':
    filepath = r'/home/bear/Desktop/base64_demo_file.txt'

with open(filepath, 'w') as f:
    f.write(example_string)

del example_string

text_file_embedder(filepath)

base64_embed = 'ClRISVMKSVMKT05MWQpBClRFU1QKT0YKRU1CRURESU5HClRFWFQKRlJPTQpBClBZVEhPTgpWQVJJQUJMRQo='

# VERIFY ENCODING
print(base64.b64decode(base64_embed).decode())
del base64_embed

# ONCE THE base64 ENCODING IS RETRIEVED, DELETE THE FILE
# os.remove(path_and_file_name)

#### END DEMO *****************************************************************

### END FROM TEXT FILE ********************************************************








