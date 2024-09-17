# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


import os, sys, inspect, io, time, base64
from copy import deepcopy
import pandas as pd, numpy as np
from debug import get_module_name as gmn
from read_write_file.generate_full_filename import (
    base_path_select as bps,
    filename_enter as fe
)




class CSVBase64RW:

    """

    DIRECTIONS FOR EMBEDDING A CSV FILE INTO A JUPYTER NOTEBOOK OR ANY
    OLD PYTHON MODULE
    TO CONVERT CSV TO BASE64:
    1) IMPORT CSVBase64RW MODULE AND INSTANTIATE CSVBase64RW LIKE
        EmbedderClass = CSVBase64RW.CSVBase64RW()
    2) PASS FULL FILE READ PATH AS ARG TO read_csv_into_base64 METHOD &
        RUN IT (OPTIONALLY ENTER encoding AS KWARG)
    3) USE METHODS print_base64_to_screen OR dump_base64_to_txt TO WRITE
        THE base64 STRING
    4) COPY & PASTE THE STRING TO ASSIGN IT TO A VARIABLE IN THE
        NOTEBOOK / .py WHERE THE DATA IS TO BE USED / STORED

    TO READ BASE64 INTO A DF:
    1) RUN IMPORT STATEMENTS IN THE NOTEBOOK / .py
    2) RUN THE CODE THAT MAKES THE ASSIGNMENT OF A VARIABLE TO THE BASE64
        STRING IN THE NOTEBOOK / .py
    3) PASS THE ASSIGNED VARIABLE TO THE base64_to_df METHOD OF A
        CSVBase64 INSTANCE, a PANDAS DF OF THE base64 IS RETURNED



    read_csv_into_base64()
    df_to_base64()
    print_base64_to_screen()
    dump_base64_to_txt()
    path_and_filename_handling()
    data_uri_handling()
    base64_to_csv()
    base64_to_df()
    dump_base64_as_df()

    """



    def __init__(self):

        """
        Converts / read csv to / from base64 encoded text string

        """

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))

        self.read_path = None
        self.data_uri = None
        self.data_uri_len = None




    def read_csv_into_base64(
        self,
        path,
        encoding=None
    ):


        """
        Convert passed csv file into base64 class attribute

        """

        self.read_path = f'{path}'

        with open(self.read_path, 'r', encoding=encoding) as file:
            csv_data = file.read()

        # Encode the CSV data using Base64
        encoded_data = base64.b64encode(csv_data.encode()).decode()

        # Create the data URI string
        self.data_uri = f'data:text/csv;base64,{encoded_data}'

        self.data_uri_len = len(self.data_uri)

        del csv_data, encoded_data

        # By executing this code in a Jupyter Notebook cell, the data URI string
        # will be created, and you can use it as needed within the notebook to
        # work with the embedded CSV data.

        # Note that embedding large CSV files in the notebook can increase the
        # file size and may impact performance. It's recommended to use this
        # approach for smaller datasets when portability is a priority. For
        # larger datasets, it may be more practical to read the CSV file from
        # the drive when needed.


    def df_to_base64(
        self,
        df
    ):

        """
        Encode a df as a csv then encode as base64.

        """


        self.data_uri = base64.b64encode(df.to_csv(index=False).encode()).decode()
        self.data_uri_len = len(self.data_uri)


    def print_base64_to_screen(self):
        """
        Print base64 encoding to screen

        """

        # Display the data URI string in the notebook / terminal
        line_len = 10000   # 100 is line length (screen width 100 not 140 because of jupyter)
        for i in range(0, len(self.data_uri), line_len):
            print(self.data_uri[i:i+line_len])
            time.sleep(.001)

        del line_len


    def dump_base64_to_txt(self, full_path_and_filename_wo_ext=None):
        """
        Dump base64 encoding to txt file

        """

        full_path_and_filename_wo_ext = \
            self.path_and_filename_handling(full_path_and_filename_wo_ext)

        with open(full_path_and_filename_wo_ext + '.txt', 'w') as f:
            f.write(str(self.data_uri))


    def path_and_filename_handling(self, full_path_and_filename_wo_ext=None):
        """
        Support module. Path and filename handling

        """
        if full_path_and_filename_wo_ext is None:
            full_path_and_filename_wo_ext = \
                os.path.join(bps.base_path_select(), fe.filename_wo_extension())

        return full_path_and_filename_wo_ext


    def data_uri_handling(self, data_uri):

        """
        Support module. Handling for data_uri when is a class attribute
        and/or passed to methods
        """

        # IF ATTR EXISTS & data_uri IS PASSED, JUST SEND THE passed data_uri
        # THRU, DONT UPDATE CLASS ATTRS
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
            raise ValueError(
                f'data_uri MUST BE PASSED AS A KWARG OR ALREADY BE AN '
                f'ATTRIBUTE OF THE INSTANCE'
            )

        return data_uri


    def base64_to_csv(self, data_uri=None):


        """Return base64 encoding as csv encoding"""

        data_uri = self.data_uri_handling(data_uri)

        # Extract the encoded CSV data
        encoded_data = data_uri.split(',', 1)[1]

        # Decode the base64-encoded data
        decoded_data = base64.b64decode(encoded_data) #data_uri)

        # Create a file-like object from the decoded data
        return io.StringIO(decoded_data.decode())


    def base64_to_df(
        self,
        data_uri=None,
        skiprows=None,
        header_row_zero_indexed=0,
        usecols=None
    ):

        """Return dataframe of base64 attribute"""

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


    def dump_base64_as_df(
            self,
            data_uri=None,
            skiprows=None,
            header_row_zero_indexed=0,
            usecols=None,
            full_path_and_filename_wo_ext=None
        ):

        """Convert base64 string attribute to df and save as csv file"""

        data_uri = self.data_uri_handling(data_uri)

        df = self.base64_to_df(
            data_uri=data_uri,
            skiprows=skiprows,
            header_row_zero_indexed=header_row_zero_indexed,
            usecols=usecols
        )

        full_path_and_filename_wo_ext = \
            self.path_and_filename_handling(full_path_and_filename_wo_ext)

        df.to_csv(full_path_and_filename_wo_ext + '.csv', index=False)









