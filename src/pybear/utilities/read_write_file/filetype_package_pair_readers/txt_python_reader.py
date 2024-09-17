# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


import numpy as np, pandas as pd



def text_python_reader(
    filename,
    delimiter,
    object_type
):

    # GET RAW DATA FROM TXT FILE
    f = open(filename, encoding="utf8")

    OBJECT = []
    for LINE in f.readlines():
        OBJECT.append([])
        OBJECT[-1].append(LINE)

    # ADD DELIMITER STUFF



    if object_type == 'LIST_OF_LISTS':
        pass
    elif object_type == 'NUMPY_ARRAY':
        OBJECT = np.array(OBJECT)
    elif object_type == 'DATAFRAME':
        OBJECT = pd.DataFrame(columns='TEXT',data=OBJECT)

    return OBJECT

