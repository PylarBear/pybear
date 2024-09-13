import numpy as n, pandas as p

#CALLED BY read_write_file.filetype_package_pair_readers.filetype_package_pair_readers
def text_python_reader(filename, delimiter, object_type):

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
        OBJECT = n.array(OBJECT)
    elif object_type == 'DATAFRAME':
        OBJECT = p.DataFrame(columns='TEXT',data=OBJECT)

    return OBJECT

