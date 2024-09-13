import numpy as np, pandas as pd
import sparse_dict as sd
from general_data_ops import get_shape as gs
from MLObjects import MLObject as mlo


def print_vectors(TARGET_HEADER, TARGET_VECTOR, target_given_orientation, OUTPUT_VECTOR, output_given_orientation, ROW_ID_VECTOR):

    while True:

        TARGET_HEADER = np.array(TARGET_HEADER).reshape((1,-1))[0]

        ROW_ID_VECTOR = np.array(ROW_ID_VECTOR).reshape((1, -1))[0]

        TARGET_VECTOR = mlo.MLObject(TARGET_VECTOR, target_given_orientation, name='TARGET', return_orientation='ROW',
                                     return_format='ARRAY', bypass_validation=False).OBJECT

        OUTPUT_VECTOR = mlo.MLObject(OUTPUT_VECTOR, output_given_orientation, name='OUTPUT', return_orientation='ROW',
                                     return_format='ARRAY', bypass_validation=False).OBJECT

        tgt_rows, tgt_cols = gs.get_shape('TARGET', TARGET_VECTOR, 'ROW')
        out_rows, out_cols = gs.get_shape('OUTPUT', OUTPUT_VECTOR, 'ROW')


        zero_dims = ''
        for _dim, name in zip((tgt_cols,tgt_rows,out_cols,out_rows),
                              ('target columns','target rows','output columns','output rows')):
            if _dim==0: zero_dims += f'\n{name} is zero'
        if len(zero_dims) > 0: print(zero_dims); break
        del zero_dims


        base_msg = f'\n*** TARGET AND OUTPUT VECTOR HAVE DIFFERENT NUMBER OF '
        if tgt_cols != out_cols:
            print(base_msg + f'COLUMNS ({tgt_cols}, {out_cols}) IN print_vectors(). ***\n')
            break
        if tgt_rows != out_rows:
            print(base_msg + f'ROWS ({tgt_rows}, {out_rows}) IN print_vectors(). ***\n')
            break
        del base_msg

        DATA = np.hstack((TARGET_VECTOR, OUTPUT_VECTOR))

        print(f'\nPREVIEW OF TARGET AND OUTPUT VECTORS:\n')
        if tgt_cols==1:  # IF SINGLE VECTOR, JUST USUAL HEADER
            HEADER = ['TARGET','OUTPUT']
        elif tgt_cols > 1:  # IF SINGLE VECTOR, JUST USUAL HEADER
            HEADER = [[*map(lambda x: 'TARGET', TARGET_HEADER), *map(lambda x: 'OUTPUT', TARGET_HEADER)],\
                                [*TARGET_HEADER, *TARGET_HEADER]
                      ]

        DF = pd.DataFrame(data=DATA, columns=HEADER, index=ROW_ID_VECTOR)
        print(DF)
        del tgt_cols, tgt_rows, out_cols, out_rows, DF, DATA, HEADER

        break










if __name__ == '__main__':
    from MLObjects.TestObjectCreators import test_header as th
    from general_data_ops import get_dummies as gd

    _rows = 20
    ROW_ID_VECTOR = np.arange(0, 20)
    BASE_TARGET = np.random.randint(0, 2, (1, _rows))
    BASE_OUTPUT = np.random.uniform(0, 1, (1, _rows))
    TARGET_HEADER = th.test_header(1)

    print(f'TESTING....')

    SINGLE_HEADER = th.test_header(1)
    SINGLE_BASE_TARGET = np.random.randint(0, 2, (1, _rows))
    SINGLE_BASE_OUTPUT = np.random.uniform(0, 1, (1, _rows))

    DUM_TARGET = np.random.choice(('A','B','C'), _rows, replace=True)
    MULTI_BASE_TARGET, MULTI_HEADER = gd.get_dummies(DUM_TARGET, OBJECT_HEADER=SINGLE_HEADER,
                                       given_orientation = 'COLUMN', return_orientation = 'COLUMN')[:2]
    del DUM_TARGET
    DUM_OUTPUT = np.random.choice(('A','B','C'), _rows, replace=True)
    MULTI_BASE_OUTPUT, MULTI_HEADER = gd.get_dummies(DUM_OUTPUT, OBJECT_HEADER=SINGLE_HEADER,
                                       given_orientation='COLUMN', return_orientation='COLUMN')[:2]
    del DUM_OUTPUT

    MASTER_CLASSES = ['single','multi']
    MASTER_TARGET_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_TARGET_ORIENT = ['COLUMN', 'ROW']
    MASTER_OUTPUT_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_OUTPUT_ORIENT = ['COLUMN', 'ROW']

    for classes in MASTER_CLASSES:
        for target_format in MASTER_TARGET_FORMAT:
            for target_orient in MASTER_TARGET_ORIENT:
                for output_format in MASTER_OUTPUT_FORMAT:
                    for output_orient in MASTER_OUTPUT_ORIENT:

                        if classes=='single':
                            GIVEN_TARGET = SINGLE_BASE_TARGET.copy()
                            GIVEN_OUTPUT = SINGLE_BASE_OUTPUT.copy()
                            GIVEN_HEADER = SINGLE_HEADER.copy()
                        elif classes=='multi':
                            GIVEN_TARGET = MULTI_BASE_TARGET.copy()
                            GIVEN_OUTPUT = MULTI_BASE_OUTPUT.copy()
                            GIVEN_HEADER = MULTI_HEADER.copy()

                        if target_orient=='ROW': GIVEN_TARGET = GIVEN_TARGET.transpose()
                        if target_format=='SPARSE_DICT': GIVEN_TARGET = sd.zip_list_as_py_int(GIVEN_TARGET)

                        if output_orient=='ROW': GIVEN_OUTPUT = GIVEN_OUTPUT.transpose()
                        if output_format=='SPARSE_DICT': GIVEN_OUTPUT = sd.zip_list_as_py_int(GIVEN_OUTPUT)

                        print_vectors(GIVEN_HEADER, GIVEN_TARGET, target_orient, GIVEN_OUTPUT, output_orient, ROW_ID_VECTOR)


    print(f'\033[92m*** ALL TESTS PASSED. VISUALLY INSPECT OUTPUT. ***\033[0m')
    # END TEST SINGLE VECTOR ###############################################################################################


    # TEST SOFTMAX #########################################################################################################


    # END TEST SOFTMAX ###################################################################################################








