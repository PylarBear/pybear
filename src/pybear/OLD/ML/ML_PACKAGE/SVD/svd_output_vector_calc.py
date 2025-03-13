import numpy as np




def svd_output_vector_calc(DATA, WINNING_COLUMNS, COEFFS, orientation='column'):

    # PUT INTO []=COLUMNS TO COMPILE WINNING_DATA
    if orientation == 'column': pass
    elif orientation == 'row': DATA = np.array(DATA, dtype=float).transpose()
    else: raise ValueError(f'\n*** INVALID orientation IN svd_output_vector_calc. ***\n')

    WINNING_DATA = [DATA[_] for _ in WINNING_COLUMNS]

    if len(WINNING_DATA) != len(COEFFS):
        print(f'\n*** UNABLE TO COMPUTE OUTPUT VECTOR, SIZE MISMATCH BETWEEN DATA AND COEFFICIENT VECTOR ***\n')
        OUTPUT_VECTOR = []
    else:
        # TRANSPOSE TO [] = ROWS FOR MATMUL
        WINNING_DATA = np.array(WINNING_DATA, dtype=float).transpose()

        OUTPUT_VECTOR = np.array([np.matmul(WINNING_DATA.astype(float), np.array(COEFFS, dtype=float))], dtype=float)

    del WINNING_DATA

    return OUTPUT_VECTOR






if __name__ == '__main__':

    rows = 20
    columns = 15
    winning_columns = 5
    DATA = np.random.randint(0,10,(columns,rows))
    COEFFS = np.random.uniform(0,1,(1,winning_columns))
    WINNING_COLUMNS = np.random.randint(0, columns, winning_columns)

    OUTPUT_VECTOR = svd_output_vector_calc(DATA, WINNING_COLUMNS, COEFFS, orientation='column')

    print(OUTPUT_VECTOR)

    from ML_PACKAGE.MUTUAL_INFORMATION import svd_error_calc as siec

    TARGET_VECTOR = [np.random.rand() for _ in range(rows)]

    error = siec.svd_error_calc(OUTPUT_VECTOR, TARGET_VECTOR)

    print(error)