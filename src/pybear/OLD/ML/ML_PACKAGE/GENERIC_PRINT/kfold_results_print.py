import numpy as np, pandas as pd









def kfold_results_print(DEV_RESULTS_MATRIX, REGULARIZATION_FACTORS):

    # THIS MAKES ALL DATAFRAME HEADERS AND INDEXES "UNSPARSE" AND CENTERS HEADERS
    pd.set_option('display.multi_sparse', False, 'display.colheader_justify', 'center')
    pd.set_option("display.max_rows", 10, "display.max_columns", 10, 'display.width', 150)
    # pd.set_option("display.float_format", lambda x: f'{x:.5g}')
    pd.options.display.float_format = '{:.5g}'.format

    BASE_HEADER = ['FACTOR', *range(1, len(DEV_RESULTS_MATRIX[0])+1), 'AVERAGE']

    KFOLD_RESULTS_HEADER = pd.MultiIndex(
        levels=[['REGULARIZATION', 'DEV SET', '      '], BASE_HEADER],
        codes=[[0,1] + list(map(lambda x: 2, DEV_RESULTS_MATRIX[0][1:])) + [2], list(range(len(BASE_HEADER)))]
    )

    del BASE_HEADER

    DEV_RESULTS_MATRIX = np.array(DEV_RESULTS_MATRIX).astype(np.float64)

    # INSERT AVERAGES RIGHTMOST
    DEV_RESULTS_MATRIX = np.insert(DEV_RESULTS_MATRIX, len(DEV_RESULTS_MATRIX[0]),
                                   np.average(DEV_RESULTS_MATRIX, axis=1), axis=1)

    # INSERT REG FACTORS LEFTMOST (DO THIS AFTER GETTING THE AVERAGE)
    DEV_RESULTS_MATRIX = np.insert(DEV_RESULTS_MATRIX, 0, REGULARIZATION_FACTORS, axis=1)

    DEV_RESULTS_DF = pd.DataFrame(data=DEV_RESULTS_MATRIX, columns=KFOLD_RESULTS_HEADER, index=None)

    print(DEV_RESULTS_DF)









if __name__ == '__main__':

    # TEST MODULE

    rows = 3
    columns = 5

    for num in (2, np.pi, 1000, 0.0000000000000000000324792, 4082794327843000000000000000000000):
        DEV_RESULTS_MATRIX = np.full((rows, columns), num)

        REGULARIZATION_FACTORS = np.arange(5, 5*(rows+1), 5).reshape((1,-1))[0]

        print(f'\n{num}:\n')
        kfold_results_print(DEV_RESULTS_MATRIX, REGULARIZATION_FACTORS)



























