import datetime, time, warnings, time
from general_sound import winlinsound as wls
import pandas as p, numpy as n
import sparse_dict as sd; from data_validation import validate_user_input as vui
from read_write_file.generate_full_filename import base_path_select as bps
p.set_option("display.max_rows", None, "display.max_columns", None)

# 9/23/22 THIS MODULE GENERATES A MATRIX WITH A RANDOM NUMBER OF CATEGORICAL COLUMNS, EACH COMPRISED OF
# A RANDOM NUMBER OF LEVELS, AND A RANDOM NUMBER OF FLOATS.  AAT IS CALCULATED FOR THIS MATRIX THEN ITS
# SPARSITY IS MEASURED.  THIS IS TO FIND A RELATIONSHIP BETWEEN THE ABOVE INPUTS AND A FINAL SPARSITY FOR AAT.




ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
HEADER_POOL = n.fromiter((f'{ALPHA[_//26]}{ALPHA[_%26]}' for _ in range(26*26)), dtype='<U2')

# rows = n.fromiter((range(100, 20001, 100)), dtype=int)

RESULTS_HEADER = ['ROWS', 'CAT_COLUMNS', 'MIN_CAT_LEVELS', 'MAX_CAT_LEVELS', 'AVG_CAT_LEVELS',
      'MIN_CAT_SP', 'MAX_CAT_SP', 'AVG_CAT_SP', 'FLOAT_COLUMNS', 'MIN_FLOAT_SP', 'MAX_FLOAT_SP',
       'AVG_FLOAT_SP', 'START_SPARSITY', 'FINAL_SPARSITY']

while True:
    duration = vui.validate_user_float(f'Enter duration cutoff in hrs > ', 0.00000001, 100)
    trial_cutoff = vui.validate_user_int(f'Enter number of trials cutoff > ', 1, 100000)
    size_cutoff = vui.validate_user_float(f'Enter size cutoff in millions of elements > ', 0.000000001, 500)
    if vui.validate_user_str(
        f'User entered time = {duration} hrs, max trials = {trial_cutoff}, and size limit = {size_cutoff} million.  '
        f'Accept? (y/n) > ', 'YN') == 'Y':
        break

DATA_HOLDER = n.zeros((14, trial_cutoff), dtype=float)

ctr = -1
t0 = time.time()
while True:
    cat_columns = n.random.randint(0,5)
    if cat_columns != 0:
        CAT_LEVELS = n.random.randint(2,50,(1,cat_columns))[0]
    else: CAT_LEVELS = n.empty((0,0), dtype=int)
    float_columns = n.random.randint(0,10)
    total_columns = n.sum(CAT_LEVELS) + float_columns
    if total_columns == 0: continue

    DATA_HEADER = HEADER_POOL[:cat_columns+float_columns]

    rows = n.random.randint(total_columns, 10000)

    print(f'\nTrying {rows} rows x {cat_columns} cat columns + {float_columns} floats ({total_columns} total)....')

    if (total_columns * rows) > size_cutoff * 1000000: continue  # IF AAT TOTAL SIZE > X TRY AGAIN

    # if (rows * rows) > 100000000: continue  # IF AAT TOTAL SIZE > X TRY AGAIN

    # if rows < columns: continue  # MORE COLUMNS THAN ROWS TRY AGAIN

    print(f'Running...')
    ctr += 1

    DATA = n.empty((0, rows), dtype=float)   # START AS [ [] = COLUMNS ]
    if cat_columns > 0:
        for feature_idx in range(cat_columns):
            DATA = n.vstack((DATA, n.random.choice(HEADER_POOL[:CAT_LEVELS[feature_idx]], (1,rows), replace=True)))

    FLOAT_SPARSITIES = n.empty((0,0), dtype=float)
    if float_columns > 0:
        for float_idx in range(float_columns):
            target_sparsity = n.random.randint(0,100)
            DUM = n.random.randint(1, 10001, (1, rows))
            cutoff = (1 - target_sparsity/100) * 10000
            DUM = DUM * (DUM <= cutoff)
            DUM = DUM / cutoff * 9
            DUM = n.ceil(DUM).astype(int)
            actual_sparsity = sd.list_sparsity(DUM)
            FLOAT_SPARSITIES = n.insert(FLOAT_SPARSITIES, len(FLOAT_SPARSITIES), actual_sparsity)
            DATA = n.vstack((DATA, DUM))

    DATA = DATA.transpose()
    DATA = p.DataFrame(data=DATA, columns=DATA_HEADER)

    # print()
    # print(DATA.iloc[:10,:5])

    if cat_columns > 0:
        DATA = p.get_dummies(DATA, columns = list(DATA.keys())[:cat_columns])

    # print()
    # print(DATA.iloc[:10,:15])

    DATA = DATA.to_numpy(dtype=float)
    DATA = DATA.transpose()

    # AT THIS POINT, get_dummies PUSHED ALL THE CATS TO THE END, AND FLOATS ARE UP FRONT
    CAT_SPARSITIES = n.empty((0,0), dtype=float)
    for cat_idx in range(len(DATA)-float_columns):
        DUM = DATA[cat_idx].copy()
        DUM.resize(1,len(DUM))
        _sparsity = sd.list_sparsity(DUM)
        CAT_SPARSITIES = n.insert(CAT_SPARSITIES, len(CAT_SPARSITIES), _sparsity)
    del DUM

    start_sparsity = sd.list_sparsity(DATA)

    # REMEMBER DATA IS [[] = COLS] FROM GETTING SPARSITIES
    AAT = n.matmul(DATA.transpose(), DATA, dtype=float)
    del DATA

    final_sparsity = sd.list_sparsity(AAT)
    del AAT

    # ['ROWS', 'CAT_COLUMNS', 'MIN_CAT_LEVELS', 'MAX_CAT_LEVELS', 'AVG_CAT_LEVLES',
    #  'MIN_CAT_SP', 'MAX_CAT_SP', 'AVG_CAT_SP',
    #  'FLOAT_COLUMNS', 'MIN_FLOAT_SP', 'MAX_FLOAT_SP', 'AVG_FLOAT_SP', 'FINAL_SPARSITY']
    DATA_HOLDER[0][ctr] = rows
    DATA_HOLDER[1][ctr] = cat_columns
    if len(CAT_LEVELS) == 0:
        DATA_HOLDER[2][ctr] = 0
        DATA_HOLDER[3][ctr] = 0
        DATA_HOLDER[4][ctr] = 0
    else:
        DATA_HOLDER[2][ctr] = n.min(CAT_LEVELS)
        DATA_HOLDER[3][ctr] = n.max(CAT_LEVELS)
        DATA_HOLDER[4][ctr] = n.average(CAT_LEVELS)
    if len(CAT_SPARSITIES)==0:
        DATA_HOLDER[5][ctr] = 100
        DATA_HOLDER[6][ctr] = 100
        DATA_HOLDER[7][ctr] = 100
    else:
        DATA_HOLDER[5][ctr] = n.min(CAT_SPARSITIES)
        DATA_HOLDER[6][ctr] = n.max(CAT_SPARSITIES)
        DATA_HOLDER[7][ctr] = n.average(CAT_SPARSITIES)
    DATA_HOLDER[8][ctr] = float_columns
    if len(FLOAT_SPARSITIES)==0:
        DATA_HOLDER[9][ctr] = 100
        DATA_HOLDER[10][ctr] = 100
        DATA_HOLDER[11][ctr] = 100
    else:
        DATA_HOLDER[9][ctr] = n.min(FLOAT_SPARSITIES)
        DATA_HOLDER[10][ctr] = n.max(FLOAT_SPARSITIES)
        DATA_HOLDER[11][ctr] = n.average(FLOAT_SPARSITIES)
    DATA_HOLDER[12][ctr] = start_sparsity
    DATA_HOLDER[13][ctr] = final_sparsity

    DH_T = DATA_HOLDER.transpose()
    DH_T = DH_T[:ctr+1]

    DF = p.DataFrame(data=DH_T, columns=RESULTS_HEADER, dtype=float).dropna()

    print(DF.tail(10))

    if ctr == trial_cutoff-1: break
    if time.time() - t0 > duration * 3600: break

basepath = bps.base_path_select()

p.DataFrame.to_excel(DF,
                     excel_writer=basepath + r'sparsity_tests_AAT.xlsx',
                     float_format='%.3f',
                     startrow=1,
                     startcol=1,
                     merge_cells=False
                     )
print('Done.')
wls.winlinsound(888, 1000)







