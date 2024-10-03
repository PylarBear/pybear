# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np




# DO IT OUT THE LONG WAY, TO KEEP COLUMNS AS uint8 AND DELETE
# DUPLICATE/SPARSE COLUMNS IN-PROCESS




# INSTEAD OF ACTUALLY ADDING THE INTERACTION COLUMNS TO X_encoded_np IN
# THIS STEP, BUILD TWO VECTORS CONTAINING idx1 & idx2 RESPECTIVELY TO
# INDICATE WHICH PAIRS OF COLUMNS TO USE TO ADD INTERACTIONS TO X_encoded_np.


# pizza, the second de-duplicator ** * ** * ** * ** * ** * ** * ** * ** *
def equality_checker(COLUMN1, COLUMN2):
    INTERACTION = (COLUMN1 * COLUMN2).astype(np.uint8)
    if INTERACTION.sum() < min_cutoff:
        return True
    elif np.array_equiv(INTERACTION, COLUMN1):
        return True
    elif np.array_equiv(INTERACTION, COLUMN2):
        return True
    else:
        return False


_ = len(COLUMNS)

print(f'# columns start = {_}')

col_idxs = range(_)

possible_columns = _ * (_ - 1) / 2
del _

IDX1_POINTERS = np.empty(0, dtype=np.uint16)
IDX2_POINTERS = np.empty(0, dtype=np.uint16)

for idx1 in col_idxs:

    if (idx1 + 1) % 100 == 0:
        print(f'Working on column number {idx1 + 1} of {len(col_idxs)}...')

    # IF LOOKING AT PAIRS WITHIN A DUMMY GROUP (EG, RI & AZ IN "STATE") THE INTERACTION WILL ALWAYS BE ZEROS.
    # THIS IS NOT THE CASE FOR "NAICS_ALL_{X}"
    # IF BOTH IDX1 & IDX2 ARE IN PORTAL, TYPE, STATE, MCAP, YEAR, MONTH, DAY, WEEKDAY, THEN JUST SKIP
    idx1_col_char4 = str(COLUMNS[idx1][:4])

    COLUMN1 = X_encoded_np[:, idx1].astype(np.uint8)

    for idx2 in range(idx1):

        idx2_col_char4 = str(COLUMNS[idx2][:4])

        if idx1_col_char4 in ['PORT', 'TYPE', 'STAT', 'MCAP', 'YEAR', 'MONT',
                              'DAY_', 'WEEK'] and \
                idx2_col_char4 == idx1_col_char4:
            continue

        COLUMN2 = X_encoded_np[:, idx2].astype(np.uint8)

        if equality_checker(COLUMN1, COLUMN2) == True:
            continue
        else:
            IDX1_POINTERS = np.insert(IDX1_POINTERS, len(IDX1_POINTERS), idx1, axis=0)
            IDX2_POINTERS = np.insert(IDX2_POINTERS, len(IDX2_POINTERS), idx2, axis=0)

print(f'\n# columns end = {len(COLUMNS) + len(IDX1_POINTERS)}')
print(
    f'Found{len(IDX1_POINTERS): ,.0f} interaction columns of{possible_columns: ,.0f} possible interactions.')

del col_idxs, possible_columns, COLUMN1, COLUMN2, equality_checker
try:
    del i
except:
    pass
try:
    del j
except:
    pass
try:
    del INTERACTION
except:
    pass

# Wall time: XXX (WINDOWS)
# Wall time: 2min 51s (UBUNTU)

%%time

# COMPUTE THE INTERACTION COLUMNS SEPARATELY FROM X_encoded_np

# STANDARD PYTHON for LOOP                            (10 in about 1 sec)

# concurrent.futures .submit UNDER ALL-AT-ONCE        (10 in about 1 sec)
# client.submit ON LocalCluster UNDER ALL-AT-ONCE     (10 in about 20 sec)
# client.submit ON 6 WORKERS UNDER ALL-AT-ONCE        (10 in about 15 sec)

# concurrent.futures .submit UNDER 4-FOLD for LOOP    (10 in about 10 secs)
# client.submit ON LocalCluster UNDER 4-FOLD for LOOP ???
# client.submit ON 6 WORKERS UNDER 4-FOLD for LOOP    (10 in about 20 secs)


INTX_COLUMNS = np.empty(0, dtype='<U200')
INTX_ARRAY = np.empty((0, X_encoded_np.shape[0]), dtype=np.uint8)

X_encoded_np = X_encoded_np.T

e = concurrent.futures.ProcessPoolExecutor(max_workers=4)
# e = client_local_1
# e = client2

for new_idx, (idx1, idx2) in enumerate(np.vstack((IDX1_POINTERS, IDX2_POINTERS)).T):
    # print(f'Calculating INTX column # {new_idx+1}...')
    if (new_idx + 1) % 5000 == 0: print(
        f'Calculating INTX column # {new_idx + 1}...')

    NEW_COLUMN = (X_encoded_np[idx1] * X_encoded_np[idx2]).astype(np.uint8)

    skip = False
    # CHECK IF NEW COLUMN IS EQUIV TO AN EXISTING COLUMN IN ORIGINAL DATA, IF SO DO NOT ADD ##################

    # FOUR-FOLD submit()s
    # for col_idx in range(0, len(X_encoded_np), 4):
    #     futures = [e.submit(np.array_equiv, NEW_COLUMN, _) for _ in X_encoded_np[col_idx:col_idx+4]]
    #     for future in client_as_completed(futures):
    #     # for future in concurrent.futures.as_completed(futures):
    #         if future.result() == True:
    #             # e.shutdown(wait=False)
    #             [_.cancel() for _ in futures]
    #             skip = True
    #             break
    # if skip: break

    # ALL-AT-ONCE submit()
    # futures = [e.submit(np.array_equiv, NEW_COLUMN, _) for _ in X_encoded_np]
    # # for future in client_as_completed(futures):
    # for future in concurrent.futures.as_completed(futures):
    #     if future.result() == True:
    #         # e.shutdown(wait=False)
    #         [_.cancel() for _ in futures]
    #         skip = True
    #         break

    # STRAIGHT UP PYTHON for LOOP
    for OLD_COLUMN in X_encoded_np:
        if np.array_equiv(NEW_COLUMN, OLD_COLUMN):
            skip = True
            break

    if skip: continue
    # END CHECK IF NEW COLUMN IS EQUIV TO AN EXISTING COLUMN IN ORIGINAL DATA, IF SO DO NOT ADD ###############

    # # CHECK IF NEW COLUMN IS EQUIV TO AN EXISTING COLUMN IN INTERACTIONS, IF SO DO NOT ADD ##################

    # FOUR-FOLD submit()s
    # for col_idx in range(0, len(X_encoded_np), 4):
    #     futures = [e.submit(np.array_equiv, NEW_COLUMN, _) for _ in INTX_ARRAY[col_idx:col_idx+4]]
    #     for future in client_as_completed(futures):
    #     # for future in concurrent.futures.as_completed(futures):
    #         if future.result() == True:
    #             # e.shutdown(wait=False)
    #             [_.cancel() for _ in futures]
    #             skip = True
    #             break
    #     if skip: break

    # ALL-AT-ONCE submit()
    # futures = [e.submit(np.array_equiv, NEW_COLUMN, _) for _ in INTX_ARRAY]
    # # for future in client_as_completed(futures):
    # for future in concurrent.futures.as_completed(futures):
    #     if future.result() == True:
    #         # e.shutdown(wait=False)
    #         [_.cancel() for _ in futures]
    #         skip = True
    #         break

    # STRAIGHT UP PYTHON for LOOP
    for INTX_COLUMN in INTX_ARRAY:
        if np.array_equiv(NEW_COLUMN, INTX_COLUMN):
            skip = True
            break

    if skip: continue

    # END CHECK IF NEW COLUMN IS EQUIV TO AN EXISTING COLUMN IN INTERACTIONS, IF SO DO NOT ADD #################

    # IF DID NOT SKIP FOR EQUIV, ADD THE INTX COLUMN
    INTX_COLUMNS = np.insert(INTX_COLUMNS, len(INTX_COLUMNS),
                             f'{COLUMNS[idx1]}_x_{COLUMNS[idx2]}', axis=0)
    INTX_ARRAY = np.insert(INTX_ARRAY, INTX_ARRAY.shape[0], NEW_COLUMN, axis=0)

del NEW_COLUMN, skip

try:
    del e
except:
    pass

X_encoded_np = X_encoded_np.T
INTX_ARRAY = INTX_ARRAY.T



print(f'X_encoded_np.shape = {X_encoded_np.shape}')
print(f'INTX_ARRAY.shape = {INTX_ARRAY.shape}')

# MERGE INTX COLUMNS ONTO ORIGINAL COLUMNS

X_encoded_COLUMNS = np.hstack((COLUMNS, INTX_COLUMNS))



# %%time

# MERGE INTX ONTO ORIGINAL DATA

# DONT NEED THIS SINCE IMPLEMENTED INTX EQUIV DELETE WHICH MAKES INTX_ARRAY MUCH SMALLER --- KEEP FOR POSTERITY
# TRANSFER SMALL CHUNKS OF X_encoded_np TO THE FRONT OF INTX_ARRAY PIECEMEAL TO PREVENT
# RAM BLOWUP WITH np.hstack((X_encoded_np, INTX_ARRAY))

# OVERWRITE THE BIGGER OBJECT (INTX_ARRAY) TO PREVENT MEMORY BLOWOUT

# columns_to_pick = 5

# for ctr, _ in enumerate(range(len(COLUMNS)-1, -1, -columns_to_pick)):
#     COLS = list(range(X_encoded_np.shape[1]))[-columns_to_pick:]
#     if ctr % (300//columns_to_pick) == 0:
#         print(f'Moving columns {", ".join(list(map(str, COLS)))}...')
#     elif 0 in COLS:
#         print(f'Moving columns {", ".join(list(map(str, COLS)))}...')
#     INTX_ARRAY = np.insert(INTX_ARRAY, 0, X_encoded_np[:, COLS].T, axis=1)
#     X_encoded_np = np.delete(X_encoded_np, COLS, axis=1)

# X_encoded_np = INTX_ARRAY

# del columns_to_pick

# # Wall time: 3min 30s

X_encoded_np = np.hstack((X_encoded_np, INTX_ARRAY))

# pizza, END the second de-duplicator ** * ** * ** * ** * ** * ** * ** * ** *

















