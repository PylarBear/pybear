# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




import numpy.typing as npt



import numpy as np




# DO IT OUT THE LONG WAY, TO KEEP COLUMNS AS uint8 AND DELETE
# DUPLICATE/SPARSE COLUMNS IN-PROCESS




# INSTEAD OF ACTUALLY ADDING THE INTERACTION COLUMNS TO X_encoded_np IN
# THIS STEP, BUILD TWO VECTORS CONTAINING idx1 & idx2 RESPECTIVELY TO
# INDICATE WHICH PAIRS OF COLUMNS TO USE TO ADD INTERACTIONS TO X_encoded_np.








# PIZZA WHERE DOES X_encoded_np COME FROM?
# PIZZA WHERE DOES COLUMNS COME FROM?

X_rows, X_cols = X_encoded_np.shape



# v^v^v^v^v^v^ FIND DUMMIES FROM WITHIN THE SAME GROUP, SKIP v^v^v^v^v^v^v^v^v^v^

IDX1_POINTERS = np.empty(0, dtype=np.uint16)
IDX2_POINTERS = np.empty(0, dtype=np.uint16)

for idx1 in range(X_cols):

    for idx2 in range(idx1):


        # IF LOOKING AT PAIRS WITHIN A DUMMY GROUP (EG, RI & AZ IN "STATE")
        # THE INTERACTION WILL ALWAYS BE ZEROS.
        # THIS IS NOT THE CASE FOR "NAICS_ALL_{X}"
        # IF BOTH IDX1 & IDX2 ARE IN PORTAL, TYPE, STATE, MCAP, YEAR, MONTH,
        # DAY, WEEKDAY, THEN JUST SKIP
        DUMMY_GROUPS = ['PORT', 'TYPE', 'STAT', 'MCAP', 'YEAR', 'MONT', 'DAY_', 'WEEK']
        idx1_col_char4 = str(COLUMNS[idx1][:4])
        idx2_col_char4 = str(COLUMNS[idx2][:4])
        if idx1_col_char4 in DUMMY_GROUPS and idx2_col_char4 == idx1_col_char4:
            continue
        # END LOOKING AT PAIRS ** * ** * ** * ** * ** * ** * ** * ** * ** *

        COLUMN1 = X_encoded_np[:, idx1].astype(np.uint8)
        COLUMN2 = X_encoded_np[:, idx2].astype(np.uint8)

        """
        Boolean indicator to delete a column.
        Delete if
        1) ct is < min_cutoff
        2) Interaction == COLUMN1
        3) Interaction == COLUMN2
        """

        INTERACTION = (COLUMN1 * COLUMN2).astype(np.uint8)
        if INTERACTION.sum() < min_cutoff:    #PIZZA!
            continue
        elif np.array_equiv(INTERACTION, COLUMN1):
            continue
        elif np.array_equiv(INTERACTION, COLUMN2):
            continue
        else:
            IDX1_POINTERS = np.insert(IDX1_POINTERS, len(IDX1_POINTERS), idx1, axis=0)
            IDX2_POINTERS = np.insert(IDX2_POINTERS, len(IDX2_POINTERS), idx2, axis=0)



print(f'\n# columns end = {len(X_cols) + len(IDX1_POINTERS)}')
print(
    f'Found{len(IDX1_POINTERS): ,.0f} interaction columns of {len(X_cols) * (len(X_cols) - 1) / 2: ,.0f} '
    f'possible interactions.'
)

del COLUMN1, COLUMN2

# v^v^v^v^v^v^ END FIND DUMMIES FROM WITHIN THE SAME GROUP, SKIP v^v^v^v^v^v^v^v^



# COMPUTE THE INTERACTION COLUMNS SEPARATELY FROM X_encoded_np


INTX_COLUMNS = np.empty(0, dtype='<U200')
INTX_ARRAY = np.empty((0, X_encoded_np.shape[0]), dtype=np.uint8)

X_encoded_np = X_encoded_np.T


for (idx1, idx2) in np.vstack((IDX1_POINTERS, IDX2_POINTERS)).T:

    NEW_INTX_COLUMN = (X_encoded_np[idx1] * X_encoded_np[idx2]).astype(np.uint8)

    skip = False
    # CHECK IF NEW COLUMN IS EQUIV TO AN EXISTING COLUMN IN ORIGINAL DATA,
    # IF SO DO NOT ADD ##################

    # STRAIGHT UP PYTHON for LOOP
    for OLD_COLUMN in X_encoded_np:
        if np.array_equiv(NEW_INTX_COLUMN, OLD_COLUMN):
            skip = True
            break

    if skip: continue
    # END CHECK IF NEW COLUMN IS EQUIV TO AN EXISTING COLUMN IN ORIGINAL DATA,
    # IF SO DO NOT ADD ###############

    # # CHECK IF NEW COLUMN IS EQUIV TO AN EXISTING COLUMN IN INTERACTIONS,
    # IF SO DO NOT ADD ##################

    # STRAIGHT UP PYTHON for LOOP
    for INTX_COLUMN in INTX_ARRAY:
        if np.array_equiv(NEW_INTX_COLUMN, INTX_COLUMN):
            skip = True
            break

    if skip: continue

    # END CHECK IF NEW COLUMN IS EQUIV TO AN EXISTING COLUMN IN INTERACTIONS,
    # IF SO DO NOT ADD #################

    # IF DID NOT SKIP FOR EQUIV, ADD THE INTX COLUMN
    INTX_COLUMNS = np.insert(
        INTX_COLUMNS,
        len(INTX_COLUMNS),
        f'{COLUMNS[idx1]}_x_{COLUMNS[idx2]}', axis=0
    )
    INTX_ARRAY = np.insert(INTX_ARRAY, INTX_ARRAY.shape[0], NEW_INTX_COLUMN, axis=0)

del NEW_INTX_COLUMN, skip


X_encoded_np = X_encoded_np.T
INTX_ARRAY = INTX_ARRAY.T



print(f'X_encoded_np.shape = {X_encoded_np.shape}')
print(f'INTX_ARRAY.shape = {INTX_ARRAY.shape}')

# MERGE INTX COLUMNS ONTO ORIGINAL COLUMNS

X_encoded_COLUMNS = np.hstack((COLUMNS, INTX_COLUMNS))


# MERGE INTX ONTO ORIGINAL DATA

X_encoded_np = np.hstack((X_encoded_np, INTX_ARRAY))



















