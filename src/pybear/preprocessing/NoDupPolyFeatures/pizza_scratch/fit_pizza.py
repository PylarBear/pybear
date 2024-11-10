# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


_duplicates: dict[tuple[int], list[tuple[int]]]
_constants: dict[tuple[int], any]


def fit_pizza(self, X):

    # get constants from X v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    _X_constant_idxs = _get_X_constants(X)
    # END get constants from X v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    # GENERATE COMBINATIONS W/O CONSTANTS # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
    # since this is one fit we know what the constants are, we can take the
    # constants in X out of the combinations
    combinations = _combination_builder(X, _constants=_X_constant_idxs, min_degree, max_degree)
    # END GENERATE COMBINATIONS W/O CONSTANTS # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v

    POLY_ARRAY = np.empty((X.shape[0], 0)).astype(X.dtype)
    IDXS_IN_POLY_ARRAY = []

    # need to iterate over the combos and find what is constant or duplicate

    # we need to get everything because set_params() might change after fit!


    for combo in combinations:
        _COLUMN = X[:, combo].product(1)

        # there are no constants put into POLY_ARRAY but are recorded in _constants
        # there are no duplicates put into POLY_ARRAY, but are recorded in _duplicates

        # determine if constant
        # use peak to peak
        if np.ptp(_COLUMN, ignore_nan=False, axis=0) == 0:

            # if the product is a column of constants, put the idx and value in the holder and
            # do not append to POLY_ARRAY

            _value = _COLUMN[0][0]

            # if a constant, also must know if it is a duplicate
            # search thru _constants for another with the same value,
            # if so, log in duplicates and continue
            for k,v in _constants:
                if v == _value:
                    _duplicates[k] = _value

            _constants[combo] = _value


        # look_for_duplicates_in_X
        out = _look_for_duplicates_in_X(_COLUMN, X)
        # _look_for_duplicates_in_X needs to return the idx in X that the combo matches
        # so both X idx and combo can be put into the _duplicates dict
        if out:
            if out not in _duplicates:
                _duplicates[out] = []
            _duplicates[out].append(combo)
            POLY_ARRAY = np.hstack((POLY_ARRAY, _COLUMN))
            IDXS_IN_POLY_ARRAY.append(combo)
            continue

        out = _look_for_duplicates_in_POLY(_COLUMN, POLY_ARRAY)
        # _look_for_duplicates_in_POLY needs to return the combo in POLY that
        # the current combo matches so both POLY combo and current combo can be
        # put into the _duplicates dict
        if out:
            if out not in _duplicates:
                _duplicates[out] = []
            _duplicates[out].append(combo)
            POLY_ARRAY = np.hstack((POLY_ARRAY, _COLUMN))
            IDXS_IN_POLY_ARRAY.append(combo)
            continue

        # if we get to this point, then we cared if _COLUMN was a constant
        # or a duplicate, but it was neither, so _COLUMN goes into POLY_ARRAY
        POLY_ARRAY = np.hstack((POLY_ARRAY, _COLUMN))
        IDXS_IN_POLY_ARRAY.append(combo)


# what do we have at this point?

# _X_constant_idxs --- good as is since its one fit
# combinations (w/o constants in X)
# the original X  --- fit to be returned
# POLY_ARRAY --- this is correctly expanded out and would be fit to be returned
# IDXS_IN_POLY_ARRAY:list[tuple[int]] nuff ced
# _constants: dict[tuple[int], any]
# _duplicates: dict[tuple[int], list[tuple[int]]
    # _duplicates needs to be converted to duplicates_:list[list[tuple]]
    # see CDT _find_duplicates.

# if transforming straight from here:
# hstack((X, POLY_ARRAY)) and return
# merge _X_constant_idxs and _constants then can be set to self.constants_
# maybe we need to find out the duplicates in X and merge that with _duplicates

























