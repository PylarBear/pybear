# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


_duplicates: dict[tuple[int], list[tuple[int]]]
_constants: dict[tuple[int], any]


def partial_fit_pizza(self, X):

    # get constants from X v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    _X_constant_idxs = _get_X_constants(X)
    # END get constants from X v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    # GENERATE COMBINATIONS W/ CONSTANTS IN # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
    # since we dont know what the constants are in future Xs, need to keep these
    # constants in the current combinations
    combinations = _combination_builder(X, _constants=[], min_degree, max_degree)
    # END GENERATE COMBINATIONS W/ CONSTANTS IN # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v

    POLY_ARRAY = np.empty((X.shape[0], 0)).astype(X.dtype)
    IDXS_IN_POLY_ARRAY = []

    # what we know for the first partial fit is that X has certain constants,
    # and that the first expansion may have certain constants and certain
    # duplicates. since we havent seen future Xs, we dont know the global
    # constants in X nor the global constants and duplicates in the expansion.
    # Future Xs may cause past constants in X to no longer be constant, and
    # may cause past constants and duplicates in the expansion to no longer
    # be constants or duplicates. But not the other way around; future
    # constants and duplicates cannot make a whole column become a constant
    # or a duplicate if it wasnt already.
    # Therefore, it doesnt matter what is currently constant or
    # duplicate for partial fits, we cant exclude them because they may
    # expose differently in the future, so we need to keep track of
    # absolutely everything.

    # we need to get everything because set_params() might change after fit!

    # need to iterate over the combos and find what is constant or duplicate

    for combo in combinations:
        _COLUMN = X[:, combo].product(1)

        # there are no constants put into POLY_ARRAY but are recorded in
        # if we are looking at duplicates, there are no duplicates in POLY_ARRAY, but may be in X

        # if we are looking at both, neither is in poly array. in this case, we
        # can look at it being constant first, that is less expensive to get.
        # we already know a column of constants cant be a duplicate of POLY, but it could be a duplicate of X

        # use peak to peak
        if np.ptp(_COLUMN, ignore_nan=False, axis=0) == 0:
            _constants[combo] = _COLUMN[0][0]
            # if the product is a column of constants, put the idx and value in the holder and
            # do not append to POLY_ARRAY
            # dont even look at duplicates, combo only needs to go into one holder


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














            elif drop_duplicates:
                if _look_for_duplicates_in_X(_COLUMN) is True:









