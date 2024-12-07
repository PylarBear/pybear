# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#





from typing import Literal








def _build_attributes(
    X_constants_: dict[tuple[int, ...]: any],
    _poly_constants: dict[tuple[int, ...]: any],
    _poly_dupls,  # pizza do we want the pre_ or post_ version (pre_ has X tuples)
    _keep: Literal['first', 'last', 'random'],
    _drop_constants: bool,
    _drop_duplicates: bool,
    _drop_collinear: bool,
    _rand_idxs: tuple[tuple[int, ...]]
) -> tuple[
    dict[tuple[int, ...]: tuple[int, ...]],
    kept_poly_duplicates_,
    dropped_poly_constants_,
    kept_poly_constants_,
    poly_collinear_,
    dropped_poly_collinear_,
    kept_poly_collinear_
]:

    """




    Paramters
    ---------



    Return
    ------
    -
        tuple[
            dropped_poly_duplicates_: dict[tuple[int, ...]: tuple[int, ...]],
            kept_poly_duplicates_: dict[tuple[int, ...]: list[tuple[int, ...]]],
            dropped_poly_constants_: dict[tuple[int, ...]: any],
            kept_poly_constants_: dict[tuple[int, ...]: any],
            poly_collinear_: ,
            dropped_poly_collinear_,
            kept_poly_collinear_
        ]

    """

    # poly duplicates -----------------------
    if self.drop_duplicates:
        # if :param: drop_duplicates is True, need to know from :param: keep which
        # one from each dupl set is kept
        # all other poly_duplicates_ are dropped
        # kept_poly_duplicates_ is a dictionary whose keys are the kept tuple and
        # values are the tuples that were deleted
        # need to have X tuples in here! use _poly_duplicates not poly_duplicates_
        self.kept_poly_duplicates_: dict[tuple[int, ...]: list[tuple[int, ...]]] = \
            _identify_poly_dupls_to_keep(
                self.poly_duplicates_,
                self.keep,
                self._rand_idxs
            )
        for _dupl_set_idx, _dupl_set in enumerate(self.poly_duplicates_):
            for _dupl_tuple in _dupl_set:
        # once we know the poly dupls being kept (if any), then dropped_poly_duplicates_
        # is just the opposite set, with some format manipulation
        self.dropped_poly_duplicates_: dict[tuple[int, ...]: tuple[int, ...]] = deepcopy(self.poly_duplicates_)
    elif not self.drop_duplicates:
        # if :param: drop_duplicates is False, all poly_duplicates_ are kept
        # no poly_duplicates_ are dropped
        self.kept_poly_duplicates_: dict[tuple[int, ...]: list[tuple[int, ...]]] = deepcopy(self.poly_constants_)
        # self.dropped_poly_constants_: dict[tuple[int, ...]: tuple[int, ...]] stays {}

    # END poly duplicates -----------------------

    # poly constants -----------------------
    self.poly_constants_: dict[tuple[int, ...]: any] = \
        _merge_constants(
            self.poly_constants_,
            _poly_constants_current_partial_fit,
            self.rtol,
            self.atol
        )

    # both of these are dict[tuple[int, ...]: any]
    if self.drop_constants:
        # if :param: drop_constants is True, all poly_constants_ are dropped
        # no poly_constants_ are kept
        # self.kept_poly_constants_ stays {}
        self.dropped_poly_constants_ = deepcopy(self.poly_constants_)
    elif not self.drop_constants:
        # if :param: drop_constants is False, all poly_constants_ are kept
        # no poly_constants_ are dropped
        self.kept_poly_constants_ = deepcopy(self.poly_constants_)
        # self.dropped_poly_constants_ stays {}
    # END poly constants -----------------------

    # poly collinear -----------------------
    if not hasattr(self, 'poly_collinear_'):
        self.poly_collinear_: dict[tuple[int, ...]: tuple[int, ...]] = {}
    if not hasattr(self, 'dropped_poly_collinear_'):
        self.dropped_poly_collinear_: dict[tuple[int, ...]: tuple[int, ...]] = {}
    if not hasattr(self, 'kept_poly_collinear_'):
        self.kept_poly_collinear_: dict[tuple[int, ...], tuple[int, ...]] = {}
    # END poly collinear -----------------------







    return dropped_poly_duplicates_, kept_poly_duplicates_, \
        dropped_poly_constants_, kept_poly_constants_, \
        poly_collinear_, dropped_poly_collinear_, kept_poly_collinear_











