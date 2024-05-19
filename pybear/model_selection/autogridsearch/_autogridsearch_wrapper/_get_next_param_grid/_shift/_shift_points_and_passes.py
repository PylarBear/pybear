# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from copy import deepcopy


def _shift_points_and_passes(
                                _params:dict,
                                _pass:int,
                                _total_passes_is_hard:bool
    ) -> dict:

    """
    Replicate the points from the last pass into the next pass. Truncate
    points if total_passes is hard.

    """

    __params = deepcopy(_params)

    for _param in __params:

        # for string params, increment
        # pass_on_which_to_use_only_the_single_best_value
        if __params[_param][-1] == 'string':
            __params[_param][-2] += 1

        # for numerical params:
        # replicate the previous points into the next pass and push the
        # next values over to the right;
        # e.g., [10, 5, 4, 3] on edge on pass 0,
        #   becomes [10, 10, 5, 4, 3] on pass 1
        # if total_passes_is_hard, drop last value
        # e.g., [10, 5, 4, 3] on edge on pass 0,
        #   becomes [10, 10, 5, 4] on pass 1
        else:
            __params[_param][1].insert(_pass, __params[_param][1][_pass - 1])

            if _total_passes_is_hard:
                __params[_param][1] = __params[_param][1][:-1]


    return __params















