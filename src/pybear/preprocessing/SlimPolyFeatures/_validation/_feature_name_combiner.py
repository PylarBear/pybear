# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Callable

import numpy as np



def _val_feature_name_combiner(
    _feature_name_combiner: Callable[[tuple[int, ...]], str],
    _min_degree: int,
    _degree: int,
    _X_num_columns: int,
    _interaction_only: bool
) -> None:

    """
    Validate feature_name_combiner. Must be a callable that takes a varibble-length
    tuple of integers as input.

    _feature_name_combiner:
        Union[Callable[[tuple[int, ...]], str], None], default = None -
        User-defined function for mapping combo tuples of integers to
        polynomial feature names. Must take in a tuple of integers of
        variable length (min length is :param: min_degree, max length is
        :param: degree) and return a string. If None, then the default
        polynomial feature name format is used. For example, if the
        feature names of X are ['x0', 'x1', ..., 'xn'] and the polynomial
        tuple is (2, 4, 2), then the default polynomial feature name is
        'x2^2_x4'.

    _min_degree:
        int - the minimum polynomial degree of the polynomial expansion

    _degree:
        int - the minimum polynomial degree of the polynomial expansion

    _X_num_columns:
        int - the number of columns in the passed data.

    _interaction_only:
        bool -

    """


    # for however many times you want to test this:
        # randomly pick a length of the combo tuple (min_degree <= x <= degree)
        # randomly pick that many integers from the column indices of X
        # pass the combo of indices to the callable
        # assert output of callable is a string

    for _trial in range(10):

        _rand_num_columns = np.random.choice(range(_min_degree, _degree+1))

        _rand_combo = tuple(
            np.random.choice(
                range(_X_num_columns),
                _rand_num_columns,
                replace=True
            )
        )

        out = _feature_name_combiner(_rand_combo)

        assert isinstance(out, str), \
            (f"validation test on combo {_rand_combo} failed, returned "
             f"type{type(out)}, must return string.")




























