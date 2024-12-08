# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#





from .._type_aliases import FeatureNameCombinerType

import numpy as np



def _val_feature_name_combiner(
    _feature_name_combiner: FeatureNameCombinerType,
    _min_degree: int,
    _degree: int,
    _X_num_columns: int,
    _interaction_only: bool
) -> None:

    """
    Validate feature_name_combiner. Must be:
    1) Literal 'as_feature_names' ... pizza pizza
    2) Literal 'as_indices' ... pizza
    3) a callable that takes a
    vector of strings (the feature names of X) and a variable-length tuple
    of integers (the polynomial indices combination) as input and always
    returns a string.


    Parameters
    ----------
    _feature_name_combiner:
        Union[Callable[[tuple[int, ...]], str], Literal['as_feature_names', 'as_indices']] default = None -
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
        bool - whether to only return pizza?


    Return
    ------
    -
        None.


    """


    # for however many times you want to test this:
        # randomly pick a length of the combo tuple (min_degree <= x <= degree)
        # randomly pick that many integers from the column indices of X
        # pass the combo of indices to the callable
        # assert output of callable is a string

    _columns = [f'x{i}' for i in range(_X_num_columns)]

    if callable(_feature_name_combiner):
        # ensure the callable returns a string

        # pizza when u get there, when this is actually called in some other module ensure the
        # returned feature names are unique

        for _trial in range(10):

            _rand_num_columns = np.random.choice(range(_min_degree, _degree+1))

            _rand_combo = tuple(
                np.random.choice(
                    range(_X_num_columns),
                    _rand_num_columns,
                    replace=True
                )
            )

            out = _feature_name_combiner(_columns, _rand_combo)

            if not isinstance(out, str):
                raise ValueError(
                    f"validation test on combo {_rand_combo} failed, returned "
                    f"type {type(out)}, must return string."
                )

    elif _feature_name_combiner == 'as_feature_names':
        # allowed, all good
        pass

    elif _feature_name_combiner == 'as_indices':
        # allowed, all good
        pass

    else:
        raise ValueError(
            f"\ninvalid :param: feature_name_combiner. must be: "
            f"\n1) Literal 'as_feature_names' ... pizza pizza "
            f"\n2) Literal 'as_indices' ... pizza "
            f"\n3) a callable that takes a vector of strings (the feature "
            f"names of X) and a variable-length tuple of integers (the "
            f"polynomial indices combination) as input and always "
            f"returns a string."
        )























