# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np



def _val_num_combinations(
    n_features_in_: int,
    _n_poly_features: int,
    _min_degree: int,
    _max_degree: int,
    _intx_only: bool
) -> None:

    """
    PIZZA REWRITE THIS
    Calculate number of terms in polynomial expansion

    This should be equivalent to counting the number of terms returned by
    _combinations(...) but much faster.


    Parameters
    ----------
    n_features_in_:
        int,
    _n_poly_features:
        int,
    _min_degree:
        int,
    _max_degree:
        int,
    _intx_only:
        bool



    Return
    ------
    -
        None


    """

    for _ in (n_features_in_, _n_poly_features, _min_degree, _max_degree):
        assert isinstance(_, int)
        assert not isinstance(_, bool)

    assert n_features_in_ >= 1
    assert _n_poly_features >= 1
    assert _min_degree >= 1, f"min_degree == 0 shouldnt be getting in here"
    assert _max_degree >= 2, f"max_degree in [0,1] shouldnt be getting in here"
    assert _max_degree >= _min_degree

    assert isinstance(_intx_only, bool)

    if _min_degree == 1:
        _n_output_features = n_features_in_ + _n_poly_features
    else:
        _n_output_features = _n_poly_features


    # this is taken almost verbatim from
    # sklearn.preprocessing._polynomial.PolynomialFeatures.fit()
    if _n_output_features > np.iinfo(np.intp).max:

        msg = (
            "The output that would result from the current configuration would"
            f" have {_n_output_features} features which is too large to be"
            f" indexed by {np.intp().dtype.name}. Please change some or all of the"
            " following:\n- The number of features in the input, currently"
            f" {n_features_in_=}\n- The range of degrees to calculate, currently"
            f" [{_min_degree}, {_max_degree}]\n- Whether to include only"
            f" interaction terms, currently {_intx_only}."
        )

        if np.intp == np.int32 and _n_output_features <= np.iinfo(np.int64).max:
            msg += (
            "\nNote that the current Python runtime has a limited 32 bit "
            "address space and that this configuration would have been "
            "admissible if run on a 64 bit Python runtime."
            )

        raise ValueError(msg)





