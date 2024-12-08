# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable, Callable
from typing_extensions import Union

import numpy.typing as npt
import numpy as np

from sklearn.exceptions import NotFittedError


def _get_feature_names_out(
    _input_features: Union[Iterable[str], None],
    feature_names_in_: Union[npt.NDArray[str], None],
    _min_degree: int,
    _active_combos: tuple[tuple[int, ...], ...],
    _X_shape: tuple[int, ...],
    _feature_name_combiner: Union[Callable[[tuple[int, ...]], str], None]
) -> npt.NDArray[object]:

    """
    Pizza!


    Parameters
    ----------
    _input_features:
        Union[Iterable[str], None] -
    feature_names_in_:
        Union[npt.NDArray[str], None] -
    _min_degree:
        int -
    _active_combos:
        tuple[tuple[int, ...], ...] -
    _X_shape:
        tuple[int, ...] - The shape of the data passed to undergo polynomial expansion.
    _feature_name_combiner:
        # pizza finalized this and put it in the main module.
        Union[Callable[[tuple[int, ...]], str], None], default = None -
        User-defined function for mapping combo tuples of integers to
        polynomial feature names. Must take in a tuple of integers of
        variable length (min length is :param: min_degree, max length is
        :param: degree) and return a string. If None, then the default
        polynomial feature name format is used. For example, if the
        feature names of X are ['x0', 'x1', ..., 'xn'] and the polynomial
        tuple is (2, 2, 4), then the default polynomial feature name is
        'x2^2_x4'.



    Return
    ------
    -
        _feature_names_out: npt.NDArray[object] - The feature names for
        the polynomial expansion.


    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * 

    try:
        if isinstance(_input_features, type(None)):
            raise UnicodeError
        iter(_input_features)
        if isinstance(_input_features, (str, dict)):
            raise Exception
        if not all(map(
                isinstance, _input_features, (str for _ in _input_features)
        )):
            raise Exception
    except UnicodeError:
        pass
    except:
        raise ValueError(
            f"'_input_features' must be a vector-like containing strings, or None"
        )


    if feature_names_in_ is not None:
        assert isinstance(feature_names_in_, np.ndarray)
        assert len(feature_names_in_.shape) == 1
        assert feature_names_in_.shape[0] == _X_shape[1]
        assert all(map(isinstance, feature_names_in_, (str for _ in feature_names_in_)))

    assert isinstance(_min_degree, int)
    assert _min_degree >= 1

    assert isinstance(_active_combos, tuple)
    for _tuple in _active_combos:
        assert isinstance(_tuple, tuple)
        assert all(map(isinstance, _tuple, (int for _ in _tuple)))

    assert callable(_feature_name_combiner)

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *  



    if _input_features is not None:

        if len(_input_features) != self.n_features_in_:
            raise ValueError("_input_features should have length equal")

        if hasattr(self, 'feature_names_in_'):

            if not np.array_equal(_input_features, self.feature_names_in_):
                raise ValueError(
                    f"_input_features is not equal to feature_names_in_"
                )

        out = np.array(_input_features, dtype=object)[self.column_mask_]

        return out

    elif hasattr(self, 'feature_names_in_'):
        return self.feature_names_in_[self.column_mask_].astype(object)

    else:
        try:
            _input_features = \
                np.array([f"x{i}" for i in range(self.n_features_in_)])
            return _input_features.astype(object)[self.column_mask_]
        except:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this "
                f"estimator."
            )

    """
    FROM sklearn.utils.validation
    
    def _check_feature_names_in(
        estimator, 
        input_features=None, 
        *, 
        generate_names=True
    ):
    
        Check `input_features` and generate names if needed.
    
        Commonly used in :term:`get_feature_names_out`.
    
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.
    
            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.
    
        generate_names : bool, default=True
            Whether to generate names when `input_features` is `None` and
            `estimator.feature_names_in_` is not defined. This is useful for transformers
            that validates `input_features` but do not require them in
            :term:`get_feature_names_out` e.g. `PCA`.
    
        Returns
        -------
        feature_names_in : ndarray of str or `None`
            Feature names in.
    

        feature_names_in_ = getattr(estimator, "feature_names_in_", None)
        n_features_in_ = getattr(estimator, "n_features_in_", None)
    
        if input_features is not None:
            input_features = np.asarray(input_features, dtype=object)
            if feature_names_in_ is not None and not np.array_equal(
                feature_names_in_, input_features
            ):
                raise ValueError("input_features is not equal to feature_names_in_")
    
            if n_features_in_ is not None and len(input_features) != n_features_in_:
                raise ValueError(
                    "input_features should have length equal to number of "
                    f"features ({n_features_in_}), got {len(input_features)}"
                )
            return input_features
    
        if feature_names_in_ is not None:
            return feature_names_in_
    
        if not generate_names:
            return
    
        # Generates feature names if `n_features_in_` is defined
        if n_features_in_ is None:
            raise ValueError("Unable to generate feature names without n_features_in_")
    
        return np.asarray([f"x{i}" for i in range(n_features_in_)], dtype=object)

    """



    """
    FROM _encoders 
    
    def _compute_transformed_categories(self, i, remove_dropped=True):
        
        Compute the transformed categories used for column `i`.

        1. If there are infrequent categories, the category is named 'infrequent_sklearn'.
        2. Dropped columns are removed when remove_dropped=True.
        
        cats = self.categories_[i]

        if self._infrequent_enabled:
            infreq_map = self._default_to_infrequent_mappings[i]
            if infreq_map is not None:
                frequent_mask = infreq_map < infreq_map.max()
                infrequent_cat = "infrequent_sklearn"
                # infrequent category is always at the end
                cats = np.concatenate(
                    (cats[frequent_mask], np.array([infrequent_cat], dtype=object))
                )

        if remove_dropped:
            cats = self._remove_dropped_categories(cats, i)
        return cats
    
    
    FROM OneHotEncoder
    def _check_get_feature_name_combiner(self):
        if self.feature_name_combiner == "concat":
            return lambda feature, category: feature + "_" + str(category)
        else:  # callable
            dry_run_combiner = self.feature_name_combiner("feature", "category")
            if not isinstance(dry_run_combiner, str):
                raise TypeError(
                    "When `feature_name_combiner` is a callable, it should return a "
                    f"Python string. Got {type(dry_run_combiner)} instead."
                )
            return self.feature_name_combiner
    
    
    FROM OneHotEncoder
    def get_feature_names_out(
        self, 
        input_features=None
    ):
    
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        
        
        
        check_is_fitted(self)
        input_features = _check_feature_names_in(self, input_features)
        cats = [
            self._compute_transformed_categories(i)
            for i, _ in enumerate(self.categories_)
        ]

        feature_names = []
        for i in range(len(cats)):
            names = [self._check_get_feature_name_combiner()(input_features[i], t) for t in cats[i]]
            feature_names.extend(names)

        return np.array(feature_names, dtype=object)

    """

    """
    FROM POLYNOMIAL FEATURES
     
    def get_feature_names_out(
        self, 
        input_features=None
    ):
    
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features is None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        
        
        powers = self.powers_
        input_features = _check_feature_names_in(self, input_features)
        feature_names = []
        for row in powers:
            inds = np.where(row)[0]
            if len(inds):
                name = " ".join(
                    (
                        "%s^%d" % (input_features[ind], exp)
                        if exp != 1
                        else input_features[ind]
                    )
                    for ind, exp in zip(inds, row[inds])
                )
            else:
                name = "1"
            feature_names.append(name)
        return np.asarray(feature_names, dtype=object)
    
    """











