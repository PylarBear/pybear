# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.preprocessing.SlimPolyFeatures._validation._feature_name_combiner \
    import _val_feature_name_combiner

import numpy as np

import pytest



class TestFeatureNameCombiner:


    @pytest.mark.parametrize('_string_literals',
        ('junk', 'trash', 'as_indices', 'garbage', 'as_feature_names')
    )
    @pytest.mark.parametrize('_min_degree, _degree', ((1,2), (1,3), (2,4)))
    @pytest.mark.parametrize('_X_num_columns', (5, 10))
    @pytest.mark.parametrize('_interaction_only', (True, False))
    def test_string_literals(
        self, _string_literals, _min_degree, _degree, _X_num_columns,
        _interaction_only
    ):

        if _string_literals in ('as_indices', 'as_feature_names'):
            _val_feature_name_combiner(
                _string_literals,
                _min_degree,
                _degree,
                _X_num_columns,
                _interaction_only
            )
        else:
            with pytest.raises(ValueError):
                _val_feature_name_combiner(
                    _string_literals,
                    _min_degree,
                    _degree,
                    _X_num_columns,
                    _interaction_only
                )




    @pytest.mark.parametrize('_fnc_output',
        (
            -np.e, -1, 0, 1, np.e, True, False, None, [0,1], (0,), {0,1},
            {'a':1}, 'string'
        )
    )
    @pytest.mark.parametrize('_min_degree, _degree', ((1,2), (1,3), (2,4)))
    @pytest.mark.parametrize('_X_num_columns', (5, 10))
    @pytest.mark.parametrize('_interaction_only', (True, False))
    def test_accepts_good_callable(
        self, _fnc_output, _min_degree, _degree, _X_num_columns, _interaction_only
    ):

        _rand_num_columns = np.random.choice(range(_min_degree, _degree+1))

        _rand_combo = tuple(
            np.random.choice(
                range(_X_num_columns),
                _rand_num_columns,
                replace=True
            )
        )

        _columns = [f'x{i}' for i in range(_X_num_columns)]

        _feature_name_combiner = lambda _columns, x: _fnc_output

        if _fnc_output == 'string':
            _val_feature_name_combiner(
                _feature_name_combiner,
                _min_degree,
                _degree,
                _X_num_columns,
                _interaction_only
            )
        else:
            with pytest.raises(ValueError):
                _val_feature_name_combiner(
                    _feature_name_combiner,
                    _min_degree,
                    _degree,
                    _X_num_columns,
                    _interaction_only
                )































