# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.base._check_shape import check_shape

import numpy as np
import pandas as pd
import scipy.sparse as ss

import pytest





class TestCheckShape:


    # def check_shape(
    #     OBJECT,
    #     min_features: numbers.Integral=1,
    #     min_samples: numbers.Integral=1,
    #     allowed_dimensionality: Iterable[numbers.Integral] = (1, 2)
    # ) -> tuple[int, ...]:


    @pytest.mark.parametrize('X_format', ('np', 'pd', 'coo'))
    @pytest.mark.parametrize('dimensionality', (1, 2, 3, 4, 5))
    @pytest.mark.parametrize('allowed_dimensionality', ((1,), (2,), (1,2)))
    def test_rejects_bad_dimensionality(
        self, X_format, dimensionality, allowed_dimensionality
    ):

        # skip impossible conditions - - - - - - - - - - - - - - - - - -

        if X_format == 'coo' and dimensionality != 2:
            pytest.skip(reason=f"scipy sparse can only be 2D")

        if X_format == 'pd' and dimensionality > 2:
            pytest.skip(reason=f"pd dataframe must be 1 or 2D")

        # END skip impossible conditions - - - - - - - - - - - - - - - -

        # create the shape tuple
        _shape = tuple(np.random.randint(2, 10, dimensionality).tolist())

        _base_X = np.random.randint(0, 10, _shape)

        if X_format == 'np':
            _X = _base_X
        elif X_format == 'pd':
            _X = pd.DataFrame(data=_base_X)
            if dimensionality == 1:
                _X = _X.squeeze()
        elif X_format == 'coo':
            _X = ss.coo_array(_base_X)
        else:
            raise Exception

        if dimensionality in allowed_dimensionality:
            out = check_shape(
                _X,
                allowed_dimensionality=allowed_dimensionality
            )

            assert out == _shape

        elif dimensionality not in allowed_dimensionality:
            with pytest.raises(ValueError):
                check_shape(
                    _X,
                    allowed_dimensionality=allowed_dimensionality
                )


    @pytest.mark.parametrize('X_format', ('np', 'pd', 'dok'))
    @pytest.mark.parametrize('dimensionality', (1, 2))
    @pytest.mark.parametrize('features', (0, 1, 2, 100))
    @pytest.mark.parametrize('min_features', (0, 1, 2))
    def test_rejects_too_few_features(
        self, X_format, dimensionality, features, min_features
    ):

        # skip impossible conditions - - - - - - - - - - - - - - - - - -

        if dimensionality == 1 and features != 1:
            pytest.skip(reason=f"impossible condition")

        if X_format == 'dok' and features != 2:
            pytest.skip(reason=f"scipy sparse must be 2D")

        # END skip impossible conditions - - - - - - - - - - - - - - - -


        if dimensionality == 1:
            # features == 1
            _shape = (100, )
            _base_X = np.random.randint(0, 10, _shape)
        elif dimensionality == 2:
            _shape = (100, features)
            _base_X = np.random.randint(0, 10, _shape)

        if X_format == 'np':
            _X = _base_X
        elif X_format == 'pd':
            _X = pd.DataFrame(data=_base_X)
            if dimensionality == 1:
                _X = _X.squeeze()
        elif X_format == 'dok':
            _X = ss.dok_array(_base_X)
        else:
            raise Exception


        if features >= min_features:
            out = check_shape(
                _X,
                min_features=min_features
            )

            assert out == _shape

        elif features < min_features:
            with pytest.raises(ValueError):
                check_shape(
                    _X,
                    min_features=min_features
                )



    @pytest.mark.parametrize('X_format', ('np', 'pd', 'lil'))
    @pytest.mark.parametrize('dimensionality', (1, 2))
    @pytest.mark.parametrize('samples', (0, 1, 2, 100))
    @pytest.mark.parametrize('min_samples', (0, 1, 2))
    def test_rejects_too_few_samples(
        self, X_format, dimensionality, samples, min_samples
    ):

        # skip impossible conditions - - - - - - - - - - - - - - - - - -
        if X_format == 'lil' and dimensionality != 2:
            pytest.skip(reason=f"scipy sparse must be 2D")
        if X_format == 'pd' and dimensionality == 1 and samples == 1:
            pytest.skip(reason=f"1x1 pandas squeezes to a number.")
        # END skip impossible conditions - - - - - - - - - - - - - - - -


        if dimensionality == 1:
            _shape = (samples,)
            _base_X = np.random.randint(0, 10, _shape)
        elif dimensionality == 2:
            _shape = (samples, 5)
            _base_X = np.random.randint(0, 10, _shape)

        if X_format == 'np':
            _X = _base_X
        elif X_format == 'pd':
            _X = pd.DataFrame(data=_base_X)
            if dimensionality == 1:
                _X = _X.squeeze()
        elif X_format == 'lil':
            _X = ss.lil_array(_base_X)
        else:
            raise Exception


        if samples >= min_samples:
            out = check_shape(
                _X,
                min_samples=min_samples
            )

            assert out == _shape

        elif samples < min_samples:
            with pytest.raises(ValueError):
                check_shape(
                    _X,
                    min_samples=min_samples
                )
































