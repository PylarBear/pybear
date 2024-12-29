# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest


import numpy as np

from pybear.preprocessing.MinCountTransformer._validation._val_feature_names \
    import _val_feature_names


class TestValFeatureNamesIn:


    @staticmethod
    @pytest.fixture
    def good_feature_names_in():
        return np.array(('abc', 'def', 'ghi', 'jkl'), dtype='<U3')


    @pytest.mark.parametrize('junk_feature_names_in',
        (0, True, None, np.pi, {'a': 1}, 'junk', min, lambda x: x)
    )
    def test_rejects_non_list_like(self, good_feature_names_in,
                                   junk_feature_names_in):
        with pytest.raises(TypeError):
            _val_feature_names(good_feature_names_in, junk_feature_names_in)


    def test_accepts_iterable_w_strs(self, good_feature_names_in):

        _val_feature_names(good_feature_names_in, good_feature_names_in)


    @pytest.mark.parametrize('junk_value',
        (0, True, None, np.pi, [1,2], (1,2), {1,2}, {'a': 1}, min, lambda x: x)
    )
    def test_rejects_junk_strs(self, junk_value, good_feature_names_in):

        bad_feature_names_in = good_feature_names_in.astype(object)

        bad_feature_names_in[0] = junk_value

        with pytest.raises(ValueError):
            _val_feature_names(bad_feature_names_in, good_feature_names_in)


    @pytest.mark.parametrize('bad_value',
        ('junk', 'trash', 'garbage', 'rubbish')
    )
    def test_rejects_column_mismatch(self, bad_value, good_feature_names_in):

        bad_feature_names_in = good_feature_names_in.astype('<U10')

        bad_feature_names_in[0] = bad_value

        with pytest.raises(ValueError):
            _val_feature_names(bad_feature_names_in, good_feature_names_in)

        bad_feature_names_in = good_feature_names_in.astype('<U10')

        bad_feature_names_in = np.insert(
            bad_feature_names_in,
            len(bad_feature_names_in),
            'mno',
            axis=0
        )

        with pytest.raises(ValueError):
            _val_feature_names(bad_feature_names_in, good_feature_names_in)







