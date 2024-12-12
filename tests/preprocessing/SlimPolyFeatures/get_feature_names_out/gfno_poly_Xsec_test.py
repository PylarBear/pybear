# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.SlimPolyFeatures._get_feature_names_out. \
    _gfno_poly import _gfno_poly

from typing import Callable, Iterable, Literal
from typing_extensions import Union

import numpy as np

import pytest



class TestGFNOPoly:


    # def _gfno_poly(
    #         feature_names_in_: npt.NDArray[str],
    #         _active_combos: tuple[tuple[int, ...], ...],
    #         _feature_name_combiner: FeatureNameCombinerType
    # ) -> npt.NDArray[object]:


    @pytest.mark.parametrize('junk_feature_names_in_',
        (-2.7, -1, 0, 1, 2.7, True, False, None, 'junk', [0, 1], (0, 1), {'a': 1}, lambda x: x)
    )
    def test_feature_names_in_rejects_junk(self, junk_feature_names_in_):

        with pytest.raises(AssertionError):

            _gfno_poly(
                junk_feature_names_in_,
                _active_combos=((0,0), (0,1), (1,1)),
                _feature_name_combiner='as_indices'
            )


    @pytest.mark.parametrize('junk_active_combos',
        (-2.7, -1, 0, 1, 2.7, True, False, None, 'junk', [0, 1], (0, 1), {'a': 1}, lambda x: x)
    )
    def test_active_combos_rejects_junk(self, junk_active_combos):

        with pytest.raises(AssertionError):

            _gfno_poly(
                feature_names_in_=np.array(['a', 'b'], dtype=object),
                _active_combos=junk_active_combos,
                _feature_name_combiner='as_indices'
            )


    @pytest.mark.parametrize('junk_feature_name_combiner',
        (-2.7, -1, 0, 1, 2.7, True, False, None, 'junk', [0, 1], (0, 1), {'a': 1})
    )
    def test_feature_name_combiner_rejects_junk(self, junk_feature_name_combiner):

        with pytest.raises(AssertionError):

            _gfno_poly(
                feature_names_in_=np.array(['a', 'b'], dtype=object),
                _active_combos=((0,0), (0,1), (1,1)),
                _feature_name_combiner=junk_feature_name_combiner
            )


    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('_feature_name_combiner_trial',
        ('as_indices', 'as_feature_names', 'callable1', 'callable2')
    )
    def test_accuracy(self, _feature_name_combiner_trial):

        _feature_names_in = np.array(['a', 'b'], dtype=object)
        _active_combos = ((0, 0), (0, 1), (1, 1))

        _feature_name_combiner: Union[
            Callable[[Iterable[str], tuple[tuple[int, ...], ...]], str],
            Literal['as_feature_names', 'as_indices']
        ]

        if _feature_name_combiner_trial == 'as_indices':
            _feature_name_combiner = 'as_indices'
        elif _feature_name_combiner_trial == 'as_feature_names':
            _feature_name_combiner = 'as_feature_names'
        elif _feature_name_combiner_trial == 'callable1':
            _feature_name_combiner = lambda feature_names, combos: '+'.join(map(str, combos))
        elif _feature_name_combiner_trial == 'callable2':
            # this deliberately creates duplicate feature names to force raise
            _feature_name_combiner = lambda feature_names, combos: 'junk'
        else:
            raise Exception()

        if _feature_name_combiner_trial == 'callable2':
            with pytest.raises(ValueError):
                _gfno_poly(
                    feature_names_in_=_feature_names_in,
                    _active_combos=_active_combos,
                    _feature_name_combiner=_feature_name_combiner
                )
        else:

            out = _gfno_poly(
                feature_names_in_=_feature_names_in,
                _active_combos=_active_combos,
                _feature_name_combiner=_feature_name_combiner
            )

            if '_feature_name_combiner' == 'as_indices':
                assert out == list(map(str, ((0, 0), (0, 1), (1, 1))))
            elif '_feature_name_combiner' == 'as_feature_names_':
                ref = []
                # scan over the combo, get the powers by counting the number of
                # occurrences of X column indices
                _idx_ct_dict = {k: 0 for k in _active_combos}  # only gets unique idxs
                for _X_idx in _active_combos:
                    _idx_ct_dict[_X_idx] += 1

                    _poly_feature_name = ''

                    for _combo_idx, _X_idx in enumerate(_active_combos):
                        _poly_feature_name += f"{_feature_names_in[_X_idx]}^{_idx_ct_dict[_X_idx]}"
                        if _combo_idx < (len(_active_combos) - 1):
                            _poly_feature_name += "_"

                    ref.append(_poly_feature_name)

                assert np.array_equiv(out, ref)
            elif '_feature_name_combiner' == 'callable1':
                assert np.array_equal(out, ['+'.join(map(str, _combo)) for _combo in _active_combos])
            # 'callable2' excepted above and skipped


            assert isinstance(out, np.ndarray)
            return np.array(out, dtype=object)











